// Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "param_search.h"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "colorspace_conversion.h"
#include "dinov3_auto_solver.h"
#include "neg_pipeline.h"

const char* to_string(FixedPointStatus status) {
  switch (status) {
    case FixedPointStatus::kConverged: return "CONVERGED";
    case FixedPointStatus::kDiverged: return "DIVERGED";
    case FixedPointStatus::kMaxIters: return "MAX_ITERS";
  }
  return "";
}

const char* to_string(HeadroomOutcome outcome) {
  switch (outcome) {
    case HeadroomOutcome::kWithinRange: return "within calibrated range";
    case HeadroomOutcome::kNoCapacity: return "no exposure capacity (E at or beyond the bracket's e_max)";
    case HeadroomOutcome::kCapacityLimited: return "capacity limited";
    case HeadroomOutcome::kPulledInside: return "pulled inside calibrated range";
  }
  return "";
}

const char* to_string(RescueOutcome outcome) {
  switch (outcome) {
    case RescueOutcome::kNotNeeded: return "not needed";
    case RescueOutcome::kNotScreened: return "next bracket not screened";
    case RescueOutcome::kNotAdmissible: return "next bracket not admissible";
    case RescueOutcome::kClipsTooMuch: return "clip ratio";
    case RescueOutcome::kShiftsLightness: return "L* shift";
    case RescueOutcome::kWorseResidual: return "residual";
    case RescueOutcome::kTaken: return "taken";
  }
  return "";
}

namespace {

// Fixed-point loop thresholds (see FixedPointStatus).
constexpr double TOL_PARAM = 0.02;     // |dE| + |dg| + |db| <= 0.02
constexpr double TOL_REL_ERR = 0.015;  // relative residual change <= 1.5 %
constexpr int MAX_STEPS = 8;
// The scene's range is taken between its SCENE_TAIL and 1 - SCENE_TAIL points.
constexpr double SCENE_TAIL = 0.005;
constexpr double SLACK_PENALTY_SCALE = 0.5;
// The previews, the screening and the final render use a fixed soft knee.
constexpr float AUTO_KNEE = 0.95f;
// Highlight rescue thresholds (see RescueOutcome).
constexpr double RESCUE_CLIP_TAU = 3e-4;
constexpr double RESCUE_CLIP_RATIO = 0.5;
constexpr double RESCUE_DL_MIN = -4.5, RESCUE_DL_MAX = 0.5;
constexpr double RESCUE_RESID_TOL = 0.10;
// The search runs as if the frame were scanned at 1/8 s, ISO 100 with the profile's film base.
constexpr double REFERENCE_SHUTTER_S = 0.125;
constexpr double REFERENCE_ISO = 100.0;

double now_ms() { return omp_get_wtime() * 1000.0; }

// v with prec decimals ("%.<prec>f", or "%+.<prec>f" with sign), for the verbose log.
std::string fixed(double v, int prec, bool sign = false) {
  std::ostringstream os;
  if (sign) os << std::showpos;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}

// v with prec significant digits ("%.<prec>g"), for the verbose log.
std::string general(double v, int prec) {
  std::ostringstream os;
  os << std::setprecision(prec) << v;
  return os.str();
}

const uint16_t* pixel_at(const RgbImage& img, int x, int y) { return &img.rgb[(size_t(y) * img.width + x) * 3]; }

// ---- Bracket enclosure -------------------------------------------------------------------------------------

// The range of E that keeps the scene inside a bracket's calibrated range. The bracket's grey wedge was
// calibrated over green values g_min .. g_max in its own capture; E scales the scene's green 0.5 % and 99.5 %
// points (g_lo, g_hi, at the reference exposure) onto them:
//
//   e_min = g_min / g_lo * k,   e_max = g_max / g_hi * k,   k = reference exposure / the bracket's exposure.
//
// With e_min <= e_max (feasible) every E in between keeps the whole scene calibrated, and that is the range E is
// searched in. Otherwise the scene spans more than the bracket, and the search gets a window of 0.60 to 0.85 stop
// either side of e_center (within 1.5 stops of highlight and 1.2 stops of shadow tolerance).
struct Enclosure {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  bool feasible = false;    // e_min <= e_max
  bool admissible = true;   // the scene's span fits the bracket's span plus the 2.7 stops of tolerance
  double e_min = 0, e_max = 0;
  double search_lo = 0, search_hi = 0;  // the range E is searched in
  double e_center = 0;                  // sqrt(e_min * e_max)
  double slack_ev = 0;                  // log2(e_max / e_min): spare stops, negative when not feasible
  double fill_ratio = 0;                // (g_hi / g_lo) / (g_max / g_min): the share of the bracket the scene uses
};

// py-ref: negicc_station/src/optimizer.py @ 93be1ec (bracket_enclosure)
bool bracket_enclosure(const BundleTarget& tgt, int target_idx, double g_lo, double g_hi, Enclosure& r) {
  if (tgt.patches_g.empty()) return false;
  const double tgt_min = tgt.g_min, tgt_max = tgt.g_max;
  const double k_t = (REFERENCE_SHUTTER_S * (REFERENCE_ISO / 100.0)) / tgt.exposure_s;
  r.target_idx = target_idx;
  r.e_min = (tgt_min / g_lo) * k_t;
  r.e_max = (tgt_max / g_hi) * k_t;
  r.e_center = std::sqrt(r.e_min * r.e_max);
  r.slack_ev = std::log2(r.e_max / r.e_min);
  r.feasible = r.e_max >= r.e_min;
  const double usable_hi_ev = 1.5, usable_sh_ev = 1.2;
  const double u_min = r.e_min * std::pow(2.0, -usable_hi_ev);
  const double u_max = r.e_max * std::pow(2.0, usable_sh_ev);
  const double usable_capacity_ev = std::log2(tgt_max / tgt_min) + usable_hi_ev + usable_sh_ev;
  r.admissible = std::log2(g_hi / g_lo) <= usable_capacity_ev;
  if (r.feasible) {
    r.search_lo = r.e_min;
    r.search_hi = r.e_max;
  } else {
    const double margin_ev = std::max(0.60, std::min(0.85, 0.60 * std::fabs(r.slack_ev)));
    r.search_lo = std::max(u_min, r.e_center * std::pow(2.0, -margin_ev));
    r.search_hi = std::min(u_max, r.e_center * std::pow(2.0, margin_ev));
  }
  r.fill_ratio = (g_hi / g_lo) / (tgt_max / tgt_min);
  return true;
}

// Brackets whose grey wedge spans at least 0.75 of the largest tonal capacity (at least three kept), when the
// bundle has five or more.
// py-ref: negicc_station/src/target_selection.py @ 93be1ec (filter_admissible_targets)
std::vector<int> filter_admissible_targets(const ProfileBundle& bundle) {
  const int n = bundle.targets.size();
  std::vector<int> out;
  if (n < 5) {
    for (int i = 0; i < n; ++i) out.push_back(i);
    return out;
  }
  double max_c = 0.0;
  for (const auto& t : bundle.targets) max_c = std::max(max_c, t.tonal_capacity_ev);
  for (int i = 0; i < n; ++i) {
    if (bundle.targets[i].tonal_capacity_ev >= 0.75 * max_c) out.push_back(i);
  }
  if (out.size() < 3) {
    std::vector<int> ranked(n);
    for (int i = 0; i < n; ++i) ranked[i] = i;
    std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
      return bundle.targets[a].tonal_capacity_ev > bundle.targets[b].tonal_capacity_ev;
    });
    out.assign(ranked.begin(), ranked.begin() + 3);
    std::sort(out.begin(), out.end());
  }
  return out;
}

// ---- Grey point (metering) ---------------------------------------------------------------------------------

// Least squares solution of the 3x3 system J x = r with singular values below rcond * max dropped (numpy lstsq).
void lstsq3(const double J[3][3], const double r[3], double rcond, double x[3]) {
  double A[3][3] = {}, V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) A[i][j] += J[k][i] * J[k][j];
  for (int sweep = 0; sweep < 50; ++sweep) {  // Jacobi eigen decomposition of J^T J
    double off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
    if (off < 1e-300) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (A[p][q] == 0.0) continue;
        const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 3; ++k) {
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  double jt_r[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) jt_r[i] += J[k][i] * r[k];
  const double smax = std::sqrt(std::max({A[0][0], A[1][1], A[2][2], 0.0}));
  x[0] = x[1] = x[2] = 0.0;
  for (int e = 0; e < 3; ++e) {
    const double lambda = A[e][e];
    if (!(lambda > 0.0) || std::sqrt(lambda) <= rcond * smax) continue;
    double proj = 0.0;
    for (int k = 0; k < 3; ++k) proj += V[k][e] * jt_r[k];
    for (int k = 0; k < 3; ++k) x[k] += V[k][e] * proj / lambda;
  }
}

// The pipeline's response to synthetic camera-linear inputs for one bracket (identity matrix, knee 0.95).
// py-ref: negicc_station/src/grey_point.py @ 93be1ec (GreyPoint)
class GreyPoint {
 public:
  explicit GreyPoint(const ProfileData& prof) : prof_(prof) {}

  // Lab of inputs t (n x 3), the scale going through E so the 16-bit inputs sit near 0.9.
  void lab(const double (*t)[3], int n, double (*out)[3]) const {
    double tmax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) tmax = std::max(tmax, t[i][c]);
    const double s = tmax / 0.9;
    static const float kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float m[9];
    candidate_matrix(kIdentity, s, 1.0f, 1.0f, m);
    for (int i = 0; i < n; ++i) {
      uint16_t px[3];
      for (int c = 0; c < 3; ++c) {
        const float v = t[i][c] / s;
        const float vv = std::isfinite(v) ? v : 0.0f;
        px[c] = std::lround(std::fmin(1.0f, std::fmax(0.0f, vv)) * 65535.0f);
      }
      const Rgb o = convert_pixel_srgb(px, m, AUTO_KNEE, false, 1.0f, prof_);
      const float rgb[3] = {o.r, o.g, o.b};
      srgb_to_lab_d(rgb, out[i]);
    }
  }

  double lstar(const double t[3]) const {
    const double in[1][3] = {{t[0], t[1], t[2]}};
    double lab_out[1][3];
    lab(in, 1, lab_out);
    return lab_out[0][0];
  }

  // The input rendered as neutral L* 50: a coarse scan along an orange-mask-like line, then Newton. False when the
  // bracket has none within 0.25 dE.
  bool reference_grey() const {
    const double dir[3] = {1.4, 1.0, 0.88};
    double best_k = 0.0, best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 27; ++i) {
      const double k = std::pow(2.0, -10.0 + 0.5 * i);
      const double t[3] = {k * dir[0], k * dir[1], k * dir[2]};
      const double d = std::fabs(lstar(t) - 50.0);
      if (d < best_d) {
        best_d = d;
        best_k = k;
      }
    }
    double log_t[3];
    for (int c = 0; c < 3; ++c) log_t[c] = std::log(best_k * dir[c]);
    const double target[3] = {50.0, 0.0, 0.0};
    return newton(target, log_t, 20) <= 0.25;
  }

 private:
  // Solves Lab(exp(log_t)) = target; returns the error in Lab units.
  double newton(const double target[3], double log_t[3], int iters) const {
    constexpr double FD_STEP = 0.05, MAX_LOG_STEP = 0.25, TOLERANCE = 0.02;
    for (int it = 0; it < iters; ++it) {
      double pts[7][3];
      for (int c = 0; c < 3; ++c) pts[0][c] = std::exp(log_t[c]);
      for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < 2; ++k) {
          const double sgn = k == 0 ? 1.0 : -1.0;
          for (int j = 0; j < 3; ++j) pts[1 + 2 * c + k][j] = std::exp(log_t[j] + (j == c ? sgn * FD_STEP : 0.0));
        }
      }
      double l[7][3];
      lab(pts, 7, l);
      const double r[3] = {target[0] - l[0][0], target[1] - l[0][1], target[2] - l[0][2]};
      if (std::max({std::fabs(r[0]), std::fabs(r[1]), std::fabs(r[2])}) < TOLERANCE) {
        return std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      }
      double J[3][3];
      for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row) J[row][c] = (l[1 + 2 * c][row] - l[2 + 2 * c][row]) / (2.0 * FD_STEP);
      double step[3];
      lstsq3(J, r, 1e-6, step);
      for (int c = 0; c < 3; ++c) log_t[c] += std::clamp(step[c], -MAX_LOG_STEP, MAX_LOG_STEP);
    }
    double t[1][3], l[1][3];
    for (int c = 0; c < 3; ++c) t[0][c] = std::exp(log_t[c]);
    lab(t, 1, l);
    const double r[3] = {target[0] - l[0][0], target[1] - l[0][1], target[2] - l[0][2]};
    return std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  }

  const ProfileData& prof_;
};

// The E (see ConversionParams) at which the scene vector p, with gains g and b, renders at L* 50 through the
// bracket: the metered exposure. A bisection on log2 E over [-6, 6] that stops at |L* - 50| <= 0.1. Its 25
// halvings, the station's count, resolve E to 12 / 2^25 stop, far below that tolerance, so they only cap the loop.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (exposure_for_lightness)
bool exposure_for_lightness(const GreyPoint& gp, const double p[3], double g, double b, double& e_out) {
  constexpr double TARGET = 50.0, TOL = 0.1;
  constexpr int MAX_BISECTIONS = 25;
  const double base_t[3] = {p[0], p[1] * g, p[2] * b};
  double lo = -6.0, hi = 6.0;
  for (int i = 0; i < MAX_BISECTIONS; ++i) {
    const double mid = 0.5 * (lo + hi);
    const double E = std::pow(2.0, mid);
    const double t[3] = {E * base_t[0], E * base_t[1], E * base_t[2]};
    const double L = gp.lstar(t);
    if (std::fabs(L - TARGET) <= TOL) {
      e_out = E;
      return true;
    }
    if (L > TARGET) lo = mid;
    else hi = mid;
  }
  const double E = std::pow(2.0, 0.5 * (lo + hi));
  const double t[3] = {E * base_t[0], E * base_t[1], E * base_t[2]};
  e_out = E;
  return std::fabs(gp.lstar(t) - TARGET) <= TOL;
}

// ---- Frame statistics --------------------------------------------------------------------------------------

// Statistics of a frame's inner crop: the green 0.5 % / 99.5 % points, the log-average RGB and the crop averaged
// onto the DINOv3 patch grid (one pixel per patch).
// py-ref: negicc_station/src/image_capture.cpp @ 93be1ec (FrameView::raw_stats)
struct FrameStats {
  float g_lo = 0.0f, g_hi = 0.0f;
  double log_avg_rgb[3] = {0, 0, 0};  // geometric mean in [0, 1]
  int grid_w = 0, grid_h = 0;
  std::vector<uint16_t> grid;  // grid_w x grid_h RGB
};

double u16_percentile_linear(const std::vector<uint64_t>& hist, uint64_t total, double q) {
  if (total == 0) return 0.0;
  const double pos = q * (total - 1);
  const uint64_t idx_lo = std::floor(pos);
  const uint64_t idx_hi = std::ceil(pos);
  const double frac = pos - idx_lo;
  uint64_t cum = 0;
  int v_lo = -1, v_hi = -1;
  for (int v = 0; v < 65536; ++v) {
    cum += hist[v];
    if (v_lo < 0 && cum > idx_lo) v_lo = v;
    if (v_hi < 0 && cum > idx_hi) {
      v_hi = v;
      break;
    }
  }
  if (v_lo < 0) v_lo = 65535;
  if (v_hi < 0) v_hi = 65535;
  return (1.0 - frac) * v_lo + frac * v_hi;
}

void compute_frame_stats(const RgbImage& frame, float crop_ratio, int max_size, int patch_size, FrameStats& st) {
  const InnerCrop crop = compute_inner_crop(frame.width, frame.height, crop_ratio);
  // compute_aspect_preserved_shape of color_pipeline.h (round half away from zero)
  const int p = std::max(1, patch_size);
  const int crop_w = std::max(crop.w, 1), crop_h = std::max(crop.h, 1);
  const double scale = std::max(max_size, p) / double(std::max(crop_w, crop_h));
  const int gw = std::max(1, int(std::round(crop_w * scale / p)));
  const int gh = std::max(1, int(std::round(crop_h * scale / p)));
  const int dino_w = gw * p, dino_h = gh * p;

  std::vector<double> log_lut(65536);
  for (int v = 0; v < 65536; ++v) log_lut[v] = std::log(std::max(v, 1));

  std::vector<uint64_t> g_hist(65536, 0);
  std::vector<double> row_logs(crop.h * 3, 0.0);
#pragma omp parallel
  {
    std::vector<uint64_t> loc(65536, 0);
#pragma omp for schedule(static)
    for (int ry = 0; ry < crop.h; ++ry) {
      const uint16_t* row = pixel_at(frame, crop.x0, crop.y0 + ry);
      double lr = 0.0, lg = 0.0, lb = 0.0;
      for (int rx = 0; rx < crop.w; ++rx, row += 3) {
        loc[row[1]]++;
        lr += log_lut[row[0]];
        lg += log_lut[row[1]];
        lb += log_lut[row[2]];
      }
      row_logs[ry * 3] = lr;
      row_logs[ry * 3 + 1] = lg;
      row_logs[ry * 3 + 2] = lb;
    }
#pragma omp critical
    for (int v = 0; v < 65536; ++v) g_hist[v] += loc[v];
  }
  const uint64_t npx = uint64_t(crop.w) * crop.h;
  if (npx > 0) {
    double tot[3] = {0, 0, 0};
    for (int ry = 0; ry < crop.h; ++ry)
      for (int c = 0; c < 3; ++c) tot[c] += row_logs[ry * 3 + c];
    for (int c = 0; c < 3; ++c) st.log_avg_rgb[c] = std::exp(tot[c] * (1.0 / npx)) / 65535.0;
    const double p_lo = u16_percentile_linear(g_hist, npx, SCENE_TAIL);
    const double p_hi = u16_percentile_linear(g_hist, npx, 1.0 - SCENE_TAIL);
    st.g_lo = std::max(1.0, p_lo);
    st.g_hi = std::max(st.g_lo + 1.0, p_hi);
  }

  // Inner crop resampled to the DINOv3 input size in double (area average, bilinear when enlarging), then each
  // patch averaged into the grid.
  std::vector<double> dino(dino_w * dino_h * 3, 0.0);
  const double sx = crop.w / double(dino_w);
  const double sy = crop.h / double(dino_h);
  if (sx >= 1.0 && sy >= 1.0) {
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      const double y0 = dy * sy, y1 = (dy + 1) * sy;
      const int ys = std::floor(y0), ye = std::min<int>(crop.h, std::ceil(y1));
      for (int dx = 0; dx < dino_w; ++dx) {
        const double x0 = dx * sx, x1 = (dx + 1) * sx;
        const int xs = std::floor(x0), xe = std::min<int>(crop.w, std::ceil(x1));
        double acc[3] = {0, 0, 0}, wsum = 0.0;
        for (int yy = ys; yy < ye; ++yy) {
          const double wy = std::min(y1, yy + 1.0) - std::max(y0, yy + 0.0);
          if (wy <= 0.0) continue;
          const uint16_t* row = pixel_at(frame, crop.x0, crop.y0 + yy);
          for (int xx = xs; xx < xe; ++xx) {
            const double wx = std::min(x1, xx + 1.0) - std::max(x0, xx + 0.0);
            if (wx <= 0.0) continue;
            const double w = wx * wy;
            acc[0] += row[xx * 3] * w;
            acc[1] += row[xx * 3 + 1] * w;
            acc[2] += row[xx * 3 + 2] * w;
            wsum += w;
          }
        }
        const double inv = (wsum > 0.0) ? (1.0 / wsum) : 0.0;
        for (int c = 0; c < 3; ++c) dino[(dy * dino_w + dx) * 3 + c] = acc[c] * inv;
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      const double fy = (dy + 0.5) * sy - 0.5;
      const int y0 = std::clamp<int>(std::floor(fy), 0, crop.h - 1);
      const int y1 = std::clamp(y0 + 1, 0, crop.h - 1);
      const double wy1 = fy - y0, wy0 = 1.0 - wy1;
      const uint16_t* row0 = pixel_at(frame, crop.x0, crop.y0 + y0);
      const uint16_t* row1 = pixel_at(frame, crop.x0, crop.y0 + y1);
      for (int dx = 0; dx < dino_w; ++dx) {
        const double fx = (dx + 0.5) * sx - 0.5;
        const int x0 = std::clamp<int>(std::floor(fx), 0, crop.w - 1);
        const int x1 = std::clamp(x0 + 1, 0, crop.w - 1);
        const double wx1 = fx - x0, wx0 = 1.0 - wx1;
        for (int c = 0; c < 3; ++c) {
          const double top = row0[x0 * 3 + c] * wx0 + row0[x1 * 3 + c] * wx1;
          const double bot = row1[x0 * 3 + c] * wx0 + row1[x1 * 3 + c] * wx1;
          dino[(dy * dino_w + dx) * 3 + c] = top * wy0 + bot * wy1;
        }
      }
    }
  }
  st.grid_w = gw;
  st.grid_h = gh;
  st.grid.assign(gw * gh * 3, 0);
  const double inv_patch = 1.0 / (p * p);
  for (int gy = 0; gy < gh; ++gy) {
    for (int gx = 0; gx < gw; ++gx) {
      double acc[3] = {0, 0, 0};
      for (int py = 0; py < p; ++py) {
        const double* drow = &dino[((gy * p + py) * dino_w + gx * p) * 3];
        for (int px = 0; px < p; ++px, drow += 3) {
          acc[0] += drow[0];
          acc[1] += drow[1];
          acc[2] += drow[2];
        }
      }
      for (int c = 0; c < 3; ++c) {
        st.grid[(gy * gw + gx) * 3 + c] = std::clamp(std::round(acc[c] * inv_patch), 0.0, 65535.0);
      }
    }
  }
}

// ---- Highlight ceiling and clipping ------------------------------------------------------------------------

// Per-channel 8-bit sRGB ceiling of a bracket: the brightest value any of its calibration patches renders to
// through its profile, read from the profcheck report the bundle stores; 255 without a report. Highlights above
// it are outside what the bracket was calibrated on.
// TODO: read the ceiling from the bundle's metadata JSON (exported by the station) instead of parsing the report.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (profile_highlight_ceiling)
std::array<double, 3> profile_highlight_ceiling(const std::string& profcheck_output) {
  static const std::regex kRow(R"(\]\s+(\S+):\s+\S+\s+\S+\s+\S+\s+->\s+(\S+)\s+(\S+)\s+(\S+)\s+should be)");
  const double inf = std::numeric_limits<double>::infinity();
  std::array<double, 3> ceil = {-inf, -inf, -inf};
  bool any = false;
  for (auto it = std::sregex_iterator(profcheck_output.begin(), profcheck_output.end(), kRow);
       it != std::sregex_iterator(); ++it) {
    double lab[3], rgb[3];
    try {
      for (int c = 0; c < 3; ++c) lab[c] = std::stod((*it)[2 + c].str());
    } catch (const std::exception&) {
      continue;
    }
    pcs_lab_to_srgb_u8(lab, rgb);
    for (int c = 0; c < 3; ++c) ceil[c] = std::max(ceil[c], rgb[c]);
    any = true;
  }
  if (!any) ceil = {255.0, 255.0, 255.0};
  return ceil;
}

// L* histogram of kLabBins bins over [0, 100].
constexpr int kLabBins = 1000;

int lab_bin(float L) { return std::clamp(static_cast<int>(L * (kLabBins / 100.0f)), 0, kLabBins - 1); }

double percentile_from_hist_L(const std::vector<uint64_t>& hist, double tot, double q) {
  if (tot <= 0.0) return 0.0;
  const double target = tot * q;
  double cum = 0.0;
  int k = 0;
  for (; k < kLabBins; ++k) {
    cum += hist[k];
    if (cum >= target) break;
  }
  return (std::min(k, kLabBins - 1) + 0.5) * (100.0 / kLabBins);
}

// Highlight clipping of one render, in percent of the pixels measured (float, as the station's kernel).
struct ClipStats {
  float exc[3];      // at or above the bracket's ceiling, per channel
  float white_clip;  // with any channel white-clipped
  float median_L;    // the median L*
};

// The highlight clipping of img's crop rendered with each conversion matrix.
// py-ref: negicc_station/src/pipeline/converter.cpp @ 93be1ec (Converter::highlight_clip)
// py-ref: negicc_station/src/pipeline/kernels_cpu.cpp @ 93be1ec (cpu_highlight_clip)
std::vector<ClipStats> measure_highlight_clip(const RgbImage& img, const InnerCrop& crop_in,
                                              const std::vector<std::array<float, 9>>& matrices, float knee,
                                              const ProfileData& prof, const std::array<double, 3>& ceil_rgb) {
  std::vector<ClipStats> res(matrices.size(), ClipStats{{0, 0, 0}, 0, 0});
  const InnerCrop crop = (crop_in.w > 0 && crop_in.h > 0) ? crop_in : InnerCrop{0, 0, img.width, img.height};
  const double tot = double(crop.w) * crop.h;
  if (tot <= 0) return res;
  float ceil_lin[3];
  for (int k = 0; k < 3; ++k) {
    const float ceil = ceil_rgb[k];
    const float c_u8 = (ceil > 1.0001f) ? ceil : (ceil * 255.0f);
    const float thr_u8 = std::min(std::floor(c_u8) + 1.0f, 255.0f);
    ceil_lin[k] = srgb_decode(std::max(0.0f, (thr_u8 - 0.5f) / 255.0f));
  }
  for (size_t c = 0; c < matrices.size(); ++c) {
    const float* m = matrices[c].data();
    uint64_t exc[3] = {0, 0, 0}, wc = 0;
    std::vector<uint64_t> hist(kLabBins, 0);
#pragma omp parallel
    {
      uint64_t l_exc[3] = {0, 0, 0}, l_wc = 0;
      std::vector<uint64_t> l_hist(kLabBins, 0);
#pragma omp for schedule(static)
      for (int y = crop.y0; y < crop.y0 + crop.h; ++y) {
        const uint16_t* row = pixel_at(img, crop.x0, y);
        for (int x = 0; x < crop.w; ++x, row += 3) {
          const Rgb lin = convert_pixel_linear(row, m, knee, prof);
          if (lin.r >= ceil_lin[0]) l_exc[0]++;
          if (lin.g >= ceil_lin[1]) l_exc[1]++;
          if (lin.b >= ceil_lin[2]) l_exc[2]++;
          if (linear_channel_white_clipped(lin.r) || linear_channel_white_clipped(lin.g) ||
              linear_channel_white_clipped(lin.b)) {
            l_wc++;
          }
          l_hist[lab_bin(linear_srgb_to_L(lin))]++;
        }
      }
#pragma omp critical
      {
        for (int k = 0; k < 3; ++k) exc[k] += l_exc[k];
        wc += l_wc;
        for (int k = 0; k < kLabBins; ++k) hist[k] += l_hist[k];
      }
    }
    for (int k = 0; k < 3; ++k) res[c].exc[k] = exc[k] / tot * 100.0;
    res[c].white_clip = wc / tot * 100.0;
    res[c].median_L = percentile_from_hist_L(hist, tot, 0.50);
  }
  return res;
}

// The largest per-channel share (0..1) of a render above the bracket's ceiling.
double max_excess(const ClipStats& c) { return std::max({c.exc[0], c.exc[1], c.exc[2]}) / 100.0; }

// ---- The frame the search works on -------------------------------------------------------------------------

// The conversion matrix of params over an input matrix: rows scaled by E, E g and E b. The pixel pipeline takes
// float matrices.
std::array<float, 9> conversion_matrix(const float input[9], const ConversionParams& p) {
  std::array<float, 9> m;
  candidate_matrix(input, p.e, p.g, p.b, m.data());
  return m;
}

// The frame cut out of the half-size decode in the output's orientation, its inner crop, statistics and input
// matrix, and the measurements every step of the search makes on them: previews, DINOv3, the residual, the
// coordinate descent and the highlight clipping. Their time is added up into the result.
class SearchFrame {
 public:
  SearchFrame(const ProfileBundle& bundle, DinoV3Engine& engine, bool verbose, ParamSearchResult& out)
      : bundle_(bundle), engine_(engine), verbose_(verbose), out_(out) {}

  // Cuts the frame (bbox in the half-size decode) and its inner crop, oriented, and takes the statistics.
  void cut(const RgbImage& half, const int bbox[4], int rot_cw, bool hflip, bool vflip, float crop_ratio) {
    crop_ratio_ = crop_ratio;
    cut_and_orient(half, bbox[0], bbox[1], bbox[2], bbox[3], rot_cw, hflip, vflip, frame_);
    compute_frame_stats(frame_, crop_ratio, engine_.max_size(), engine_.patch_size(), stats_);
    const InnerCrop in = compute_inner_crop(bbox[2], bbox[3], crop_ratio);
    cut_and_orient(half, bbox[0] + in.x0, bbox[1] + in.y0, in.w, in.h, rot_cw, hflip, vflip, inner_crop_);
  }

  // Sets the merged input matrix of this frame (ParamSearchResult: M_frame) and e_norm; the search runs on
  // M_frame / e_norm.
  void set_frame_matrix(const double frame_matrix[9], double e_norm) {
    std::copy(frame_matrix, frame_matrix + 9, frame_matrix_);
    e_norm_ = e_norm;
    for (int i = 0; i < 9; ++i) input_matrix_[i] = frame_matrix_[i] / e_norm_;
  }

  const RgbImage& inner_crop() const { return inner_crop_; }
  const FrameStats& stats() const { return stats_; }
  const double* frame_matrix() const { return frame_matrix_; }
  const float* input_matrix() const { return input_matrix_; }
  double e_norm() const { return e_norm_; }

  // The inner crop rendered to 8-bit sRGB through bracket target_idx at params: what DINOv3 sees.
  void render_preview(int target_idx, const ConversionParams& params, std::vector<uint8_t>& out) {
    const double t0 = now_ms();
    const std::array<float, 9> m = conversion_matrix(input_matrix_, params);
    const ProfileData& prof = bundle_.targets[target_idx].film_prof.data;
    const size_t n = inner_crop_.rgb.size() / 3;
    out.resize(inner_crop_.rgb.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
      const Rgb v = convert_pixel_srgb(&inner_crop_.rgb[i * 3], m.data(), AUTO_KNEE, false, 1.0f, prof);
      out[i * 3] = quantize_u8(v.r);
      out[i * 3 + 1] = quantize_u8(v.g);
      out[i * 3 + 2] = quantize_u8(v.b);
    }
    render_ms_ += now_ms() - t0;
  }

  // The intent field (mu, sigma) DINOv3 predicts from a preview, on the statistics' grid.
  bool predict(const std::vector<uint8_t>& preview, std::vector<float>& mu, std::vector<float>& sigma) {
    const double t0 = now_ms();
    DinoV3InferenceResult r;
    try {
      r = engine_.infer(preview.data(), inner_crop_.width, inner_crop_.height, inner_crop_.width * 3);
    } catch (const std::exception& e) {
      std::cerr << "ERROR! DINOv3 inference failed: " << e.what() << std::endl;
      return false;
    }
    infer_ms_ += now_ms() - t0;
    out_.inferences++;
    const size_t n = stats_.grid.size();
    if (r.grid_w != stats_.grid_w || r.grid_h != stats_.grid_h || r.mu.size() != n || r.sigma.size() != n) {
      std::cerr << "ERROR! DINOv3 intent field " << r.grid_w << "x" << r.grid_h << " does not match the frame grid "
                << stats_.grid_w << "x" << stats_.grid_h << std::endl;
      return false;
    }
    mu = std::move(r.mu);
    sigma = std::move(r.sigma);
    return true;
  }

  // The self-consistency residual of a preview against the field predicted from it (FixedPointResult::residual).
  double residual(const std::vector<uint8_t>& preview, const std::vector<float>& mu, const std::vector<float>& sigma) {
    const double t0 = now_ms();
    const double err = self_consistency_residual(preview.data(), inner_crop_.width, inner_crop_.height,
                                                 stats_.grid_w, stats_.grid_h, mu.data(), sigma.data());
    residual_ms_ += now_ms() - t0;
    return err;
  }

  // The coordinate descent of (E, g, b) of bracket target_idx against an intent field, from initial (or the
  // search's defaults when null) with E within e_bounds = {lo, hi, centre}; returns the intent loss
  // (ScreeningResult::intent_loss).
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (optimize_target_for_intent)
  double search(int target_idx, const std::vector<float>& mu, const std::vector<float>& sigma,
                const ConversionParams* initial, const double e_bounds[3], ConversionParams& out) {
    const double t0 = now_ms();
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    const double init[3] = {initial ? initial->e : 0.0, initial ? initial->g : 0.0, initial ? initial->b : 0.0};
    double res[3];
    const double loss = optimize_target_for_intent_d(cfg, input_matrix_, stats_.grid.data(), stats_.grid_w,
                                                     stats_.grid_h, mu.data(), sigma.data(),
                                                     initial ? init : nullptr, e_bounds, 0, res);
    out = {res[0], res[1], res[2]};
    search_ms_ += now_ms() - t0;
    return loss;
  }

  // The fixed-point loop of bracket target_idx (FixedPointResult), from start (its screening result) with E
  // kept within the trust region e_bounds = {lo, hi, centre}; the centre is where the search's penalty pulls E.
  // py-ref: negicc_station/src/per_target_conversion.py @ 93be1ec (_FrameContext.fixed_point)
  bool fixed_point(int target_idx, const ConversionParams& start, double e_bounds[3], FixedPointResult& fp) {
    if (e_bounds[0] > e_bounds[1]) {
      std::swap(e_bounds[0], e_bounds[1]);
      e_bounds[2] = (e_bounds[0] + e_bounds[1]) / 2.0;
    }
    ConversionParams curr = start, best;
    double prev_err = std::numeric_limits<double>::infinity();
    FixedPointStatus status = FixedPointStatus::kMaxIters;
    int steps = 0;
    bool have_best = false;
    double best_err = 0.0;
    std::vector<uint8_t> preview;
    std::vector<float> mu, sigma;
    for (int it = 0; it < MAX_STEPS; ++it) {
      render_preview(target_idx, curr, preview);
      if (!predict(preview, mu, sigma)) return false;
      const double err = residual(preview, mu, sigma);
      if (!have_best || err < best_err) {
        have_best = true;
        best_err = err;
        best = curr;
      }
      if (it > 0 && err > prev_err * 1.25 && (err - prev_err) > 1.5) {
        status = FixedPointStatus::kDiverged;
        steps++;
        break;
      }
      ConversionParams next;
      search(target_idx, mu, sigma, &curr, e_bounds, next);
      const double dp = std::fabs(next.e - curr.e) + std::fabs(next.g - curr.g) + std::fabs(next.b - curr.b);
      const double rel = it > 0 ? std::fabs(err - prev_err) / std::max(1e-4, prev_err) : 1.0;
      steps++;
      if (verbose_) {
        std::cout << "    T" << target_idx + 1 << " step " << it << ": E " << fixed(curr.e, 4) << " g "
                  << fixed(curr.g, 4) << " b " << fixed(curr.b, 4) << " resid " << fixed(err, 4) << "\n";
      }
      if (it > 0 && (dp <= TOL_PARAM || rel <= TOL_REL_ERR)) {
        status = FixedPointStatus::kConverged;
        break;
      }
      curr = next;
      prev_err = err;
    }
    fp.target_idx = target_idx;
    fp.params = best;
    fp.residual = best_err;
    fp.status = status;
    fp.steps = steps;
    return true;
  }

  // The highlight clipping of the frame's inner crop through bracket target_idx, per candidate (E, g, b).
  std::vector<ClipStats> highlight_clip(int target_idx, const std::vector<ConversionParams>& candidates,
                                        const std::array<double, 3>& ceil) {
    const InnerCrop crop = compute_inner_crop(frame_.width, frame_.height, crop_ratio_);
    std::vector<std::array<float, 9>> matrices;
    for (const ConversionParams& p : candidates) matrices.push_back(conversion_matrix(input_matrix_, p));
    return measure_highlight_clip(frame_, crop, matrices, AUTO_KNEE, bundle_.targets[target_idx].film_prof.data,
                                  ceil);
  }

  // Writes the time spent per measurement into the result, in whole milliseconds.
  void write_timings() const {
    out_.timing_render_ms = std::lround(render_ms_);
    out_.timing_infer_ms = std::lround(infer_ms_);
    out_.timing_search_ms = std::lround(search_ms_);
    out_.timing_residual_ms = std::lround(residual_ms_);
  }

 private:
  const ProfileBundle& bundle_;
  DinoV3Engine& engine_;
  const bool verbose_;
  ParamSearchResult& out_;

  float crop_ratio_ = 0.90f;
  RgbImage frame_;       // the frame within bbox, oriented as the output
  RgbImage inner_crop_;  // its centred inner crop (crop_ratio of each side): what DINOv3 sees
  FrameStats stats_;     // over the inner crop
  double frame_matrix_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // M_frame (ParamSearchResult)
  float input_matrix_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // M_frame / e_norm: the matrix the search runs on
  double e_norm_ = 1.0;
  double render_ms_ = 0, infer_ms_ = 0, search_ms_ = 0, residual_ms_ = 0;
};

// ---- Highlight headroom ------------------------------------------------------------------------------------

struct Headroom {
  HeadroomOutcome outcome = HeadroomOutcome::kWithinRange;
  double delta_ev = 0.0;  // the stops E is raised by
};

// The exposure increase (stops) that brings every output channel's (1 - SCENE_TAIL) point inside the bracket's
// calibrated range, bounded by the bracket's exposure capacity log2(e_max / E); a two-stage batched dyadic
// bisection over E at the fixed gains g and b.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (compute_highlight_headroom_ev)
Headroom highlight_headroom(SearchFrame& fr, int target_idx, const ConversionParams& params, bool has_e_max,
                            double e_max, const std::array<double, 3>& ceil) {
  constexpr double tail = SCENE_TAIL, tol_ev = 0.01;
  const double e = params.e;
  auto at_ev = [&](double ev) { return ConversionParams{e * std::pow(2.0, ev), params.g, params.b}; };
  Headroom h;
  const bool cap_known = has_e_max && e_max > 0 && e > 0;
  const double capacity_ev = cap_known ? std::log2(e_max / e) : 0.0;
  const bool has_cap = cap_known && capacity_ev > 0.0;
  std::vector<ConversionParams> init = {params};
  if (has_cap) init.push_back(at_ev(capacity_ev));
  const auto hc_init = fr.highlight_clip(target_idx, init, ceil);
  if (max_excess(hc_init[0]) <= tail) {
    h.outcome = HeadroomOutcome::kWithinRange;
    return h;
  }
  if (!has_cap) {
    h.outcome = HeadroomOutcome::kNoCapacity;
    return h;
  }
  if (max_excess(hc_init[1]) > tail) {
    h.outcome = HeadroomOutcome::kCapacityLimited;
    h.delta_ev = capacity_ev;
    return h;
  }
  // Stage 1 evaluates 2^s1 - 1 evenly spaced exposures in one batch and bisects over them; stage 2 does the same
  // inside the bracketing interval until it is narrower than tol_ev.
  const int total_steps = std::max(0, int(std::ceil(std::log2(std::max(capacity_ev / tol_ev, 1.0)))));
  const int stage1_steps = std::min(4, total_steps);
  double lo = 0.0, hi = capacity_ev;
  if (stage1_steps > 0) {
    const int n1 = 1 << stage1_steps;
    std::vector<ConversionParams> cands;
    for (int i = 1; i < n1; ++i) cands.push_back(at_ev(lo + (i / double(n1)) * (hi - lo)));
    const auto hc1 = fr.highlight_clip(target_idx, cands, ceil);
    int ilo = 0, ihi = n1;
    for (int s = 0; s < stage1_steps; ++s) {
      const int mid = (ilo + ihi) / 2;
      if (max_excess(hc1[mid - 1]) <= tail) ihi = mid;
      else ilo = mid;
    }
    lo = (ilo / double(n1)) * capacity_ev;
    hi = (ihi / double(n1)) * capacity_ev;
  }
  const int rem_steps = total_steps - stage1_steps;
  if (rem_steps > 0 && (hi - lo) > tol_ev) {
    const int n2 = 1 << rem_steps;
    std::vector<ConversionParams> cands;
    for (int i = 1; i < n2; ++i) cands.push_back(at_ev(lo + (i / double(n2)) * (hi - lo)));
    const auto hc2 = fr.highlight_clip(target_idx, cands, ceil);
    const double span = hi - lo, base_lo = lo;
    int ilo = 0, ihi = n2;
    for (int s = 0; s < rem_steps; ++s) {
      const int mid = (ilo + ihi) / 2;
      if (max_excess(hc2[mid - 1]) <= tail) ihi = mid;
      else ilo = mid;
    }
    lo = base_lo + (ilo / double(n2)) * span;
    hi = base_lo + (ihi / double(n2)) * span;
  }
  h.outcome = HeadroomOutcome::kPulledInside;
  h.delta_ev = hi;
  return h;
}

// The merged input matrix M_frame (ParamSearchResult): the crosstalk rows scaled by normalization_target / film
// base and by the exposure ratio between the film base capture and the scan.
// py-ref: negicc_station/src/film_profiling.py @ 93be1ec (compute_merged_crosstalk, compute_exposure_ratio)
void merged_matrix(const ProfileBundle& bundle, const double fb[3], double exposure_ratio, double out[9]) {
  for (int r = 0; r < 3; ++r) {
    const double scale = fb[r] > 0 ? (bundle.normalization_target / fb[r]) * exposure_ratio : 1.0;
    for (int c = 0; c < 3; ++c) out[r * 3 + c] = bundle.crosstalk_matrix_d[r * 3 + c] * scale;
  }
}

// A screened bracket with its enclosure.
struct Screened {
  ScreeningResult result;
  Enclosure enclosure;
};

}  // namespace

// ---- Public API ----------------------------------------------------------------------------------------------

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& half,
                              const int bbox_in[4], int rot_cw, bool hflip, bool vflip, double scan_shutter_s,
                              double scan_iso, const ParamSearchOptions& opts, ParamSearchResult& out) {
  const double t_start = now_ms();
  const bool verbose = opts.verbose;
  if (bundle.targets.empty()) {
    std::cerr << "ERROR! The profile bundle has no targets" << std::endl;
    return false;
  }
  SearchFrame fr(bundle, engine, verbose, out);
  auto name_of = [&](int target_idx) { return bundle.targets[target_idx].name; };

  // The frame and its inner crop, oriented.
  int bbox[4] = {bbox_in[0], bbox_in[1], bbox_in[2], bbox_in[3]};
  clamp_rect(bbox, half.width, half.height);
  std::copy(bbox, bbox + 4, out.bbox);
  fr.cut(half, bbox, rot_cw, hflip, vflip, opts.crop_ratio);
  const FrameStats& stats = fr.stats();

  // Step 1: the enclosure of each bracket.
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (compute_gamut_volume_enclosure)
  const double g_lo = std::max<double>(1.0, stats.g_lo);
  const double g_hi = std::max<double>(g_lo + 1.0, stats.g_hi);
  out.g_lo = g_lo;
  out.g_hi = g_hi;
  out.scene_ev = std::log2(g_hi / g_lo);
  std::vector<Enclosure> enclosure;
  for (int i : filter_admissible_targets(bundle)) {
    Enclosure r;
    if (bracket_enclosure(bundle.targets[i], i, g_lo, g_hi, r)) enclosure.push_back(r);
  }
  if (enclosure.empty()) {
    std::cerr << "ERROR! No bracket of the bundle has patch readings" << std::endl;
    return false;
  }

  // The input matrix in reference units (ParamSearchResult).
  const double* fb = opts.film_base_rgb ? opts.film_base_rgb : bundle.film_base_avg;
  double fb_t = 0.125;
  try {
    fb_t = shutter_to_seconds(bundle.film_base_shutter);
  } catch (const std::exception&) {
  }
  const double fb_exposure = fb_t * (bundle.film_base_iso / 100.0);
  // A measured base comes from this capture: no exposure ratio.
  const double ratio_frame = opts.film_base_rgb ? 1.0 : fb_exposure / (scan_shutter_s * (scan_iso / 100.0));
  const double ratio_ref = fb_exposure / (REFERENCE_SHUTTER_S * (REFERENCE_ISO / 100.0));
  double m_frame[9], m_ref[9];
  merged_matrix(bundle, fb, ratio_frame, m_frame);
  merged_matrix(bundle, bundle.film_base_avg, ratio_ref, m_ref);
  fr.set_frame_matrix(m_frame, (m_ref[4] != 0.0) ? m_frame[4] / m_ref[4] : 1.0);
  const double e_norm = fr.e_norm();
  out.e_norm = e_norm;
  std::copy(fr.input_matrix(), fr.input_matrix() + 9, out.input_matrix);
  for (int r = 0; r < 3; ++r) {
    out.row_scale[r] = (fb[r] > 0 ? (bundle.normalization_target / fb[r]) * ratio_frame : 1.0) / e_norm;
  }

  if (verbose) {
    std::ostringstream os;
    os << "Auto: frame " << bbox[2] << "x" << bbox[3] << " at (" << bbox[0] << ", " << bbox[1] << ") (half size), crop "
       << fr.inner_crop().width << "x" << fr.inner_crop().height << ", grid " << stats.grid_w << "x" << stats.grid_h
       << "\n";
    os << "Auto: scene green " << fixed(g_lo, 1) << ".." << fixed(g_hi, 1) << " (" << fixed(out.scene_ev, 2)
       << " EV), scan " << general(scan_shutter_s, 6) << " s ISO " << fixed(scan_iso, 0) << ", e_norm "
       << general(e_norm, 6) << "\n";
    os << "  bracket            E_min     E_max  slack_EV   fill  feasible admissible\n";
    for (const auto& r : enclosure) {
      os << "  " << std::left << std::setw(14) << name_of(r.target_idx) << std::right << "  " << std::setw(8)
         << fixed(r.e_min, 4) << "  " << std::setw(8) << fixed(r.e_max, 4) << "  " << std::setw(8)
         << fixed(r.slack_ev, 3, true) << "  " << std::setw(5) << fixed(r.fill_ratio, 3) << "  " << std::setw(8)
         << (r.feasible ? "yes" : "no") << " " << std::setw(10) << (r.admissible ? "yes" : "no") << "\n";
    }
    std::cout << os.str();
  }

  auto rank = [](const Enclosure& r) {
    return std::fabs(r.fill_ratio - 0.85) + SLACK_PENALTY_SCALE * std::max(0.0, -r.slack_ev);
  };
  std::vector<Enclosure> ranked = enclosure;
  std::stable_sort(ranked.begin(), ranked.end(),
                   [&](const Enclosure& a, const Enclosure& b) { return rank(a) < rank(b); });
  std::vector<Enclosure> top;
  for (const auto& r : ranked)
    if (r.admissible && top.size() < 2) top.push_back(r);
  if (top.empty()) top.assign(ranked.begin(), ranked.begin() + std::min<size_t>(2, ranked.size()));

  // Step 2: meter each admissible bracket to L* 50 and bootstrap from the one deepest inside its bounds.
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (select_metered_bootstrap_candidate)
  std::vector<Enclosure> pool;
  for (const auto& r : enclosure)
    if (r.admissible) pool.push_back(r);
  if (pool.empty()) pool = enclosure;
  double p_scene[3];
  for (int r = 0; r < 3; ++r) {
    p_scene[r] = 0.0;
    for (int c = 0; c < 3; ++c) p_scene[r] += (fr.frame_matrix()[r * 3 + c] / e_norm) * stats.log_avg_rgb[c];
  }
  struct Depth {
    Enclosure cand;
    double e_met, depth, lo, hi;
  };
  std::vector<Depth> depths;
  for (const auto& c : pool) {
    const GreyPoint gp(bundle.targets[c.target_idx].film_prof.data);
    double e_met = 0.0;
    if (!gp.reference_grey()) {
      if (verbose) std::cout << "  " << name_of(c.target_idx) << ": no neutral L* 50 in this bracket, not metered\n";
      continue;
    }
    if (!exposure_for_lightness(gp, p_scene, 1.10, 1.15, e_met)) {
      if (verbose) std::cout << "  " << name_of(c.target_idx) << ": metering did not converge\n";
      continue;
    }
    const double lo = c.search_lo, hi = c.search_hi;
    const double depth = std::min(std::log2(e_met / std::max(1e-6, lo)), std::log2(std::max(1e-6, hi) / e_met)) -
                         SLACK_PENALTY_SCALE * std::max(0.0, -c.slack_ev);
    depths.push_back({c, e_met, depth, lo, hi});
  }
  Enclosure cand1;
  double e1_init;
  if (!depths.empty()) {
    std::stable_sort(depths.begin(), depths.end(), [](const Depth& a, const Depth& b) { return a.depth > b.depth; });
    cand1 = depths[0].cand;
    e1_init = std::clamp(depths[0].e_met, depths[0].lo, depths[0].hi);
  } else {
    cand1 = top[0];
    e1_init = cand1.e_center;
  }
  if (verbose) {
    std::ostringstream os;
    for (const auto& d : depths) {
      os << "  metered " << name_of(d.cand.target_idx) << ": E " << fixed(d.e_met, 4) << " depth "
         << fixed(d.depth, 3, true) << "\n";
    }
    os << "Auto: bootstrap from " << name_of(cand1.target_idx) << " at E " << fixed(e1_init, 4) << "\n";
    std::cout << os.str();
  }

  std::vector<uint8_t> preview;
  std::vector<float> mu1, sig1;
  fr.render_preview(cand1.target_idx, {e1_init, 1.10, 1.15}, preview);
  if (!fr.predict(preview, mu1, sig1)) return false;

  // Step 3: screen the brackets against the bootstrap field.
  std::vector<Screened> screening;
  auto screen = [&](const Enclosure& c, double lo, double hi, double e_met) {
    Screened s;
    s.enclosure = c;
    s.result.target_idx = c.target_idx;
    s.result.e_lo = lo;
    s.result.e_hi = hi;
    s.result.e_metered = e_met;
    const double eb[3] = {lo, hi, c.e_center};
    s.result.intent_loss = fr.search(c.target_idx, mu1, sig1, nullptr, eb, s.result.params);
    screening.push_back(s);
  };
  if (!depths.empty()) {
    for (const auto& d : depths) screen(d.cand, d.lo, d.hi, d.e_met);
  } else {
    for (const auto& c : pool) screen(c, c.search_lo, c.search_hi, c.e_center);
  }
  std::stable_sort(screening.begin(), screening.end(), [](const Screened& a, const Screened& b) {
    return a.result.intent_loss < b.result.intent_loss;
  });
  std::vector<Screened> shortlist;
  for (const auto& s : screening)
    if (s.result.intent_loss <= screening[0].result.intent_loss + 2.0 && shortlist.size() < 2) shortlist.push_back(s);
  if (shortlist.size() < 2 && screening.size() >= 2) shortlist.assign(screening.begin(), screening.begin() + 2);
  for (const auto& s : screening) out.screening.push_back(s.result);
  if (verbose) {
    std::ostringstream os;
    os << "Auto: screening\n";
    for (const auto& s : screening) {
      const ConversionParams& p = s.result.params;
      os << "  " << std::left << std::setw(14) << name_of(s.result.target_idx) << std::right << " E " << fixed(p.e, 4)
         << " g " << fixed(p.g, 4) << " b " << fixed(p.b, 4) << " loss " << fixed(s.result.intent_loss, 4) << "\n";
    }
    std::cout << os.str();
  }

  // Step 4: the fixed-point loop per shortlisted bracket, within a trust region of +/- 0.5 stop around its
  // screened E.
  auto trust_bounds = [](const Screened& c, double eb[3]) {
    eb[0] = std::max(c.result.e_lo, c.result.params.e * std::pow(2.0, -0.5));
    eb[1] = std::min(c.result.e_hi, c.result.params.e * std::pow(2.0, 0.5));
    eb[2] = c.enclosure.e_center;
  };
  std::vector<FixedPointResult> adaptive;
  for (const auto& c : shortlist) {
    double eb[3];
    trust_bounds(c, eb);
    FixedPointResult fp;
    if (verbose) std::cout << "Auto: fixed point on " << name_of(c.result.target_idx) << "\n";
    if (!fr.fixed_point(c.result.target_idx, c.result.params, eb, fp)) return false;
    adaptive.push_back(fp);
    out.tournament.push_back(fp);
  }
  // The lowest residual wins, among the converged brackets when any converged.
  bool any_converged = false;
  for (const auto& a : adaptive) any_converged |= (a.status == FixedPointStatus::kConverged);
  const FixedPointResult* win = nullptr;
  for (const auto& a : adaptive) {
    if (any_converged && a.status != FixedPointStatus::kConverged) continue;
    if (!win || a.residual < win->residual) win = &a;
  }
  FixedPointResult chosen = *win;
  ConversionParams params = chosen.params;

  // Step 5: the highlight headroom.
  auto bounds_of = [&](int target_idx, Enclosure& r) {
    for (const auto& c : enclosure)
      if (c.target_idx == target_idx) {
        r = c;
        return true;
      }
    return bracket_enclosure(bundle.targets[target_idx], target_idx, g_lo, g_hi, r);
  };
  std::map<int, std::array<double, 3>> ceilings;
  auto ceiling_of = [&](int target_idx) {
    auto it = ceilings.find(target_idx);
    if (it == ceilings.end()) {
      it = ceilings.emplace(target_idx, profile_highlight_ceiling(bundle.targets[target_idx].profcheck_output)).first;
    }
    return it->second;
  };
  auto apply_headroom = [&](int target_idx, ConversionParams& p) {
    Enclosure hb;
    const bool has_hb = bounds_of(target_idx, hb);
    const Headroom h = highlight_headroom(fr, target_idx, p, has_hb, has_hb ? hb.e_max : 0.0, ceiling_of(target_idx));
    if (h.delta_ev > 0.0) p.e *= std::pow(2.0, h.delta_ev);
    return h;
  };
  Headroom headroom = apply_headroom(chosen.target_idx, params);
  if (verbose) {
    std::cout << "Auto: winner " << name_of(chosen.target_idx) << " (resid " << fixed(chosen.residual, 4) << ", "
              << to_string(chosen.status) << "), highlight headroom " << fixed(headroom.delta_ev, 3, true) << " EV ("
              << to_string(headroom.outcome) << ")\n";
  }

  // Step 5: the highlight rescue (RescueOutcome).
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (measure_rescue_candidates, select_highlight_rescue)
  const ClipStats hc0 = fr.highlight_clip(chosen.target_idx, {params}, ceiling_of(chosen.target_idx))[0];
  const double clip_0 = hc0.white_clip / 100.0;
  out.rescue_white_clip = clip_0;
  out.rescue = RescueOutcome::kNotNeeded;
  if (clip_0 > RESCUE_CLIP_TAU) {
    const int t_adj = chosen.target_idx + 1;
    const Screened* s_adj = nullptr;
    for (const auto& s : screening)
      if (s.result.target_idx == t_adj) s_adj = &s;
    if (!s_adj) {
      out.rescue = RescueOutcome::kNotScreened;
    } else if (!s_adj->enclosure.admissible) {
      out.rescue = RescueOutcome::kNotAdmissible;
    } else {
      const FixedPointResult* run = nullptr;
      for (const auto& a : adaptive)
        if (a.target_idx == t_adj) run = &a;
      FixedPointResult fp_adj;
      if (run) {
        fp_adj = *run;
      } else {
        double eb[3];
        trust_bounds(*s_adj, eb);
        if (verbose) std::cout << "Auto: fixed point on " << name_of(t_adj) << " (highlight rescue)\n";
        if (!fr.fixed_point(t_adj, s_adj->result.params, eb, fp_adj)) return false;
      }
      ConversionParams p_adj = fp_adj.params;
      const Headroom h_adj = apply_headroom(t_adj, p_adj);
      const ClipStats hc_adj = fr.highlight_clip(t_adj, {p_adj}, ceiling_of(t_adj))[0];
      const double clip_adj = hc_adj.white_clip / 100.0;
      const double dl = double(hc_adj.median_L) - hc0.median_L;
      if (clip_adj > RESCUE_CLIP_RATIO * clip_0) out.rescue = RescueOutcome::kClipsTooMuch;
      else if (!(RESCUE_DL_MIN <= dl && dl <= RESCUE_DL_MAX)) out.rescue = RescueOutcome::kShiftsLightness;
      else if (fp_adj.residual > (1.0 + RESCUE_RESID_TOL) * chosen.residual) out.rescue = RescueOutcome::kWorseResidual;
      else out.rescue = RescueOutcome::kTaken;
      if (verbose) {
        std::cout << "Auto: highlight rescue: " << name_of(t_adj) << " clips " << fixed(clip_adj * 100.0, 4)
                  << "% vs " << fixed(clip_0 * 100.0, 4) << "%, dL* " << fixed(dl, 2, true) << ", resid "
                  << fixed(fp_adj.residual, 4) << ": " << to_string(out.rescue) << "\n";
      }
      if (out.rescue == RescueOutcome::kTaken) {
        chosen = fp_adj;
        params = p_adj;
        headroom = h_adj;
      }
    }
    if (verbose && (out.rescue == RescueOutcome::kNotScreened || out.rescue == RescueOutcome::kNotAdmissible)) {
      std::cout << "Auto: highlight rescue: " << to_string(out.rescue) << "\n";
    }
  }

  out.target_idx = chosen.target_idx;
  out.target_name = name_of(chosen.target_idx);
  out.params = {params.e / e_norm, params.g, params.b};
  out.residual = chosen.residual;
  out.status = chosen.status;
  out.headroom = headroom.outcome;
  out.headroom_ev = headroom.delta_ev;
  fr.write_timings();
  out.timing_total_ms = std::lround(now_ms() - t_start);
  return true;
}
