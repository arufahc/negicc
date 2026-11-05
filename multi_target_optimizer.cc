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

#include "multi_target_optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

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

const char* to_string(ClipFallbackOutcome outcome) {
  switch (outcome) {
    case ClipFallbackOutcome::kNotNeeded: return "not needed";
    case ClipFallbackOutcome::kNotScreened: return "next bracket not screened";
    case ClipFallbackOutcome::kNotAdmissible: return "next bracket not admissible";
    case ClipFallbackOutcome::kClipsTooMuch: return "clip ratio";
    case ClipFallbackOutcome::kShiftsLightness: return "L* shift";
    case ClipFallbackOutcome::kWorseResidual: return "residual";
    case ClipFallbackOutcome::kTaken: return "taken";
  }
  return "";
}

namespace {

// Fixed-point loop thresholds (see FixedPointStatus).
constexpr float TOL_PARAM = 0.02f;     // |dE| + |dg| + |db| <= 0.02
constexpr float TOL_REL_ERR = 0.015f;  // relative residual change <= 1.5 %
// The scene's range is taken between its SCENE_TAIL and 1 - SCENE_TAIL points.
constexpr float SCENE_TAIL = 0.005f;
constexpr float SLACK_PENALTY_SCALE = 0.5f;
// The previews, the screening and the final render use a fixed soft knee.
constexpr float AUTO_KNEE = 0.95f;
// Clip fallback thresholds (see ClipFallbackOutcome).
constexpr float CLIP_FALLBACK_CLIP_TAU = 3e-4f;
constexpr float CLIP_FALLBACK_CLIP_RATIO = 0.5f;
constexpr float CLIP_FALLBACK_DL_MIN = -4.5f, CLIP_FALLBACK_DL_MAX = 0.5f;
constexpr float CLIP_FALLBACK_RESID_TOL = 0.10f;
// The search runs as if the frame were scanned at 1/8 s, ISO 100 with the profile's film base.
constexpr float REFERENCE_SHUTTER_S = 0.125f;
constexpr float REFERENCE_ISO = 100.0f;

using Clock = std::chrono::steady_clock;

// v with prec decimals ("%.<prec>f", or "%+.<prec>f" with sign), for the verbose log.
std::string fixed(float v, int prec, bool sign = false) {
  std::ostringstream os;
  if (sign) os << std::showpos;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}

// v with prec significant digits ("%.<prec>g"), for the verbose log.
std::string general(float v, int prec) {
  std::ostringstream os;
  os << std::setprecision(prec) << v;
  return os.str();
}

// ---- The inner crop ---------------------------------------------------------------------------------------

using negicc::InnerCrop;
using negicc::compute_inner_crop;

// ---- Bracket enclosure -------------------------------------------------------------------------------------

// The range of E that keeps the scene inside a bracket's calibrated range. The bracket's grey wedge was
// calibrated over green values tgt_min_g .. tgt_max_g in its own capture; E scales the scene's green 0.5 % and 99.5 %
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
  float e_min = 0, e_max = 0;
  float search_lo = 0, search_hi = 0;  // the range E is searched in
  float e_center = 0;                  // sqrt(e_min * e_max)
  float slack_ev = 0;                  // log2(e_max / e_min): spare stops, negative when not feasible
  float fill_ratio = 0;                // (g_hi / g_lo) / (g_max / g_min): the share of the bracket the scene uses
};

// py-ref: negicc_station/src/optimizer.py @ 93be1ec (bracket_enclosure)
bool bracket_enclosure(const BundleTarget& tgt, int target_idx, float g_lo, float g_hi, Enclosure& r) {
  if (tgt.patches_g.empty()) return false;
  const float tgt_min = tgt.tgt_min_g, tgt_max = tgt.tgt_max_g;
  const float k_t = (REFERENCE_SHUTTER_S * (REFERENCE_ISO / 100.0f)) / tgt.exposure_s;
  r.target_idx = target_idx;
  r.e_min = (tgt_min / g_lo) * k_t;
  r.e_max = (tgt_max / g_hi) * k_t;
  r.e_center = std::sqrt(r.e_min * r.e_max);
  r.slack_ev = std::log2(r.e_max / r.e_min);
  r.feasible = r.e_max >= r.e_min;
  const float usable_hi_ev = 1.5f, usable_sh_ev = 1.2f;
  const float u_min = r.e_min * std::pow(2.0f, -usable_hi_ev);
  const float u_max = r.e_max * std::pow(2.0f, usable_sh_ev);
  const float usable_capacity_ev = std::log2(tgt_max / tgt_min) + usable_hi_ev + usable_sh_ev;
  r.admissible = std::log2(g_hi / g_lo) <= usable_capacity_ev;
  if (r.feasible) {
    r.search_lo = r.e_min;
    r.search_hi = r.e_max;
  } else {
    const float margin_ev = std::max(0.60f, std::min(0.85f, 0.60f * std::fabs(r.slack_ev)));
    r.search_lo = std::max(u_min, r.e_center * std::pow(2.0f, -margin_ev));
    r.search_hi = std::min(u_max, r.e_center * std::pow(2.0f, margin_ev));
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
  float max_c = 0.0f;
  for (const auto& t : bundle.targets) max_c = std::max(max_c, t.tonal_capacity_ev);
  for (int i = 0; i < n; ++i) {
    if (bundle.targets[i].tonal_capacity_ev >= 0.75f * max_c) out.push_back(i);
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
void lstsq3(const float J[3][3], const float r[3], float rcond, float x[3]) {
  float A[3][3] = {}, V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) A[i][j] += J[k][i] * J[k][j];
  for (int sweep = 0; sweep < 50; ++sweep) {  // Jacobi eigen decomposition of J^T J
    float off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
    if (off < 1e-30f) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (A[p][q] == 0.0f) continue;
        const float theta = (A[q][q] - A[p][p]) / (2.0f * A[p][q]);
        const float t = (theta >= 0 ? 1.0f : -1.0f) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
        const float c = 1.0f / std::sqrt(t * t + 1.0f), s = t * c;
        for (int k = 0; k < 3; ++k) {
          const float akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const float apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const float vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  float jt_r[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) jt_r[i] += J[k][i] * r[k];
  const float smax = std::sqrt(std::max({A[0][0], A[1][1], A[2][2], 0.0f}));
  x[0] = x[1] = x[2] = 0.0f;
  for (int e = 0; e < 3; ++e) {
    const float lambda = A[e][e];
    if (!(lambda > 0.0f) || std::sqrt(lambda) <= rcond * smax) continue;
    float proj = 0.0f;
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
  void lab(const float (*t)[3], int n, float (*out)[3]) const {
    float tmax = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) tmax = std::max(tmax, t[i][c]);
    const float s = tmax / 0.9f;
    static const float kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float m[9];
    negicc::candidate_matrix(kIdentity, s, 1.0f, 1.0f, m);
    const negicc::PipelineConfig pc = make_pipeline_config(prof_, AUTO_KNEE, 1.0f);
    for (int i = 0; i < n; ++i) {
      uint16_t px[3];
      for (int c = 0; c < 3; ++c) {
        const float v = t[i][c] / s;
        const float vv = std::isfinite(v) ? v : 0.0f;
        px[c] = std::lround(std::fmin(1.0f, std::fmax(0.0f, vv)) * 65535.0f);
      }
      const Lab lab = negicc::srgb_to_lab(negicc::convert_pixel(pc, m, px));
      out[i][0] = lab.L;
      out[i][1] = lab.a;
      out[i][2] = lab.b;
    }
  }

  float lstar(const float t[3]) const {
    const float in[1][3] = {{t[0], t[1], t[2]}};
    float lab_out[1][3];
    lab(in, 1, lab_out);
    return lab_out[0][0];
  }

  // Whether the bracket can render some camera-linear input as a neutral mid grey (L* 50, a* = b* = 0) within
  // 0.25 dE. The bootstrap does not meter a bracket that cannot.
  //
  // The start point: inputs k * (1.4, 1.0, 0.88), the green-normalised neutral input of the Portra brackets (a
  // colour like the orange mask, only a starting direction), for k = 2^-10 .. 2^3 in half stops; the k whose L* is
  // nearest 50 wins. newton() then moves all three channels freely from there to hit (50, 0, 0) exactly.
  bool reference_grey(float log_t0[3] = nullptr) const {
    const float dir[3] = {1.4f, 1.0f, 0.88f};
    float best_k = 0.0f, best_d = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 27; ++i) {
      const float k = std::pow(2.0f, -10.0f + 0.5f * i);
      const float t[3] = {k * dir[0], k * dir[1], k * dir[2]};
      const float d = std::fabs(lstar(t) - 50.0f);
      if (d < best_d) {
        best_d = d;
        best_k = k;
      }
    }
    float log_t[3];
    for (int c = 0; c < 3; ++c) log_t[c] = std::log(best_k * dir[c]);
    const float target[3] = {50.0f, 0.0f, 0.0f};
    const bool ok = newton(target, log_t, 20) <= 0.25f;
    if (log_t0) {
      for (int c = 0; c < 3; ++c) log_t0[c] = log_t[c];
    }
    return ok;
  }

  // Solves the gains (E, g, b) that move the grey that base renders as neutral L* 50 (at log_t0) to target = (L*, a*, b*).
  // Returns the residual dE76 of the Newton solve.
  float gains(const float log_t0[3], const float base[3], const float target[3], float egb_out[3]) const {
    float log_t[3] = {log_t0[0], log_t0[1], log_t0[2]};
    const float err = newton(target, log_t, 16);
    const float log_base[3] = {std::log(base[0]), std::log(base[0] * base[1]), std::log(base[0] * base[2])};
    const float G[3] = {
        std::exp(log_base[0] + log_t[0] - log_t0[0]),
        std::exp(log_base[1] + log_t[1] - log_t0[1]),
        std::exp(log_base[2] + log_t[2] - log_t0[2]),
    };
    egb_out[0] = G[0];
    egb_out[1] = G[1] / G[0];
    egb_out[2] = G[2] / G[0];
    return err;
  }

  // Evaluates the (L*, a*, b*) of the grey that base renders as neutral L* 50, when rendered with gains egb.
  void grey(const float log_t0[3], const float base[3], const float egb[3], float lab_out[3]) const {
    const float log_base[3] = {std::log(base[0]), std::log(base[0] * base[1]), std::log(base[0] * base[2])};
    const float log_egb[3] = {std::log(egb[0]), std::log(egb[0] * egb[1]), std::log(egb[0] * egb[2])};
    float t[1][3], l[1][3];
    for (int c = 0; c < 3; ++c) t[0][c] = std::exp(log_t0[c] + log_egb[c] - log_base[c]);
    lab(t, 1, l);
    for (int c = 0; c < 3; ++c) lab_out[c] = l[0][c];
  }

 private:
  // Finds the camera-linear input t that the bracket renders as the Lab colour target, starting from log_t, and
  // leaves it in log_t (as log(t), so every step is a ratio and t stays positive). Returns the distance from the
  // render of the final t to target, in Lab units (dE76).
  //
  // Each iteration renders t and, for the Jacobian d Lab / d log t by central differences, t with each channel
  // scaled by exp(+-FD_STEP): 7 renders in one lab() call. It stops when every Lab component is within TOLERANCE
  // of target. Otherwise it takes the least-squares Newton step J * step = target - Lab(t) (lstsq3 drops
  // directions the render barely responds to), clamps each channel's step to +-MAX_LOG_STEP (a factor of about
  // 1.28) so the knee and the profile's curves cannot throw it far, and repeats, at most iters times.
  float newton(const float target[3], float log_t[3], int iters) const {
    constexpr float FD_STEP = 0.05f, MAX_LOG_STEP = 0.25f, TOLERANCE = 0.02f;
    for (int it = 0; it < iters; ++it) {
      float pts[7][3];
      for (int c = 0; c < 3; ++c) pts[0][c] = std::exp(log_t[c]);
      for (int c = 0; c < 3; ++c) {
        for (int k = 0; k < 2; ++k) {
          const float sgn = k == 0 ? 1.0f : -1.0f;
          for (int j = 0; j < 3; ++j) pts[1 + 2 * c + k][j] = std::exp(log_t[j] + (j == c ? sgn * FD_STEP : 0.0f));
        }
      }
      float l[7][3];
      lab(pts, 7, l);
      const float r[3] = {target[0] - l[0][0], target[1] - l[0][1], target[2] - l[0][2]};
      if (std::max({std::fabs(r[0]), std::fabs(r[1]), std::fabs(r[2])}) < TOLERANCE) {
        return std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      }
      float J[3][3];
      for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row) J[row][c] = (l[1 + 2 * c][row] - l[2 + 2 * c][row]) / (2.0f * FD_STEP);
      float step[3];
      lstsq3(J, r, 1e-6f, step);
      for (int c = 0; c < 3; ++c) log_t[c] += std::clamp(step[c], -MAX_LOG_STEP, MAX_LOG_STEP);
    }
    float t[1][3], l[1][3];
    for (int c = 0; c < 3; ++c) t[0][c] = std::exp(log_t[c]);
    lab(t, 1, l);
    const float r[3] = {target[0] - l[0][0], target[1] - l[0][1], target[2] - l[0][2]};
    return std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  }

  const ProfileData& prof_;
};

// The E (see ConversionParams) at which the scene vector p, with gains g and b, renders at L* 50 through the
// bracket: the metered exposure. A bisection on log2 E over [-6, 6] that stops at |L* - 50| <= 0.1. Its 25
// halvings, the station's count, resolve E to 12 / 2^25 stop, far below that tolerance, so they only cap the loop.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (exposure_for_lightness)
bool exposure_for_lightness(const GreyPoint& gp, const float p[3], float g, float b, float& e_out) {
  constexpr float TARGET = 50.0f, TOL = 0.1f;
  constexpr int MAX_BISECTIONS = 25;
  const float base_t[3] = {p[0], p[1] * g, p[2] * b};
  float lo = -6.0f, hi = 6.0f;
  for (int i = 0; i < MAX_BISECTIONS; ++i) {
    const float mid = 0.5f * (lo + hi);
    const float E = std::pow(2.0f, mid);
    const float t[3] = {E * base_t[0], E * base_t[1], E * base_t[2]};
    const float L = gp.lstar(t);
    if (std::fabs(L - TARGET) <= TOL) {
      e_out = E;
      return true;
    }
    if (L > TARGET) lo = mid;
    else hi = mid;
  }
  const float E = std::pow(2.0f, 0.5f * (lo + hi));
  const float t[3] = {E * base_t[0], E * base_t[1], E * base_t[2]};
  e_out = E;
  return std::fabs(gp.lstar(t) - TARGET) <= TOL;
}

// ---- Frame statistics --------------------------------------------------------------------------------------

// Statistics of a frame's inner crop: the green 0.5 % / 99.5 % points, the log-average RGB and the crop averaged
// onto the DINOv3 patch grid (one pixel per patch). The grid is the inner crop's shape at DINOv3's input size: its
// longer side scaled to the engine's max_size, each side rounded to whole patches of patch_size pixels, so
// grid_w x grid_h patches (32 x 21 for a 3:2 landscape crop at 512 and 16).
// py-ref: negicc_station/src/image_capture.cpp @ 93be1ec (FrameView::raw_stats)
struct FrameStats {
  float g_lo = 0.0f, g_hi = 0.0f;
  float log_avg_rgb[3] = {0, 0, 0};  // geometric mean in [0, 1]
  int grid_w = 0, grid_h = 0;
  std::vector<uint16_t> grid;  // grid_w x grid_h pixels, row-major, 3 values (camera RGB) each
};

float u16_percentile_linear(const std::vector<uint64_t>& hist, uint64_t total, float q) {
  if (total == 0) return 0.0f;
  const float pos = q * (total - 1);
  const uint64_t idx_lo = std::floor(pos);
  const uint64_t idx_hi = std::ceil(pos);
  const float frac = pos - idx_lo;
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
  return (1.0f - frac) * v_lo + frac * v_hi;
}

void compute_frame_stats(const negicc::RgbView& frame, float crop_ratio, int max_size, int patch_size, FrameStats& st) {
  const InnerCrop crop = compute_inner_crop(frame.width, frame.height, crop_ratio);
  const negicc::AspectPreservedShape shape =
      negicc::compute_aspect_preserved_shape(crop.w, crop.h, max_size, patch_size);
  const int p = std::max(1, patch_size);
  const int gw = shape.grid_w, gh = shape.grid_h;
  const int dino_w = shape.target_w, dino_h = shape.target_h;  // DINOv3's input size: the grid in pixels

  std::vector<float> log_lut(65536);
  for (int v = 0; v < 65536; ++v) log_lut[v] = std::log(std::max(v, 1));

  std::vector<uint64_t> g_hist(65536, 0);
  std::vector<float> row_logs(crop.h * 3, 0.0f);  // per inner crop row, the sum of log R, log G and log B
  const bool contiguous_rgb = frame.is_contiguous_rgb();
#pragma omp parallel
  {
    std::vector<uint64_t> loc(65536, 0);
#pragma omp for schedule(static)
    for (int ry = 0; ry < crop.h; ++ry) {
      float lr = 0.0f, lg = 0.0f, lb = 0.0f;
      if (contiguous_rgb) {
        const uint16_t* p = frame.r_ptr + (ptrdiff_t)(crop.y0 + ry) * frame.stride +
                            (ptrdiff_t)crop.x0 * 3;
        for (int rx = 0; rx < crop.w; ++rx, p += 3) {
          loc[p[1]]++;
          lr += log_lut[p[0]];
          lg += log_lut[p[1]];
          lb += log_lut[p[2]];
        }
      } else {
        for (int rx = 0; rx < crop.w; ++rx) {
          uint16_t px[3];
          frame.sample(crop.x0 + rx, crop.y0 + ry, px);
          loc[px[1]]++;
          lr += log_lut[px[0]];
          lg += log_lut[px[1]];
          lb += log_lut[px[2]];
        }
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
    float tot[3] = {0, 0, 0};
    for (int ry = 0; ry < crop.h; ++ry)
      for (int c = 0; c < 3; ++c) tot[c] += row_logs[ry * 3 + c];
    for (int c = 0; c < 3; ++c) st.log_avg_rgb[c] = std::exp(tot[c] * (1.0f / npx)) / 65535.0f;
    const float p_lo = u16_percentile_linear(g_hist, npx, SCENE_TAIL);
    const float p_hi = u16_percentile_linear(g_hist, npx, 1.0f - SCENE_TAIL);
    st.g_lo = std::max(1.0f, p_lo);
    st.g_hi = std::max(st.g_lo + 1.0f, p_hi);
  }

  // Inner crop resampled to the DINOv3 input size (area average, bilinear when enlarging), then each
  // patch averaged into the grid. dino is dino_w x dino_h pixels, row-major, 3 floats (camera RGB) each.
  std::vector<float> dino(dino_w * dino_h * 3, 0.0f);
  const float sx = crop.w / float(dino_w);
  const float sy = crop.h / float(dino_h);
  if (sx >= 1.0f && sy >= 1.0f) {
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      const float y0 = dy * sy, y1 = (dy + 1) * sy;
      const int ys = std::floor(y0), ye = std::min<int>(crop.h, std::ceil(y1));
      for (int dx = 0; dx < dino_w; ++dx) {
        const float x0 = dx * sx, x1 = (dx + 1) * sx;
        const int xs = std::floor(x0), xe = std::min<int>(crop.w, std::ceil(x1));
        float acc[3] = {0, 0, 0}, wsum = 0.0f;
        for (int yy = ys; yy < ye; ++yy) {
          const float wy = std::min(y1, yy + 1.0f) - std::max(y0, yy + 0.0f);
          if (wy <= 0.0f) continue;
          for (int xx = xs; xx < xe; ++xx) {
            const float wx = std::min(x1, xx + 1.0f) - std::max(x0, xx + 0.0f);
            if (wx <= 0.0f) continue;
            const float w = wx * wy;
            uint16_t px[3];
            frame.sample(crop.x0 + xx, crop.y0 + yy, px);
            acc[0] += px[0] * w;
            acc[1] += px[1] * w;
            acc[2] += px[2] * w;
            wsum += w;
          }
        }
        const float inv = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;
        for (int c = 0; c < 3; ++c) dino[(dy * dino_w + dx) * 3 + c] = acc[c] * inv;
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      const float fy = (dy + 0.5f) * sy - 0.5f;
      const int y0 = std::clamp<int>(std::floor(fy), 0, crop.h - 1);
      const int y1 = std::clamp(y0 + 1, 0, crop.h - 1);
      const float wy1 = fy - y0, wy0 = 1.0f - wy1;
      for (int dx = 0; dx < dino_w; ++dx) {
        const float fx = (dx + 0.5f) * sx - 0.5f;
        const int x0 = std::clamp<int>(std::floor(fx), 0, crop.w - 1);
        const int x1 = std::clamp(x0 + 1, 0, crop.w - 1);
        const float wx1 = fx - x0, wx0 = 1.0f - wx1;
        uint16_t p00[3], p01[3], p10[3], p11[3];
        frame.sample(crop.x0 + x0, crop.y0 + y0, p00);
        frame.sample(crop.x0 + x1, crop.y0 + y0, p01);
        frame.sample(crop.x0 + x0, crop.y0 + y1, p10);
        frame.sample(crop.x0 + x1, crop.y0 + y1, p11);
        for (int c = 0; c < 3; ++c) {
          const float top = p00[c] * wx0 + p01[c] * wx1;
          const float bot = p10[c] * wx0 + p11[c] * wx1;
          dino[(dy * dino_w + dx) * 3 + c] = top * wy0 + bot * wy1;
        }
      }
    }
  }
  st.grid_w = gw;
  st.grid_h = gh;
  st.grid.assign(gw * gh * 3, 0);
  const float inv_patch = 1.0f / (p * p);
  for (int gy = 0; gy < gh; ++gy) {
    for (int gx = 0; gx < gw; ++gx) {
      float acc[3] = {0, 0, 0};
      for (int py = 0; py < p; ++py) {
        const float* drow = &dino[((gy * p + py) * dino_w + gx * p) * 3];
        for (int px = 0; px < p; ++px, drow += 3) {
          acc[0] += drow[0];
          acc[1] += drow[1];
          acc[2] += drow[2];
        }
      }
      for (int c = 0; c < 3; ++c) {
        st.grid[(gy * gw + gx) * 3 + c] = std::clamp(std::round(acc[c] * inv_patch), 0.0f, 65535.0f);
      }
    }
  }
}

// ---- Highlight ceiling and clipping ------------------------------------------------------------------------

// ICC PCS L*a*b* (D50) to 8-bit sRGB code values in [0, 255], through Bradford D50 -> D65.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (_pcs_lab_to_srgb)
void pcs_lab_to_srgb_u8(const float lab[3], float out[3]) {
  static const float kD50[3] = {0.9642f, 1.0f, 0.8249f};
  static const float kBradford[3][3] = {{0.9555766f, -0.0230393f, 0.0631636f},
                                        {-0.0282895f, 1.0099416f, 0.0210077f},
                                        {0.0122982f, -0.0204830f, 1.3299098f}};
  static const float kXyzToSrgb[3][3] = {{3.2406255f, -1.5372080f, -0.4986286f},
                                         {-0.9689307f, 1.8757561f, 0.0415175f},
                                         {0.0557101f, -0.2040211f, 1.0569959f}};
  const float fy = (lab[0] + 16.0f) / 116.0f;
  const float f[3] = {fy + lab[1] / 500.0f, fy, fy - lab[2] / 200.0f};
  float xyz[3], d65[3];
  for (int c = 0; c < 3; ++c) {
    const float f3 = f[c] * f[c] * f[c];
    xyz[c] = (f3 > 0.008856f ? f3 : (f[c] - 16.0f / 116.0f) / 7.787f) * kD50[c];
  }
  for (int r = 0; r < 3; ++r) d65[r] = kBradford[r][0] * xyz[0] + kBradford[r][1] * xyz[1] + kBradford[r][2] * xyz[2];
  for (int r = 0; r < 3; ++r) {
    const float lin = std::clamp(kXyzToSrgb[r][0] * d65[0] + kXyzToSrgb[r][1] * d65[1] + kXyzToSrgb[r][2] * d65[2],
                                 0.0f, 1.0f);
    const float enc = (lin <= 0.0031308f) ? lin * 12.92f : 1.055f * std::pow(lin, 1.0f / 2.4f) - 0.055f;
    out[r] = std::clamp(enc, 0.0f, 1.0f) * 255.0f;
  }
}

// Per-channel 8-bit sRGB ceiling of a bracket: the brightest value any of its calibration patches renders to
// through its profile, read from the profcheck report the bundle stores; 255 without a report. Highlights above
// it are outside what the bracket was calibrated on.
// TODO: read the ceiling from the bundle's metadata JSON (exported by the station) instead of parsing the report.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (profile_highlight_ceiling)
void profile_highlight_ceiling(const std::string& profcheck_output, float ceil[3]) {
  static const std::regex kRow(R"(\]\s+(\S+):\s+\S+\s+\S+\s+\S+\s+->\s+(\S+)\s+(\S+)\s+(\S+)\s+should be)");
  const float inf = std::numeric_limits<float>::infinity();
  for (int c = 0; c < 3; ++c) ceil[c] = -inf;
  bool any = false;
  for (auto it = std::sregex_iterator(profcheck_output.begin(), profcheck_output.end(), kRow);
       it != std::sregex_iterator(); ++it) {
    float lab[3], rgb[3];
    try {
      for (int c = 0; c < 3; ++c) lab[c] = std::stof((*it)[2 + c].str());
    } catch (const std::exception&) {
      continue;
    }
    pcs_lab_to_srgb_u8(lab, rgb);
    for (int c = 0; c < 3; ++c) ceil[c] = std::max(ceil[c], rgb[c]);
    any = true;
  }
  if (!any) {
    for (int c = 0; c < 3; ++c) ceil[c] = 255.0f;
  }
}

// Highlight clipping of one render, in percent of the pixels measured (float, as the station's kernel).
struct ClipStats {
  float exc[3];      // at or above the bracket's ceiling, per channel
  float white_clip;  // with any channel white-clipped
  float median_L;    // the median L*
};

// The highlight clipping of img's crop rendered with each conversion matrix (9 floats per candidate in matrices).
// py-ref: negicc_station/src/pipeline/converter.cpp @ 93be1ec (Converter::highlight_clip)
// py-ref: negicc_station/src/pipeline/kernels_cpu.cpp @ 93be1ec (cpu_highlight_clip)
std::vector<ClipStats> measure_highlight_clip(const negicc::RgbView& img, const InnerCrop& crop_in,
                                              const std::vector<float>& matrices, float knee,
                                              const ProfileData& prof, const float ceil_rgb[3],
                                              float shoulder = 1.0f, uint64_t* max_hist = nullptr) {
  const size_t n = matrices.size() / 9;
  std::vector<ClipStats> res(n, ClipStats{{0, 0, 0}, 0, 0});
  const InnerCrop crop = (crop_in.w > 0 && crop_in.h > 0) ? crop_in : InnerCrop{0, 0, img.width, img.height};
  const uint64_t npx = uint64_t(crop.w) * crop.h;
  if (npx == 0 || n == 0) return res;
  std::vector<float> hc_out(n * 5, 0.0f);
  negicc::HighlightClipArgs ha{};
  ha.in_view = &img;
  ha.width = img.width;
  ha.height = img.height;
  ha.crop = {crop.x0, crop.y0, crop.w, crop.h};
  ha.matrices = matrices.data();
  ha.n = (int)n;
  ha.cfg = make_pipeline_config(prof, knee, shoulder);
  ha.max_hist = max_hist;
  // A pixel is above the ceiling when its 8-bit sRGB code is above it (codes floor(ceil) + 1 and up), the threshold
  // negicc-station's Converter::highlight_clip uses: counting from ceil / 255 itself also took pixels that encode to
  // the ceiling's own code, so neg_process's excess read higher than the station's and its highlight pullback
  // decided differently on white light. Ceilings at or below 1 are fractions of full scale.
  for (int c = 0; c < 3; ++c) {
    const float c_u8 = (ceil_rgb[c] > 1.0001f) ? ceil_rgb[c] : (ceil_rgb[c] * 255.0f);
    const float thr_u8 = std::min(std::floor(c_u8) + 1.0f, 255.0f);
    ha.ceil_lin[c] = negicc::srgb_decode(std::max(0.0f, (thr_u8 - 0.5f) / 255.0f));
  }
  ha.out = hc_out.data();
  negicc::cpu_highlight_clip(ha);
  for (size_t c = 0; c < n; ++c) {
    res[c].exc[0] = hc_out[c * 5 + 0];
    res[c].exc[1] = hc_out[c * 5 + 1];
    res[c].exc[2] = hc_out[c * 5 + 2];
    res[c].white_clip = hc_out[c * 5 + 3];
    res[c].median_L = hc_out[c * 5 + 4];
  }
  return res;
}

// The largest per-channel share (0..1) of a render above the bracket's ceiling.
float max_excess(const ClipStats& c) { return std::max({c.exc[0], c.exc[1], c.exc[2]}) / 100.0f; }

// ---- The frame the search works on -------------------------------------------------------------------------

// The upright frame, its inner crop, statistics and input matrix, and the measurements every step of the search
// makes on them: previews, DINOv3, the residual, the coordinate descent and the highlight clipping. Their time is
// added up into the result.
class SearchFrame {
 public:
  SearchFrame(const ProfileBundle& bundle, DinoV3Engine& engine, bool verbose, int max_steps, ParamSearchResult& out)
      : bundle_(bundle), engine_(engine), verbose_(verbose), max_steps_(std::max(1, max_steps)), out_(out) {}

  // Takes the upright frame view, its inner crop and the statistics.
  void set_frame(const negicc::RgbView& frame, float crop_ratio) {
    frame_ = frame;
    crop_ = compute_inner_crop(frame.width, frame.height, crop_ratio);
    compute_frame_stats(frame, crop_ratio, engine_.max_size(), engine_.patch_size(), stats_);
  }

  // Sets the merged input matrix of this frame (ParamSearchResult: M_frame) and e_norm; on TriLED the search runs on
  // M_frame / e_norm for bit-exact station parity, while on white-light bundles (no crosstalk) it runs on M_frame
  // with g_lo and g_hi scaled by e_norm in bracket_enclosure.
  void set_frame_matrix(const float frame_matrix[9], float e_norm) {
    e_norm_ = e_norm;
    const float mat_div = bundle_.has_crosstalk ? e_norm : 1.0f;
    for (int i = 0; i < 9; ++i) {
      frame_matrix_[i] = frame_matrix[i];
      input_matrix_[i] = frame_matrix[i] / mat_div;
    }
  }

  // The shoulder: a fixed x_hi for every render, or (auto) the one each ConversionParams carries, which
  // fixed_point() measures. The screening passes carry 1.
  void set_shoulder(float x_hi, bool automatic) {
    shoulder_fixed_ = x_hi;
    shoulder_auto_ = automatic;
  }
  float shoulder_of(const ConversionParams& p) const { return shoulder_auto_ ? p.shoulder : shoulder_fixed_; }

  const InnerCrop& inner_crop() const { return crop_; }
  const FrameStats& stats() const { return stats_; }
  const float* frame_matrix() const { return frame_matrix_; }
  const float* input_matrix() const { return input_matrix_; }
  float e_norm() const { return e_norm_; }

  // The inner crop rendered to 8-bit sRGB through bracket target_idx at params: what DINOv3 sees. out is
  // crop_.w x crop_.h pixels, row-major, 3 bytes each; DINOv3 resamples it to its input size.
  void render_preview(int target_idx, const ConversionParams& params, std::vector<uint8_t>& out) {
    const Clock::time_point t0 = Clock::now();
    float m[9];  // input_matrix_ with its rows scaled by E, E g and E b
    negicc::candidate_matrix(input_matrix_, params.e, params.g, params.b, m);
    const ProfileData& prof = bundle_.targets[target_idx].film_prof.data;
    out.resize((size_t)crop_.w * crop_.h * 3);
    negicc::RgbView crop_view = frame_.crop_and_orient(crop_.x0, crop_.y0, crop_.w, crop_.h, 0, false, false);
    negicc::RenderArgs ra{};
    ra.in_view = &crop_view;
    ra.width = crop_.w;
    ra.height = crop_.h;
    ra.matrix = m;
    ra.cfg = make_pipeline_config(prof, AUTO_KNEE, shoulder_of(params));
    ra.out_type = 1;
    ra.out = out.data();
    negicc::cpu_render(ra);
    render_time_ += Clock::now() - t0;
  }

  // The intent field DINOv3 predicts from a preview, on the statistics' grid: mu (the target Lab) and sigma (its
  // tolerance) are grid_w x grid_h patches, row-major, 3 floats (L*, a*, b*) each.
  bool predict(const std::vector<uint8_t>& preview, std::vector<float>& mu, std::vector<float>& sigma) {
    const Clock::time_point t0 = Clock::now();
    DinoV3InferenceResult r;
    try {
      r = engine_.infer(preview.data(), crop_.w, crop_.h, crop_.w * 3);
    } catch (const std::exception& e) {
      std::cerr << "ERROR! DINOv3 inference failed: " << e.what() << std::endl;
      return false;
    }
    infer_time_ += Clock::now() - t0;
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
  float residual(const std::vector<uint8_t>& preview, const std::vector<float>& mu, const std::vector<float>& sigma) {
    const Clock::time_point t0 = Clock::now();
    const float err = self_consistency_residual(preview.data(), crop_.w, crop_.h,
                                                 stats_.grid_w, stats_.grid_h, mu.data(), sigma.data());
    residual_time_ += Clock::now() - t0;
    return err;
  }

  // The coordinate descent of (E, g, b) of bracket target_idx against an intent field, from initial (or the
  // search's defaults when null) with E within e_bounds = {lo, hi, centre}; returns the intent loss
  // (ScreeningResult::intent_loss).
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (optimize_target_for_intent)
  float search(int target_idx, const std::vector<float>& mu, const std::vector<float>& sigma,
                const ConversionParams* initial, const float e_bounds[3], ConversionParams& out,
                bool pullback_search = false) {
    const Clock::time_point t0 = Clock::now();
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    cfg.shoulder = initial ? shoulder_of(*initial) : shoulder_fixed_;
    if (pullback_search) {
      Enclosure enc;
      const float g_scale = bundle_.has_crosstalk ? 1.0f : e_norm_;
      const float g_lo_ref = std::max(1.0f, stats_.g_lo) * g_scale;
      const float g_hi_ref = std::max(std::max(1.0f, stats_.g_lo) + 1.0f, stats_.g_hi) * g_scale;
      if (bracket_enclosure(bundle_.targets[target_idx], target_idx, g_lo_ref, g_hi_ref, enc)) {
        cfg.enc_e_min = enc.e_min;
        cfg.enc_e_max = enc.e_max;
      }
    }
    IntentGains init, res;
    if (initial) init = {initial->e, initial->g, initial->b};
    const float loss = optimize_target_for_intent(cfg, input_matrix_, stats_.grid.data(), stats_.grid_w,
                                                  stats_.grid_h, mu.data(), sigma.data(), initial ? &init : nullptr,
                                                  e_bounds, 0, res);
    out = {res.e, res.g, res.b, cfg.shoulder};
    search_time_ += Clock::now() - t0;
    return loss;
  }

  // The fixed-point loop of bracket target_idx (FixedPointResult), from start (its screening result) with E
  // kept within the trust region e_bounds = {lo, hi, centre}; the centre is where the search's penalty pulls E.
  // py-ref: negicc_station/src/per_target_conversion.py @ 93be1ec (_FrameContext.fixed_point)
  bool fixed_point(int target_idx, const ConversionParams& start, float e_bounds[3], FixedPointResult& fp,
                   bool pullback_search = false) {
    if (e_bounds[0] > e_bounds[1]) {
      std::swap(e_bounds[0], e_bounds[1]);
      e_bounds[2] = (e_bounds[0] + e_bounds[1]) / 2.0f;
    }
    ConversionParams curr = start, best;
    curr.shoulder = shoulder_of(start);
    float prev_err = std::numeric_limits<float>::infinity();
    float cur_q995 = 0.0f, best_q995 = 0.0f;
    std::vector<float> best_mu, best_sigma;
    FixedPointStatus status = FixedPointStatus::kMaxIters;
    int steps = 0;
    bool have_best = false;
    float best_err = 0.0f;
    std::vector<uint8_t> preview;
    std::vector<float> mu, sigma;
    for (int it = 0; it < max_steps_; ++it) {
      // x_hi from the pre-shoulder render at the incoming parameters; held through this iteration's search.
      if (shoulder_auto_) curr.shoulder = measure_shoulder(target_idx, curr, &cur_q995);
      render_preview(target_idx, curr, preview);
      if (!predict(preview, mu, sigma)) return false;
      const float err = residual(preview, mu, sigma);
      if (!have_best || err < best_err) {
        have_best = true;
        best_err = err;
        best = curr;
        best_q995 = cur_q995;
        best_mu = mu;  // kept for the shoulder guard and for the sidecar's intent field
        best_sigma = sigma;
      }
      if (it > 0 && err > prev_err * 1.25f && (err - prev_err) > 1.5f) {
        status = FixedPointStatus::kDiverged;
        steps++;
        break;
      }
      ConversionParams next;
      search(target_idx, mu, sigma, &curr, e_bounds, next, pullback_search);
      const float dp = std::fabs(next.e - curr.e) + std::fabs(next.g - curr.g) + std::fabs(next.b - curr.b);
      const float rel = it > 0 ? std::fabs(err - prev_err) / std::max(1e-4f, prev_err) : 1.0f;
      steps++;
      if (verbose_) {
        std::cout << "    T" << target_idx + 1 << " step " << it << ": E " << fixed(curr.e, 4) << " g "
                  << fixed(curr.g, 4) << " b " << fixed(curr.b, 4) << " resid " << fixed(err, 4);
        if (shoulder_auto_ || curr.shoulder > negicc::kShoulderOff) {
          std::cout << " x_hi " << fixed(curr.shoulder, 3) << " (q995 " << fixed(cur_q995, 3)
                    << ", enclosure log2(E/e_max) " << fixed(enclosure_log2(target_idx, curr.e), 2, true) << ")";
        }
        std::cout << "\n";
      }
      // On non-crosstalk (white-light) bundles, stop once the residual has risen for two consecutive steps
      // past best_err and keep best as a converged tournament candidate (matching negicc-station optimizer.py).
      const bool rel_conv = bundle_.has_crosstalk
                                ? ((!pullback_search || err <= prev_err) && rel <= TOL_REL_ERR)
                                : (err <= prev_err && err <= best_err * (1.0f + TOL_REL_ERR) && rel <= TOL_REL_ERR);
      const bool past_min = !bundle_.has_crosstalk && it >= 2 && err >= prev_err && prev_err > best_err;
      if (it > 0 && (dp <= TOL_PARAM || rel_conv || past_min)) {
        if (verbose_ && past_min) std::cout << "    T" << target_idx + 1 << ": residual passed minimum, keeping best\n";
        status = FixedPointStatus::kConverged;
        break;
      }
      curr = next;
      prev_err = err;
    }
    fp.target_idx = target_idx;
    fp.params = best;
    fp.shoulder_q995 = best_q995;
    fp.shoulder_knee = 1.0f;
    fp.shoulder_guard = false;
    if (best.shoulder > negicc::kShoulderOff) shoulder_guard(target_idx, best, best_mu, best_sigma, fp);
    fp.residual = best_err;
    fp.status = status;
    fp.steps = steps;
    fp.mu = std::move(best_mu);
    fp.sigma = std::move(best_sigma);
    fp.grid_w = stats_.grid_w;
    fp.grid_h = stats_.grid_h;
    return true;
  }

  // Re-scores the best step with its x_hi and with the shoulder off against the field DINOv3 predicted from it. The
  // guard compares the L* loss alone: the shoulder removes lightness, so a 2% worse mean of L*, a*, b* would be about
  // 6% on L* (the a* and b* terms barely move). The penalties do not depend on the shoulder. The shoulder is kept either
  // way: the guard is only recorded when the loss is more than 2% worse, so the evaluation sees how often it disagrees.
  void shoulder_guard(int target_idx, const ConversionParams& p, const std::vector<float>& mu,
                      const std::vector<float>& sigma, FixedPointResult& fp) {
    negicc::PipelineConfig pc{};
    negicc::set_shoulder(pc, p.shoulder);
    fp.shoulder_knee = pc.shoulder_x_k;
    if (mu.empty()) return;
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    const IntentGains cand{p.e, p.g, p.b};
    auto mean_loss = [&](float shoulder) {
      cfg.shoulder = shoulder;
      IntentLoss l;
      intent_loss(cfg, input_matrix_, stats_.grid.data(), stats_.grid_w * stats_.grid_h, &cand, 1, mu.data(),
                  sigma.data(), &l);
      return (double)l.L;
    };
    const double with = mean_loss(p.shoulder), without = mean_loss(1.0f);
    fp.shoulder_guard = with > 1.02 * without;
    if (verbose_) {
      std::cout << "    T" << target_idx + 1 << " Shoulder: q995=" << fixed(fp.shoulder_q995, 3)
                << ", x_hi=" << fixed(p.shoulder, 3) << ", knee=" << fixed(fp.shoulder_knee, 3)
                << ", enclosure log2(E/e_max)=" << fixed(enclosure_log2(target_idx, p.e), 2, true) << ", L* loss "
                << fixed((float)with, 3) << " vs " << fixed((float)without, 3) << " without"
                << (fp.shoulder_guard ? " (guard)" : "") << "\n";
    }
  }

  // The highlight clipping of the frame's inner crop through bracket target_idx, per candidate (E, g, b).
  std::vector<ClipStats> highlight_clip(int target_idx, const std::vector<ConversionParams>& candidates,
                                        const float ceil[3]) {
    std::vector<float> matrices(candidates.size() * 9);
    for (size_t i = 0; i < candidates.size(); ++i) {
      const ConversionParams& p = candidates[i];
      negicc::candidate_matrix(input_matrix_, p.e, p.g, p.b, &matrices[i * 9]);
    }
    // The candidates share one shoulder (they differ in E, g, b around one render): the first one's. The clip
    // counts read the pre-shoulder value; only the median L* is post-shoulder.
    return measure_highlight_clip(frame_, crop_, matrices, AUTO_KNEE, bundle_.targets[target_idx].film_prof.data,
                                  ceil, candidates.empty() ? shoulder_fixed_ : shoulder_of(candidates[0]));
  }

  // log2(E / e_max) of bracket target_idx's enclosure at E: the same quantity as the shoulder's q995 for green (the
  // cross-check printed next to x_hi, never used for its value); NaN when the bracket has no enclosure.
  float enclosure_log2(int target_idx, float e) const {
    Enclosure enc;
    const float g_lo = std::max(1.0f, stats_.g_lo);
    const float g_hi = std::max(g_lo + 1.0f, stats_.g_hi);
    const float g_scale = bundle_.has_crosstalk ? 1.0f : e_norm_;
    if (!(e > 0.0f) ||
        !bracket_enclosure(bundle_.targets[target_idx], target_idx, g_lo * g_scale, g_hi * g_scale, enc) ||
        !(enc.e_max > 0.0f))
      return std::numeric_limits<float>::quiet_NaN();
    return std::log2(e / enc.e_max);
  }

  // The shoulder's x_hi at p: the 99.5% point of the max channel of the pre-shoulder render of the inner crop,
  // mapped to sRGB 242 (shoulder_x_hi_anchored); q995 gets the point.
  float measure_shoulder(int target_idx, const ConversionParams& p, float* q995) {
    const Clock::time_point t0 = Clock::now();  // the extra pre-shoulder pass per iteration is render time
    static const float kCeil[3] = {255.0f, 255.0f, 255.0f};
    float m[9];
    negicc::candidate_matrix(input_matrix_, p.e, p.g, p.b, m);
    std::vector<float> matrices(m, m + 9);
    std::vector<uint64_t> hist(negicc::kMaxHistBins, 0);
    measure_highlight_clip(frame_, crop_, matrices, AUTO_KNEE, bundle_.targets[target_idx].film_prof.data, kCeil,
                           1.0f, hist.data());
    const float x_hi = shoulder_x_hi_from_hist(hist.data(), q995);
    render_time_ += Clock::now() - t0;
    return x_hi;
  }

  // Writes the time spent per measurement into the result.
  void write_timings() const {
    out_.render_time = render_time_;
    out_.infer_time = infer_time_;
    out_.search_time = search_time_;
    out_.residual_time = residual_time_;
  }

 private:
  const ProfileBundle& bundle_;
  DinoV3Engine& engine_;
  const bool verbose_;
  const int max_steps_;
  ParamSearchResult& out_;

  negicc::RgbView frame_;  // the upright frame (search_conversion_params)
  // The centred inner crop of the frame (compute_inner_crop): what DINOv3 sees, read in place from frame_.
  InnerCrop crop_;
  FrameStats stats_;  // over the inner crop
  float frame_matrix_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // M_frame (ParamSearchResult)
  float input_matrix_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // M_frame / e_norm: the matrix the search runs on
  float e_norm_ = 1.0f;
  float shoulder_fixed_ = 1.0f;
  bool shoulder_auto_ = false;
  Clock::duration render_time_{}, infer_time_{}, search_time_{}, residual_time_{};  // added up per measurement
};

// The merged input matrix M_frame (ParamSearchResult): the crosstalk rows scaled by normalization_target / film
// base and by the exposure ratio between the film base capture and the scan.
// py-ref: negicc_station/src/film_profiling.py @ 93be1ec (compute_merged_crosstalk, compute_exposure_ratio)
void merged_matrix(const ProfileBundle& bundle, const float fb[3], float exposure_ratio, float out[9]) {
  for (int r = 0; r < 3; ++r) {
    const float scale = fb[r] > 0 ? (bundle.normalization_target / fb[r]) * exposure_ratio : 1.0f;
    for (int c = 0; c < 3; ++c) out[r * 3 + c] = bundle.crosstalk_matrix[r * 3 + c] * scale;
  }
}

// A screened bracket with its enclosure.
struct Screened {
  ScreeningResult result;
  Enclosure enclosure;
};

}  // namespace

// ---- Public API ----------------------------------------------------------------------------------------------

// The merged input matrix of a scan (M_frame, ParamSearchResult) and its exposure normalisation e_norm.
struct FrameScale {
  float fb[3];
  float ratio_frame;
  float frame_matrix[9];
  float e_norm;
};
static FrameScale frame_scale(const ProfileBundle& bundle, float scan_shutter_s, float scan_iso,
                              const ParamSearchOptions& opts) {
  FrameScale s;
  const float* fb = opts.film_base_rgb;
  if (!fb && opts.film_base_rgb_d) {
    for (int i = 0; i < 3; ++i) s.fb[i] = (float)opts.film_base_rgb_d[i];
    fb = s.fb;
  }
  if (!fb) fb = bundle.film_base_avg;
  if (fb != s.fb) std::copy(fb, fb + 3, s.fb);
  const float fb_t = bundle.film_base_shutter.empty() ? 0.125f : shutter_to_seconds(bundle.film_base_shutter);
  const float fb_exposure = fb_t * (bundle.film_base_iso / 100.0f);
  // A measured base comes from this capture: no exposure ratio.
  const bool has_custom_fb = (opts.film_base_rgb != nullptr || opts.film_base_rgb_d != nullptr) && opts.film_base_from_scan;
  s.ratio_frame = has_custom_fb ? 1.0f : fb_exposure / (scan_shutter_s * (scan_iso / 100.0f));
  const float ratio_ref = fb_exposure / (REFERENCE_SHUTTER_S * (REFERENCE_ISO / 100.0f));
  float ref_matrix[9];
  merged_matrix(bundle, s.fb, s.ratio_frame, s.frame_matrix);
  merged_matrix(bundle, bundle.film_base_avg, ratio_ref, ref_matrix);
  s.e_norm = (ref_matrix[4] != 0.0f) ? s.frame_matrix[4] / ref_matrix[4] : 1.0f;
  return s;
}

float measure_rolloff_x_hi(const ProfileBundle& bundle, int target_idx, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995, const double* film_base_rgb) {
  return measure_rolloff_x_hi(bundle, bundle.targets[target_idx].film_prof.data, frame, scan_shutter_s, scan_iso, e, g,
                              b, crop_ratio, q995, film_base_rgb);
}

float measure_rolloff_x_hi(const ProfileBundle& bundle, const negicc::ProfileData& prof_in,
                           const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995, const double* film_base_rgb) {
  ProfileData prof;
  static_cast<negicc::ProfileData&>(prof) = prof_in;
  ParamSearchOptions base_opts;
  base_opts.film_base_rgb_d = film_base_rgb;
  base_opts.film_base_from_scan = film_base_rgb != nullptr;
  const FrameScale scale = frame_scale(bundle, (float)scan_shutter_s, (float)scan_iso, base_opts);
  // As SearchFrame::measure_shoulder: the matrix in reference units (M_frame / e_norm) at the search's internal E.
  float input_matrix[9], m[9];
  for (int i = 0; i < 9; ++i) input_matrix[i] = scale.frame_matrix[i] / scale.e_norm;
  negicc::candidate_matrix(input_matrix, e * scale.e_norm, g, b, m);
  static const float kCeil[3] = {255.0f, 255.0f, 255.0f};
  std::vector<uint64_t> hist(negicc::kMaxHistBins, 0);
  measure_highlight_clip(frame, compute_inner_crop(frame.width, frame.height, crop_ratio), std::vector<float>(m, m + 9),
                         AUTO_KNEE, prof, kCeil, 1.0f, hist.data());
  return shoulder_x_hi_from_hist(hist.data(), q995);
}

// L* of a grey whose luminance is multiplied by 2 ** stops (CIE L* <-> relative luminance Y).
// py-ref: negicc_station/src/grey_point.py (lstar_after_stops)
static float lstar_after_stops(float L, float stops) {
  double Y = L > 8.0f ? std::pow(((double)L + 16.0) / 116.0, 3.0) : (double)L / 903.3;
  Y *= std::pow(2.0, (double)stops);
  return (float)(Y > 0.008856 ? 116.0 * std::cbrt(Y) - 16.0 : 903.3 * Y);
}

// py-ref: negicc_station/src/conversion_session.py (ConversionSession._solve_knobs)
bool solve_grey_knobs(const ProfileData& prof, float base_e, float base_g, float base_b, float brightness_stops,
                      float tint_a, float warmth_b, KnobSolveResult& out) {
  out.e = base_e;
  out.g = base_g;
  out.b = base_b;
  out.limited = false;
  out.grey_lab[0] = 50.0f;
  out.grey_lab[1] = 0.0f;
  out.grey_lab[2] = 0.0f;
  if (!(base_e > 0.0f) || !(base_g > 0.0f) || !(base_b > 0.0f)) return false;

  const GreyPoint gp(prof);
  float log_t0[3];
  if (!gp.reference_grey(log_t0)) return false;
  const float base[3] = {base_e, base_g, base_b};

  if (brightness_stops == 0.0f && tint_a == 0.0f && warmth_b == 0.0f) {
    gp.grey(log_t0, base, base, out.grey_lab);
    return true;
  }

  constexpr float REACHABLE = 0.25f;
  const float anchor[3] = {50.0f, 0.0f, 0.0f};
  const float full[3] = {lstar_after_stops(50.0f, brightness_stops), tint_a, warmth_b};
  float egb[3];
  if (gp.gains(log_t0, base, full, egb) <= REACHABLE) {
    out.e = egb[0];
    out.g = egb[1];
    out.b = egb[2];
    out.limited = false;
    gp.grey(log_t0, base, egb, out.grey_lab);
    return true;
  }

  auto largest = [&](auto make_target, float best_out[3]) -> bool {
    float lo = 0.0f, hi = 1.0f;
    bool found = false;
    for (int i = 0; i < 7; ++i) {
      const float f = 0.5f * (lo + hi);
      float tgt[3], r[3];
      make_target(f, tgt);
      if (gp.gains(log_t0, base, tgt, r) <= REACHABLE) {
        lo = f;
        best_out[0] = r[0];
        best_out[1] = r[1];
        best_out[2] = r[2];
        found = true;
      } else {
        hi = f;
      }
    }
    return found;
  };

  const float colour_only[3] = {50.0f, tint_a, warmth_b};
  float colour_egb[3];
  float best[3] = {base_e, base_g, base_b};
  if (gp.gains(log_t0, base, colour_only, colour_egb) <= REACHABLE) {
    if (!largest(
            [&](float f, float tgt[3]) {
              tgt[0] = 50.0f + f * (full[0] - 50.0f);
              tgt[1] = tint_a;
              tgt[2] = warmth_b;
            },
            best)) {
      best[0] = colour_egb[0];
      best[1] = colour_egb[1];
      best[2] = colour_egb[2];
    }
  } else {
    largest(
        [&](float f, float tgt[3]) {
          for (int c = 0; c < 3; ++c) tgt[c] = anchor[c] + f * (full[c] - anchor[c]);
        },
        best);
  }
  out.e = best[0];
  out.g = best[1];
  out.b = best[2];
  out.limited = true;
  gp.grey(log_t0, base, best, out.grey_lab);
  return true;
}

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& frame,
                              float scan_shutter_s, float scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out) {
  return search_conversion_params(
      bundle, engine, negicc::RgbView::from_interleaved_rgb(frame.rgb.data(), frame.width, frame.height),
      (double)scan_shutter_s, (double)scan_iso, opts, out);
}

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                              double scan_shutter_s_in, double scan_iso_in, const ParamSearchOptions& opts,
                              ParamSearchResult& out) {
  const float scan_shutter_s = (float)scan_shutter_s_in;
  const float scan_iso = (float)scan_iso_in;
  const Clock::time_point t_start = Clock::now();
  const bool verbose = opts.verbose;
  if (bundle.targets.empty()) {
    std::cerr << "ERROR! The profile bundle has no targets" << std::endl;
    return false;
  }
  SearchFrame fr(bundle, engine, verbose, opts.max_steps, out);
  auto name_of = [&](int target_idx) { return bundle.targets[target_idx].name; };

  fr.set_shoulder(1.0f, true);  // the roll-off, measured by the fixed-point loop
  fr.set_frame(frame, opts.crop_ratio);
  const FrameStats& stats = fr.stats();

  const bool white_light = !bundle.has_crosstalk;
  // The input matrix in reference units (ParamSearchResult).
  const FrameScale scale = frame_scale(bundle, scan_shutter_s, scan_iso, opts);
  fr.set_frame_matrix(scale.frame_matrix, scale.e_norm);
  const float e_norm = fr.e_norm();
  out.e_norm = e_norm;
  for (int i = 0; i < 9; ++i) out.input_matrix[i] = scale.frame_matrix[i] / e_norm;
  for (int r = 0; r < 3; ++r) {
    out.row_scale[r] = (scale.fb[r] > 0 ? (bundle.normalization_target / scale.fb[r]) * scale.ratio_frame : 1.0f) /
                       e_norm;
  }

  // Step 1: the enclosure of each bracket (on white-light bundles, g_lo and g_hi are scaled by e_norm into reference
  // station ADU units; on TriLED, raw g_lo and g_hi are kept for bit-exact station parity).
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (compute_gamut_volume_enclosure)
  const float g_lo = std::max(1.0f, stats.g_lo);
  const float g_hi = std::max(g_lo + 1.0f, stats.g_hi);
  const float g_scale = white_light ? e_norm : 1.0f;
  const float g_lo_ref = g_lo * g_scale;
  const float g_hi_ref = g_hi * g_scale;
  out.g_lo = g_lo;
  out.g_hi = g_hi;
  out.scene_ev = std::log2(g_hi / g_lo);
  std::vector<Enclosure> enclosure;
  for (int i : filter_admissible_targets(bundle)) {
    Enclosure r;
    if (bracket_enclosure(bundle.targets[i], i, g_lo_ref, g_hi_ref, r)) {
      if (white_light && r.search_lo > r.search_hi) std::swap(r.search_lo, r.search_hi);
      enclosure.push_back(r);
    }
  }
  if (enclosure.empty()) {
    std::cerr << "ERROR! No bracket of the bundle has patch readings" << std::endl;
    return false;
  }

  if (verbose) {
    std::ostringstream os;
    os << "Auto: frame " << frame.width << "x" << frame.height << " (half size, upright), crop "
       << fr.inner_crop().w << "x" << fr.inner_crop().h << ", grid " << stats.grid_w << "x" << stats.grid_h
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
    return std::fabs(r.fill_ratio - 0.85f) + SLACK_PENALTY_SCALE * std::max(0.0f, -r.slack_ev);
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
  float p_scene[3];
  for (int r = 0; r < 3; ++r) {
    p_scene[r] = 0.0f;
    for (int c = 0; c < 3; ++c) p_scene[r] += fr.input_matrix()[r * 3 + c] * stats.log_avg_rgb[c];
  }
  struct Depth {
    Enclosure cand;
    float e_met, depth, lo, hi;
  };
  std::vector<Depth> depths;
  for (const auto& c : pool) {
    const GreyPoint gp(bundle.targets[c.target_idx].film_prof.data);
    float e_met = 0.0f;
    if (!gp.reference_grey()) {
      if (verbose) std::cout << "  " << name_of(c.target_idx) << ": no neutral L* 50 in this bracket, not metered\n";
      continue;
    }
    if (!exposure_for_lightness(gp, p_scene, 1.10f, 1.15f, e_met)) {
      if (verbose) std::cout << "  " << name_of(c.target_idx) << ": metering did not converge\n";
      continue;
    }
    const float lo = c.search_lo, hi = c.search_hi;
    const float depth = std::min(std::log2(e_met / std::max(1e-6f, lo)), std::log2(std::max(1e-6f, hi) / e_met)) -
                         SLACK_PENALTY_SCALE * std::max(0.0f, -c.slack_ev);
    depths.push_back({c, e_met, depth, lo, hi});
  }
  Enclosure cand1;
  float e1_init;
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
  fr.render_preview(cand1.target_idx, {e1_init, 1.10f, 1.15f}, preview);
  if (!fr.predict(preview, mu1, sig1)) return false;

  // Step 3: screen the brackets against the bootstrap field.
  std::vector<Screened> screening;
  auto screen = [&](const Enclosure& c, float lo, float hi, float e_met) {
    Screened s;
    s.enclosure = c;
    s.result.target_idx = c.target_idx;
    s.result.e_lo = lo;
    s.result.e_hi = hi;
    s.result.e_metered = e_met;
    const float eb[3] = {lo, hi, c.e_center};
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
    if (s.result.intent_loss <= screening[0].result.intent_loss + 2.0f && shortlist.size() < 2) shortlist.push_back(s);
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
  auto trust_bounds = [](const Screened& c, float eb[3]) {
    eb[0] = std::max(c.result.e_lo, c.result.params.e * std::pow(2.0f, -0.5f));
    eb[1] = std::min(c.result.e_hi, c.result.params.e * std::pow(2.0f, 0.5f));
    eb[2] = c.enclosure.e_center;
  };
  std::vector<FixedPointResult> adaptive;
  for (const auto& c : shortlist) {
    float eb[3];
    trust_bounds(c, eb);
    FixedPointResult fp;
    if (verbose) std::cout << "Auto: fixed point on " << name_of(c.result.target_idx) << "\n";
    if (!fr.fixed_point(c.result.target_idx, c.result.params, eb, fp)) return false;
    adaptive.push_back(fp);
    out.tournament.push_back(fp);
  }
  // The winner is the lowest plain residual among the converged brackets (when any converged), below. tourney_score,
  // the enclosure-regularized residual, only ranks the candidates of the highlight pullback.
  auto tourney_score = [&](const FixedPointResult& a) {
    for (const auto& c : enclosure) {
      if (c.target_idx == a.target_idx && a.params.e > 0.0f) {
        const float d_hi = std::max(0.0f, std::log2(c.e_min / a.params.e));
        const float d_sh = std::max(0.0f, std::log2(a.params.e / c.e_max));
        const float neg_slack = std::max(0.0f, -c.slack_ev);
        return a.residual + 4.5f * d_hi * d_hi + 1.2f * d_sh * d_sh + 0.8f * neg_slack * neg_slack;
      }
    }
    return a.residual;
  };
  bool any_converged = false;
  for (const auto& a : adaptive) any_converged |= (a.status == FixedPointStatus::kConverged);
  const FixedPointResult* win = nullptr;
  for (const auto& a : adaptive) {
    if (any_converged && a.status != FixedPointStatus::kConverged) continue;
    if (!win || a.residual < win->residual) win = &a;
  }
  FixedPointResult chosen = *win;
  ConversionParams params = chosen.params;

  // Each bracket's enclosure (the screened one, else computed).
  auto bounds_of = [&](int target_idx, Enclosure& r) {
    for (const auto& c : enclosure)
      if (c.target_idx == target_idx) {
        r = c;
        return true;
      }
    return bracket_enclosure(bundle.targets[target_idx], target_idx, g_lo_ref, g_hi_ref, r);
  };
  // Each bracket's ceiling, parsed the first time it is needed.
  std::vector<float> ceilings(bundle.targets.size() * 3);
  std::vector<bool> has_ceiling(bundle.targets.size(), false);
  auto ceiling_of = [&](int target_idx) -> const float* {
    float* ceil = &ceilings[target_idx * 3];
    if (!has_ceiling[target_idx]) {
      profile_highlight_ceiling(bundle.targets[target_idx].profcheck_output, ceil);
      has_ceiling[target_idx] = true;
    }
    return ceil;
  };
  // (The highlight headroom that raised E here was removed with negicc-station deaf9cc, 2026-10-29: it changed none
  // of the 55 TriLED development frames, and on white light it made neg_process's Auto differ from the station's.)
  if (verbose) {
    std::cout << "Auto: winner " << name_of(chosen.target_idx) << " (resid " << fixed(chosen.residual, 4) << ", "
              << to_string(chosen.status) << ")\n";
  }

  // Step 5: the clip fallback (ClipFallbackOutcome): move to the next bracket when the chosen one white-clips.
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (measure_rescue_candidates, select_highlight_rescue)
  const ClipStats hc0 = fr.highlight_clip(chosen.target_idx, {params}, ceiling_of(chosen.target_idx))[0];
  const float clip_0 = hc0.white_clip / 100.0f;
  out.clip_fallback_white_clip = clip_0;
  out.clip_fallback = ClipFallbackOutcome::kNotNeeded;
  if (clip_0 > CLIP_FALLBACK_CLIP_TAU) {
    const int t_adj = chosen.target_idx + 1;
    const Screened* s_adj = nullptr;
    for (const auto& s : screening)
      if (s.result.target_idx == t_adj) s_adj = &s;
    if (!s_adj) {
      out.clip_fallback = ClipFallbackOutcome::kNotScreened;
    } else if (!s_adj->enclosure.admissible) {
      out.clip_fallback = ClipFallbackOutcome::kNotAdmissible;
    } else {
      const FixedPointResult* run = nullptr;
      for (const auto& a : adaptive)
        if (a.target_idx == t_adj) run = &a;
      FixedPointResult fp_adj;
      if (run) {
        fp_adj = *run;
      } else {
        float eb[3];
        trust_bounds(*s_adj, eb);
        if (verbose) std::cout << "Auto: fixed point on " << name_of(t_adj) << " (clip fallback)\n";
        if (!fr.fixed_point(t_adj, s_adj->result.params, eb, fp_adj)) return false;
      }
      ConversionParams p_adj = fp_adj.params;
      const ClipStats hc_adj = fr.highlight_clip(t_adj, {p_adj}, ceiling_of(t_adj))[0];
      const float clip_adj = hc_adj.white_clip / 100.0f;
      const float dl = hc_adj.median_L - hc0.median_L;
      if (clip_adj > CLIP_FALLBACK_CLIP_RATIO * clip_0) out.clip_fallback = ClipFallbackOutcome::kClipsTooMuch;
      else if (!(CLIP_FALLBACK_DL_MIN <= dl && dl <= CLIP_FALLBACK_DL_MAX)) out.clip_fallback = ClipFallbackOutcome::kShiftsLightness;
      else if (fp_adj.residual > (1.0f + CLIP_FALLBACK_RESID_TOL) * chosen.residual)
        out.clip_fallback = ClipFallbackOutcome::kWorseResidual;
      else out.clip_fallback = ClipFallbackOutcome::kTaken;
      if (verbose) {
        std::cout << "Auto: clip fallback: " << name_of(t_adj) << " clips " << fixed(clip_adj * 100.0f, 4)
                  << "% vs " << fixed(clip_0 * 100.0f, 4) << "%, dL* " << fixed(dl, 2, true) << ", resid "
                  << fixed(fp_adj.residual, 4) << ": " << to_string(out.clip_fallback) << "\n";
      }
      if (out.clip_fallback == ClipFallbackOutcome::kTaken) {
        chosen = fp_adj;
        params = p_adj;
      }
    }
    if (verbose && (out.clip_fallback == ClipFallbackOutcome::kNotScreened || out.clip_fallback == ClipFallbackOutcome::kNotAdmissible)) {
      std::cout << "Auto: clip fallback: " << to_string(out.clip_fallback) << "\n";
    }
  }

  // Step 6: the highlight pullback (formerly Part 1B, the enclosure-guided rescue), when the chosen bracket is
  // highlight-constrained (much of the pre-shoulder render above sRGB 242 or 235) without being a dark or backlit
  // silhouette: pull E up into the bracket's calibrated range, or re-search the brackets for one whose range holds
  // the frame.
  if (out.clip_fallback != ClipFallbackOutcome::kTaken && bundle.targets.size() > 1) {
    const float ceil_242[3] = {242.0f, 242.0f, 242.0f};
    const float ceil_235[3] = {235.0f, 235.0f, 235.0f};
    const ClipStats hc_242 = fr.highlight_clip(chosen.target_idx, {params}, ceil_242)[0];
    const ClipStats hc_235 = fr.highlight_clip(chosen.target_idx, {params}, ceil_235)[0];
    const float exc_242 = max_excess(hc_242);
    const float exc_235 = max_excess(hc_235);
    Enclosure enc_chosen{};
    const bool has_enc_chosen = bounds_of(chosen.target_idx, enc_chosen);
    const bool tier1_hi = (hc_242.median_L >= 58.0f) && (exc_242 >= 0.022f);
    const bool tier2_hi =
        (hc_242.median_L >= 48.0f) && (exc_242 >= 0.075f || (chosen.target_idx >= 3 && exc_235 >= 0.110f));
    if (has_enc_chosen && enc_chosen.feasible && params.e < enc_chosen.e_min && (tier1_hi || tier2_hi)) {
      const float e_pull = std::min(enc_chosen.e_max, std::min(enc_chosen.e_min, params.e * std::pow(2.0f, 0.15f)));
      if (e_pull > params.e) {
        params.e = e_pull;
        out.highlight_pullback = true;
      }
    } else if ((!has_enc_chosen || !enc_chosen.feasible) && (tier1_hi || tier2_hi)) {
      std::vector<Depth> depths_res;
      for (const auto& c : pool) {
        float e_met = 0.0f;
        bool found_met = false;
        for (const auto& d : depths) {
          if (d.cand.target_idx == c.target_idx) {
            e_met = d.e_met;
            found_met = true;
            break;
          }
        }
        if (!found_met) continue;
        const float u_min = c.e_min * std::pow(2.0f, -1.5f);
        const float u_max = c.e_max * std::pow(2.0f, 1.2f);
        const float margin_ev = std::max(0.60f, std::min(0.85f, 0.60f * std::fabs(c.slack_ev)));
        const float win_lo = std::max(u_min, c.e_center * std::pow(2.0f, -margin_ev));
        const float win_hi = std::min(u_max, c.e_center * std::pow(2.0f, margin_ev));
        const float base_lo = c.feasible ? std::min(c.e_min, win_lo) : win_lo;
        const float base_hi = c.feasible ? std::max(c.e_max, win_hi) : win_hi;
        float lo = base_lo;
        float hi = std::max(base_hi, std::min(u_max, e_met * std::pow(2.0f, 0.20f)));
        // On an infeasible bracket the window can cross (win_lo > win_hi when the centre's margin falls outside
        // [u_min, u_max]); std::clamp below needs lo <= hi, and the depth would be measured on a reversed window.
        if (lo > hi) std::swap(lo, hi);
        const float depth_r = std::min(std::log2(e_met / std::max(1e-6f, lo)), std::log2(std::max(1e-6f, hi) / e_met)) -
                              SLACK_PENALTY_SCALE * std::max(0.0f, -c.slack_ev);
        depths_res.push_back({c, e_met, depth_r, lo, hi});
      }
      std::vector<float> mu_res = mu1, sig_res = sig1;
      if (!depths_res.empty()) {
        std::stable_sort(depths_res.begin(), depths_res.end(),
                         [](const Depth& a, const Depth& b) { return a.depth > b.depth; });
        const Enclosure cand_res = depths_res[0].cand;
        const float e_res_init = std::clamp(depths_res[0].e_met, depths_res[0].lo, depths_res[0].hi);
        if (cand_res.target_idx != cand1.target_idx || std::fabs(e_res_init - e1_init) > 1e-4f) {
          std::vector<uint8_t> preview_res;
          fr.render_preview(cand_res.target_idx, {e_res_init, 1.10f, 1.15f}, preview_res);
          if (!fr.predict(preview_res, mu_res, sig_res)) return false;
        }
      }
      std::vector<Screened> res_screening;
      auto screen_res = [&](const Enclosure& c, float lo, float hi, float e_met) {
        Screened s;
        s.enclosure = c;
        s.result.target_idx = c.target_idx;
        s.result.e_lo = lo;
        s.result.e_hi = hi;
        s.result.e_metered = e_met;
        const float eb[3] = {lo, hi, c.e_center};
        s.result.intent_loss =
            fr.search(c.target_idx, mu_res, sig_res, nullptr, eb, s.result.params, /*pullback_search=*/true);
        res_screening.push_back(s);
      };
      if (!depths_res.empty()) {
        for (const auto& d : depths_res) screen_res(d.cand, d.lo, d.hi, d.e_met);
      } else {
        for (const auto& c : pool) screen_res(c, c.search_lo, c.search_hi, c.e_center);
      }
      if (!res_screening.empty()) {
        std::stable_sort(res_screening.begin(), res_screening.end(), [](const Screened& a, const Screened& b) {
          return a.result.intent_loss < b.result.intent_loss;
        });
        std::vector<Screened> res_shortlist;
        for (const auto& s : res_screening) {
          if (s.result.intent_loss <= res_screening[0].result.intent_loss + 2.0f && res_shortlist.size() < 2) {
            res_shortlist.push_back(s);
          }
        }
        if (res_shortlist.size() < 2 && res_screening.size() >= 2) {
          res_shortlist.assign(res_screening.begin(), res_screening.begin() + 2);
        }
        std::vector<FixedPointResult> res_adaptive;
        for (const auto& c : res_shortlist) {
          float eb[3];
          trust_bounds(c, eb);
          FixedPointResult fp;
          if (!fr.fixed_point(c.result.target_idx, c.result.params, eb, fp, /*pullback_search=*/true)) return false;
          res_adaptive.push_back(fp);
        }
        bool res_any_conv = false;
        for (const auto& a : res_adaptive) res_any_conv |= (a.status == FixedPointStatus::kConverged);
        const FixedPointResult* win_res = nullptr;
        for (const auto& a : res_adaptive) {
          if (res_any_conv && a.status != FixedPointStatus::kConverged) continue;
          if (!win_res || tourney_score(a) < tourney_score(*win_res)) win_res = &a;
        }
        if (win_res) {
          FixedPointResult fp_res = *win_res;
          ConversionParams p_res = fp_res.params;
          ClipStats hc_res_242 = fr.highlight_clip(fp_res.target_idx, {p_res}, ceil_242)[0];
          const float dl_min = (hc_242.median_L >= 58.0f) ? -7.5f : -4.5f;
          float dl = hc_res_242.median_L - hc_242.median_L;
          if (dl < dl_min) {
            p_res.e *= std::pow(2.0f, (dl - dl_min) * 0.038f);
            hc_res_242 = fr.highlight_clip(fp_res.target_idx, {p_res}, ceil_242)[0];
            dl = hc_res_242.median_L - hc_242.median_L;
          }
          const float min_allowed_L = std::min(48.0f, hc_242.median_L - 6.0f);
          const float exc_res_242 = max_excess(hc_res_242);
          if (exc_res_242 <= exc_242 + 0.005f && hc_res_242.median_L >= min_allowed_L &&
              (fp_res.target_idx > chosen.target_idx || p_res.e > params.e * 1.02f)) {
            chosen = fp_res;
            params = p_res;
            out.highlight_pullback = true;
          }
        }
      }
    }
  }

  // The clip fallback and the pullback move E after the loop froze x_hi; measure it once more at the final parameters
  // (the same pre-shoulder pass on the inner crop) so the written image's shoulder matches its highlights. The
  // guard stays that of the converged iteration.
  {
    float q995 = 0.0f;
    params.shoulder = fr.measure_shoulder(chosen.target_idx, params, &q995);
    chosen.shoulder_q995 = q995;
    negicc::PipelineConfig pc{};
    negicc::set_shoulder(pc, params.shoulder);
    chosen.shoulder_knee = pc.shoulder_x_k;
    if (verbose) {
      std::cout << "Auto: Shoulder: q995=" << fixed(q995, 3) << ", x_hi=" << fixed(params.shoulder, 3)
                << ", knee=" << fixed(chosen.shoulder_knee, 3) << ", enclosure log2(E/e_max)="
                << fixed(fr.enclosure_log2(chosen.target_idx, params.e), 2, true) << " (final parameters)\n";
    }
  }

  out.target_idx = chosen.target_idx;
  out.target_name = name_of(chosen.target_idx);
  out.params = {white_light ? params.e : params.e / e_norm, params.g, params.b, params.shoulder};
  out.residual = chosen.residual;
  out.shoulder_q995 = chosen.shoulder_q995;
  out.shoulder_knee = chosen.shoulder_knee;
  out.shoulder_guard = chosen.shoulder_guard;
  out.status = chosen.status;
  out.intent_mu = chosen.mu;
  out.intent_sigma = chosen.sigma;
  out.intent_grid_w = chosen.grid_w;
  out.intent_grid_h = chosen.grid_h;
  out.intent_params = chosen.params;
  fr.write_timings();
  out.total_time = Clock::now() - t_start;
  return true;
}

// ---- Fitted G/B input curves ---------------------------------------------------------------------------------

void apply_input_curves(ParsedFilmProfile& prof, const negicc::InputCurves& curves) {
  if (!curves.active() || !prof.data.has_profile) return;
  for (int ch = 1; ch <= 2; ++ch) {
    std::vector<float>& table = prof.in_trc[ch];
    if (table.size() < 2) continue;
    std::vector<float> composed(table.size());
    const int i = ch - 1;
    negicc::compose_input_table(table.data(), (int)table.size(), curves.ref[i], curves.u_med[i], curves.a[i],
                                curves.k[i], composed.data());
    table.swap(composed);
  }
  prof.rebind_pointers();
}

// py-ref: negicc_station/src/conversion_session.py (ConversionSession.set_fit_curves) and
// negicc_station/src/per_target_conversion.py (PerTargetConverter.fit_bracket)
bool fit_input_curves_after_search(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                                   double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                                   const ParamSearchResult& res, const negicc::CurveFitOptions& fit_opts,
                                   int encoding, CurveRefitResult& out) {
  if (bundle.has_crosstalk) {
    std::cerr << "ERROR! Fitted input curves are for white-light bundles: with a crosstalk matrix the film base does "
                 "not mark the dark end of every channel" << std::endl;
    return false;
  }
  const int idx = res.target_idx;
  if (idx < 0 || idx >= (int)bundle.targets.size()) {
    std::cerr << "ERROR! Fitted input curves need the bracket of a finished search" << std::endl;
    return false;
  }
  const Clock::time_point t_fit0 = Clock::now();
  ParamSearchResult scratch;  // the inferences and timings of the frame below; not reported
  SearchFrame fr(bundle, engine, opts.verbose, opts.max_steps, scratch);
  fr.set_shoulder(1.0f, true);
  fr.set_frame(frame, opts.crop_ratio);
  const FrameStats& stats = fr.stats();
  const FrameScale scale = frame_scale(bundle, (float)scan_shutter_s, (float)scan_iso, opts);
  fr.set_frame_matrix(scale.frame_matrix, scale.e_norm);
  const size_t n = stats.grid.size() / 3;
  if (res.intent_grid_w != stats.grid_w || res.intent_grid_h != stats.grid_h || res.intent_mu.size() != n * 3 ||
      res.intent_sigma.size() != n * 3 || n == 0) {
    std::cerr << "ERROR! The search's intent field does not match the frame's grid" << std::endl;
    return false;
  }
  // The search's matrix and E: white-light bundles search on M_frame with E in this frame's units.
  const float* merged = fr.input_matrix();
  const float e_int = res.params.e;  // white light (checked above): the search's E is already in this frame's units
  out.params_before = res.params;

  // 1. The fit with E fixed (the station's Converter.fit_input_curves).
  float fb[3];
  if (opts.film_base_rgb_d) {
    for (int c = 0; c < 3; ++c) fb[c] = (float)opts.film_base_rgb_d[c];
  } else if (opts.film_base_rgb) {
    for (int c = 0; c < 3; ++c) fb[c] = opts.film_base_rgb[c];
  } else {
    // No film base read from the scan: the brightest grid cell per channel, as the station takes it.
    for (int c = 0; c < 3; ++c) fb[c] = 0.0f;
    for (size_t i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) fb[c] = std::max(fb[c], (float)stats.grid[i * 3 + c]);
  }
  ParsedFilmProfile& prof = bundle.targets[idx].film_prof;
  // The export's render of the grid: its encoding and the roll-off it measures (measure_rolloff_x_hi, as neg_process
  // measures every conversion's), which only the sRGB encoding applies. Before, the fit and the re-meter rendered
  // sRGB with the search's own roll-off, whatever the export's encoding, and the export then re-measured the roll-off
  // at the re-metered E on the combined profile, so the median L* it rendered was not the one the re-meter matched.
  out.encoding = encoding;
  auto rolloff_at = [&](const negicc::ProfileData& p, float e, float g, float b) {
    if (encoding != ENCODE_SRGB) return 1.0f;
    return measure_rolloff_x_hi(bundle, p, frame, scan_shutter_s, scan_iso, e, g, b, opts.crop_ratio, nullptr,
                                opts.film_base_rgb_d);
  };
  out.x_hi_fit = rolloff_at(prof.data, e_int, res.params.g, res.params.b);
  const negicc::PipelineConfig cfg = make_pipeline_config(prof.data, AUTO_KNEE, out.x_hi_fit, encoding);
  negicc::CurveFrame cf;
  cf.merged = merged;
  cf.knee = AUTO_KNEE;
  std::copy(fb, fb + 3, cf.film_base);
  cf.grid = stats.grid.data();
  cf.n = (int)n;
  // Auto's grid median L* on the bracket's own profile, as the export renders Auto's result: the lightness the fit
  // keeps.
  const float target_L = negicc::grid_median_L(cfg, merged, e_int, res.params.g, res.params.b, stats.grid.data(), (int)n);
  // Auto's loss parts on the same render; both before step 3 replaces the profile's tables (cfg points at them).
  out.parts_auto = negicc::grid_intent_loss_parts(cfg, merged, e_int, res.params.g, res.params.b, stats.grid.data(),
                                                  (int)n, res.intent_mu.data(), res.intent_sigma.data());

  // 1. The joint fit: g, b and the curves, each candidate scored at the E that keeps target_L (negicc::
  //    fit_input_curves_joint), its curves anchored at its own E, g and b. Before, the fit ran at Auto's E and the
  //    re-meter moved E afterwards, which moved the colour the fit had chosen.
  const int table_n[2] = {prof.data.in_trc_size[1], prof.data.in_trc_size[2]};
  const negicc::JointCurveFit jf = negicc::fit_input_curves_joint(cfg, cf, e_int, res.params.g, res.params.b,
                                                                  res.intent_mu.data(), res.intent_sigma.data(),
                                                                  target_L, table_n, fit_opts);
  out.fit.curves = jf.curves;
  out.fit.g = jf.g;
  out.fit.b = jf.b;
  out.fit.loss_before = jf.loss_before;
  out.fit.loss_after = jf.loss_after;
  out.fit.evaluations = jf.evaluations;
  out.fit_e = jf.e;
  out.fit_renders = jf.renders;
  out.fit_iterations = jf.iterations;
  out.fit_converged = jf.converged;
  out.fit_polish = jf.polish_evaluations;
  if (opts.verbose) {
    const negicc::InputCurves& c = out.fit.curves;
    std::cout << "Curves: G a " << fixed(c.a[0], 4) << " k " << fixed(c.k[0], 4) << ", B a " << fixed(c.a[1], 4)
              << " k " << fixed(c.k[1], 4) << "; E " << fixed(jf.e, 4) << " g " << fixed(out.fit.g, 4) << " b "
              << fixed(out.fit.b, 4) << ", loss " << fixed((float)out.fit.loss_before, 3) << " -> "
              << fixed((float)out.fit.loss_after, 3) << " (" << jf.iterations << " steps, " << jf.evaluations
              << " evaluations, " << jf.renders << " renders for E)\n";
  }
  {  // the loss's parts right after the fit, at its own E, on the real combined tables
    negicc::CombinedProfile c0;
    c0.build(prof.data, out.fit.curves);
    negicc::PipelineConfig c0cfg = cfg;
    c0cfg.profile = c0.data;
    out.parts_fit = negicc::grid_intent_loss_parts(c0cfg, merged, jf.e, out.fit.g, out.fit.b, stats.grid.data(),
                                                   (int)n, res.intent_mu.data(), res.intent_sigma.data());
  }

  // 2. E re-metered on the real combined tables (the fit solved it on its per-cell model, which composes the curve
  //    analytically instead of through the tables), the curves anchored at each E it tries; alternated with the
  //    roll-off the export measures at that E on the combined profile, until the roll-off holds still.
  negicc::CombinedProfile comb;
  float x = out.x_hi_fit, e_new = e_int;
  constexpr int kMaxPasses = 6;
  auto remeter = [&]() {
    out.remeter_converged = false;
    for (out.remeter_passes = 1; out.remeter_passes <= kMaxPasses; ++out.remeter_passes) {
      e_new = negicc::remeter_e_median(make_pipeline_config(prof.data, AUTO_KNEE, x, encoding), cf, out.fit.curves,
                                       e_int, out.fit.g, out.fit.b, target_L);
      comb.build(prof.data, negicc::anchor_curves(out.fit.curves, cf, e_new, out.fit.g, out.fit.b));
      const float x_new = rolloff_at(comb.data, e_new, out.fit.g, out.fit.b);
      if (x_new == x) {
        out.remeter_converged = true;
        break;
      }
      x = x_new;
    }
    out.remeter_passes = std::min(out.remeter_passes, kMaxPasses);
  };
  remeter();

  // The shape (monotonic, within max_dev_stops of the base TRC over the whole table) was checked where the fit anchored
  // the curves, at Auto's E; anchored at the re-metered E, ref moves and the table reaches further below it, so the
  // curves can step just past the limit (on the 41 white-LED frames up to 0.002 stops, on 4 frames). There a and k are
  // scaled down by the largest factor that keeps the shape at the export's anchor, and E is re-metered again.
  auto shape_ok_at = [&](const negicc::InputCurves& shape, float e) {
    const negicc::InputCurves c = negicc::anchor_curves(shape, cf, e, out.fit.g, out.fit.b);
    const float dev = fit_opts.max_dev_stops;
    return negicc::curve_shape_ok(c.a[0], c.k[0], c.u_med[0], c.ref[0], prof.data.in_trc_size[1], dev) &&
           negicc::curve_shape_ok(c.a[1], c.k[1], c.u_med[1], c.ref[1], prof.data.in_trc_size[2], dev);
  };
  auto scaled = [](negicc::InputCurves c, float t) {
    for (int ch = 0; ch < 2; ++ch) {
      c.a[ch] *= t;
      c.k[ch] *= t;
    }
    return c;
  };
  out.shape_scale = 1.0f;
  for (int round = 0; round < 3 && !shape_ok_at(out.fit.curves, e_new); ++round) {
    float lo = 0.0f, hi = 1.0f;  // the identity (t = 0) always passes
    for (int it = 0; it < 24; ++it) {
      const float mid = 0.5f * (lo + hi);
      if (shape_ok_at(scaled(out.fit.curves, mid), e_new)) lo = mid;
      else hi = mid;
    }
    out.fit.curves = scaled(out.fit.curves, lo);
    out.shape_scale *= lo;
    remeter();
  }
  if (!shape_ok_at(out.fit.curves, e_new)) {  // still past the limit after three rounds: no curves
    out.fit.curves = scaled(out.fit.curves, 0.0f);
    out.shape_scale = 0.0f;
    remeter();
  }

  // 3. The curves anchored at the parameters the export renders with (the re-metered E, the fitted g and b), and the
  //    combined profile built from them in place: every later use of the bracket renders through it.
  out.fit.curves = negicc::anchor_curves(out.fit.curves, cf, e_new, out.fit.g, out.fit.b);
  apply_input_curves(prof, out.fit.curves);
  out.x_hi = rolloff_at(prof.data, e_new, out.fit.g, out.fit.b);
  const negicc::PipelineConfig cfg_comb = make_pipeline_config(prof.data, AUTO_KNEE, out.x_hi, encoding);
  out.params = {e_new, out.fit.g, out.fit.b, out.x_hi};
  out.target_median_L = target_L;
  out.median_L = negicc::grid_median_L(cfg_comb, merged, e_new, out.fit.g, out.fit.b, stats.grid.data(), (int)n);
  out.parts_export = negicc::grid_intent_loss_parts(cfg_comb, merged, e_new, out.fit.g, out.fit.b, stats.grid.data(),
                                                    (int)n, res.intent_mu.data(), res.intent_sigma.data());
  out.loss_export = out.parts_export.total();
  out.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_fit0).count();
  if (opts.verbose) {
    std::cout << "Curves: E re-metered " << fixed(e_int, 4) << " -> " << fixed(e_new, 4) << " (grid median L* "
              << fixed(out.median_L, 2) << ", Auto's " << fixed(target_L, 2) << "); g " << fixed(out.params.g, 4)
              << " b " << fixed(out.params.b, 4) << "\n";
  }
  return true;
}
