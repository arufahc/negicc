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
#include "dinov3_preprocess.h"
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
// Ranking and metering depth: the cost, per stop, of a scene that overflows an infeasible bracket's calibrated range.
constexpr float SLACK_PENALTY_SCALE = 0.5f;
// The gains (g, b) the bootstrap meters and renders at: before any intent field exists the search needs fixed, roughly
// neutral gains (the station's).
constexpr float BOOTSTRAP_G = 1.10f, BOOTSTRAP_B = 1.15f;
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
// either side of e_center (within 1.5 stops of highlight and 1.2 stops of shadow tolerance): the wider the scene's
// overflow, the wider the window (0.6 stop per stop of overflow), so E can trade highlights against shadows. The
// tolerances let a scene overflow the calibration a little; their values are kept from negicc-station, which records
// no reason for them.
struct Enclosure {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  bool feasible = false;    // e_min <= e_max
  bool admissible = true;   // the scene's span fits the bracket's span plus the 2.7 stops of tolerance
  float e_min = 0, e_max = 0;
  float search_lo = 0, search_hi = 0;  // the range E is searched in
  float e_center = 0;                  // sqrt(e_min * e_max)
  float slack_ev = 0;                  // log2(e_max / e_min): spare stops, negative when not feasible
  float fill_ratio = 0;                // (g_hi / g_lo) / (g_max / g_min): the scene's contrast ratio over the bracket's
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
    // The usable range can cut the window from both sides past each other (u_min above e_center * 2^margin, or u_max
    // below e_center / 2^margin): ordered here, so no caller searches an inverted window.
    if (r.search_lo > r.search_hi) std::swap(r.search_lo, r.search_hi);
  }
  r.fill_ratio = (g_hi / g_lo) / (tgt_max / tgt_min);
  return true;
}

// Brackets whose grey wedge spans at least 0.75 of the largest tonal capacity (at least three kept), when the
// bundle has five or more: a bracket that calibrates a much shorter range than the best ones cannot hold a whole
// scene, so it is not worth screening.
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
  // the ceiling's own code, which reads the excess higher than the station does. Ceilings at or below 1 are
  // fractions of full scale.
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

  // The inner crop rendered to 8-bit sRGB through bracket target_idx at params: what DINOv3 sees. out is crop_.w x
  // crop_.h pixels, row-major, 3 bytes each; DINOv3 resamples it to its input size.
  void render_preview(int target_idx, const ConversionParams& params, std::vector<uint8_t>& out) {
    const Clock::time_point t0 = Clock::now();
    float m[9];  // input_matrix_ with its rows scaled by E, E g and E b
    negicc::candidate_matrix(input_matrix_, params.e, params.g, params.b, m);
    out.resize((size_t)crop_.w * crop_.h * 3);
    negicc::RgbView crop_view = frame_.crop_and_orient(crop_.x0, crop_.y0, crop_.w, crop_.h, 0, false, false);
    negicc::RenderArgs ra{};
    ra.in_view = &crop_view;
    ra.width = crop_.w;
    ra.height = crop_.h;
    ra.matrix = m;
    ra.cfg = make_pipeline_config(bundle_.targets[target_idx].film_prof.data, AUTO_KNEE, shoulder_of(params));
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
    // Every loss divides by sigma: a non-positive or non-finite sigma, or a non-finite mu, would make the search's
    // and the curve fit's losses NaN or infinite (as dinov3_auto_solver.cc's infer_checked rejects them).
    for (size_t i = 0; i < n; ++i) {
      if (!(r.sigma[i] > 0.0f) || !std::isfinite(r.sigma[i]) || !std::isfinite(r.mu[i])) {
        std::cerr << "ERROR! DINOv3 intent field has a non-positive or non-finite sigma or mu at " << i << std::endl;
        return false;
      }
    }
    mu = std::move(r.mu);
    sigma = std::move(r.sigma);
    last_neutral_p_ = r.neutral_p.size() == n / 3 ? std::move(r.neutral_p) : std::vector<float>();
    last_tone_features_.clear();
    if (r.pooled.size() == 1152 && r.token_sd.size() == 768) {
      last_tone_features_ = r.pooled;
      last_tone_features_.insert(last_tone_features_.end(), r.token_sd.begin(), r.token_sd.end());
    }
    return true;
  }

  // The neutral head's p per cell from the last predict() (empty without a neutral head).
  const std::vector<float>& last_neutral_p() const { return last_neutral_p_; }
  // The tone heads' input from the last predict() (FixedPointResult::tone_features; empty without the token spread).
  const std::vector<float>& last_tone_features() const { return last_tone_features_; }

  // The inner crop rendered to 8-bit sRGB through the tables pd (a bracket's profile with changed input tables) at
  // params, then resampled to DINOv3's input size as the engine resamples it (net_w x net_h pixels).
  void render_net_preview(const ProfileData& pd, const ConversionParams& params, std::vector<uint8_t>& out,
                          int& net_w, int& net_h) {
    const Clock::time_point t0 = Clock::now();
    float m[9];
    negicc::candidate_matrix(input_matrix_, params.e, params.g, params.b, m);
    std::vector<uint8_t> full((size_t)crop_.w * crop_.h * 3);
    negicc::RgbView crop_view = frame_.crop_and_orient(crop_.x0, crop_.y0, crop_.w, crop_.h, 0, false, false);
    negicc::RenderArgs ra{};
    ra.in_view = &crop_view;
    ra.width = crop_.w;
    ra.height = crop_.h;
    ra.matrix = m;
    ra.cfg = make_pipeline_config(pd, AUTO_KNEE, shoulder_of(params));
    ra.out_type = 1;
    ra.out = full.data();
    negicc::cpu_render(ra);
    const negicc::AspectPreservedShape sh =
        negicc::compute_aspect_preserved_shape(crop_.w, crop_.h, engine_.max_size(), engine_.patch_size());
    net_w = sh.target_w;
    net_h = sh.target_h;
    out.resize((size_t)net_w * net_h * 3);
    antialiased_bilinear_resize_rgb(full.data(), crop_.w, crop_.h, crop_.w * 3, out.data(), net_w, net_h);
    render_time_ += Clock::now() - t0;
  }

  // The intent field DINOv3 predicts from a preview already at its input size (render_net_preview), on the
  // statistics' grid, and the tone heads' input from the same inference (empty without the token spread). False
  // (reported) when the inference fails or its grid is not the frame's.
  bool predict_net(const std::vector<uint8_t>& preview, int w, int h, std::vector<float>& mu, std::vector<float>& sigma,
                   std::vector<float>& tone_features) {
    const Clock::time_point t0 = Clock::now();
    DinoV3InferenceResult r;
    try {
      r = engine_.infer(preview.data(), w, h, w * 3);
    } catch (const std::exception& e) {
      std::cerr << "ERROR! DINOv3 inference failed: " << e.what() << std::endl;
      return false;
    }
    infer_time_ += Clock::now() - t0;
    out_.inferences++;
    const size_t n = stats_.grid.size();
    if (r.grid_w != stats_.grid_w || r.grid_h != stats_.grid_h || r.mu.size() != n || r.sigma.size() != n) return false;
    mu = std::move(r.mu);
    sigma = std::move(r.sigma);
    tone_features.clear();
    if (r.pooled.size() == 1152 && r.token_sd.size() == 768) {
      tone_features = r.pooled;
      tone_features.insert(tone_features.end(), r.token_sd.begin(), r.token_sd.end());
    }
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
  // (ScreeningResult::intent_loss). pullback_search (the highlight pullback's re-search) replaces the pull of E toward
  // the centre with penalties on E outside the bracket's calibrated range and, when initial is null (its screening),
  // searches g and b on wider grids (dinov3_auto_solver.cc).
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (optimize_target_for_intent)
  float search(int target_idx, const std::vector<float>& mu, const std::vector<float>& sigma,
                const ConversionParams* initial, const float e_bounds[3], ConversionParams& out,
                bool pullback_search = false) {
    const Clock::time_point t0 = Clock::now();
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    cfg.shoulder = initial ? shoulder_of(*initial) : shoulder_fixed_;
    cfg.wide_pullback_grids = !bundle_.has_crosstalk;
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
  // kept within the trust region e_bounds = {lo, hi, centre}; outside pullback_search the search's penalty pulls E
  // toward the centre. An inverted region is ordered and centred on its midpoint. pullback_search as for search().
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
    std::vector<float> best_mu, best_sigma, best_p, best_tone;
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
        best_p = last_neutral_p_;
        best_tone = last_tone_features_;
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
      // What counts as converged, besides parameters that hold still (dp). TriLED: a residual that holds still within
      // TOL_REL_ERR. White light: a residual past its minimum, and not a residual that holds still: there it often
      // moves by about 1.5% a step while g and b are still walking, and a change of 0.1% in one channel then decided
      // whether the loop stopped at its second step or ran on to a residual 15% lower. Another bracket won and the
      // picture's median moved by up to 19 L*.
      const bool rel_conv = bundle_.has_crosstalk && (!pullback_search || err <= prev_err) && rel <= TOL_REL_ERR;
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
    fp.neutral_p = std::move(best_p);
    fp.tone_features = std::move(best_tone);
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
  std::vector<float> last_neutral_p_;  // predict(): the neutral head's p of the last inference
  std::vector<float> last_tone_features_;  // predict(): the tone heads' input from the last inference
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

void meter_bracket_exposures(const ProfileBundle& bundle, const negicc::RgbView& frame, double scan_shutter_s,
                             double scan_iso, float crop_ratio, const double* film_base_rgb,
                             std::vector<BracketExposure>& out) {
  out.assign(bundle.targets.size(), BracketExposure{});
  ParamSearchOptions opts;
  opts.film_base_rgb_d = film_base_rgb;
  opts.film_base_from_scan = film_base_rgb != nullptr;
  const FrameScale scale = frame_scale(bundle, (float)scan_shutter_s, (float)scan_iso, opts);
  // The statistics the search reads (SearchFrame::set_frame); the grid's size does not change g_lo and g_hi.
  FrameStats stats;
  compute_frame_stats(frame, crop_ratio, 224, 16, stats);
  // As search_conversion_params' Step 1.
  const bool white_light = !bundle.has_crosstalk;
  const float g_lo = std::max(1.0f, stats.g_lo);
  const float g_hi = std::max(g_lo + 1.0f, stats.g_hi);
  const float g_scale = white_light ? scale.e_norm : 1.0f;
  // The scene vector Step 2 meters: the search's input matrix (SearchFrame::set_frame_matrix: M_frame on white light,
  // M_frame / e_norm on TriLED) applied to the log-average RGB.
  const float mat_div = white_light ? 1.0f : scale.e_norm;
  float p_scene[3];
  for (int r = 0; r < 3; ++r) {
    p_scene[r] = 0.0f;
    for (int c = 0; c < 3; ++c) p_scene[r] += scale.frame_matrix[r * 3 + c] / mat_div * stats.log_avg_rgb[c];
  }
  for (size_t i = 0; i < bundle.targets.size(); ++i) {
    if (!(bundle.targets[i].exposure_s > 0.0f)) continue;
    Enclosure r;
    if (!bracket_enclosure(bundle.targets[i], (int)i, g_lo * g_scale, g_hi * g_scale, r)) continue;
    // The search's E is the station's on white light; on TriLED it runs on M_frame / e_norm, and its result is
    // params.e / e_norm (search_conversion_params' out.params).
    const float to_station = white_light ? 1.0f : 1.0f / scale.e_norm;
    BracketExposure& b = out[i];
    b.feasible = r.feasible;
    b.e_min = r.e_min * to_station;
    b.e_max = r.e_max * to_station;
    b.e_center = r.e_center * to_station;
    b.search_lo = r.search_lo * to_station;
    b.search_hi = r.search_hi * to_station;
    b.valid = b.e_min > 0.0f && b.e_max > 0.0f && std::isfinite(b.e_center) && b.e_center > 0.0f;
    b.e_metered = b.e_center;
    const GreyPoint gp(bundle.targets[i].film_prof.data);
    float e_met = 0.0f;
    if (gp.reference_grey() && exposure_for_lightness(gp, p_scene, BOOTSTRAP_G, BOOTSTRAP_B, e_met) && e_met > 0.0f) {
      b.has_metered = true;
      b.e_metered = std::clamp(e_met, r.search_lo, r.search_hi) * to_station;
    }
  }
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
  // As SearchFrame::measure_shoulder: M_frame at E (Auto's units), written as (M_frame / e_norm) * (E * e_norm).
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
                      float tint_a, float warmth_b, KnobSolveResult& out, const ProfileData* grey_prof) {
  out.e = base_e;
  out.g = base_g;
  out.b = base_b;
  out.limited = false;
  out.grey_lab[0] = 50.0f;
  out.grey_lab[1] = 0.0f;
  out.grey_lab[2] = 0.0f;
  if (!(base_e > 0.0f) || !(base_g > 0.0f) || !(base_b > 0.0f)) return false;

  const GreyPoint gp(prof);
  // The grey: the input grey_prof (else prof) renders as neutral L* 50.
  float log_t0[3];
  if (!GreyPoint(grey_prof ? *grey_prof : prof).reference_grey(log_t0)) return false;
  const float base[3] = {base_e, base_g, base_b};

  if (brightness_stops == 0.0f && tint_a == 0.0f && warmth_b == 0.0f) {
    gp.grey(log_t0, base, base, out.grey_lab);
    return true;
  }

  // A target Lab counts as reached when the Newton solve lands within 0.25 (Lab units) of it. An unreachable target
  // (past the profile's gamut at that grey) keeps as much as reachable: the whole colour change with the brightness
  // scaled back, else colour and brightness scaled back together, each by a 7-step bisection (to 1% of the change).
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

bool knob_grey_lab(const ProfileData& grey_prof, const ProfileData& prof, const float base[3], const float egb[3],
                   float lab_out[3]) {
  float log_t0[3];
  if (!GreyPoint(grey_prof).reference_grey(log_t0)) return false;
  GreyPoint(prof).grey(log_t0, base, egb, lab_out);
  return true;
}

float knob_grey_miss_stops(float brightness_stops, float grey_L) {
  auto Y = [](double L) { return L > 8.0 ? std::pow((L + 16.0) / 116.0, 3.0) : L / 903.3; };
  return (float)std::log2(Y(grey_L) / Y(lstar_after_stops(50.0f, brightness_stops)));
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

  // The bracket whose calibrated range the scene fits best, among the admissible ones when there are any: the lowest
  // |fill_ratio - 0.85| (the scene's green contrast ratio over the bracket's) plus SLACK_PENALTY_SCALE per stop the
  // scene overflows an infeasible bracket by. Only the bootstrap falls back to it, when no bracket meters.
  auto rank = [](const Enclosure& r) {
    return std::fabs(r.fill_ratio - 0.85f) + SLACK_PENALTY_SCALE * std::max(0.0f, -r.slack_ev);
  };
  const Enclosure best_fit = *std::min_element(
      enclosure.begin(), enclosure.end(), [&](const Enclosure& a, const Enclosure& b) {
        if (a.admissible != b.admissible) return a.admissible;
        return rank(a) < rank(b);
      });

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
  // A metered bracket: its E for L* 50 (e_met), the window [lo, hi] E is searched in, and the depth of e_met inside
  // that window in stops, less SLACK_PENALTY_SCALE per stop of overflow on an infeasible bracket. The deepest bracket
  // renders the bootstrap: its metered E sits furthest from the ends of its search window.
  struct Depth {
    Enclosure cand;
    float e_met, depth, lo, hi;
  };
  auto metered = [](const Enclosure& c, float e_met, float lo, float hi) {
    const float depth = std::min(std::log2(e_met / std::max(1e-6f, lo)), std::log2(std::max(1e-6f, hi) / e_met)) -
                        SLACK_PENALTY_SCALE * std::max(0.0f, -c.slack_ev);
    return Depth{c, e_met, depth, lo, hi};
  };
  auto deepest_first = [](std::vector<Depth>& v) {
    std::stable_sort(v.begin(), v.end(), [](const Depth& a, const Depth& b) { return a.depth > b.depth; });
  };
  std::vector<Depth> depths;
  for (const auto& c : pool) {
    const GreyPoint gp(bundle.targets[c.target_idx].film_prof.data);
    float e_met = 0.0f;
    if (!gp.reference_grey()) {
      if (verbose) std::cout << "  " << name_of(c.target_idx) << ": no neutral L* 50 in this bracket, not metered\n";
      continue;
    }
    if (!exposure_for_lightness(gp, p_scene, BOOTSTRAP_G, BOOTSTRAP_B, e_met)) {
      if (verbose) std::cout << "  " << name_of(c.target_idx) << ": metering did not converge\n";
      continue;
    }
    depths.push_back(metered(c, e_met, c.search_lo, c.search_hi));
  }
  Enclosure cand1;
  float e1_init;
  if (!depths.empty()) {
    deepest_first(depths);
    cand1 = depths[0].cand;
    e1_init = std::clamp(depths[0].e_met, depths[0].lo, depths[0].hi);
  } else {
    cand1 = best_fit;
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
  fr.render_preview(cand1.target_idx, {e1_init, BOOTSTRAP_G, BOOTSTRAP_B}, preview);
  if (!fr.predict(preview, mu1, sig1)) return false;

  // Step 3: screen the metered brackets (every bracket of the pool when none metered) against one intent field, best
  // first. The pullback's re-search (Step 6) screens with the enclosure penalties in place of the pull toward the
  // centre (pullback_search).
  auto screen_all = [&](const std::vector<Depth>& ds, const std::vector<float>& mu, const std::vector<float>& sigma,
                        bool pullback_search) {
    std::vector<Screened> rows;
    auto screen = [&](const Enclosure& c, float lo, float hi, float e_met) {
      Screened s;
      s.enclosure = c;
      s.result.target_idx = c.target_idx;
      s.result.e_lo = lo;
      s.result.e_hi = hi;
      s.result.e_metered = e_met;
      const float eb[3] = {lo, hi, c.e_center};
      s.result.intent_loss = fr.search(c.target_idx, mu, sigma, nullptr, eb, s.result.params, pullback_search);
      rows.push_back(s);
    };
    if (!ds.empty()) {
      for (const auto& d : ds) screen(d.cand, d.lo, d.hi, d.e_met);
    } else {
      for (const auto& c : pool) screen(c, c.search_lo, c.search_hi, c.e_center);
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Screened& a, const Screened& b) {
      return a.result.intent_loss < b.result.intent_loss;
    });
    return rows;
  };
  const std::vector<Screened> screening = screen_all(depths, mu1, sig1, false);
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

  // Step 4: the fixed-point loop on the best two screened brackets, each within a trust region of +/- 0.5 stop
  // around its screened E (the screening's single field is a rough guide; the loop refines, not relocates, E).
  // (The station also drops a second bracket screened more than 2.0 worse, then takes the best two anyway when fewer
  // than two remain, so the margin never binds.)
  auto trust_bounds = [](const Screened& c, float eb[3]) {
    eb[0] = std::max(c.result.e_lo, c.result.params.e * std::pow(2.0f, -0.5f));
    eb[1] = std::min(c.result.e_hi, c.result.params.e * std::pow(2.0f, 0.5f));
    eb[2] = c.enclosure.e_center;
  };
  auto tournament = [&](const std::vector<Screened>& rows, bool pullback_search, std::vector<FixedPointResult>& runs) {
    for (size_t i = 0; i < std::min<size_t>(2, rows.size()); ++i) {
      const Screened& c = rows[i];
      float eb[3];
      trust_bounds(c, eb);
      FixedPointResult fp;
      if (verbose && !pullback_search) std::cout << "Auto: fixed point on " << name_of(c.result.target_idx) << "\n";
      if (!fr.fixed_point(c.result.target_idx, c.result.params, eb, fp, pullback_search)) return false;
      runs.push_back(fp);
    }
    return true;
  };
  // The winner of a tournament: the lowest score, with prefer_converged among the converged runs when any converged.
  auto winner = [](const std::vector<FixedPointResult>& runs, bool prefer_converged,
                   auto score) -> const FixedPointResult* {
    bool any_converged = false;
    for (const auto& a : runs) any_converged |= prefer_converged && a.status == FixedPointStatus::kConverged;
    const FixedPointResult* win = nullptr;
    for (const auto& a : runs) {
      if (any_converged && a.status != FixedPointStatus::kConverged) continue;
      if (!win || score(a) < score(*win)) win = &a;
    }
    return win;
  };
  std::vector<FixedPointResult> adaptive;
  if (!tournament(screening, false, adaptive)) return false;
  out.tournament = adaptive;
  // The winner is the lowest plain residual; on TriLED among the converged brackets when any converged. On white light
  // the status does not count: each loop's best step is a render with a measured residual, and whether a loop ended as
  // converged, diverged or out of steps flipped with a 0.1% change of the input.
  FixedPointResult chosen =
      *winner(adaptive, bundle.has_crosstalk, [](const FixedPointResult& a) { return a.residual; });
  ConversionParams params = chosen.params;

  // The enclosure-regularised residual that ranks the pullback's tournament (Step 6): the residual plus quadratic
  // penalties on E outside the bracket's calibrated range, heaviest on the highlight side (E below e_min), and on the
  // scene overflowing an infeasible bracket. The weights are those of the pullback search's own penalties
  // (dinov3_auto_solver.cc compute_penalties).
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
  // Each bracket's enclosure. Every bracket the search can choose was screened, so it is in enclosure.
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
      } else if (white_light && out.clip_fallback == ClipFallbackOutcome::kClipsTooMuch && clip_0 > 0.02f) {
        // When severe highlight clipping (>2% white-clipped) is only partially reduced by chosen + 1, continue
        // walking denser brackets with pullback_search (wide g/b range) to find the deepest bracket that clears
        // the highlight clipping or highlight overshoot within the lightness and residual gates.
        float best_clip = clip_adj;
        float best_q995 = fp_adj.shoulder_q995;
        const float overshoot_0 = std::max(0.0f, chosen.shoulder_q995 - 1.0f);
        const float resid_0 = chosen.residual;
        const int start_idx = chosen.target_idx + 2;
        for (int t_cand = start_idx; t_cand < (int)bundle.targets.size(); ++t_cand) {
          const Screened* s_cand = nullptr;
          for (const auto& s : screening)
            if (s.result.target_idx == t_cand) s_cand = &s;
          if (!s_cand || !s_cand->enclosure.admissible) continue;
          ConversionParams p_start = s_cand->result.params;
          const float eb_screen[3] = {s_cand->result.e_lo, s_cand->result.e_hi, s_cand->enclosure.e_center};
          fr.search(t_cand, mu1, sig1, nullptr, eb_screen, p_start, true);
          Screened s_pb = *s_cand;
          s_pb.result.params = p_start;
          float eb[3];
          trust_bounds(s_pb, eb);
          FixedPointResult fp_cand;
          if (verbose) std::cout << "Auto: fixed point on " << name_of(t_cand) << " (clip fallback multi-step)\n";
          if (!fr.fixed_point(t_cand, p_start, eb, fp_cand, true)) return false;
          ConversionParams p_cand = fp_cand.params;
          ClipStats hc_cand = fr.highlight_clip(t_cand, {p_cand}, ceiling_of(t_cand))[0];
          float dl_cand = hc_cand.median_L - hc0.median_L;
          if (dl_cand < CLIP_FALLBACK_DL_MIN) {
            p_cand.e *= std::pow(2.0f, (dl_cand - CLIP_FALLBACK_DL_MIN) * 0.038f);
            hc_cand = fr.highlight_clip(t_cand, {p_cand}, ceiling_of(t_cand))[0];
            dl_cand = hc_cand.median_L - hc0.median_L;
          }
          const float clip_cand = hc_cand.white_clip / 100.0f;
          const float overshoot_cand = std::max(0.0f, fp_cand.shoulder_q995 - 1.0f);
          const bool clears_overshoot =
              (overshoot_0 > 0.15f && overshoot_cand <= 0.10f * overshoot_0 && clip_cand < 0.70f * clip_0);
          ClipFallbackOutcome cand_outcome = ClipFallbackOutcome::kTaken;
          if (clip_cand > CLIP_FALLBACK_CLIP_RATIO * clip_0 && !clears_overshoot)
            cand_outcome = ClipFallbackOutcome::kClipsTooMuch;
          else if (!(CLIP_FALLBACK_DL_MIN <= dl_cand && dl_cand <= CLIP_FALLBACK_DL_MAX))
            cand_outcome = ClipFallbackOutcome::kShiftsLightness;
          else if (fp_cand.residual > (1.0f + (clears_overshoot ? 0.12f : CLIP_FALLBACK_RESID_TOL)) * resid_0)
            cand_outcome = ClipFallbackOutcome::kWorseResidual;
          if (verbose) {
            std::cout << "Auto: clip fallback: " << name_of(t_cand) << " clips " << fixed(clip_cand * 100.0f, 4)
                      << "% vs " << fixed(clip_0 * 100.0f, 4) << "%, dL* " << fixed(dl_cand, 2, true) << ", resid "
                      << fixed(fp_cand.residual, 4) << ": " << to_string(cand_outcome) << "\n";
          }
          if (cand_outcome == ClipFallbackOutcome::kTaken &&
              (out.clip_fallback != ClipFallbackOutcome::kTaken ||
               (clears_overshoot ? fp_cand.shoulder_q995 < best_q995 : clip_cand < best_clip))) {
            out.clip_fallback = ClipFallbackOutcome::kTaken;
            fp_cand.params = p_cand;
            chosen = fp_cand;
            params = p_cand;
            best_clip = clip_cand;
            best_q995 = fp_cand.shoulder_q995;
          }
        }
      }
    }
    if (verbose && (out.clip_fallback == ClipFallbackOutcome::kNotScreened || out.clip_fallback == ClipFallbackOutcome::kNotAdmissible)) {
      std::cout << "Auto: clip fallback: " << to_string(out.clip_fallback) << "\n";
    }
  }

  // Step 6: the highlight pullback. A frame whose render puts many highlights above sRGB 242 (or 235) while its
  // midtones are bright (median L* >= 58, or >= 48 with more highlights) was rendered too bright by an E below the
  // bracket's calibrated range, or by a bracket whose range cannot hold the scene. A dark or backlit frame (low
  // median L*) is left alone: its highlights are the subject's surroundings, and darkening would sink the subject.
  //   tier 1: median L* >= 58 and >= 2.2% of the inner crop above sRGB 242 in the worst channel;
  //   tier 2: median L* >= 48 and >= 7.5% above 242 in the worst channel, or, on bracket index 3 and above, >= 11%
  //           above 235 in the worst channel.
  // When the chosen bracket is feasible, E is raised toward the bracket's e_min by at most 0.15 stop (a nudge, not a
  // re-solve: the loop's intent stays close). When it is not, Steps 2-4 run again on windows that reach a little
  // further (pullback_search: enclosure penalties instead of the pull toward the centre), ranked by the
  // enclosure-regularised residual (tourney_score) among the converged runs when any converged, on white light as on
  // TriLED, and the result is taken only if it clips no more, does not darken the midtones past what the scene allows,
  // and moves to a later bracket or raises E by more than 2%.
  // The thresholds are negicc-station's, ported unchanged so both pick the same result.
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
      // On white light the re-search runs only when the next bracket (chosen + 1) was competitive in the shared-intent
      // screening pass, over chosen and chosen + 1 alone, and its result must keep the clip fallback's residual
      // tolerance of chosen (the B5 port). TriLED keeps the station's re-search over every metered bracket, for parity.
      float chosen_screen_loss = 0.0f;
      bool found_chosen_screen = false;
      bool adj_screen_ok = !white_light;
      if (white_light) for (const auto& s : screening) {
        if (s.result.target_idx == chosen.target_idx) {
          chosen_screen_loss = s.result.intent_loss;
          found_chosen_screen = true;
          break;
        }
      }
      if (white_light && found_chosen_screen) {
        for (const auto& s : screening) {
          if (s.result.target_idx == chosen.target_idx + 1 &&
              s.result.intent_loss <= (1.0f + CLIP_FALLBACK_RESID_TOL) * chosen_screen_loss) {
            adj_screen_ok = true;
            break;
          }
        }
      }
      // The re-search window of each metered bracket: its calibrated range united with the infeasible window
      // (bracket_enclosure), its upper end raised to at least 0.2 stop above its metered E (within the usable range),
      // so the pullback can reach the darker-than-metered side the first search could not.
      std::vector<Depth> depths_res;
      if (adj_screen_ok) {
        for (const auto& c : pool) {
          if (white_light && (c.target_idx < chosen.target_idx || c.target_idx > chosen.target_idx + 1)) continue;
          const auto d = std::find_if(depths.begin(), depths.end(),
                                      [&](const Depth& x) { return x.cand.target_idx == c.target_idx; });
          if (d == depths.end()) continue;
          const float e_met = d->e_met;
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
          depths_res.push_back(metered(c, e_met, lo, hi));
        }
      }
      // A new bootstrap field when the deepest bracket or its E changed (else the first one is the same field).
      std::vector<float> mu_res = mu1, sig_res = sig1;
      if (!depths_res.empty()) {
        deepest_first(depths_res);
        const Enclosure cand_res = depths_res[0].cand;
        const float e_res_init = std::clamp(depths_res[0].e_met, depths_res[0].lo, depths_res[0].hi);
        if (cand_res.target_idx != cand1.target_idx || std::fabs(e_res_init - e1_init) > 1e-4f) {
          std::vector<uint8_t> preview_res;
          fr.render_preview(cand_res.target_idx, {e_res_init, BOOTSTRAP_G, BOOTSTRAP_B}, preview_res);
          if (!fr.predict(preview_res, mu_res, sig_res)) return false;
        }
      }
      std::vector<FixedPointResult> res_adaptive;
      if ((!white_light || !depths_res.empty()) &&
          !tournament(screen_all(depths_res, mu_res, sig_res, true), true, res_adaptive))
        return false;
      if (const FixedPointResult* win_res = winner(res_adaptive, bundle.has_crosstalk, tourney_score)) {
        FixedPointResult fp_res = *win_res;
        ConversionParams p_res = fp_res.params;
        ClipStats hc_res_242 = fr.highlight_clip(fp_res.target_idx, {p_res}, ceil_242)[0];
        // The midtones may darken by up to 7.5 L* on a bright frame (median L* >= 58), 4.5 otherwise; past that, E
        // is lowered (brightening) by 0.038 stop per L* of overshoot, one partial correction, and measured again.
        const float dl_min = (hc_242.median_L >= 58.0f) ? -7.5f : -4.5f;
        float dl = hc_res_242.median_L - hc_242.median_L;
        if (dl < dl_min) {
          p_res.e *= std::pow(2.0f, (dl - dl_min) * 0.038f);
          hc_res_242 = fr.highlight_clip(fp_res.target_idx, {p_res}, ceil_242)[0];
          dl = hc_res_242.median_L - hc_242.median_L;
        }
        // Taken when it clips no more (at most 0.5 percentage point more of the crop above 242 in the worst channel),
        // keeps the midtones at L* 48 or within 6 of the first choice's, stays within the clip fallback's residual
        // tolerance of chosen, and actually moves: a later bracket, or E up by more than 2%.
        const float min_allowed_L = std::min(48.0f, hc_242.median_L - 6.0f);
        const float exc_res_242 = max_excess(hc_res_242);
        if (exc_res_242 <= exc_242 + 0.005f && hc_res_242.median_L >= min_allowed_L &&
            (!white_light || fp_res.residual <= (1.0f + CLIP_FALLBACK_RESID_TOL) * chosen.residual) &&
            (fp_res.target_idx > chosen.target_idx || p_res.e > params.e * 1.02f)) {
          chosen = fp_res;
          params = p_res;
          out.highlight_pullback = true;
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
  out.intent_neutral_p = chosen.neutral_p;
  out.intent_tone_features = chosen.tone_features;
  out.intent_grid_w = chosen.grid_w;
  out.intent_grid_h = chosen.grid_h;
  out.intent_params = chosen.params;
  {
    Enclosure enc{};
    if (bounds_of(chosen.target_idx, enc) && enc.feasible) {
      out.e_range_lo = white_light ? enc.e_min : enc.e_min / e_norm;
      out.e_range_hi = white_light ? enc.e_max : enc.e_max / e_norm;
    }
  }
  fr.write_timings();
  out.total_time = Clock::now() - t_start;
  return true;
}

// ---- Fitted G/B input curves ---------------------------------------------------------------------------------

void apply_input_curves(ParsedFilmProfile& prof, const negicc::InputCurves& curves) {
  if (!curves.active() || !prof.data.has_profile) return;
  const float kr = (curves.k_r != 0.0f) ? curves.k_r : (curves.k[0] < 0.0f ? 0.8228f * curves.k[0] : 0.0f);
  if (kr < 0.0f && prof.in_trc[0].size() >= 2) {
    std::vector<float>& table = prof.in_trc[0];
    std::vector<float> composed(table.size());
    const float refr = (curves.ref_r > 0.0f) ? curves.ref_r : std::min(1.0f, curves.ref[0] * 1.0905f);
    negicc::compose_input_table(table.data(), (int)table.size(), refr, 0.0f, 0.0f, kr, composed.data(),
                                curves.u_hold_r(), 0.36f, 0.76f);
    table.swap(composed);
  }
  const float ts_gb[2] = {0.30f, 0.46f};
  for (int ch = 1; ch <= 2; ++ch) {
    std::vector<float>& table = prof.in_trc[ch];
    if (table.size() < 2) continue;
    std::vector<float> composed(table.size());
    const int i = ch - 1;
    negicc::compose_input_table(table.data(), (int)table.size(), curves.ref[i], curves.u_med[i], curves.a[i],
                                curves.k[i], composed.data(), curves.u_hold(i), ts_gb[i], 0.76f);
    table.swap(composed);
  }
  prof.rebind_pointers();
}

void apply_input_curves(ParsedFilmProfile& prof, const ParsedFilmProfile& base, const negicc::InputCurves& curves) {
  const float kr = (curves.k_r != 0.0f) ? curves.k_r : (curves.k[0] < 0.0f ? 0.8228f * curves.k[0] : 0.0f);
  {
    const std::vector<float>& table_r = base.in_trc[0];
    if (!curves.active() || !(kr < 0.0f) || !base.data.has_profile || table_r.size() < 2) {
      prof.in_trc[0] = table_r;
    } else {
      std::vector<float> composed(table_r.size());
      const float refr = (curves.ref_r > 0.0f) ? curves.ref_r : std::min(1.0f, curves.ref[0] * 1.0905f);
      negicc::compose_input_table(table_r.data(), (int)table_r.size(), refr, 0.0f, 0.0f, kr, composed.data(),
                                  curves.u_hold_r(), 0.36f, 0.76f);
      prof.in_trc[0].swap(composed);
    }
  }
  const float ts_gb[2] = {0.30f, 0.46f};
  for (int ch = 1; ch <= 2; ++ch) {
    const std::vector<float>& table = base.in_trc[ch];
    if (!curves.active() || !base.data.has_profile || table.size() < 2) {
      prof.in_trc[ch] = table;
      continue;
    }
    std::vector<float> composed(table.size());
    const int i = ch - 1;
    negicc::compose_input_table(table.data(), (int)table.size(), curves.ref[i], curves.u_med[i], curves.a[i],
                                curves.k[i], composed.data(), curves.u_hold(i), ts_gb[i], 0.76f);
    prof.in_trc[ch].swap(composed);
  }
  prof.rebind_pointers();
}

// What anchors the curves on a frame (negicc::CurveFrame) from its statistics: the search's matrix, the knee, and the
// film base read from the scan (opts), else the brightest grid cell per channel, as the station takes it. stats must
// outlive the result (its grid is not copied).
static negicc::CurveFrame curve_frame_of(const FrameStats& stats, const float* merged, const ParamSearchOptions& opts) {
  negicc::CurveFrame cf;
  cf.merged = merged;
  cf.knee = AUTO_KNEE;
  const size_t n = stats.grid.size() / 3;
  if (opts.film_base_rgb_d) {
    for (int c = 0; c < 3; ++c) cf.film_base[c] = (float)opts.film_base_rgb_d[c];
  } else if (opts.film_base_rgb) {
    for (int c = 0; c < 3; ++c) cf.film_base[c] = opts.film_base_rgb[c];
  } else {
    for (int c = 0; c < 3; ++c) cf.film_base[c] = 0.0f;
    for (size_t i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) cf.film_base[c] = std::max(cf.film_base[c], (float)stats.grid[i * 3 + c]);
  }
  cf.grid = stats.grid.data();
  cf.n = (int)n;
  return cf;
}

// The largest factor t in [0, 1] (bisection; t = 0, the identity, always passes) by which channel ch's a and k are
// scaled to keep the shape where ok says.
template <typename Ok>
static float shape_scale_of(Ok ok) {
  if (ok(1.0f)) return 1.0f;
  float lo = 0.0f, hi = 1.0f;
  for (int it = 0; it < 24; ++it) {
    const float mid = 0.5f * (lo + hi);
    if (ok(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

bool reanchor_input_curves(const ProfileBundle& bundle, const DinoV3Engine& engine, int target_idx,
                           const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                           const ParamSearchOptions& opts, const negicc::InputCurves& curves, const float from[3],
                           const float to[3], float max_dev_stops, negicc::InputCurves& out, float scale[2]) {
  out = curves;
  scale[0] = scale[1] = 1.0f;
  if (bundle.has_crosstalk || target_idx < 0 || target_idx >= (int)bundle.targets.size()) {
    std::cerr << "ERROR! Re-anchoring the input curves needs a white-light bundle's bracket" << std::endl;
    return false;
  }
  if (!curves.active()) return true;
  // The frame as the fit saw it: the same statistics' grid and the search's matrix (white light: M_frame).
  FrameStats stats;
  compute_frame_stats(frame, opts.crop_ratio, engine.max_size(), engine.patch_size(), stats);
  const FrameScale fs = frame_scale(bundle, (float)scan_shutter_s, (float)scan_iso, opts);
  const negicc::CurveFrame cf = curve_frame_of(stats, fs.frame_matrix, opts);
  if (cf.n <= 0) return true;
  // Moved by the difference of two fresh anchors rather than replaced by the one at to, so a replay's recorded ref and
  // u_med carry over exactly where nothing moves.
  const negicc::InputCurves a0 = negicc::anchor_curves(curves, cf, from[0], from[1], from[2]);
  const negicc::InputCurves a1 = negicc::anchor_curves(curves, cf, to[0], to[1], to[2]);
  if (a0.ref_r > 0.0f && a1.ref_r > 0.0f && curves.ref_r > 0.0f)
    out.ref_r = curves.ref_r * (a1.ref_r / a0.ref_r);
  out.u_lo_r = curves.u_lo_r + (a1.u_lo_r - a0.u_lo_r);
  const ProfileData& pd = bundle.targets[target_idx].film_prof.data;
  for (int ch = 0; ch < 2; ++ch) {
    if (a0.ref[ch] > 0.0f && a1.ref[ch] > 0.0f) out.ref[ch] = curves.ref[ch] * (a1.ref[ch] / a0.ref[ch]);
    out.u_med[ch] = curves.u_med[ch] + (a1.u_med[ch] - a0.u_med[ch]);
    out.u_lo[ch] = curves.u_lo[ch] + (a1.u_lo[ch] - a0.u_lo[ch]);
    const int table_n = pd.in_trc_size[ch + 1];
    scale[ch] = shape_scale_of([&](float t) {
      return negicc::curve_shape_ok(t * out.a[ch], t * out.k[ch], out.u_med[ch], out.ref[ch], table_n, max_dev_stops,
                                    out.u_hold(ch));
    });
    out.a[ch] *= scale[ch];
    out.k[ch] *= scale[ch];
  }
  out.k_r *= scale[0];
  return true;
}

// ---- Step 4: the network's magenta-green offset on the fitted state ------------------------------------------------
//
// The joint fit balances the scene-neutral anchors; whether the whole picture then reads too green or too magenta is
// a judgement the intent network makes about the content. It barely tracks a re-rendered preview (its field moves
// little with the render), so one inference on the fitted state gives a field the other offsets can be scored
// against: each lattice point's cells are predicted as the pivot's cells plus the change the grid model gives, and
// scored by the a*-only residual S(s) = mean_i ((a_i - mu_a,i) / sigma_a,i)^2. The few best points are rendered and
// scored on their own fields; the best of those, moved one bootstrap standard error of the residual's minimum toward
// magenta (the owner prefers a frame a little magenta to a little green), is the offset applied. Lattice points that
// make the grid greener than Auto (the joint fit's guard, read on the export's render) are not candidates.
constexpr double kOffsetScanMax = 8.0;       // lattice s in [-8, 8] a*,
constexpr double kOffsetScanStep = 0.5;      // in steps of 0.5 a*
constexpr int kOffsetScanCandidates = 3;     // lattice points rendered and scored on their own fields
constexpr double kOffsetScanGuardSlack = 0.1;  // a*: the scan's guard allows this much beyond kGreenGuardTolAb
constexpr int kOffsetBootstrapResamples = 300;
constexpr int kOffsetBootstrapBlock = 4;     // cells: blocks of 4 x 4 grid cells are resampled together
constexpr double kOffsetReadoutSE = 1.0;     // the pick moves this many standard errors toward magenta
// Test only, off: no scan when the joint fit settled for lack of neutral evidence (the neutral head's conf at or below
// 1e-4, the fit's own settle test), e.g. a blank or near-uniform frame, where the field alone would set the trim.
constexpr bool kOffsetScanNeedsEvidence = false;
// The direction solve: probe steps of (d_G, d_B, k_G, k_B), the ridge on the step-scaled columns (a layer the grid
// cannot move, e.g. a blue floor, would otherwise ask for an offset and a contrast change that cancel at many D),
// and the bounds of the density change.
constexpr double kOffsetProbe[4] = {0.01, 0.01, 0.05, 0.05};
constexpr double kOffsetRidge = 1e-2;
constexpr double kOffsetMaxDensity = 0.3, kOffsetMaxContrast = 0.6;

// The fitted tables with t's density change composed in front of G and B (R is the reference): a value's density
// above the film base D becomes D + d + k (D - P), P the frame's median cell density of that channel (piv).
static void offset_tables(const ProfileData& fitted, const OffsetTables& t, float e, float g, float b,
                          std::vector<float> out[3]) {
  const double* th = t.theta;
  const double* piv = t.piv;
  const float gain[3] = {e, e * g, e * b};
  for (int c = 0; c < 3; ++c) {
    const int n = fitted.in_trc_size[c];
    out[c].assign(fitted.in_trc[c], fitted.in_trc[c] + n);
    if (c == 0 || n < 2) continue;
    const double d = th[c - 1], k = th[c + 1];
    if (d == 0.0 && k == 0.0) continue;
    const double v_base = (double)gain[c] * t.diag[c] * t.film_base[c] / 65535.0;
    if (!(v_base > 0)) continue;
    for (int j = 0; j < n; ++j) {
      const double v = (double)j / (n - 1);
      if (v <= 0.0) continue;
      const double D = -std::log10(v / v_base);
      const double dD = d + k * (D - piv[c]);
      const double v2 = std::min(1.0, std::max(0.0, v * std::pow(10.0, -dD)));
      out[c][j] = negicc::apply_trc((float)v2, fitted.in_trc[c], n);
    }
  }
}

// The cells' CIELAB as the network's grid sees them: the 16 x 16 pixel block means of an 8-bit preview at DINOv3's
// input size, sRGB (D65) to Lab in double precision. lab gets 3 values per cell, row-major.
static void net_cell_lab(const std::vector<uint8_t>& u8, int w, int h, std::vector<double>& lab) {
  const int gw = w / 16, gh = h / 16;
  lab.assign((size_t)gw * gh * 3, 0.0);
  for (int gy = 0; gy < gh; ++gy)
    for (int gx = 0; gx < gw; ++gx) {
      double rgb[3] = {0, 0, 0};
      for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x)
          for (int c = 0; c < 3; ++c) rgb[c] += u8[((size_t)(gy * 16 + y) * w + gx * 16 + x) * 3 + c];
      double lin[3];
      for (int c = 0; c < 3; ++c) {
        const double v = rgb[c] / 256.0 / 255.0;
        lin[c] = v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
      }
      const double X = (0.4124564 * lin[0] + 0.3575761 * lin[1] + 0.1804375 * lin[2]) / 0.95047;
      const double Y = 0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2];
      const double Z = (0.0193339 * lin[0] + 0.1191920 * lin[1] + 0.9503041 * lin[2]) / 1.08883;
      auto f = [](double t) { return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0; };
      double* o = &lab[((size_t)gy * gw + gx) * 3];
      o[0] = 116.0 * f(Y) - 16.0;
      o[1] = 500.0 * (f(X) - f(Y));
      o[2] = 200.0 * (f(Y) - f(Z));
    }
}

// The a*-only residual of cells' Lab against a field.
static double offset_residual(const std::vector<double>& lab, const std::vector<float>& mu,
                              const std::vector<float>& sigma) {
  const size_t n = lab.size() / 3;
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double z = (lab[3 * i + 1] - mu[3 * i + 1]) / sigma[3 * i + 1];
    s += z * z;
  }
  return n ? s / (double)n : 0.0;
}

// A fixed PCG64 stream (128-bit LCG, XSL-RR output; the state and increment of the reference implementation's seed
// 0), so the bootstrap and with it the offset are reproducible frame by frame.
class OffsetBootstrapRng {
 public:
  // A uniform integer in [0, n), n >= 1 (Lemire's multiply-shift with rejection on 32-bit draws).
  uint32_t below(uint32_t n) {
    uint64_t m = (uint64_t)next32() * n;
    uint32_t left = (uint32_t)m;
    if (left < n) {
      const uint32_t threshold = (UINT32_MAX - (n - 1)) % n;
      while (left < threshold) {
        m = (uint64_t)next32() * n;
        left = (uint32_t)m;
      }
    }
    return (uint32_t)(m >> 32);
  }

 private:
  static unsigned __int128 u128(uint64_t hi, uint64_t lo) { return ((unsigned __int128)hi << 64) | lo; }
  uint32_t next32() {
    if (has_half_) {
      has_half_ = false;
      return half_;
    }
    state_ = state_ * u128(0x2360ED051FC65DA4ULL, 0x4385DF649FCCF645ULL) + inc_;
    const uint64_t x = (uint64_t)(state_ >> 64) ^ (uint64_t)state_;
    const unsigned rot = (unsigned)(state_ >> 122);
    const uint64_t out = (x >> rot) | (x << ((64 - rot) & 63));
    half_ = (uint32_t)(out >> 32);
    has_half_ = true;
    return (uint32_t)out;
  }
  unsigned __int128 state_ = u128(0x1aa1b5345996452dULL, 0x09585eb7a69561e3ULL);
  unsigned __int128 inc_ = u128(0x418ddadb3af71a82ULL, 0x588133bc447873a9ULL);
  uint32_t half_ = 0;
  bool has_half_ = false;
};

// The standard error of the residual's minimum over s (sorted ascending; the refinement takes s[1] - s[0] as the
// step): pred[k] is the predicted cells' Lab at s[k]. Blocks of kOffsetBootstrapBlock^2 grid cells are resampled with
// replacement; each resample's minimum, when inside the range, is refined by a parabola through it and its two
// neighbours (by at most one step); the standard error is the population standard deviation of the minima.
static double offset_bootstrap_se(const std::vector<std::vector<double>>& pred, const std::vector<double>& s,
                                  const std::vector<float>& mu, const std::vector<float>& sigma, int gw, int gh) {
  const int K = (int)s.size();
  const int bx = (gw + kOffsetBootstrapBlock - 1) / kOffsetBootstrapBlock;
  const int by = (gh + kOffsetBootstrapBlock - 1) / kOffsetBootstrapBlock;
  const int nb = bx * by;
  std::vector<double> blocks((size_t)K * nb, 0.0);
  std::vector<double> cnt(nb, 0.0);
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) {
      const int blk = (y / kOffsetBootstrapBlock) * bx + x / kOffsetBootstrapBlock;
      const size_t i = (size_t)y * gw + x;
      cnt[blk] += 1.0;
      for (int k = 0; k < K; ++k) {
        const double z = (pred[k][3 * i + 1] - mu[3 * i + 1]) / sigma[3 * i + 1];
        blocks[(size_t)k * nb + blk] += z * z;
      }
    }
  OffsetBootstrapRng rng;
  std::vector<int> idx(nb);
  std::vector<double> mins, curve(K);
  mins.reserve(kOffsetBootstrapResamples);
  for (int rep = 0; rep < kOffsetBootstrapResamples; ++rep) {
    for (int j = 0; j < nb; ++j) idx[j] = (int)rng.below((uint32_t)nb);
    double n = 0.0;
    for (int j = 0; j < nb; ++j) n += cnt[idx[j]];
    int best = 0;
    for (int k = 0; k < K; ++k) {
      double v = 0.0;
      for (int j = 0; j < nb; ++j) v += blocks[(size_t)k * nb + idx[j]];
      curve[k] = v / n;
      if (curve[k] < curve[best]) best = k;
    }
    double m = s[best];
    if (best > 0 && best < K - 1) {
      const double y0 = curve[best - 1], y1 = curve[best], y2 = curve[best + 1], den = y0 - 2 * y1 + y2;
      const double off = den > 0 ? 0.5 * (y0 - y2) / den : 0.0;
      m = s[best] + std::clamp(off, -1.0, 1.0) * (s[1] - s[0]);
    }
    mins.push_back(m);
  }
  double mean = 0.0;
  for (double m : mins) mean += m;
  mean /= (double)mins.size();
  double v = 0.0;
  for (double m : mins) v += (m - mean) * (m - mean);
  return std::sqrt(v / (double)mins.size());
}

// Step 4 on the fitted state (e, g, b, x_hi on entry; prof holds the fitted tables). On return prof's G and B tables
// carry the offset, and e and x_hi are those it renders at (E re-solved to target_L over fit_grid within [e_lo, e_hi],
// the roll-off measured again). grid_all is every cell of the network's grid (the fit may have left some out), with
// p_all the neutral head's p of each (may be empty); auto_pd, e_auto, g_auto, b_auto and x_auto render Auto's result;
// q995 gets the measured roll-off's 99.5% point, and tone_features the tone heads' input from the inference of the
// winning candidate (the state closest to the export: it differs by the standard error), for step 5.
template <typename Rolloff>
static void offset_scan(SearchFrame& fr, ParsedFilmProfile& prof, const negicc::CurveFrame& cf, const float* merged,
                        const std::vector<uint16_t>& grid_all, int grid_w, int grid_h, const uint16_t* fit_grid,
                        size_t nf, int encoding, float target_L, float e_lo, float e_hi, Rolloff rolloff_at,
                        const ProfileData& auto_pd, float e_auto, float g_auto, float b_auto, float x_auto,
                        const std::vector<float>& p_all, float& e, float g, float b, float& x_hi, float& q995,
                        std::vector<float>& tone_features, OffsetScanResult& R) {
  const Clock::time_point t0 = Clock::now();
  const int N = (int)nf, NA = (int)(grid_all.size() / 3);
  const ProfileData fitted = prof.data;
  double piv[3] = {0, 0, 0};
  for (int c = 0; c < 3; ++c) {
    std::vector<double> d(N);
    for (int i = 0; i < N; ++i) d[i] = -std::log10(std::max(1e-3, (double)fit_grid[3 * i + c]) / cf.film_base[c]);
    std::nth_element(d.begin(), d.begin() + N / 2, d.end());
    piv[c] = d[N / 2];
  }
  OffsetTables frame_t;  // the frame's part; theta varies
  for (int c = 0; c < 3; ++c) {
    frame_t.piv[c] = piv[c];
    frame_t.diag[c] = merged[4 * c];
    frame_t.film_base[c] = cf.film_base[c];
  }
  const float x0 = x_hi;
  struct Tables {
    std::vector<float> in[3];
    ProfileData pd;
  };
  auto build = [&](const double th[4], float ee, Tables& T) {
    OffsetTables t = frame_t;
    for (int k = 0; k < 4; ++k) t.theta[k] = th[k];
    offset_tables(fitted, t, ee, g, b, T.in);
    T.pd = fitted;
    for (int c = 0; c < 3; ++c) T.pd.in_trc[c] = T.in[c].data();
  };
  auto median_L_at = [&](const double th[4], float ee, float xx) {
    Tables T;
    build(th, ee, T);
    return negicc::grid_median_L(make_pipeline_config(T.pd, AUTO_KNEE, xx, encoding), merged, ee, g, b, fit_grid, N);
  };
  // E with the grid's median L* at target_L: bisection on log E (the median falls as E rises).
  auto solve_e = [&](const double th[4], float xx) {
    double lo = std::log(e_lo), hi = std::log(e_hi);
    for (int it = 0; it < 40; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (median_L_at(th, (float)std::exp(mid), xx) > target_L) lo = mid;
      else hi = mid;
    }
    return (float)std::exp(0.5 * (lo + hi));
  };
  auto lab_all = [&](const ProfileData& pd, float ee, float gg, float bb, float xx, std::vector<float>& lab) {
    const negicc::PipelineConfig cfg = make_pipeline_config(pd, AUTO_KNEE, xx, encoding);
    float m[9];
    negicc::candidate_matrix(merged, ee, gg, bb, m);
    lab.resize(3 * (size_t)NA);
    for (int i = 0; i < NA; ++i) {
      const negicc::Lab l = negicc::linear_srgb_to_lab(
          negicc::to_display(cfg, negicc::convert_pixel_linear(cfg, m, grid_all.data() + 3 * i)));
      lab[3 * i] = l.L;
      lab[3 * i + 1] = l.a;
      lab[3 * i + 2] = l.b;
    }
  };
  // The grid model at theta: E re-solved with the fitted state's roll-off.
  auto lab_th = [&](const double th[4], std::vector<float>& lab) {
    const float ee = solve_e(th, x0);
    Tables T;
    build(th, ee, T);
    lab_all(T.pd, ee, g, b, x0, lab);
  };
  // The export at theta: E solved, the roll-off measured, E solved again and the roll-off measured again.
  auto finish = [&](const double th[4], float& ee, float& xx, float* q995 = nullptr) {
    ee = solve_e(th, x0);
    Tables T;
    build(th, ee, T);
    float q = 0.0f;
    xx = rolloff_at(T.pd, ee, g, b, &q);
    ee = solve_e(th, xx);
    build(th, ee, T);
    xx = rolloff_at(T.pd, ee, g, b, &q);
    if (q995) *q995 = q;
  };

  // The direction on the grid model: mean a*, b* and their slopes over L* (per 50 L*) over the fitted state's cells at
  // L* 5..95; u solves J u = (1, 0, 0, 0) (mean a* +1, the rest held).
  std::vector<float> lab0;
  const double zero[4] = {0, 0, 0, 0};
  lab_th(zero, lab0);
  std::vector<int> use;
  double Lm = 0;
  for (int i = 0; i < NA; ++i)
    if (lab0[3 * i] > 5 && lab0[3 * i] < 95) {
      use.push_back(i);
      Lm += lab0[3 * i];
    }
  if (use.size() < 8) {
    use.clear();
    Lm = 0;
    for (int i = 0; i < NA; ++i) {
      use.push_back(i);
      Lm += lab0[3 * i];
    }
  }
  Lm /= use.size();
  auto moments = [&](const std::vector<float>& lab, double mo[4]) {
    double sa = 0, sb = 0, sLL = 0, sLa = 0, sLb = 0;
    for (int i : use) {
      sa += lab[3 * i + 1];
      sb += lab[3 * i + 2];
    }
    sa /= use.size();
    sb /= use.size();
    for (int i : use) {
      const double z = (lab0[3 * i] - Lm) / 50.0;
      sLL += z * z;
      sLa += z * (lab[3 * i + 1] - sa);
      sLb += z * (lab[3 * i + 2] - sb);
    }
    mo[0] = sa;
    mo[1] = sb;
    mo[2] = sLa / std::max(sLL, 1e-9);
    mo[3] = sLb / std::max(sLL, 1e-9);
  };
  double m0[4];
  moments(lab0, m0);
  double J[4][4];
  // The probes and the lattice points are independent renders of the grid model (each solves its own E): run in
  // parallel, each writing only its own slot, so the result is the same as in order.
#pragma omp parallel for schedule(dynamic)
  for (int j = 0; j < 4; ++j) {
    double th[4] = {0, 0, 0, 0};
    th[j] = kOffsetProbe[j];
    std::vector<float> l;
    lab_th(th, l);
    double mj[4];
    moments(l, mj);
    for (int r = 0; r < 4; ++r) J[r][j] = (mj[r] - m0[r]) / kOffsetProbe[j];
  }
  double u[4];
  {
    const double* h = kOffsetProbe;
    const double y[4] = {1, 0, 0, 0};
    double M[4][5];
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        double v = 0;
        for (int k = 0; k < 4; ++k) v += J[k][r] * h[r] * J[k][c] * h[c];
        M[r][c] = v + (r == c ? kOffsetRidge : 0.0);
      }
      double v = 0;
      for (int k = 0; k < 4; ++k) v += J[k][r] * h[r] * y[k];
      M[r][4] = v;
    }
    for (int c = 0; c < 4; ++c) {
      int pv = c;
      for (int r = c + 1; r < 4; ++r)
        if (std::fabs(M[r][c]) > std::fabs(M[pv][c])) pv = r;
      for (int k = 0; k < 5; ++k) std::swap(M[c][k], M[pv][k]);
      for (int r = 0; r < 4; ++r) {
        if (r == c) continue;
        const double f = M[r][c] / M[c][c];
        for (int k = 0; k < 5; ++k) M[r][k] -= f * M[c][k];
      }
    }
    for (int k = 0; k < 4; ++k) u[k] = M[k][4] / M[k][k] * h[k];
  }
  auto theta_of = [&](double s, double th[4]) {
    for (int k = 0; k < 4; ++k)
      th[k] = std::clamp(s * u[k], k < 2 ? -kOffsetMaxDensity : -kOffsetMaxContrast,
                         k < 2 ? kOffsetMaxDensity : kOffsetMaxContrast);
  };
  auto net_preview = [&](double s, std::vector<uint8_t>& pv, int& w, int& h) {
    double th[4];
    theta_of(s, th);
    float ee, xx;
    finish(th, ee, xx);
    Tables T;
    build(th, ee, T);
    fr.render_net_preview(T.pd, {ee, g, b, xx}, pv, w, h);
  };

  // The pivot: the fitted state's preview, its field and its cells.
  std::vector<uint8_t> pv;
  int pw = 0, ph = 0;
  net_preview(0.0, pv, pw, ph);
  std::vector<float> mu0, sg0, tf0;
  if (pw / 16 != grid_w || ph / 16 != grid_h) {
    R.skipped = "grid";
  } else if (!fr.predict_net(pv, pw, ph, mu0, sg0, tf0)) {
    R.skipped = "infer";
  }
  if (R.skipped) {
    R.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return;
  }
  R.inferences = 1;
  std::vector<double> x_piv;
  net_cell_lab(pv, pw, ph, x_piv);

  // The lattice on the grid model, and the guard on each point: Auto's cells at L* 2..92, the mean a* over all of
  // them and weighted by p^2 not more than kGreenGuardTolAb (+ kOffsetScanGuardSlack) below Auto's.
  std::vector<float> lab_auto;
  lab_all(auto_pd, e_auto, g_auto, b_auto, x_auto, lab_auto);
  std::vector<char> in_band(NA, 0);
  std::vector<double> w2(NA, 0.0);
  double n_in = 0.0, w_in = 0.0, a_auto = 0.0, aw_auto = 0.0;
  for (int i = 0; i < NA; ++i) {
    if (!(lab_auto[3 * i] > 2.0f && lab_auto[3 * i] <= 92.0f)) continue;
    in_band[i] = 1;
    const double p = (size_t)NA == p_all.size() ? std::clamp((double)p_all[i], 0.0, 1.0) : 0.0;
    w2[i] = p * p;
    n_in += 1.0;
    w_in += w2[i];
    a_auto += lab_auto[3 * i + 1];
    aw_auto += w2[i] * lab_auto[3 * i + 1];
  }
  const double tol = (double)negicc::kGreenGuardTolAb + kOffsetScanGuardSlack;
  auto guard_ok = [&](const std::vector<float>& lab) {
    double a = 0.0, aw = 0.0;
    for (int i = 0; i < NA; ++i)
      if (in_band[i]) {
        a += lab[3 * i + 1];
        aw += w2[i] * lab[3 * i + 1];
      }
    if (n_in >= negicc::kGreenGuardMinCells && a / n_in < negicc::green_guard_ref(a_auto / n_in) - tol) return false;
    if (w_in >= negicc::kGreenGuardMinMass && aw / w_in < negicc::green_guard_ref(aw_auto / w_in) - tol) return false;
    return true;
  };
  const int K = (int)std::lround(2.0 * kOffsetScanMax / kOffsetScanStep) + 1;
  const int k_zero = K / 2;
  std::vector<double> s_of(K), score(K);
  std::vector<char> kept(K);
  std::vector<std::vector<double>> pred(K);
#pragma omp parallel for schedule(dynamic)
  for (int k = 0; k < K; ++k) {
    s_of[k] = -kOffsetScanMax + k * kOffsetScanStep;
    double th[4];
    theta_of(s_of[k], th);
    std::vector<float> l;
    lab_th(th, l);
    pred[k].resize(3 * (size_t)NA);
    for (int i = 0; i < 3 * NA; ++i) pred[k][i] = x_piv[i] + ((double)l[i] - (double)lab0[i]);
    score[k] = offset_residual(pred[k], mu0, sg0);
    kept[k] = guard_ok(l) || k == k_zero;
  }
  R.lattice = K;
  R.residual_pivot = (float)score[k_zero];
  for (char c : kept) R.lattice_kept += c;

  // The candidates: the kOffsetScanCandidates kept points of lowest predicted residual (ties in lattice order).
  std::vector<int> order;
  for (int k = 0; k < K; ++k)
    if (kept[k]) order.push_back(k);
  std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return score[i] < score[j]; });
  if ((int)order.size() > kOffsetScanCandidates) order.resize(kOffsetScanCandidates);

  // The standard error, over the kept points in lattice order.
  std::vector<std::vector<double>> pk;
  std::vector<double> sk;
  for (int k = 0; k < K; ++k)
    if (kept[k]) {
      pk.push_back(pred[k]);
      sk.push_back(s_of[k]);
    }
  const double se = sk.size() >= 3 ? offset_bootstrap_se(pk, sk, mu0, sg0, grid_w, grid_h) : 0.0;

  // Each candidate scored on its own field (the pivot's own, for s = 0); the lowest wins, the first on a tie.
  double best_s = 0.0, best_v = 0.0;
  bool have = false;
  tone_features = tf0;
  for (int k : order) {
    double v;
    std::vector<float> tf;
    if (k == k_zero) {
      v = offset_residual(x_piv, mu0, sg0);
      tf = tf0;
    } else {
      std::vector<uint8_t> cpv;
      int cw = 0, ch = 0;
      net_preview(s_of[k], cpv, cw, ch);
      std::vector<float> mu, sg;
      ++R.inferences;
      if (!fr.predict_net(cpv, cw, ch, mu, sg, tf)) continue;
      std::vector<double> xc;
      net_cell_lab(cpv, cw, ch, xc);
      v = offset_residual(xc, mu, sg);
    }
    R.candidates.push_back({(float)s_of[k], (float)score[k], (float)v});
    if (!have || v < best_v) {
      have = true;
      best_s = s_of[k];
      best_v = v;
      tone_features = std::move(tf);
    }
  }
  const double s_final = best_s + kOffsetReadoutSE * se;
  R.s_min = best_s;
  R.se = se;
  R.s = s_final;

  // The export: the offset composed into the fitted tables, at its E and roll-off.
  double th[4];
  theta_of(s_final, th);
  float ee, xx;
  finish(th, ee, xx, &q995);
  Tables T;
  build(th, ee, T);
  for (int c = 0; c < 3; ++c) prof.in_trc[c] = T.in[c];
  prof.rebind_pointers();
  R.tables = frame_t;
  for (int k = 0; k < 4; ++k) R.tables.theta[k] = th[k];
  e = ee;
  x_hi = xx;
  R.ran = true;
  R.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void compose_offset_tables(ParsedFilmProfile& prof, const OffsetTables& t, float e, float g, float b) {
  std::vector<float> in[3];
  offset_tables(prof.data, t, e, g, b, in);
  for (int c = 0; c < 3; ++c) prof.in_trc[c] = in[c];
  prof.rebind_pointers();
}

// py-ref: negicc_station/src/conversion_session.py (ConversionSession.set_fit_curves) and
// negicc_station/src/per_target_conversion.py (PerTargetConverter.fit_bracket)
bool fit_input_curves_after_search(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                                   double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                                   const ParamSearchResult& res, const negicc::CurveFitOptions& fit_opts,
                                   int encoding, bool offset_scan_on, const ToneHead* tone_head,
                                   CurveRefitResult& out) {
  // TODO: the deferred tri-LED extension of the fit; see the TODO at the --fit_curves gate in np_pipeline.cc.
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
  out.options = fit_opts;

  // The intent field the fit scores against: the search's, predicted from the preview at intent_params. The highlight
  // pullback (and the clip fallback) move E, or the bracket's parameters, after that preview, so there the field is
  // predicted again from a preview at the search's final parameters on the bracket's own profile, so the fit does not
  // run at one E against a field predicted at another. One DINOv3 inference, only then.
  std::vector<float> mu = res.intent_mu, sigma = res.intent_sigma;
  std::vector<float> neutral_p = res.intent_neutral_p;  // the neutral head's p of the same preview, if it ran
  std::vector<float> tone_features = res.intent_tone_features;  // the tone heads' input from it, if given
  {
    const ConversionParams& ip = res.intent_params;  // the search's own units, which are this frame's on white light
    auto moved = [](float a, float b) { return std::fabs(a - b) > 1e-6f * std::max(1.0f, std::fabs(b)); };
    if (moved(ip.e, res.params.e) || moved(ip.g, res.params.g) || moved(ip.b, res.params.b) ||
        moved(ip.shoulder, res.params.shoulder)) {
      std::vector<uint8_t> preview;
      fr.render_preview(idx, res.params, preview);
      if (!fr.predict(preview, mu, sigma)) return false;
      neutral_p = fr.last_neutral_p();
      tone_features = fr.last_tone_features();
      out.intent_repredicted = true;
      out.guard_p_source = "refit";
    }
  }

  // The curves' anchor on this frame: the film base read from the scan, else the brightest grid cell per channel (of
  // the whole grid, before any cell is left out below).
  negicc::CurveFrame cf = curve_frame_of(stats, merged, opts);
  // On a fallback crop (the whole image, not a detected frame) the grid's cells at the film base's level (rebate, gaps,
  // bare light: G and B within kBaseCellStops of a rebate base; a statistical base is the 99.5th percentile of the film
  // itself, so its level is the thinnest picture highlights) or holder-dark (G more than kHolderCellStops below the
  // base) are not picture: they are left out of the fit, the re-meter and the losses, with their intent cells. Both
  // limits are assumptions: that the base varies by less than a quarter stop, and that no picture is 9 stops below it.
  constexpr float kBaseCellStops = 0.25f, kHolderCellStops = 9.0f;
  const std::vector<float> neutral_p_all = neutral_p;  // every cell's, for step 4 (which scores the whole grid)
  std::vector<uint16_t> kept_grid;
  std::vector<size_t> kept_idx;
  const uint16_t* fit_grid = stats.grid.data();
  size_t nf = n;
  out.cells_total = (int)n;
  auto filter_intent = [&](std::vector<float>& m, std::vector<float>& s) {
    if (kept_idx.empty()) return;
    std::vector<float> km, ks;
    km.reserve(kept_idx.size() * 3);
    ks.reserve(kept_idx.size() * 3);
    for (size_t i : kept_idx) {
      km.insert(km.end(), m.begin() + i * 3, m.begin() + i * 3 + 3);
      ks.insert(ks.end(), s.begin() + i * 3, s.begin() + i * 3 + 3);
    }
    m.swap(km);
    s.swap(ks);
  };
  if (opts.fallback_crop && cf.film_base[1] > 0.0f && cf.film_base[2] > 0.0f) {
    std::vector<size_t> try_idx;
    const float base_lim[2] = {cf.film_base[1] * std::exp2(-kBaseCellStops), cf.film_base[2] * std::exp2(-kBaseCellStops)};
    const float holder_lim = cf.film_base[1] * std::exp2(-kHolderCellStops);
    for (size_t i = 0; i < n; ++i) {
      const float g = stats.grid[i * 3 + 1], b = stats.grid[i * 3 + 2];
      if ((opts.film_base_is_rebate && g >= base_lim[0] && b >= base_lim[1]) || g <= holder_lim) continue;
      kept_grid.insert(kept_grid.end(), stats.grid.begin() + i * 3, stats.grid.begin() + i * 3 + 3);
      try_idx.push_back(i);
    }
    // Only with enough picture left (at least 16 cells and a quarter of the grid, enough to constrain the fit's six
    // numbers); else the whole grid.
    if (kept_grid.size() / 3 >= std::max<size_t>(16, n / 4) && kept_grid.size() / 3 < n) {
      out.cells_excluded = (int)(n - kept_grid.size() / 3);
      nf = kept_grid.size() / 3;
      fit_grid = kept_grid.data();
      kept_idx.swap(try_idx);
      filter_intent(mu, sigma);
      if (neutral_p.size() == n) {
        std::vector<float> kp;
        kp.reserve(kept_idx.size());
        for (size_t i : kept_idx) kp.push_back(neutral_p[i]);
        neutral_p.swap(kp);
      }
      cf.grid = fit_grid;
      cf.n = (int)nf;
      if (opts.verbose) {
        std::cout << "Curves: fallback crop, " << out.cells_excluded << " of " << n
                  << " grid cells at the film base's level or holder-dark left out\n";
      }
    }
  }
  if (neutral_p.size() == nf) cf.neutral_p = neutral_p.data();
  ParsedFilmProfile& prof = bundle.targets[idx].film_prof;
  // The export's render of the grid: its encoding and the roll-off it measures (measure_rolloff_x_hi, as neg_process
  // measures every conversion's), which only the sRGB encoding applies, so the median L* the re-meter matches is the
  // one the export renders.
  out.encoding = encoding;
  auto rolloff_at = [&](const negicc::ProfileData& p, float e, float g, float b, float* q995) {
    if (encoding != ENCODE_SRGB) return 1.0f;
    return measure_rolloff_x_hi(bundle, p, frame, scan_shutter_s, scan_iso, e, g, b, opts.crop_ratio, q995,
                                opts.film_base_rgb_d);
  };
  float q995_auto = 0.0f;
  out.x_hi_auto = rolloff_at(prof.data, e_int, res.params.g, res.params.b, &q995_auto);
  negicc::PipelineConfig cfg_auto = make_pipeline_config(prof.data, AUTO_KNEE, out.x_hi_auto, encoding);
  // Auto's grid median L* on the bracket's own profile, as the export renders Auto's result: the lightness the fit
  // keeps. On low-key frames (Auto's target_L < 25 and median green density < 0.80 stops above film base), cap
  // fit_target_L by the physical toe ceiling so an unexposed night sky near film base is not lifted to kIntentMinL,
  // while never darkening a bright/flat frame (target_L >= 25).
  const float target_L =
      negicc::grid_median_L(cfg_auto, merged, e_int, res.params.g, res.params.b, fit_grid, (int)nf);
  const float d_med = -negicc::anchor_curves(negicc::InputCurves{}, cf, e_int, res.params.g, res.params.b).u_med[0];
  const float toe_max_L =
      (target_L < 25.0f && d_med > 0.0f && d_med < 0.80f) ? std::max(10.0f, 22.0f * d_med) : kIntentMaxL;
  const float fit_target_L = std::min(std::clamp(target_L, kIntentMinL, kIntentMaxL), toe_max_L);
  // Auto's loss parts on the same render; both before step 3 replaces the profile's tables (cfg points at them).
  out.parts_auto = negicc::grid_intent_loss_parts(cfg_auto, merged, e_int, res.params.g, res.params.b,
                                                  fit_grid, (int)nf, mu.data(), sigma.data());
  const int table_n[2] = {prof.data.in_trc_size[1], prof.data.in_trc_size[2]};

  // The bounds of E for the fit and the re-meter: the bracket's calibrated range (its enclosure, which the search
  // measured) widened to hold Auto's E with a margin that scales with the gain bounds (0.1 + 0.5 * max(s_g, s_b)
  // stops), so a gain change has E room to hold Auto's median L*; without a range, 2 stops either side of Auto's E.
  const float s_gain = std::max(fit_opts.g_bound_stops > 0.0f ? fit_opts.g_bound_stops : 0.0f,
                                fit_opts.b_bound_stops > 0.0f ? fit_opts.b_bound_stops : 0.0f);
  const float margin = std::exp2(0.1f + 0.5f * s_gain);
  if (res.e_range_lo > 0.0f && res.e_range_hi >= res.e_range_lo) {
    out.e_lo = std::min(res.e_range_lo, e_int / margin);
    out.e_hi = std::max(res.e_range_hi, e_int * margin);
  } else {
    out.e_lo = e_int * 0.25f;
    out.e_hi = e_int * 4.0f;
  }

  float e_start = e_int;
  out.intent_lightness_failed = (fit_target_L != target_L);
  if (out.intent_lightness_failed) {
    out.fallback_target_L = fit_target_L;
    out.e_lo = std::min(out.e_lo, e_int * 0.25f);
    out.e_hi = std::max(out.e_hi, e_int * 4.0f);
    e_start = negicc::remeter_e_median_in(cfg_auto, cf, negicc::InputCurves{}, e_int, out.e_lo, out.e_hi, res.params.g,
                                          res.params.b, fit_target_L, &out.fallback_clipped);
    out.fallback_e_moved = std::fabs(e_start - e_int) > 1e-6f * std::max(1.0f, std::fabs(e_int));
    out.x_hi_auto = rolloff_at(prof.data, e_start, res.params.g, res.params.b, &q995_auto);
    cfg_auto = make_pipeline_config(prof.data, AUTO_KNEE, out.x_hi_auto, encoding);
    out.e_lo = std::min(out.e_lo, e_start / margin);
    out.e_hi = std::max(out.e_hi, e_start * margin);
  }
  negicc::JointCurveModel m_neu(cfg_auto, cf, mu.data(), sigma.data());
  m_neu.set_scene_neutral(e_start, res.params.g, res.params.b);
  out.neutral_auto = m_neu.neutral_stats_on(cfg_auto, e_start, res.params.g, res.params.b);
  out.neutral_anchors = m_neu.has_anchors();  // as the fit's own selection: the same frame, E, g and b
  out.neutral_decisions = m_neu.decisions();

  negicc::CombinedProfile comb;
  float x = out.x_hi_auto, q995 = q995_auto, e_new = e_start;
  constexpr int kMaxPasses = 6;
  bool clipped = false;
  out.remeter_passes = 0;
  // Auto's own result, no curves (the fallback of the steps below).
  auto fall_back = [&](const char* why) {
    out.fit.curves.k_r = 0.0f;
    for (int ch = 0; ch < 2; ++ch) {
      out.fit.curves.a[ch] = out.fit.curves.k[ch] = 0.0f;
      out.shape_scale_ch[ch] = 0.0f;
    }
    out.fit.g = res.params.g;
    out.fit.b = res.params.b;
    e_new = e_start;
    x = out.x_hi_auto;
    q995 = q995_auto;
    out.fallback = why;
  };
  auto remeter = [&](float e_centre) {
    out.remeter_converged = false;
    for (int pass = 1; pass <= kMaxPasses; ++pass) {
      ++out.remeter_passes;
      e_new = negicc::remeter_e_median_in(make_pipeline_config(prof.data, AUTO_KNEE, x, encoding), cf, out.fit.curves,
                                          e_centre, out.e_lo, out.e_hi, out.fit.g, out.fit.b, fit_target_L, &clipped);
      comb.build(prof.data, negicc::anchor_curves(out.fit.curves, cf, e_new, out.fit.g, out.fit.b));
      float q = 0.0f;
      const float x_new = rolloff_at(comb.data, e_new, out.fit.g, out.fit.b, &q);
      q995 = q;
      if (x_new == x) {
        out.remeter_converged = true;
        break;
      }
      x = x_new;
    }
  };
  auto shape_scales = [&](const negicc::InputCurves& shape, float e, float t[2]) {
    const negicc::InputCurves c = negicc::anchor_curves(shape, cf, e, out.fit.g, out.fit.b);
    bool all = true;
    for (int ch = 0; ch < 2; ++ch) {
      t[ch] = shape_scale_of([&](float s) {
        return negicc::curve_shape_ok(s * c.a[ch], s * c.k[ch], c.u_med[ch], c.ref[ch], table_n[ch],
                                      fit_opts.max_dev_stops, c.u_hold(ch));
      });
      all = all && t[ch] == 1.0f;
    }
    return all;
  };

  // 1. 4x4 scene-neutral color & crossover solver for (g, b, a_G, a_B) at fit_target_L, then 2. re-meter and shape check.
  constexpr int kMaxFitPasses = 3;
  float x_fit = out.x_hi_auto;
  const float e_lo_fit = std::min(e_start, out.e_lo * 1.001f);
  const float e_hi_fit = std::max(e_start, out.e_hi / 1.001f);
  out.fit_passes = 0;
  for (int pass = 0; pass < kMaxFitPasses; ++pass) {
    ++out.fit_passes;
    out.x_hi_fit = x_fit;
    out.fallback = nullptr;
    const negicc::PipelineConfig cfg = make_pipeline_config(prof.data, AUTO_KNEE, x_fit, encoding);
    const negicc::JointCurveFit jf =
        negicc::fit_input_curves_joint(cfg, cf, e_start, res.params.g, res.params.b, mu.data(), sigma.data(),
                                       fit_target_L, table_n, fit_opts, e_lo_fit, e_hi_fit);
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
    out.fit_start = jf.start;
    out.guard = jf.guard;
    if (opts.verbose) {
      const negicc::InputCurves& c = out.fit.curves;
      std::cout << "Curves: G a " << fixed(c.a[0], 4) << " k " << fixed(c.k[0], 4) << ", B a " << fixed(c.a[1], 4)
                << " k " << fixed(c.k[1], 4) << "; E " << fixed(jf.e, 4) << " g " << fixed(out.fit.g, 4) << " b "
                << fixed(out.fit.b, 4) << ", loss " << fixed((float)out.fit.loss_before, 3) << " -> "
                << fixed((float)out.fit.loss_after, 3) << " (" << jf.iterations << " steps, " << jf.evaluations
                << " evaluations, " << jf.renders << " renders for E, roll-off " << fixed(x_fit, 3) << ")\n";
    }
    if (!jf.feasible) {
      fall_back("clip");
      break;
    }
    {
      negicc::CombinedProfile c0;
      c0.build(prof.data, out.fit.curves);
      negicc::PipelineConfig c0cfg = cfg;
      c0cfg.profile = c0.data;
      out.parts_fit = negicc::grid_intent_loss_parts(c0cfg, merged, jf.e, out.fit.g, out.fit.b, fit_grid,
                                                     (int)nf, mu.data(), sigma.data());
    }

    x = x_fit;
    remeter(jf.e);
    out.shape_scale_ch[0] = out.shape_scale_ch[1] = 1.0f;
    bool shape_ok = false;
    for (int round = 0; round < 4 && !clipped; ++round) {
      float t[2];
      if (shape_scales(out.fit.curves, e_new, t)) {
        shape_ok = true;
        break;
      }
      if (round == 3) break;
      for (int ch = 0; ch < 2; ++ch) {
        out.fit.curves.a[ch] *= t[ch];
        out.fit.curves.k[ch] *= t[ch];
        out.shape_scale_ch[ch] *= t[ch];
      }
      remeter(e_new);
    }
    if (clipped || !shape_ok) {
      fall_back(clipped ? "clip" : "shape");
      break;
    }
    if (x == x_fit) break;
    x_fit = x;
  }
  out.shape_scale = std::min(out.shape_scale_ch[0], out.shape_scale_ch[1]);

  // 3. The curves anchored at the parameters the export renders with (the re-metered E, the fitted g and b), and the
  //    combined profile built from them in place: every later use of the bracket renders through it.
  out.fit.curves = negicc::anchor_curves(out.fit.curves, cf, e_new, out.fit.g, out.fit.b);
  ParsedFilmProfile auto_prof = prof;  // the bracket's own tables, which render Auto's result (step 4's guard)
  auto_prof.rebind_pointers();
  apply_input_curves(prof, out.fit.curves);
  out.x_hi = x;  // measured on this profile at these parameters (step 2), or Auto's on the fallback
  out.x_hi_q995 = q995;
  out.rolloff_measured = encoding == ENCODE_SRGB;

  // 4. The network's magenta-green offset, composed into the same tables (E and the roll-off measured again).
  std::vector<float> scan_features;
  out.tone.features = "search";
  if (offset_scan_on && kOffsetScanNeedsEvidence && !(out.neutral_auto.conf > 1e-4f)) {
    out.offset.skipped = "evidence";
    if (opts.verbose) std::cout << "Offset: not run (evidence)\n";
  } else if (offset_scan_on) {
    const float e_lo_s = std::min(e_new, out.e_lo * 1.001f), e_hi_s = std::max(e_new, out.e_hi / 1.001f);
    offset_scan(fr, prof, cf, merged, stats.grid, stats.grid_w, stats.grid_h, fit_grid, nf, encoding, fit_target_L,
                e_lo_s, e_hi_s, rolloff_at, auto_prof.data, e_start, res.params.g, res.params.b, out.x_hi_auto,
                neutral_p_all, e_new, out.fit.g, out.fit.b, out.x_hi, out.x_hi_q995, scan_features, out.offset);
    if (out.offset.ran && scan_features.size() == (size_t)ToneHead::kInDim) {
      tone_features.swap(scan_features);
      out.tone.features = "offset";
    }
    if (opts.verbose) {
      if (out.offset.ran) {
        std::cout << "Offset: a* " << fixed(out.offset.s, 3, true) << " (minimum " << fixed(out.offset.s_min, 2, true)
                  << " + SE " << fixed(out.offset.se, 3) << "), " << out.offset.lattice_kept << " of "
                  << out.offset.lattice << " lattice points kept, " << out.offset.inferences << " inferences, E "
                  << fixed(e_new, 4) << ", roll-off " << fixed(out.x_hi, 3) << "\n";
      } else {
        std::cout << "Offset: not run (" << out.offset.skipped << ")\n";
      }
    }
  } else {
    out.offset.skipped = "off";
  }
  const negicc::PipelineConfig cfg_comb = make_pipeline_config(prof.data, AUTO_KNEE, out.x_hi, encoding);
  out.params = {e_new, out.fit.g, out.fit.b, out.x_hi};
  out.target_median_L = fit_target_L;
  out.median_L = negicc::grid_median_L(cfg_comb, merged, e_new, out.fit.g, out.fit.b, fit_grid, (int)nf);
  out.parts_export = negicc::grid_intent_loss_parts(cfg_comb, merged, e_new, out.fit.g, out.fit.b, fit_grid,
                                                    (int)nf, mu.data(), sigma.data());
  out.neutral_export = m_neu.neutral_stats_on(cfg_comb, e_new, out.fit.g, out.fit.b);
  out.loss_export = out.parts_export.total();

  // 5. The tone stage's plan: the heads' predictions (no inference of its own: from the offset's winning candidate, the
  //    state closest to the export, else from the search's intent preview), the descriptors of the final render's
  //    preview, the curve between them and the chroma factor that keeps the mean chroma.
  ToneStagePlan& tp = out.tone;
  // The heads read the image the stage will be applied to: the offset's winning candidate when the scan ran (it differs
  // from the export by the standard error only); otherwise the final state's preview, rendered below for the
  // descriptors anyway, costs one inference (the search's intent preview has the pre-fit colour).
  std::vector<uint8_t> pv;
  int pw = 0, ph = 0;
  if (tone_head && encoding == ENCODE_SRGB) {
    fr.render_net_preview(prof.data, out.params, pv, pw, ph);
    if (std::strcmp(tp.features, "offset") != 0) {
      std::vector<float> mu_f, sg_f, tf;
      if (fr.predict_net(pv, pw, ph, mu_f, sg_f, tf) && tf.size() == (size_t)ToneHead::kInDim) {
        tone_features.swap(tf);
        tp.features = "final";
      }
    }
  }
  if (!tone_head) {
    tp.skipped = "off";
  } else if (encoding != ENCODE_SRGB) {
    tp.skipped = "encoding";
  } else if (tone_features.size() != (size_t)ToneHead::kInDim ||
             !tone_head->run(tone_features.data(), tp.pred_mean, tp.pred_sd)) {
    tp.skipped = "features";
  } else {
    const size_t npx = (size_t)pw * ph;
    tp.render = tone_descriptors(pv.data(), npx);
    tp.curve = tone_plan(tp.render, tp.pred_mean[ToneHead::kL2], tp.pred_mean[ToneHead::kL98],
                         tp.pred_mean[ToneHead::kR9010], pv.data(), npx, 1.0, &tp.targets);
    tp.chroma_after = tone_mean_chroma_after(pv.data(), npx, tp.curve);
    tp.preview.swap(pv);
    tp.planned = true;
    if (opts.verbose) {
      std::cout << "Tone: L2 " << fixed((float)tp.render.L2, 1) << " -> " << fixed((float)tp.targets.L2, 1) << ", L98 "
                << fixed((float)tp.render.L98, 1) << " -> " << fixed((float)tp.targets.L98, 1) << ", L90-L10 "
                << fixed((float)tp.render.range(), 1) << " -> " << fixed((float)tp.targets.range, 1) << "; curve black "
                << fixed((float)tp.curve.black, 2) << " white " << fixed((float)tp.curve.white, 2) << " gamma "
                << fixed((float)tp.curve.gamma, 3) << " contrast " << fixed((float)tp.curve.contrast, 3) << " chroma x"
                << fixed((float)tp.curve.chroma, 4) << "\n";
    }
  }
  out.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_fit0).count();
  if (opts.verbose) {
    std::cout << "Curves: E re-metered " << fixed(e_int, 4) << " -> " << fixed(e_new, 4) << " (grid median L* "
              << fixed(out.median_L, 2) << ", Auto's " << fixed(target_L, 2) << "); g " << fixed(out.params.g, 4)
              << " b " << fixed(out.params.b, 4)
              << (!out.fallback ? "" : "; no curves, Auto's result (")
              << (out.fallback ? out.fallback : "")
              << (out.fallback ? ")" : "") << "\n";
  }
  return true;
}
