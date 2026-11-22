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
#include "film_base_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "dinov3_auto_solver.h"
#include "dinov3_preprocess.h"
#include "profile_bundle.h"
#include "nlohmann/json.hpp"

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

// The range of E that keeps the scene inside a bracket's calibrated range. The bracket's patches, all of them, span
// green values tgt_min_g .. tgt_max_g in its own capture; E scales the scene's green 0.5 % and 99.5 % points (g_lo,
// g_hi, at the reference exposure) onto them:
//
//   e_min = g_min / g_lo * rho_b,   e_max = g_max / g_hi * rho_b,   rho_b = reference exposure / the bracket's exposure
//
// With e_min <= e_max (feasible) every E in between keeps the whole scene calibrated. The search may go beyond the
// calibrated range by the usable tolerance, 1.5 stops on the highlight side and 1.2 on the shadow side, on every
// bracket alike: u_min = e_min / 2^1.5 .. u_max = e_max * 2^1.2. The same tolerances decide whether a bracket is
// admissible at all; their values are kept from negicc-station, which records no reason for them.
struct Enclosure {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  bool feasible = false;    // e_min <= e_max
  bool admissible = true;   // the scene's span fits the bracket's span plus the 2.7 stops of tolerance
  float e_min = 0, e_max = 0;
  float search_lo = 0, search_hi = 0;  // the range E is searched in
  float e_center = 0;                  // sqrt(e_min * e_max)
  // The usable range inverted (u_min above u_max): no E keeps both ends of the scene within the highlight and shadow
  // tolerance, so the search window is the set of compromises between them; see pull_centre().
  bool usable_inverted = false;
  float slack_ev = 0;                  // log2(e_max / e_min): spare stops, negative when not feasible
  float fill_ratio = 0;                // (g_hi / g_lo) / (g_max / g_min): the scene's contrast ratio over the bracket's
};

// The search window is the usable range on every bracket.
bool bracket_enclosure(const BundleTarget& tgt, int target_idx, float g_lo, float g_hi, Enclosure& r);

// The centre the search's exposure penalty pulls E toward (compute_penalties: 5 log2(E / centre)^2), or 0 for none. For
// a bracket whose usable range is inverted (the scene overflows it by more than the two tolerances together) there is
// no pull. Its window, ordered, runs from E = e_max x 2^1.2, where the shadows sit exactly at their tolerance and the
// highlights carry all the overflow, to E = e_min / 2^1.5, where the reverse holds; outside it one end is sacrificed
// more than needed while the other is already inside. Where to stand among those compromises is the question the intent
// field answers, so inside the window the field decides E, with the white-clip penalty kept and the window's two ends
// as its bounds. A pull toward e_center would answer it with "split the overflow equally in log E": on a night scene of
// 8.1 EV that puts E in the middle of the scene's log range, a picture with a median of L* 2.
float pull_centre(const Enclosure& c) { return c.usable_inverted ? 0.0f : c.e_center; }

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
  // Every bracket, feasible or not, is searched over its usable range: no window that differs between brackets, so
  // no bracket wins or loses because its window let it reach a lightness another's did not (DESIGN.md, "The
  // Auto search"). The order is kept for an inadmissible bracket, whose range can invert.
  r.usable_inverted = u_min > u_max;
  r.search_lo = std::min(u_min, u_max);
  r.search_hi = std::max(u_min, u_max);
  r.fill_ratio = (g_hi / g_lo) / (tgt_max / tgt_min);
  return true;
}

// Brackets whose grey wedge spans at least 0.75 of the largest tonal capacity (at least three kept), when the
// bundle has five or more: a bracket that calibrates a much shorter range than the best ones cannot hold a whole
// scene, so it is not worth screening.
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
class GreyPoint {
 public:
  // steady: for the knobs' solves, a reference grey that the solve misses is sought again (reference_grey). The
  // bootstrap's check of each bracket goes without it, so the brackets Auto meters do not depend on it.
  explicit GreyPoint(const ProfileData& prof, bool steady = false) : prof_(prof), steady_(steady) {}

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
  //
  // With steady_ (the knobs), a solve that misses runs again from the same start with newton()'s step control, and
  // if that misses too, from the grid of the same inputs taken per channel (each channel at any of the 27 levels
  // k * dir[c]), from the point nearest (50, 0, 0) in dE76, with the step control. The start on the line can render
  // with an sRGB channel clipped (a bracket whose profile carries fitted input curves has its neutral off the Portra
  // direction), and there the render does not move with that channel: no step computed from it finds the way out.
  // A solve that reaches the grey the first time is the one the bootstrap and every earlier build ran.
  bool reference_grey(float log_t0[3] = nullptr, float* miss = nullptr) const {
    const float dir[3] = {1.4f, 1.0f, 0.88f};
    constexpr int kLevels = 27;
    auto level = [](int i) { return std::pow(2.0f, -10.0f + 0.5f * i); };
    float best_k = 0.0f, best_d = std::numeric_limits<float>::infinity();
    for (int i = 0; i < kLevels; ++i) {
      const float k = level(i);
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
    float err = newton(target, log_t, 20);
    if (steady_ && !(err <= 0.25f)) {
      for (int c = 0; c < 3; ++c) log_t[c] = std::log(best_k * dir[c]);
      err = newton(target, log_t, 20, true);
    }
    if (steady_ && !(err <= 0.25f)) {
      float best_e = std::numeric_limits<float>::infinity();
      float lab_t[1][3], lab_out[1][3];
      for (int i = 0; i < kLevels; ++i)
        for (int j = 0; j < kLevels; ++j)
          for (int k = 0; k < kLevels; ++k) {
            lab_t[0][0] = level(i) * dir[0];
            lab_t[0][1] = level(j) * dir[1];
            lab_t[0][2] = level(k) * dir[2];
            lab(lab_t, 1, lab_out);
            const float dL = lab_out[0][0] - target[0], da = lab_out[0][1], db = lab_out[0][2];
            const float e = std::sqrt(dL * dL + da * da + db * db);
            if (e < best_e) {
              best_e = e;
              for (int c = 0; c < 3; ++c) log_t[c] = std::log(lab_t[0][c]);
            }
          }
      err = newton(target, log_t, 20, true);
    }
    const bool ok = err <= 0.25f;
    if (miss) *miss = err;
    if (log_t0) {
      for (int c = 0; c < 3; ++c) log_t0[c] = log_t[c];
    }
    return ok;
  }

  // Solves the gains (E, g, b) that move the grey that base renders as neutral L* 50 (at log_t0) to
  // target = (L*, a*, b*). Returns the residual dE76 of the Newton solve.
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
  //
  // Clamping each channel by itself changes the step's direction. Where the render is clipped (an sRGB channel at 0
  // does not move with its input) the Jacobian is nearly singular and the step is thousands of stops long in every
  // channel: clamped channel by channel, it becomes an equal move of all three channels, an exposure change, which
  // the next such step can undo, so the solve alternates between two points without end. With control (the knobs'
  // second try, reference_grey), the first step that lands no closer to target than the point it left is taken back,
  // and from that point on each step is shortened as a whole, so that no channel moves by more than MAX_LOG_STEP,
  // and keeps the Newton direction.
  float newton(const float target[3], float log_t[3], int iters, bool control = false) const {
    constexpr float FD_STEP = 0.05f, MAX_LOG_STEP = 0.25f, TOLERANCE = 0.02f;
    float prev[3] = {log_t[0], log_t[1], log_t[2]};
    float prev_miss = std::numeric_limits<float>::infinity();
    bool whole = false;  // control: steps shortened as a whole from here on
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
      const float miss = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
      if (std::max({std::fabs(r[0]), std::fabs(r[1]), std::fabs(r[2])}) < TOLERANCE) return miss;
      if (control && !whole && !(miss < prev_miss)) {
        whole = true;
        for (int c = 0; c < 3; ++c) log_t[c] = prev[c];
        continue;
      }
      float J[3][3];
      for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row) J[row][c] = (l[1 + 2 * c][row] - l[2 + 2 * c][row]) / (2.0f * FD_STEP);
      float step[3];
      lstsq3(J, r, 1e-6f, step);
      for (int c = 0; c < 3; ++c) prev[c] = log_t[c];
      prev_miss = miss;
      if (whole) {
        const float longest = std::max({std::fabs(step[0]), std::fabs(step[1]), std::fabs(step[2])});
        const float shorten = longest > MAX_LOG_STEP ? MAX_LOG_STEP / longest : 1.0f;
        for (int c = 0; c < 3; ++c) log_t[c] += step[c] * shorten;
      } else {
        for (int c = 0; c < 3; ++c) log_t[c] += std::clamp(step[c], -MAX_LOG_STEP, MAX_LOG_STEP);
      }
    }
    float t[1][3], l[1][3];
    for (int c = 0; c < 3; ++c) t[0][c] = std::exp(log_t[c]);
    lab(t, 1, l);
    const float r[3] = {target[0] - l[0][0], target[1] - l[0][1], target[2] - l[0][2]};
    return std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  }

  const ProfileData& prof_;
  const bool steady_;
};

// The E (see ConversionParams) at which the scene vector p, with gains g and b, renders at L* 50 through the
// bracket: the metered exposure. A bisection on log2 E over [-6, 6] that stops at |L* - 50| <= 0.1. Its 25
// halvings, the station's count, resolve E to 12 / 2^25 stop, far below that tolerance, so they only cap the loop.
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
// The search uses this grid only for the optimiser's cells, corrected per bracket by the trained cells of its last
// preview (update_correction); the station's copy has no such correction.
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
    // An axis that enlarges (scale < 1): per axis, an area average where it shrinks and a clamped bilinear pair where
    // it enlarges (rgb_view.h resample_taps).
    const auto ty = negicc::resample_taps<float>(crop.h, dino_h, sy);
    const auto tx = negicc::resample_taps<float>(crop.w, dino_w, sx);
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      for (int dx = 0; dx < dino_w; ++dx) {
        float acc[3] = {0, 0, 0}, wsum = 0.0f;
        for (const auto& [yy, wy] : ty[(size_t)dy]) {
          for (const auto& [xx, wx] : tx[(size_t)dx]) {
            uint16_t px[3];
            frame.sample(crop.x0 + xx, crop.y0 + yy, px);
            const float w = wx * wy;
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

// Highlight clipping of one render, in percent of the pixels measured (float, as the station's kernel).
struct ClipStats {
  float exc[3];      // at or above the bracket's ceiling, per channel
  float white_clip;  // with any channel white-clipped
  float median_L;    // the median L*
};

// The highlight clipping of img's crop rendered with each conversion matrix (9 floats per candidate in matrices).
std::vector<ClipStats> measure_highlight_clip(const negicc::RgbView& img, const InnerCrop& crop_in,
                                              const std::vector<float>& matrices, float knee,
                                              const ProfileData& prof, const float ceil_rgb[3],
                                              float shoulder = 1.0f, uint64_t* max_hist = nullptr,
                                              PreviewRenderer* renderer = nullptr) {
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
  if (renderer) {
    renderer->highlight_clip(ha);
  } else {
    negicc::cpu_highlight_clip(ha);
  }
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

  // The passes over the frame (ParamSearchOptions::renderer; null: the CPU's). Set before set_frame.
  void set_renderer(PreviewRenderer* renderer) { renderer_ = renderer; }

  // A preview pass (render_preview, render_net_preview) through the renderer, or the CPU's.
  void render(const negicc::RenderArgs& ra) {
    if (renderer_) {
      renderer_->render(ra);
    } else {
      negicc::cpu_render(ra);
    }
  }

  // Takes the upright frame view, its inner crop and the statistics.
  void set_frame(const negicc::RgbView& frame, float crop_ratio) {
    frame_ = frame;
    if (renderer_) renderer_->bind_frame(frame_);
    crop_ = compute_inner_crop(frame.width, frame.height, crop_ratio);
    compute_frame_stats(frame, crop_ratio, engine_.max_size(), engine_.patch_size(), stats_);
  }
  // The same with the crop and the statistics already taken of this frame at this engine's grid (several searches of
  // one frame: the print's materials, search_print_continuous).
  void set_frame(const negicc::RgbView& frame, const InnerCrop& crop, const FrameStats& stats) {
    frame_ = frame;
    if (renderer_) renderer_->bind_frame(frame_);
    crop_ = crop;
    stats_ = stats;
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
  // The print's gains in its curves (SolverConfig::gain_curves) for every render and optimiser pass; null: in the
  // matrix. The callee outlives the frame.
  void set_gain_curves(const GainCurves* gc) { gain_curves_ = gc; }
  void set_shoulder(float x_hi, bool automatic) {
    shoulder_fixed_ = x_hi;
    shoulder_auto_ = automatic;
  }
  float shoulder_of(const ConversionParams& p) const { return shoulder_auto_ ? p.shoulder : shoulder_fixed_; }
  // The parameters every search() holds (ParamSearchOptions::pin_mask), E in the search's units.
  void set_pins(unsigned pin_mask, float e, float g, float b) {
    pin_mask_ = pin_mask;
    pin_e_ = e;
    pin_g_ = g;
    pin_b_ = b;
  }

  const InnerCrop& inner_crop() const { return crop_; }
  const FrameStats& stats() const { return stats_; }
  // For --probe_intent: the preview's trained cells (trained_cells_lab), what the residual reads and what the network
  // is trained to predict.
  std::vector<float> patch_lab(const std::vector<uint8_t>& preview) const {
    std::vector<float> lab((size_t)stats_.grid_w * stats_.grid_h * 3);
    trained_cells_lab(preview.data(), crop_.w, crop_.h, stats_.grid_w, stats_.grid_h, engine_.patch_size(), lab.data());
    return lab;
  }
  // What intent_loss scores at these parameters: the patch-averaged camera-RGB grid (stats_.grid) rendered as the
  // (E, g, b) optimiser renders it.
  std::vector<float> optimiser_lab(int target_idx, const ConversionParams& cp) const {
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    cfg.shoulder = cp.shoulder;
    cfg.gain_curves = gain_curves_;
    const int n = stats_.grid_w * stats_.grid_h;
    std::vector<float> lab((size_t)n * 3);
    intent_cells(cfg, input_matrix_, stats_.grid.data(), n, {cp.e, cp.g, cp.b}, lab.data());
    return lab;
  }
  const float* frame_matrix() const { return frame_matrix_; }
  const float* input_matrix() const { return input_matrix_; }
  float e_norm() const { return e_norm_; }

  // The optimiser's correction for one bracket (DESIGN.md, "The Auto search"): intent_loss renders the
  // patch-averaged camera RGB, and rendering an average is not the average of the render, so its cells read darker
  // than the picture's (2.1 L* on TriLED, 1.7 on white LED, differently per frame) and bluer. After each preview of a
  // bracket, the trained cells of that preview minus the optimiser's cells at the same parameters are kept, and the
  // bracket's next search() aims the optimiser at the field minus that difference, so that at the fixed point the
  // picture, not the optimiser's reading of it, matches the field.
  void update_correction(int target_idx, const ConversionParams& params, const std::vector<uint8_t>& preview) {
    const int n = stats_.grid_w * stats_.grid_h;
    std::vector<float> picture((size_t)n * 3), optimiser((size_t)n * 3);
    trained_cells_lab(preview.data(), crop_.w, crop_.h, stats_.grid_w, stats_.grid_h, engine_.patch_size(),
                      picture.data());
    cells_ = picture;  // kept for residual() on this same preview
    cells_gen_ = preview_gen_;
    cells_data_ = preview.data();
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    cfg.shoulder = params.shoulder;
    cfg.gain_curves = gain_curves_;
    intent_cells(cfg, input_matrix_, stats_.grid.data(), n, {params.e, params.g, params.b}, optimiser.data());
    std::vector<float>& c = correction_[target_idx];
    c.resize(picture.size());
    for (size_t i = 0; i < picture.size(); ++i) c[i] = picture[i] - optimiser[i];
  }

  // The inner crop rendered to 8-bit sRGB through bracket target_idx at params: what DINOv3 sees. out is crop_.w x
  // crop_.h pixels, row-major, 3 bytes each; DINOv3 resamples it to its input size.
  void render_preview(int target_idx, const ConversionParams& params, std::vector<uint8_t>& out) {
    ++preview_gen_;  // a new preview: the cells kept from the last one are not reused
    const Clock::time_point t0 = Clock::now();
    // input_matrix_ with its rows scaled by E, E g and E b (with gain curves: by E, and g and b in the curves)
    SolverConfig sc;
    sc.prof = &bundle_.targets[target_idx].film_prof.data;
    sc.knee = AUTO_KNEE;
    sc.shoulder = shoulder_of(params);
    sc.gain_curves = gain_curves_;
    float m[9];
    std::vector<float> trc[3];
    const negicc::PipelineConfig pc = candidate_config(sc, input_matrix_, {params.e, params.g, params.b}, m, trc);
    out.resize((size_t)crop_.w * crop_.h * 3);
    negicc::RgbView crop_view = frame_.crop_and_orient(crop_.x0, crop_.y0, crop_.w, crop_.h, 0, false, false);
    negicc::RenderArgs ra{};
    ra.in_view = &crop_view;
    ra.width = crop_.w;
    ra.height = crop_.h;
    ra.matrix = m;
    ra.cfg = pc;
    ra.out_type = 1;
    ra.out = out.data();
    render(ra);
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
  // The frame's one lightness target: the L* channel (mu and sigma) of the bootstrap field. Every fixed-point loop
  // scores and searches L* against it, while a* and b* are re-predicted from each step's render (DESIGN.md, "The
  // Auto search").
  void set_lightness_target(const std::vector<float>& mu, const std::vector<float>& sigma) {
    const size_t n = mu.size() / 3;
    target_mu_l_.resize(n);
    target_sigma_l_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      target_mu_l_[i] = mu[i * 3];
      target_sigma_l_[i] = sigma[i * 3];
    }
  }

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
    render(ra);
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
    if (r.grid_w != stats_.grid_w || r.grid_h != stats_.grid_h || r.mu.size() != n || r.sigma.size() != n) {
      std::cerr << "ERROR! DINOv3's intent field is on a " << r.grid_w << " x " << r.grid_h << " grid ("
                << r.mu.size() << " cells), the frame's statistics on " << stats_.grid_w << " x " << stats_.grid_h
                << std::endl;
      return false;
    }
    mu = std::move(r.mu);
    sigma = std::move(r.sigma);
    tone_features.clear();
    if (r.pooled.size() == 1152 && r.token_sd.size() == 768) {
      tone_features = r.pooled;
      tone_features.insert(tone_features.end(), r.token_sd.begin(), r.token_sd.end());
    }
    return true;
  }

  // The residual of a preview against a field (FixedPointResult::residual): in the loops, the field predicted from the
  // preview with L* replaced by the frame's lightness target.
  float residual(const std::vector<uint8_t>& preview, const std::vector<float>& mu, const std::vector<float>& sigma) {
    const Clock::time_point t0 = Clock::now();
    const int n = stats_.grid_w * stats_.grid_h;
    const bool reuse = cells_gen_ == preview_gen_ && cells_data_ == preview.data() && cells_.size() == (size_t)n * 3;
    const float err = reuse ? residual_from_cells(cells_.data(), n, mu.data(), sigma.data())
                            : self_consistency_residual(preview.data(), crop_.w, crop_.h, stats_.grid_w,
                                                        stats_.grid_h, engine_.patch_size(), mu.data(), sigma.data());
    residual_time_ += Clock::now() - t0;
    return err;
  }

  // The coordinate descent of (E, g, b) of bracket target_idx against an intent field, from initial (or the
  // search's defaults when null) with E within e_bounds = {lo, hi, centre}; returns the intent loss
  // (ScreeningResult::intent_loss). pullback_search (the highlight pullback's re-search) replaces the pull of E toward
  // the centre with penalties on E outside the bracket's calibrated range and, when initial is null (its screening),
  // searches g and b on wider grids (dinov3_auto_solver.cc).
  // The field the optimiser aims at is corrected per bracket by the picture its last preview gave
  // (update_correction), and a bracket whose usable range is inverted gets no centre pull (pull_centre).
  float search(int target_idx, const std::vector<float>& mu, const std::vector<float>& sigma,
                const ConversionParams* initial, const float e_bounds[3], ConversionParams& out,
                bool pullback_search = false) {
    const Clock::time_point t0 = Clock::now();
    SolverConfig cfg;
    cfg.prof = &bundle_.targets[target_idx].film_prof.data;
    cfg.knee = AUTO_KNEE;
    cfg.shoulder = initial ? shoulder_of(*initial) : shoulder_fixed_;
    cfg.gain_curves = gain_curves_;
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
    cfg.pin_e = pin_e_;
    cfg.pin_g = pin_g_;
    cfg.pin_b = pin_b_;
    IntentGains init, res;
    if (initial) {
      init = {(pin_mask_ & PIN_E) ? pin_e_ : initial->e, (pin_mask_ & PIN_G) ? pin_g_ : initial->g,
              (pin_mask_ & PIN_B) ? pin_b_ : initial->b};
    }
    // The field the optimiser aims at: the field minus this bracket's last measured correction (update_correction).
    std::vector<float> mu_aim;
    const float* mu_opt = mu.data();
    const auto corr = correction_.find(target_idx);
    if (corr != correction_.end() && corr->second.size() == mu.size()) {
      mu_aim.resize(mu.size());
      for (size_t i = 0; i < mu.size(); ++i) mu_aim[i] = mu[i] - corr->second[i];
      mu_opt = mu_aim.data();
    }
    const float loss = optimize_target_for_intent(cfg, input_matrix_, stats_.grid.data(), stats_.grid_w,
                                                  stats_.grid_h, mu_opt, sigma.data(), initial ? &init : nullptr,
                                                  e_bounds, pin_mask_, res);
    out = {res.e, res.g, res.b, cfg.shoulder};
    search_time_ += Clock::now() - t0;
    return loss;
  }

  // The fixed-point loop of bracket target_idx (FixedPointResult), from start (its screening result) with E
  // kept within the trust region e_bounds = {lo, hi, centre}; outside pullback_search the search's penalty pulls E
  // toward the caller's centre (none when it is 0). pullback_search as for search(). Every caller passes an ordered
  // region: the tournament's and the clip fallback's trust bounds lie around a screened E that is inside its window;
  // an unordered one is ordered here and keeps the caller's centre.
  // L* is held to the frame's lightness target, and the correction of update_correction is measured at every step.
  bool fixed_point(int target_idx, const ConversionParams& start, float e_bounds[3], FixedPointResult& fp,
                   bool pullback_search = false) {
    if (e_bounds[0] > e_bounds[1]) std::swap(e_bounds[0], e_bounds[1]);  // the centre stays the caller's
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
      update_correction(target_idx, curr, preview);
      if (!predict(preview, mu, sigma)) return false;
      // The L* the model asks for follows the picture it is shown, so a loop re-predicting L* from its own render
      // drifts lighter or darker by itself, and residuals measured against different fields cannot rank lightness:
      // two finalists that screening placed at the same L* ended 10-14 L* apart and the lower residual picked one.
      // L* is held to the frame's one lightness target; a* and b* keep the re-prediction.
      if (target_mu_l_.size() * 3 == mu.size()) {
        for (size_t i = 0; i < target_mu_l_.size(); ++i) {
          mu[i * 3] = target_mu_l_[i];
          sigma[i * 3] = target_sigma_l_[i];
        }
      }
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

  // Re-scores the best step with its x_hi and with the shoulder off against the field it was scored against. The guard
  // compares the L* loss alone: the shoulder removes lightness, so a 2% worse mean of L*, a*, b* would be about 6% on
  // L* (the a* and b* terms barely move). The penalties do not depend on the shoulder. The shoulder is kept either way:
  // the guard is only recorded when the loss is more than 2% worse, so the evaluation sees how often it disagrees.
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
                                  ceil, candidates.empty() ? shoulder_fixed_ : shoulder_of(candidates[0]), nullptr,
                                  renderer_);
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
                           1.0f, hist.data(), renderer_);
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
  PreviewRenderer* renderer_ = nullptr;   // ParamSearchOptions::renderer: null for the CPU's passes
  const GainCurves* gain_curves_ = nullptr;   // set_gain_curves
  float shoulder_fixed_ = 1.0f;
  unsigned pin_mask_ = 0;
  float pin_e_ = 1.0f, pin_g_ = 1.0f, pin_b_ = 1.0f;
  bool shoulder_auto_ = false;
  Clock::duration render_time_{}, infer_time_{}, search_time_{}, residual_time_{};  // added up per measurement
  std::vector<float> last_neutral_p_;  // predict(): the neutral head's p of the last inference
  std::vector<float> last_tone_features_;  // predict(): the tone heads' input from the last inference
  std::map<int, std::vector<float>> correction_;  // update_correction(): per bracket, picture minus optimiser cells
  // update_correction(): the preview's trained cells, and the preview they came from (its render's generation and
  // buffer). residual() takes them when it is given that same preview; render_preview advances the generation, so
  // the cells of an earlier render are never reused.
  std::vector<float> cells_;
  unsigned long preview_gen_ = 0, cells_gen_ = ~0ul;
  const uint8_t* cells_data_ = nullptr;
  std::vector<float> target_mu_l_, target_sigma_l_;  // set_lightness_target(): the bootstrap field's L* channel
};

// The merged input matrix M_frame (ParamSearchResult): the crosstalk rows scaled by normalization_target / film
// base and by the exposure ratio between the film base capture and the scan.
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
  const bool has_custom_fb =
      (opts.film_base_rgb != nullptr || opts.film_base_rgb_d != nullptr) && opts.film_base_from_scan;
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
                           float* q995, const double* film_base_rgb, PreviewRenderer* renderer) {
  return measure_rolloff_x_hi(bundle, bundle.targets[target_idx].film_prof.data, frame, scan_shutter_s, scan_iso, e, g,
                              b, crop_ratio, q995, film_base_rgb, renderer);
}

float measure_rolloff_x_hi(const ProfileBundle& bundle, const negicc::ProfileData& prof_in,
                           const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995, const double* film_base_rgb, PreviewRenderer* renderer) {
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
  if (renderer) renderer->bind_frame(frame);
  measure_highlight_clip(frame, compute_inner_crop(frame.width, frame.height, crop_ratio), std::vector<float>(m, m + 9),
                         AUTO_KNEE, prof, kCeil, 1.0f, hist.data(), renderer);
  return shoulder_x_hi_from_hist(hist.data(), q995);
}

// L* of a grey whose luminance is multiplied by 2 ** stops (CIE L* <-> relative luminance Y). CIE 1976 L* with CIE
// 15's exact kLabKappa and kLabEpsilon, whose product is the break in L*, 8.
static float lstar_after_stops(float L, float stops) {
  double Y = L > 8.0f ? std::pow(((double)L + 16.0) / 116.0, 3.0) : (double)L / negicc::kLabKappa;
  Y *= std::pow(2.0, (double)stops);
  return (float)(Y > negicc::kLabEpsilon ? 116.0 * std::cbrt(Y) - 16.0 : negicc::kLabKappa * Y);
}

bool solve_grey_knobs(const ProfileData& prof, float base_e, float base_g, float base_b, float brightness_stops,
                      float tint_a, float warmth_b, KnobSolveResult& out, const ProfileData* grey_prof) {
  out.e = base_e;
  out.g = base_g;
  out.b = base_b;
  out.limited = false;
  out.grey_lab[0] = 50.0f;
  out.grey_lab[1] = 0.0f;
  out.grey_lab[2] = 0.0f;
  out.grey_miss = 0.0f;
  if (!(base_e > 0.0f) || !(base_g > 0.0f) || !(base_b > 0.0f)) return false;

  const GreyPoint gp(prof, true);
  // The grey: the input grey_prof (else prof) renders as neutral L* 50.
  float log_t0[3], miss = 0.0f;
  if (!GreyPoint(grey_prof ? *grey_prof : prof, true).reference_grey(log_t0, &miss)) {
    out.grey_miss = miss;
    return false;
  }
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

// The knobs on a print (negicc #120): brightness moves the print's mid grey A (its curves recomputed at A), tint and
// warmth move g and b (in the print's curves), E held. The grey is the film point the print at (a0, g0, b0) renders as
// neutral L* 50; the solve finds (A, g, b) so that it renders at the target through the print at (A, g, b). Newton on
// (A, ln g, ln b) with central differences (kPrintKnobDA in A, GreyPoint's FD_STEP in ln g and ln b); unreachable
// targets are scaled back as solve_grey_knobs scales them. A stays in [a_lo, a_hi]: the print's curves hold flat past
// the ends of the grey balance curve, where the grey would not move with A.
bool solve_print_knobs(const std::function<void(double a, double g, double b, ParsedFilmProfile& prof)>& print_at,
                       double a0, float g0, float b0, double a_lo, double a_hi, float brightness_stops, float tint_a,
                       float warmth_b, PrintKnobResult& out) {
  constexpr double kPrintKnobDA = 1e-3;      // APD: the step of d Lab / d A
  constexpr double kPrintKnobMaxDA = 0.1;    // APD: the largest Newton step in A (the paper's curve is
                                             // steep, and a full step overshoots there)
  constexpr int kPrintKnobIter = 30;         // Newton iterations at most
  constexpr double FD_STEP = 0.05, MAX_LOG_STEP = 0.25;   // ln g, ln b: GreyPoint's own values
  constexpr float TOLERANCE = 0.02f;         // GreyPoint's: every Lab component within it ends the solve
  constexpr float REACHABLE = 0.25f;         // as solve_grey_knobs
  out = PrintKnobResult{};
  out.a = a0; out.g = g0; out.b = b0;
  if (!std::isfinite(a0) || !std::isfinite(g0) || !std::isfinite(b0) || !(g0 > 0.0f) || !(b0 > 0.0f) ||
      !std::isfinite(a_lo) || !std::isfinite(a_hi) || !(a_lo < a_hi) || !(a_lo <= a0 && a0 <= a_hi)) {
    return false;
  }
  ParsedFilmProfile prof;
  print_at(a0, g0, b0, prof);
  float log_t0[3];
  if (!GreyPoint(prof.data, true).reference_grey(log_t0)) return false;
  const float one[3] = {1.0f, 1.0f, 1.0f};
  auto render = [&](double a, double lg, double lb, float lab[3]) {
    ParsedFilmProfile p;
    print_at(a, std::exp(lg), std::exp(lb), p);
    GreyPoint(p.data).grey(log_t0, one, one, lab);
  };
  if (brightness_stops == 0.0f && tint_a == 0.0f && warmth_b == 0.0f) {   // as solve_grey_knobs: nothing to move
    render(a0, std::log((double)g0), std::log((double)b0), out.grey_lab);
    return true;
  }
  auto solve_to = [&](const float target[3], double x_out[3]) -> float {
    double x[3] = {a0, std::log((double)g0), std::log((double)b0)};
    float lab[3];
    render(x[0], x[1], x[2], lab);
    for (int it = 0; it < kPrintKnobIter; ++it) {
      double r[3];
      bool within = true;
      for (int c = 0; c < 3; ++c) { r[c] = target[c] - lab[c]; within = within && std::fabs(r[c]) <= TOLERANCE; }
      if (within) break;
      double J[9];
      const double h[3] = {kPrintKnobDA, FD_STEP, FD_STEP};
      for (int v = 0; v < 3; ++v) {     // central differences, as GreyPoint; one-sided at an end of A's range
        double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
        xp[v] += h[v];
        xm[v] -= h[v];
        if (v == 0) {
          xp[0] = std::min(xp[0], a_hi);
          xm[0] = std::max(xm[0], a_lo);
        }
        float lp[3], lm[3];
        render(xp[0], xp[1], xp[2], lp);
        render(xm[0], xm[1], xm[2], lm);
        for (int c = 0; c < 3; ++c) J[c * 3 + v] = (lp[c] - lm[c]) / (xp[v] - xm[v]);
      }
      const double det = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) +
                         J[2] * (J[3] * J[7] - J[4] * J[6]);
      if (!(std::fabs(det) > 1e-12)) break;
      double inv[9] = {
          (J[4] * J[8] - J[5] * J[7]) / det, (J[2] * J[7] - J[1] * J[8]) / det, (J[1] * J[5] - J[2] * J[4]) / det,
          (J[5] * J[6] - J[3] * J[8]) / det, (J[0] * J[8] - J[2] * J[6]) / det, (J[2] * J[3] - J[0] * J[5]) / det,
          (J[3] * J[7] - J[4] * J[6]) / det, (J[1] * J[6] - J[0] * J[7]) / det, (J[0] * J[4] - J[1] * J[3]) / det};
      double step[3];
      for (int v = 0; v < 3; ++v) step[v] = inv[v * 3] * r[0] + inv[v * 3 + 1] * r[1] + inv[v * 3 + 2] * r[2];
      step[0] = std::clamp(step[0], -kPrintKnobMaxDA, kPrintKnobMaxDA);
      step[1] = std::clamp(step[1], -MAX_LOG_STEP, MAX_LOG_STEP);
      step[2] = std::clamp(step[2], -MAX_LOG_STEP, MAX_LOG_STEP);
      for (int v = 0; v < 3; ++v) x[v] += step[v];
      x[0] = std::clamp(x[0], a_lo, a_hi);
      render(x[0], x[1], x[2], lab);
    }
    float e2 = 0.0f;
    for (int c = 0; c < 3; ++c) e2 += (target[c] - lab[c]) * (target[c] - lab[c]);
    for (int v = 0; v < 3; ++v) x_out[v] = x[v];
    return std::sqrt(e2);
  };
  const float full[3] = {lstar_after_stops(50.0f, brightness_stops), tint_a, warmth_b};
  double x[3];
  auto take = [&](const double xv[3], bool limited) {
    out.a = xv[0]; out.g = (float)std::exp(xv[1]); out.b = (float)std::exp(xv[2]); out.limited = limited;
    render(xv[0], xv[1], xv[2], out.grey_lab);
  };
  if (solve_to(full, x) <= REACHABLE) { take(x, false); return true; }
  auto largest = [&](auto make_target, double best[3]) -> bool {
    float lo = 0.0f, hi = 1.0f; bool found = false;
    for (int i = 0; i < 7; ++i) {
      const float f = 0.5f * (lo + hi); float tgt[3]; double xr[3];
      make_target(f, tgt);
      if (solve_to(tgt, xr) <= REACHABLE) { lo = f; for (int v = 0; v < 3; ++v) best[v] = xr[v]; found = true; }
      else hi = f;
    }
    return found;
  };
  const float colour_only[3] = {50.0f, tint_a, warmth_b};
  double best[3] = {a0, std::log((double)g0), std::log((double)b0)}, xc[3];
  if (solve_to(colour_only, xc) <= REACHABLE) {
    for (int v = 0; v < 3; ++v) best[v] = xc[v];
    largest([&](float f, float t[3]) { t[0] = 50.0f + f * (full[0] - 50.0f); t[1] = tint_a; t[2] = warmth_b; }, best);
  } else {
    largest([&](float f, float t[3]) { t[0] = 50.0f + f * (full[0] - 50.0f); t[1] = f * tint_a; t[2] = f * warmth_b; },
            best);
  }
  take(best, true);
  return true;
}

bool knob_grey_lab(const ProfileData& grey_prof, const ProfileData& prof, const float base[3], const float egb[3],
                   float lab_out[3]) {
  float log_t0[3];
  if (!GreyPoint(grey_prof, true).reference_grey(log_t0)) return false;
  GreyPoint(prof).grey(log_t0, base, egb, lab_out);
  return true;
}

float knob_grey_miss_stops(float brightness_stops, float grey_L) {
  auto Y = [](double L) { return L > 8.0 ? std::pow((L + 16.0) / 116.0, 3.0) : L / negicc::kLabKappa; };
  return (float)std::log2(Y(grey_L) / Y(lstar_after_stops(50.0f, brightness_stops)));
}

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& frame,
                              float scan_shutter_s, float scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out) {
  return search_conversion_params(
      bundle, engine, negicc::RgbView::from_interleaved_rgb(frame.rgb.data(), frame.width, frame.height),
      (double)scan_shutter_s, (double)scan_iso, opts, out);
}

namespace {
// Inside an entry point of the bundle search, process-wide, so that a pass started on another thread (which the
// contract does not allow) is counted too.
std::atomic<int> g_audit_depth{0};
thread_local bool t_in_renderer = false;  // inside PassThroughRenderer's forwarding, on this thread
struct AuditScope {
  AuditScope() { ++g_audit_depth; }
  ~AuditScope() { --g_audit_depth; }
};
// Marks PassThroughRenderer's forwarding, and clears the mark when the forwarded pass returns or throws.
struct InRenderer {
  InRenderer() { t_in_renderer = true; }
  ~InRenderer() { t_in_renderer = false; }
};
}  // namespace

PassAudit& pass_audit() {
  static PassAudit a;
  return a;
}

void audit_cpu_pass(int kind) {
  (void)kind;
  if (g_audit_depth.load() > 0 && !t_in_renderer) pass_audit().outside++;
}

void PassThroughRenderer::bind_frame(const negicc::RgbView& frame) {
  (void)frame;
  pass_audit().binds++;
}
void PassThroughRenderer::render(const negicc::RenderArgs& a) {
  pass_audit().renders++;
  const InRenderer in_renderer;
  negicc::cpu_render(a);
}
void PassThroughRenderer::highlight_clip(const negicc::HighlightClipArgs& a) {
  pass_audit().clips++;
  const InRenderer in_renderer;
  negicc::cpu_highlight_clip(a);
}

// ---- The GPU renderer (multi_target_optimizer.h GpuRenderer) -------------------------------------------------------

namespace {

// A device buffer that grows, freed with the renderer.
struct DeviceBuffer {
  void* ptr = nullptr;
  size_t size = 0;
  void* get(size_t bytes) {
    if (bytes > size) {
      negicc::gpu_free(ptr);
      ptr = nullptr;
      size = 0;
      ptr = negicc::gpu_malloc(bytes);
      size = bytes;
    }
    return ptr;
  }
  void release() {
    negicc::gpu_free(ptr);
    ptr = nullptr;
    size = 0;
  }
  ~DeviceBuffer() { negicc::gpu_free(ptr); }
};

// One table on the device, with the host contents it was copied from (compared on every call).
struct DeviceTable {
  DeviceBuffer buf;
  std::vector<float> host;
  const float* sync(const float* src, int n) {
    if (!src || n <= 0) return nullptr;
    if ((int)host.size() != n || std::memcmp(host.data(), src, (size_t)n * sizeof(float)) != 0) {
      host.assign(src, src + n);
      negicc::gpu_upload(buf.get((size_t)n * sizeof(float)), src, (size_t)n * sizeof(float));
    }
    return static_cast<const float*>(buf.ptr);
  }
};

struct DeviceProfile {
  DeviceTable clut, out_trc[3], in_trc[3];
};

bool same_view(const negicc::RgbView& a, const negicc::RgbView& b) {
  return a.width == b.width && a.height == b.height && a.r_ptr == b.r_ptr && a.g_ptr == b.g_ptr &&
         a.g2_ptr == b.g2_ptr && a.b_ptr == b.b_ptr && a.pitch == b.pitch && a.stride == b.stride &&
         a.flat.gain == b.flat.gain && a.flat.ax_i == b.flat.ax_i && a.flat.ay_i == b.flat.ay_i;
}

}  // namespace

struct GpuRenderer::Impl {
  negicc::RgbView frame;            // the bound frame, as given
  bool bound = false;
  const uint16_t* dev_frame = nullptr;
  bool copied = false;              // dev_frame is the renderer's copy: contiguous, frame.width x frame.height
  DeviceBuffer frame_copy, out;
  DeviceTable tone;
  // Per profile, keyed by its cLUT and first output table, which the search keeps for its whole run.
  std::map<std::pair<const float*, const float*>, std::unique_ptr<DeviceProfile>> profiles;

  // The pass's view on the device: the bound frame or an upright crop inside it with the same layout, at its offset.
  // The crop's origin is found from byte addresses, so a view of another buffer is refused without comparing pointers
  // into different allocations.
  negicc::RgbView device_view(const negicc::RgbView* v) const {
    if (!bound) throw negicc::GpuError("GPU renderer: a pass before bind_frame");
    if (!v || !v->valid()) throw negicc::GpuError("GPU renderer: a pass without a view of the bound frame");
    if (v->pitch != frame.pitch || v->stride != frame.stride || (v->g2_ptr != nullptr) != (frame.g2_ptr != nullptr) ||
        v->flat.gain != frame.flat.gain)
      throw negicc::GpuError("GPU renderer: a view that is not an upright crop of the bound frame");
    auto addr = [](const uint16_t* p) { return (long long)reinterpret_cast<uintptr_t>(p); };
    const long long off = addr(v->r_ptr) - addr(frame.r_ptr), es = (long long)sizeof(uint16_t);
    const bool same_layout = off % es == 0 && addr(v->g_ptr) - addr(frame.g_ptr) == off &&
                             addr(v->b_ptr) - addr(frame.b_ptr) == off &&
                             (!v->g2_ptr || addr(v->g2_ptr) - addr(frame.g2_ptr) == off);
    // The origin (x, y) with off / es = y * stride + x * pitch, the crop inside the frame.
    long long ox = -1, oy = -1;
    if (same_layout && frame.pitch != 0) {
      const long long e = off / es;
      for (long long y = 0; y + v->height <= frame.height; ++y) {
        const long long rem = e - y * (long long)frame.stride;
        if (rem % frame.pitch != 0) continue;
        const long long x = rem / frame.pitch;
        if (x < 0 || x + v->width > frame.width) continue;
        ox = x;
        oy = y;
        break;
      }
    }
    if (ox < 0) throw negicc::GpuError("GPU renderer: a view that is not inside the bound frame");
    negicc::RgbView d = *v;
    d.g2_ptr = nullptr;
    d.flat = negicc::FlatSampleArgs{};
    d.flat_tables.reset();
    if (!copied) {   // the caller's frame on the device, in its own layout
      d.r_ptr = dev_frame + (oy * frame.stride + ox * frame.pitch);
    } else {         // the copy: contiguous, frame.width x frame.height
      d.r_ptr = dev_frame + ((size_t)oy * frame.width + (size_t)ox) * 3;
      d.pitch = 3;
      d.stride = 3 * (ptrdiff_t)frame.width;
    }
    d.g_ptr = d.r_ptr + 1;
    d.b_ptr = d.r_ptr + 2;
    return d;
  }

  negicc::PipelineConfig device_config(const negicc::PipelineConfig& c) {
    negicc::PipelineConfig d = c;
    const negicc::ProfileData& p = c.profile;
    if (p.has_profile) {
      auto& dp = profiles[{p.clut, p.out_trc[0]}];
      if (!dp) dp = std::make_unique<DeviceProfile>();
      d.profile.clut = dp->clut.sync(p.clut, p.clut_dim[0] * p.clut_dim[1] * p.clut_dim[2] * 3);
      for (int ch = 0; ch < 3; ++ch) {
        d.profile.out_trc[ch] = dp->out_trc[ch].sync(p.out_trc[ch], p.out_trc_size[ch]);
        d.profile.in_trc[ch] = dp->in_trc[ch].sync(p.in_trc[ch], p.in_trc_size[ch]);
      }
    }
    if (c.tone.lut && c.tone.n > 0) d.tone.lut = tone.sync(c.tone.lut, c.tone.n + 1);
    return d;
  }
};

GpuRenderer::GpuRenderer() : impl_(std::make_unique<Impl>()) {
  if (!negicc::gpu_device_available()) throw negicc::GpuError("no CUDA device that runs this build's GPU passes");
}
GpuRenderer::~GpuRenderer() = default;

void GpuRenderer::bind_frame(const negicc::RgbView& frame) {
  ++binds;
  Impl& m = *impl_;
  if (m.bound && same_view(m.frame, frame)) return;   // the same frame again (each roll-off measurement binds it)
  m.frame = frame;
  m.bound = false;
  const uint16_t* given = device_frame ? device_frame(frame) : nullptr;
  if (given) {
    if (frame.g2_ptr || frame.flat.gain || frame.g_ptr != frame.r_ptr + 1 || frame.b_ptr != frame.r_ptr + 2)
      throw negicc::GpuError("GPU renderer: a frame read in place must be interleaved RGB without a flat field");
    m.dev_frame = given;
    m.copied = false;
  } else {
    const int w = frame.width, h = frame.height;
    std::vector<uint16_t> host((size_t)w * h * 3);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) frame.sample(x, y, host.data() + ((size_t)y * w + x) * 3);
    void* dev = m.frame_copy.get(host.size() * sizeof(uint16_t));
    negicc::gpu_upload(dev, host.data(), host.size() * sizeof(uint16_t));
    m.dev_frame = static_cast<const uint16_t*>(dev);
    m.copied = true;
  }
  m.bound = true;
}

void GpuRenderer::render(const negicc::RenderArgs& a) {
  ++renders;
  Impl& m = *impl_;
  const negicc::RgbView view = m.device_view(a.in_view);
  negicc::RenderArgs d = a;
  d.in = nullptr;
  d.in_view = &view;
  d.cfg = m.device_config(a.cfg);
  const size_t elem = a.out_type == 1 ? 1 : a.out_type == 0 ? 2 : 4;
  const size_t bytes = (size_t)a.width * a.height * 3 * elem;
  if (bytes == 0) return;
  d.out = m.out.get(bytes);
  negicc::gpu_render(d);
  negicc::gpu_download(a.out, d.out, bytes);
}

void GpuRenderer::release_frame() {
  Impl& m = *impl_;
  m.bound = false;
  m.dev_frame = nullptr;
  m.copied = false;
  m.frame_copy.release();
  m.out.release();
}

void GpuRenderer::highlight_clip(const negicc::HighlightClipArgs& a) {
  ++clips;
  Impl& m = *impl_;
  const negicc::RgbView view = m.device_view(a.in_view);
  negicc::HighlightClipArgs d = a;
  d.in = nullptr;
  d.in_view = &view;
  d.cfg = m.device_config(a.cfg);
  negicc::gpu_highlight_clip(d);
}

// ---- The comparison (multi_target_optimizer.h CompareRenderer) ------------------------------------------------------

void CompareRenderer::bind_frame(const negicc::RgbView& frame) {
  PassThroughRenderer::bind_frame(frame);
  gpu.bind_frame(frame);
}

namespace {
// One compared quantity of a pass: the difference, counted as differing when above 0 and as not finite when either
// side is not (NaN or infinite), which within_tolerance refuses.
double compared(CompareRenderer::Pass& p, double cpu, double gpu) {
  const double diff = std::fabs(cpu - gpu);
  if (!std::isfinite(diff)) {
    ++p.nonfinite;
    ++p.differing;
    return 0.0;
  }
  if (diff > 0.0) ++p.differing;
  return diff;
}
}  // namespace

void CompareRenderer::render(const negicc::RenderArgs& a) {
  PassThroughRenderer::render(a);
  const size_t elem = a.out_type == 1 ? 1 : a.out_type == 0 ? 2 : 4;
  const size_t n = (size_t)a.width * a.height * 3;
  std::vector<uint8_t> other(n * elem);
  negicc::PositiveStats other_stats{};
  negicc::RenderArgs g = a;
  g.out = other.data();
  g.stats = a.stats ? &other_stats : nullptr;
  gpu.render(g);
  Pass p;
  p.render = true;
  p.values = (long long)n;
  auto value = [&](const void* buf, size_t i) -> double {
    if (elem == 1) return static_cast<const uint8_t*>(buf)[i];
    if (elem == 2) return static_cast<const uint16_t*>(buf)[i];
    return static_cast<const float*>(buf)[i];
  };
  // Each pixel at the output's full scale (255, 65535, or 1 for a float render), read in the render's encoding.
  const double full = elem == 1 ? 255.0 : elem == 2 ? 65535.0 : 1.0;
  const bool linear = a.cfg.encoding == negicc::ENCODE_SRGB_LINEAR;
  auto lstar = [&](const void* buf, size_t i) {
    const negicc::Rgb v = {(float)(value(buf, i) / full), (float)(value(buf, i + 1) / full),
                           (float)(value(buf, i + 2) / full)};
    return (double)(linear ? negicc::linear_srgb_to_lab(v) : negicc::srgb_to_lab(v)).L;
  };
  double sum_dl = 0.0;
  for (size_t i = 0; i + 2 < n; i += 3) {
    bool same = true;
    for (size_t k = i; k < i + 3; ++k) {
      const long long before = p.differing;
      p.max_abs = std::max(p.max_abs, compared(p, value(a.out, k), value(other.data(), k)));
      same = same && p.differing == before;
    }
    if (same) continue;
    const double dl = std::fabs(lstar(a.out, i) - lstar(other.data(), i));   // the pixel's values were counted above
    if (!std::isfinite(dl)) {
      ++p.nonfinite;
      continue;
    }
    sum_dl += dl;
    p.max_dl = std::max(p.max_dl, dl);
  }
  if (n) p.mean_dl = sum_dl / (double)(n / 3);
  if (a.stats) {
    p.stats = true;
    const negicc::PositiveStats& c = *a.stats;
    const negicc::PositiveStats& q = other_stats;
    const negicc::InnerCrop crop = (a.stats_crop.w > 0 && a.stats_crop.h > 0)
                                       ? a.stats_crop
                                       : negicc::InnerCrop{0, 0, a.width, a.height};
    const double pct = 100.0 / std::max(1.0, (double)crop.w * crop.h);   // a count as a share of the crop's pixels
    p.max_share = std::max(compared(p, c.white_clip, q.white_clip), compared(p, c.black_clip, q.black_clip));
    for (int ch = 0; ch < 3; ++ch)
      for (int k = 0; k < 256; ++k)
        p.max_hist = std::max(p.max_hist, compared(p, c.hist_rgb[ch][k], q.hist_rgb[ch][k]) * pct);
    for (int k = 0; k < negicc::kLabBins; ++k)
      p.max_hist = std::max(p.max_hist, compared(p, c.hist_L[k], q.hist_L[k]) * pct);
    const float zc[6] = {c.mid_a, c.mid_b, c.hi_a, c.hi_b, c.sh_a, c.sh_b};
    const float zq[6] = {q.mid_a, q.mid_b, q.hi_a, q.hi_b, q.sh_a, q.sh_b};
    for (int k = 0; k < 6; ++k) p.max_zone = std::max(p.max_zone, compared(p, zc[k], zq[k]));
    p.values += 2 + 3 * 256 + negicc::kLabBins + 6;
  }
  passes.push_back(p);
}

void CompareRenderer::highlight_clip(const negicc::HighlightClipArgs& a) {
  PassThroughRenderer::highlight_clip(a);
  std::vector<float> out((size_t)a.n * 5);
  std::vector<uint64_t> hist(a.max_hist ? (size_t)a.n * negicc::kMaxHistBins : 0);
  negicc::HighlightClipArgs g = a;
  g.out = out.data();
  g.max_hist = a.max_hist ? hist.data() : nullptr;
  gpu.highlight_clip(g);
  Pass p;
  p.values = (long long)a.n * 5 + (long long)hist.size();
  for (int c = 0; c < a.n; ++c)
    for (int k = 0; k < 5; ++k) {
      const double diff = compared(p, a.out[c * 5 + k], out[c * 5 + k]);
      p.max_abs = std::max(p.max_abs, diff);
      if (k < 4) p.max_share = std::max(p.max_share, diff);
      else p.max_median_l = std::max(p.max_median_l, diff);
    }
  const negicc::InnerCrop crop = (a.crop.w > 0 && a.crop.h > 0) ? a.crop : negicc::InnerCrop{0, 0, a.width, a.height};
  const double pct = 100.0 / std::max(1.0, (double)crop.w * crop.h);
  for (size_t i = 0; i < hist.size(); ++i)
    p.max_hist = std::max(p.max_hist, compared(p, (double)a.max_hist[i], (double)hist[i]) * pct);
  passes.push_back(p);
}

bool CompareRenderer::within_tolerance() const {
  for (const Pass& p : passes) {
    if (p.nonfinite) return false;
    if (p.render && p.mean_dl > kCompareMeanDL) return false;
    if (p.max_zone > kCompareMeanDL) return false;
    if (p.max_share > kCompareShare || p.max_hist > kCompareShare) return false;
    if (p.max_median_l > kCompareMedianL + 1e-9) return false;
  }
  return !passes.empty();
}

std::string CompareRenderer::summary() const {
  long renders = 0, renders_diff = 0, with_stats = 0, clips = 0, clips_diff = 0;
  long long nonfinite = 0;
  double mean_dl = 0.0, max_dl = 0.0, max_code = 0.0, r_share = 0.0, r_hist = 0.0, zone = 0.0, share = 0.0,
         median = 0.0, hist = 0.0;
  for (const Pass& p : passes) {
    nonfinite += p.nonfinite;
    if (p.render) {
      ++renders;
      renders_diff += p.differing > 0;
      with_stats += p.stats;
      mean_dl = std::max(mean_dl, p.mean_dl);
      max_dl = std::max(max_dl, p.max_dl);
      max_code = std::max(max_code, p.max_abs);
      r_share = std::max(r_share, p.max_share);
      r_hist = std::max(r_hist, p.max_hist);
      zone = std::max(zone, p.max_zone);
    } else {
      ++clips;
      clips_diff += p.differing > 0;
      share = std::max(share, p.max_share);
      median = std::max(median, p.max_median_l);
      hist = std::max(hist, p.max_hist);
    }
  }
  std::ostringstream s;
  s << std::setprecision(3) << renders << " renders (" << renders_diff << " differ; largest mean change " << mean_dl
    << " L*, largest pixel change " << max_dl << " L*, largest value difference " << max_code;
  if (with_stats)
    s << "; " << with_stats << " with statistics: clip shares " << r_share << " %, histogram bins " << r_hist
      << " %, zone means " << zone;
  s << "), " << clips << " clip passes (" << clips_diff << " differ; shares " << share << " %, median " << median
    << " L*, max-channel histogram bins " << hist << " %)";
  if (nonfinite) s << "; " << nonfinite << " values not finite";
  return s.str();
}

// For --probe_intent: the mean of a flat (c0, c1, c2) cell vector per channel, and writing the probe file.
static std::vector<double> probe_mean3(const std::vector<float>& v) {
  double s[3] = {0, 0, 0};
  const size_t n = v.size() / 3;
  for (size_t i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) s[c] += v[i * 3 + c];
  // No cells: NaN, which the JSON writes as null, so that "no cells" stays apart from a black, neutral Lab.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  return n ? std::vector<double>{s[0] / n, s[1] / n, s[2] / n} : std::vector<double>{nan, nan, nan};
}

static bool write_probe_file(const std::string& path, const nlohmann::json& doc, size_t n_renders, bool verbose) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "ERROR! Cannot write the probe file " << path << std::endl;
    return false;
  }
  f << doc.dump() << "\n";
  if (!f) {
    std::cerr << "ERROR! Cannot write the probe file " << path << std::endl;
    return false;
  }
  if (verbose) std::cout << "Auto: probe: " << n_renders << " renders written to " << path << "\n";
  return true;
}

// Stage 1, the Auto search: picks the bracket and the exposure and gains (E, g, b) on the half-size scan. It takes the
// profile bundle, the intent engine, the scan and its exposure; screens the brackets against the bootstrap field (the
// mean over every metered bracket's render), runs the best through a fixed-point tournament (predicting a* and b*
// again as the render changes, L* held to the frame's lightness target), moves to a denser bracket or pulls E back
// where the chosen one clips, and freezes the highlight shoulder. It hands on the bracket, (E, g, b, x_hi) and the
// winning inference's intent field, neutral-patch probabilities and tone features to the white-light stages
// (run_white_light_fit).
bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                              double scan_shutter_s_in, double scan_iso_in, const ParamSearchOptions& opts,
                              ParamSearchResult& out) {
  const AuditScope audit_scope;  // PassAudit (neg_process --check_renderer)
  const float scan_shutter_s = (float)scan_shutter_s_in;
  const float scan_iso = (float)scan_iso_in;
  const Clock::time_point t_start = Clock::now();
  const bool verbose = opts.verbose;
  if (bundle.targets.empty()) {
    std::cerr << "ERROR! The profile bundle has no targets" << std::endl;
    return false;
  }
  SearchFrame fr(bundle, engine, verbose, opts.max_steps, out);
  fr.set_renderer(opts.renderer);
  auto name_of = [&](int target_idx) { return bundle.targets[target_idx].name; };

  fr.set_shoulder(1.0f, true);  // the roll-off, measured by the fixed-point loop
  fr.set_frame(frame, opts.crop_ratio);
  const FrameStats& stats = fr.stats();
  // Kept for the tone plan of this same frame (plan_auto_tone_stage), which needs the same statistics.
  out.frame_stats = std::make_shared<FrameStats>(stats);
  out.frame_stats_view = frame;
  out.frame_stats_crop_ratio = opts.crop_ratio;
  out.frame_stats_max_size = engine.max_size();
  out.frame_stats_patch = engine.patch_size();

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
  const float g_lo = std::max(1.0f, stats.g_lo);
  const float g_hi = std::max(g_lo + 1.0f, stats.g_hi);
  const float g_scale = white_light ? e_norm : 1.0f;
  const float g_lo_ref = g_lo * g_scale;
  const float g_hi_ref = g_hi * g_scale;
  out.g_lo = g_lo;
  out.g_hi = g_hi;
  out.scene_ev = std::log2(g_hi / g_lo);
  // The pins (ParamSearchOptions): the bracket, and E (in the search's units), g and b.
  const bool pin_target = opts.pin_target >= 0;
  const bool pin_e = (opts.pin_mask & PIN_E) != 0;
  const float boot_g = (opts.pin_mask & PIN_G) ? opts.pin_g : BOOTSTRAP_G;
  const float boot_b = (opts.pin_mask & PIN_B) ? opts.pin_b : BOOTSTRAP_B;
  const float pin_e_search = white_light ? opts.pin_e : opts.pin_e * e_norm;
  fr.set_pins(opts.pin_mask, pin_e_search, opts.pin_g, opts.pin_b);
  if (pin_target && opts.pin_target >= (int)bundle.targets.size()) {
    std::cerr << "ERROR! The pinned bracket " << opts.pin_target << " is not in the bundle" << std::endl;
    return false;
  }
  std::vector<Enclosure> enclosure;
  if (pin_target) {
    // The pinned bracket alone, admissible or not: it is the only one metered, screened and searched.
    Enclosure r;
    if (bracket_enclosure(bundle.targets[opts.pin_target], opts.pin_target, g_lo_ref, g_hi_ref, r)) {
      enclosure.push_back(r);
    }
  } else {
    for (int i : filter_admissible_targets(bundle)) {
      Enclosure r;
      if (bracket_enclosure(bundle.targets[i], i, g_lo_ref, g_hi_ref, r)) {
        enclosure.push_back(r);
      }
    }
  }
  if (enclosure.empty()) {
    std::cerr << (pin_target ? "ERROR! The pinned bracket has no patch readings"
                             : "ERROR! No bracket of the bundle has patch readings") << std::endl;
    return false;
  }
  if (verbose && (pin_target || opts.pin_mask)) {
    std::ostringstream os;
    os << "Auto: pinned:";
    if (pin_target) os << " " << name_of(opts.pin_target);
    if (pin_e) os << " E " << general(opts.pin_e, 6);
    if (opts.pin_mask & PIN_G) os << " g " << general(opts.pin_g, 6);
    if (opts.pin_mask & PIN_B) os << " b " << general(opts.pin_b, 6);
    std::cout << os.str() << "\n";
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

  // Step 2: meter each admissible bracket to L* 50; the bootstrap field is the mean over every metered bracket's
  // render.
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
  // (its metered E furthest from the ends of its search window) renders the bootstrap alone when E is pinned.
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
    if (!exposure_for_lightness(gp, p_scene, boot_g, boot_b, e_met)) {
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
  if (pin_e) e1_init = pin_e_search;
  if (verbose) {
    std::ostringstream os;
    for (const auto& d : depths) {
      os << "  metered " << name_of(d.cand.target_idx) << ": E " << fixed(d.e_met, 4) << " depth "
         << fixed(d.depth, 3, true) << "\n";
    }
    os << "Auto: bootstrap from " << name_of(cand1.target_idx) << " at E " << fixed(e1_init, 4)
       << (depths.empty() || pin_e
               ? " (the field is this render's)"
               : " (the deepest metered bracket; the field is the mean over every metered bracket's render)")
       << "\n";
    std::cout << os.str();
  }

  // The bootstrap field: the mean of the fields predicted from every metered bracket's render at its metered E,
  // clamped to its search window. When E is pinned, the deepest metered bracket renders it alone at the pinned E; when
  // no bracket meters, the best-fit bracket renders it alone at its e_center (at the pinned E when E is pinned). A
  // field from one bracket would depend on which bracket is deepest, and the two deepest are often within 0.02 stop of
  // each other: a 1 % tighter crop could swap them and move the result by several L*. The mean has no such switch. It
  // is the screening's field and the frame's lightness target.
  std::vector<uint8_t> preview;
  std::vector<float> mu1, sig1;
  // The bootstrap's renders, (bracket, E) each, at the bootstrap gains.
  std::vector<std::pair<int, float>> renders;
  if (depths.empty() || pin_e) {
    renders.push_back({cand1.target_idx, e1_init});
  } else {
    for (const auto& d : depths) renders.push_back({d.cand.target_idx, std::clamp(d.e_met, d.lo, d.hi)});
  }
  auto mean_field = [&](std::vector<float>& mu, std::vector<float>& sig) {
    std::vector<float> m, sg;
    for (size_t k = 0; k < renders.size(); ++k) {
      fr.render_preview(renders[k].first, {renders[k].second, boot_g, boot_b}, preview);
      fr.update_correction(renders[k].first, {renders[k].second, boot_g, boot_b}, preview);
      if (!fr.predict(preview, m, sg)) return false;
      if (k == 0) {
        mu.assign(m.size(), 0.0f);
        sig.assign(sg.size(), 0.0f);
      }
      for (size_t i = 0; i < m.size(); ++i) {
        mu[i] += m[i] / renders.size();
        sig[i] += sg[i] / renders.size();
      }
    }
    if (verbose) std::cout << "Auto: bootstrap field averaged over " << renders.size() << " render(s)\n";
    return true;
  };
  if (!mean_field(mu1, sig1)) return false;
  fr.set_lightness_target(mu1, sig1);

  // --probe_intent: what the network predicts from chosen renders (ParamSearchOptions::probe_path), then stop.
  if (!opts.probe_path.empty()) {
    nlohmann::json recs = nlohmann::json::array();
    std::vector<uint8_t> pv;
    std::vector<float> mu, sg;
    auto record = [&](const char* kind, int set_idx, float stop, int target_idx, const ConversionParams& cp) -> bool {
      fr.render_preview(target_idx, cp, pv);
      if (!fr.predict(pv, mu, sg)) return false;
      const std::vector<float> lab = fr.patch_lab(pv), opt = fr.optimiser_lab(target_idx, cp);
      recs.push_back({{"kind", kind}, {"set", set_idx}, {"stop", stop}, {"target_idx", target_idx},
                      {"target", bundle.targets[target_idx].name}, {"e", cp.e}, {"g", cp.g}, {"b", cp.b},
                      {"shoulder", cp.shoulder}, {"predicted", probe_mean3(mu)}, {"sigma", probe_mean3(sg)},
                      {"render", probe_mean3(lab)}, {"optimiser_render", probe_mean3(opt)}, {"mu", mu},
                      {"sigma_cells", sg}, {"render_cells", lab}, {"optimiser_cells", opt}});
      return true;
    };
    int k = 0;
    for (const auto& d : depths) {
      if (!record("metered", k, 0.0f, d.cand.target_idx, {d.e_met, boot_g, boot_b})) return false;
      const float ec = std::clamp(d.e_met, d.lo, d.hi);
      if (ec != d.e_met && !record("metered_clamped", k, 0.0f, d.cand.target_idx, {ec, boot_g, boot_b})) return false;
      ++k;
    }
    for (size_t si = 0; si < opts.probe_params.size(); ++si) {
      const auto& p = opts.probe_params[si];
      const int t = (int)p[0];
      if (t < 0 || t >= (int)bundle.targets.size()) continue;
      for (const float stop : opts.probe_stops) {
        ConversionParams cp{p[1] * std::pow(2.0f, stop), p[2], p[3]};
        float q995 = 0.0f;
        cp.shoulder = fr.measure_shoulder(t, cp, &q995);
        if (!record("params", (int)si, stop, t, cp)) return false;
      }
    }
    const nlohmann::json doc = {{"search", "scene"}, {"grid_w", stats.grid_w}, {"grid_h", stats.grid_h},
                                {"e_norm", out.e_norm}, {"has_crosstalk", bundle.has_crosstalk},
                                {"bootstrap_target", cand1.target_idx}, {"bootstrap_e", e1_init},
                                {"target_L", probe_mean3(mu1)[0]}, {"records", recs}};
    if (!write_probe_file(opts.probe_path, doc, recs.size(), verbose)) return false;
    out.probed = true;
    return true;
  }

  // Step 3: screen the metered brackets (every bracket of the pool, from its e_center, when none metered) against the
  // bootstrap field, each aimed at the field minus its own correction (SearchFrame::search), best first. The
  // pullback's re-search (Step 6) screens with the enclosure penalties in place of the pull toward the centre
  // (pullback_search).
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
      const float eb[3] = {lo, hi, pull_centre(c)};
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
  // Each screened bracket is aimed at the field minus its own correction, and the screening ranks their losses. The
  // bootstrap measured a correction for each bracket it rendered; when it rendered one bracket alone (E pinned, or no
  // bracket metered), every other screened bracket is rendered once at the E its screening starts from (the pinned E,
  // else its e_center) and the bootstrap gains, as the bootstrap's bracket was, to measure its own (no inference).
  if (depths.empty() || pin_e) {
    auto measure = [&](int target_idx, float e) {
      if (target_idx == cand1.target_idx) return;
      fr.render_preview(target_idx, {e, boot_g, boot_b}, preview);
      fr.update_correction(target_idx, {e, boot_g, boot_b}, preview);
    };
    if (!depths.empty()) {
      for (const auto& d : depths) measure(d.cand.target_idx, pin_e_search);
    } else {
      for (const auto& c : pool) measure(c.target_idx, pin_e ? pin_e_search : c.e_center);
    }
  }
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
    eb[2] = pull_centre(c.enclosure);
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
  // The winner of a tournament: the lowest score, whichever way each loop ended.
  auto winner = [](const std::vector<FixedPointResult>& runs, auto score) -> const FixedPointResult* {
    const FixedPointResult* win = nullptr;
    for (const auto& a : runs) {
      if (!win || score(a) < score(*win)) win = &a;
    }
    return win;
  };
  std::vector<FixedPointResult> adaptive;
  if (!tournament(screening, false, adaptive)) return false;
  out.tournament = adaptive;
  // The winner is the lowest plain residual, on both lights. How a loop ended does not count: each loop's best step is a
  // render with a measured residual, and whether a loop ended as converged, diverged or out of steps flipped with a
  // 0.1% change of the input on white light, and with small changes of the search on TriLED.
  FixedPointResult chosen = *winner(adaptive, [](const FixedPointResult& a) { return a.residual; });
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
  if (verbose) {
    std::cout << "Auto: winner " << name_of(chosen.target_idx) << " (resid " << fixed(chosen.residual, 4) << ", "
              << to_string(chosen.status) << ")\n";
  }

  // Step 5: the clip fallback (ClipFallbackOutcome): move to the next bracket when the chosen one white-clips.
  // It reads the white-clip share and the median L*, which no ceiling enters (the ceiling counts only exc).
  static const float kCeil[3] = {255.0f, 255.0f, 255.0f};
  const ClipStats hc0 = fr.highlight_clip(chosen.target_idx, {params}, kCeil)[0];
  const float clip_0 = hc0.white_clip / 100.0f;
  out.clip_fallback_white_clip = clip_0;
  out.clip_fallback = ClipFallbackOutcome::kNotNeeded;
  if (clip_0 > CLIP_FALLBACK_CLIP_TAU && !pin_target) {  // a pinned bracket stays
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
      const ClipStats hc_adj = fr.highlight_clip(t_adj, {p_adj}, kCeil)[0];
      const float clip_adj = hc_adj.white_clip / 100.0f;
      const float dl = hc_adj.median_L - hc0.median_L;
      if (clip_adj > CLIP_FALLBACK_CLIP_RATIO * clip_0) out.clip_fallback = ClipFallbackOutcome::kClipsTooMuch;
      else if (!(CLIP_FALLBACK_DL_MIN <= dl && dl <= CLIP_FALLBACK_DL_MAX))
        out.clip_fallback = ClipFallbackOutcome::kShiftsLightness;
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
    if (verbose && (out.clip_fallback == ClipFallbackOutcome::kNotScreened ||
                    out.clip_fallback == ClipFallbackOutcome::kNotAdmissible)) {
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
  // re-solve: the loop's intent stays close). When it is not, the screening and the tournament run again on the
  // bootstrap field (the first one, which is also the field when no bracket metered) and the same windows with the
  // enclosure penalties in place of the pull toward the centre (pullback_search), and the runs are ranked by the
  // enclosure-regularised residual (tourney_score), whichever way each loop ended. The result is taken only if it clips at most 0.5 percentage point more, does not
  // darken the midtones past what the scene allows, and moves to a later bracket or raises E by more than 2%.
  // The thresholds are stated values, not derived (DESIGN.md 4.2, "Limits").
  // The pullback is skipped after the clip fallback took the next bracket, on a one-bracket bundle, and with a pinned
  // E: it only moves E or the bracket.
  if (out.clip_fallback != ClipFallbackOutcome::kTaken && bundle.targets.size() > 1 && !pin_e) {
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
      // The re-search screens the same metered brackets on the same windows (their usable range, bracket_enclosure)
      // against the same bootstrap field as the main search: it differs from it by its penalties on E outside the
      // calibrated range, not by its window or its field, so the field is not predicted again. The screening starts,
      // as the first one did, from the bootstrap renders' corrections, which the tournament has since replaced: each
      // metered bracket is rendered again at its bootstrap parameters to measure its correction (no inference). When no
      // bracket metered there are no bootstrap renders to repeat, and each bracket starts from its latest measured
      // correction: the tournament's last for the two finalists, which it rendered again (and the clip fallback's for
      // the bracket it tried, when it ran), and for every other bracket the one measured before the first screening
      // (the bootstrap's render for its own bracket, a render at its e_center for the others), which nothing has
      // rendered since. That is the best information each bracket has. No test frame reaches this case.
      if (!depths.empty()) {
        for (const auto& r : renders) {
          fr.render_preview(r.first, {r.second, boot_g, boot_b}, preview);
          fr.update_correction(r.first, {r.second, boot_g, boot_b}, preview);
        }
      }
      std::vector<FixedPointResult> res_adaptive;
      if (!tournament(screen_all(depths, mu1, sig1, true), true, res_adaptive)) return false;
      if (const FixedPointResult* win_res = winner(res_adaptive, tourney_score)) {
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
        // Taken when it clips at most 0.5 percentage point more of the crop above 242 in the worst channel, keeps the
        // midtones at L* 48 or within 6 of the first choice's, and actually moves: a later bracket, or E up by more
        // than 2%.
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

// ---- Stage 3, the profile tables: the fitted G/B input curves --------------------------------------------------
//
// The curves live in the bracket's input tables, composed at an E, g and b (apply_input_curves): the fit's, a
// replay's, or the knobs', which compose them again from the bracket's own tables (np_pipeline.cc). Every render after
// that, on the CPU or the GPU, reads one ordinary profile.

void apply_input_curves(ParsedFilmProfile& prof, const negicc::InputCurves& curves) {
  // In place: the profile is its own base (each table is composed into a temporary before it replaces the old one).
  if (!curves.active() || !prof.data.has_profile) return;
  apply_input_curves(prof, prof, curves);
}

void apply_input_curves(ParsedFilmProfile& prof, const ParsedFilmProfile& base, const negicc::InputCurves& curves) {
  for (int ch = 1; ch <= 2; ++ch) {
    const std::vector<float>& table = base.in_trc[ch];
    if (!curves.active() || !base.data.has_profile || table.size() < 2) {
      prof.in_trc[ch] = table;
      continue;
    }
    std::vector<float> composed(table.size());
    const int i = ch - 1;
    negicc::compose_input_table(table.data(), (int)table.size(), curves.ref[i], curves.u_med[i], curves.a[i],
                                curves.k[i], composed.data(), curves.u_hold(i));
    prof.in_trc[ch].swap(composed);
  }
  prof.rebind_pointers();
}

// On a fallback crop (the whole image, not a detected frame) the grid's cells at the film base's level (rebate, gaps,
// bare light: G and B within kBaseCellStops of a rebate base; a statistical base is the 99.5th percentile of the film
// itself, so its level is the thinnest picture highlights) or holder-dark (G more than kHolderCellStops below the
// base) are not picture. The indices of the cells kept, when that leaves enough picture (at least 16 cells and a
// quarter of the grid, enough to constrain the curve fit's four numbers) and drops some; else none (the whole grid).
// Both limits are assumptions: that the base varies by less than a quarter stop, and that no picture is 9 stops below
// it. The curve fit (WhiteLightFitCore::prepare) and the knobs' re-anchoring (reanchor_input_curves) both take them.
static std::vector<size_t> fallback_picture_cells(const uint16_t* grid, size_t n, const float film_base[3],
                                                  bool base_is_rebate) {
  constexpr float kBaseCellStops = 0.25f, kHolderCellStops = 9.0f;
  std::vector<size_t> kept;
  if (!(film_base[1] > 0.0f && film_base[2] > 0.0f)) return kept;
  const float base_lim[2] = {film_base[1] * std::exp2(-kBaseCellStops), film_base[2] * std::exp2(-kBaseCellStops)};
  const float holder_lim = film_base[1] * std::exp2(-kHolderCellStops);
  for (size_t i = 0; i < n; ++i) {
    const float g = grid[i * 3 + 1], b = grid[i * 3 + 2];
    if ((base_is_rebate && g >= base_lim[0] && b >= base_lim[1]) || g <= holder_lim) continue;
    kept.push_back(i);
  }
  if (kept.size() < std::max<size_t>(16, n / 4) || kept.size() == n) kept.clear();
  return kept;
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
  const AuditScope audit_scope;  // PassAudit (neg_process --check_renderer)
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
  negicc::CurveFrame cf = curve_frame_of(stats, fs.frame_matrix, opts);
  if (cf.n <= 0) return true;
  // On a fallback crop, the cells the curve fit anchored on (fallback_picture_cells): rebate and holder cells stay put
  // while picture cells move with the knobs, so counting them would move the anchors by less than the picture does.
  std::vector<uint16_t> kept_grid;
  if (opts.fallback_crop) {
    const std::vector<size_t> kept = fallback_picture_cells(cf.grid, (size_t)cf.n, cf.film_base,
                                                            opts.film_base_is_rebate);
    for (size_t i : kept) kept_grid.insert(kept_grid.end(), cf.grid + i * 3, cf.grid + i * 3 + 3);
    if (!kept.empty()) {
      cf.grid = kept_grid.data();
      cf.n = (int)kept.size();
    }
  }
  // Moved by the difference of two fresh anchors rather than replaced by the one at to, so a replay's recorded ref and
  // u_med carry over exactly where nothing moves.
  const negicc::InputCurves a0 = negicc::anchor_curves(curves, cf, from[0], from[1], from[2]);
  const negicc::InputCurves a1 = negicc::anchor_curves(curves, cf, to[0], to[1], to[2]);
  const ProfileData& pd = bundle.targets[target_idx].film_prof.data;
  for (int ch = 0; ch < 2; ++ch) {
    if (a0.ref[ch] > 0.0f && a1.ref[ch] > 0.0f) out.ref[ch] = curves.ref[ch] * (a1.ref[ch] / a0.ref[ch]);
    out.u_med[ch] = curves.u_med[ch] + (a1.u_med[ch] - a0.u_med[ch]);
    // At most 0, as input_curve_reference holds the cells it is read from: a hold point above ref would step the
    // curve there (curve_shape_ok). At 0 the held curve is the identity below ref.
    out.u_lo[ch] = std::min(curves.u_lo[ch] + (a1.u_lo[ch] - a0.u_lo[ch]), 0.0f);
    const int table_n = pd.in_trc_size[ch + 1];
    scale[ch] = shape_scale_of([&](float t) {
      return negicc::curve_shape_ok(t * out.a[ch], t * out.k[ch], out.u_med[ch], out.ref[ch], table_n, max_dev_stops,
                                    out.u_hold(ch));
    });
    out.a[ch] *= scale[ch];
    out.k[ch] *= scale[ch];
  }
  return true;
}

// The tone stage's plan on the render the conversion exports: the preview of profile pd at params (the search's units)
// at DINOv3's size, one inference on it for the tone heads' input, and the heads' predictions. features holds the
// search's own tone features on entry (read only if that inference fails) and the ones the heads read on return.
static void plan_tone_on_render(SearchFrame& fr, const ProfileData& pd, const ConversionParams& params,
                                const ToneHead* tone_head, int encoding, std::vector<float>& features,
                                ToneStagePlan& tp) {
  std::vector<uint8_t> pv;
  int pw = 0, ph = 0;
  if (tone_head && encoding == ENCODE_SRGB) {
    fr.render_net_preview(pd, params, pv, pw, ph);
    std::vector<float> mu_f, sg_f, tf;
    if (fr.predict_net(pv, pw, ph, mu_f, sg_f, tf) && tf.size() == (size_t)ToneHead::kInDim) {
      features.swap(tf);
      tp.features = "final";
    }
  }
  if (!tone_head) {
    tp.skipped = "off";
  } else if (encoding != ENCODE_SRGB) {
    tp.skipped = "encoding";
  } else if (features.size() != (size_t)ToneHead::kInDim ||
             !tone_head->run(features.data(), tp.pred_mean, tp.pred_sd)) {
    tp.skipped = "features";
  } else {
    tp.preview.swap(pv);
  }
}

// The white-light fit on plain data (fit_white_light_curves): the steps after the frame and its intent field are
// known, in the order run() takes them, and the state they share. neg_process's WhiteLightFit and negicc-station
// both run it.
class WhiteLightFitCore {
 public:
  WhiteLightFitCore(const WhiteLightFitData& d, const negicc::CurveFitOptions& fit_opts,
                    const WhiteLightRolloff& rolloff, WhiteLightFitResult& out)
      : d_(d), fit_opts_(fit_opts), rolloff_(rolloff), out_(out) {}

  bool run() {
    // Pinned parameters: g and b stay at the search's, which held them, and E is neither solved per candidate nor
    // re-metered.
    if (d_.pin_mask & PIN_G) fit_opts_.g_bound_stops = 0.0f;
    if (d_.pin_mask & PIN_B) fit_opts_.b_bound_stops = 0.0f;
    if (pin_e()) fit_opts_.solve_e = false;
    if (!prepare()) return false;
    fit_and_remeter();
    compose_curves();
    record();
    return true;
  }

 private:
  bool pin_e() const { return (d_.pin_mask & PIN_E) != 0; }

  // The export's roll-off on profile p (the bracket's tables composed with curves c) at (e, g, b), as the
  // conversion measures it (sRGB only).
  float rolloff_at(const negicc::ProfileData& p, const negicc::InputCurves& c, float e, float g, float b,
                   float* q995) const {
    if (d_.encoding != ENCODE_SRGB) return 1.0f;
    return rolloff_(p, c, e, g, b, q995);
  }

  // ---- 0. Prepare: the curves' anchor, the cells the fit uses, Auto's render and lightness, E's bounds, and the
  // scene-neutral anchors ----
  bool prepare() {
    const size_t n = (size_t)d_.n;
    if (d_.mu.size() != n * 3 || d_.sigma.size() != n * 3 || n == 0 || !d_.prof || !d_.grid || !d_.merged) {
      std::cerr << "ERROR! The curve fit's intent field does not match the frame's grid" << std::endl;
      return false;
    }
    static_cast<negicc::ProfileData&>(base_) = *d_.prof;
    out_.options = fit_opts_;
    mu_ = d_.mu;
    sigma_ = d_.sigma;
    std::vector<float> neutral_p = d_.neutral_p;
    // The curves' anchor on this frame: the film base given, else the brightest grid cell per channel (of the whole
    // grid, before any cell is left out below).
    cf_ = negicc::CurveFrame{};
    cf_.merged = d_.merged;
    cf_.knee = AUTO_KNEE;
    if (d_.has_film_base) {
      for (int c = 0; c < 3; ++c) cf_.film_base[c] = d_.film_base[c];
    } else {
      for (size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) cf_.film_base[c] = std::max(cf_.film_base[c], (float)d_.grid[i * 3 + c]);
    }
    cf_.grid = d_.grid;
    cf_.n = (int)n;
    // On a fallback crop the cells that are not picture (fallback_picture_cells) are left out of the fit, the re-meter
    // and the losses, with their intent cells.
    fit_grid_ = d_.grid;
    nf_ = n;
    out_.cells_total = (int)n;
    std::vector<size_t> kept_idx;
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
    if (d_.fallback_crop) {
      std::vector<size_t> try_idx = fallback_picture_cells(d_.grid, n, cf_.film_base, d_.film_base_is_rebate);
      for (size_t i : try_idx) kept_grid_.insert(kept_grid_.end(), d_.grid + i * 3, d_.grid + i * 3 + 3);
      if (!try_idx.empty()) {
        out_.cells_excluded = (int)(n - kept_grid_.size() / 3);
        nf_ = kept_grid_.size() / 3;
        fit_grid_ = kept_grid_.data();
        kept_idx.swap(try_idx);
        filter_intent(mu_, sigma_);
        if (neutral_p.size() == n) {
          std::vector<float> kp;
          kp.reserve(kept_idx.size());
          for (size_t i : kept_idx) kp.push_back(neutral_p[i]);
          neutral_p.swap(kp);
        }
        cf_.grid = fit_grid_;
        cf_.n = (int)nf_;
        if (d_.verbose) {
          std::cout << "Curves: fallback crop, " << out_.cells_excluded << " of " << n
                    << " grid cells at the film base's level or holder-dark left out\n";
        }
      }
    }
    neutral_p_ = neutral_p;
    if (neutral_p_.size() == nf_) cf_.neutral_p = neutral_p_.data();
    // The intent field over the fit's cells for the fit's fifth residual, and each cell's 4 x 4 block of the grid.
    field_.mu = mu_.data();
    field_.sigma = sigma_.data();
    field_.bx.clear();
    field_.by.clear();
    for (size_t k = 0; k < nf_; ++k) {
      const size_t i = kept_idx.empty() ? k : kept_idx[k];
      field_.bx.push_back((int)(i % d_.grid_w) / 4);
      field_.by.push_back((int)(i / d_.grid_w) / 4);
    }
    field_.nbx = (d_.grid_w + 3) / 4;
    // The export's render of the grid: its encoding and the roll-off it measures, which only the sRGB encoding
    // applies, so the median L* the re-meter matches is the one the export renders.
    out_.encoding = d_.encoding;
    q995_auto_ = 0.0f;
    out_.x_hi_auto = rolloff_at(base_, negicc::InputCurves{}, d_.e, d_.g, d_.b, &q995_auto_);
    cfg_auto_ = make_pipeline_config(base_, AUTO_KNEE, out_.x_hi_auto, d_.encoding);
    // Auto's grid median L* on the bracket's own profile, as the export renders Auto's result: the lightness the fit
    // keeps (within kIntentMinL..kIntentMaxL).
    target_L_ = negicc::grid_median_L(cfg_auto_, d_.merged, d_.e, d_.g, d_.b, fit_grid_, (int)nf_);
    // A pinned E keeps its own lightness, whatever it is.
    fit_target_L_ = pin_e() ? target_L_ : std::clamp(target_L_, kIntentMinL, kIntentMaxL);
    // Auto's loss parts on the same render; both before step 3 replaces the profile's tables (cfg points at them).
    out_.parts_auto = negicc::grid_intent_loss_parts(cfg_auto_, d_.merged, d_.e, d_.g, d_.b,
                                                     fit_grid_, (int)nf_, mu_.data(), sigma_.data());
    table_n_[0] = base_.in_trc_size[1];
    table_n_[1] = base_.in_trc_size[2];

    // The bounds of E for the fit and the re-meter: the bracket's calibrated range (its enclosure, which the search
    // measured) widened to hold Auto's E with a margin that scales with the gain bounds (0.1 + 0.5 * max(s_g, s_b)
    // stops), so a gain change has E room to hold Auto's median L*; without a range, 2 stops either side of Auto's E.
    const float s_gain = std::max(fit_opts_.g_bound_stops, fit_opts_.b_bound_stops);
    const float margin = std::exp2(0.1f + 0.5f * s_gain);
    if (d_.e_range_lo > 0.0f && d_.e_range_hi >= d_.e_range_lo) {
      out_.e_lo = std::min(d_.e_range_lo, d_.e / margin);
      out_.e_hi = std::max(d_.e_range_hi, d_.e * margin);
    } else {
      out_.e_lo = d_.e * 0.25f;
      out_.e_hi = d_.e * 4.0f;
    }

    e_start_ = d_.e;
    out_.intent_lightness_failed = (fit_target_L_ != target_L_);
    if (out_.intent_lightness_failed) {
      out_.fallback_target_L = fit_target_L_;
      out_.e_lo = std::min(out_.e_lo, d_.e * 0.25f);
      out_.e_hi = std::max(out_.e_hi, d_.e * 4.0f);
      e_start_ = negicc::solve_e_for_median_L(cfg_auto_, cf_, negicc::InputCurves{}, d_.e, out_.e_lo, out_.e_hi,
                                              d_.g, d_.b, fit_target_L_, &out_.fallback_clipped);
      out_.fallback_e_moved = std::fabs(e_start_ - d_.e) > 1e-6f * std::max(1.0f, std::fabs(d_.e));
      out_.x_hi_auto = rolloff_at(base_, negicc::InputCurves{}, e_start_, d_.g, d_.b, &q995_auto_);
      cfg_auto_ = make_pipeline_config(base_, AUTO_KNEE, out_.x_hi_auto, d_.encoding);
      out_.e_lo = std::min(out_.e_lo, e_start_ / margin);
      out_.e_hi = std::max(out_.e_hi, e_start_ * margin);
    }
    m_neu_ = std::make_unique<negicc::JointCurveModel>(cfg_auto_, cf_, mu_.data(), sigma_.data());
    m_neu_->set_scene_neutral(e_start_, d_.g, d_.b);
    out_.neutral_auto = m_neu_->neutral_stats_on(cfg_auto_, e_start_, d_.g, d_.b);
    out_.neutral_anchors = m_neu_->has_anchors();  // as the fit's own selection: the same frame, E, g and b
    x_ = out_.x_hi_auto;
    q995_ = q995_auto_;
    e_new_ = e_start_;
    return true;
  }

  // Auto's own result, no curves (the fallback of the steps below).
  void fall_back(const char* why) {
    for (int ch = 0; ch < 2; ++ch) {
      out_.fit.curves.a[ch] = out_.fit.curves.k[ch] = 0.0f;
      out_.shape_scale_ch[ch] = 0.0f;
    }
    out_.fit.g = d_.g;
    out_.fit.b = d_.b;
    e_new_ = e_start_;
    x_ = out_.x_hi_auto;
    q995_ = q995_auto_;
    out_.fallback = why;
  }

  // ---- 1. The fit: the joint scene-neutral colour and crossover solver for (g, b, a_G, a_B) at fit_target_L, run
  // at the roll-off x_fit; then 2. the re-meter and the shape check, and the fit again (up to kMaxFitPasses) while
  // the re-metered roll-off differs from the one it ran with ----
  void fit_and_remeter() {
    constexpr int kMaxFitPasses = 3;
    float x_fit = out_.x_hi_auto;
    out_.remeter_passes = 0;
    out_.fit_passes = 0;
    for (int pass = 0; pass < kMaxFitPasses; ++pass) {
      ++out_.fit_passes;
      if (!fit_pass(x_fit)) break;
      if (!remeter_and_check_shape(x_fit)) break;
      if (x_ == x_fit) break;
      x_fit = x_;
    }
    out_.shape_scale = std::min(out_.shape_scale_ch[0], out_.shape_scale_ch[1]);
  }

  // One joint fit at roll-off x_fit; false (Auto's result taken) when it found no feasible point.
  bool fit_pass(float x_fit) {
    out_.x_hi_fit = x_fit;
    out_.fallback = nullptr;
    const float e_lo_fit = std::min(e_start_, out_.e_lo * 1.001f);
    const float e_hi_fit = std::max(e_start_, out_.e_hi / 1.001f);
    const negicc::PipelineConfig cfg = make_pipeline_config(base_, AUTO_KNEE, x_fit, d_.encoding);
    out_.fit = negicc::fit_input_curves_joint(cfg, cf_, field_, e_start_, d_.g, d_.b,
                                              fit_target_L_, table_n_, fit_opts_, e_lo_fit, e_hi_fit);
    if (!out_.fit.ok) {
      // Fewer than 8 cells in range, or no feasible start: Auto's colour (its g and b, no curves).
      negicc::JointCurveModel mdl(cfg, cf_);
      out_.fit.curves = mdl.anchored(negicc::InputCurves{}, e_start_, d_.g, d_.b);
      out_.fit.g = d_.g;
      out_.fit.b = d_.b;
      out_.fit.e = e_start_;
      out_.fit.feasible = true;
      out_.fallback = out_.fit.skipped;  // reported only: the re-meter still runs on Auto's values
      if (d_.verbose) std::cout << "Curves: no fit (" << out_.fit.skipped << "): Auto's g and b, no curves\n";
    }
    const negicc::JointCurveFit& jf = out_.fit;
    if (d_.verbose) {
      const negicc::InputCurves& c = out_.fit.curves;
      std::cout << "Curves: G a " << fixed(c.a[0], 4) << " k " << fixed(c.k[0], 4) << ", B a " << fixed(c.a[1], 4)
                << " k " << fixed(c.k[1], 4) << "; E " << fixed(jf.e, 4) << " g " << fixed(out_.fit.g, 4) << " b "
                << fixed(out_.fit.b, 4) << ", loss " << fixed((float)out_.fit.loss_before, 3) << " -> "
                << fixed((float)out_.fit.loss_after, 3) << " (" << jf.iterations << " steps, " << jf.evaluations
                << " evaluations, " << jf.renders << " renders for E, roll-off " << fixed(x_fit, 3) << ")\n";
    }
    if (!jf.feasible) {
      fall_back("clip");
      return false;
    }
    negicc::CombinedProfile c0;
    c0.build(base_, out_.fit.curves);
    negicc::PipelineConfig c0cfg = cfg;
    c0cfg.profile = c0.data;
    out_.parts_fit = negicc::grid_intent_loss_parts(c0cfg, d_.merged, jf.e, out_.fit.g, out_.fit.b, fit_grid_,
                                                    (int)nf_, mu_.data(), sigma_.data());
    return true;
  }

  // E solved on the real combined tables around e_centre, alternated with the roll-off measured there until the
  // roll-off holds still (at most kMaxPasses).
  void remeter(float e_centre) {
    constexpr int kMaxPasses = 6;
    out_.remeter_converged = false;
    for (int pass = 1; pass <= kMaxPasses; ++pass) {
      ++out_.remeter_passes;
      if (pin_e()) {
        e_new_ = e_centre;  // held: only the roll-off is measured on the combined tables
        clipped_ = false;
      } else {
        e_new_ = negicc::solve_e_for_median_L(make_pipeline_config(base_, AUTO_KNEE, x_, d_.encoding), cf_,
                                              out_.fit.curves, e_centre, out_.e_lo, out_.e_hi, out_.fit.g, out_.fit.b,
                                              fit_target_L_, &clipped_);
      }
      const negicc::InputCurves anchored = negicc::anchor_curves(out_.fit.curves, cf_, e_new_, out_.fit.g, out_.fit.b);
      comb_.build(base_, anchored);
      float q = 0.0f;
      const float x_new = rolloff_at(comb_.data, anchored, e_new_, out_.fit.g, out_.fit.b, &q);
      q995_ = q;
      if (x_new == x_) {
        out_.remeter_converged = true;
        break;
      }
      x_ = x_new;
    }
  }

  // The factors t[ch] that keep each curve's shape at E e; true when both are 1.
  bool shape_scales(const negicc::InputCurves& shape, float e, float t[2]) const {
    const negicc::InputCurves c = negicc::anchor_curves(shape, cf_, e, out_.fit.g, out_.fit.b);
    bool all = true;
    for (int ch = 0; ch < 2; ++ch) {
      t[ch] = shape_scale_of([&](float s) {
        return negicc::curve_shape_ok(s * c.a[ch], s * c.k[ch], c.u_med[ch], c.ref[ch], table_n_[ch],
                                      fit_opts_.max_dev_stops, c.u_hold(ch));
      });
      all = all && t[ch] == 1.0f;
    }
    return all;
  }

  // ---- 2. The re-meter after a fit pass: E on the real combined tables, each channel's curve scaled down where its
  // shape fails at the re-metered anchor (up to four rounds); false (Auto's result taken) when E leaves its bounds or
  // the shape still fails ----
  bool remeter_and_check_shape(float x_fit) {
    x_ = x_fit;
    remeter(out_.fit.e);
    out_.shape_scale_ch[0] = out_.shape_scale_ch[1] = 1.0f;
    bool shape_ok = false;
    for (int round = 0; round < 4 && !clipped_; ++round) {
      float t[2];
      if (shape_scales(out_.fit.curves, e_new_, t)) {
        shape_ok = true;
        break;
      }
      if (round == 3) break;
      for (int ch = 0; ch < 2; ++ch) {
        out_.fit.curves.a[ch] *= t[ch];
        out_.fit.curves.k[ch] *= t[ch];
        out_.shape_scale_ch[ch] *= t[ch];
      }
      remeter(e_new_);
    }
    if (clipped_ || !shape_ok) {
      fall_back(clipped_ ? "clip" : "shape");
      return false;
    }
    return true;
  }

  // ---- 3. The curves anchored at the parameters the export renders with (the re-metered E, the fitted g and b), and
  // the bracket's tables composed with them (the caller composes them into its own profile the same way) ----
  void compose_curves() {
    out_.fit.curves = negicc::anchor_curves(out_.fit.curves, cf_, e_new_, out_.fit.g, out_.fit.b);
    comb_final_.build(base_, out_.fit.curves);
    out_.x_hi = x_;  // measured on this profile at these parameters (step 2), or Auto's on the fallback
    out_.x_hi_q995 = q995_;
    out_.rolloff_measured = d_.encoding == ENCODE_SRGB;
  }

  // ---- 4. Record: the export's parameters, lightness, losses and neutral statistics ----
  void record() {
    negicc::PipelineConfig cfg_comb = make_pipeline_config(base_, AUTO_KNEE, out_.x_hi, d_.encoding);
    cfg_comb.profile = comb_final_.data;
    out_.params = {e_new_, out_.fit.g, out_.fit.b, out_.x_hi};
    out_.target_median_L = fit_target_L_;
    out_.median_L = negicc::grid_median_L(cfg_comb, d_.merged, e_new_, out_.fit.g, out_.fit.b, fit_grid_, (int)nf_);
    out_.parts_export = negicc::grid_intent_loss_parts(cfg_comb, d_.merged, e_new_, out_.fit.g, out_.fit.b, fit_grid_,
                                                       (int)nf_, mu_.data(), sigma_.data());
    out_.neutral_export = m_neu_->neutral_stats_on(cfg_comb, e_new_, out_.fit.g, out_.fit.b);
    out_.loss_export = out_.parts_export.total();
    if (d_.verbose) {
      std::cout << "Curves: E re-metered " << fixed(d_.e, 4) << " -> " << fixed(e_new_, 4) << " (grid median L* "
                << fixed(out_.median_L, 2) << ", Auto's " << fixed(target_L_, 2) << "); g "
                << fixed(out_.params.g, 4) << " b " << fixed(out_.params.b, 4)
                << (!out_.fallback ? "" : "; no curves, Auto's result (") << (out_.fallback ? out_.fallback : "")
                << (out_.fallback ? ")" : "") << "\n";
    }
  }

  const WhiteLightFitData& d_;
  negicc::CurveFitOptions fit_opts_;  // the caller's, with the pins applied (run)
  const WhiteLightRolloff& rolloff_;
  WhiteLightFitResult& out_;
  ProfileData base_;                  // the bracket's own profile (its tables are the caller's)
  std::vector<float> mu_, sigma_, neutral_p_;
  negicc::CurveFrame cf_;
  std::vector<uint16_t> kept_grid_;
  const uint16_t* fit_grid_ = nullptr;
  size_t nf_ = 0;
  float q995_auto_ = 0.0f;
  negicc::PipelineConfig cfg_auto_{};
  float target_L_ = 0.0f, fit_target_L_ = 0.0f;
  int table_n_[2] = {0, 0};
  float e_start_ = 0.0f;
  std::unique_ptr<negicc::JointCurveModel> m_neu_;
  float x_ = 1.0f, q995_ = 0.0f, e_new_ = 0.0f;  // the re-metered roll-off, its q995 and E
  bool clipped_ = false;
  negicc::CombinedProfile comb_;        // the re-meter's combined tables
  negicc::CombinedProfile comb_final_;  // the result's combined tables
  negicc::JointFitField field_;         // the intent field the fit's fifth residual reads
};

// neg_process's white-light fit (run_white_light_fit) on the frame the search ran on: the search's intent field
// (predicted again where the search moved its result afterwards), the fit on that data (WhiteLightFitCore), the
// bracket's tables composed with the curves in place, and the tone stage's plan.
class WhiteLightFit {
 public:
  WhiteLightFit(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame, double scan_shutter_s,
                double scan_iso, const ParamSearchOptions& opts, const ParamSearchResult& res,
                const negicc::CurveFitOptions& fit_opts, int encoding, const ToneHead* tone_head,
                WhiteLightFitResult& out)
      : bundle_(bundle), frame_(frame), scan_shutter_s_(scan_shutter_s), scan_iso_(scan_iso), opts_(opts), res_(res),
        fit_opts_(fit_opts), encoding_(encoding), tone_head_(tone_head), out_(out),
        fr_(bundle, engine, opts.verbose, opts.max_steps, scratch_) {}

  bool run() {
    // White light only (see the --fit_curves gate in np_pipeline.cc): a TriLED bundle's curves are the measured ones.
    if (bundle_.has_crosstalk) {
      std::cerr << "ERROR! Fitted input curves are for white-light bundles: a TriLED bundle converts through its "
                   "measured curves" << std::endl;
      return false;
    }
    const int idx = res_.target_idx;
    if (idx < 0 || idx >= (int)bundle_.targets.size()) {
      std::cerr << "ERROR! Fitted input curves need the bracket of a finished search" << std::endl;
      return false;
    }
    const Clock::time_point t_fit0 = Clock::now();
    fr_.set_renderer(opts_.renderer);
    fr_.set_shoulder(1.0f, true);
    fr_.set_frame(frame_, opts_.crop_ratio);
    const FrameStats& stats = fr_.stats();
    const FrameScale scale = frame_scale(bundle_, (float)scan_shutter_s_, (float)scan_iso_, opts_);
    fr_.set_frame_matrix(scale.frame_matrix, scale.e_norm);
    const size_t n = stats.grid.size() / 3;
    if (res_.intent_grid_w != stats.grid_w || res_.intent_grid_h != stats.grid_h ||
        res_.intent_mu.size() != n * 3 || res_.intent_sigma.size() != n * 3 || n == 0) {
      std::cerr << "ERROR! The search's intent field does not match the frame's grid" << std::endl;
      return false;
    }
    out_.params_before = res_.params;

    // The fit's data. The search's matrix and E: white-light bundles search on M_frame with E in this frame's units.
    WhiteLightFitData d;
    ParsedFilmProfile& prof = bundle_.targets[idx].film_prof;
    d.prof = &prof.data;
    d.merged = fr_.input_matrix();
    d.grid = stats.grid.data();
    d.n = (int)n;
    d.grid_w = stats.grid_w;
    d.has_film_base = opts_.film_base_rgb_d || opts_.film_base_rgb;
    for (int c = 0; c < 3 && d.has_film_base; ++c) {
      d.film_base[c] = opts_.film_base_rgb_d ? (float)opts_.film_base_rgb_d[c] : opts_.film_base_rgb[c];
    }
    d.fallback_crop = opts_.fallback_crop;
    d.film_base_is_rebate = opts_.film_base_is_rebate;
    d.e = res_.params.e;
    d.g = res_.params.g;
    d.b = res_.params.b;
    d.e_range_lo = res_.e_range_lo;
    d.e_range_hi = res_.e_range_hi;
    d.pin_mask = opts_.pin_mask;
    d.verbose = opts_.verbose;
    d.encoding = encoding_;
    // The intent field the fit scores against: the search's, predicted from the preview at intent_params. The
    // highlight pullback (and the clip fallback) move E, or the bracket's parameters, after that preview, so there the
    // field is predicted again from a preview at the search's final parameters on the bracket's own profile, so the
    // fit does not run at one E against a field predicted at another. One DINOv3 inference, only then. Its L* is held
    // to the frame's lightness target (the search's field's L*), as the fixed point holds it, so the fit's L* losses
    // are measured against one reference whether or not the field was predicted again; a* and b* keep the new
    // prediction.
    d.mu = res_.intent_mu;
    d.sigma = res_.intent_sigma;
    d.neutral_p = res_.intent_neutral_p;                  // the neutral head's p of the same preview, if it ran
    std::vector<float> tone_features = res_.intent_tone_features;  // the tone heads' input from it, if given
    {
      const ConversionParams& ip = res_.intent_params;  // the search's own units, which are this frame's on white light
      auto moved = [](float a, float b) { return std::fabs(a - b) > 1e-6f * std::max(1.0f, std::fabs(b)); };
      if (moved(ip.e, res_.params.e) || moved(ip.g, res_.params.g) || moved(ip.b, res_.params.b) ||
          moved(ip.shoulder, res_.params.shoulder)) {
        std::vector<uint8_t> preview;
        fr_.render_preview(idx, res_.params, preview);
        if (!fr_.predict(preview, d.mu, d.sigma)) return false;
        for (size_t i = 0; i < n; ++i) {
          d.mu[i * 3] = res_.intent_mu[i * 3];
          d.sigma[i * 3] = res_.intent_sigma[i * 3];
        }
        d.neutral_p = fr_.last_neutral_p();
        tone_features = fr_.last_tone_features();
        out_.intent_repredicted = true;
        out_.guard_p_source = "refit";
      }
    }
    // The export's roll-off, as neg_process measures every conversion's.
    const WhiteLightRolloff rolloff = [&](const negicc::ProfileData& p, const negicc::InputCurves&, float e, float g,
                                          float b, float* q995) {
      return measure_rolloff_x_hi(bundle_, p, frame_, scan_shutter_s_, scan_iso_, e, g, b, opts_.crop_ratio, q995,
                                  opts_.film_base_rgb_d, opts_.renderer);
    };
    if (!fit_white_light_curves(d, fit_opts_, rolloff, out_)) return false;
    // The bracket's tables composed with the curves in place (apply_input_curves): every later use of the bracket
    // renders through them.
    // A pass that throws after this (the tone plan's render) leaves the bracket's own tables restored.
    const std::vector<float> own_g = prof.in_trc[1], own_b = prof.in_trc[2];
    apply_input_curves(prof, out_.fit.curves);
    // The tone stage's plan: the heads read the image the stage will be applied to, the final state's preview, at one
    // inference (the search's intent preview has the pre-fit colour, and is the fallback when that inference fails).
    out_.tone.features = "search";
    try {
      plan_tone_on_render(fr_, prof.data, out_.params, tone_head_, encoding_, tone_features, out_.tone);
    } catch (...) {
      prof.in_trc[1] = own_g;
      prof.in_trc[2] = own_b;
      prof.rebind_pointers();
      throw;
    }
    out_.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_fit0).count();
    return true;
  }

 private:
  ProfileBundle& bundle_;
  const negicc::RgbView& frame_;
  const double scan_shutter_s_, scan_iso_;
  const ParamSearchOptions& opts_;
  const ParamSearchResult& res_;
  const negicc::CurveFitOptions fit_opts_;
  const int encoding_;
  const ToneHead* tone_head_;
  WhiteLightFitResult& out_;
  ParamSearchResult scratch_;  // the inferences and timings of the frame below; not reported
  SearchFrame fr_;
};

bool fit_white_light_curves(const WhiteLightFitData& data, const negicc::CurveFitOptions& fit_opts,
                            const WhiteLightRolloff& rolloff, WhiteLightFitResult& out) {
  return WhiteLightFitCore(data, fit_opts, rolloff, out).run();
}

bool run_white_light_fit(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                         double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                         const ParamSearchResult& res, const negicc::CurveFitOptions& fit_opts,
                         int encoding, const ToneHead* tone_head,
                         WhiteLightFitResult& out) {
  const AuditScope audit_scope;  // PassAudit (neg_process --check_renderer)
  return WhiteLightFit(bundle, engine, frame, scan_shutter_s, scan_iso, opts, res, fit_opts, encoding, tone_head, out)
      .run();
}
bool plan_auto_tone_stage(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                          double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                          const ParamSearchResult& res, int encoding, const ToneHead& tone_head, ToneStagePlan& out) {
  const AuditScope audit_scope;  // PassAudit (neg_process --check_renderer)
  const int idx = res.target_idx;
  if (idx < 0 || idx >= (int)bundle.targets.size()) {
    std::cerr << "ERROR! The tone stage needs the bracket of a finished search" << std::endl;
    return false;
  }
  out = ToneStagePlan{};
  ParamSearchResult scratch;  // the inference and timings of the frame below; not reported
  SearchFrame fr(bundle, engine, opts.verbose, opts.max_steps, scratch);
  fr.set_renderer(opts.renderer);
  fr.set_shoulder(1.0f, true);
  {
    // The frame statistics: the search's own (ParamSearchResult::frame_stats), when it computed them from this same
    // view, crop ratio and network size; computed here otherwise.
    const bool same = res.frame_stats && res.frame_stats_view.same_view(frame) &&
                      res.frame_stats_crop_ratio == opts.crop_ratio && res.frame_stats_max_size == engine.max_size() &&
                      res.frame_stats_patch == engine.patch_size();
    if (same) {
      fr.set_frame(frame, compute_inner_crop(frame.width, frame.height, opts.crop_ratio),
                   *std::static_pointer_cast<const FrameStats>(res.frame_stats));
    } else {
      fr.set_frame(frame, opts.crop_ratio);
    }
  }
  const FrameScale scale = frame_scale(bundle, (float)scan_shutter_s, (float)scan_iso, opts);
  fr.set_frame_matrix(scale.frame_matrix, scale.e_norm);
  const ProfileData& pd = bundle.targets[idx].film_prof.data;
  // The export's roll-off at the search's result, as the conversion measures it (only the sRGB encoding applies one).
  ConversionParams p = res.params;
  float q995 = 0.0f;
  p.shoulder = encoding == ENCODE_SRGB
                   ? measure_rolloff_x_hi(bundle, pd, frame, scan_shutter_s, scan_iso, p.e, p.g, p.b, opts.crop_ratio,
                                          &q995, opts.film_base_rgb_d, opts.renderer)
                   : 1.0f;
  // The preview renders in the search's units: a TriLED search runs on M_frame / e_norm with E in the reference's
  // units, while res.params.e is in this frame's (white light searches in the frame's units).
  if (bundle.has_crosstalk) p.e *= fr.e_norm();
  std::vector<float> features = res.intent_tone_features;
  plan_tone_on_render(fr, pd, p, &tone_head, encoding, features, out);
  return true;
}

float print_floor_factor(const ProfileBundle& bundle, const negicc::RgbView& frame, float crop_ratio,
                         const double fb[3]) {
  const InnerCrop c = compute_inner_crop(frame.width, frame.height, crop_ratio);
  const double* x = bundle.crosstalk_matrix_d;
  std::vector<float> v[3];
  uint16_t px[3];
  for (int y = c.y0; y < c.y0 + c.h; y += 4)
    for (int xx = c.x0; xx < c.x0 + c.w; xx += 4) {
      frame.sample(xx, y, px);
      for (int r = 0; r < 3; ++r)
        if (fb[r] > 0.0)
          v[r].push_back((float)((x[r * 3] * px[0] + x[r * 3 + 1] * px[1] + x[r * 3 + 2] * px[2]) / fb[r]));
    }
  float k = 1.0f;
  for (int r = 0; r < 3; ++r) {
    if (v[r].empty()) continue;   // also a channel without a positive base
    const size_t q = std::min(v[r].size() - 1, (size_t)(negicc::kBasePercentile / 100.0 * v[r].size()));
    std::nth_element(v[r].begin(), v[r].begin() + q, v[r].end());
    k = std::max(k, v[r][q]);
  }
  return k;
}

float raise_base_to_print_floor(const ProfileBundle& bundle, const negicc::RgbView& frame, float crop_ratio,
                                const double* found, double scan_shutter_s, double scan_iso, double base_in[3],
                                double base_out[3]) {
  if (found) {
    for (int c = 0; c < 3; ++c) base_in[c] = found[c];
  } else {
    const double fb_t = bundle.film_base_shutter.empty() ? 0.125 : shutter_to_seconds_d(bundle.film_base_shutter);
    const double ratio = (scan_shutter_s * (scan_iso / 100.0)) / (fb_t * (bundle.film_base_iso / 100.0));
    for (int c = 0; c < 3; ++c) base_in[c] = bundle.film_base_avg_d[c] * ratio;
  }
  const float k = print_floor_factor(bundle, frame, crop_ratio, base_in);
  for (int c = 0; c < 3; ++c) base_out[c] = base_in[c] * k;
  return k;
}

// What the print searches of one frame share across its materials (search_print_continuous), taken once.
struct PrintFrame {
  ProfileBundle pb;   // the bundle with one target, which carries the material's print at the current A
  InnerCrop crop;
  FrameStats stats;   // over the inner crop, at the engine's grid (SearchFrame::set_frame)
  FrameScale scale;
  // The frame's shares at E, measured with the first material: E is held, and the gains are in the print's curves,
  // after the knee, so the knee sees the matrix at E alone whatever the material and its g and b.
  bool shares_measured = false;
  float above_knee = 0.0f, past_knee[3] = {0.0f, 0.0f, 0.0f}, past_cut[3] = {0.0f, 0.0f, 0.0f};
};

int nearest_bracket(const std::vector<double>& a_mid, double a) {
  int nearest = 0;
  for (size_t i = 0; i < a_mid.size(); ++i)
    if (std::fabs(a_mid[i] - a) < std::fabs(a_mid[nearest] - a)) nearest = (int)i;
  return nearest;
}

// One material's search on the shared frame pf, its time counted from t0.
static bool search_print_material(PrintFrame& pf, const ProfileBundle& bundle, const ParsedFilmProfile& print_base,
                                  const ContinuousPrint& in, DinoV3Engine& engine, const negicc::RgbView& frame,
                                  const ParamSearchOptions& opts, Clock::time_point t0, ParamSearchResult& out,
                                  ContinuousPrintResult& cr) {
  cr = ContinuousPrintResult{};
  constexpr double kContinuousStep = 0.1, kContinuousTol = 0.001;
  ProfileBundle& pb = pf.pb;
  pb.targets[0].film_prof = print_base;
  pb.targets[0].film_prof.rebind_pointers();
  ParsedFilmProfile& prof = pb.targets[0].film_prof;
  // The print at A: its input tables at A (cur_a), without gains, in prof. Every render and optimiser pass at (A, g, b)
  // puts g and b into the tables at cur_a, after the knee, with the matrix at E alone (gain_curves), as the conversion
  // renders the print, so the search scores the picture the conversion writes.
  double cur_a = 0.0;
  auto set_curves_at = [&](double a) {
    cur_a = a;
    in.trc_at(a, nullptr, prof.in_trc);
    for (int c = 0; c < 3; ++c) prof.data.in_trc_size[c] = (int)prof.in_trc[c].size();
    prof.rebind_pointers();
  };
  const GainCurves gain_curves = [&](float g, float b, std::vector<float> trc[3]) {
    const double gains[3] = {1.0, g, b};
    in.trc_at(cur_a, gains, trc);
  };
  out = ParamSearchResult{};
  SearchFrame fr(pb, engine, opts.verbose, opts.max_steps, out);
  fr.set_renderer(opts.renderer);
  fr.set_gain_curves(&gain_curves);
  fr.set_shoulder(1.0f, false);   // a print keeps no highlight shoulder (the paper rolls off to its white)
  fr.set_frame(frame, pf.crop, pf.stats);
  const FrameScale& scale = pf.scale;
  fr.set_frame_matrix(scale.frame_matrix, scale.e_norm);
  const float e_norm = fr.e_norm();
  out.e_norm = e_norm;
  for (int i = 0; i < 9; ++i) out.input_matrix[i] = scale.frame_matrix[i] / e_norm;
  for (int r = 0; r < 3; ++r) {
    out.row_scale[r] = (scale.fb[r] > 0 ? (pb.normalization_target / scale.fb[r]) * scale.ratio_frame : 1.0f) / e_norm;
  }
  // E = 1 relative to the frame's own film base: the base reads n0, as the brackets were measured. The search runs on
  // M_frame / e_norm, which in green is the reference scan's matrix whatever the frame's base, so E = 1 there would put
  // the base at n0 / e_norm and make the print depend on how long the scan was exposed; E = e_norm puts it at n0. The
  // result's E (E / e_norm) is then 1. On white light the search runs on M_frame itself
  // (SearchFrame::set_frame_matrix), so E = 1 puts the base at n0.
  const bool white_light = !bundle.has_crosstalk;
  const float E = white_light ? 1.0f : e_norm;
  float m[9];
  negicc::candidate_matrix(fr.input_matrix(), E, 1.0f, 1.0f, m);
  // The frame's shares at E (the matrix at E alone, every 4th pixel each way): of the inner crop above the knee in any
  // channel (at the bootstrap), and per channel past the knee and past the cut (at the result, whose g and b are in the
  // curves, after the knee).
  if (!pf.shares_measured) {
    const float cut = AUTO_KNEE + (1.0f - AUTO_KNEE) * std::atanh((0.9999f - AUTO_KNEE) / (1.0f - AUTO_KNEE));
    const InnerCrop& c = fr.inner_crop();
    long above = 0, knee_n[3] = {0, 0, 0}, cut_n[3] = {0, 0, 0}, total = 0;
    uint16_t px[3];
    for (int y = c.y0; y < c.y0 + c.h; y += 4)
      for (int x = c.x0; x < c.x0 + c.w; x += 4) {
        frame.sample(x, y, px);
        float u_max = 0.0f;
        for (int r = 0; r < 3; ++r) {
          const float u = (m[r * 3] * px[0] + m[r * 3 + 1] * px[1] + m[r * 3 + 2] * px[2]) / 65535.0f;
          u_max = std::max(u_max, u);
          knee_n[r] += u > AUTO_KNEE;
          cut_n[r] += u >= cut;
        }
        above += u_max > AUTO_KNEE;
        ++total;
      }
    pf.above_knee = total ? (float)above / total : 0.0f;
    for (int r = 0; r < 3; ++r) {
      pf.past_knee[r] = total ? (float)knee_n[r] / total : 0.0f;
      pf.past_cut[r] = total ? (float)cut_n[r] / total : 0.0f;
    }
    pf.shares_measured = true;
  }
  cr.above_knee = pf.above_knee;
  for (int r = 0; r < 3; ++r) {
    cr.past_knee[r] = pf.past_knee[r];
    cr.past_cut[r] = pf.past_cut[r];
  }
  // A0, as the scene's search meters a bracket: the A at which the frame's log-average cell prints at L* 50 through
  // the print profile (bisection; L* falls as A rises).
  const FrameStats& st = fr.stats();
  const size_t n = st.grid.size() / 3;
  if (n == 0) {
    std::cerr << "ERROR! The print's search has no grid cells on this frame" << std::endl;
    return false;
  }
  double a0;
  {
    uint16_t px[3];
    for (int c = 0; c < 3; ++c) px[c] = (uint16_t)std::lround(std::clamp(st.log_avg_rgb[c], 0.0f, 1.0f) * 65535.0f);
    auto lstar = [&](double a) {
      set_curves_at(a);
      const negicc::PipelineConfig pc = make_pipeline_config(prof.data, AUTO_KNEE, 1.0f);
      return negicc::linear_srgb_to_lab(negicc::convert_pixel_display(pc, m, px)).L;
    };
    double lo = in.a_lo, hi = in.a_hi;
    for (int it = 0; it < 60; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (lstar(mid) > 50.0f) lo = mid; else hi = mid;
    }
    a0 = 0.5 * (lo + hi);
    cr.a0_at_range_end = a0 - in.a_lo < 1e-6 || in.a_hi - a0 < 1e-6;
  }
  cr.a0 = a0;
  // The bootstrap: the mean field of the renders at A0 - h, A0, A0 + h; its L* is the frame's lightness target.
  std::vector<uint8_t> preview;
  std::vector<float> mu1, sig1, mu, sg;
  std::vector<double> boot = {a0};
  if (in.boot_h > 0.0) boot = {a0 - in.boot_h, a0, a0 + in.boot_h};
  for (size_t k = 0; k < boot.size(); ++k) {
    set_curves_at(std::clamp(boot[k], in.a_lo, in.a_hi));
    fr.render_preview(0, {E, 1.0f, 1.0f}, preview);
    // The render at A0 gives the correction the screening at A0 below aims with, as each bracket's bootstrap render
    // gives the scene's screening its correction.
    if (boot[k] == a0) fr.update_correction(0, {E, 1.0f, 1.0f}, preview);
    if (!fr.predict(preview, mu, sg)) return false;
    if (k == 0) {
      mu1.assign(mu.size(), 0.0f);
      sig1.assign(sg.size(), 0.0f);
    }
    for (size_t i = 0; i < mu.size(); ++i) {
      mu1[i] += mu[i] / boot.size();
      sig1[i] += sg[i] / boot.size();
    }
  }
  {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += mu1[i * 3];
    cr.target_mean_l = (float)(s / n);
  }
  if (opts.verbose) {
    std::cout << "Print (continuous): A0 " << a0 << " (range " << in.a_lo << " to " << in.a_hi
              << "), above the knee at E = 1: " << 100.0f * cr.above_knee << " %, lightness target L* "
              << cr.target_mean_l << " from " << boot.size() << " render(s)"
              << (cr.a0_at_range_end ? " (A0 at an end of the range: bootstrap renders there share an A)" : "") << "\n";
  }
  // The fixed point over (A, g, b), from g and b screened at A0 against the bootstrap field minus the correction of
  // the render at A0 (the optimiser's wide grids, as the scene search screens its brackets).
  fr.set_pins(PIN_E, E, 1.0f, 1.0f);
  float e_bounds[3] = {E, E, E};
  ConversionParams curr{E, 1.0f, 1.0f, 1.0f};
  {
    set_curves_at(a0);
    ConversionParams scr;
    fr.search(0, mu1, sig1, nullptr, e_bounds, scr);
    curr.g = scr.g;
    curr.b = scr.b;
  }
  ConversionParams best = curr;
  double a = a0, best_a = a0;
  float best_err = INFINITY, prev_err = INFINITY;
  std::vector<float> best_mu, best_sg;
  bool stop_next = false;
  for (int it = 0; it < std::max(1, opts.max_steps); ++it) {
    set_curves_at(a);
    fr.render_preview(0, curr, preview);
    fr.update_correction(0, curr, preview);
    if (!fr.predict(preview, mu, sg)) return false;
    for (size_t i = 0; i < n; ++i) {
      mu[i * 3] = mu1[i * 3];
      sg[i * 3] = sig1[i * 3];
    }
    const float err = fr.residual(preview, mu, sg);
    cr.steps = it + 1;
    if (err < best_err) {
      best_err = err;
      best = curr;
      best_a = a;
      best_mu = mu;
      best_sg = sg;
    }
    if (opts.verbose) {
      std::cout << "  continuous step " << it + 1 << ": A " << a << " g " << curr.g << " b " << curr.b << " residual "
                << err << "\n";
    }
    if (stop_next || (std::isfinite(prev_err) && std::fabs(err - prev_err) <= TOL_REL_ERR * prev_err)) {
      cr.converged = true;
      cr.stopped_by = stop_next ? "parameters" : "residual";
      break;
    }
    prev_err = err;
    // The last step allowed: its result is scored above, and a further move would be neither rendered nor scored.
    if (it + 1 >= std::max(1, opts.max_steps)) break;
    // A by a scan of its range then golden section; at each A, g and b by the optimiser (E pinned). Every A tried is
    // aimed at the field minus the correction measured at this step's A above: update_correction keys it by the
    // print's one target, whose curves change with A, and it is not measured again at each A (that would take a
    // render each), so the line search takes the correction to change little across the A it tries.
    ConversionParams gb;
    auto loss = [&](double x, ConversionParams& p) {
      set_curves_at(x);
      return fr.search(0, mu, sg, &curr, e_bounds, p);
    };
    // Bracket the minimum: from the current A, steps of kContinuousStep downhill, within the range. When both
    // neighbours are lower, each side is marched and the lower of the two brackets kept.
    auto clampA = [&](double x) { return std::clamp(x, in.a_lo, in.a_hi); };
    struct Bracket { double lo, hi; float f; };
    auto march = [&](double x0, float f0, double dir) {
      double xm = x0, xo = clampA(x0 - dir * kContinuousStep);
      float fm = f0;
      double xn = clampA(xm + dir * kContinuousStep);
      float fn = xn == xm ? fm : loss(xn, gb);   // at an end of the range: no step to take
      while (fn < fm && xn != xm) {
        xo = xm;
        xm = xn;
        fm = fn;
        xn = clampA(xm + dir * kContinuousStep);
        if (xn == xm) break;
        fn = loss(xn, gb);
      }
      return Bracket{std::min(xo, xn), std::max(xo, xn), fm};
    };
    const double x0 = clampA(a);
    const float f0 = loss(x0, gb);
    const double xl0 = clampA(x0 - kContinuousStep), xr0 = clampA(x0 + kContinuousStep);
    // At an end of the range a neighbour is x0 itself: its loss is f0, not computed again.
    const float fl0 = xl0 == x0 ? f0 : loss(xl0, gb), fr0 = xr0 == x0 ? f0 : loss(xr0, gb);
    Bracket br{xl0, xr0, f0};
    if (fl0 < f0 && fr0 < f0) {
      const Bracket left = march(xl0, fl0, -1.0), right = march(xr0, fr0, 1.0);
      br = left.f <= right.f ? left : right;
    } else if (fl0 < f0) {
      br = march(xl0, fl0, -1.0);
    } else if (fr0 < f0) {
      br = march(xr0, fr0, 1.0);
    }
    double lo = br.lo, hi = br.hi;
    const double phi = 0.6180339887498949;   // (sqrt(5) - 1) / 2: golden-section search (Kiefer, 1953)
    double x1 = hi - phi * (hi - lo), x2 = lo + phi * (hi - lo);
    float f1 = loss(x1, gb), f2 = loss(x2, gb);
    while (hi - lo > kContinuousTol) {
      if (f1 <= f2) {
        hi = x2;
        x2 = x1;
        f2 = f1;
        x1 = hi - phi * (hi - lo);
        f1 = loss(x1, gb);
      } else {
        lo = x1;
        x1 = x2;
        f1 = f2;
        x2 = lo + phi * (hi - lo);
        f2 = loss(x2, gb);
      }
    }
    const double next_a = 0.5 * (lo + hi);
    ConversionParams next;
    loss(next_a, next);
    cr.last_move[0] = (float)(std::fabs(next_a - a) / kStopDensity);
    cr.last_move[1] = std::fabs(next.g - curr.g);
    cr.last_move[2] = std::fabs(next.b - curr.b);
    const float move = cr.last_move[0] + cr.last_move[1] + cr.last_move[2];
    a = next_a;
    curr.g = next.g;
    curr.b = next.b;
    stop_next = move <= TOL_PARAM;   // the moved parameters are rendered and scored once more
  }
  // --probe_intent: what the network predicts from the print at the result's g, b and A moved by each probe stop
  // (stops of printing exposure, A + stop log10 2); written beside the search, which goes on to convert. Each record's
  // e is the search's E, in its units as the scene probe's: e_norm on TriLED (the base at n0), 1 on white light, where
  // the search runs on M_frame itself (has_crosstalk false); the result's E is 1 on both. As in the scene probe, each
  // record's predicted is the network's own prediction and the file's target_L the mean L* of the lightness target
  // the search held.
  if (!opts.probe_path.empty()) {
    nlohmann::json recs = nlohmann::json::array();
    for (const float stop : opts.probe_stops) {
      const double ap = std::clamp(best_a + stop * kStopDensity, in.a_lo, in.a_hi);
      const double applied_stop = (ap - best_a) / kStopDensity;
      set_curves_at(ap);
      fr.render_preview(0, best, preview);
      if (!fr.predict(preview, mu, sg)) return false;
      const std::vector<float> lab = fr.patch_lab(preview);
      recs.push_back({{"kind", "params"}, {"stop", applied_stop}, {"requested_stop", stop}, {"a", ap}, {"e", best.e},
                      {"g", best.g}, {"b", best.b}, {"predicted", probe_mean3(mu)}, {"sigma", probe_mean3(sg)},
                      {"render", probe_mean3(lab)}, {"mu", mu}, {"sigma_cells", sg}, {"render_cells", lab}});
    }
    const nlohmann::json doc = {{"search", "print"}, {"grid_w", st.grid_w}, {"grid_h", st.grid_h},
                                {"e_norm", e_norm}, {"has_crosstalk", bundle.has_crosstalk}, {"a", best_a},
                                {"target_L", cr.target_mean_l}, {"records", recs}};
    if (!write_probe_file(opts.probe_path, doc, recs.size(), opts.verbose)) return false;
  }
  set_curves_at(best_a);
  {   // the crop's share with a channel above 1 that the print cuts (above 1.002 linear, a margin over the paper's
      // white, whose sRGB values relative to the paper are 1.0001, 1.0000, 0.9994), and its share at the paper's white,
      // in the picture the conversion writes (g and b in the curves, the matrix at E alone)
    SolverConfig sc;
    sc.prof = &prof.data;
    sc.knee = AUTO_KNEE;
    sc.gain_curves = &gain_curves;
    float mb[9];
    std::vector<float> trc[3];
    const negicc::PipelineConfig white_cfg = candidate_config(sc, fr.input_matrix(), {best.e, best.g, best.b}, mb, trc);
    const InnerCrop& c0 = fr.inner_crop();
    auto share = [&](float ceil, float out3[3]) {
      std::vector<float> hc(5, 0.0f);
      negicc::HighlightClipArgs ha{};
      ha.in_view = &frame;
      ha.width = frame.width;
      ha.height = frame.height;
      ha.crop = {c0.x0, c0.y0, c0.w, c0.h};
      ha.matrices = mb;
      ha.n = 1;
      ha.cfg = white_cfg;
      for (int c = 0; c < 3; ++c) ha.ceil_lin[c] = ceil;
      ha.out = hc.data();
      if (opts.renderer) opts.renderer->highlight_clip(ha); else negicc::cpu_highlight_clip(ha);
      for (int c = 0; c < 3; ++c) out3[c] = hc[c] / 100.0f;   // highlight_clip's percent, as a share
    };
    if (opts.renderer) opts.renderer->bind_frame(frame);
    share(1.002f, cr.above_white);
    share(0.999f, cr.at_white);
  }
  const int nearest = nearest_bracket(in.a_mid, best_a);
  out.target_idx = nearest;
  out.target_name = bundle.targets[nearest].name;
  out.params = {white_light ? E : E / e_norm, best.g, best.b, best.shoulder};   // as search_conversion_params
  out.residual = best_err;
  out.status = cr.converged ? FixedPointStatus::kConverged : FixedPointStatus::kMaxIters;
  out.intent_mu = best_mu;
  out.intent_sigma = best_sg;
  out.intent_grid_w = st.grid_w;
  out.intent_grid_h = st.grid_h;
  out.intent_params = best;
  out.g_lo = st.g_lo;
  out.g_hi = st.g_hi;
  cr.a = best_a;
  fr.write_timings();
  out.total_time = Clock::now() - t0;
  return true;
}

bool search_print_continuous(const ProfileBundle& bundle, const std::vector<PrintMaterialSearch>& prints,
                             DinoV3Engine& engine, const negicc::RgbView& frame, double scan_shutter_s,
                             double scan_iso, const ParamSearchOptions& opts) {
  const AuditScope audit_scope;  // PassAudit (neg_process --check_renderer)
  const Clock::time_point t0 = Clock::now();
  for (const PrintMaterialSearch& p : prints) {
    if (bundle.targets.empty() || !p.print_base->data.has_profile || !p.in->trc_at || !(p.in->a_hi > p.in->a_lo)) {
      std::cerr << "ERROR! The continuous print search needs a bundle's print (its grey balance curve)" << std::endl;
      return false;
    }
  }
  PrintFrame pf;
  pf.pb = bundle;
  pf.pb.targets.resize(1);
  pf.crop = compute_inner_crop(frame.width, frame.height, opts.crop_ratio);
  compute_frame_stats(frame, opts.crop_ratio, engine.max_size(), engine.patch_size(), pf.stats);
  pf.scale = frame_scale(pf.pb, (float)scan_shutter_s, (float)scan_iso, opts);
  for (size_t i = 0; i < prints.size(); ++i) {
    // The first material's time includes what the others share.
    if (!search_print_material(pf, bundle, *prints[i].print_base, *prints[i].in, engine, frame, opts,
                               i == 0 ? t0 : Clock::now(), *prints[i].out, *prints[i].res)) {
      return false;
    }
  }
  return true;
}
