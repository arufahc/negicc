#ifndef NEGICC_INPUT_CURVES_H
#define NEGICC_INPUT_CURVES_H

// Fitted G and B input curves (negicc-station designs/plan_fit_curves.md): per channel, on the value v entering the
// profile's input TRC (after the input matrix and the knee),
//
//   u = log2(v / ref)  for v < ref,   s = 1 + a + k tanh((u - u_med) / 1 stop),   v' = ref 2^(u s);   v >= ref: v
//
// ref is the film base at that point at Auto's E, g, b (frozen when the curves are fitted), u_med the median over the
// intent grid's cells of min(u, 0). The combined profile is the bracket profile with its G and B input tables
// replaced by in(curve(v)) sampled on the table's own entries; red, the matrix, the cLUT and the output curves are
// unchanged. White light only: with a crosstalk matrix the film base does not mark the dark end of every channel.
//
// Header-only: neg_process and negicc-station compile the same code, so both front ends fit the same curves.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "color_pipeline.h"

namespace negicc {

struct InputCurves {
  // [0] = G, [1] = B
  float a[2] = {0.0f, 0.0f};
  float k[2] = {0.0f, 0.0f};
  float ref[2] = {1.0f, 1.0f};
  float u_med[2] = {0.0f, 0.0f};
  bool active() const { return a[0] != 0.0f || k[0] != 0.0f || a[1] != 0.0f || k[1] != 0.0f; }
};

inline float input_curve(float v, float ref, float u_med, float a, float k) {
  if (!(v < ref) || (a == 0.0f && k == 0.0f)) return v;
  const float vv = std::max(v, 1e-12f);
  const float u = std::min(std::log2(vv / ref), 0.0f);
  const float s = 1.0f + a + k * std::tanh(u - u_med);
  return ref * std::exp2(u * s);
}

// out[i] = in(curve(i / (n - 1))), in sampled with apply_trc's linear interpolation.
inline void compose_input_table(const float* in, int n, float ref, float u_med, float a, float k, float* out) {
  for (int i = 0; i < n; ++i) {
    float v = input_curve(static_cast<float>(i) / static_cast<float>(n - 1), ref, u_med, a, k);
    v = std::min(1.0f, std::max(0.0f, v));
    out[i] = apply_trc(v, in, n);
  }
}

// The combined profile's G and B input tables (owned) and the ProfileData pointing at them; red and everything else
// point at the base profile, which must outlive it.
struct CombinedProfile {
  std::vector<float> in_g, in_b;
  ProfileData data{};

  void build(const ProfileData& base, const InputCurves& c) {
    data = base;
    if (!c.active() || !base.has_profile) return;
    const int ng = base.in_trc_size[1], nb = base.in_trc_size[2];
    if (ng > 1 && base.in_trc[1]) {
      in_g.resize(ng);
      compose_input_table(base.in_trc[1], ng, c.ref[0], c.u_med[0], c.a[0], c.k[0], in_g.data());
      data.in_trc[1] = in_g.data();
    }
    if (nb > 1 && base.in_trc[2]) {
      in_b.resize(nb);
      compose_input_table(base.in_trc[2], nb, c.ref[1], c.u_med[1], c.a[1], c.k[1], in_b.data());
      data.in_trc[2] = in_b.data();
    }
  }
};

// ref and u_med for G and B: the film base and the grid's cells through the merged input matrix at (e, g, b) and the
// knee. merged: the 3x3 input matrix before the per-candidate rows (candidate_matrix); film_base: camera units (as the
// grid). grid: n cells of 16-bit camera-linear RGB.
inline void input_curve_reference(const float* merged, float e, float g, float b, float knee, const float film_base[3],
                                  const uint16_t* grid, int n, InputCurves& c) {
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  const Rgb fb = apply_matrix(m, {film_base[0] / 65535.0f, film_base[1] / 65535.0f, film_base[2] / 65535.0f});
  const float refs[2] = {knee_compress(fb.g, knee), knee_compress(fb.b, knee)};
  std::vector<float> ug(n), ub(n);
  for (int i = 0; i < n; ++i) {
    const Rgb v = apply_matrix(m, load_u16(grid + 3 * i));
    const float vg = std::max(knee_compress(v.g, knee), 1e-12f), vb = std::max(knee_compress(v.b, knee), 1e-12f);
    ug[i] = std::min(std::log2(vg / std::max(refs[0], 1e-12f)), 0.0f);
    ub[i] = std::min(std::log2(vb / std::max(refs[1], 1e-12f)), 0.0f);
  }
  auto median = [](std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    float hi = v[h];
    if (v.size() % 2) return hi;
    const float lo = *std::max_element(v.begin(), v.begin() + h);
    return 0.5f * (lo + hi);
  };
  for (int ch = 0; ch < 2; ++ch) c.ref[ch] = refs[ch];
  c.u_med[0] = median(ug);
  c.u_med[1] = median(ub);
}

// The intent loss of the grid through cfg (its profile possibly combined) at (e, g, b): mean over cells of
// |(Lab - mu) / sigma|^2, the sum of cpu_intent_loss's three terms.
inline double grid_intent_loss(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                               const uint16_t* grid, int n, const float* mu, const float* sigma) {
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const Rgb lin = convert_pixel_linear(cfg, m, grid + 3 * i);
    const Lab lab = linear_srgb_to_lab(to_display(cfg, lin));
    const float dL = (lab.L - mu[i * 3]) / sigma[i * 3];
    const float da = (lab.a - mu[i * 3 + 1]) / sigma[i * 3 + 1];
    const float db = (lab.b - mu[i * 3 + 2]) / sigma[i * 3 + 2];
    s += static_cast<double>(dL) * dL + static_cast<double>(da) * da + static_cast<double>(db) * db;
  }
  return n > 0 ? s / n : 0.0;
}

struct CurveFitOptions {
  float curve_bound = 0.25f;  // |a|, |k| <= this
  float g_bound = 0.5f;       // |g - g0| <= this
  float b_bound = 0.6f;       // |b - b0| <= this
  float lambda = 0.0f;        // penalty on a^2 + k^2 (Iterations 227 and 228: 0)
  float step0_gain = 0.08f;
  float step0_curve = 0.05f;
  float step_min = 1e-4f;
  int max_evals = 8000;
};

struct CurveFitResult {
  InputCurves curves;
  float g = 1.0f, b = 1.0f;
  double loss_before = 0.0, loss_after = 0.0;
  int evaluations = 0;
};

// Joint fit of g, b and the four curve numbers with E fixed, against the frozen intent field mu, sigma on the grid:
// a deterministic bounded pattern search (each coordinate tried at +step and -step in turn, kept when the objective
// falls; all steps halve after a sweep without improvement). c carries ref and u_med (input_curve_reference).
//
// TODO(cuda): this evaluates on the CPU (about 700 cells x 1,000 to 2,500 evaluations). On a CUDA build it should run
// as a kernel that composes the input tables and scores the cells on the device, using the profile's existing device
// copy, so the fit never copies the grid render back to the host per evaluation.
inline CurveFitResult fit_input_curves(PipelineConfig cfg, const float* merged, float e, float g0, float b0,
                                       const uint16_t* grid, int n, const float* mu, const float* sigma,
                                       InputCurves c, const CurveFitOptions& o = CurveFitOptions()) {
  const ProfileData base = cfg.profile;
  CombinedProfile comb;
  auto objective = [&](const float* p, int& evals) {
    InputCurves t = c;
    t.a[0] = p[2];
    t.k[0] = p[3];
    t.a[1] = p[4];
    t.k[1] = p[5];
    comb.build(base, t);
    cfg.profile = comb.data;
    ++evals;
    const double pen = o.lambda * (static_cast<double>(p[2]) * p[2] + static_cast<double>(p[3]) * p[3] +
                                   static_cast<double>(p[4]) * p[4] + static_cast<double>(p[5]) * p[5]);
    return grid_intent_loss(cfg, merged, e, p[0], p[1], grid, n, mu, sigma) + pen;
  };
  const float lo[6] = {g0 - o.g_bound, b0 - o.b_bound, -o.curve_bound, -o.curve_bound, -o.curve_bound, -o.curve_bound};
  const float hi[6] = {g0 + o.g_bound, b0 + o.b_bound, o.curve_bound, o.curve_bound, o.curve_bound, o.curve_bound};
  float p[6] = {g0, b0, 0.0f, 0.0f, 0.0f, 0.0f};
  float step[6] = {o.step0_gain, o.step0_gain, o.step0_curve, o.step0_curve, o.step0_curve, o.step0_curve};
  CurveFitResult r;
  int evals = 0;
  double best = objective(p, evals);
  r.loss_before = best;
  while (evals < o.max_evals) {
    bool improved = false;
    for (int d = 0; d < 6 && evals < o.max_evals; ++d) {
      for (int sgn : {1, -1}) {
        float q[6];
        std::copy(p, p + 6, q);
        q[d] = std::min(hi[d], std::max(lo[d], p[d] + sgn * step[d]));
        if (q[d] == p[d]) continue;
        const double f = objective(q, evals);
        if (f < best) {
          best = f;
          std::copy(q, q + 6, p);
          improved = true;
          break;
        }
      }
    }
    if (!improved) {
      float mx = 0.0f;
      for (float& s : step) {
        s *= 0.5f;
        mx = std::max(mx, s);
      }
      if (mx < o.step_min) break;
    }
  }
  r.curves = c;
  r.curves.a[0] = p[2];
  r.curves.k[0] = p[3];
  r.curves.a[1] = p[4];
  r.curves.k[1] = p[5];
  r.g = p[0];
  r.b = p[1];
  r.loss_after = grid_intent_loss([&] {
    PipelineConfig c2 = cfg;
    comb.build(base, r.curves);
    c2.profile = comb.data;
    return c2;
  }(), merged, e, r.g, r.b, grid, n, mu, sigma);
  r.evaluations = evals;
  return r;
}

// ---- Fit curves auto (negicc-station designs/plan_fit_curves_auto.md) ------------------------------------------
// After Auto: try the curves only when Auto leaves a cast (S_cast at or above tau_cast), and keep them only when the
// fitted result removes enough of it without moving the tones or clipping more. Both front ends call these, so they
// decide the same.

struct FitCurvesAutoOptions {
  float tau_cast = 7.17f;  // S_cast that triggers the fit (Iteration 227's threshold, fixed before Iteration 229)
  float r_cast = 0.7f;     // keep only if S_cast after <= r_cast x S_cast before
  float d_l50 = 1.5f;      // |median L* change| of the grid's cells
  float d_ltail = 2.5f;    // |2nd and 98th percentile L* change| of the grid's cells
  float d_clip = 0.1f;     // white-clip rise, percentage points of the frame's pixels
};

// The grid's colour residual and tones at (e, g, b) through cfg (its profile possibly combined): S_cast, the mean
// over cells of the a* and b* terms of the intent loss against mu, sigma; and the 2nd, 50th and 98th percentile of
// the cells' L* (nearest rank on the sorted cells). white_clip (% of the frame's pixels) is the caller's: it needs
// the frame, not the grid.
struct GridToneCast {
  double s_cast = 0.0;
  float p2_L = 0.0f, median_L = 0.0f, p98_L = 0.0f;
  float white_clip = 0.0f;
};

inline GridToneCast grid_tone_cast(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                                   const uint16_t* grid, int n, const float* mu, const float* sigma) {
  GridToneCast t;
  if (n <= 0) return t;
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  std::vector<float> L(n);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const Rgb lin = convert_pixel_linear(cfg, m, grid + 3 * i);
    const Lab lab = linear_srgb_to_lab(to_display(cfg, lin));
    const float da = (lab.a - mu[i * 3 + 1]) / sigma[i * 3 + 1];
    const float db = (lab.b - mu[i * 3 + 2]) / sigma[i * 3 + 2];
    s += static_cast<double>(da) * da + static_cast<double>(db) * db;
    L[i] = lab.L;
  }
  t.s_cast = s / n;
  std::sort(L.begin(), L.end());
  auto rank = [&](float q) { return L[std::min(n - 1, std::max(0, (int)std::lround(q * (n - 1))))]; };
  t.p2_L = rank(0.02f);
  t.median_L = rank(0.5f);
  t.p98_L = rank(0.98f);
  return t;
}

struct FitCurvesAutoCheck {
  const char* name;
  double value;  // the measured quantity the limit applies to
  double limit;
  bool passed;
};

struct FitCurvesAutoDecision {
  bool triggered = false;  // S_cast before >= tau_cast: the fit ran
  bool applied = false;    // the curves are kept
  const char* outcome = "not needed";  // "not needed", "applied" or the first failing check's name
  double s_cast_before = 0.0, s_cast_after = 0.0;
  std::vector<FitCurvesAutoCheck> checks;  // in order: cast, median L*, p2 L*, p98 L*, white clip, converged
};

inline bool fit_curves_auto_triggered(const GridToneCast& before, const FitCurvesAutoOptions& o) {
  return before.s_cast >= o.tau_cast;
}

// The usefulness test of the fitted result (after, at the fixed point's E, g, b on the combined profile, against its
// own fresh intent field) against Auto's (before).
inline FitCurvesAutoDecision fit_curves_auto_decide(const GridToneCast& before, const GridToneCast& after,
                                                    bool converged, const FitCurvesAutoOptions& o) {
  FitCurvesAutoDecision d;
  d.s_cast_before = before.s_cast;
  d.s_cast_after = after.s_cast;
  d.triggered = fit_curves_auto_triggered(before, o);
  if (!d.triggered) return d;
  const double ratio = before.s_cast > 0.0 ? after.s_cast / before.s_cast : 1.0;
  d.checks = {
      {"cast", ratio, o.r_cast, ratio <= o.r_cast},
      {"median L*", std::fabs(after.median_L - before.median_L), o.d_l50,
       std::fabs(after.median_L - before.median_L) <= o.d_l50},
      {"p2 L*", std::fabs(after.p2_L - before.p2_L), o.d_ltail, std::fabs(after.p2_L - before.p2_L) <= o.d_ltail},
      {"p98 L*", std::fabs(after.p98_L - before.p98_L), o.d_ltail, std::fabs(after.p98_L - before.p98_L) <= o.d_ltail},
      {"white clip", after.white_clip - before.white_clip, o.d_clip, after.white_clip - before.white_clip <= o.d_clip},
      {"converged", converged ? 1.0 : 0.0, 1.0, converged},
  };
  d.applied = true;
  d.outcome = "applied";
  for (const auto& c : d.checks) {
    if (!c.passed) {
      d.applied = false;
      d.outcome = c.name;
      break;
    }
  }
  return d;
}

}  // namespace negicc

#endif  // NEGICC_INPUT_CURVES_H
