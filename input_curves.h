#ifndef NEGICC_INPUT_CURVES_H
#define NEGICC_INPUT_CURVES_H

// Fitted G and B input curves (negicc-station designs/plan_fit_curves.md): per channel, on the value v entering the
// profile's input TRC (after the input matrix and the knee),
//
//   u = log2(v / ref)  for v < ref,   s = 1 + a + k tanh((u - u_med) / 1 stop),   v' = ref 2^(u s);   v >= ref: v
//
// ref is the film base at that point and u_med the median over the intent grid's cells of min(u, 0), both at the E, g
// and b the conversion renders with (anchor_curves): the fit anchors each candidate at its own g and b, and the
// re-meter each E it tries, so the recorded curves pivot on the film base of the export's parameters. A replay (and the
// knobs, which move E, g and b afterwards) keeps the recorded ref and u_med. The combined profile is the bracket profile with its G and B input tables
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
  // The densest grid cell's u (<= 0): the low end of the frame's density range, over which the fit keeps each curve
  // monotonic and within CurveFitOptions::max_dev_stops of the base TRC. Fitting only; replay does not need it.
  float u_lo[2] = {0.0f, 0.0f};
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
  c.u_lo[0] = ug.empty() ? 0.0f : *std::min_element(ug.begin(), ug.end());
  c.u_lo[1] = ub.empty() ? 0.0f : *std::min_element(ub.begin(), ub.end());
  c.u_med[0] = median(ug);
  c.u_med[1] = median(ub);
}

// What anchors a curve on a frame: the 3x3 input matrix before the per-candidate rows (candidate_matrix), the knee,
// the film base (camera units, as the grid) and the grid's n cells of 16-bit camera-linear RGB.
struct CurveFrame {
  const float* merged = nullptr;
  float knee = 0.95f;
  float film_base[3] = {0.0f, 0.0f, 0.0f};
  const uint16_t* grid = nullptr;
  int n = 0;
};

// shape's a and k anchored at (e, g, b): ref, u_med and u_lo from the frame there (input_curve_reference).
inline InputCurves anchor_curves(const InputCurves& shape, const CurveFrame& f, float e, float g, float b) {
  InputCurves c = shape;
  input_curve_reference(f.merged, e, g, b, f.knee, f.film_base, f.grid, f.n, c);
  return c;
}

// One channel's curve over the frame's density range [u_lo, 0] (u in stops below the film base): monotonic (denser
// film never renders lighter in this channel) and within max_dev stops of the base TRC (|u s(u) - u|), checked on
// samples+1 evenly spaced points. The identity (a = k = 0) always passes.
inline bool curve_shape_ok(float a, float k, float u_med, float u_lo, float max_dev, int samples = 64) {
  if (a == 0.0f && k == 0.0f) return true;
  if (!(u_lo < 0.0f)) return true;
  float prev = -INFINITY;
  for (int i = 0; i <= samples; ++i) {
    const float u = u_lo * (1.0f - static_cast<float>(i) / samples);  // u_lo up to 0
    const float f = u * (1.0f + a + k * std::tanh(u - u_med));
    if (!(f > prev) || std::fabs(f - u) > max_dev) return false;
    prev = f;
  }
  return true;
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
  float max_dev_stops = 1.0f; // each curve within this many stops of the base TRC over the frame (curve_shape_ok)
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

// Joint fit of g, b and the four curve numbers with E fixed, against the frozen intent field mu, sigma on the frame's
// grid: a deterministic bounded pattern search (each coordinate tried at +step and -step in turn, kept when the
// objective falls; all steps halve after a sweep without improvement). Each candidate's curves are anchored at its own
// g and b (anchor_curves), so the curves the result carries pivot on the film base at the fitted g and b.
//
// TODO(cuda): this evaluates on the CPU (about 700 cells x 1,000 to 2,500 evaluations). On a CUDA build it should run
// as a kernel that composes the input tables and scores the cells on the device, using the profile's existing device
// copy, so the fit never copies the grid render back to the host per evaluation.
inline CurveFitResult fit_input_curves(PipelineConfig cfg, const CurveFrame& f, float e, float g0, float b0,
                                       const float* mu, const float* sigma, const CurveFitOptions& o = CurveFitOptions()) {
  const ProfileData base = cfg.profile;
  CombinedProfile comb;
  auto curves_of = [&](const float* p) {
    InputCurves t;
    t.a[0] = p[2];
    t.k[0] = p[3];
    t.a[1] = p[4];
    t.k[1] = p[5];
    return anchor_curves(t, f, e, p[0], p[1]);
  };
  auto objective = [&](const float* p, int& evals) {
    const InputCurves t = curves_of(p);
    ++evals;
    // Monotonic and bounded over the frame's density range, or not a candidate.
    if (!curve_shape_ok(t.a[0], t.k[0], t.u_med[0], t.u_lo[0], o.max_dev_stops) ||
        !curve_shape_ok(t.a[1], t.k[1], t.u_med[1], t.u_lo[1], o.max_dev_stops))
      return static_cast<double>(INFINITY);
    comb.build(base, t);
    cfg.profile = comb.data;
    const double pen = o.lambda * (static_cast<double>(p[2]) * p[2] + static_cast<double>(p[3]) * p[3] +
                                   static_cast<double>(p[4]) * p[4] + static_cast<double>(p[5]) * p[5]);
    return grid_intent_loss(cfg, f.merged, e, p[0], p[1], f.grid, f.n, mu, sigma) + pen;
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
        const double fq = objective(q, evals);
        if (fq < best) {
          best = fq;
          std::copy(q, q + 6, p);
          improved = true;
          break;
        }
      }
    }
    if (!improved) {
      float mx = 0.0f;
      for (float& st : step) {
        st *= 0.5f;
        mx = std::max(mx, st);
      }
      if (mx < o.step_min) break;
    }
  }
  r.curves = curves_of(p);
  r.g = p[0];
  r.b = p[1];
  comb.build(base, r.curves);
  cfg.profile = comb.data;
  r.loss_after = grid_intent_loss(cfg, f.merged, e, r.g, r.b, f.grid, f.n, mu, sigma);
  r.evaluations = evals;
  return r;
}

// The grid's median L* through cfg at (e, g, b) (nearest rank over the cells).
inline float grid_median_L(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                           const uint16_t* grid, int n) {
  if (n <= 0) return 0.0f;
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  std::vector<float> L(n);
  for (int i = 0; i < n; ++i) L[i] = linear_srgb_to_lab(to_display(cfg, convert_pixel_linear(cfg, m, grid + 3 * i))).L;
  std::nth_element(L.begin(), L.begin() + n / 2, L.end());
  return L[n / 2];
}

// E re-metered so the grid's median L* through the combined profile at the fitted g, b equals target_L (Auto's): a
// bisection in log E over [e0 / 4, e0 * 4] (a larger E renders darker), the curves (shape's a and k) anchored at each E
// it tries, so the profile it measures is the one the export renders at that E. cfg carries the bracket's own profile.
// negicc-station Iteration 230's F6L, which replaces the E/g/b fixed point after the curve fit (the fixed point
// followed its own darker render).
inline float remeter_e_median(PipelineConfig cfg, const CurveFrame& f, const InputCurves& shape, float e0, float g,
                              float b, float target_L, int iters = 40) {
  const ProfileData base = cfg.profile;
  CombinedProfile comb;
  auto median_at = [&](float e) {
    comb.build(base, anchor_curves(shape, f, e, g, b));
    cfg.profile = comb.data;
    return grid_median_L(cfg, f.merged, e, g, b, f.grid, f.n);
  };
  float lo = e0 * 0.25f, hi = e0 * 4.0f;
  for (int i = 0; i < iters; ++i) {
    const float mid = std::sqrt(lo * hi);
    if (median_at(mid) < target_L) hi = mid;
    else lo = mid;
  }
  return std::sqrt(lo * hi);
}

}  // namespace negicc

#endif  // NEGICC_INPUT_CURVES_H
