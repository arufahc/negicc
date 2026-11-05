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
// knobs, which move E, g and b afterwards) keeps the recorded ref and u_med. The combined profile is the bracket
// profile with its G and B input tables replaced by in(curve(v)) sampled on the table's own entries; red, the matrix,
// the cLUT and the output curves are unchanged. White light only: with a crosstalk matrix the film base does not mark
// the dark end of every channel.
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
  // The densest grid cell's u (<= 0): the low end of the frame's density range. Recorded only (the shape is checked
  // over the whole input table, curve_shape_ok); replay does not need it.
  float u_lo[2] = {0.0f, 0.0f};
  // Below u_lo the curve stops bending and keeps the offset it has there (slope 1 in stops): its departure from the
  // base TRC beyond the frame's density range is the one at u_lo. Off: the curve's form continues to the table's end.
  bool hold = false;
  float u_hold(int i) const { return hold ? u_lo[i] : -INFINITY; }
  bool active() const { return a[0] != 0.0f || k[0] != 0.0f || a[1] != 0.0f || k[1] != 0.0f; }
};

// f(u) = u s(u), held below u_hold at f(u_hold) + (u - u_hold) (u_hold = -inf: not held).
inline float curve_stops(float u, float u_med, float a, float k, float u_hold = -INFINITY) {
  if (u < u_hold) return u_hold * (1.0f + a + k * std::tanh(u_hold - u_med)) + (u - u_hold);
  return u * (1.0f + a + k * std::tanh(u - u_med));
}

inline float input_curve(float v, float ref, float u_med, float a, float k, float u_hold = -INFINITY) {
  if (!(v < ref) || (a == 0.0f && k == 0.0f)) return v;
  const float vv = std::max(v, 1e-12f);
  const float u = std::min(std::log2(vv / ref), 0.0f);
  return ref * std::exp2(curve_stops(u, u_med, a, k, u_hold));
}

// out[i] = in(curve(i / (n - 1))), in sampled with apply_trc's linear interpolation.
inline void compose_input_table(const float* in, int n, float ref, float u_med, float a, float k, float* out,
                                float u_hold = -INFINITY) {
  for (int i = 0; i < n; ++i) {
    float v = input_curve(static_cast<float>(i) / static_cast<float>(n - 1), ref, u_med, a, k, u_hold);
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
      compose_input_table(base.in_trc[1], ng, c.ref[0], c.u_med[0], c.a[0], c.k[0], in_g.data(), c.u_hold(0));
      data.in_trc[1] = in_g.data();
    }
    if (nb > 1 && base.in_trc[2]) {
      in_b.resize(nb);
      compose_input_table(base.in_trc[2], nb, c.ref[1], c.u_med[1], c.a[1], c.k[1], in_b.data(), c.u_hold(1));
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

// One channel's curve over the whole input table it is composed into (compose_input_table): at every entry
// v = i / (n - 1) below ref, u = log2(v / ref), the curve is monotonic (denser film never renders lighter in this
// channel) and within max_dev stops of the base TRC (|u s(u) - u|). Before, this was checked only over the frame's
// density range [u_lo, 0] (the densest grid cell), while the table carries the curve down to 1 / (n - 1), where
// pixels denser than any grid cell land: there the curve could fold over or move arbitrarily far from the TRC. The
// identity (a = k = 0) always passes.
inline bool curve_shape_ok(float a, float k, float u_med, float ref, int table_n, float max_dev,
                           float u_hold = -INFINITY) {
  if (a == 0.0f && k == 0.0f) return true;
  if (!(ref > 0.0f) || table_n < 2) return true;
  float prev = INFINITY;  // f at the entry above; the table is walked from ref down
  for (int i = table_n - 1; i >= 1; --i) {
    const float v = static_cast<float>(i) / static_cast<float>(table_n - 1);
    if (!(v < ref)) continue;
    const float u = std::log2(v / ref);
    const float f = curve_stops(u, u_med, a, k, u_hold);
    if (!(f < prev) || std::fabs(f - u) > max_dev) return false;
    prev = f;
  }
  return true;
}

// The intent loss of the grid through cfg (its profile possibly combined) at (e, g, b): mean over cells of
// |(Lab - mu) / sigma|^2, the sum of cpu_intent_loss's three terms.
// The loss's lightness and colour parts: mean over cells of ((L - mu_L) / sigma_L)^2, and of the a* and b* terms.
struct LossParts {
  double L = 0.0, ab = 0.0;
  double total() const { return L + ab; }
};

inline LossParts grid_intent_loss_parts(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                                        const uint16_t* grid, int n, const float* mu, const float* sigma) {
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  LossParts s;
  for (int i = 0; i < n; ++i) {
    const Rgb lin = convert_pixel_linear(cfg, m, grid + 3 * i);
    const Lab lab = linear_srgb_to_lab(to_display(cfg, lin));
    const float dL = (lab.L - mu[i * 3]) / sigma[i * 3];
    const float da = (lab.a - mu[i * 3 + 1]) / sigma[i * 3 + 1];
    const float db = (lab.b - mu[i * 3 + 2]) / sigma[i * 3 + 2];
    s.L += static_cast<double>(dL) * dL;
    s.ab += static_cast<double>(da) * da + static_cast<double>(db) * db;
  }
  if (n > 0) {
    s.L /= n;
    s.ab /= n;
  }
  return s;
}

inline double grid_intent_loss(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                               const uint16_t* grid, int n, const float* mu, const float* sigma) {
  return grid_intent_loss_parts(cfg, merged, e, g, b, grid, n, mu, sigma).total();
}

struct CurveFitOptions {
  float curve_bound = 0.25f;  // |a|, |k| <= this
  float max_dev_stops = 1.0f; // each curve within this many stops of the base TRC over the whole table (curve_shape_ok)
  float g_bound = 0.5f;       // |g - g0| <= this
  float b_bound = 0.6f;       // |b - b0| <= this
  float lambda = 0.0f;        // penalty on a^2 + k^2 (Iterations 227 and 228: 0)
  float step0_gain = 0.08f;
  float step0_curve = 0.05f;
  float step_min = 1e-4f;
  int max_evals = 8000;
  float polish_step0 = 0.02f;   // the joint fit's coordinate polish: first step, and its budget
  int polish_max_evals = 600;
  // Step 2 options, off by default (the defaults reproduce the joint fit):
  int starts = 1;               // seeds: no curves, then a = +-start_a on G and B in turn; the lowest loss wins
  float start_a = 0.06f;
  bool hold_slope = false;      // InputCurves::hold for the fitted curves
  float l_weight = 1.0f;        // weight of the cells' L* residuals in the loss (lightness is held by the constraint)
  float huber = 0.0f;           // > 0: each residual through a Huber function with this threshold (in sigma units)
  // The joint fit (fit_input_curves_joint): E solved for each candidate so the grid's median L* equals Auto's. Off:
  // E stays at Auto's (the measurement of the solver against the pattern search).
#ifdef NEGICC_CURVE_FIT_E_FIXED
  bool solve_e = false;
#else
  bool solve_e = true;
#endif
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
                                       const float* mu, const float* sigma,
                                       const CurveFitOptions& o = CurveFitOptions()) {
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
    // Monotonic and bounded over the whole input table, or not a candidate.
    if (!curve_shape_ok(t.a[0], t.k[0], t.u_med[0], t.ref[0], base.in_trc_size[1], o.max_dev_stops) ||
        !curve_shape_ok(t.a[1], t.k[1], t.u_med[1], t.ref[1], base.in_trc_size[2], o.max_dev_stops))
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


// ---- The joint fit -------------------------------------------------------------------------------------------
//
// g, b and the four curve numbers fitted against the intent field with E tied to the lightness constraint: each
// candidate is scored at the E that renders the grid's median L* at target_L (Auto's), so the fit chooses its colour at
// the lightness the export will have, instead of fitting at Auto's E and moving E afterwards (which moved the colour
// it had fitted). The loss is the same (grid_intent_loss), evaluated on a per-cell model:
//   - each cell's three matrix rows are taken once (merged · rgb), and a candidate's (e, g, b) only scales them;
//   - the curve is applied to each cell's G and B value analytically, in(curve(v)), in place of composing the two
//     4096-entry tables per candidate (the export's tables sample the same function at their entries);
//   - E is solved by secant steps on log E warm-started from the previous candidate's E (one to three grid renders),
//     with a bisection as the fallback;
//   - the solver is Levenberg-Marquardt on the cells' residuals (dL, da, db) / sigma, with a forward-difference
//     Jacobian, box bounds by projection and the shape (curve_shape_ok over the whole table, at each candidate's anchor)
//     by rejecting a step and raising the damping.
// The result's curves are anchored at its E, g and b; the caller re-meters E on the real combined tables, which moves
// it by the small difference between the model and the tables.
struct JointCurveFit {
  InputCurves curves;
  float e = 1.0f, g = 1.0f, b = 1.0f;
  double loss_before = 0.0, loss_after = 0.0;
  int evaluations = 0;  // residual evaluations (each with its E solved)
  int renders = 0;      // grid median renders spent solving E
  int iterations = 0;   // accepted Levenberg-Marquardt steps
  int polish_evaluations = 0;  // of them, the coordinate polish's
  int start = 0;               // the seed that won (CurveFitOptions::starts)
  bool converged = false;
};

class JointCurveModel {
 public:
  JointCurveModel(const PipelineConfig& cfg, const CurveFrame& f, const float* mu, const float* sigma)
      : cfg_(cfg), f_(f), mu_(mu), sigma_(sigma), rows_(3 * f.n), L_(f.n), ug_(f.n), ub_(f.n) {
    for (int i = 0; i < f.n; ++i) {
      const Rgb x = load_u16(f.grid + 3 * i);
      for (int r = 0; r < 3; ++r)
        rows_[3 * i + r] = f.merged[3 * r] * x.r + f.merged[3 * r + 1] * x.g + f.merged[3 * r + 2] * x.b;
    }
    for (int r = 0; r < 3; ++r)
      fb_[r] = (f.merged[3 * r] * f.film_base[0] + f.merged[3 * r + 1] * f.film_base[1] +
                f.merged[3 * r + 2] * f.film_base[2]) / 65535.0f;
  }

  int n() const { return f_.n; }

  // The curves of shape (its a and k) anchored at (e, g, b), as anchor_curves anchors them.
  InputCurves anchored(const InputCurves& shape, float e, float g, float b) {
    InputCurves c = shape;
    const float refs[2] = {knee_compress(e * g * fb_[1], f_.knee), knee_compress(e * b * fb_[2], f_.knee)};
    for (int i = 0; i < f_.n; ++i) {
      const float vg = std::max(knee_compress(e * g * rows_[3 * i + 1], f_.knee), 1e-12f);
      const float vb = std::max(knee_compress(e * b * rows_[3 * i + 2], f_.knee), 1e-12f);
      ug_[i] = std::min(std::log2(vg / std::max(refs[0], 1e-12f)), 0.0f);
      ub_[i] = std::min(std::log2(vb / std::max(refs[1], 1e-12f)), 0.0f);
    }
    c.ref[0] = refs[0];
    c.ref[1] = refs[1];
    c.u_lo[0] = f_.n ? *std::min_element(ug_.begin(), ug_.end()) : 0.0f;
    c.u_lo[1] = f_.n ? *std::min_element(ub_.begin(), ub_.end()) : 0.0f;
    c.u_med[0] = median(ug_);
    c.u_med[1] = median(ub_);
    return c;
  }

  // One cell through the pipeline with the curves applied to its G and B value (c anchored).
  Lab cell_lab(int i, float e, float g, float b, const InputCurves& c) const {
    Rgb v = {knee_compress(e * rows_[3 * i], f_.knee), knee_compress(e * g * rows_[3 * i + 1], f_.knee),
             knee_compress(e * b * rows_[3 * i + 2], f_.knee)};
    v.g = clamp01(input_curve(v.g, c.ref[0], c.u_med[0], c.a[0], c.k[0], c.u_hold(0)));
    v.b = clamp01(input_curve(v.b, c.ref[1], c.u_med[1], c.a[1], c.k[1], c.u_hold(1)));
    return linear_srgb_to_lab(to_display(cfg_, pcs_to_linear_srgb_pre_shoulder(apply_profile(cfg_.profile, v))));
  }

  float median_L(float e, float g, float b, const InputCurves& shape) {
    const InputCurves c = anchored(shape, e, g, b);
    for (int i = 0; i < f_.n; ++i) L_[i] = cell_lab(i, e, g, b, c).L;
    std::nth_element(L_.begin(), L_.begin() + f_.n / 2, L_.end());
    return L_[f_.n / 2];
  }

  // The residuals (dL, da, db) / sigma of every cell at (e, g, b) with shape anchored there; returns the mean of
  // their squares (grid_intent_loss).
  // l_weight and huber reshape the residuals (CurveFitOptions); 1 and 0 leave them as grid_intent_loss's.
  void set_weighting(float l_weight, float huber) {
    l_scale_ = std::sqrt(std::max(l_weight, 0.0f));
    huber_ = huber;
  }

  double residuals(float e, float g, float b, const InputCurves& shape, std::vector<double>& r) {
    const InputCurves c = anchored(shape, e, g, b);
    r.resize(3 * (size_t)f_.n);
    double s = 0.0;
    auto robust = [&](double x) {
      if (!(huber_ > 0.0f) || std::fabs(x) <= huber_) return x;
      const double ax = std::fabs(x);
      return (x < 0 ? -1.0 : 1.0) * std::sqrt(2.0 * huber_ * ax - (double)huber_ * huber_);  // r'^2 = 2 Huber(r)
    };
    for (int i = 0; i < f_.n; ++i) {
      const Lab lab = cell_lab(i, e, g, b, c);
      r[3 * i] = robust(l_scale_ * (lab.L - mu_[3 * i]) / sigma_[3 * i]);
      r[3 * i + 1] = robust((lab.a - mu_[3 * i + 1]) / sigma_[3 * i + 1]);
      r[3 * i + 2] = robust((lab.b - mu_[3 * i + 2]) / sigma_[3 * i + 2]);
      s += r[3 * i] * r[3 * i] + r[3 * i + 1] * r[3 * i + 1] + r[3 * i + 2] * r[3 * i + 2];
    }
    return f_.n ? s / f_.n : 0.0;
  }

  // E with the grid's median L* at target_L (a larger E renders darker): secant steps on log E from e_warm, then a
  // bisection over [e0 / 4, e0 * 4] when they do not settle within tol.
  float solve_e(float g, float b, const InputCurves& shape, float target_L, float e_warm, float e0, int& renders,
                float tol = 1e-3f) {
    const float lo_t = std::log(e0 * 0.25f), hi_t = std::log(e0 * 4.0f);
    auto f = [&](float t) {
      ++renders;
      return median_L(std::exp(t), g, b, shape) - target_L;
    };
    float t0 = std::min(hi_t, std::max(lo_t, std::log(e_warm))), f0 = f(t0);
    if (std::fabs(f0) <= tol) return std::exp(t0);
    float t1 = std::min(hi_t, std::max(lo_t, t0 + (f0 > 0.0f ? 0.05f : -0.05f))), f1 = f(t1);
    for (int it = 0; it < 6; ++it) {
      if (std::fabs(f1) <= tol) return std::exp(t1);
      if (f1 == f0) break;
      const float t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
      if (!(t2 > lo_t && t2 < hi_t)) break;
      t0 = t1;
      f0 = f1;
      t1 = t2;
      f1 = f(t1);
    }
    float lo = lo_t, hi = hi_t;  // the fallback: median L* falls with log E
    for (int it = 0; it < 40; ++it) {
      const float mid = 0.5f * (lo + hi);
      if (f(mid) < 0.0f) hi = mid;
      else lo = mid;
    }
    return std::exp(0.5f * (lo + hi));
  }

 private:
  static float median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    const size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const float hi = v[h];
    if (v.size() % 2) return hi;
    return 0.5f * (*std::max_element(v.begin(), v.begin() + h) + hi);
  }

  PipelineConfig cfg_;
  CurveFrame f_;
  const float* mu_;
  const float* sigma_;
  float fb_[3] = {0.0f, 0.0f, 0.0f};
  float l_scale_ = 1.0f, huber_ = 0.0f;
  std::vector<float> rows_, L_, ug_, ub_;
};

// The joint fit from Auto's (e0, g0, b0) and no curves; cfg carries the bracket's own profile and the export's encoding
// and roll-off. target_L: Auto's grid median L*. table_n: the G and B input tables' sizes, for the shape check.
inline JointCurveFit fit_input_curves_joint(const PipelineConfig& cfg, const CurveFrame& f, float e0, float g0, float b0,
                                            const float* mu, const float* sigma, float target_L, const int table_n[2],
                                            const CurveFitOptions& o = CurveFitOptions()) {
  JointCurveModel model(cfg, f, mu, sigma);
  model.set_weighting(o.l_weight, o.huber);
  const float lo[6] = {g0 - o.g_bound, b0 - o.b_bound, -o.curve_bound, -o.curve_bound, -o.curve_bound, -o.curve_bound};
  const float hi[6] = {g0 + o.g_bound, b0 + o.b_bound, o.curve_bound, o.curve_bound, o.curve_bound, o.curve_bound};
  auto shape_of = [&](const float* p) {
    InputCurves c;
    c.hold = o.hold_slope;
    c.a[0] = p[2];
    c.k[0] = p[3];
    c.a[1] = p[4];
    c.k[1] = p[5];
    return c;
  };
  JointCurveFit out;
  int renders = 0;
  // A point: its E (solved, or Auto's) and its residuals; false when its curves leave the shape at its anchor.
  auto evaluate = [&](const float* p, float e_warm, float& e, std::vector<double>& r, double& cost, bool check_shape) {
    const InputCurves shape = shape_of(p);
    e = o.solve_e ? model.solve_e(p[0], p[1], shape, target_L, e_warm, e0, renders) : e0;
    if (check_shape) {
      const InputCurves c = model.anchored(shape, e, p[0], p[1]);
      if (!curve_shape_ok(c.a[0], c.k[0], c.u_med[0], c.ref[0], table_n[0], o.max_dev_stops, c.u_hold(0)) ||
          !curve_shape_ok(c.a[1], c.k[1], c.u_med[1], c.ref[1], table_n[1], o.max_dev_stops, c.u_hold(1)))
        return false;
    }
    cost = model.residuals(e, p[0], p[1], shape, r);
    ++out.evaluations;
    return true;
  };

  // One run of the solver and the polish from p_init; its result in p_out, e_out, cost_out.
  auto run = [&](const float* p_init, bool first, float* p_out, float& e_out, double& cost_out, bool& settled_out) {
    float p[6];
    std::copy(p_init, p_init + 6, p);
    float e = e0;
    std::vector<double> r, rt, rj;
    double cost = 0.0;
    evaluate(p, e0, e, r, cost, false);
    if (first) out.loss_before = cost;
    const int m = 6;
    double lambda = 1e-3;
    bool settled = false;
    for (int iter = 0; iter < 60 && !settled; ++iter) {
      // Forward-difference Jacobian of the residuals (E solved at each probe, warm-started at this point's E).
      const size_t nr = r.size();
      std::vector<double> J(nr * m);
      for (int j = 0; j < m; ++j) {
        float q[6];
        std::copy(p, p + 6, q);
        const float h = (q[j] + 1e-3f <= hi[j]) ? 1e-3f : -1e-3f;
        q[j] += h;
        float ej = e;
        double cj = 0.0;
        evaluate(q, e, ej, rj, cj, false);
        for (size_t k = 0; k < nr; ++k) J[k * m + j] = (rj[k] - r[k]) / h;
      }
      double A[6][6] = {}, gvec[6] = {};
      for (size_t k = 0; k < nr; ++k) {
        for (int a = 0; a < m; ++a) {
          gvec[a] += J[k * m + a] * r[k];
          for (int bb = a; bb < m; ++bb) A[a][bb] += J[k * m + a] * J[k * m + bb];
        }
      }
      for (int a = 0; a < m; ++a)
        for (int bb = 0; bb < a; ++bb) A[a][bb] = A[bb][a];
      bool accepted = false;
      for (int tries = 0; tries < 12 && !accepted; ++tries) {
        // (A + lambda diag(A)) delta = -g, by Gaussian elimination with partial pivoting.
        double M[6][7];
        for (int a = 0; a < m; ++a) {
          for (int bb = 0; bb < m; ++bb) M[a][bb] = A[a][bb] + (a == bb ? lambda * std::max(A[a][a], 1e-12) : 0.0);
          M[a][6] = -gvec[a];
        }
        for (int c = 0; c < m; ++c) {
          int piv = c;
          for (int rr = c + 1; rr < m; ++rr)
            if (std::fabs(M[rr][c]) > std::fabs(M[piv][c])) piv = rr;
          for (int cc = 0; cc < 7; ++cc) std::swap(M[c][cc], M[piv][cc]);
          if (std::fabs(M[c][c]) < 1e-300) M[c][c] = 1e-300;
          for (int rr = c + 1; rr < m; ++rr) {
            const double fct = M[rr][c] / M[c][c];
            for (int cc = c; cc < 7; ++cc) M[rr][cc] -= fct * M[c][cc];
          }
        }
        double delta[6];
        for (int c = m - 1; c >= 0; --c) {
          double sum = M[c][6];
          for (int cc = c + 1; cc < m; ++cc) sum -= M[c][cc] * delta[cc];
          delta[c] = sum / M[c][c];
        }
        float q[6];
        float step = 0.0f;
        for (int j = 0; j < m; ++j) {
          q[j] = std::min(hi[j], std::max(lo[j], p[j] + (float)delta[j]));
          step = std::max(step, std::fabs(q[j] - p[j]));
        }
        if (step < 1e-6f) {
          settled = true;
          break;
        }
        float eq = e;
        double cq = 0.0;
        bool feasible = evaluate(q, e, eq, rt, cq, true);
        if (!feasible) {
          // The step leaves the shape: the furthest point along it that keeps the shape (the optimum of a strong cast
          // usually sits on that boundary, and rejecting the whole step stalled the solver inside it).
          float t_ok = 0.0f, t_bad = 1.0f;
          for (int it = 0; it < 12; ++it) {
            const float t = 0.5f * (t_ok + t_bad);
            float qt[6];
            for (int j = 0; j < m; ++j) qt[j] = p[j] + t * (q[j] - p[j]);
            const InputCurves c = model.anchored(shape_of(qt), e, qt[0], qt[1]);
            if (curve_shape_ok(c.a[0], c.k[0], c.u_med[0], c.ref[0], table_n[0], o.max_dev_stops, c.u_hold(0)) &&
                curve_shape_ok(c.a[1], c.k[1], c.u_med[1], c.ref[1], table_n[1], o.max_dev_stops, c.u_hold(1)))
              t_ok = t;
            else
              t_bad = t;
          }
          if (t_ok > 0.0f) {
            for (int j = 0; j < m; ++j) q[j] = p[j] + t_ok * (q[j] - p[j]);
            feasible = evaluate(q, e, eq, rt, cq, true);
          }
        }
        if (feasible && cq < cost) {
          const double gain = cost - cq;
          std::copy(q, q + 6, p);
          e = eq;
          r.swap(rt);
          cost = cq;
          lambda = std::max(lambda / 3.0, 1e-9);
          accepted = true;
          ++out.iterations;
          if (gain < 1e-7 * std::max(1.0, cost)) settled = true;
        } else {
          lambda *= 4.0;
        }
      }
      if (!accepted) {
        settled = true;  // no step lowers the loss at any damping tried
        break;
      }
    }
    // A coordinate polish from the solver's result (each parameter tried at +step and -step, kept when the loss falls,
    // steps halving after a sweep without one): it slides along an active bound or the shape's boundary, where the
    // Levenberg-Marquardt step, which moves all parameters at once, keeps leaving the feasible set.
    {
      float step = o.polish_step0;
      int polish = 0;
      while (step >= o.step_min && polish < o.polish_max_evals) {
        bool improved = false;
        for (int d = 0; d < m && polish < o.polish_max_evals; ++d) {
          for (int sgn : {1, -1}) {
            float q[6];
            std::copy(p, p + 6, q);
            q[d] = std::min(hi[d], std::max(lo[d], p[d] + sgn * step));
            if (q[d] == p[d]) continue;
            float eq = e;
            double cq = 0.0;
            ++polish;
            if (evaluate(q, e, eq, rt, cq, true) && cq < cost) {
              std::copy(q, q + 6, p);
              e = eq;
              r.swap(rt);
              cost = cq;
              improved = true;
              break;
            }
          }
        }
        if (!improved) step *= 0.5f;
      }
      out.polish_evaluations += polish;
    }
    std::copy(p, p + 6, p_out);
    e_out = e;
    cost_out = cost;
    settled_out = settled;
  };

  // The seeds: no curves, then a = +start_a and -start_a on G and on B in turn (CurveFitOptions::starts of them); the
  // run with the lowest loss wins, the first on a tie.
  const float seeds[5][2] = {{0.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}};
  float p[6], e = e0;
  double cost = INFINITY;
  out.converged = true;
  for (int si = 0; si < std::max(1, std::min(o.starts, 5)); ++si) {
    float p0[6] = {g0, b0, seeds[si][0] * o.start_a, 0.0f, seeds[si][1] * o.start_a, 0.0f};
    float ps[6], es = e0;
    double cs = 0.0;
    bool settled = false;
    run(p0, si == 0, ps, es, cs, settled);
    out.converged = out.converged && settled;
    if (cs < cost) {
      std::copy(ps, ps + 6, p);
      e = es;
      cost = cs;
      out.start = si;
    }
  }
  out.g = p[0];
  out.b = p[1];
  out.e = e;
  out.curves = model.anchored(shape_of(p), e, p[0], p[1]);
  out.loss_after = cost;
  out.renders = renders;
  return out;
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
