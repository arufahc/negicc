#ifndef NEGICC_INPUT_CURVES_H
#define NEGICC_INPUT_CURVES_H

// Fitted G and B input curves (negicc-station designs/plan_fit_curves.md): per channel, on the value v entering the
// profile's input TRC (after the input matrix and the knee),
//
//   u = log2(v / ref)  for v < ref,   s = 1 + a + k tanh((u - u_med) / 1 stop),   v' = ref 2^(u s);   v >= ref: v
//
// ref is the film base at that point and u_med the median over the intent grid's cells of min(u, 0), both at the E, g
// and b the conversion renders with (anchor_curves): the fit anchors each candidate at its own g and b, and the
// re-meter each E it tries, so the recorded curves pivot on the film base of the export's parameters. A replay at the
// recorded E, g and b keeps the recorded ref and u_med. The knobs (--brightness, --tint, --warmth) move E, g and b
// after the curves were anchored: there the curves are re-anchored at the knobbed E, g and b (ref and u_med moved by
// what the frame's film base and grid move between the two anchors) and the input tables composed again from the
// bracket's un-curved ones (neg_process's reanchor_input_curves), so the curves stay on the same film densities
// instead of sliding along the table with E. The combined profile is the bracket
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

// The smallest sigma the losses divide by. The intent field's sigma is checked positive and finite where it is
// predicted (SearchFrame::predict), but a sigma just above 0 would still let one cell's residual swamp the fit; the
// floor is far below any sigma DINOv3 predicts (a few L* or a*/b* units), so it never moves an ordinary loss.
constexpr float kSigmaFloor = 1e-3f;

// The intent loss of the grid through cfg (its profile possibly combined) at (e, g, b): mean over cells of
// |(Lab - mu) / sigma|^2, the sum of cpu_intent_loss's three terms (3 z2 in multi_target_optimizer.h's terms).
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
    const float dL = (lab.L - mu[i * 3]) / std::max(sigma[i * 3], kSigmaFloor);
    const float da = (lab.a - mu[i * 3 + 1]) / std::max(sigma[i * 3 + 1], kSigmaFloor);
    const float db = (lab.b - mu[i * 3 + 2]) / std::max(sigma[i * 3 + 2], kSigmaFloor);
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

// The lowest g or b a fit may reach (CurveFitOptions::g_bound, b_bound below g0, b0 are floored here).
constexpr float kGainFloor = 1e-3f;

struct CurveFitOptions {
  float curve_bound = 0.25f;  // |a|, |k| <= this
  float max_dev_stops = 1.0f; // each curve within this many stops of the base TRC over the whole table (curve_shape_ok)
  float g_bound = 0.5f;       // |g - g0| <= this
  float b_bound = 0.6f;       // |b - b0| <= this
  // Penalty on the four curve numbers, lambda (aG^2 + kG^2 + aB^2 + kB^2), added to the fits' objective z2 (the mean
  // of the squared residuals over the 3 n cell terms). The gains and the curves' slope terms (1 + a) move the
  // residuals almost alike, so without it the Levenberg-Marquardt normal matrix is nearly singular along that
  // direction; the small default only fixes the direction the residuals leave free (Iterations 227 and 228 ran 0).
  float lambda = 1e-3f;
  float step0_gain = 0.08f;
  float step0_curve = 0.05f;
  float step_min = 1e-4f;
  int max_evals = 8000;
  float polish_step0 = 0.02f;   // the joint fit's coordinate polish: first step, and its budget
  int polish_max_evals = 600;
  // Step 2 options (hold_slope on; the others off, which reproduces the joint fit):
  int starts = 1;               // seeds: no curves, then a = +-start_a on G and B in turn; the lowest loss wins
  float start_a = 0.06f;
  bool hold_slope = true;       // InputCurves::hold for the fitted curves (on: Alpha Lam, 2026-11-05)
  float l_weight = 1.0f;        // weight of the cells' L* residuals in the loss (lightness is held by the constraint)
  float huber = 0.0f;           // > 0: each residual through a Huber function with this threshold (in sigma units)
  // The joint fit (fit_input_curves_joint): E solved for each candidate so the grid's median L* equals Auto's. Off:
  // E stays at Auto's (the measurement of the solver against the pattern search).
#ifdef NEGICC_CURVE_FIT_E_FIXED
  bool solve_e = false;
#else
  bool solve_e = true;
#endif
  // The tolerance, in L*, of each candidate's E solve (JointCurveModel::solve_e). The Jacobian's probes move a
  // parameter by 1e-3, which moves the grid's median L* by about 0.01: a tolerance of 1e-3 (before) left the probe's E
  // either unchanged or moved by up to the tolerance, so the derivative through E was noise. 2e-5 is a few float steps
  // of L* near 50.
  float e_tol_L = 2e-5f;
};

struct CurveFitResult {
  InputCurves curves;
  float g = 1.0f, b = 1.0f;
  double loss_before = 0.0, loss_after = 0.0;
  int evaluations = 0;
};

// The first fit, which the joint fit below replaced in neg_process: g, b and the four curve numbers with E fixed,
// against the frozen intent field mu, sigma on the frame's grid: a deterministic bounded pattern search (each
// coordinate tried at +step and -step in turn, kept when the objective falls; all steps halve after a sweep without
// improvement) on the same objective (z2 plus the lambda penalty) and with the same held curves (hold_slope). Each
// candidate's curves are anchored at its own g and b (anchor_curves), so the curves the result carries pivot on the
// film base at the fitted g and b.
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
    t.hold = o.hold_slope;  // as the joint fit's curves (before, this fit ignored hold_slope)
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
    if (!curve_shape_ok(t.a[0], t.k[0], t.u_med[0], t.ref[0], base.in_trc_size[1], o.max_dev_stops, t.u_hold(0)) ||
        !curve_shape_ok(t.a[1], t.k[1], t.u_med[1], t.ref[1], base.in_trc_size[2], o.max_dev_stops, t.u_hold(1)))
      return static_cast<double>(INFINITY);
    comb.build(base, t);
    cfg.profile = comb.data;
    const double pen = o.lambda * (static_cast<double>(p[2]) * p[2] + static_cast<double>(p[3]) * p[3] +
                                   static_cast<double>(p[4]) * p[4] + static_cast<double>(p[5]) * p[5]);
    // z2 (grid_intent_loss is the sum of the three per-cell means, 3 z2) plus the penalty, as the joint fit's.
    return grid_intent_loss(cfg, f.merged, e, p[0], p[1], f.grid, f.n, mu, sigma) / 3.0 + pen;
  };
  // The gains keep a positive floor: at g or b = 0 the channel's ref (the film base through the gain) is 0 and the
  // curve's anchor collapses (u = log2(v / ref) is undefined), so a wide g_bound or b_bound never reaches it.
  const float lo[6] = {std::max(kGainFloor, g0 - o.g_bound), std::max(kGainFloor, b0 - o.b_bound), -o.curve_bound,
                       -o.curve_bound, -o.curve_bound, -o.curve_bound};
  const float hi[6] = {g0 + o.g_bound, b0 + o.b_bound, o.curve_bound, o.curve_bound, o.curve_bound, o.curve_bound};
  float p[6] = {g0, b0, 0.0f, 0.0f, 0.0f, 0.0f};
  float step[6] = {o.step0_gain, o.step0_gain, o.step0_curve, o.step0_curve, o.step0_curve, o.step0_curve};
  CurveFitResult r;
  int evals = 0;
  double best = objective(p, evals);
  {
    comb.build(base, curves_of(p));
    cfg.profile = comb.data;
    r.loss_before = grid_intent_loss(cfg, f.merged, e, p[0], p[1], f.grid, f.n, mu, sigma);
  }
  while (evals < o.max_evals) {
    bool improved = false;
    for (int d = 0; d < 6 && evals < o.max_evals; ++d) {
      for (int sgn : {1, -1}) {
        if (evals >= o.max_evals) break;  // checked per trial, so the budget is never overrun by the second sign
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
// it had fitted). The objective is z2 (the mean of the cells' squared residuals over their 3 n terms, grid_intent_loss
// / 3) plus CurveFitOptions::lambda times the curves' squared numbers, evaluated on a per-cell model:
//   - each cell's three matrix rows are taken once (merged · rgb), and a candidate's (e, g, b) only scales them;
//   - the curve is applied to each cell's G and B value analytically, in(curve(v)), in place of composing the two
//     4096-entry tables per candidate (the export's tables sample the same function at their entries);
//   - E is solved within [e_lo, e_hi] by secant steps on log E warm-started from the previous candidate's E (one to
//     three grid renders), with a bracketed bisection as the fallback; a candidate whose E would leave the bounds is
//     not one (the export could not render it at target_L there);
//   - the solver is Levenberg-Marquardt on the residuals (dL, da, db) / (sigma sqrt(3 n)) and sqrt(lambda) times the
//     curve numbers, with a forward-difference Jacobian, box bounds by projection and the shape (curve_shape_ok over
//     the whole table, at each candidate's anchor) by cutting a step back to the furthest point that keeps it (its E
//     solved at each point tried) or rejecting it and raising the damping.
// The result's curves are anchored at its E, g and b; the caller re-meters E on the real combined tables, which moves
// it by the small difference between the model and the tables.
struct JointCurveFit {
  InputCurves curves;
  float e = 1.0f, g = 1.0f, b = 1.0f;
  // The data loss (grid_intent_loss's units: the sum of the three per-cell means, without the penalty) at the first
  // seed (Auto's g and b, no curves, E solved) and at the result.
  double loss_before = 0.0, loss_after = 0.0;
  int evaluations = 0;  // points whose E was solved and whose shape and residuals were evaluated (every one tried)
  int renders = 0;      // grid median renders spent solving E (the evaluations' and the cut-backs')
  int iterations = 0;   // accepted Levenberg-Marquardt steps
  int polish_evaluations = 0;  // of the evaluations, the coordinate polish's (counted the same way)
  int start = 0;               // the seed that won (CurveFitOptions::starts)
  bool converged = false;      // every feasible run settled (a step or gain below tolerance), none stalled
  // False when no seed is feasible (not even Auto's g and b without curves keeps target_L within the E bounds): the
  // result is then Auto's g, b and e0 with no curves, which the caller should not take as a fit.
  bool feasible = true;
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

  // The residuals (dL, da, db) / sigma of every cell at (e, g, b) with shape anchored there; returns the sum of their
  // squares over the n cells (grid_intent_loss's units, the sum of the three per-cell means; the fit's objective
  // divides the residuals by sqrt(3 n)).
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
      r[3 * i] = robust(l_scale_ * (lab.L - mu_[3 * i]) / std::max(sigma_[3 * i], kSigmaFloor));
      r[3 * i + 1] = robust((lab.a - mu_[3 * i + 1]) / std::max(sigma_[3 * i + 1], kSigmaFloor));
      r[3 * i + 2] = robust((lab.b - mu_[3 * i + 2]) / std::max(sigma_[3 * i + 2], kSigmaFloor));
      s += r[3 * i] * r[3 * i] + r[3 * i + 1] * r[3 * i + 1] + r[3 * i + 2] * r[3 * i + 2];
    }
    return f_.n ? s / f_.n : 0.0;
  }

  // E with the grid's median L* at target_L within [e_lo, e_hi] (a larger E renders darker: the median falls with
  // log E, each cell's L* does). Secant steps on log E from e_warm (clamped into the bounds), each new point kept in a
  // bracket of the two nearest points either side of the target; where the secant leaves the bracket or stalls, the
  // bounds the bracket lacks are tested and the bracket bisected. False when the target lies outside the bounds (the
  // median at one bound is already past it): e is that bound, and the point is not one the export can render at
  // target_L. Before, a warm start at a bound was stepped outward and clamped back onto itself (the secant then saw
  // two equal points and gave up), the fallback bisected [e0 / 4, 4 e0] without testing its ends, and an unreachable
  // target returned a bound as if it were the answer.
  bool solve_e(float g, float b, const InputCurves& shape, float target_L, float e_warm, float e_lo, float e_hi,
               float tol, int& renders, float& e) {
    const float lo_t = std::log(e_lo), hi_t = std::log(e_hi);
    auto f = [&](float t) {
      ++renders;
      return median_L(std::exp(t), g, b, shape) - target_L;
    };
    // The bracket: ta the largest log E seen with the median above the target (f > 0), tc the smallest below it.
    bool has_a = false, has_c = false;
    float ta = lo_t, tc = hi_t;
    auto note = [&](float t, float ft) {
      if (ft > 0.0f) {
        if (!has_a || t > ta) ta = t;
        has_a = true;
      } else {
        if (!has_c || t < tc) tc = t;
        has_c = true;
      }
    };
    auto at = [&](float t, bool reached) {
      e = std::exp(t);
      return reached;
    };
    float t0 = std::min(hi_t, std::max(lo_t, std::log(e_warm))), f0 = f(t0);
    if (std::fabs(f0) <= tol) return at(t0, true);
    note(t0, f0);
    // The second point: 0.05 in log E towards the target. Where t0 sits on the bound that step would cross, the
    // target is beyond the bound (the median is monotonic in log E): not reachable.
    const float d = f0 > 0.0f ? 0.05f : -0.05f;
    const float t_bound = d > 0.0f ? hi_t : lo_t;
    if (t0 == t_bound) return at(t0, false);
    float t1 = (d > 0.0f) ? std::min(hi_t, t0 + d) : std::max(lo_t, t0 + d), f1 = f(t1);
    if (std::fabs(f1) <= tol) return at(t1, true);
    note(t1, f1);
    if (t1 == t_bound && (f1 > 0.0f) == (f0 > 0.0f)) return at(t1, false);  // still short at the bound
    for (int it = 0; it < 8; ++it) {
      if (f1 == f0) break;
      const float t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
      const float b_lo = has_a ? ta : lo_t, b_hi = has_c ? tc : hi_t;
      if (!(t2 > b_lo && t2 < b_hi)) break;  // outside the bracket (or the bounds): bisect instead
      t0 = t1;
      f0 = f1;
      t1 = t2;
      f1 = f(t1);
      if (std::fabs(f1) <= tol) return at(t1, true);
      note(t1, f1);
    }
    // The bisection needs both sides: test the bound the bracket lacks (the target beyond it: not reachable).
    if (!has_a) {
      const float fl = f(lo_t);
      if (std::fabs(fl) <= tol) return at(lo_t, true);
      if (fl < 0.0f) return at(lo_t, false);
      note(lo_t, fl);
    }
    if (!has_c) {
      const float fh = f(hi_t);
      if (std::fabs(fh) <= tol) return at(hi_t, true);
      if (fh > 0.0f) return at(hi_t, false);
      note(hi_t, fh);
    }
    for (int it = 0; it < 40 && tc - ta > 1e-6f; ++it) {
      const float mid = 0.5f * (ta + tc), fm = f(mid);
      if (std::fabs(fm) <= tol) return at(mid, true);
      note(mid, fm);
    }
    e = std::exp(0.5f * (ta + tc));
    return true;
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
// and roll-off. target_L: Auto's grid median L*. table_n: the G and B input tables' sizes, for the shape check. e_lo,
// e_hi: the bounds of every candidate's E (0: e0 / 4 and 4 e0, as before they were given); they should hold e0.
inline JointCurveFit fit_input_curves_joint(const PipelineConfig& cfg, const CurveFrame& f, float e0, float g0, float b0,
                                            const float* mu, const float* sigma, float target_L, const int table_n[2],
                                            const CurveFitOptions& o = CurveFitOptions(), float e_lo = 0.0f,
                                            float e_hi = 0.0f) {
  JointCurveModel model(cfg, f, mu, sigma);
  model.set_weighting(o.l_weight, o.huber);
  if (!(e_lo > 0.0f)) e_lo = e0 * 0.25f;
  if (!(e_hi > e_lo)) e_hi = std::max(e_lo, e0 * 4.0f);
  // The gains' positive floor, as in fit_input_curves.
  const float lo[6] = {std::max(kGainFloor, g0 - o.g_bound), std::max(kGainFloor, b0 - o.b_bound), -o.curve_bound,
                       -o.curve_bound, -o.curve_bound, -o.curve_bound};
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
  const double inv_sqrt_3n = f.n > 0 ? 1.0 / std::sqrt(3.0 * f.n) : 0.0;
  const double sqrt_lambda = std::sqrt(std::max(0.0, static_cast<double>(o.lambda)));
  // A point's E (solved within the bounds, or e0) and whether the point is a candidate: its E reaches target_L within
  // the bounds and (check_shape) its curves keep the shape at its anchor.
  auto solve_point = [&](const float* p, float e_warm, float& e, bool check_shape) {
    const InputCurves shape = shape_of(p);
    if (o.solve_e) {
      if (!model.solve_e(p[0], p[1], shape, target_L, e_warm, e_lo, e_hi, o.e_tol_L, renders, e)) return false;
    } else {
      e = e0;
    }
    if (!check_shape) return true;
    const InputCurves c = model.anchored(shape, e, p[0], p[1]);
    return curve_shape_ok(c.a[0], c.k[0], c.u_med[0], c.ref[0], table_n[0], o.max_dev_stops, c.u_hold(0)) &&
           curve_shape_ok(c.a[1], c.k[1], c.u_med[1], c.ref[1], table_n[1], o.max_dev_stops, c.u_hold(1));
  };
  // A point evaluated: its E, the solver's residual vector r (the cells' over sqrt(3 n), then the penalty's four),
  // the objective (the sum of r's squares) and the data loss; false when it is not a candidate (solve_point).
  auto evaluate = [&](const float* p, float e_warm, float& e, std::vector<double>& r, double& cost, double& data,
                      bool check_shape) {
    ++out.evaluations;
    if (!solve_point(p, e_warm, e, check_shape)) return false;
    data = model.residuals(e, p[0], p[1], shape_of(p), r);
    const size_t nd = r.size();
    double sum = 0.0;
    for (size_t k = 0; k < nd; ++k) {
      r[k] *= inv_sqrt_3n;
      sum += r[k] * r[k];
    }
    r.resize(nd + 4);
    for (int j = 0; j < 4; ++j) {
      r[nd + j] = sqrt_lambda * p[2 + j];
      sum += r[nd + j] * r[nd + j];
    }
    cost = sum;
    return std::isfinite(cost);
  };
  // The furthest point p + t (q - p), t in [0, 1], that is a candidate (its E solved at each t tried, warm-started at
  // e; t = 0, p itself, is one), by bisection; its t, and its E in e_t. Before, the shape was checked at p's E and the
  // point then evaluated at its own, where it could fail again, so the solver stalled on the boundary.
  auto cut_back = [&](const float* p, const float* q, float e, float& e_t) {
    float t_ok = 0.0f, t_bad = 1.0f;
    e_t = e;
    for (int it = 0; it < 10; ++it) {
      const float t = 0.5f * (t_ok + t_bad);
      float qt[6];
      for (int j = 0; j < 6; ++j) qt[j] = p[j] + t * (q[j] - p[j]);
      float et = e;
      if (solve_point(qt, e, et, true)) {
        t_ok = t;
        e_t = et;
      } else {
        t_bad = t;
      }
    }
    return t_ok;
  };

  // One run of the solver and the polish from p_init; its result in p_out, e_out, cost_out (the objective), data_out
  // (the data loss); feasible_out false when not even the seed's no-curve end is a candidate.
  auto run = [&](const float* p_init, bool first, float* p_out, float& e_out, double& cost_out, double& data_out,
                 bool& settled_out, bool& feasible_out) {
    float p[6];
    std::copy(p_init, p_init + 6, p);
    float e = e0;
    std::vector<double> r, rt, rj;
    double cost = 0.0, data = 0.0;
    settled_out = false;
    feasible_out = false;
    // The seed must be a candidate (before, it was evaluated without the shape check, so an infeasible seed could
    // win): where its curves leave the shape or its E the bounds, they are cut back towards none.
    if (!evaluate(p, e0, e, r, cost, data, true)) {
      float none[6];
      std::copy(p, p + 6, none);
      for (int j = 2; j < 6; ++j) none[j] = 0.0f;
      float e_none = e0;
      if (!solve_point(none, e0, e_none, true)) return;
      float e_t = e_none;
      const float t = cut_back(none, p, e_none, e_t);
      for (int j = 2; j < 6; ++j) p[j] = t * p[j];
      if (!evaluate(p, e_t, e, r, cost, data, true)) return;
    }
    feasible_out = true;
    if (first) out.loss_before = data;
    const int m = 6;
    double lambda = 1e-3;
    bool settled = false;
    for (int iter = 0; iter < 60 && !settled; ++iter) {
      // Forward-difference Jacobian of the residuals (E solved at each probe, warm-started at this point's E, to
      // CurveFitOptions::e_tol_L, so the probe's E follows the parameter instead of the solver's tolerance).
      const size_t nr = r.size();
      std::vector<double> J(nr * m);
      for (int j = 0; j < m; ++j) {
        float q[6];
        std::copy(p, p + 6, q);
        const float h = (q[j] + 1e-3f <= hi[j]) ? 1e-3f : -1e-3f;
        q[j] += h;
        float ej = e;
        double cj = 0.0, dj = 0.0;
        if (!evaluate(q, e, ej, rj, cj, dj, false) || rj.size() != nr) {
          // The probe's E left the bounds: no derivative along j this iteration.
          for (size_t k = 0; k < nr; ++k) J[k * m + j] = 0.0;
          continue;
        }
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
        double cq = 0.0, dq = 0.0;
        bool feasible = evaluate(q, e, eq, rt, cq, dq, true);
        if (!feasible) {
          // The step leaves the shape or the E bounds: the furthest candidate along it (the optimum of a strong cast
          // usually sits on the shape's boundary, and rejecting the whole step stalled the solver inside it).
          float e_t = e;
          const float t_ok = cut_back(p, q, e, e_t);
          if (t_ok > 0.0f) {
            for (int j = 0; j < m; ++j) q[j] = p[j] + t_ok * (q[j] - p[j]);
            feasible = evaluate(q, e_t, eq, rt, cq, dq, true);
          }
        }
        if (feasible && cq < cost) {
          const double gain = cost - cq;
          std::copy(q, q + 6, p);
          e = eq;
          r.swap(rt);
          cost = cq;
          data = dq;
          lambda = std::max(lambda / 3.0, 1e-9);
          accepted = true;
          ++out.iterations;
          if (gain < 1e-7 * std::max(1.0, cost)) settled = true;
        } else {
          lambda *= 4.0;
        }
      }
      // No step lowers the objective at any damping tried while the steps were not yet negligible: the solver stalled
      // (before, this was reported as settled).
      if (!accepted && !settled) break;
    }
    // A coordinate polish from the solver's result (each parameter tried at +step and -step, kept when the objective
    // falls, steps halving after a sweep without one): it slides along an active bound or the shape's boundary, where
    // the Levenberg-Marquardt step, which moves all parameters at once, keeps leaving the feasible set.
    {
      float step = o.polish_step0;
      int polish = 0;
      while (step >= o.step_min && polish < o.polish_max_evals) {
        bool improved = false;
        for (int d = 0; d < m && polish < o.polish_max_evals; ++d) {
          for (int sgn : {1, -1}) {
            if (polish >= o.polish_max_evals) break;  // per trial, so the budget is never overrun
            float q[6];
            std::copy(p, p + 6, q);
            q[d] = std::min(hi[d], std::max(lo[d], p[d] + sgn * step));
            if (q[d] == p[d]) continue;
            float eq = e;
            double cq = 0.0, dq = 0.0;
            ++polish;
            if (evaluate(q, e, eq, rt, cq, dq, true) && cq < cost) {
              std::copy(q, q + 6, p);
              e = eq;
              r.swap(rt);
              cost = cq;
              data = dq;
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
    data_out = data;
    settled_out = settled;
  };

  // The seeds: no curves, then a = +start_a and -start_a on G and on B in turn (CurveFitOptions::starts of them); the
  // feasible run with the lowest objective wins, the first on a tie. The result starts as the first seed's point, so it
  // is defined even when no run is feasible.
  const float seeds[5][2] = {{0.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}};
  float p[6] = {g0, b0, 0.0f, 0.0f, 0.0f, 0.0f}, e = e0;
  double cost = INFINITY, data = 0.0;
  bool any = false;
  out.converged = true;
  for (int si = 0; si < std::max(1, std::min(o.starts, 5)); ++si) {
    float p0[6] = {g0, b0, seeds[si][0] * o.start_a, 0.0f, seeds[si][1] * o.start_a, 0.0f};
    float ps[6], es = e0;
    double cs = 0.0, ds = 0.0;
    bool settled = false, feasible = false;
    run(p0, si == 0, ps, es, cs, ds, settled, feasible);
    if (!feasible) continue;
    out.converged = out.converged && settled;
    if (!any || cs < cost) {
      std::copy(ps, ps + 6, p);
      e = es;
      cost = cs;
      data = ds;
      out.start = si;
      any = true;
    }
  }
  out.feasible = any;
  if (!any) out.converged = false;
  out.g = p[0];
  out.b = p[1];
  out.e = e;
  out.curves = model.anchored(shape_of(p), e, p[0], p[1]);
  out.loss_after = any ? data : 0.0;
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
// bisection in log E (a larger E renders darker), the curves (shape's a and k) anchored at each E it tries, so the
// profile it measures is the one the export renders at that E. cfg carries the bracket's own profile. The bisection
// runs over [e_centre / 2, 2 e_centre] within [e_lo, e_hi] when the target lies there (both ends tested first), else
// over [e_lo, e_hi]; where the target lies outside those too, *clipped is set and E is the bound it reached. Before,
// it bisected [e0 / 4, 4 e0] around Auto's E whatever E the fit had solved, without testing the ends, and an
// unreachable target returned a bound as if it were the answer.
// negicc-station Iteration 230's F6L, which replaces the E/g/b fixed point after the curve fit (the fixed point
// followed its own darker render).
inline float remeter_e_median_in(PipelineConfig cfg, const CurveFrame& f, const InputCurves& shape, float e_centre,
                                 float e_lo, float e_hi, float g, float b, float target_L, bool* clipped = nullptr,
                                 int iters = 40) {
  const ProfileData base = cfg.profile;
  CombinedProfile comb;
  auto median_at = [&](float e) {
    comb.build(base, anchor_curves(shape, f, e, g, b));
    cfg.profile = comb.data;
    return grid_median_L(cfg, f.merged, e, g, b, f.grid, f.n);
  };
  if (clipped) *clipped = false;
  float lo = std::max(e_lo, e_centre * 0.5f), hi = std::min(e_hi, e_centre * 2.0f);
  const bool enclosed = lo < hi && median_at(lo) >= target_L && median_at(hi) <= target_L;
  if (!enclosed) {
    lo = e_lo;
    hi = e_hi;
    const float L_lo = median_at(lo), L_hi = median_at(hi);
    const bool dark_at_lo = L_lo + 0.05f < target_L;  // darker than the target even at the lowest E (0.05 L* LUT tolerance)
    if (dark_at_lo || L_hi - 0.05f > target_L) {      // or lighter even at the highest
      if (clipped) *clipped = true;
      return L_lo < target_L ? lo : hi;
    }
    if (L_lo <= target_L) return lo;
    if (L_hi >= target_L) return hi;
  }
  for (int i = 0; i < iters; ++i) {
    const float mid = std::sqrt(lo * hi);
    if (median_at(mid) < target_L) hi = mid;
    else lo = mid;
  }
  return std::sqrt(lo * hi);
}

// The same around e0 over [e0 / 4, 4 e0] (the bounds before they were given).
inline float remeter_e_median(PipelineConfig cfg, const CurveFrame& f, const InputCurves& shape, float e0, float g,
                              float b, float target_L, int iters = 40) {
  return remeter_e_median_in(cfg, f, shape, e0, e0 * 0.25f, e0 * 4.0f, g, b, target_L, nullptr, iters);
}

}  // namespace negicc

#endif  // NEGICC_INPUT_CURVES_H
