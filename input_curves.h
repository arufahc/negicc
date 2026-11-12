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
  // The neutral-patch head's p(neutral) per grid cell (n values, DinoV3InferenceResult::neutral_p), or null when the
  // head did not run. The joint fit's scene-neutral anchors come from it (JointCurveModel::set_scene_neutral).
  const float* neutral_p = nullptr;
};

// shape's a and k anchored at (e, g, b): ref, u_med and u_lo from the frame there (input_curve_reference).
inline InputCurves anchor_curves(const InputCurves& shape, const CurveFrame& f, float e, float g, float b) {
  InputCurves c = shape;
  input_curve_reference(f.merged, e, g, b, f.knee, f.film_base, f.grid, f.n, c);
  return c;
}

// One channel's curve over the whole input table it is composed into (compose_input_table): at every entry
// v = i / (n - 1) below ref, u = log2(v / ref), the curve is monotonic (denser film never renders lighter in this
// channel) and within max_dev stops of the base TRC (|u s(u) - u|). The whole table is checked, not only the frame's
// density range [u_lo, 0] (the densest grid cell): pixels denser than any grid cell land below it, where the curve
// could otherwise fold over or move arbitrarily far from the TRC. The identity (a = k = 0) always passes.
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

// The lowest g or b a fit may reach (CurveFitOptions::g_bound, b_bound below g0, b0 are floored here).
constexpr float kGainFloor = 1e-3f;

struct CurveFitOptions {
  float curve_bound = 0.35f;  // |a|, |k| <= this
  float max_dev_stops = 2.2f; // each curve within this many stops of the base TRC over the whole table (curve_shape_ok)
  float g_bound = 0.5f;       // |g - g0| <= this (joint fit, when g_bound_stops <= 0)
  float b_bound = 0.6f;       // |b - b0| <= this (joint fit, when b_bound_stops <= 0)
  float g_bound_stops = 0.75f;  // joint fit: g in [g0 * 2^-s, g0 * 2^s] (<= 0: additive g_bound)
  float b_bound_stops = 1.25f;  // joint fit: b in [b0 * 2^-s, b0 * 2^s] (<= 0: additive b_bound)
  // Penalty on the curve numbers, joint_lambda (a_G^2 + a_B^2), added to the joint fit's objective. The gains and the
  // curves' slope terms (1 + a) move the residuals almost alike, so without it the Levenberg-Marquardt normal matrix
  // is nearly singular along that direction; damping non-essential curve bends also prevents shadow-to-highlight
  // crossover on clean frames.
  float joint_lambda = 1.5f;
  bool hold_slope = true;       // InputCurves::hold for the fitted curves (on: Alpha Lam, 2026-11-05)
  // The joint fit (fit_input_curves_joint): E solved for each candidate so the grid's median L* equals Auto's. Off:
  // E stays at Auto's (the measurement of the solver against the pattern search).
#ifdef NEGICC_CURVE_FIT_E_FIXED
  bool solve_e = false;
#else
  bool solve_e = true;
#endif
  // The tolerance, in L*, of each candidate's E solve (JointCurveModel::solve_e). The Jacobian's probes move a
  // parameter by 1e-3, which moves the grid's median L* by about 0.01: a tolerance of 1e-3 would leave the probe's E
  // either unchanged or moved by up to the tolerance, so the derivative through E would be noise. 2e-5 is a few float
  // steps of L* near 50.
  float e_tol_L = 2e-5f;
};

// ---- The joint fit -------------------------------------------------------------------------------------------
//
// g, b and the G and B curves' bend a_G, a_B (k is 0) fitted so the frame's scene-neutral anchors render grey, with E
// tied to the lightness constraint: each candidate is scored at the E that renders the grid's median L* at target_L
// (Auto's), so the fit chooses its colour at the lightness the export will have. The objective is the four
// scene-neutral residuals (JointCurveModel::residuals: the anchors' p^2-weighted mean a* and b*, and their slopes over
// L* against the slope shrink's target), the never-greener guard's two hinges on the guarded pass (JointFitGuard), and
// sqrt(CurveFitOptions::joint_lambda) times a_G and a_B. The intent field is not part of it (its losses are only
// reported). It is evaluated on a per-cell model:
//   - each cell's three matrix rows are taken once (merged · rgb), and a candidate's (e, g, b) only scales them;
//   - the curve is applied to each cell's G and B value analytically, in(curve(v)), in place of composing the two
//     4096-entry tables per candidate (the export's tables sample the same function at their entries);
//   - E is solved within [e_lo, e_hi] by secant steps on log E warm-started from the previous candidate's E (one to
//     three grid renders), with a bracketed bisection as the fallback; a candidate whose E would leave the bounds is
//     not one (the export could not render it at target_L there);
//   - the solver is Levenberg-Marquardt on those residuals with a forward-difference Jacobian, box bounds by
//     projection and the shape (curve_shape_ok over the whole table, at each candidate's anchor) by cutting a step
//     back to the furthest point that keeps it (its E solved at each point tried) or rejecting it and raising the
//     damping.
// The solved correction is then handed over by evidence: its displacement from Auto in (ln g, ln b, a_G, a_B) is
// scaled by w = m / (m + JointCurveModel::kEvidenceHalfMass), m the anchors' evidence mass, and E solved there again.
// The result's curves are anchored at its E, g and b; the caller re-meters E on the real combined tables, which moves
// it by the small difference between the model and the tables.
struct NeutralStats {
  int m = 0;
  float a_bar = 0.0f, b_bar = 0.0f;
  float k_a = 0.0f, k_b = 0.0f;
  float q_a = 0.0f, q_b = 0.0f;
  float off() const { return std::hypot(a_bar, b_bar); }
  float slope() const { return std::hypot(k_a, k_b); }
  float quad() const { return std::hypot(q_a, q_b); }
};

// What JointCurveModel::set_scene_neutral decided beyond the anchors themselves, for the sidecar.
struct NeutralTargetDecisions {
  float evidence = 0.0f;     // the anchors' evidence mass, sum of p^2 (0: no anchors, the fit keeps Auto's colour)
  float span_L = 0.0f;       // the anchors' weighted L* standard deviation the slope shrink measured
  float slope_share = 0.0f;  // the share of the anchors' slope at Auto the fit removes: sd^2 / (sd^2 + s0^2)
  float hand_over = 0.0f;    // the share of the fitted correction kept: evidence / (evidence + kEvidenceHalfMass)
};

// The joint fit's never-greener guard (fit_input_curves_joint): the grid's mean a* at Auto and at the fit's result,
// over all cells and weighted by the neutral head's p^2, for the sidecar.
struct GreenGuard {
  bool counted[2] = {false, false};  // [all cells, p^2-weighted]: the term had enough cells / evidence to count
  float auto_a[2] = {0.0f, 0.0f};    // Auto's mean a* (g0, b0, no curves, E solved)
  float fit_a[2] = {0.0f, 0.0f};     // the fit's mean a* at its result
  bool binding = false;              // a counted term ended more than the tolerance greener than Auto: the hinge pulled
  // The fit runs first without the guard's residuals; refit: its result broke the bound, so this is the guarded re-run.
  // free_fit_a: the unguarded result's two means (equal to fit_a when it was kept).
  bool refit = false;
  float free_fit_a[2] = {0.0f, 0.0f};
  // Diagnostics only (the guard acts on the means above): the same means per L* band of Auto's render ([2, 22],
  // (22, 45], (45, 70], (70, 92)), the band's cells and evidence mass. A band with no cells has zeros.
  int band_cells[4] = {0, 0, 0, 0};
  float band_mass[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float band_auto_a[4][2] = {}, band_fit_a[4][2] = {};
};

struct JointCurveFit {
  InputCurves curves;
  float e = 1.0f, g = 1.0f, b = 1.0f;
  // The data loss (the sum of the four squared scene-neutral residuals, JointCurveModel::residuals, without the guard
  // and the penalty) at Auto's g and b with no curves (E solved) and at the result.
  double loss_before = 0.0, loss_after = 0.0;
  int evaluations = 0;  // points whose E was solved and whose shape and residuals were evaluated (every one tried)
  int renders = 0;      // grid median renders spent solving E (the evaluations' and the cut-backs')
  int iterations = 0;   // accepted Levenberg-Marquardt steps
  bool converged = false;      // every feasible run settled (a step or gain below tolerance), none stalled
  // False when no seed is feasible (not even Auto's g and b without curves keeps target_L within the E bounds): the
  // result is then Auto's g, b and e0 with no curves, which the caller should not take as a fit.
  bool feasible = true;
  NeutralStats neutral_before, neutral_after;
  GreenGuard guard;
};

class JointCurveModel {
 public:
  JointCurveModel(const PipelineConfig& cfg, const CurveFrame& f, const float* = nullptr, const float* = nullptr)
      : cfg_(cfg), f_(f), rows_(3 * f.n), L_(f.n), ug_(f.n), ub_(f.n) {
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

  // Slope shrink: s0 of sd^2 / (sd^2 + s0^2), in L*; most of a slope measured over an L* spread well under s0 is kept.
  static constexpr float kSlopeSpanL = 6.0f;
  // Evidence hand-over: the evidence mass (sum of p^2) at which the fit's correction is applied at half strength
  // (w = m / (m + this)), so that a correction resting on very little neutral evidence is not trusted. A policy
  // value, not derived.
  // TODO: the value was checked on few low-evidence frames; re-check it on the scans that arrive through the web
  // portal as they accumulate (their evidence distribution, and the corrections of low-evidence scans at 6, 12, 24
  // and 48).
  static constexpr double kEvidenceHalfMass = 12.0;
  static double hand_over_weight(double evidence) { return evidence / (evidence + kEvidenceHalfMass); }

  // Builds the scene-neutral anchors from the rendered scan cells Lab0(i) at (e0, g0, b0): every cell Auto renders at
  // L* 2..92 is an anchor, weighted by the neutral-patch head's p(neutral) squared (CurveFrame::neutral_p), with no
  // threshold, no minimum count and no balancing between tone ranges; the weights are normalised to sum to 1. Their
  // sum of p^2 before that is the evidence. The fit then removes the anchors' cast in full: the offset residuals aim
  // at a* = b* = 0, and the slope residuals at (1 - f) k0, k0 the anchors' slope at Auto and f = sd^2 / (sd^2 +
  // kSlopeSpanL^2), sd the anchors' weighted L* standard deviation (the slope shrink: a slope measured over a narrow
  // tone range is not extrapolated into a bend across the whole range).
  //
  // Without the head's p (no head loaded, or no inference), or when every p^2 is 0, there are no anchors
  // (has_anchors() false): the neutral residuals are 0, and the joint fit stays at its start, Auto's g and b with no
  // G/B bend (E still solved to Auto's median L*; the re-meter and the steps after the fit run as on any frame).
  // decisions() records the evidence, the hand-over weight and the slope shrink.
  void set_scene_neutral(float e0, float g0, float b0) {
    neu_idx_.clear();
    neu_w_.clear();
    has_anchors_ = false;
    w_off_ = w_slope_ = 0.0f;
    k_target_a_ = k_target_b_ = 0.0f;
    decisions_ = NeutralTargetDecisions{};
    if (f_.n <= 0 || !f_.neutral_p) return;

    const InputCurves zero_c{};
    std::vector<Lab> lab0(f_.n);
    std::vector<float> c0(f_.n);
    for (int i = 0; i < f_.n; ++i) {
      lab0[i] = cell_lab(i, e0, g0, b0, zero_c);
      const float c = std::hypot(lab0[i].a, lab0[i].b);
      c0[i] = std::isfinite(c) && std::isfinite(lab0[i].L) ? c : INFINITY;
    }

    std::vector<int> bands[4];
    int total = 0;
    for (int i = 0; i < f_.n; ++i) {
      if (!std::isfinite(c0[i])) continue;
      const float L = lab0[i].L;
      if (L > 2.0f && L < 92.0f) {
        if (L <= 22.0f) bands[0].push_back(i);
        else if (L <= 45.0f) bands[1].push_back(i);
        else if (L <= 70.0f) bands[2].push_back(i);
        else bands[3].push_back(i);
        ++total;
      }
    }
    if (total < 8) return;

    std::vector<int> b_sel[4];
    std::vector<float> pw(f_.n, 0.0f);  // p^2 per cell
    double mass = 0.0;
    for (int b = 0; b < 4; ++b) {
      for (int i : bands[b]) {
        const float p = std::clamp(f_.neutral_p[i], 0.0f, 1.0f);
        pw[i] = p * p;
        if (pw[i] > 0.0f) {
          b_sel[b].push_back(i);
          mass += pw[i];
        }
      }
    }
    decisions_.evidence = (float)mass;
    decisions_.hand_over = (float)hand_over_weight(decisions_.evidence);

    float w_sum = 0.0f;
    for (int b = 0; b < 4; ++b) {
      for (int i : b_sel[b]) {
        neu_idx_.push_back(i);
        neu_w_.push_back(pw[i]);
        w_sum += pw[i];
      }
    }
    if (!(w_sum > 0.0f)) {
      neu_idx_.clear();
      neu_w_.clear();
      return;
    }
    for (float& w : neu_w_) w /= w_sum;
    has_anchors_ = true;
    const int m = (int)neu_idx_.size();

    double sL = 0.0, sa = 0.0, sb = 0.0;
    for (int k = 0; k < m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      sL += w * lab0[i].L;
      sa += w * lab0[i].a;
      sb += w * lab0[i].b;
    }
    double vL0 = 0.0, covLa0 = 0.0, covLb0 = 0.0;
    for (int k = 0; k < m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      const double dL = lab0[i].L - sL;
      vL0 += w * dL * dL;
      covLa0 += w * dL * (lab0[i].a - sa);
      covLb0 += w * dL * (lab0[i].b - sb);
    }
    const double ka0 = 50.0 * covLa0 / (vL0 + 25.0);
    const double kb0 = 50.0 * covLb0 / (vL0 + 25.0);

    w_off_ = 1.0f / 1.5f;
    w_slope_ = std::sqrt((float)(vL0 / (vL0 + 100.0))) / 2.0f;

    // Slope shrink: the anchors' weighted L* spread (their weights renormalised over them; a separate pass from vL0,
    // which is centred on a mean summed in another order).
    double w_span = 0.0, L_span = 0.0;
    for (int k = 0; k < m; ++k) {
      w_span += neu_w_[k];
      L_span += neu_w_[k] * lab0[neu_idx_[k]].L;
    }
    L_span /= w_span;
    double vL_span = 0.0;
    for (int k = 0; k < m; ++k) {
      const double dL = lab0[neu_idx_[k]].L - L_span;
      vL_span += neu_w_[k] * dL * dL;
    }
    vL_span /= w_span;
    const double s0_sq = (double)kSlopeSpanL * (double)kSlopeSpanL;
    const float slope_share = (float)(vL_span / (vL_span + s0_sq));
    k_target_a_ = (float)((1.0 - slope_share) * ka0);
    k_target_b_ = (float)((1.0 - slope_share) * kb0);
    decisions_.span_L = (float)std::sqrt(vL_span);
    decisions_.slope_share = slope_share;

  }

  // set_scene_neutral found anchors (the head's p, with some evidence).
  bool has_anchors() const { return has_anchors_; }
  // What set_scene_neutral decided: the evidence, the hand-over weight and the slope shrink.
  const NeutralTargetDecisions& decisions() const { return decisions_; }

  NeutralStats neutral_stats(float e, float g, float b, const InputCurves& shape) {
    const InputCurves c = anchored(shape, e, g, b);
    return neutral_stats_anchored(e, g, b, c);
  }

  NeutralStats neutral_stats_on(const PipelineConfig& cfg, float e, float g, float b) const {
    NeutralStats st;
    st.m = (int)neu_idx_.size();
    if (st.m <= 0) return st;
    float m[9];
    candidate_matrix(f_.merged, e, g, b, m);
    std::vector<Lab> labs(st.m);
    double sL = 0.0, sa = 0.0, sb = 0.0;
    for (int k = 0; k < st.m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      labs[k] = linear_srgb_to_lab(to_display(cfg, convert_pixel_linear(cfg, m, f_.grid + 3 * i)));
      sL += w * labs[k].L;
      sa += w * labs[k].a;
      sb += w * labs[k].b;
    }
    return compute_neutral_moments(labs, sL, sa, sb, st);
  }

  // 4-equation scene-neutral residuals: [w_off * a_bar, w_off * b_bar, w_slope * (k_a - kt_a),
  // w_slope * (k_b - kt_b)], kt the slope target set_scene_neutral's slope shrink left.
  double residuals(float e, float g, float b, const InputCurves& shape, std::vector<double>& r) {
    const NeutralStats st = neutral_stats(e, g, b, shape);
    r.assign(4, 0.0);
    if (st.m <= 0 || !has_anchors_) return 0.0;
    r[0] = (double)w_off_ * st.a_bar;
    r[1] = (double)w_off_ * st.b_bar;
    r[2] = (double)w_slope_ * (st.k_a - k_target_a_);
    r[3] = (double)w_slope_ * (st.k_b - k_target_b_);
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
  }

  // E with the grid's median L* at target_L within [e_lo, e_hi] (a larger E renders darker: the median falls with log
  // E, each cell's L* does). Secant steps on log E from e_warm (clamped into the bounds), each new point kept in a
  // bracket of the two nearest points either side of the target; where the secant leaves the bracket or stalls, the
  // bounds the bracket lacks are tested and the bracket bisected. False when the target lies outside the bounds (the
  // median at one bound is already past it): e is that bound, and the point is not one the export can render at
  // target_L.
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
  NeutralStats compute_neutral_moments(const std::vector<Lab>& labs, double sL, double sa, double sb,
                                       NeutralStats& st) const {
    double vL = 0.0, sLa = 0.0, sLb = 0.0, sz3 = 0.0;
    for (int k = 0; k < st.m; ++k) {
      const double w = neu_w_[k];
      const double dL = labs[k].L - sL;
      const double z = dL / 50.0;
      vL += w * dL * dL;
      sz3 += w * z * z * z;
      sLa += w * dL * (labs[k].a - sa);
      sLb += w * dL * (labs[k].b - sb);
    }
    const double ka = 50.0 * sLa / (vL + 25.0);
    const double kb = 50.0 * sLb / (vL + 25.0);
    const double vz = vL / 2500.0;
    const double mu3 = sz3 / (vz + 1e-4);
    double vq = 0.0, sqa = 0.0, sqb = 0.0;
    for (int k = 0; k < st.m; ++k) {
      const double w = neu_w_[k];
      const double z = (labs[k].L - sL) / 50.0;
      const double uq = 4.0 * (z * z - vz - mu3 * z);
      const double ea = (labs[k].a - sa) - ka * z;
      const double eb = (labs[k].b - sb) - kb * z;
      vq += w * uq * uq;
      sqa += w * uq * ea;
      sqb += w * uq * eb;
    }
    st.a_bar = (float)sa;
    st.b_bar = (float)sb;
    st.k_a = (float)ka;
    st.k_b = (float)kb;
    st.q_a = (float)(sqa / (vq + 0.04));
    st.q_b = (float)(sqb / (vq + 0.04));
    return st;
  }

  NeutralStats neutral_stats_anchored(float e, float g, float b, const InputCurves& c) const {
    NeutralStats st;
    st.m = (int)neu_idx_.size();
    if (st.m <= 0) return st;
    std::vector<Lab> labs(st.m);
    double sL = 0.0, sa = 0.0, sb = 0.0;
    for (int k = 0; k < st.m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      labs[k] = cell_lab(i, e, g, b, c);
      sL += w * labs[k].L;
      sa += w * labs[k].a;
      sb += w * labs[k].b;
    }
    return compute_neutral_moments(labs, sL, sa, sb, st);
  }

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
  float fb_[3] = {0.0f, 0.0f, 0.0f};
  float w_off_ = 0.0f, w_slope_ = 0.0f;
  float k_target_a_ = 0.0f, k_target_b_ = 0.0f;  // the slope residuals' target (slope shrink)
  NeutralTargetDecisions decisions_;
  std::vector<float> rows_, L_, ug_, ub_;
  std::vector<int> neu_idx_;
  std::vector<float> neu_w_;
  bool has_anchors_ = false;
};

// The never-greener guard: the joint fit may move the grid's colour anywhere the neutral term asks, except greener
// than Auto. Over the grid cells Auto renders at L* 2..92 (one band: the whole tone range), the candidate's mean a*
// over all cells, and its mean a* weighted by the neutral head's p^2, may not fall more than kGreenGuardTolAb below
// Auto's. The bound applies to the result, not to the solver's path: the fit runs first without the guard, and that
// result stands when neither counted mean ends more than the tolerance greener than Auto's. Otherwise the fit runs
// again from the same start with each excess in the Levenberg-Marquardt residuals as kGreenGuardWeight x
// max(0, a*_Auto - tol - a*), 0 at Auto and wherever the candidate is not greener: a penalty in the objective rather
// than a clamp afterwards. (A hinge on every evaluation also blocked steps that pass through a slightly greener point
// on the way to a less green result, stalling the fit at its start.) A term counts with at least kGreenGuardMinCells
// cells, the weighted one with an evidence mass (sum p^2) of at least kGreenGuardMinMass.
// Shadows carry no debit (green_guard_cell_a): in both means, a cell Auto renders in the deep shadows that ends greener
// than at Auto counts at Auto's a*, so the fit may correct a shadow cast toward green; a shadow cell that ends more
// magenta keeps its credit, and mid-tones and highlights count as they are. The tint trim's guard check is not
// affected.
constexpr float kGreenGuardTolAb = 0.5f;
// The shadow exemption's weight w_s = 1 - smoothstep((L*_Auto - kGreenGuardShadowL0) / kGreenGuardShadowWidth + 0.5):
// 1 in the deep shadows, 0.5 at L* 15, 0 above L* 20.
constexpr double kGreenGuardShadowL0 = 15.0;
constexpr double kGreenGuardShadowWidth = 10.0;
// A cell's a* as the guard's means count it: a_Auto + d, d = a - a_Auto, with a negative d (the cell greener than at
// Auto) scaled by 1 - w_s, w_s the shadow weight of the cell's L* at Auto.
inline double green_guard_cell_a(double a, double a_auto, double L_auto) {
  if (!std::isfinite(a) || !std::isfinite(a_auto)) return a;
  const double d = a - a_auto;
  if (d >= 0.0) return a;
  const double x = std::clamp((L_auto - kGreenGuardShadowL0) / kGreenGuardShadowWidth + 0.5, 0.0, 1.0);
  const double ws = 1.0 - x * x * (3.0 - 2.0 * x);
  return a_auto + (1.0 - ws) * d;
}
constexpr float kGreenGuardWeight = 4.0f;
constexpr double kGreenGuardMinCells = 8.0;
constexpr double kGreenGuardMinMass = 2.0;

// The never-greener guard on one pass of the joint fit: its cells (Auto's render at L* 2..92), their p^2, Auto's a*
// and L* per cell, Auto's two means (ref) and which of them count. set_cells() takes the cells from Auto's render once
// Auto's E is solved; means() gives a candidate's two means; hinge() a mean's excess as the residual the guarded pass
// adds; binds() whether a result breaks the bound. (The tint trim's guard check differs in its tolerance, its L* edge,
// the shadow exemption and how it compares the means, so it keeps its own.)
struct JointFitGuard {
  std::vector<char> in;
  std::vector<double> pw, auto_a, auto_L;  // p^2; Auto's a* (a cell a candidate renders non-finite counts at it) and L*
  double ref[2] = {0.0, 0.0};
  bool counted[2] = {false, false};
  bool ready = false;  // set_cells() and ref done: the residuals carry the guard's entries from then on

  // lab_auto(i): cell i's Lab on Auto's render; neutral_p: the head's p per cell, or null.
  template <typename AutoLab>
  void set_cells(int n, AutoLab lab_auto, const float* neutral_p) {
    in.assign(n, 0);
    pw.assign(n, 0.0);
    auto_a.assign(n, 0.0);
    auto_L.assign(n, 0.0);
    double na = 0.0, wp = 0.0;
    for (int i = 0; i < n; ++i) {
      const Lab l = lab_auto(i);
      if (!std::isfinite(l.L) || !std::isfinite(l.a) || !(l.L > 2.0f && l.L < 92.0f)) continue;
      in[i] = 1;
      auto_a[i] = l.a;
      auto_L[i] = l.L;
      const double p = neutral_p ? std::clamp((double)neutral_p[i], 0.0, 1.0) : 0.0;
      pw[i] = p * p;
      na += 1.0;
      wp += pw[i];
    }
    counted[0] = na >= kGreenGuardMinCells;
    counted[1] = wp >= kGreenGuardMinMass;
  }
  // a_of(i): the candidate's a* of cell i. v: the mean over the guard's cells and the p^2-weighted mean, each cell's
  // a* as green_guard_cell_a counts it.
  template <typename CellA>
  void means(CellA a_of, double v[2]) const {
    double sa = 0.0, na = 0.0, sp = 0.0, wp = 0.0;
    for (int i = 0; i < (int)in.size(); ++i) {
      if (!in[i]) continue;
      // The same cells for every candidate (the ones set_cells counted): a cell a candidate renders non-finite counts
      // at Auto's a*, so the two means never compare different populations.
      double a = a_of(i);
      if (!std::isfinite(a)) a = auto_a[i];
      a = green_guard_cell_a(a, auto_a[i], auto_L[i]);
      sa += a;
      na += 1.0;
      sp += pw[i] * a;
      wp += pw[i];
    }
    v[0] = na > 0 ? sa / na : 0.0;
    v[1] = wp > 0 ? sp / wp : 0.0;
  }
  double hinge(int k, const double v[2]) const {
    return counted[k] ? (double)kGreenGuardWeight * std::max(0.0, ref[k] - kGreenGuardTolAb - v[k]) : 0.0;
  }
  bool binds(int k, const double v[2]) const { return counted[k] && ref[k] - kGreenGuardTolAb - v[k] > 0.0; }
};

// One pass of the joint fit (fit_input_curves_joint); guard_path: the guard's hinge joins the residuals at every
// evaluation. Off, the hinge residuals are 0 (same residual count) and the guard is only measured at the result.
inline JointCurveFit fit_input_curves_joint_pass(const PipelineConfig& cfg, const CurveFrame& f, float e0, float g0,
                                                 float b0, const float* mu, const float* sigma, float target_L,
                                                 const int table_n[2], const CurveFitOptions& o, float e_lo, float e_hi,
                                                 bool guard_path) {
  JointCurveModel model(cfg, f, mu, sigma);
  model.set_scene_neutral(e0, g0, b0);
  if (!(e_lo > 0.0f)) e_lo = e0 * 0.25f;
  if (!(e_hi > e_lo)) e_hi = std::max(e_lo, e0 * 4.0f);
  const float g_lo = o.g_bound_stops > 0.0f ? g0 * std::exp2(-o.g_bound_stops) : g0 - o.g_bound;
  const float g_hi = o.g_bound_stops > 0.0f ? g0 * std::exp2(o.g_bound_stops) : g0 + o.g_bound;
  const float b_lo = o.b_bound_stops > 0.0f ? b0 * std::exp2(-o.b_bound_stops) : b0 - o.b_bound;
  const float b_hi = o.b_bound_stops > 0.0f ? b0 * std::exp2(o.b_bound_stops) : b0 + o.b_bound;
  const float lo[4] = {std::max(kGainFloor, g_lo), std::max(kGainFloor, b_lo), -o.curve_bound, -o.curve_bound};
  const float hi[4] = {g_hi, b_hi, o.curve_bound, o.curve_bound};
  auto shape_of = [&](const float* p) {
    InputCurves c;
    c.hold = o.hold_slope;
    c.a[0] = p[2];
    c.k[0] = 0.0f;
    c.a[1] = p[3];
    c.k[1] = 0.0f;
    return c;
  };
  JointCurveFit out;
  out.neutral_before = model.neutral_stats(e0, g0, b0, InputCurves{});
  int renders = 0;
  const double sqrt_lambda = std::sqrt(std::max(0.0, static_cast<double>(o.joint_lambda)));

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

  // The guard (JointFitGuard), set once Auto's E is solved (guard_init), before which evaluate adds no guard
  // residuals.
  JointFitGuard guard;
  auto guard_means = [&](const float* q, float eq, double v[2]) {
    const InputCurves c = model.anchored(shape_of(q), eq, q[0], q[1]);
    guard.means([&](int i) { return (double)model.cell_lab(i, eq, q[0], q[1], c).a; }, v);
  };
  auto guard_init = [&](float eq) {
    const InputCurves c = model.anchored(InputCurves{}, eq, g0, b0);
    guard.set_cells(f.n, [&](int i) { return model.cell_lab(i, eq, g0, b0, c); }, f.neutral_p);
    const float p0[4] = {g0, b0, 0.0f, 0.0f};
    guard_means(p0, eq, guard.ref);
    out.guard.counted[0] = guard.counted[0];
    out.guard.counted[1] = guard.counted[1];
    guard.ready = true;
  };
  auto evaluate = [&](const float* p, float e_warm, float& e, std::vector<double>& r, double& cost, double& data,
                      bool check_shape) {
    ++out.evaluations;
    if (!solve_point(p, e_warm, e, check_shape)) return false;
    data = model.residuals(e, p[0], p[1], shape_of(p), r);
    double sum = data;
    if (guard.ready) {
      // The hinges are 0 on the unguarded pass (the guard is measured at its result only), so the means are needed
      // only on the guarded one; the residual vector keeps its two guard entries either way.
      double v[2] = {0.0, 0.0};
      if (guard_path) guard_means(p, e, v);
      for (int k = 0; k < 2; ++k) {
        const double h = guard_path ? guard.hinge(k, v) : 0.0;
        r.push_back(h);
        sum += h * h;
      }
    }
    const size_t nd = r.size();
    r.resize(nd + 2);
    for (int j = 0; j < 2; ++j) {
      r[nd + j] = sqrt_lambda * p[2 + j];
      sum += r[nd + j] * r[nd + j];
    }
    cost = sum;
    return std::isfinite(cost);
  };

  auto cut_back = [&](const float* p, const float* q, float e, float& e_t) {
    float t_ok = 0.0f, t_bad = 1.0f;
    e_t = e;
    for (int it = 0; it < 10; ++it) {
      const float t = 0.5f * (t_ok + t_bad);
      float qt[4];
      for (int j = 0; j < 4; ++j) qt[j] = p[j] + t * (q[j] - p[j]);
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

  float p[4] = {g0, b0, 0.0f, 0.0f}, e = e0;
  std::vector<double> r, rt, rj;
  double cost = 0.0, data = 0.0;
  if (!evaluate(p, e0, e, r, cost, data, true)) {
    out.feasible = false;
    out.converged = false;
    out.g = g0;
    out.b = b0;
    out.e = e0;
    out.curves = model.anchored(shape_of(p), e0, g0, b0);
    out.neutral_after = out.neutral_before;
    out.renders = renders;
    return out;
  }
  out.feasible = true;
  out.loss_before = data;
  // The guard's reference is Auto at its solved E; Auto's point again with the guard's residuals (all 0 there).
  guard_init(e);
  evaluate(p, e, e, r, cost, data, true);

  const int m = 4;
  double lambda = 1e-3;
  bool settled = (!model.has_anchors() || data <= 1e-6);
  for (int iter = 0; iter < 25 && !settled; ++iter) {
    const size_t nr = r.size();
    std::vector<double> J(nr * m, 0.0);
    for (int j = 0; j < m; ++j) {
      float q[4];
      std::copy(p, p + 4, q);
      const float h = (q[j] + 1e-3f <= hi[j]) ? 1e-3f : -1e-3f;
      q[j] += h;
      float ej = e;
      double cj = 0.0, dj = 0.0;
      if (!evaluate(q, e, ej, rj, cj, dj, false) || rj.size() != nr) continue;
      for (size_t k = 0; k < nr; ++k) J[k * m + j] = (rj[k] - r[k]) / h;
    }
    double A[4][4] = {}, gvec[4] = {};
    for (size_t k = 0; k < nr; ++k) {
      for (int a = 0; a < m; ++a) {
        gvec[a] += J[k * m + a] * r[k];
        for (int bb = a; bb < m; ++bb) A[a][bb] += J[k * m + a] * J[k * m + bb];
      }
    }
    for (int a = 0; a < m; ++a)
      for (int bb = 0; bb < a; ++bb) A[a][bb] = A[bb][a];
    for (int a = 0; a < m; ++a) {
      if ((p[a] <= lo[a] + 1e-5f && gvec[a] > 0.0) || (p[a] >= hi[a] - 1e-5f && gvec[a] < 0.0)) {
        for (int bb = 0; bb < m; ++bb) A[a][bb] = A[bb][a] = 0.0;
        A[a][a] = 1.0;
        gvec[a] = 0.0;
      }
    }

    bool accepted = false;
    for (int tries = 0; tries < 10 && !accepted; ++tries) {
      double M[4][5];
      for (int a = 0; a < m; ++a) {
        for (int bb = 0; bb < m; ++bb) M[a][bb] = A[a][bb] + (a == bb ? lambda * std::max(A[a][a], 1e-6) : 0.0);
        M[a][4] = -gvec[a];
      }
      for (int c = 0; c < m; ++c) {
        int piv = c;
        for (int rr = c + 1; rr < m; ++rr)
          if (std::fabs(M[rr][c]) > std::fabs(M[piv][c])) piv = rr;
        for (int cc = 0; cc < 5; ++cc) std::swap(M[c][cc], M[piv][cc]);
        if (std::fabs(M[c][c]) < 1e-300) M[c][c] = 1e-300;
        for (int rr = c + 1; rr < m; ++rr) {
          const double fct = M[rr][c] / M[c][c];
          for (int cc = c; cc < 5; ++cc) M[rr][cc] -= fct * M[c][cc];
        }
      }
      double delta[4];
      for (int c = m - 1; c >= 0; --c) {
        double sum = M[c][4];
        for (int cc = c + 1; cc < m; ++cc) sum -= M[c][cc] * delta[cc];
        delta[c] = sum / M[c][c];
      }
      float q[4];
      float step = 0.0f;
      for (int j = 0; j < m; ++j) {
        q[j] = std::min(hi[j], std::max(lo[j], p[j] + (float)delta[j]));
        step = std::max(step, std::fabs(q[j] - p[j]));
      }
      if (step < 1e-5f) {
        settled = true;
        break;
      }
      float eq = e;
      double cq = 0.0, dq = 0.0;
      bool feasible = false;
      for (float step_s : {1.0f, 0.5f, 0.25f, 0.1f, 0.05f}) {
        float qt[4];
        for (int j = 0; j < m; ++j) {
          qt[j] = std::min(hi[j], std::max(lo[j], p[j] + step_s * (float)delta[j]));
        }
        float eq_t = e;
        double cq_t = 0.0, dq_t = 0.0;
        bool ok = evaluate(qt, e, eq_t, rt, cq_t, dq_t, true);
        if (!ok && step_s == 1.0f) {
          float e_t = e;
          const float t_ok = cut_back(p, qt, e, e_t);
          if (t_ok > 0.0f) {
            for (int j = 0; j < m; ++j) qt[j] = p[j] + t_ok * (qt[j] - p[j]);
            ok = evaluate(qt, e_t, eq_t, rt, cq_t, dq_t, true);
          }
        }
        if (ok && cq_t < cost) {
          std::copy(qt, qt + 4, q);
          eq = eq_t;
          cq = cq_t;
          dq = dq_t;
          feasible = true;
          break;
        }
      }
      if (feasible && cq < cost) {
        const double gain = cost - cq;
        std::copy(q, q + 4, p);
        e = eq;
        r.swap(rt);
        cost = cq;
        data = dq;
        lambda = std::max(lambda / 3.0, 1e-9);
        accepted = true;
        ++out.iterations;
        if (gain < 1e-6 * std::max(1.0, cost) || data < 1e-4) settled = true;
      } else {
        lambda *= 4.0;
      }
    }
    if (!accepted && !settled) break;
  }
  // The evidence hand-over: the solved (ln g, ln b, a_G, a_B) moves from Auto's (g0, b0, 0, 0) by w, E solved there
  // again (the result stays at the solved point if that E cannot be found). The guard below measures this result.
  {
    const double w = JointCurveModel::hand_over_weight(model.decisions().evidence);
    float q[4] = {(float)(g0 * std::pow((double)p[0] / g0, w)), (float)(b0 * std::pow((double)p[1] / b0, w)),
                  (float)(w * p[2]), (float)(w * p[3])};
    float eq = e;
    if (solve_point(q, e, eq, false)) {
      std::copy(q, q + 4, p);
      e = eq;
    }
  }
  out.converged = settled;
  out.g = p[0];
  out.b = p[1];
  out.e = e;
  out.curves = model.anchored(shape_of(p), e, p[0], p[1]);
  {
    double v[2];
    guard_means(p, e, v);
    for (int k = 0; k < 2; ++k) {
      out.guard.auto_a[k] = (float)guard.ref[k];
      out.guard.fit_a[k] = (float)v[k];
      if (guard.binds(k, v)) out.guard.binding = true;
    }
    // Per band of Auto's render, for the sidecar: Auto's cells as guard_init rendered them (at Auto's E, as auto_a),
    // the fit's at its result.
    const InputCurves c_fit = model.anchored(shape_of(p), e, p[0], p[1]);
    double s[4][4] = {}, n[4] = {}, w[4] = {};
    for (int i = 0; i < f.n; ++i) {
      if (!guard.in[i]) continue;
      const Lab lf = model.cell_lab(i, e, p[0], p[1], c_fit);
      if (!std::isfinite(lf.a)) continue;
      const double la_a = guard.auto_a[i], la_L = guard.auto_L[i], pw = guard.pw[i];
      const int bd = la_L <= 22.0 ? 0 : la_L <= 45.0 ? 1 : la_L <= 70.0 ? 2 : 3;
      n[bd] += 1.0;
      w[bd] += pw;
      s[bd][0] += la_a;
      s[bd][1] += pw * la_a;
      s[bd][2] += lf.a;
      s[bd][3] += pw * lf.a;
    }
    for (int bd = 0; bd < 4; ++bd) {
      out.guard.band_cells[bd] = (int)n[bd];
      out.guard.band_mass[bd] = (float)w[bd];
      if (n[bd] > 0) {
        out.guard.band_auto_a[bd][0] = (float)(s[bd][0] / n[bd]);
        out.guard.band_fit_a[bd][0] = (float)(s[bd][2] / n[bd]);
      }
      if (w[bd] > 0) {
        out.guard.band_auto_a[bd][1] = (float)(s[bd][1] / w[bd]);
        out.guard.band_fit_a[bd][1] = (float)(s[bd][3] / w[bd]);
      }
    }
  }

  out.neutral_after = model.neutral_stats(e, p[0], p[1], out.curves);
  out.loss_after = data;
  out.renders = renders;
  return out;
}

// Stage 2, the curve fit: removes what is left of the colour cast, and its change from shadows to highlights, on the
// bracket Auto chose. It takes the grid cells, the film base, Auto's (E, g, b) and the neutral head's probabilities,
// builds p^2-weighted neutral anchors (JointCurveModel) and solves p = (g, b, a_G, a_B) by Levenberg-Marquardt, E
// solved at each candidate to keep target_L, then hands the solved correction over by the anchors' evidence
// (kEvidenceHalfMass). The fit runs first without the never-greener guard (kGreenGuard*); when its handed-over result
// is greener than Auto's (deep shadows exempt) it runs again with it. It hands on the curves and the fitted (E, g, b),
// which run_white_light_fit re-meters on the combined profile. evaluations and renders count both passes.
inline JointCurveFit fit_input_curves_joint(const PipelineConfig& cfg, const CurveFrame& f, float e0, float g0, float b0,
                                            const float* mu, const float* sigma, float target_L, const int table_n[2],
                                            const CurveFitOptions& o = CurveFitOptions(), float e_lo = 0.0f,
                                            float e_hi = 0.0f) {
  const JointCurveFit free_fit =
      fit_input_curves_joint_pass(cfg, f, e0, g0, b0, mu, sigma, target_L, table_n, o, e_lo, e_hi, false);
  JointCurveFit out = free_fit;
  if (free_fit.guard.binding) {
    out = fit_input_curves_joint_pass(cfg, f, e0, g0, b0, mu, sigma, target_L, table_n, o, e_lo, e_hi, true);
    out.guard.refit = true;
    out.evaluations += free_fit.evaluations;
    out.renders += free_fit.renders;
  }
  out.guard.free_fit_a[0] = free_fit.guard.fit_a[0];
  out.guard.free_fit_a[1] = free_fit.guard.fit_a[1];
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
// over [e_lo, e_hi]; where the target lies outside those too, *clipped is set and E is the bound it reached. It is used
// instead of a fixed point on E, g and b after the fit, which would follow its own darker render.
inline float solve_e_for_median_L(PipelineConfig cfg, const CurveFrame& f, const InputCurves& shape, float e_centre,
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
  return solve_e_for_median_L(cfg, f, shape, e0, e0 * 0.25f, e0 * 4.0f, g, b, target_L, nullptr, iters);
}

}  // namespace negicc

#endif  // NEGICC_INPUT_CURVES_H
