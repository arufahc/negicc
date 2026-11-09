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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstdint>
#include <vector>

#include "color_pipeline.h"

namespace negicc {

struct InputCurves {
  static constexpr int kMaxKnots = 4;
  static constexpr int kMaxSegs = kMaxKnots + 1;
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
  // Red highlight cliff parameters when k[0] < 0 (windowed [t_s, 0.76] highlight cliff de-surging).
  float k_r = 0.0f;
  float ref_r = 0.0f;
  float u_lo_r = 0.0f;
  // Phase 2C monotone piecewise-linear knot curve in log2 transmittance u = log2(v / ref) <= 0:
  // n_knots > 0 activates the knot curve with knots 0 > u_knot[ch][0] > ... > u_knot[ch][n_knots - 1] and segment
  // slope offsets a_seg[ch][0 .. n_knots] (slope on segment m is 1 + a_seg[ch][m]).
  int n_knots = 0;
  float u_knot[2][kMaxKnots] = {};
  float a_seg[2][kMaxSegs] = {};
  float u_hold(int i) const { return hold ? u_lo[i] : -INFINITY; }
  float u_hold_r() const { return hold ? (u_lo_r < 0.0f ? u_lo_r : u_lo[0]) : -INFINITY; }
  bool active() const {
    if (n_knots > 0) {
      for (int ch = 0; ch < 2; ++ch)
        for (int s = 0; s <= n_knots && s < kMaxSegs; ++s)
          if (a_seg[ch][s] != 0.0f) return true;
    }
    return a[0] != 0.0f || k[0] != 0.0f || a[1] != 0.0f || k[1] != 0.0f || k_r != 0.0f;
  }
};

inline float curve_stops_knots(float u, int n_knots, const float* u_knot, const float* a_seg,
                               float u_hold = -INFINITY) {
  auto eval_at = [&](float uu) {
    float f = 0.0f;
    float prev_u = 0.0f;
    for (int m = 0; m < n_knots; ++m) {
      const float uk = u_knot[m];
      const float slope = std::max(0.15f, 1.0f + a_seg[m]);
      if (uu >= uk) return f + slope * (uu - prev_u);
      f += slope * (uk - prev_u);
      prev_u = uk;
    }
    const float slope_last = std::max(0.15f, 1.0f + a_seg[n_knots]);
    return f + slope_last * (uu - prev_u);
  };
  if (u < u_hold) return eval_at(u_hold) + (u - u_hold);
  return eval_at(u);
}

inline float cliff_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return std::exp(x);
  return std::log1p(std::exp(x));
}

inline float trc_f_at_level(const float* in, int n, float ref, float t_lvl) {
  if (!in || n < 2 || !(ref > 0.0f)) return -2.0f;
  float lo = -8.0f, hi = 0.0f;
  for (int it = 0; it < 36; ++it) {
    const float mid = 0.5f * (lo + hi);
    const float v = std::min(1.0f, std::max(0.0f, ref * std::exp2(mid)));
    if (apply_trc(v, in, n) > t_lvl) lo = mid;
    else hi = mid;
  }
  return 0.5f * (lo + hi);
}

// f(u) = u s(u) for k >= 0; for k < 0, windowed [f_start, f_end] softplus cliff ramp with midtone slope 1 + a and
// cliff slope reduction 1 + k, held below u_hold at f(u_hold) + (u - u_hold) (u_hold = -inf: not held).
inline float curve_stops(float u, float u_med, float a, float k, float u_hold = -INFINITY, float f_start = -1.78f,
                         float f_end = -2.51f) {
  if (k < 0.0f && f_end < f_start) {
    auto eval_cliff = [&](float uu) {
      const float s0 = std::max(1.0f + a, 0.25f);
      const float s1 = std::max(1.0f + k, 0.20f);
      const float w = 0.12f;
      const float u0 = f_start / s0;
      const float u1 = u0 + (f_end - f_start) / (s0 * s1);
      const float sp0 = cliff_softplus(u0 / w) - cliff_softplus(u1 / w);
      const float ramp = w * (cliff_softplus(-(uu - u0) / w) - cliff_softplus(-(uu - u1) / w) - sp0);
      return s0 * (uu - k * ramp);
    };
    if (u < u_hold) return eval_cliff(u_hold) + (u - u_hold);
    return eval_cliff(u);
  }
  if (u < u_hold) return u_hold * (1.0f + a + k * std::tanh(u_hold - u_med)) + (u - u_hold);
  return u * (1.0f + a + k * std::tanh(u - u_med));
}

inline float input_curve(float v, float ref, float u_med, float a, float k, float u_hold = -INFINITY,
                         float f_start = -1.78f, float f_end = -2.51f) {
  if (!(v < ref) || (a == 0.0f && k == 0.0f)) return v;
  const float vv = std::max(v, 1e-12f);
  const float u = std::min(std::log2(vv / ref), 0.0f);
  return ref * std::exp2(curve_stops(u, u_med, a, k, u_hold, f_start, f_end));
}

inline float input_curve_ch(const InputCurves& c, int ch, float v, float f_start = -1.78f, float f_end = -2.51f) {
  const float ref = c.ref[ch];
  if (!(v < ref)) return v;
  if (c.n_knots > 0) {
    bool any = false;
    for (int s = 0; s <= c.n_knots && s < InputCurves::kMaxSegs; ++s)
      if (c.a_seg[ch][s] != 0.0f) any = true;
    if (!any) return v;
    const float vv = std::max(v, 1e-12f);
    const float u = std::min(std::log2(vv / ref), 0.0f);
    return ref * std::exp2(curve_stops_knots(u, c.n_knots, c.u_knot[ch], c.a_seg[ch], c.u_hold(ch)));
  }
  return input_curve(v, ref, c.u_med[ch], c.a[ch], c.k[ch], c.u_hold(ch), f_start, f_end);
}

// out[i] = in(curve(i / (n - 1))), in sampled with apply_trc's linear interpolation.
inline void compose_input_table(const float* in, int n, float ref, float u_med, float a, float k, float* out,
                                float u_hold = -INFINITY, float t_s = 0.30f, float t_e = 0.76f) {
  float f_start = -1.78f, f_end = -2.51f;
  if (k < 0.0f && in && n >= 2 && ref > 0.0f) {
    f_start = trc_f_at_level(in, n, ref, t_s);
    f_end = trc_f_at_level(in, n, ref, t_e);
  }
  for (int i = 0; i < n; ++i) {
    float v = input_curve(static_cast<float>(i) / static_cast<float>(n - 1), ref, u_med, a, k, u_hold, f_start, f_end);
    v = std::min(1.0f, std::max(0.0f, v));
    out[i] = apply_trc(v, in, n);
  }
}

inline void compose_input_table_ch(const float* in, int n, const InputCurves& c, int ch, float* out,
                                   float t_s = 0.30f, float t_e = 0.76f) {
  if (c.n_knots > 0) {
    for (int i = 0; i < n; ++i) {
      float v = input_curve_ch(c, ch, static_cast<float>(i) / static_cast<float>(n - 1));
      v = std::min(1.0f, std::max(0.0f, v));
      out[i] = apply_trc(v, in, n);
    }
    return;
  }
  compose_input_table(in, n, c.ref[ch], c.u_med[ch], c.a[ch], c.k[ch], out, c.u_hold(ch), t_s, t_e);
}

// The combined profile's input tables (owned) and the ProfileData pointing at them; everything else points at the base
// profile, which must outlive it.
struct CombinedProfile {
  std::vector<float> in_r, in_g, in_b;
  ProfileData data{};

  void build(const ProfileData& base, const InputCurves& c) {
    data = base;
    if (!c.active() || !base.has_profile) return;
    const int nr = base.in_trc_size[0], ng = base.in_trc_size[1], nb = base.in_trc_size[2];
    const float kr = (c.n_knots > 0) ? 0.0f : ((c.k_r != 0.0f) ? c.k_r : (c.k[0] < 0.0f ? 0.8228f * c.k[0] : 0.0f));
    if (kr < 0.0f && nr > 1 && base.in_trc[0]) {
      const float refr = (c.ref_r > 0.0f) ? c.ref_r : std::min(1.0f, c.ref[0] * 1.0905f);
      in_r.resize(nr);
      compose_input_table(base.in_trc[0], nr, refr, 0.0f, 0.0f, kr, in_r.data(), c.u_hold_r(), 0.36f, 0.76f);
      data.in_trc[0] = in_r.data();
    }
    if (ng > 1 && base.in_trc[1]) {
      in_g.resize(ng);
      compose_input_table_ch(base.in_trc[1], ng, c, 0, in_g.data(), 0.30f, 0.76f);
      data.in_trc[1] = in_g.data();
    }
    if (nb > 1 && base.in_trc[2]) {
      in_b.resize(nb);
      compose_input_table_ch(base.in_trc[2], nb, c, 1, in_b.data(), 0.46f, 0.76f);
      data.in_trc[2] = in_b.data();
    }
  }
};

// ref and u_med for G and B (and ref_r, u_lo_r for R): the film base and the grid's cells through the merged input
// matrix at (e, g, b) and the knee. merged: the 3x3 input matrix before the per-candidate rows (candidate_matrix);
// film_base: camera units (as the grid). grid: n cells of 16-bit camera-linear RGB.
inline void input_curve_reference(const float* merged, float e, float g, float b, float knee, const float film_base[3],
                                  const uint16_t* grid, int n, InputCurves& c) {
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  const Rgb fb = apply_matrix(m, {film_base[0] / 65535.0f, film_base[1] / 65535.0f, film_base[2] / 65535.0f});
  c.ref_r = knee_compress(fb.r, knee);
  const float refs[2] = {knee_compress(fb.g, knee), knee_compress(fb.b, knee)};
  std::vector<float> ur(n), ug(n), ub(n);
  for (int i = 0; i < n; ++i) {
    const Rgb v = apply_matrix(m, load_u16(grid + 3 * i));
    const float vr = std::max(knee_compress(v.r, knee), 1e-12f);
    const float vg = std::max(knee_compress(v.g, knee), 1e-12f), vb = std::max(knee_compress(v.b, knee), 1e-12f);
    ur[i] = std::min(std::log2(vr / std::max(c.ref_r, 1e-12f)), 0.0f);
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
  c.u_lo_r = ur.empty() ? 0.0f : *std::min_element(ur.begin(), ur.end());
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

inline bool curve_shape_ok_ch(const InputCurves& c, int ch, int table_n, float max_dev) {
  if (c.n_knots > 0) {
    const float ref = c.ref[ch];
    if (!(ref > 0.0f) || table_n < 2) return true;
    bool any = false;
    for (int s = 0; s <= c.n_knots && s < InputCurves::kMaxSegs; ++s)
      if (c.a_seg[ch][s] != 0.0f) any = true;
    if (!any) return true;
    float prev = INFINITY;
    for (int i = table_n - 1; i >= 1; --i) {
      const float v = static_cast<float>(i) / static_cast<float>(table_n - 1);
      if (!(v < ref)) continue;
      const float u = std::log2(v / ref);
      const float f = curve_stops_knots(u, c.n_knots, c.u_knot[ch], c.a_seg[ch], c.u_hold(ch));
      if (!(f < prev) || std::fabs(f - u) > max_dev) return false;
      prev = f;
    }
    return true;
  }
  return curve_shape_ok(c.a[ch], c.k[ch], c.u_med[ch], c.ref[ch], table_n, max_dev, c.u_hold(ch));
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
  float curve_bound = 0.35f;  // |a|, |k| <= this
  float max_dev_stops = 2.2f; // each curve within this many stops of the base TRC over the whole table (curve_shape_ok)
  float g_bound = 0.5f;       // |g - g0| <= this (fit_input_curves)
  float b_bound = 0.6f;       // |b - b0| <= this (fit_input_curves)
  float g_bound_stops = 0.75f;  // joint fit: g in [g0 * 2^-s, g0 * 2^s] (<= 0: additive g_bound)
  float b_bound_stops = 1.25f;  // joint fit: b in [b0 * 2^-s, b0 * 2^s] (<= 0: additive b_bound)
  // Penalty on the four curve numbers, lambda (aG^2 + kG^2 + aB^2 + kB^2), added to the fits' objective z2 (the mean
  // of the squared residuals over the 3 n cell terms). The gains and the curves' slope terms (1 + a) move the
  // residuals almost alike, so without it the Levenberg-Marquardt normal matrix is nearly singular along that
  // direction; damping non-essential curve bends also prevents shadow-to-highlight crossover on clean frames.
  float lambda = 0.25f;
  float joint_lambda = 1.5f;
  float step0_gain = 0.08f;
  float step0_curve = 0.05f;
  float step_min = 1e-4f;
  int max_evals = 8000;
  bool hold_slope = true;       // InputCurves::hold for the fitted curves (on: Alpha Lam, 2026-11-05)
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
struct NeutralStats {
  int m = 0;
  float conf = 0.0f;
  float c_a = 0.0f, c_b = 0.0f;
  float a_bar = 0.0f, b_bar = 0.0f;
  float k_a = 0.0f, k_b = 0.0f;
  float q_a = 0.0f, q_b = 0.0f;
  float off() const { return std::hypot(a_bar - c_a, b_bar - c_b); }
  float slope() const { return std::hypot(k_a, k_b); }
  float quad() const { return std::hypot(q_a, q_b); }
};

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
  NeutralStats neutral_before, neutral_after;
};

// FNV-1a over the bytes of a float field (experiment hook: keys external per-cell data to a DINOv3 mu field).
inline unsigned long long mu_field_hash(const float* v, size_t n) {
  unsigned long long h = 1469598103934665603ULL;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(v);
  for (size_t k = 0; k < n * sizeof(float); ++k) { h ^= p[k]; h *= 1099511628211ULL; }
  return h;
}

class JointCurveModel {
 public:
  // Experiment hook (B5 + head anchors): external neutral-patch probabilities per cell; empty = raw-chroma selection.
  std::vector<float> ext_p_;
  float ext_thr_ = 0.9f;
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
    v.g = clamp01(input_curve_ch(c, 0, v.g));
    v.b = clamp01(input_curve_ch(c, 1, v.b));
    return linear_srgb_to_lab(to_display(cfg_, pcs_to_linear_srgb_pre_shoulder(apply_profile(cfg_.profile, v))));
  }

  float median_L(float e, float g, float b, const InputCurves& shape) {
    const InputCurves c = anchored(shape, e, g, b);
    for (int i = 0; i < f_.n; ++i) L_[i] = cell_lab(i, e, g, b, c).L;
    std::nth_element(L_.begin(), L_.begin() + f_.n / 2, L_.end());
    return L_[f_.n / 2];
  }

  // Builds the low-chroma scene-neutral set N directly from the rendered scan cells Lab0(i) at (e0, g0, b0)
  // across four luma bands ([2, 22], [22, 45], [45, 70], [70, 92]), with collinearity gating on midtone bands
  // and equalized band weights, and zero dependency on DINOv3's mu.
  void set_scene_neutral(float e0, float g0, float b0, float strength = 0.75f) {
    neu_idx_.clear();
    neu_band_.clear();
    neu_pre_gate_idx_.clear();
    neu_w_.clear();
    conf_ = w_off_ = w_slope_ = w_quad_ = 0.0f;
    c_a_ = c_b_ = 0.0f;
    if (f_.n <= 0) return;

    const InputCurves zero_c{};
    std::vector<Lab> lab0(f_.n);
    std::vector<float> c0(f_.n);
    for (int i = 0; i < f_.n; ++i) {
      lab0[i] = cell_lab(i, e0, g0, b0, zero_c);
      const float c = std::hypot(lab0[i].a, lab0[i].b);
      c0[i] = std::isfinite(c) && std::isfinite(lab0[i].L) ? c : INFINITY;
    }

    std::vector<int> bands[4];
    std::vector<float> mid_c;
    mid_c.reserve(f_.n);
    int total = 0;
    for (int i = 0; i < f_.n; ++i) {
      if (!std::isfinite(c0[i])) continue;
      const float L = lab0[i].L;
      if (L > 2.0f && L < 92.0f) {
        if (L <= 22.0f) bands[0].push_back(i);
        else if (L <= 45.0f) bands[1].push_back(i);
        else if (L <= 70.0f) bands[2].push_back(i);
        else bands[3].push_back(i);
        if (L > 20.0f) mid_c.push_back(c0[i]);
        ++total;
      }
    }
    if (total < 8) return;
    if (mid_c.empty()) {
      for (int i : bands[0]) mid_c.push_back(c0[i]);
    }
    if (mid_c.empty()) return;
    // On extreme low-key scenes (>= 75% of cells in B0 and < 8 cells in B3, e.g. a dark tunnel with a small
    // daylight portal), merge B2 and B3 into B3 (L* > 45) so the highlight endpoint is populated.
    const bool head_mode = (int)ext_p_.size() == f_.n;
    if (!head_mode && (int)bands[0].size() >= 3 * total / 4 && (int)bands[3].size() < 8 &&
        (int)(bands[2].size() + bands[3].size()) >= 12) {
      bands[3].insert(bands[3].end(), bands[2].begin(), bands[2].end());
      bands[2].clear();
      std::fprintf(stderr, "GATE lowkey_merge\n");
    }
    std::vector<float> mc_sorted = mid_c;
    std::sort(mc_sorted.begin(), mc_sorted.end());
    const float c_q15 = mc_sorted[std::min(mc_sorted.size() - 1, (size_t)(0.15 * (double)mc_sorted.size()))];
    const float c_q30 = mc_sorted[std::min(mc_sorted.size() - 1, (size_t)(0.30 * (double)mc_sorted.size()))];
    const float c_q50 = mc_sorted[std::min(mc_sorted.size() - 1, (size_t)(0.50 * (double)mc_sorted.size()))];
    const bool chart_like = !head_mode && (c_q15 < 6.5f && c_q50 > 15.0f && c_q50 > 2.5f * c_q15);
    if (chart_like) std::fprintf(stderr, "GATE chart_like\n");
    const float c_gate = chart_like ? std::max(5.0f, 1.5f * c_q15) : std::max(15.0f, 1.75f * c_q30);
    const double sel_frac = chart_like ? 0.15 : 0.30;

    std::vector<int> b_sel[4];
    double bL[4] = {}, ba[4] = {}, bb[4] = {}, bmed_a[4] = {}, bmed_b[4] = {};
    for (int b = 0; b < 4; ++b) {
      if (bands[b].empty() || ((int)bands[b].size() < 8 && total >= 64)) continue;
      std::vector<float> va, vb;
      va.reserve(bands[b].size());
      vb.reserve(bands[b].size());
      for (int i : bands[b]) {
        va.push_back(lab0[i].a);
        vb.push_back(lab0[i].b);
      }
      bmed_a[b] = median(va);
      bmed_b[b] = median(vb);
      std::vector<int> ord = bands[b];
      const bool use_cluster = (b == 0 && (int)bands[0].size() >= f_.n / 3 && std::hypot(bmed_a[0], bmed_b[0]) > 12.0);
      if (use_cluster) std::fprintf(stderr, "GATE shadow_cluster\n");
      std::sort(ord.begin(), ord.end(), [&](int i, int j) {
        const float ci = use_cluster ? std::hypot(lab0[i].a - (float)bmed_a[0], lab0[i].b - (float)bmed_b[0]) : c0[i];
        const float cj = use_cluster ? std::hypot(lab0[j].a - (float)bmed_a[0], lab0[j].b - (float)bmed_b[0]) : c0[j];
        return ci != cj ? ci < cj : i < j;
      });
      if (head_mode) {
        for (int i : bands[b])
          if (ext_p_[i] >= ext_thr_) b_sel[b].push_back(i);
      } else {
        const int mb = std::max(1, (int)std::lround(sel_frac * (double)ord.size()));
        for (int k = 0; k < mb; ++k) {
          if (use_cluster || c0[ord[k]] <= c_gate) b_sel[b].push_back(ord[k]);
        }
      }
      for (int idx_pre : b_sel[b]) neu_pre_gate_idx_.push_back(idx_pre);
      if (!b_sel[b].empty()) {
        for (int i : b_sel[b]) {
          bL[b] += lab0[i].L;
          ba[b] += lab0[i].a;
          bb[b] += lab0[i].b;
        }
        const double inv_nb = 1.0 / (double)b_sel[b].size();
        bL[b] *= inv_nb;
        ba[b] *= inv_nb;
        bb[b] *= inv_nb;
      }
    }

    // Collinearity gate on midtone bands B1, B2 when both shadow B0 (at least 16 cells) and highlight B3 are populated:
    // if a midtone band deviates from the linear L* trajectory between B0 and B3 by > 8.0 Lab units (e.g. blue sky in
    // B1/B2 on 02, or sodium-vapor tunnel wall in B1 on 00), drop that midtone band so B0 and B3 anchor the neutral line.
    if (!head_mode && b_sel[0].size() >= 16 && !b_sel[3].empty() && b_sel[0].size() + b_sel[3].size() >= 32 && bL[3] - bL[0] > 25.0) {
      for (int b = 1; b <= 2; ++b) {
        if (b_sel[b].empty()) continue;
        const double t = std::clamp((bL[b] - bL[0]) / (bL[3] - bL[0]), 0.0, 1.0);
        const double a_hat_med = bmed_a[0] + t * (bmed_a[3] - bmed_a[0]);
        const double b_hat_med = bmed_b[0] + t * (bmed_b[3] - bmed_b[0]);
        const double dev_med = std::hypot(bmed_a[b] - a_hat_med, bmed_b[b] - b_hat_med);
        const double a_hat_sel = ba[0] + t * (ba[3] - ba[0]);
        const double b_hat_sel = bb[0] + t * (bb[3] - bb[0]);
        const double dev_sel = std::hypot(ba[b] - a_hat_sel, bb[b] - b_hat_sel);
        if (std::max(dev_med, dev_sel) > 8.0) {
          std::fprintf(stderr, "GATE collinearity_drop_B%d\n", b);
          b_sel[b].clear();
        }
      }
    }

    float w_sum = 0.0f;
    double sa_act = 0.0, sb_act = 0.0, w_act = 0.0;
    int n_bands = 0;
    for (int b = 0; b < 4; ++b) {
      if (b_sel[b].empty()) continue;
      ++n_bands;
      const size_t min_denom = (b == 3 && (int)bands[0].size() >= 3 * total / 4) ? 8 : 16;
      const float wb = 1.0f / (float)std::max<size_t>(b_sel[b].size(), min_denom);
      for (int i : b_sel[b]) {
        neu_idx_.push_back(i);
        neu_w_.push_back(wb);
        w_sum += wb;
      }
      const double wb_all = 1.0 / (double)bands[b].size();
      for (int i : bands[b]) {
        sa_act += wb_all * lab0[i].a;
        sb_act += wb_all * lab0[i].b;
      }
      w_act += 1.0;
    }
    if (w_sum > 0.0f) {
      for (float& w : neu_w_) w /= w_sum;
    }
    const int m = (int)neu_idx_.size();
    if (m < 16 || w_act <= 0.0) {
      neu_idx_.clear();
      neu_w_.clear();
      return;
    }
    const double a_act = sa_act / w_act, b_act = sb_act / w_act;

    double sL = 0.0, sa = 0.0, sb = 0.0;
    for (int k = 0; k < m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      sL += w * lab0[i].L;
      sa += w * lab0[i].a;
      sb += w * lab0[i].b;
    }
    double vL0 = 0.0, covLa0 = 0.0, covLb0 = 0.0, sz3_0 = 0.0;
    for (int k = 0; k < m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      const double dL = lab0[i].L - sL;
      const double z = dL / 50.0;
      vL0 += w * dL * dL;
      sz3_0 += w * z * z * z;
      covLa0 += w * dL * (lab0[i].a - sa);
      covLb0 += w * dL * (lab0[i].b - sb);
    }
    const double ka0 = 50.0 * covLa0 / (vL0 + 25.0);
    const double kb0 = 50.0 * covLb0 / (vL0 + 25.0);
    const double vz0 = vL0 / 2500.0;
    const double mu3_0 = sz3_0 / (vz0 + 1e-4);
    double vq0 = 0.0, sqa0 = 0.0, sqb0 = 0.0;
    for (int k = 0; k < m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      const double z = (lab0[i].L - sL) / 50.0;
      const double uq = 4.0 * (z * z - vz0 - mu3_0 * z);
      const double ea = (lab0[i].a - sa) - ka0 * z;
      const double eb = (lab0[i].b - sb) - kb0 * z;
      vq0 += w * uq * uq;
      sqa0 += w * uq * ea;
      sqb0 += w * uq * eb;
    }

    std::vector<float> dev(m);
    for (int k = 0; k < m; ++k) {
      const int i = neu_idx_[k];
      const double z = (lab0[i].L - sL) / 50.0;
      dev[k] = (float)std::hypot(lab0[i].a - (sa + ka0 * z), lab0[i].b - (sb + kb0 * z));
    }
    const float spread = median(dev);
    const bool has_endpoints = (b_sel[0].size() >= 15 && (!b_sel[3].empty() || b_sel[0].size() >= 64));
    const double cast_a = has_endpoints
                              ? sa
                              : ((sa * a_act > 0.0) ? std::copysign(std::min(std::fabs(sa), std::fabs(a_act)), sa) : 0.0);
    const double cast_b = has_endpoints
                              ? sb
                              : ((sb * b_act > 0.0) ? std::copysign(std::min(std::fabs(sb), std::fabs(b_act)), sb) : 0.0);
    const float unaligned = (float)std::hypot(sa - cast_a, sb - cast_b);
    conf_ = std::clamp(1.0f - (spread - 3.5f) / 8.0f, 0.25f, 1.0f) *
            std::clamp(1.0f - unaligned / 6.0f, 0.0f, 1.0f) *
            std::min(1.0f, (float)m / 48.0f);
    const float strength_eff = has_endpoints ? 0.95f : strength;
    const float alpha_off = std::clamp(strength_eff * conf_, 0.0f, 1.0f);
    c_a_ = (float)(sa - alpha_off * cast_a);
    c_b_ = (float)(sb - alpha_off * cast_b);
    w_off_ = conf_ / 1.5f;
    w_slope_ = chart_like ? 0.0f : conf_ * std::sqrt((float)(vL0 / (vL0 + 100.0))) / 2.0f;
    w_quad_ = (n_bands >= 3) ? conf_ * std::sqrt((float)(vq0 / (vq0 + 0.08))) / 4.0f : 0.0f;
  }

  NeutralStats neutral_stats(float e, float g, float b, const InputCurves& shape) {
    const InputCurves c = anchored(shape, e, g, b);
    return neutral_stats_anchored(e, g, b, c);
  }

  NeutralStats neutral_stats_on(const PipelineConfig& cfg, float e, float g, float b) const {
    NeutralStats st;
    st.m = (int)neu_idx_.size();
    st.conf = conf_;
    st.c_a = c_a_;
    st.c_b = c_b_;
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

  // 4-equation scene-neutral residuals: [w_off * (a_bar - c_a), w_off * (b_bar - c_b), w_slope * k_a, w_slope * k_b].
  double residuals(float e, float g, float b, const InputCurves& shape, std::vector<double>& r) {
    const NeutralStats st = neutral_stats(e, g, b, shape);
    r.assign(4, 0.0);
    if (st.m <= 0 || conf_ <= 0.0f) return 0.0;
    r[0] = (double)w_off_ * (st.a_bar - c_a_);
    r[1] = (double)w_off_ * (st.b_bar - c_b_);
    r[2] = (double)w_slope_ * st.k_a;
    r[3] = (double)w_slope_ * st.k_b;
    return r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3];
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

  const std::vector<int>& neu_idx() const { return neu_idx_; }
  const std::vector<int>& neu_band() const { return neu_band_; }
  const std::vector<int>& neu_pre_gate_idx() const { return neu_pre_gate_idx_; }
  const std::vector<int>& holdout_idx() const { return holdout_idx_; }
  int active_bands() const { return n_active_bands_; }
  const int* band_counts() const { return band_counts_; }
  float u_knot(int ch, int k) const { return u_knot_[ch][k]; }
  bool use_rc_joint() const { return use_rc_joint_; }

  // Phase 1 / Phase 2: Select neutral anchors from DINOv3 prediction (mu_sel, sigma_sel) on the statistics grid:
  //   hypot(mu_a - med_a, mu_b - med_b) + |sigma_ab| < c_max
  // excluding clipped cells and cells outside the bracket's calibrated input range, and bucket into 4 bands.
  void set_dinov3_neutral_anchors(float e0, float g0, float b0, const float* mu_sel, const float* sigma_sel,
                                  float c_max = 6.0f, bool variant_1b = false, int band_mode = 0,
                                  int checker_mode = 0, int grid_w = 32, int min_band_anchors = 2,
                                  float g_cal_min = 0.0f, float g_cal_max = 0.0f, float g_cal_scale = 0.0f,
                                  int f_mode = 0) {
    neu_idx_.clear();
    neu_band_.clear();
    neu_pre_gate_idx_.clear();
    neu_w_.clear();
    holdout_idx_.clear();
    holdout_w_.clear();
    n_active_bands_ = 0;
    for (int b = 0; b < 4; ++b) band_counts_[b] = 0;
    for (int ch = 0; ch < 2; ++ch)
      for (int k = 0; k < InputCurves::kMaxKnots; ++k) u_knot_[ch][k] = 0.0f;
    conf_ = w_off_ = w_slope_ = w_quad_ = 0.0f;
    c_a_ = c_b_ = 0.0f;
    if (f_.n <= 0 || !mu_sel || !sigma_sel) return;

    const float min_L_floor = (f_mode >= 1) ? 6.5f : 2.0f;
    const int min_anc = (f_mode >= 1) ? std::max(5, min_band_anchors) : min_band_anchors;

    const InputCurves zero_c = anchored(InputCurves{}, e0, g0, b0);
    std::vector<Lab> lab0(f_.n);
    std::vector<bool> valid(f_.n, false);
    std::vector<float> valid_mu_a, valid_mu_b;
    valid_mu_a.reserve(f_.n);
    valid_mu_b.reserve(f_.n);

    for (int i = 0; i < f_.n; ++i) {
      lab0[i] = cell_lab(i, e0, g0, b0, zero_c);
      if (!std::isfinite(lab0[i].L) || !std::isfinite(lab0[i].a) || !std::isfinite(lab0[i].b)) continue;
      if (lab0[i].L <= min_L_floor || lab0[i].L >= 99.5f) continue;
      const uint16_t r16 = f_.grid[3 * i + 0], g16 = f_.grid[3 * i + 1], b16 = f_.grid[3 * i + 2];
      if (std::max({r16, g16, b16}) >= 65500 || std::min({r16, g16, b16}) <= 1) continue;
      if (g_cal_max > g_cal_min && g_cal_scale > 0.0f) {
        const float g_in = g_cal_scale * e0 * static_cast<float>(g16);
        if (g_in < g_cal_min * std::exp2(-1.5f) || g_in > g_cal_max * std::exp2(1.2f)) continue;
      }
      valid[i] = true;
      valid_mu_a.push_back(mu_sel[3 * i + 1]);
      valid_mu_b.push_back(mu_sel[3 * i + 2]);
    }
    const int n_valid = (int)valid_mu_a.size();
    if (n_valid < 8) return;

    // Experiment hook (arm N): neutral-patch probabilities from an external head, keyed by a hash of mu_sel.
    std::vector<float> ext_p;
    float ext_thr = 0.9f;
    if (const char* pdir = std::getenv("NP_NEUTRAL_P_DIR")) {
      if (const char* t = std::getenv("NP_NEUTRAL_P_THR")) ext_thr = std::strtof(t, nullptr);
      const unsigned long long hsh = mu_field_hash(mu_sel, (size_t)f_.n * 3);
      char nm[64];
      std::snprintf(nm, sizeof(nm), "/%016llx.txt", hsh);
      FILE* fp = std::fopen((std::string(pdir) + nm).c_str(), "r");
      if (fp) {
        float v;
        while (std::fscanf(fp, "%f", &v) == 1) ext_p.push_back(v);
        std::fclose(fp);
      }
      if ((int)ext_p.size() != f_.n) {
        std::fprintf(stderr, "ARM_N: no p for mu %016llx (got %zu of %d)\n", hsh, ext_p.size(), f_.n);
        ext_p.clear();
      } else {
        std::fprintf(stderr, "ARM_N: using p for mu %016llx thr %.2f\n", hsh, ext_thr);
      }
    }
    const float med_a = variant_1b ? median(valid_mu_a) : 0.0f;
    const float med_b = variant_1b ? median(valid_mu_b) : 0.0f;
    const float u_lo_g = zero_c.u_lo[0];
    const float q1 = 0.25f * u_lo_g, q2 = 0.50f * u_lo_g, q3 = 0.75f * u_lo_g;

    std::vector<int> bands_all[4], b_sel_all[4];
    for (int i = 0; i < f_.n; ++i) {
      if (!valid[i]) continue;
      int b = -1;
      if (band_mode == 1 && u_lo_g < -0.4f) {
        const float u = ug_[i];
        if (u > q1) b = 0;
        else if (u > q2) b = 1;
        else if (u > q3) b = 2;
        else b = 3;
      } else {
        const float L = lab0[i].L;
        if (L <= 22.0f) b = 0;
        else if (L <= 45.0f) b = 1;
        else if (L <= 70.0f) b = 2;
        else b = 3;
      }
      bands_all[b].push_back(i);
      const float c_mu = std::hypot(mu_sel[3 * i + 1] - med_a, mu_sel[3 * i + 2] - med_b);
      const float s_ab = std::hypot(sigma_sel[3 * i + 1], sigma_sel[3 * i + 2]) / 1.41421356f;
      const bool sel_ok = ext_p.empty() ? (c_mu + s_ab < c_max) : (ext_p[i] >= ext_thr);
      if (sel_ok) {
        b_sel_all[b].push_back(i);
        neu_pre_gate_idx_.push_back(i);
      }
    }

    bool is_strict_band[4] = {false, false, false, false};
    int n_strict_bands = 0;
    for (int b = 0; b < 4; ++b) {
      if ((int)b_sel_all[b].size() >= min_anc) {
        is_strict_band[b] = true;
        ++n_strict_bands;
      }
    }

    // F2 (f_mode >= 2): Extend correction to unanchored dark/bright tone halves using collinearity-gated
    // low-raw-chroma scene-neutral cells when at least one band has strict DINOv3 anchors, then apply the
    // B0 -> B3 collinearity gate on midtone bands B1, B2.
    use_rc_joint_ = false;
    if ((f_mode == 2 || f_mode == 3 || f_mode == 4) && n_strict_bands >= 1 && band_mode == 0) {
      const bool has_dark = (is_strict_band[0] || is_strict_band[1]);
      const bool only_b0 = (n_strict_bands == 1 && is_strict_band[0]);
      if (!has_dark || only_b0) {
        use_rc_joint_ = true;
        JointCurveModel m_rc(cfg_, f_);
        m_rc.set_scene_neutral(e0, g0, b0);
        std::vector<int> rc_bands[4];
        for (int idx : m_rc.neu_idx()) {
          if (!valid[idx]) continue;
          const float L = lab0[idx].L;
          const int b = (L <= 22.0f) ? 0 : ((L <= 45.0f) ? 1 : ((L <= 70.0f) ? 2 : 3));
          rc_bands[b].push_back(idx);
        }
        if (!has_dark && (int)rc_bands[0].size() >= 16) {
          b_sel_all[0] = rc_bands[0];
        }
        if (only_b0) {
          for (int b : {1, 2}) {
            if ((int)b_sel_all[b].size() < min_anc && (int)rc_bands[b].size() >= min_anc) {
              b_sel_all[b] = rc_bands[b];
            }
          }
        }
      }
      if ((int)b_sel_all[0].size() >= 16 && (int)b_sel_all[3].size() >= min_anc &&
          (int)(b_sel_all[0].size() + b_sel_all[3].size()) >= 24) {
        double bL[4] = {}, ba[4] = {}, bb[4] = {};
        for (int b = 0; b < 4; ++b) {
          if (b_sel_all[b].empty()) continue;
          for (int idx : b_sel_all[b]) {
            bL[b] += lab0[idx].L;
            ba[b] += lab0[idx].a;
            bb[b] += lab0[idx].b;
          }
          const double inv = 1.0 / (double)b_sel_all[b].size();
          bL[b] *= inv;
          ba[b] *= inv;
          bb[b] *= inv;
        }
        if (bL[3] - bL[0] > 25.0) {
          for (int b = 1; b <= 2; ++b) {
            if ((int)b_sel_all[b].size() < min_anc) continue;
            const double t = std::clamp((bL[b] - bL[0]) / (bL[3] - bL[0]), 0.0, 1.0);
            const double a_hat = ba[0] + t * (ba[3] - ba[0]);
            const double b_hat = bb[0] + t * (bb[3] - bb[0]);
            if (std::hypot(ba[b] - a_hat, bb[b] - b_hat) > 5.5) {
              b_sel_all[b].clear();
              is_strict_band[b] = false;
            }
          }
        }
      }
    }

    int n_strict_kept = 0;
    int n_strict_total = 0;
    double strict_mu_a = 0.0, strict_mu_b = 0.0;
    for (int b = 0; b < 4; ++b) {
      if (is_strict_band[b] && (int)b_sel_all[b].size() >= min_anc) {
        ++n_strict_kept;
        double ba = 0.0, bb = 0.0;
        for (int idx : b_sel_all[b]) {
          ba += mu_sel[3 * idx + 1];
          bb += mu_sel[3 * idx + 2];
          ++n_strict_total;
        }
        strict_mu_a += ba / (double)b_sel_all[b].size();
        strict_mu_b += bb / (double)b_sel_all[b].size();
      }
    }
    if (n_strict_kept > 0) {
      strict_mu_a /= (double)n_strict_kept;
      strict_mu_b /= (double)n_strict_kept;
    }

    std::vector<int> b_fit[4], b_hold[4];
    int active_b[4] = {};
    for (int b = 0; b < 4; ++b) {
      band_counts_[b] = (int)b_sel_all[b].size();
      if (bands_all[b].empty() || ((int)bands_all[b].size() < 8 && n_valid >= 64)) continue;
      if ((int)b_sel_all[b].size() < min_anc) continue;
      if (checker_mode == 1 && grid_w > 0) {
        for (int idx : b_sel_all[b]) {
          const int r_idx = idx / grid_w, c_idx = idx % grid_w;
          if (((r_idx + c_idx) & 1) == 0) b_fit[b].push_back(idx);
          else b_hold[b].push_back(idx);
        }
        if (b_fit[b].empty()) b_fit[b] = b_sel_all[b];
      } else {
        b_fit[b] = b_sel_all[b];
      }
      active_b[n_active_bands_++] = b;
    }

    if (n_active_bands_ == 0) return;

    float w_sum = 0.0f, wh_sum = 0.0f;
    for (int k = 0; k < n_active_bands_; ++k) {
      const int b = active_b[k];
      const float wb = 1.0f / static_cast<float>(n_active_bands_ * b_fit[b].size());
      for (int idx : b_fit[b]) {
        neu_idx_.push_back(idx);
        neu_band_.push_back(b);
        neu_w_.push_back(wb);
        w_sum += wb;
      }
      if (!b_hold[b].empty()) {
        const float wh = 1.0f / static_cast<float>(n_active_bands_ * b_hold[b].size());
        for (int idx : b_hold[b]) {
          holdout_idx_.push_back(idx);
          holdout_w_.push_back(wh);
          wh_sum += wh;
        }
      }
      std::vector<float> vg_b, vb_b;
      vg_b.reserve(b_sel_all[b].size());
      vb_b.reserve(b_sel_all[b].size());
      for (int idx : b_sel_all[b]) {
        vg_b.push_back(ug_[idx]);
        vb_b.push_back(ub_[idx]);
      }
      u_knot_[0][k] = median(vg_b);
      u_knot_[1][k] = median(vb_b);
    }
    if (w_sum > 0.0f) {
      for (float& w : neu_w_) w /= w_sum;
    }
    if (wh_sum > 0.0f) {
      for (float& w : holdout_w_) w /= wh_sum;
    }
    // F1 (f_mode == 1 or 3): Target the common DINOv3 offset o = (strict_mu_a, strict_mu_b) across all bands.
    // F3 (f_mode == 4): Damp the common offset toward (0, 0) when broad multi-band / high-count neutral coverage
    // indicates a true neutral daylight/chart scene, while preserving full o on low-key / golden-hour scenes.
    if (f_mode == 1 || f_mode == 3) {
      c_a_ = static_cast<float>(strict_mu_a);
      c_b_ = static_cast<float>(strict_mu_b);
    } else if (f_mode == 4) {
      const bool broad_neutral = (n_strict_kept >= 3 && n_strict_total >= 64) || (n_strict_total >= 150);
      const double w_o = broad_neutral ? 0.35 : 1.0;
      double oa = w_o * strict_mu_a;
      double ob = w_o * strict_mu_b;
      const double mag = std::hypot(oa, ob);
      if (mag > 2.5) {
        oa *= 2.5 / mag;
        ob *= 2.5 / mag;
      }
      c_a_ = static_cast<float>(oa);
      c_b_ = static_cast<float>(ob);
    }
    // Experiment hook: partial-strength target, a fraction s of the anchors' common DINOv3 offset (0 = neutral).
    if (const char* so = std::getenv("NP_OFFSET_STRENGTH")) {
      const float s = std::strtof(so, nullptr);
      c_a_ = static_cast<float>(s * strict_mu_a);
      c_b_ = static_cast<float>(s * strict_mu_b);
      std::fprintf(stderr, "ARM_N: offset strength %.2f -> target (%.2f, %.2f)\n", s, c_a_, c_b_);
    }
    // Ensure knots are strictly ordered: 0 > u_knot[ch][0] > u_knot[ch][1] > ...
    for (int ch = 0; ch < 2; ++ch) {
      float prev = -0.05f;
      for (int k = 0; k < n_active_bands_; ++k) {
        u_knot_[ch][k] = std::min(u_knot_[ch][k], prev);
        prev = u_knot_[ch][k] - 0.12f;
      }
    }
    conf_ = neu_idx_.empty() ? 0.0f : 1.0f;
    w_off_ = 1.0f;
    w_slope_ = (n_active_bands_ >= 2) ? 0.5f : 0.0f;
    w_quad_ = (n_active_bands_ >= 3) ? 0.25f : 0.0f;
  }

  // Per-anchor-cell neutrality residuals (target a* = c_a_, b* = c_b_ after cLUT) with band-equalized weights w_j.
  double dinov3_residuals(float e, float g, float b, const InputCurves& shape, std::vector<double>& r) {
    const int m = (int)neu_idx_.size();
    r.assign(2 * m, 0.0);
    if (m <= 0 || conf_ <= 0.0f) return 0.0;
    const InputCurves c = anchored(shape, e, g, b);
    double sum = 0.0;
    for (int j = 0; j < m; ++j) {
      const int i = neu_idx_[j];
      const double sw = std::sqrt(static_cast<double>(neu_w_[j]));
      const Lab l = cell_lab(i, e, g, b, c);
      r[2 * j + 0] = sw * static_cast<double>(l.a - c_a_);
      r[2 * j + 1] = sw * static_cast<double>(l.b - c_b_);
      sum += r[2 * j + 0] * r[2 * j + 0] + r[2 * j + 1] * r[2 * j + 1];
    }
    return sum;
  }

  // Band-equalized mean Delta E_ab of fitted anchors at (e, g, b, shape).
  float anchor_mean_de(float e, float g, float b, const InputCurves& shape) {
    const int m = (int)neu_idx_.size();
    if (m <= 0) return 0.0f;
    const InputCurves c = anchored(shape, e, g, b);
    double sum = 0.0;
    for (int j = 0; j < m; ++j) {
      const Lab l = cell_lab(neu_idx_[j], e, g, b, c);
      sum += static_cast<double>(neu_w_[j]) * std::hypot(l.a - c_a_, l.b - c_b_);
    }
    return static_cast<float>(sum);
  }

  // Band-equalized mean Delta E_ab of held-out checkerboard anchors at (e, g, b, shape).
  float heldout_anchor_mean_de(float e, float g, float b, const InputCurves& shape) {
    const int m = (int)holdout_idx_.size();
    if (m <= 0) return anchor_mean_de(e, g, b, shape);
    const InputCurves c = anchored(shape, e, g, b);
    double sum = 0.0;
    for (int j = 0; j < m; ++j) {
      const Lab l = cell_lab(holdout_idx_[j], e, g, b, c);
      sum += static_cast<double>(holdout_w_[j]) * std::hypot(l.a - c_a_, l.b - c_b_);
    }
    return static_cast<float>(sum);
  }

  // Mean across active bands of hypot(mean_a_k - c_a_, mean_b_k - c_b_) at (e, g, b, shape).
  float anchor_band_mean_de(float e, float g, float b, const InputCurves& shape) {
    const int m = (int)neu_idx_.size();
    if (m <= 0 || n_active_bands_ <= 0) return 0.0f;
    const InputCurves c = anchored(shape, e, g, b);
    double ba[4] = {0.0, 0.0, 0.0, 0.0};
    double bb[4] = {0.0, 0.0, 0.0, 0.0};
    int bc[4] = {0, 0, 0, 0};
    for (int j = 0; j < m; ++j) {
      const int bk = (j < (int)neu_band_.size()) ? neu_band_[j] : -1;
      if (bk < 0 || bk >= 4) continue;
      const Lab l = cell_lab(neu_idx_[j], e, g, b, c);
      ba[bk] += (l.a - c_a_);
      bb[bk] += (l.b - c_b_);
      bc[bk] += 1;
    }
    double sum = 0.0;
    int nb = 0;
    for (int k = 0; k < 4; ++k) {
      if (bc[k] > 0) {
        sum += std::hypot(ba[k] / bc[k], bb[k] / bc[k]);
        ++nb;
      }
    }
    return nb > 0 ? static_cast<float>(sum / nb) : 0.0f;
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
    st.conf = conf_;
    st.c_a = c_a_;
    st.c_b = c_b_;
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
  float conf_ = 0.0f, c_a_ = 0.0f, c_b_ = 0.0f, w_off_ = 0.0f, w_slope_ = 0.0f, w_quad_ = 0.0f;
  std::vector<float> rows_, L_, ug_, ub_;
  std::vector<int> neu_idx_;
  std::vector<int> neu_band_;
  std::vector<int> neu_pre_gate_idx_;
  std::vector<float> neu_w_;
  std::vector<int> holdout_idx_;
  std::vector<float> holdout_w_;
  int n_active_bands_ = 0;
  int band_counts_[4] = {0, 0, 0, 0};
  float u_knot_[2][InputCurves::kMaxKnots] = {};
  bool use_rc_joint_ = false;
};

struct DinoNeutralConfig {
  float c_max = 6.0f;           // Tunable #1: hypot(mu_a, mu_b) + |sigma_ab| < C_max
  float lambda = 0.25f;         // Tunable #2: regularizer weight on curve slopes toward identity
  float tau_bracket = 1.0f;     // Tunable #3: bracket tiebreaker tolerance in Delta E
  float jaccard_min = 0.90f;    // Tunable #4: Phase 3 Jaccard overlap threshold
  float eps_fallback = 3.0f;    // Tunable #5: Phase 5 leftover anchor error threshold in Delta E
  bool variant_1b = false;      // false: 1a (raw mu), true: 1b (subtract frame median mu)
  int force_arm = 0;            // 0: adaptive (2A/2B/2C + Phase 5 eps fallback), 1: 2A, 2: 2B, 3: 2C
  int band_mode = 0;            // 0: 4 luma bands (2,22], (22,45], (45,70], (70,99.5); 1: 4 u-quarters
  int checker_mode = 0;         // 0: all anchors, 1: fit even (r+c)%2==0 and hold out odd
  int grid_w = 32;
  int min_band_anchors = 2;
  float g_cal_min = 0.0f;
  float g_cal_max = 0.0f;
  float g_cal_scale = 0.0f;
  int f_mode = 0;               // 0: 2auto, 1: F1, 2: F2, 3: F1+F2, 4: F1+F2+F3
};

struct DinoNeutralFit {
  JointCurveFit jf;
  int arm_used = 0;             // 0: no anchors, 1: 2A, 2: 2B, 3: 2C
  int initial_arm = 0;
  int active_bands = 0;
  int band_counts[4] = {0, 0, 0, 0};
  int anchor_count = 0;
  float anchor_de_before = 0.0f;
  float anchor_de_after = 0.0f;
  float heldout_de_before = 0.0f;
  float heldout_de_after = 0.0f;
  bool dropped_down = false;
};

inline void apply_highlight_cliff_desurge(const PipelineConfig& cfg, const CurveFrame& f, float e, float g, float b,
                                          float max_dev_stops, InputCurves& curves) {
  if (curves.n_knots > 0 || !cfg.profile.has_profile || !cfg.profile.in_trc[0] || !cfg.profile.in_trc[1] ||
      !cfg.profile.in_trc[2] || cfg.profile.in_trc_size[0] <= 1 || cfg.profile.in_trc_size[1] <= 1 ||
      cfg.profile.in_trc_size[2] <= 1 || f.n < 64) {
    return;
  }
  const ProfileData& base = cfg.profile;
  const float aG1 = curves.a[0], aB1 = curves.a[1];
  const InputCurves c_p1 = curves;
  float m_p1[9];
  candidate_matrix(f.merged, e, g, b, m_p1);

  auto render_p2 = [&](float kR, float aG, float kG, float aB, float kB, std::vector<Lab>& out_lab) -> bool {
    InputCurves cand = c_p1;
    cand.a[0] = aG;
    cand.k[0] = kG;
    cand.a[1] = aB;
    cand.k[1] = kB;
    cand.k_r = kR;
    if (!curve_shape_ok(cand.a[0], cand.k[0], cand.u_med[0], cand.ref[0], base.in_trc_size[1], max_dev_stops,
                        cand.u_hold(0)) ||
        !curve_shape_ok(cand.a[1], cand.k[1], cand.u_med[1], cand.ref[1], base.in_trc_size[2], max_dev_stops,
                        cand.u_hold(1)) ||
        !curve_shape_ok(0.0f, cand.k_r, 0.0f, cand.ref_r, base.in_trc_size[0], max_dev_stops, cand.u_hold_r())) {
      return false;
    }
    CombinedProfile comb_p2;
    comb_p2.build(base, cand);
    PipelineConfig cfg_p2 = cfg;
    cfg_p2.profile = comb_p2.data;
    set_shoulder(cfg_p2, 1.0f);
    out_lab.resize(f.n);
    for (int i = 0; i < f.n; ++i) {
      out_lab[i] = linear_srgb_to_lab(to_display(cfg_p2, convert_pixel_linear(cfg_p2, m_p1, f.grid + 3 * i)));
    }
    return true;
  };

  std::vector<Lab> lab_p1;
  if (!render_p2(0.0f, aG1, 0.0f, aB1, 0.0f, lab_p1)) return;
  std::vector<int> blo, bhi;
  std::vector<float> l_lo, a_lo, b_lo_v, l_hi, a_hi, b_hi_v;
  for (int i = 0; i < f.n; ++i) {
    if (f.n >= 640 && f.n % 32 == 0) {
      const int gh = f.n / 32, r_idx = i / 32, c_idx = i % 32;
      if (r_idx < 1 || r_idx >= gh - 1 || c_idx < 2 || c_idx >= 30) continue;
    }
    if (lab_p1[i].L >= 58.0f && lab_p1[i].L < 70.0f) {
      blo.push_back(i);
      l_lo.push_back(lab_p1[i].L);
      a_lo.push_back(lab_p1[i].a);
      b_lo_v.push_back(lab_p1[i].b);
    } else if (lab_p1[i].L >= 75.0f && lab_p1[i].L <= 91.5f) {
      bhi.push_back(i);
      l_hi.push_back(lab_p1[i].L);
      a_hi.push_back(lab_p1[i].a);
      b_hi_v.push_back(lab_p1[i].b);
    }
  }
  auto med_vec = [](std::vector<float> v) -> float {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const float med_L_lo = med_vec(l_lo), med_b_lo = med_vec(b_lo_v);
  const float med_a_hi = med_vec(a_hi), med_b_hi = med_vec(b_hi_v);
  if (blo.size() < 8 || bhi.size() < 10 || !(med_b_lo < -1.0f && med_a_hi < -2.5f && (med_b_hi - med_b_lo) > 4.0f)) {
    return;
  }
  std::vector<int> sel_cyan, sel_warm;
  double sum_L_cyan = 0.0;
  for (int idx : bhi) {
    if (lab_p1[idx].a <= med_a_hi) {
      sel_cyan.push_back(idx);
      sum_L_cyan += lab_p1[idx].L;
    } else {
      sel_warm.push_back(idx);
    }
  }
  if (sel_cyan.empty()) return;
  const double mean_L1_cyan = sum_L_cyan / sel_cyan.size();
  const double target_L_hi = med_L_lo + 0.60 * (mean_L1_cyan - med_L_lo);
  const double target_a_hi = 0.2;
  const double target_b_hi = std::clamp(0.42 * static_cast<double>(med_b_lo), -4.2, -2.8);
  auto loss_p2 = [&](const float* pk) -> double {
    if (pk[0] > 0.0f || pk[0] < -0.65f || pk[1] > 0.0f || pk[1] < -0.65f || pk[2] > 0.0f || pk[2] < -0.45f)
      return 1e18;
    const float aG = aG1 - 0.085f * pk[1];
    const float aB = aB1 + 0.035f * pk[2];
    std::vector<Lab> lab;
    if (!render_p2(pk[0], aG, pk[1], aB, pk[2], lab)) return 1e18;
    double mL = 0.0, ma = 0.0, mb = 0.0;
    for (int idx : sel_cyan) {
      mL += lab[idx].L;
      ma += lab[idx].a;
      mb += lab[idx].b;
    }
    mL /= sel_cyan.size();
    ma /= sel_cyan.size();
    mb /= sel_cyan.size();
    double w_a = 0.0;
    if (!sel_warm.empty()) {
      for (int idx : sel_warm) w_a += (lab[idx].a - lab_p1[idx].a);
      w_a /= sel_warm.size();
    }
    const double dL = mL - target_L_hi;
    const double da = ma - target_a_hi;
    const double db = mb - target_b_hi;
    const double reg = 15.0 * (pk[0] - 0.823 * pk[1]) * (pk[0] - 0.823 * pk[1]) + 0.5 * (w_a * w_a);
    return dL * dL + 2.0 * da * da + 2.0 * db * db + reg;
  };
  float pk[3] = {-0.42f, -0.48f, -0.15f};
  float st_k[3] = {0.02f, 0.02f, 0.02f};
  double cur_k = loss_p2(pk);
  for (int it = 0; it < 60; ++it) {
    bool imp = false;
    for (int d = 0; d < 3; ++d) {
      for (int sgn : {-1, 1}) {
        float qk[3] = {pk[0], pk[1], pk[2]};
        qk[d] += sgn * st_k[d];
        const double v = loss_p2(qk);
        if (v < cur_k) {
          cur_k = v;
          pk[0] = qk[0];
          pk[1] = qk[1];
          pk[2] = qk[2];
          imp = true;
        }
      }
    }
    if (!imp) {
      for (int d = 0; d < 3; ++d) st_k[d] *= 0.6f;
    }
  }
  if (cur_k < 1e17) {
    curves.k_r = pk[0];
    curves.a[0] = aG1 - 0.085f * pk[1];
    curves.k[0] = pk[1];
    curves.a[1] = aB1 + 0.035f * pk[2];
    curves.k[1] = pk[2];
  }
}

inline DinoNeutralFit fit_dinov3_neutral_curves(const PipelineConfig& cfg, const CurveFrame& f, float e0, float g0,
                                                float b0, const float* mu_sel, const float* sigma_sel, float target_L,
                                                const int table_n[2], const CurveFitOptions& o = CurveFitOptions(),
                                                float e_lo = 0.0f, float e_hi = 0.0f,
                                                const DinoNeutralConfig& dcfg = DinoNeutralConfig()) {
  DinoNeutralFit res;
  JointCurveModel model(cfg, f, mu_sel, sigma_sel);
  model.set_dinov3_neutral_anchors(e0, g0, b0, mu_sel, sigma_sel, dcfg.c_max, dcfg.variant_1b, dcfg.band_mode,
                                   dcfg.checker_mode, dcfg.grid_w, dcfg.min_band_anchors, dcfg.g_cal_min,
                                   dcfg.g_cal_max, dcfg.g_cal_scale, dcfg.f_mode);
  res.active_bands = model.active_bands();
  for (int b = 0; b < 4; ++b) res.band_counts[b] = model.band_counts()[b];
  res.anchor_count = (int)model.neu_idx().size();
  res.jf.neutral_before = model.neutral_stats(e0, g0, b0, InputCurves{});
  res.anchor_de_before = model.anchor_mean_de(e0, g0, b0, InputCurves{});
  res.heldout_de_before = model.heldout_anchor_mean_de(e0, g0, b0, InputCurves{});

  if (!(e_lo > 0.0f)) e_lo = e0 * 0.25f;
  if (!(e_hi > e_lo)) e_hi = std::max(e_lo, e0 * 4.0f);
  const float g_lo = o.g_bound_stops > 0.0f ? g0 * std::exp2(-o.g_bound_stops) : g0 - o.g_bound;
  const float g_hi = o.g_bound_stops > 0.0f ? g0 * std::exp2(o.g_bound_stops) : g0 + o.g_bound;
  const float b_lo = o.b_bound_stops > 0.0f ? b0 * std::exp2(-o.b_bound_stops) : b0 - o.b_bound;
  const float b_hi = o.b_bound_stops > 0.0f ? b0 * std::exp2(o.b_bound_stops) : b0 + o.b_bound;

  // Phase 5: No anchors (or below the band minimum everywhere) -> B5's Auto render with no curves.
  if (res.active_bands == 0 || res.anchor_count < 2) {
    res.arm_used = 0;
    res.initial_arm = 0;
    res.jf.feasible = true;
    res.jf.converged = true;
    res.jf.e = e0;
    res.jf.g = g0;
    res.jf.b = b0;
    res.jf.curves = model.anchored(InputCurves{}, e0, g0, b0);
    res.jf.neutral_after = res.jf.neutral_before;
    res.anchor_de_after = res.anchor_de_before;
    res.heldout_de_after = res.heldout_de_before;
    return res;
  }

  int start_level = (res.active_bands >= 2 && res.anchor_count >= 12) ? 3
                  : (res.active_bands >= 2)                           ? 2
                                                                      : 1;
  if (dcfg.f_mode >= 2) {
    start_level = (res.active_bands >= 3 && res.anchor_count >= 12) ? 3
                : (res.active_bands >= 2)                           ? 2
                                                                    : 1;
  }
  if (dcfg.force_arm >= 1 && dcfg.force_arm <= 3) {
    start_level = dcfg.force_arm;
  }
  res.initial_arm = start_level;

  const double sqrt_lam = 15.0 * std::sqrt(std::max(0.0, static_cast<double>(dcfg.lambda)));
  const double sqrt_lam_rc = std::sqrt(std::max(0.0, static_cast<double>(o.joint_lambda)));
  JointCurveModel rc_model(cfg, f, mu_sel, sigma_sel);
  if (model.use_rc_joint()) {
    rc_model.set_scene_neutral(e0, g0, b0);
  }
  int total_renders = 0, total_evals = 0;

  auto run_level = [&](int level, JointCurveFit& out_jf, float& out_de) -> bool {
    const int n_knots = (level == 3) ? std::max(1, res.active_bands) : 0;
    const int n_segs_ch = (level == 1) ? 0 : ((level == 2) ? 1 : (n_knots + 1));
    const int m = 2 + 2 * n_segs_ch;
    std::vector<float> lo(m, -o.curve_bound), hi(m, o.curve_bound);
    lo[0] = std::max(kGainFloor, g_lo);
    hi[0] = g_hi;
    lo[1] = std::max(kGainFloor, b_lo);
    hi[1] = b_hi;

    auto shape_of = [&](const float* p) {
      InputCurves c;
      c.hold = o.hold_slope;
      if (level == 2) {
        c.a[0] = p[2];
        c.a[1] = p[3];
      } else if (level == 3) {
        c.n_knots = n_knots;
        for (int ch = 0; ch < 2; ++ch) {
          for (int k = 0; k < n_knots; ++k) c.u_knot[ch][k] = model.u_knot(ch, k);
          for (int s = 0; s <= n_knots; ++s) c.a_seg[ch][s] = p[2 + ch * n_segs_ch + s];
        }
      }
      return c;
    };

    auto solve_point = [&](const float* p, float e_warm, float& e, bool check_shape) {
      const InputCurves shape = shape_of(p);
      if (o.solve_e) {
        if (!model.solve_e(p[0], p[1], shape, target_L, e_warm, e_lo, e_hi, o.e_tol_L, total_renders, e))
          return false;
      } else {
        e = e0;
      }
      if (!check_shape) return true;
      const InputCurves c = model.anchored(shape, e, p[0], p[1]);
      return curve_shape_ok_ch(c, 0, table_n[0], o.max_dev_stops) &&
             curve_shape_ok_ch(c, 1, table_n[1], o.max_dev_stops);
    };

    auto evaluate = [&](const float* p, float e_warm, float& e, std::vector<double>& r, double& cost, double& data,
                        bool check_shape) {
      ++total_evals;
      if (!solve_point(p, e_warm, e, check_shape)) return false;
      if (model.use_rc_joint() && level == 2) {
        data = rc_model.residuals(e, p[0], p[1], shape_of(p), r);
        double sum = data;
        for (int j = 2; j < m; ++j) {
          const double v = sqrt_lam_rc * static_cast<double>(p[j]);
          r.push_back(v);
          sum += v * v;
        }
        cost = sum;
        return std::isfinite(cost);
      }
      data = model.dinov3_residuals(e, p[0], p[1], shape_of(p), r);
      double sum = data;
      if (n_segs_ch > 0) {
        if (dcfg.f_mode >= 2 && n_segs_ch >= 2) {
          for (int ch = 0; ch < 2; ++ch) {
            double mean_s = 0.0;
            for (int s = 0; s < n_segs_ch; ++s) mean_s += static_cast<double>(p[2 + ch * n_segs_ch + s]);
            mean_s /= static_cast<double>(n_segs_ch);
            const double vm = sqrt_lam * mean_s;
            r.push_back(vm);
            sum += vm * vm;
            for (int s = 0; s < n_segs_ch; ++s) {
              const double v = 0.15 * sqrt_lam * static_cast<double>(p[2 + ch * n_segs_ch + s]);
              r.push_back(v);
              sum += v * v;
            }
            for (int s = 1; s < n_segs_ch; ++s) {
              const double dv = 1.5 * sqrt_lam *
                                static_cast<double>(p[2 + ch * n_segs_ch + s] - p[2 + ch * n_segs_ch + s - 1]);
              r.push_back(dv);
              sum += dv * dv;
            }
          }
        } else {
          for (int j = 2; j < m; ++j) {
            const double v = sqrt_lam * static_cast<double>(p[j]);
            r.push_back(v);
            sum += v * v;
          }
          if (n_segs_ch >= 2) {
            for (int ch = 0; ch < 2; ++ch) {
              for (int s = 1; s < n_segs_ch; ++s) {
                const double dv = sqrt_lam * static_cast<double>(p[2 + ch * n_segs_ch + s] - p[2 + ch * n_segs_ch + s - 1]);
                r.push_back(dv);
                sum += dv * dv;
              }
            }
          }
        }
      }
      cost = sum;
      return std::isfinite(cost);
    };

    auto cut_back = [&](const float* p, const float* q, float e, float& e_t) {
      float t_ok = 0.0f, t_bad = 1.0f;
      e_t = e;
      std::vector<float> qt(m);
      for (int it = 0; it < 10; ++it) {
        const float t = 0.5f * (t_ok + t_bad);
        for (int j = 0; j < m; ++j) qt[j] = p[j] + t * (q[j] - p[j]);
        float et = e;
        if (solve_point(qt.data(), e, et, true)) {
          t_ok = t;
          e_t = et;
        } else {
          t_bad = t;
        }
      }
      return t_ok;
    };

    std::vector<float> p(m, 0.0f);
    p[0] = g0;
    p[1] = b0;
    float e = e0;
    std::vector<double> r, rt, rj;
    double cost = 0.0, data = 0.0;
    if (!evaluate(p.data(), e0, e, r, cost, data, true)) {
      out_jf.feasible = false;
      return false;
    }
    out_jf.feasible = true;
    out_jf.loss_before = data;

    double lm_lambda = 1e-3;
    bool settled = (data <= 1e-6);
    for (int iter = 0; iter < 25 && !settled; ++iter) {
      const size_t nr = r.size();
      std::vector<double> J(nr * m, 0.0);
      for (int j = 0; j < m; ++j) {
        std::vector<float> q = p;
        const float h = (q[j] + 1e-3f <= hi[j]) ? 1e-3f : -1e-3f;
        q[j] += h;
        float ej = e;
        double cj = 0.0, dj = 0.0;
        if (!evaluate(q.data(), e, ej, rj, cj, dj, false) || rj.size() != nr) continue;
        for (size_t k = 0; k < nr; ++k) J[k * m + j] = (rj[k] - r[k]) / h;
      }
      std::vector<double> A(m * m, 0.0), gvec(m, 0.0);
      for (size_t k = 0; k < nr; ++k) {
        for (int a = 0; a < m; ++a) {
          gvec[a] += J[k * m + a] * r[k];
          for (int bb = a; bb < m; ++bb) A[a * m + bb] += J[k * m + a] * J[k * m + bb];
        }
      }
      for (int a = 0; a < m; ++a)
        for (int bb = 0; bb < a; ++bb) A[a * m + bb] = A[bb * m + a];
      for (int a = 0; a < m; ++a) {
        if ((p[a] <= lo[a] + 1e-5f && gvec[a] > 0.0) || (p[a] >= hi[a] - 1e-5f && gvec[a] < 0.0)) {
          for (int bb = 0; bb < m; ++bb) A[a * m + bb] = A[bb * m + a] = 0.0;
          A[a * m + a] = 1.0;
          gvec[a] = 0.0;
        }
      }

      bool accepted = false;
      for (int tries = 0; tries < 10 && !accepted; ++tries) {
        std::vector<double> M(m * (m + 1), 0.0);
        for (int a = 0; a < m; ++a) {
          for (int bb = 0; bb < m; ++bb)
            M[a * (m + 1) + bb] = A[a * m + bb] + (a == bb ? lm_lambda * std::max(A[a * m + a], 1e-6) : 0.0);
          M[a * (m + 1) + m] = -gvec[a];
        }
        for (int c = 0; c < m; ++c) {
          int piv = c;
          for (int rr = c + 1; rr < m; ++rr)
            if (std::fabs(M[rr * (m + 1) + c]) > std::fabs(M[piv * (m + 1) + c])) piv = rr;
          for (int cc = 0; cc <= m; ++cc) std::swap(M[c * (m + 1) + cc], M[piv * (m + 1) + cc]);
          if (std::fabs(M[c * (m + 1) + c]) < 1e-300) M[c * (m + 1) + c] = 1e-300;
          for (int rr = c + 1; rr < m; ++rr) {
            const double fct = M[rr * (m + 1) + c] / M[c * (m + 1) + c];
            for (int cc = c; cc <= m; ++cc) M[rr * (m + 1) + cc] -= fct * M[c * (m + 1) + cc];
          }
        }
        std::vector<double> delta(m, 0.0);
        for (int c = m - 1; c >= 0; --c) {
          double sum = M[c * (m + 1) + m];
          for (int cc = c + 1; cc < m; ++cc) sum -= M[c * (m + 1) + cc] * delta[cc];
          delta[c] = sum / M[c * (m + 1) + c];
        }
        std::vector<float> q(m, 0.0f);
        float step = 0.0f;
        for (int j = 0; j < m; ++j) {
          q[j] = std::min(hi[j], std::max(lo[j], p[j] + static_cast<float>(delta[j])));
          step = std::max(step, std::fabs(q[j] - p[j]));
        }
        if (step < 1e-5f) {
          settled = true;
          break;
        }
        float eq = e;
        double cq = 0.0, dq = 0.0;
        bool step_ok = false;
        for (float step_s : {1.0f, 0.5f, 0.25f, 0.1f, 0.05f}) {
          std::vector<float> qt(m);
          for (int j = 0; j < m; ++j)
            qt[j] = std::min(hi[j], std::max(lo[j], p[j] + step_s * static_cast<float>(delta[j])));
          float eq_t = e;
          double cq_t = 0.0, dq_t = 0.0;
          bool ok = evaluate(qt.data(), e, eq_t, rt, cq_t, dq_t, true);
          if (!ok && step_s == 1.0f) {
            float e_t = e;
            const float t_ok = cut_back(p.data(), qt.data(), e, e_t);
            if (t_ok > 0.0f) {
              for (int j = 0; j < m; ++j) qt[j] = p[j] + t_ok * (qt[j] - p[j]);
              ok = evaluate(qt.data(), e_t, eq_t, rt, cq_t, dq_t, true);
            }
          }
          if (ok && cq_t < cost) {
            q = qt;
            eq = eq_t;
            cq = cq_t;
            dq = dq_t;
            step_ok = true;
            break;
          }
        }
        if (step_ok && cq < cost) {
          const double gain = cost - cq;
          p = q;
          e = eq;
          r.swap(rt);
          cost = cq;
          data = dq;
          lm_lambda = std::max(lm_lambda / 3.0, 1e-9);
          accepted = true;
          ++out_jf.iterations;
          if (gain < 1e-6 * std::max(1.0, cost) || data < 1e-4) settled = true;
        } else {
          lm_lambda *= 4.0;
        }
      }
      if (!accepted && !settled) break;
    }
    out_jf.converged = settled;
    out_jf.g = p[0];
    out_jf.b = p[1];
    out_jf.e = e;
    out_jf.curves = model.anchored(shape_of(p.data()), e, p[0], p[1]);
    out_jf.loss_after = data;
    out_jf.neutral_after = model.neutral_stats(e, p[0], p[1], out_jf.curves);
    out_de = model.anchor_mean_de(e, p[0], p[1], out_jf.curves);
    return true;
  };

  int cur_level = start_level;
  bool has_best = false;
  float best_band_de = 1e30f;
  while (cur_level >= 1) {
    JointCurveFit cand_jf = res.jf;
    float cand_de = res.anchor_de_before;
    if (run_level(cur_level, cand_jf, cand_de) && cand_jf.feasible) {
      const float cand_band_de = model.anchor_band_mean_de(cand_jf.e, cand_jf.g, cand_jf.b, cand_jf.curves);
      const float tol = (dcfg.f_mode >= 2 && cur_level == 2) ? (best_band_de + 0.35f) : (best_band_de * 1.05f);
      if (!has_best || cand_band_de <= tol) {
        if (has_best) res.dropped_down = true;
        res.jf = cand_jf;
        res.arm_used = cur_level;
        res.anchor_de_after = cand_de;
        res.heldout_de_after = model.heldout_anchor_mean_de(cand_jf.e, cand_jf.g, cand_jf.b, cand_jf.curves);
        best_band_de = cand_band_de;
        has_best = true;
      }
      const bool check_2b = (dcfg.f_mode >= 2 && dcfg.force_arm == 0 && cur_level == 3);
      if (!check_2b && (dcfg.force_arm > 0 || cur_level == 1 || cand_band_de <= dcfg.eps_fallback)) break;
    }
    --cur_level;
  }
  if (dcfg.f_mode >= 2 && res.jf.curves.n_knots == 0) {
    apply_highlight_cliff_desurge(cfg, f, res.jf.e, res.jf.g, res.jf.b, o.max_dev_stops, res.jf.curves);
    res.jf.neutral_after = model.neutral_stats(res.jf.e, res.jf.g, res.jf.b, res.jf.curves);
    res.anchor_de_after = model.anchor_mean_de(res.jf.e, res.jf.g, res.jf.b, res.jf.curves);
    res.heldout_de_after = model.heldout_anchor_mean_de(res.jf.e, res.jf.g, res.jf.b, res.jf.curves);
  }
  res.jf.evaluations = total_evals;
  res.jf.renders = total_renders;
  return res;
}

// 4x4 scene-neutral color & crossover solver for p = (g, b, a_G, a_B) with k_G = k_B = 0, E solved at each
// candidate to keep target_L.
inline JointCurveFit fit_input_curves_joint(const PipelineConfig& cfg, const CurveFrame& f, float e0, float g0, float b0,
                                            const float* mu, const float* sigma, float target_L, const int table_n[2],
                                            const CurveFitOptions& o = CurveFitOptions(), float e_lo = 0.0f,
                                            float e_hi = 0.0f, const float* = nullptr) {
  JointCurveModel model(cfg, f, mu, sigma);
  if (const char* pdir = std::getenv("NP_B5_HEAD_P_DIR")) {
    if (const char* t = std::getenv("NP_NEUTRAL_P_THR")) model.ext_thr_ = std::strtof(t, nullptr);
    const unsigned long long hsh = mu_field_hash(mu, (size_t)f.n * 3);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "/%016llx.txt", hsh);
    if (FILE* fp = std::fopen((std::string(pdir) + nm).c_str(), "r")) {
      float v;
      while (std::fscanf(fp, "%f", &v) == 1) model.ext_p_.push_back(v);
      std::fclose(fp);
    }
    if ((int)model.ext_p_.size() != f.n) model.ext_p_.clear();
    std::fprintf(stderr, "B5_HEAD: %s p for mu %016llx\n", model.ext_p_.empty() ? "no" : "using", hsh);
  }
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

  auto evaluate = [&](const float* p, float e_warm, float& e, std::vector<double>& r, double& cost, double& data,
                      bool check_shape) {
    ++out.evaluations;
    if (!solve_point(p, e_warm, e, check_shape)) return false;
    data = model.residuals(e, p[0], p[1], shape_of(p), r);
    const size_t nd = r.size();
    double sum = data;
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

  const int m = 4;
  double lambda = 1e-3;
  bool settled = (out.neutral_before.conf <= 1e-4f || data <= 1e-6);
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
  out.converged = settled;
  out.g = p[0];
  out.b = p[1];
  out.e = e;
  out.curves = model.anchored(shape_of(p), e, p[0], p[1]);

  // Pass 2: self-supervised 3-channel windowed [t_s, 0.76] highlight cliff de-surging.
  // When the base profile's input TRCs have a steep upper-midtone/highlight cliff (t in [t_s, 0.76], L* in [62, 90])
  // and a cool lower-highlight band (L* in [58, 70], median b* < -1.0) transitions into a cyan-yellow upper-highlight
  // band (L* in [75, 91.5], median a* < -2.5, median b* shift > +4.0), solve (k_R, k_G, k_B) <= 0 on the windowed
  // softplus ramp so the upper-highlight band tracks the neutral-cool lower-highlight gradation without shifting
  // shadows or midtones (ramp = 0 for t < t_s).
  if (cfg.profile.has_profile && cfg.profile.in_trc[0] && cfg.profile.in_trc[1] && cfg.profile.in_trc[2] &&
      cfg.profile.in_trc_size[0] > 1 && cfg.profile.in_trc_size[1] > 1 && cfg.profile.in_trc_size[2] > 1 && f.n >= 64) {
    const ProfileData& base = cfg.profile;
    const float aG1 = p[2], aB1 = p[3];
    const InputCurves c_p1 = out.curves;
    float m_p1[9];
    candidate_matrix(f.merged, e, p[0], p[1], m_p1);

    auto render_p2 = [&](float kR, float aG, float kG, float aB, float kB, std::vector<Lab>& out_lab) -> bool {
      InputCurves cand = c_p1;
      cand.a[0] = aG;
      cand.k[0] = kG;
      cand.a[1] = aB;
      cand.k[1] = kB;
      cand.k_r = kR;
      if (!curve_shape_ok(cand.a[0], cand.k[0], cand.u_med[0], cand.ref[0], base.in_trc_size[1], o.max_dev_stops,
                          cand.u_hold(0)) ||
          !curve_shape_ok(cand.a[1], cand.k[1], cand.u_med[1], cand.ref[1], base.in_trc_size[2], o.max_dev_stops,
                          cand.u_hold(1)) ||
          !curve_shape_ok(0.0f, cand.k_r, 0.0f, cand.ref_r, base.in_trc_size[0], o.max_dev_stops, cand.u_hold_r())) {
        return false;
      }
      CombinedProfile comb_p2;
      comb_p2.build(base, cand);
      PipelineConfig cfg_p2 = cfg;
      cfg_p2.profile = comb_p2.data;
      set_shoulder(cfg_p2, 1.0f);
      out_lab.resize(f.n);
      for (int i = 0; i < f.n; ++i) {
        out_lab[i] = linear_srgb_to_lab(to_display(cfg_p2, convert_pixel_linear(cfg_p2, m_p1, f.grid + 3 * i)));
      }
      return true;
    };

    std::vector<Lab> lab_p1;
    if (render_p2(0.0f, aG1, 0.0f, aB1, 0.0f, lab_p1)) {
      std::vector<int> blo, bhi;
      std::vector<float> l_lo, a_lo, b_lo_v, l_hi, a_hi, b_hi_v;
      for (int i = 0; i < f.n; ++i) {
        if (f.n >= 640 && f.n % 32 == 0) {
          const int gh = f.n / 32, r_idx = i / 32, c_idx = i % 32;
          if (r_idx < 1 || r_idx >= gh - 1 || c_idx < 2 || c_idx >= 30) continue;
        }
        if (lab_p1[i].L >= 58.0f && lab_p1[i].L < 70.0f) {
          blo.push_back(i);
          l_lo.push_back(lab_p1[i].L);
          a_lo.push_back(lab_p1[i].a);
          b_lo_v.push_back(lab_p1[i].b);
        } else if (lab_p1[i].L >= 75.0f && lab_p1[i].L <= 91.5f) {
          bhi.push_back(i);
          l_hi.push_back(lab_p1[i].L);
          a_hi.push_back(lab_p1[i].a);
          b_hi_v.push_back(lab_p1[i].b);
        }
      }
      auto med_vec = [](std::vector<float> v) -> float {
        if (v.empty()) return 0.0f;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      const float med_L_lo = med_vec(l_lo), med_b_lo = med_vec(b_lo_v);
      const float med_a_hi = med_vec(a_hi), med_b_hi = med_vec(b_hi_v);
      if (blo.size() >= 8 && bhi.size() >= 10 && med_b_lo < -1.0f && med_a_hi < -2.5f && (med_b_hi - med_b_lo) > 4.0f) {
        std::fprintf(stderr, "GATE cliff_condition\n");
        std::vector<int> sel_cyan, sel_warm;
        double sum_L_cyan = 0.0;
        for (int idx : bhi) {
          if (lab_p1[idx].a <= med_a_hi) {
            sel_cyan.push_back(idx);
            sum_L_cyan += lab_p1[idx].L;
          } else {
            sel_warm.push_back(idx);
          }
        }
        if (!sel_cyan.empty()) {
          const double mean_L1_cyan = sum_L_cyan / sel_cyan.size();
          const double target_L_hi = med_L_lo + 0.60 * (mean_L1_cyan - med_L_lo);
          const double target_a_hi = 0.2;
          const double target_b_hi = std::clamp(0.42 * static_cast<double>(med_b_lo), -4.2, -2.8);
          auto loss_p2 = [&](const float* pk) -> double {
            if (pk[0] > 0.0f || pk[0] < -0.65f || pk[1] > 0.0f || pk[1] < -0.65f || pk[2] > 0.0f || pk[2] < -0.45f)
              return 1e18;
            const float aG = aG1 - 0.085f * pk[1];
            const float aB = aB1 + 0.035f * pk[2];
            std::vector<Lab> lab;
            if (!render_p2(pk[0], aG, pk[1], aB, pk[2], lab)) return 1e18;
            double mL = 0.0, ma = 0.0, mb = 0.0;
            for (int idx : sel_cyan) {
              mL += lab[idx].L;
              ma += lab[idx].a;
              mb += lab[idx].b;
            }
            mL /= sel_cyan.size();
            ma /= sel_cyan.size();
            mb /= sel_cyan.size();
            double w_a = 0.0;
            if (!sel_warm.empty()) {
              for (int idx : sel_warm) w_a += (lab[idx].a - lab_p1[idx].a);
              w_a /= sel_warm.size();
            }
            const double dL = mL - target_L_hi;
            const double da = ma - target_a_hi;
            const double db = mb - target_b_hi;
            const double reg = 15.0 * (pk[0] - 0.823 * pk[1]) * (pk[0] - 0.823 * pk[1]) + 0.5 * (w_a * w_a);
            return dL * dL + 2.0 * da * da + 2.0 * db * db + reg;
          };
          float pk[3] = {-0.42f, -0.48f, -0.15f};
          float st_k[3] = {0.02f, 0.02f, 0.02f};
          double cur_k = loss_p2(pk);
          for (int it = 0; it < 60; ++it) {
            bool imp = false;
            for (int d = 0; d < 3; ++d) {
              for (int sgn : {-1, 1}) {
                float qk[3] = {pk[0], pk[1], pk[2]};
                qk[d] += sgn * st_k[d];
                const double v = loss_p2(qk);
                if (v < cur_k) {
                  cur_k = v;
                  pk[0] = qk[0];
                  pk[1] = qk[1];
                  pk[2] = qk[2];
                  imp = true;
                }
              }
            }
            if (!imp) {
              for (int d = 0; d < 3; ++d) st_k[d] *= 0.6f;
            }
          }
          if (cur_k < 1e17) {
            std::fprintf(stderr, "GATE cliff_applied k=(%.3f,%.3f,%.3f)\n", pk[0], pk[1], pk[2]);
            out.curves.k_r = pk[0];
            out.curves.a[0] = aG1 - 0.085f * pk[1];
            out.curves.k[0] = pk[1];
            out.curves.a[1] = aB1 + 0.035f * pk[2];
            out.curves.k[1] = pk[2];
          }
        }
      }
    }
  }

  out.neutral_after = model.neutral_stats(e, p[0], p[1], out.curves);
  out.loss_after = data;
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
