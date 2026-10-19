// Copyright 2024 Alpha Lam <arufa.hc@gmail.com>
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

#include "auto_convert.h"

#include <libraw/libraw.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>

#include "dinov3_auto_solver.h"
#include "dinov3_preprocess.h"
#include "frame_analysis.h"
#include "neg_pipeline.h"

namespace {

// Fixed-point loop thresholds.
constexpr double TOL_PARAM = 0.02;     // |dE| + |dg| + |db| <= 0.02
constexpr double TOL_REL_ERR = 0.015;  // relative residual change <= 1.5 %
constexpr int MAX_STEPS = 8;
// The scene's range is taken between its SCENE_TAIL and 1 - SCENE_TAIL points.
constexpr double SCENE_TAIL = 0.005;
constexpr double SLACK_PENALTY_SCALE = 0.5;
// The previews, the screening and the final render use a fixed soft knee.
constexpr float AUTO_KNEE = 0.95f;
// Highlight rescue thresholds.
constexpr double RESCUE_CLIP_TAU = 3e-4;
constexpr double RESCUE_CLIP_RATIO = 0.5;
constexpr double RESCUE_DL_MIN = -4.5, RESCUE_DL_MAX = 0.5;
constexpr double RESCUE_RESID_TOL = 0.10;
constexpr int RESCUE_MAX_HOP = 1;
// The search runs as if the frame were scanned at 1/8 s, ISO 100 with the profile's film base.
constexpr double REFERENCE_SHUTTER_S = 0.125;
constexpr double REFERENCE_ISO = 100.0;

double now_ms() { return omp_get_wtime() * 1000.0; }

// ---- Bracket enclosure -------------------------------------------------------------------------------------

struct Enclosure {
  int idx = 0;
  bool feasible = false;
  bool admissible = true;
  double e_min = 0, e_max = 0, search_lo = 0, search_hi = 0, e_center = 0, slack_ev = 0, fill_ratio = 0;
};

// py-ref: negicc_station/src/optimizer.py @ 93be1ec (bracket_enclosure)
bool bracket_enclosure(const BundleTarget& tgt, int idx, double g_lo, double g_hi, Enclosure& r) {
  if (tgt.patches_g.empty()) return false;
  const double tgt_min = tgt.g_min, tgt_max = tgt.g_max;
  const double k_t = (REFERENCE_SHUTTER_S * (REFERENCE_ISO / 100.0)) / tgt.exposure_s;
  r.idx = idx;
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
  const int n = (int)bundle.targets.size();
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

// CIELAB of encoded sRGB in double (OpenCV's constants), as grey_point.to_lab.
void to_lab_d(const float rgb[3], double lab[3]) {
  double lin[3];
  for (int c = 0; c < 3; ++c) {
    const double v = rgb[c];
    lin[c] = (v <= 0.04045) ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
  }
  const double X = (lin[0] * 0.412453 + lin[1] * 0.357580 + lin[2] * 0.180423) / 0.950456;
  const double Y = lin[0] * 0.212671 + lin[1] * 0.715160 + lin[2] * 0.072169;
  const double Z = (lin[0] * 0.019334 + lin[1] * 0.119193 + lin[2] * 0.950227) / 1.088754;
  auto f = [](double t) { return t > 0.008856 ? std::cbrt(t) : 7.787 * t + 16.0 / 116.0; };
  const double fx = f(X), fy = f(Y), fz = f(Z);
  lab[0] = (Y > 0.008856) ? 116.0 * fy - 16.0 : 903.3 * Y;
  lab[1] = 500.0 * (fx - fy);
  lab[2] = 200.0 * (fy - fz);
}

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
  explicit GreyPoint(const ProfileData& prof) : m_prof(prof) {}

  // Lab of inputs t (n x 3), the scale going through E so the 16-bit inputs sit near 0.9.
  void lab(const double (*t)[3], int n, double (*out)[3]) const {
    double tmax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) tmax = std::max(tmax, t[i][c]);
    const double s = tmax / 0.9;
    static const float kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float m[9];
    candidate_matrix(kIdentity, (float)s, 1.0f, 1.0f, m);
    for (int i = 0; i < n; ++i) {
      uint16_t px[3];
      for (int c = 0; c < 3; ++c) {
        const float v = (float)(t[i][c] / s);
        const float vv = std::isfinite(v) ? v : 0.0f;
        px[c] = (uint16_t)std::lround(std::fmin(1.0f, std::fmax(0.0f, vv)) * 65535.0f);
      }
      const Rgb o = convert_pixel_srgb(px, m, AUTO_KNEE, false, 1.0f, m_prof);
      const float rgb[3] = {o.r, o.g, o.b};
      to_lab_d(rgb, out[i]);
    }
  }

  double lstar(const double t[3]) const {
    double lab_out[1][3];
    lab((const double (*)[3])t, 1, lab_out);
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

  const ProfileData& m_prof;
};

// Exposure E that renders scene vector p at L* 50 with gains (g, b): bisection on log2 E in [-6, 6].
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (exposure_for_lightness)
bool exposure_for_lightness(const GreyPoint& gp, const double p[3], double g, double b, double& e_out) {
  constexpr double TARGET = 50.0, TOL = 0.1;
  const double base_t[3] = {p[0], p[1] * g, p[2] * b};
  double lo = -6.0, hi = 6.0;
  for (int i = 0; i < 25; ++i) {
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

// ---- One frame prepared for the search -------------------------------------------------------------------

struct ClipStats {
  float exc[3];      // % of pixels at or above the bracket's ceiling, per channel
  float white_clip;  // % with any channel white-clipped
  float median_L;
};

struct FixedPoint {
  double e = 0, g = 0, b = 0;
  double final_err = 0;
  std::string status;
  int iters = 0;
};

struct Screened {
  int idx = 0;
  double e = 0, g = 0, b = 0, loss = 0;
  double lo = 0, hi = 0, e_met = 0;
  Enclosure meta;
};

class Frame {
 public:
  Frame(const ProfileBundle& bundle, DinoV3Engine& engine, AutoConvertResult& out)
      : m_bundle(bundle), m_engine(engine), m_out(out) {}

  RgbImage frame_view, crop_view;
  FrameStats stats;
  float conv_matrix[9];  // m_frame / e_norm
  double m_frame_d[9];
  double e_norm = 1.0;

  SolverConfig cfg(int idx) const {
    SolverConfig c;
    c.prof = &m_bundle.targets[idx].film_prof.data;
    c.knee = AUTO_KNEE;
    return c;
  }

  void render_preview(int idx, double e, double g, double b, std::vector<uint8_t>& out) {
    const double t0 = now_ms();
    float m[9];
    candidate_matrix(conv_matrix, (float)e, (float)g, (float)b, m);
    out.resize((size_t)crop_view.width * crop_view.height * 3);
    render_srgb_u8(crop_view, m, AUTO_KNEE, m_bundle.targets[idx].film_prof.data, out.data());
    m_out.timing_render_ms += now_ms() - t0;
  }

  // Intent field (mu, sigma) of a preview on the statistics' grid.
  bool predict(const std::vector<uint8_t>& preview, std::vector<float>& mu, std::vector<float>& sigma) {
    const double t0 = now_ms();
    DinoV3InferenceResult r;
    try {
      r = m_engine.infer(preview.data(), crop_view.width, crop_view.height, crop_view.width * 3);
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! DINOv3 inference failed: %s\n", e.what());
      return false;
    }
    m_out.timing_infer_ms += now_ms() - t0;
    m_out.inferences++;
    const size_t n = (size_t)stats.grid_w * stats.grid_h * 3;
    if (r.grid_w != stats.grid_w || r.grid_h != stats.grid_h || r.mu.size() != n || r.sigma.size() != n) {
      fprintf(stderr, "ERROR! DINOv3 intent field %dx%d does not match the frame grid %dx%d\n", r.grid_w, r.grid_h,
              stats.grid_w, stats.grid_h);
      return false;
    }
    mu = std::move(r.mu);
    sigma = std::move(r.sigma);
    return true;
  }

  double residual(const std::vector<uint8_t>& preview, const std::vector<float>& mu, const std::vector<float>& sigma) {
    const double t0 = now_ms();
    const double err = self_consistency_residual(preview.data(), crop_view.width, crop_view.height, stats.grid_w,
                                                 stats.grid_h, mu.data(), sigma.data());
    m_out.timing_residual_ms += now_ms() - t0;
    return err;
  }

  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (optimize_target_for_intent)
  double search(int idx, const std::vector<float>& mu, const std::vector<float>& sigma, const double* initial,
                const double* e_bounds, double out[3]) {
    const double t0 = now_ms();
    const double score = optimize_target_for_intent_d(cfg(idx), conv_matrix, stats.grid.data(), stats.grid_w,
                                                      stats.grid_h, mu.data(), sigma.data(), initial, e_bounds, 0,
                                                      out);
    m_out.timing_search_ms += now_ms() - t0;
    return score;
  }

  // Render -> intent field -> search, until the gains or the self-consistency residual settle; the gains with the
  // lowest residual win.
  // py-ref: negicc_station/src/per_target_conversion.py @ 93be1ec (_FrameContext.fixed_point)
  bool fixed_point(int idx, const double start[3], double e_bounds[3], FixedPoint& fp) {
    if (e_bounds[0] > e_bounds[1]) {
      std::swap(e_bounds[0], e_bounds[1]);
      e_bounds[2] = (e_bounds[0] + e_bounds[1]) / 2.0;
    }
    double curr[3] = {start[0], start[1], start[2]};
    double prev_err = std::numeric_limits<double>::infinity();
    std::string status = "MAX_ITERS";
    int steps = 0;
    bool have_best = false;
    double best_err = 0.0, best[3] = {0, 0, 0};
    std::vector<uint8_t> preview;
    std::vector<float> mu, sigma;
    for (int it = 0; it < MAX_STEPS; ++it) {
      render_preview(idx, curr[0], curr[1], curr[2], preview);
      if (!predict(preview, mu, sigma)) return false;
      const double err = residual(preview, mu, sigma);
      if (!have_best || err < best_err) {
        have_best = true;
        best_err = err;
        std::copy(curr, curr + 3, best);
      }
      if (it > 0 && err > prev_err * 1.25 && (err - prev_err) > 1.5) {
        status = "DIVERGED";
        steps++;
        break;
      }
      double nw[3];
      search(idx, mu, sigma, curr, e_bounds, nw);
      const double dp = std::fabs(nw[0] - curr[0]) + std::fabs(nw[1] - curr[1]) + std::fabs(nw[2] - curr[2]);
      const double rel = it > 0 ? std::fabs(err - prev_err) / std::max(1e-4, prev_err) : 1.0;
      steps++;
      if (m_verbose) {
        printf("    T%d step %d: E %.4f g %.4f b %.4f resid %.4f\n", idx + 1, it, curr[0], curr[1], curr[2], err);
      }
      if (it > 0 && (dp <= TOL_PARAM || rel <= TOL_REL_ERR)) {
        status = "CONVERGED";
        break;
      }
      std::copy(nw, nw + 3, curr);
      prev_err = err;
    }
    fp.e = best[0];
    fp.g = best[1];
    fp.b = best[2];
    fp.final_err = best_err;
    fp.status = status;
    fp.iters = steps;
    return true;
  }

  // Highlight excess per channel, white clip and median L* over the frame's inner crop, per candidate exposure.
  std::vector<ClipStats> highlight_clip(int idx, const std::vector<double>& es, double g, double b,
                                        const double ceil[3], float crop_ratio) {
    const InnerCrop crop = compute_inner_crop(frame_view.width, frame_view.height, crop_ratio);
    const float ceil_f[3] = {(float)ceil[0], (float)ceil[1], (float)ceil[2]};
    const int n = (int)es.size();
    std::vector<float> mats((size_t)n * 9), rows((size_t)n * 5);
    for (int c = 0; c < n; ++c) candidate_matrix(conv_matrix, (float)es[c], (float)g, (float)b, &mats[(size_t)c * 9]);
    ::highlight_clip(frame_view, crop, mats.data(), n, AUTO_KNEE, m_bundle.targets[idx].film_prof.data, ceil_f,
                     rows.data());
    std::vector<ClipStats> res(n);
    for (int c = 0; c < n; ++c) {
      const float* r = &rows[(size_t)c * 5];
      res[c] = {{r[0], r[1], r[2]}, r[3], r[4]};
    }
    return res;
  }

  bool m_verbose = true;

 private:
  const ProfileBundle& m_bundle;
  DinoV3Engine& m_engine;
  AutoConvertResult& m_out;
};

struct Headroom {
  bool applied = false;
  double delta_ev = 0.0;
  std::string reason;
};

double max_excess(const ClipStats& c) { return std::max({(double)c.exc[0], (double)c.exc[1], (double)c.exc[2]}) / 100.0; }

// Exposure increase (stops) that brings every output channel's (1 - tail) point inside the bracket's calibrated
// range, bounded by the bracket's exposure capacity; a two-stage batched dyadic bisection.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (compute_highlight_headroom_ev)
Headroom highlight_headroom(Frame& fr, int idx, double e, bool has_e_max, double e_max, const double ceil[3], double g,
                            double b, float crop_ratio) {
  constexpr double tail = SCENE_TAIL, tol_ev = 0.01;
  Headroom h;
  const bool cap_known = has_e_max && e_max > 0 && e > 0;
  const double capacity_ev = cap_known ? std::log2(e_max / e) : 0.0;
  const bool has_cap = cap_known && capacity_ev > 0.0;
  std::vector<double> init_es = {e};
  if (has_cap) init_es.push_back(e * std::pow(2.0, capacity_ev));
  const auto hc_init = fr.highlight_clip(idx, init_es, g, b, ceil, crop_ratio);
  const double f0 = max_excess(hc_init[0]);
  if (f0 <= tail) {
    h.reason = "within calibrated range";
    return h;
  }
  if (!has_cap) {
    h.reason = "no exposure capacity (E at or beyond the bracket's e_max)";
    return h;
  }
  if (max_excess(hc_init[1]) > tail) {
    h.applied = true;
    h.delta_ev = capacity_ev;
    h.reason = "capacity limited";
    return h;
  }
  const int total_steps = std::max(0, (int)std::ceil(std::log2(std::max(capacity_ev / tol_ev, 1.0))));
  const int stage1_steps = std::min(4, total_steps);
  double lo = 0.0, hi = capacity_ev;
  if (stage1_steps > 0) {
    const int n1 = (1 << stage1_steps) - 1;
    std::vector<double> es1;
    for (int i = 1; i <= n1; ++i) es1.push_back(e * std::pow(2.0, lo + (i / (double)(1 << stage1_steps)) * (hi - lo)));
    const auto hc1 = fr.highlight_clip(idx, es1, g, b, ceil, crop_ratio);
    int ilo = 0, ihi = 1 << stage1_steps;
    for (int s = 0; s < stage1_steps; ++s) {
      const int mid = (ilo + ihi) / 2;
      if (max_excess(hc1[mid - 1]) <= tail) ihi = mid;
      else ilo = mid;
    }
    lo = (ilo / (double)(1 << stage1_steps)) * capacity_ev;
    hi = (ihi / (double)(1 << stage1_steps)) * capacity_ev;
  }
  const int rem_steps = total_steps - stage1_steps;
  if (rem_steps > 0 && (hi - lo) > tol_ev) {
    const int n2 = (1 << rem_steps) - 1;
    std::vector<double> es2;
    for (int i = 1; i <= n2; ++i) es2.push_back(e * std::pow(2.0, lo + (i / (double)(1 << rem_steps)) * (hi - lo)));
    const auto hc2 = fr.highlight_clip(idx, es2, g, b, ceil, crop_ratio);
    const double span = hi - lo, base_lo = lo;
    int ilo = 0, ihi = 1 << rem_steps;
    for (int s = 0; s < rem_steps; ++s) {
      const int mid = (ilo + ihi) / 2;
      if (max_excess(hc2[mid - 1]) <= tail) ihi = mid;
      else ilo = mid;
    }
    lo = base_lo + (ilo / (double)(1 << rem_steps)) * span;
    hi = base_lo + (ihi / (double)(1 << rem_steps)) * span;
  }
  h.applied = true;
  h.delta_ev = hi;
  h.reason = "pulled inside calibrated range";
  return h;
}

// Merged input matrix: crosstalk rows scaled by normalization_target / film base and the exposure ratio between
// the film base capture and the scan.
// py-ref: negicc_station/src/film_profiling.py @ 93be1ec (compute_merged_crosstalk, compute_exposure_ratio)
void merged_matrix(const ProfileBundle& bundle, const double fb[3], double exposure_ratio, double out[9]) {
  for (int r = 0; r < 3; ++r) {
    const double scale = fb[r] > 0 ? (bundle.normalization_target / fb[r]) * exposure_ratio : 1.0;
    for (int c = 0; c < 3; ++c) out[r * 3 + c] = bundle.crosstalk_matrix_d[r * 3 + c] * scale;
  }
}

}  // namespace

// ---- Public API ----------------------------------------------------------------------------------------------

bool run_auto_convert(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& half, const int bbox_in[4],
                      int rot_cw, bool hflip, bool vflip, double scan_shutter_s, double scan_iso,
                      const AutoConvertOptions& opts, AutoConvertResult& out) {
  const double t_start = now_ms();
  const bool verbose = opts.verbose;
  if (bundle.targets.empty()) {
    fprintf(stderr, "ERROR! The profile bundle has no targets\n");
    return false;
  }
  Frame fr(bundle, engine, out);
  fr.m_verbose = verbose;

  // Frame and its inner crop, oriented.
  int bbox[4] = {bbox_in[0], bbox_in[1], bbox_in[2], bbox_in[3]};
  clamp_rect(bbox, half.width, half.height);
  std::copy(bbox, bbox + 4, out.bbox);
  cut_and_orient(half, bbox[0], bbox[1], bbox[2], bbox[3], rot_cw, hflip, vflip, fr.frame_view);
  compute_frame_stats(fr.frame_view, opts.crop_ratio, engine.max_size(), engine.patch_size(), fr.stats);
  const InnerCrop in = compute_inner_crop(bbox[2], bbox[3], opts.crop_ratio);
  cut_and_orient(half, bbox[0] + in.x0, bbox[1] + in.y0, in.w, in.h, rot_cw, hflip, vflip, fr.crop_view);

  // Bracket enclosure.
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (compute_gamut_volume_enclosure)
  const double g_lo = std::max(1.0, (double)fr.stats.g_lo);
  const double g_hi = std::max(g_lo + 1.0, (double)fr.stats.g_hi);
  out.g_lo = g_lo;
  out.g_hi = g_hi;
  out.scene_ev = std::log2(g_hi / g_lo);
  std::vector<Enclosure> enclosure;
  for (int i : filter_admissible_targets(bundle)) {
    Enclosure r;
    if (bracket_enclosure(bundle.targets[i], i, g_lo, g_hi, r)) enclosure.push_back(r);
  }
  if (enclosure.empty()) {
    fprintf(stderr, "ERROR! No bracket of the bundle has patch readings\n");
    return false;
  }

  // Input matrix in the reference normalisation.
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
  double m_ref[9];
  merged_matrix(bundle, fb, ratio_frame, fr.m_frame_d);
  merged_matrix(bundle, bundle.film_base_avg, ratio_ref, m_ref);
  fr.e_norm = (m_ref[4] != 0.0) ? fr.m_frame_d[4] / m_ref[4] : 1.0;
  for (int i = 0; i < 9; ++i) fr.conv_matrix[i] = (float)(fr.m_frame_d[i] / fr.e_norm);
  out.e_norm = fr.e_norm;
  std::copy(fr.conv_matrix, fr.conv_matrix + 9, out.matrix);
  for (int r = 0; r < 3; ++r) {
    out.row_scale[r] = (fb[r] > 0 ? (bundle.normalization_target / fb[r]) * ratio_frame : 1.0) / fr.e_norm;
  }

  if (verbose) {
    printf("Auto: frame %dx%d at (%d, %d) (half size), crop %dx%d, grid %dx%d\n", bbox[2], bbox[3], bbox[0], bbox[1],
           fr.crop_view.width, fr.crop_view.height, fr.stats.grid_w, fr.stats.grid_h);
    printf("Auto: scene green %.1f..%.1f (%.2f EV), scan %.6g s ISO %.0f, e_norm %.6g\n", g_lo, g_hi, out.scene_ev,
           scan_shutter_s, scan_iso, fr.e_norm);
    printf("  bracket            E_min     E_max  slack_EV   fill  feasible admissible\n");
    for (const auto& r : enclosure) {
      printf("  %-14s  %8.4f  %8.4f  %+8.3f  %5.3f  %8s %10s\n", bundle.targets[r.idx].name.c_str(), r.e_min,
             r.e_max, r.slack_ev, r.fill_ratio, r.feasible ? "yes" : "no", r.admissible ? "yes" : "no");
    }
  }

  auto rank = [](const Enclosure& r) {
    return std::fabs(r.fill_ratio - 0.85) + SLACK_PENALTY_SCALE * std::max(0.0, -r.slack_ev);
  };
  std::vector<Enclosure> ranked = enclosure;
  std::stable_sort(ranked.begin(), ranked.end(), [&](const Enclosure& a, const Enclosure& b) { return rank(a) < rank(b); });
  std::vector<Enclosure> top;
  for (const auto& r : ranked)
    if (r.admissible && top.size() < 2) top.push_back(r);
  if (top.empty()) top.assign(ranked.begin(), ranked.begin() + std::min<size_t>(2, ranked.size()));

  // Pass 1: meter each admissible bracket to L* 50 and bootstrap from the one deepest inside its bounds.
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (select_metered_bootstrap_candidate)
  std::vector<Enclosure> pool;
  for (const auto& r : enclosure)
    if (r.admissible) pool.push_back(r);
  if (pool.empty()) pool = enclosure;
  double p_scene[3];
  for (int r = 0; r < 3; ++r) {
    p_scene[r] = 0.0;
    for (int c = 0; c < 3; ++c) p_scene[r] += (fr.m_frame_d[r * 3 + c] / fr.e_norm) * fr.stats.log_avg_rgb[c];
  }
  struct Depth {
    Enclosure cand;
    double e_met, depth, lo, hi;
  };
  std::vector<Depth> depths;
  for (const auto& c : pool) {
    const GreyPoint gp(bundle.targets[c.idx].film_prof.data);
    double e_met = 0.0;
    if (!gp.reference_grey()) {
      if (verbose) printf("  %s: no neutral L* 50 in this bracket, not metered\n", bundle.targets[c.idx].name.c_str());
      continue;
    }
    if (!exposure_for_lightness(gp, p_scene, 1.10, 1.15, e_met)) {
      if (verbose) printf("  %s: metering did not converge\n", bundle.targets[c.idx].name.c_str());
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
    for (const auto& d : depths) {
      printf("  metered %s: E %.4f depth %+.3f\n", bundle.targets[d.cand.idx].name.c_str(), d.e_met, d.depth);
    }
    printf("Auto: bootstrap from %s at E %.4f\n", bundle.targets[cand1.idx].name.c_str(), e1_init);
  }

  std::vector<uint8_t> preview;
  std::vector<float> mu1, sig1;
  fr.render_preview(cand1.idx, e1_init, 1.10, 1.15, preview);
  if (!fr.predict(preview, mu1, sig1)) return false;

  // Stage 1: screen the brackets against the bootstrap field.
  std::vector<Screened> screening;
  auto screen = [&](const Enclosure& c, double lo, double hi, double e_met) {
    Screened s;
    s.idx = c.idx;
    s.meta = c;
    s.lo = lo;
    s.hi = hi;
    s.e_met = e_met;
    double eb[3] = {lo, hi, c.e_center}, res[3];
    s.loss = fr.search(c.idx, mu1, sig1, nullptr, eb, res);
    s.e = res[0];
    s.g = res[1];
    s.b = res[2];
    screening.push_back(s);
  };
  if (!depths.empty()) {
    for (const auto& d : depths) screen(d.cand, d.lo, d.hi, d.e_met);
  } else {
    for (const auto& c : pool) screen(c, c.search_lo, c.search_hi, c.e_center);
  }
  std::stable_sort(screening.begin(), screening.end(), [](const Screened& a, const Screened& b) { return a.loss < b.loss; });
  std::vector<Screened> shortlist;
  for (const auto& s : screening)
    if (s.loss <= screening[0].loss + 2.0 && shortlist.size() < 2) shortlist.push_back(s);
  if (shortlist.size() < 2 && screening.size() >= 2) shortlist.assign(screening.begin(), screening.begin() + 2);
  for (const auto& s : screening) out.screening.push_back({s.idx, s.e, s.g, s.b, s.loss, s.lo, s.hi, s.e_met});
  if (verbose) {
    printf("Auto: screening\n");
    for (const auto& s : screening) {
      printf("  %-14s E %.4f g %.4f b %.4f loss %.4f\n", bundle.targets[s.idx].name.c_str(), s.e, s.g, s.b, s.loss);
    }
  }

  // Stage 2: fixed-point loop per shortlisted bracket, within a trust region of +/- 0.5 stop.
  auto trust_bounds = [](const Screened& c, double eb[3]) {
    eb[0] = std::max(c.lo, c.e * std::pow(2.0, -0.5));
    eb[1] = std::min(c.hi, c.e * std::pow(2.0, 0.5));
    eb[2] = c.meta.e_center;
  };
  std::vector<std::pair<int, FixedPoint>> adaptive;
  for (const auto& c : shortlist) {
    double eb[3];
    trust_bounds(c, eb);
    const double start[3] = {c.e, c.g, c.b};
    FixedPoint fp;
    if (verbose) printf("Auto: fixed point on %s\n", bundle.targets[c.idx].name.c_str());
    if (!fr.fixed_point(c.idx, start, eb, fp)) return false;
    adaptive.push_back({c.idx, fp});
    out.tournament.push_back({c.idx, fp.e, fp.g, fp.b, fp.final_err, fp.status, fp.iters});
  }
  bool any_converged = false;
  for (const auto& a : adaptive) any_converged |= (a.second.status == "CONVERGED");
  int win_i = -1;
  for (int i = 0; i < (int)adaptive.size(); ++i) {
    if (any_converged && adaptive[i].second.status != "CONVERGED") continue;
    if (win_i < 0 || adaptive[i].second.final_err < adaptive[win_i].second.final_err) win_i = i;
  }
  int chosen = adaptive[win_i].first;
  FixedPoint win = adaptive[win_i].second;
  double e = win.e, g = win.g, b = win.b;

  // Highlight headroom.
  auto bounds_of = [&](int idx, Enclosure& r) {
    for (const auto& c : enclosure)
      if (c.idx == idx) {
        r = c;
        return true;
      }
    return bracket_enclosure(bundle.targets[idx], idx, g_lo, g_hi, r);
  };
  std::map<int, std::array<double, 3>> ceilings;
  auto ceiling_of = [&](int idx) {
    auto it = ceilings.find(idx);
    if (it == ceilings.end()) {
      std::array<double, 3> c;
      profile_highlight_ceiling(bundle.targets[idx].profcheck_output, c.data());
      it = ceilings.emplace(idx, c).first;
    }
    return it->second;
  };
  auto apply_headroom = [&](int idx, double& e_io, double g_c, double b_c) {
    Enclosure hb;
    const bool has_hb = bounds_of(idx, hb);
    const auto ceil = ceiling_of(idx);
    const Headroom h = highlight_headroom(fr, idx, e_io, has_hb, has_hb ? hb.e_max : 0.0, ceil.data(), g_c, b_c,
                                          opts.crop_ratio);
    if (h.delta_ev > 0.0) e_io *= std::pow(2.0, h.delta_ev);
    return h;
  };
  Headroom headroom = apply_headroom(chosen, e, g, b);
  if (verbose) {
    printf("Auto: winner %s (resid %.4f, %s), highlight headroom %+.3f EV (%s)\n",
           bundle.targets[chosen].name.c_str(), win.final_err, win.status.c_str(), headroom.delta_ev,
           headroom.reason.c_str());
  }

  // Highlight rescue: the next bracket when the winner clips and it clips clearly less.
  // py-ref: negicc_station/src/optimizer.py @ 93be1ec (measure_rescue_candidates, select_highlight_rescue)
  {
    const auto c0 = ceiling_of(chosen);
    const ClipStats hc0 = fr.highlight_clip(chosen, {e}, g, b, c0.data(), opts.crop_ratio)[0];
    const double clip_0 = hc0.white_clip / 100.0;
    const double l_0 = hc0.median_L;
    out.rescue_clip = clip_0;
    out.rescue_branch = clip_0 > RESCUE_CLIP_TAU ? "R" : "N";
    if (clip_0 > RESCUE_CLIP_TAU) {
      const int t_adj = chosen + 1;
      const Screened* s_adj = nullptr;
      for (const auto& s : screening)
        if (s.idx == t_adj) s_adj = &s;
      if (!s_adj) {
        if (verbose) printf("Auto: highlight rescue: %d not in the screened pool\n", t_adj + 1);
      } else if (!s_adj->meta.admissible) {
        if (verbose) printf("Auto: highlight rescue: %d not admissible\n", t_adj + 1);
      } else {
        FixedPoint fp_adj;
        bool have = false;
        for (const auto& a : adaptive)
          if (a.first == t_adj) {
            fp_adj = a.second;
            have = true;
          }
        if (!have) {
          double eb[3];
          trust_bounds(*s_adj, eb);
          const double start[3] = {s_adj->e, s_adj->g, s_adj->b};
          if (verbose) printf("Auto: fixed point on %s (highlight rescue)\n", bundle.targets[t_adj].name.c_str());
          if (!fr.fixed_point(t_adj, start, eb, fp_adj)) return false;
        }
        double e_c = fp_adj.e;
        const Headroom h_c = apply_headroom(t_adj, e_c, fp_adj.g, fp_adj.b);
        const auto cc = ceiling_of(t_adj);
        const ClipStats hcc = fr.highlight_clip(t_adj, {e_c}, fp_adj.g, fp_adj.b, cc.data(), opts.crop_ratio)[0];
        const double clip_c = hcc.white_clip / 100.0;
        const double dl = (double)hcc.median_L - l_0;
        const char* fail = nullptr;
        if (std::abs(t_adj - chosen) > RESCUE_MAX_HOP) fail = "hop";
        else if (clip_c > RESCUE_CLIP_RATIO * clip_0) fail = "clip ratio";
        else if (!(RESCUE_DL_MIN <= dl && dl <= RESCUE_DL_MAX)) fail = "L* shift";
        else if (fp_adj.final_err > (1.0 + RESCUE_RESID_TOL) * win.final_err) fail = "residual";
        if (verbose) {
          printf("Auto: highlight rescue: %s clips %.4f%% vs %.4f%%, dL* %+.2f, resid %.4f: %s\n",
                 bundle.targets[t_adj].name.c_str(), clip_c * 100.0, clip_0 * 100.0, dl, fp_adj.final_err,
                 fail ? fail : "taken");
        }
        if (!fail) {
          chosen = t_adj;
          win = fp_adj;
          e = e_c;
          g = fp_adj.g;
          b = fp_adj.b;
          headroom = h_c;
          out.rescue_changed = true;
        }
      }
    }
  }

  out.target_idx = chosen;
  out.target_name = bundle.targets[chosen].name;
  out.e = e / fr.e_norm;
  out.g = g;
  out.b = b;
  out.residual = win.final_err;
  out.status = win.status;
  out.headroom_applied = headroom.applied;
  out.headroom_ev = headroom.delta_ev;
  out.headroom_reason = headroom.reason;
  out.timing_total_ms = now_ms() - t_start;
  return true;
}
