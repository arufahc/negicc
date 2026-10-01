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

#include "dinov3_auto_solver.h"

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "dinov3_preprocess.h"

static constexpr double TOL_PARAM = 0.02;
static constexpr double TOL_REL_ERR = 0.015;

void resample_grid_rgb16(const uint16_t (*src)[4], int src_w, int src_h,
                         uint16_t* dst, int dst_w, int dst_h) {
  if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;

  const double scale_x = (double)src_w / (double)dst_w;
  const double scale_y = (double)src_h / (double)dst_h;

  #pragma omp parallel for schedule(static) if((size_t)dst_w * dst_h > 8192)
  for (int dy = 0; dy < dst_h; ++dy) {
    const double y0 = dy * scale_y;
    const double y1 = (dy + 1) * scale_y;
    const int sy_start = (int)std::floor(y0);
    const int sy_end = std::min(src_h, (int)std::ceil(y1));

    for (int dx = 0; dx < dst_w; ++dx) {
      const double x0 = dx * scale_x;
      const double x1 = (dx + 1) * scale_x;
      const int sx_start = (int)std::floor(x0);
      const int sx_end = std::min(src_w, (int)std::ceil(x1));

      double r_acc = 0.0, g_acc = 0.0, b_acc = 0.0, w_acc = 0.0;

      for (int sy = sy_start; sy < sy_end; ++sy) {
        const double wy = std::min(y1, (double)(sy + 1)) - std::max(y0, (double)sy);
        if (wy <= 0.0) continue;
        const uint16_t* row_ptr = src[(size_t)sy * src_w];

        for (int sx = sx_start; sx < sx_end; ++sx) {
          const double wx = std::min(x1, (double)(sx + 1)) - std::max(x0, (double)sx);
          if (wx <= 0.0) continue;
          const double w = wx * wy;

          const size_t idx = (size_t)sx * 4;
          r_acc += row_ptr[idx] * w;
          g_acc += row_ptr[idx + 1] * w;
          b_acc += row_ptr[idx + 2] * w;
          w_acc += w;
        }
      }

      const double inv_w = (w_acc > 0.0) ? (1.0 / w_acc) : 0.0;
      const size_t dst_idx = ((size_t)dy * dst_w + dx) * 3;
      dst[dst_idx]     = (uint16_t)std::clamp(std::round(r_acc * inv_w), 0.0, 65535.0);
      dst[dst_idx + 1] = (uint16_t)std::clamp(std::round(g_acc * inv_w), 0.0, 65535.0);
      dst[dst_idx + 2] = (uint16_t)std::clamp(std::round(b_acc * inv_w), 0.0, 65535.0);
    }
  }
}

void intent_loss(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                 const IntentGains* cand, int n_cand, const float* mu, const float* sigma, IntentLoss* out) {
  #pragma omp parallel for schedule(dynamic)
  for (int c = 0; c < n_cand; ++c) {
    float m[9];
    candidate_matrix(merged, cand[c].e, cand[c].g, cand[c].b, m);
    double sL = 0.0, sa = 0.0, sb = 0.0, clip = 0.0;
    for (int i = 0; i < n_px; ++i) {
      const uint16_t* px = grid_rgb16 + i * 3;
      const Rgb v = convert_pixel_srgb(px, m, cfg.knee, cfg.has_gamma, cfg.inv_gamma, *cfg.prof);
      clip += channel_white_clipped(v.r) + channel_white_clipped(v.g) + channel_white_clipped(v.b);
      const Lab lab = srgb_to_lab(v);
      const float dL = (lab.L - mu[i * 3]) / sigma[i * 3];
      const float da = (lab.a - mu[i * 3 + 1]) / sigma[i * 3 + 1];
      const float db = (lab.b - mu[i * 3 + 2]) / sigma[i * 3 + 2];
      sL += dL * dL;
      sa += da * da;
      sb += db * db;
    }
    out[c].L = (float)(sL / n_px);
    out[c].a = (float)(sa / n_px);
    out[c].b = (float)(sb / n_px);
    out[c].clip = (float)(clip / (3.0 * n_px) * 100.0);
  }
}

// Search state in double, as the reference's Python scalars; candidates are rounded to float only at intent_loss.
struct GainsD {
  double e, g, b;
};

static inline double compute_penalties(double e, float hi_clip, double e_center, double& pen_clip, double& pen_e) {
  const double clip_excess = std::max(0.0, (double)hi_clip - 2.5) / 2.0;
  pen_clip = 15.0 * (clip_excess * clip_excess);
  if (e_center > 0.0) {
    const double log2_ratio = std::log2(e / e_center);
    pen_e = 5.0 * log2_ratio * log2_ratio;
  } else {
    pen_e = 0.0;
  }
  return pen_clip + pen_e;
}

static inline IntentLoss eval_one(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                                  double e, double g, double b, const float* mu, const float* sigma) {
  const IntentGains cand{(float)e, (float)g, (float)b};
  IntentLoss r;
  intent_loss(cfg, merged, grid_rgb16, n_px, &cand, 1, mu, sigma, &r);
  return r;
}

static double optimize_impl(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                            int gw, int gh, const float* mu, const float* sigma,
                            const GainsD* initial, const double* e_bounds,
                            unsigned pin_mask, GainsD& out) {
  const int n_cells = gw * gh;
  // A pinned gain stays at its start value (relative 1 without initial), never at the reference's free defaults.
  double curr_g = initial ? initial->g : ((pin_mask & PIN_G) ? 1.0 : 1.22);
  double curr_b = initial ? initial->b : ((pin_mask & PIN_B) ? 1.0 : 1.15);

  double e_min, e_max, e_center;
  std::vector<double> e_cands_d;

  if (pin_mask & PIN_E) {
    const double fixed_e = initial ? initial->e : 1.0;
    e_cands_d.push_back(fixed_e);
    e_min = e_max = e_center = fixed_e;
  } else if (e_bounds != nullptr) {
    e_min = e_bounds[0];
    e_max = e_bounds[1];
    e_center = e_bounds[2];
    // np.linspace: start + i * step, endpoint pinned to stop
    const double lin_step = (e_max - e_min) / 12.0;
    for (int i = 0; i < 12; ++i) e_cands_d.push_back(e_min + i * lin_step);
    e_cands_d.push_back(e_max);
    if (initial) {
      for (double de : {-0.06, -0.03, 0.0, 0.03, 0.06}) {
        const double c = initial->e + de;
        if (c >= e_min && c <= e_max) e_cands_d.push_back(c);
      }
    }
    std::sort(e_cands_d.begin(), e_cands_d.end());
    e_cands_d.erase(std::unique(e_cands_d.begin(), e_cands_d.end()), e_cands_d.end());
  } else if (initial) {
    e_min = 0.20;
    e_max = 3.0;
    e_center = initial->e;
    for (double d : {-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15}) {
      if (initial->e + d >= 0.20) e_cands_d.push_back(initial->e + d);
    }
  } else {
    e_min = 0.40;
    e_max = 1.80;
    e_center = 1.0;
    e_cands_d = {0.40, 0.50, 0.60, 0.70, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70};
  }

  // 1. Exposure sweep (first minimum)
  std::vector<IntentGains> cands(e_cands_d.size());
  for (size_t i = 0; i < e_cands_d.size(); ++i) cands[i] = {(float)e_cands_d[i], (float)curr_g, (float)curr_b};
  std::vector<IntentLoss> losses(cands.size());
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());

  double p_clip, p_e;
  size_t best_k = 0;
  double best_score_e = 0.0;
  for (size_t i = 0; i < e_cands_d.size(); ++i) {
    const double score = (double)losses[i].L + compute_penalties(e_cands_d[i], losses[i].clip, e_center, p_clip, p_e);
    if (i == 0 || score < best_score_e) {
      best_score_e = score;
      best_k = i;
    }
  }
  double best_e = e_cands_d[best_k];

  // Exposure refinement (sequential, fixed base)
  if (!(pin_mask & PIN_E)) {
    const double step = (e_max > e_min) ? std::max(0.01, (e_max - e_min) / 40.0) : 0.02;
    const double base_e = best_e;
    for (double de : {-2.0 * step, -step, step, 2.0 * step}) {
      const double e_test = base_e + de;
      if (e_test < e_min || e_test > e_max) continue;
      const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, e_test, curr_g, curr_b, mu, sigma);
      const double s = (double)r.L + compute_penalties(e_test, r.clip, e_center, p_clip, p_e);
      if (s < best_score_e) {
        best_score_e = s;
        best_e = e_test;
      }
    }
  }

  // 2. Chromatic gains: g sweep, then b sweep
  std::vector<double> g_cands_d, b_cands_d;
  if (pin_mask & PIN_G) {
    g_cands_d = {curr_g};
  } else if (initial) {
    for (double d : {-0.04, -0.02, 0.0, 0.02, 0.04}) g_cands_d.push_back(initial->g + d);
  } else {
    g_cands_d = {0.98, 1.04, 1.10, 1.16, 1.22, 1.28};
  }
  if (pin_mask & PIN_B) {
    b_cands_d = {curr_b};
  } else if (initial) {
    for (double d : {-0.06, -0.03, 0.0, 0.03, 0.06}) b_cands_d.push_back(initial->b + d);
  } else {
    b_cands_d = {0.85, 0.95, 1.05, 1.15, 1.25, 1.35};
  }

  cands.resize(g_cands_d.size());
  losses.resize(g_cands_d.size());
  for (size_t i = 0; i < g_cands_d.size(); ++i) cands[i] = {(float)best_e, (float)g_cands_d[i], (float)curr_b};
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
  size_t k = 0;
  for (size_t i = 1; i < g_cands_d.size(); ++i) {
    if (losses[i].a < losses[k].a) k = i;
  }
  double best_g = g_cands_d[k];
  const float best_loss_a = losses[k].a;

  cands.resize(b_cands_d.size());
  losses.resize(b_cands_d.size());
  for (size_t i = 0; i < b_cands_d.size(); ++i) cands[i] = {(float)best_e, (float)best_g, (float)b_cands_d[i]};
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
  k = 0;
  for (size_t i = 1; i < b_cands_d.size(); ++i) {
    if (losses[i].b < losses[k].b) k = i;
  }
  double best_b = b_cands_d[k];
  const float best_loss_b = losses[k].b;

  // 3. g/b fine-tune (sequential, chained)
  double best_gb = (double)best_loss_a + (double)best_loss_b;
  const double chain[4][2] = {{-0.02, 0.0}, {0.02, 0.0}, {0.0, -0.02}, {0.0, 0.02}};
  for (const auto& d : chain) {
    if ((pin_mask & PIN_G) && d[0] != 0.0) continue;
    if ((pin_mask & PIN_B) && d[1] != 0.0) continue;
    const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, best_e, best_g + d[0], best_b + d[1], mu, sigma);
    if ((double)r.a + (double)r.b < best_gb) {
      best_gb = (double)r.a + (double)r.b;
      best_g += d[0];
      best_b += d[1];
    }
  }

  // 4. Final balanced intent loss
  const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, best_e, best_g, best_b, mu, sigma);
  const double loss_intent = ((double)r.L + (double)r.a + (double)r.b) / 3.0;
  out = {best_e, best_g, best_b};
  return loss_intent + compute_penalties(best_e, r.clip, e_center, p_clip, p_e);
}

float optimize_target_for_intent(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                                 int gw, int gh, const float* mu, const float* sigma,
                                 const IntentGains* initial, const float* e_bounds,
                                 unsigned pin_mask, IntentGains& out) {
  GainsD init_d, out_d;
  if (initial) init_d = {initial->e, initial->g, initial->b};
  double bounds_d[3];
  if (e_bounds) {
    for (int i = 0; i < 3; ++i) bounds_d[i] = e_bounds[i];
  }
  const double score = optimize_impl(cfg, merged, grid_rgb16, gw, gh, mu, sigma, initial ? &init_d : nullptr,
                                     e_bounds ? bounds_d : nullptr, pin_mask, out_d);
  out = {(float)out_d.e, (float)out_d.g, (float)out_d.b};
  return (float)score;
}

static float self_consistency_residual(const uint8_t* preview_u8, int w, int h, int gw, int gh,
                                       const float* mu, const float* sigma) {
  std::vector<uint8_t> resized(gw * gh * 3);
  cv_linear_resize_rgb(preview_u8, w, h, w * 3, resized.data(), gw, gh);

  double sum_res = 0.0;
  const int n = gw * gh;
  for (int i = 0; i < n; ++i) {
    const Rgb c = { resized[i * 3 + 0] / 255.0f,
                    resized[i * 3 + 1] / 255.0f,
                    resized[i * 3 + 2] / 255.0f };
    const Lab lab = srgb_to_lab(c);
    const float dL = (lab.L - mu[i * 3 + 0]) / sigma[i * 3 + 0];
    const float da = (lab.a - mu[i * 3 + 1]) / sigma[i * 3 + 1];
    const float db = (lab.b - mu[i * 3 + 2]) / sigma[i * 3 + 2];
    sum_res += (dL * dL + da * da + db * db) / 3.0;
  }
  return (float)(sum_res / n);
}

static void render_crop_preview(const SolverConfig& cfg, const float* merged_norm,
                                const uint16_t (*image)[4], int w, int h,
                                float e, float g, float b, uint8_t* out_u8) {
  float m[9];
  candidate_matrix(merged_norm, e, g, b, m);

  #pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const uint16_t* in_row = image[(size_t)y * w];
    uint8_t* out_row = out_u8 + (size_t)y * w * 3;
    for (int x = 0; x < w; ++x) {
      const Rgb v = convert_pixel_srgb(in_row + x * 4, m, cfg.knee, cfg.has_gamma, cfg.inv_gamma, *cfg.prof);
      out_row[x * 3 + 0] = quantize_u8(v.r);
      out_row[x * 3 + 1] = quantize_u8(v.g);
      out_row[x * 3 + 2] = quantize_u8(v.b);
    }
  }
}

// Inference with the shape and sigma checks the loss relies on (no sigma floor: the reference divides by raw sigma).
static bool infer_checked(DinoV3Engine& eng, const uint8_t* preview_u8, int w, int h, int gw, int gh,
                          DinoV3InferenceResult& res, DinoSolveResult& out) {
  const double t0 = omp_get_wtime();
  try {
    res = eng.infer(preview_u8, w, h, w * 3);
  } catch (const std::exception& e) {
    fprintf(stderr, "ERROR! DINOv3 inference failed: %s\n", e.what());
    return false;
  }
  out.timing_infer_ms += (omp_get_wtime() - t0) * 1000.0;
  out.total_inferences++;
  const size_t n = (size_t)gw * gh * 3;
  if (res.grid_w != gw || res.grid_h != gh || res.mu.size() != n || res.sigma.size() != n) {
    fprintf(stderr, "ERROR! DINOv3 intent field shape %dx%d (%zu values) does not match grid %dx%d.\n",
            res.grid_w, res.grid_h, res.mu.size(), gw, gh);
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!(res.sigma[i] > 0.0f) || !std::isfinite(res.sigma[i]) || !std::isfinite(res.mu[i])) {
      fprintf(stderr, "ERROR! DINOv3 intent field has non-positive or non-finite sigma/mu at %zu.\n", i);
      return false;
    }
  }
  return true;
}

static DinoIterationLog make_log(char tag, int step, const GainsD& p, float residual, const IntentLoss& l,
                                 double e_center) {
  DinoIterationLog log;
  log.tag = tag;
  log.step = step;
  log.e = (float)p.e;
  log.g = (float)p.g;
  log.b = (float)p.b;
  log.residual = residual;
  log.loss_L = l.L;
  log.loss_a = l.a;
  log.loss_b = l.b;
  log.clip_pct = l.clip;
  double p_clip, p_e;
  const double pen = compute_penalties(p.e, l.clip, e_center, p_clip, p_e);
  log.penalty_clip = (float)p_clip;
  log.penalty_anchor = (float)p_e;
  log.objective = (float)(((double)l.L + (double)l.a + (double)l.b) / 3.0 + pen);
  return log;
}

bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const uint16_t (*image)[4], int w, int h,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out) {
  if (!image || w <= 0 || h <= 0 || !cfg.prof || !user_merged) return false;

  const double t0_total = omp_get_wtime();

  int target_w = 0, target_h = 0, grid_w = 0, grid_h = 0;
  compute_aspect_preserved_shape(w, h, eng.max_size(), eng.patch_size(), target_w, target_h, grid_w, grid_h);
  out.grid_w = grid_w;
  out.grid_h = grid_h;
  out.target_w = target_w;
  out.target_h = target_h;

  const int n_cells = grid_w * grid_h;
  if (n_cells <= 0) return false;
  std::vector<uint16_t> grid_rgb16((size_t)n_cells * 3);

  const double t0_views = omp_get_wtime();
  resample_grid_rgb16(image, w, h, grid_rgb16.data(), grid_w, grid_h);
  out.timing_views_ms += (omp_get_wtime() - t0_views) * 1000.0;

  // Bootstrap gains (reference (e_c, 1.10, 1.15)); a pinned gain stays at relative 1.
  const double boot_g = (pin_mask & PIN_G) ? 1.0 : 1.10;
  const double boot_b = (pin_mask & PIN_B) ? 1.0 : 1.15;

  // 1. Exposure anchor E_c: median L* of the grid render = 50, bisection on log2 E over [-4, 4].
  double e_center = 1.0;
  if (!(pin_mask & PIN_E)) {
    const double lo0 = -4.0, hi0 = 4.0;
    double log2_lo = lo0, log2_hi = hi0;
    std::vector<float> l_vals(n_cells);
    for (int iter = 0; iter < 16; ++iter) {
      const double mid = 0.5 * (log2_lo + log2_hi);
      float m[9];
      candidate_matrix(user_merged, (float)std::pow(2.0, mid), (float)boot_g, (float)boot_b, m);
      for (int i = 0; i < n_cells; ++i) {
        const Rgb v = convert_pixel_srgb(grid_rgb16.data() + (size_t)i * 3, m, cfg.knee, cfg.has_gamma,
                                         cfg.inv_gamma, *cfg.prof);
        l_vals[i] = srgb_to_lab(v).L;
      }
      std::nth_element(l_vals.begin(), l_vals.begin() + n_cells / 2, l_vals.end());
      // Negative: more transmittance (larger E) renders a darker positive, so median L* falls with E.
      if (l_vals[n_cells / 2] < 50.0f) {
        log2_hi = mid;
      } else {
        log2_lo = mid;
      }
    }
    out.anchor_clamped = (log2_lo == lo0 || log2_hi == hi0);
    e_center = std::pow(2.0, 0.5 * (log2_lo + log2_hi));
  }
  out.e_center = (float)e_center;

  // Search in E_c-normalized units: all rows scaled once by E_c.
  float merged_norm[9];
  for (int i = 0; i < 9; ++i) merged_norm[i] = user_merged[i] * (float)e_center;

  const double e_bounds[3] = {std::pow(2.0, -1.5), std::pow(2.0, 1.5), 1.0};
  out.bounds_lo = (float)e_bounds[0];
  out.bounds_hi = (float)e_bounds[1];

  std::vector<uint8_t> preview_u8((size_t)w * h * 3);
  DinoV3InferenceResult inf;

  // All pinned: no search; one render + inference for the intent field at the user's parameters.
  if (pin_mask == (PIN_E | PIN_G | PIN_B)) {
    const double t0_r = omp_get_wtime();
    render_crop_preview(cfg, merged_norm, image, w, h, 1.0f, 1.0f, 1.0f, preview_u8.data());
    out.timing_render_ms += (omp_get_wtime() - t0_r) * 1000.0;
    if (!infer_checked(eng, preview_u8.data(), w, h, grid_w, grid_h, inf, out)) return false;

    out.status = "PINNED";
    out.best_iteration = 0;
    out.best_residual = self_consistency_residual(preview_u8.data(), w, h, grid_w, grid_h,
                                                  inf.mu.data(), inf.sigma.data());
    out.solved_relative = {1.0f, 1.0f, 1.0f};
    out.solved_absolute = user_gains;
    out.best_mu = std::move(inf.mu);
    out.best_sigma = std::move(inf.sigma);
    out.tone_mass = std::move(inf.tone_mass);
    out.timing_total_ms = (omp_get_wtime() - t0_total) * 1000.0;
    return true;
  }

  // 2. Bootstrap: render at (1, boot_g, boot_b), infer, search with initial = null.
  const GainsD boot{1.0, boot_g, boot_b};
  const double t0_r_boot = omp_get_wtime();
  render_crop_preview(cfg, merged_norm, image, w, h, (float)boot.e, (float)boot.g, (float)boot.b, preview_u8.data());
  out.timing_render_ms += (omp_get_wtime() - t0_r_boot) * 1000.0;
  if (!infer_checked(eng, preview_u8.data(), w, h, grid_w, grid_h, inf, out)) return false;

  const double anchor_center = (pin_mask & PIN_E) ? 0.0 : 1.0;  // pinned E drops P_anchor
  out.iterations.push_back(make_log('b', -1, boot, -1.0f,
                                    eval_one(cfg, merged_norm, grid_rgb16.data(), n_cells, boot.e, boot.g, boot.b,
                                             inf.mu.data(), inf.sigma.data()),
                                    anchor_center));

  const double t0_search_boot = omp_get_wtime();
  GainsD curr;
  optimize_impl(cfg, merged_norm, grid_rgb16.data(), grid_w, grid_h, inf.mu.data(), inf.sigma.data(),
                nullptr, e_bounds, pin_mask, curr);
  out.timing_search_ms += (omp_get_wtime() - t0_search_boot) * 1000.0;

  // 3. Fixed-point loop (reference fixed_point).
  double prev_err = HUGE_VAL;
  double best_err = HUGE_VAL;
  GainsD best_params = curr;
  DinoV3InferenceResult best_inf;
  int best_it = -1;
  std::string status = "MAX_ITERS";
  const int loop_limit = std::max(1, max_iters);

  for (int it = 0; it < loop_limit; ++it) {
    const double t0_r = omp_get_wtime();
    render_crop_preview(cfg, merged_norm, image, w, h, (float)curr.e, (float)curr.g, (float)curr.b,
                        preview_u8.data());
    out.timing_render_ms += (omp_get_wtime() - t0_r) * 1000.0;
    if (!infer_checked(eng, preview_u8.data(), w, h, grid_w, grid_h, inf, out)) return false;

    const double t0_resid = omp_get_wtime();
    const double err = self_consistency_residual(preview_u8.data(), w, h, grid_w, grid_h,
                                                 inf.mu.data(), inf.sigma.data());
    out.timing_residual_ms += (omp_get_wtime() - t0_resid) * 1000.0;

    const IntentLoss it_loss = eval_one(cfg, merged_norm, grid_rgb16.data(), n_cells, curr.e, curr.g, curr.b,
                                        inf.mu.data(), inf.sigma.data());
    DinoIterationLog it_log = make_log((char)('0' + it % 10), it, curr, (float)err, it_loss, anchor_center);

    if (err < best_err) {
      best_err = err;
      best_params = curr;
      best_inf = inf;
      best_it = it;
    }

    // Diverged: checked before the search, as the reference.
    if (it > 0 && err > prev_err * 1.25 && (err - prev_err) > 1.5) {
      status = "DIVERGED";
      out.iterations.push_back(it_log);
      break;
    }

    const double t0_s = omp_get_wtime();
    GainsD next;
    optimize_impl(cfg, merged_norm, grid_rgb16.data(), grid_w, grid_h, inf.mu.data(), inf.sigma.data(),
                  &curr, e_bounds, pin_mask, next);
    out.timing_search_ms += (omp_get_wtime() - t0_s) * 1000.0;

    const double dp = std::abs(next.e - curr.e) + std::abs(next.g - curr.g) + std::abs(next.b - curr.b);
    const double rel = (it > 0) ? std::abs(err - prev_err) / std::max(1e-4, prev_err) : 1.0;
    it_log.dp = (float)dp;
    it_log.rel = (float)rel;
    out.iterations.push_back(it_log);

    curr = next;
    prev_err = err;
    if (it > 0 && (dp <= TOL_PARAM || rel <= TOL_REL_ERR)) {
      status = "CONVERGED";
      break;
    }
  }

  if (best_it < 0) {
    fprintf(stderr, "ERROR! DINOv3 solver produced no finite residual.\n");
    return false;
  }

  out.status = status;
  out.best_iteration = best_it;
  out.best_residual = (float)best_err;
  out.solved_relative = {(float)(e_center * best_params.e), (float)best_params.g, (float)best_params.b};
  out.solved_absolute = {(float)((double)user_gains.e * e_center * best_params.e),
                         (float)((double)user_gains.g * best_params.g),
                         (float)((double)user_gains.b * best_params.b)};
  out.best_mu = std::move(best_inf.mu);
  out.best_sigma = std::move(best_inf.sigma);
  out.tone_mass = std::move(best_inf.tone_mass);
  out.timing_total_ms = (omp_get_wtime() - t0_total) * 1000.0;
  return true;
}
