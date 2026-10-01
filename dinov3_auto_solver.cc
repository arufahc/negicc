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
#include <cstring>
#include <algorithm>
#include <vector>

#include "dinov3_preprocess.h"

static constexpr float TOL_PARAM = 0.02f;
static constexpr float TOL_REL_ERR = 0.015f;

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

static inline float compute_penalties(double e, float hi_clip, double e_center, float& pen_clip, float& pen_e) {
  const float clip_excess = std::max(0.0f, hi_clip - 2.5f) / 2.0f;
  pen_clip = 15.0f * (clip_excess * clip_excess);
  if (e_center > 0.0) {
    const double log2_ratio = std::log2(e / e_center);
    pen_e = (float)(5.0 * log2_ratio * log2_ratio);
  } else {
    pen_e = 0.0f;
  }
  return pen_clip + pen_e;
}

float optimize_target_for_intent(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                                 int gw, int gh, const float* mu, const float* sigma,
                                 const IntentGains* initial, const float* e_bounds,
                                 unsigned pin_mask, IntentGains& out) {
  const int n_cells = gw * gh;
  double curr_g = (initial != nullptr) ? (double)initial->g : 1.22;
  double curr_b = (initial != nullptr) ? (double)initial->b : 1.15;

  double e_min = 0.40, e_max = 1.80, e_center = 1.0;
  std::vector<double> e_cands_d;

  if (pin_mask & PIN_E) {
    const double fixed_e = (initial != nullptr) ? (double)initial->e : 1.0;
    e_cands_d.push_back(fixed_e);
    e_min = fixed_e;
    e_max = fixed_e;
    e_center = fixed_e;
  } else if (e_bounds != nullptr) {
    e_min = e_bounds[0];
    e_max = e_bounds[1];
    e_center = e_bounds[2];
    for (int i = 0; i < 13; ++i) {
      e_cands_d.push_back(e_min + (e_max - e_min) * ((double)i / 12.0));
    }
    if (initial != nullptr) {
      const double de_list[5] = {-0.06, -0.03, 0.0, 0.03, 0.06};
      for (double de : de_list) {
        const double c = (double)initial->e + de;
        if (c >= e_min && c <= e_max) {
          e_cands_d.push_back(c);
        }
      }
    }
    std::sort(e_cands_d.begin(), e_cands_d.end());
    e_cands_d.erase(std::unique(e_cands_d.begin(), e_cands_d.end()), e_cands_d.end());
  } else if (initial != nullptr) {
    const double init_e = (double)initial->e;
    e_min = 0.20;
    e_max = 3.0;
    e_center = init_e;
    const double de_list[9] = {-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15};
    for (double d : de_list) {
      if (init_e + d >= 0.20) e_cands_d.push_back(init_e + d);
    }
  } else {
    e_min = 0.40;
    e_max = 1.80;
    e_center = 1.0;
    const double fixed_cands[11] = {0.40, 0.50, 0.60, 0.70, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70};
    for (double val : fixed_cands) e_cands_d.push_back(val);
  }

  std::vector<IntentGains> cands_e(e_cands_d.size());
  for (size_t i = 0; i < e_cands_d.size(); ++i) {
    cands_e[i] = {(float)e_cands_d[i], (float)curr_g, (float)curr_b};
  }

  std::vector<IntentLoss> losses_e(cands_e.size());
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands_e.data(), (int)cands_e.size(), mu, sigma, losses_e.data());

  size_t best_k = 0;
  float best_score_e = 1e30f;
  for (size_t i = 0; i < e_cands_d.size(); ++i) {
    float p_clip = 0.0f, p_e = 0.0f;
    const float score = losses_e[i].L + compute_penalties(e_cands_d[i], losses_e[i].clip, e_center, p_clip, p_e);
    if (score < best_score_e) {
      best_score_e = score;
      best_k = i;
    }
  }
  double best_e = e_cands_d[best_k];

  // E refinement (sequential)
  if (!(pin_mask & PIN_E)) {
    const double step = (e_max > e_min) ? std::max(0.01, (e_max - e_min) / 40.0) : 0.02;
    const double de_steps[4] = {-2.0 * step, -step, step, 2.0 * step};
    for (double de : de_steps) {
      const double e_test = best_e + de;
      if (e_test < e_min || e_test > e_max) continue;
      const IntentGains cand{(float)e_test, (float)curr_g, (float)curr_b};
      IntentLoss r;
      intent_loss(cfg, merged, grid_rgb16, n_cells, &cand, 1, mu, sigma, &r);
      float p_clip = 0.0f, p_e = 0.0f;
      const float s = r.L + compute_penalties(e_test, r.clip, e_center, p_clip, p_e);
      if (s < best_score_e) {
        best_score_e = s;
        best_e = e_test;
      }
    }
  }

  // Chromatic gains: g sweep then b sweep
  std::vector<double> g_cands_d;
  std::vector<double> b_cands_d;
  if (pin_mask & PIN_G) {
    g_cands_d.push_back(curr_g);
  } else if (initial != nullptr) {
    const double dg_list[5] = {-0.04, -0.02, 0.0, 0.02, 0.04};
    for (double d : dg_list) g_cands_d.push_back((double)initial->g + d);
  } else {
    const double def_g[6] = {0.98, 1.04, 1.10, 1.16, 1.22, 1.28};
    for (double v : def_g) g_cands_d.push_back(v);
  }

  if (pin_mask & PIN_B) {
    b_cands_d.push_back(curr_b);
  } else if (initial != nullptr) {
    const double db_list[5] = {-0.06, -0.03, 0.0, 0.03, 0.06};
    for (double d : db_list) b_cands_d.push_back((double)initial->b + d);
  } else {
    const double def_b[6] = {0.85, 0.95, 1.05, 1.15, 1.25, 1.35};
    for (double v : def_b) b_cands_d.push_back(v);
  }

  std::vector<IntentGains> cands_g(g_cands_d.size());
  for (size_t i = 0; i < g_cands_d.size(); ++i) {
    cands_g[i] = {(float)best_e, (float)g_cands_d[i], (float)curr_b};
  }
  std::vector<IntentLoss> losses_g(cands_g.size());
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands_g.data(), (int)cands_g.size(), mu, sigma, losses_g.data());

  size_t best_g_idx = 0;
  float best_loss_a = 1e30f;
  for (size_t i = 0; i < g_cands_d.size(); ++i) {
    if (losses_g[i].a < best_loss_a) {
      best_loss_a = losses_g[i].a;
      best_g_idx = i;
    }
  }
  double best_g = g_cands_d[best_g_idx];

  std::vector<IntentGains> cands_b(b_cands_d.size());
  for (size_t i = 0; i < b_cands_d.size(); ++i) {
    cands_b[i] = {(float)best_e, (float)best_g, (float)b_cands_d[i]};
  }
  std::vector<IntentLoss> losses_b(cands_b.size());
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands_b.data(), (int)cands_b.size(), mu, sigma, losses_b.data());

  size_t best_b_idx = 0;
  float best_loss_b = 1e30f;
  for (size_t i = 0; i < b_cands_d.size(); ++i) {
    if (losses_b[i].b < best_loss_b) {
      best_loss_b = losses_b[i].b;
      best_b_idx = i;
    }
  }
  double best_b = b_cands_d[best_b_idx];

  // g/b fine-tune (sequential, chained)
  double best_gb = (double)best_loss_a + (double)best_loss_b;
  const double dg_chain[4] = {-0.02, 0.02, 0.0, 0.0};
  const double db_chain[4] = {0.0, 0.0, -0.02, 0.02};
  for (int step = 0; step < 4; ++step) {
    if ((pin_mask & PIN_G) && dg_chain[step] != 0.0) continue;
    if ((pin_mask & PIN_B) && db_chain[step] != 0.0) continue;
    const double test_g = best_g + dg_chain[step];
    const double test_b = best_b + db_chain[step];
    const IntentGains cand{(float)best_e, (float)test_g, (float)test_b};
    IntentLoss r;
    intent_loss(cfg, merged, grid_rgb16, n_cells, &cand, 1, mu, sigma, &r);
    if ((double)r.a + (double)r.b < best_gb) {
      best_gb = (double)r.a + (double)r.b;
      best_g = test_g;
      best_b = test_b;
    }
  }

  // Final balanced intent loss and score
  const IntentGains final_cand{(float)best_e, (float)best_g, (float)best_b};
  IntentLoss r_final;
  intent_loss(cfg, merged, grid_rgb16, n_cells, &final_cand, 1, mu, sigma, &r_final);
  float p_clip = 0.0f, p_e = 0.0f;
  const float penalties = compute_penalties(best_e, r_final.clip, e_center, p_clip, p_e);
  const float loss_intent = (r_final.L + r_final.a + r_final.b) / 3.0f;

  out = final_cand;
  return loss_intent + penalties;
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

bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const uint16_t (*image)[4], int w, int h,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out) {
  if (!image || w <= 0 || h <= 0 || !cfg.prof) return false;

  const double t0_total = omp_get_wtime();

  int target_w = 0, target_h = 0, grid_w = 0, grid_h = 0;
  compute_aspect_preserved_shape(w, h, eng.max_size(), eng.patch_size(), target_w, target_h, grid_w, grid_h);

  out.grid_w = grid_w;
  out.grid_h = grid_h;
  out.target_w = target_w;
  out.target_h = target_h;

  const int n_cells = grid_w * grid_h;
  std::vector<uint16_t> grid_rgb16(n_cells * 3);

  const double t0_views = omp_get_wtime();
  resample_grid_rgb16(image, w, h, grid_rgb16.data(), grid_w, grid_h);
  out.timing_views_ms += (omp_get_wtime() - t0_views) * 1000.0;

  // 1. Exposure anchor E_c bisection (if E is not pinned)
  double e_center = 1.0;
  if (!(pin_mask & PIN_E)) {
    double log2_lo = -4.0, log2_hi = 4.0;
    for (int iter = 0; iter < 16; ++iter) {
      const double mid = 0.5 * (log2_lo + log2_hi);
      const double e_cand = std::pow(2.0, mid);
      float m[9];
      candidate_matrix(user_merged, (float)e_cand, 1.10f, 1.15f, m);

      std::vector<float> l_vals(n_cells);
      for (int i = 0; i < n_cells; ++i) {
        const Rgb v = convert_pixel_srgb(grid_rgb16.data() + i * 3, m, cfg.knee, cfg.has_gamma, cfg.inv_gamma, *cfg.prof);
        const Lab lab = srgb_to_lab(v);
        l_vals[i] = lab.L;
      }
      std::nth_element(l_vals.begin(), l_vals.begin() + n_cells / 2, l_vals.end());
      const float med_L = l_vals[n_cells / 2];
      if (med_L < 50.0f) {
        log2_lo = mid;
      } else {
        log2_hi = mid;
      }
    }
    e_center = std::pow(2.0, 0.5 * (log2_lo + log2_hi));
  }
  out.e_center = (float)e_center;

  // Build merged_norm matrix: all rows scaled by E_c
  float merged_norm[9];
  for (int i = 0; i < 9; ++i) {
    merged_norm[i] = user_merged[i] * (float)e_center;
  }

  const float e_bounds[3] = {
    (float)std::pow(2.0, -1.5),
    (float)std::pow(2.0, 1.5),
    1.0f
  };
  out.bounds_lo = e_bounds[0];
  out.bounds_hi = e_bounds[1];

  std::vector<uint8_t> preview_u8((size_t)w * h * 3);

  // If all parameters are pinned, render once to acquire intent features and exit
  if (pin_mask == (PIN_E | PIN_G | PIN_B)) {
    const double t0_r = omp_get_wtime();
    render_crop_preview(cfg, merged_norm, image, w, h, 1.0f / (float)e_center, 1.0f, 1.0f, preview_u8.data());
    out.timing_render_ms += (omp_get_wtime() - t0_r) * 1000.0;

    const double t0_inf = omp_get_wtime();
    const auto inf = eng.infer(preview_u8.data(), w, h, w * 3);
    out.timing_infer_ms += (omp_get_wtime() - t0_inf) * 1000.0;
    out.total_inferences = 1;

    out.status = "PINNED";
    out.best_iteration = 0;
    out.best_residual = 0.0f;
    out.solved_relative = {1.0f, 1.0f, 1.0f};
    out.solved_absolute = user_gains;
    out.best_mu = inf.mu;
    out.best_sigma = inf.sigma;
    out.tone_mass = inf.tone_mass;
    return true;
  }

  // Bootstrap inference at normalized theta_boot = (1.0, 1.10, 1.15)
  const double t0_r_boot = omp_get_wtime();
  render_crop_preview(cfg, merged_norm, image, w, h, 1.0f, 1.10f, 1.15f, preview_u8.data());
  out.timing_render_ms += (omp_get_wtime() - t0_r_boot) * 1000.0;

  const double t0_inf_boot = omp_get_wtime();
  const auto inf_boot = eng.infer(preview_u8.data(), w, h, w * 3);
  out.timing_infer_ms += (omp_get_wtime() - t0_inf_boot) * 1000.0;
  out.total_inferences++;

  // Log bootstrap row
  IntentGains boot_gains{1.0f, 1.10f, 1.15f};
  IntentLoss boot_loss;
  intent_loss(cfg, merged_norm, grid_rgb16.data(), n_cells, &boot_gains, 1,
              inf_boot.mu.data(), inf_boot.sigma.data(), &boot_loss);
  float boot_p_clip = 0.0f, boot_p_e = 0.0f;
  const float boot_penalties = compute_penalties(1.0, boot_loss.clip, 1.0, boot_p_clip, boot_p_e);
  const float boot_obj = (boot_loss.L + boot_loss.a + boot_loss.b) / 3.0f + boot_penalties;

  DinoIterationLog boot_log;
  boot_log.tag = 'b';
  boot_log.step = -1;
  boot_log.e = 1.0f;
  boot_log.g = 1.10f;
  boot_log.b = 1.15f;
  boot_log.residual = -1.0f;
  boot_log.loss_L = boot_loss.L;
  boot_log.loss_a = boot_loss.a;
  boot_log.loss_b = boot_loss.b;
  boot_log.clip_pct = boot_loss.clip;
  boot_log.penalty_clip = boot_p_clip;
  boot_log.penalty_anchor = boot_p_e;
  boot_log.objective = boot_obj;
  out.iterations.push_back(boot_log);

  // Bootstrap initial search
  const double t0_search_boot = omp_get_wtime();
  IntentGains curr;
  optimize_target_for_intent(cfg, merged_norm, grid_rgb16.data(), grid_w, grid_h,
                             inf_boot.mu.data(), inf_boot.sigma.data(),
                             nullptr, e_bounds, pin_mask, curr);
  out.timing_search_ms += (omp_get_wtime() - t0_search_boot) * 1000.0;

  float prev_err = 1e30f;
  float best_err = 1e30f;
  IntentGains best_params = curr;
  DinoV3InferenceResult best_inf = inf_boot;
  int best_it = -1;

  std::string status = "MAX_ITERS";
  const int loop_limit = std::max(1, max_iters);

  for (int it = 0; it < loop_limit; ++it) {
    const double t0_r = omp_get_wtime();
    render_crop_preview(cfg, merged_norm, image, w, h, curr.e, curr.g, curr.b, preview_u8.data());
    out.timing_render_ms += (omp_get_wtime() - t0_r) * 1000.0;

    const double t0_inf = omp_get_wtime();
    const auto inf = eng.infer(preview_u8.data(), w, h, w * 3);
    out.timing_infer_ms += (omp_get_wtime() - t0_inf) * 1000.0;
    out.total_inferences++;

    const double t0_resid = omp_get_wtime();
    const float err = self_consistency_residual(preview_u8.data(), w, h, grid_w, grid_h,
                                               inf.mu.data(), inf.sigma.data());
    out.timing_residual_ms += (omp_get_wtime() - t0_resid) * 1000.0;

    if (err < best_err) {
      best_err = err;
      best_params = curr;
      best_inf = inf;
      best_it = it;
    }

    // Divergence check
    if (it > 0 && err > prev_err * 1.25f && (err - prev_err) > 1.5f) {
      status = "DIVERGED";
      DinoIterationLog div_log;
      div_log.tag = '0' + (char)it;
      div_log.step = it;
      div_log.e = curr.e;
      div_log.g = curr.g;
      div_log.b = curr.b;
      div_log.residual = err;
      out.iterations.push_back(div_log);
      break;
    }

    // Search against latest mu, sigma
    const double t0_s = omp_get_wtime();
    IntentGains next_params;
    optimize_target_for_intent(cfg, merged_norm, grid_rgb16.data(), grid_w, grid_h,
                               inf.mu.data(), inf.sigma.data(),
                               &curr, e_bounds, pin_mask, next_params);
    out.timing_search_ms += (omp_get_wtime() - t0_s) * 1000.0;

    const float dp = std::abs(next_params.e - curr.e) +
                     std::abs(next_params.g - curr.g) +
                     std::abs(next_params.b - curr.b);
    const float rel = (it > 0) ? (std::abs(err - prev_err) / std::max(1e-4f, prev_err)) : 1.0f;

    // Evaluate iteration diagnostic losses
    IntentLoss it_loss;
    intent_loss(cfg, merged_norm, grid_rgb16.data(), n_cells, &curr, 1,
                inf.mu.data(), inf.sigma.data(), &it_loss);
    float it_p_clip = 0.0f, it_p_e = 0.0f;
    const float it_pen = compute_penalties(curr.e, it_loss.clip, 1.0, it_p_clip, it_p_e);
    const float it_obj = (it_loss.L + it_loss.a + it_loss.b) / 3.0f + it_pen;

    DinoIterationLog it_log;
    it_log.tag = '0' + (char)it;
    it_log.step = it;
    it_log.e = curr.e;
    it_log.g = curr.g;
    it_log.b = curr.b;
    it_log.residual = err;
    it_log.loss_L = it_loss.L;
    it_log.loss_a = it_loss.a;
    it_log.loss_b = it_loss.b;
    it_log.clip_pct = it_loss.clip;
    it_log.penalty_clip = it_p_clip;
    it_log.penalty_anchor = it_p_e;
    it_log.objective = it_obj;
    it_log.dp = dp;
    it_log.rel = rel;
    out.iterations.push_back(it_log);

    if (it > 0 && (dp <= TOL_PARAM || rel <= TOL_REL_ERR)) {
      status = "CONVERGED";
      curr = next_params;
      prev_err = err;
      break;
    }

    curr = next_params;
    prev_err = err;
  }

  out.status = status;
  out.best_iteration = best_it;
  out.best_residual = best_err;
  out.solved_relative = {
    (float)(e_center * best_params.e),
    best_params.g,
    best_params.b
  };
  out.solved_absolute = {
    (float)(user_gains.e * e_center * best_params.e),
    user_gains.g * best_params.g,
    user_gains.b * best_params.b
  };
  out.best_mu = best_inf.mu;
  out.best_sigma = best_inf.sigma;
  out.tone_mass = best_inf.tone_mass;
  out.timing_total_ms = (omp_get_wtime() - t0_total) * 1000.0;

  return true;
}
