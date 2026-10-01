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

#ifndef DINOV3_AUTO_SOLVER_H
#define DINOV3_AUTO_SOLVER_H

#include <cstdint>
#include <string>
#include <vector>

#include "dinov3_engine.h"
#include "neg_pipeline.h"

// Flags for pin_mask (manual argument overrides)
constexpr unsigned PIN_E = 1 << 0;
constexpr unsigned PIN_G = 1 << 1;
constexpr unsigned PIN_B = 1 << 2;

struct IntentGains {
  float e = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
};

struct IntentLoss {
  float L = 0.0f;
  float a = 0.0f;
  float b = 0.0f;
  float clip = 0.0f;  // in percent [0, 100]
};

struct SolverConfig {
  const ProfileData* prof = nullptr;
  float knee = 0.95f;
  bool has_gamma = false;
  float inv_gamma = 1.0f;
};

struct DinoIterationLog {
  char tag = ' ';  // 'b' for bootstrap, '0', '1', etc.
  int step = 0;
  float e = 1.0f;  // normalized E (relative to E_c)
  float g = 1.0f;
  float b = 1.0f;
  float residual = -1.0f;  // -1 if unmeasured (bootstrap)
  float loss_L = 0.0f;
  float loss_a = 0.0f;
  float loss_b = 0.0f;
  float clip_pct = 0.0f;
  float penalty_clip = 0.0f;
  float penalty_anchor = 0.0f;
  float objective = 0.0f;
  float dp = 0.0f;
  float rel = 0.0f;
};

struct DinoSolveResult {
  std::string status = "CONVERGED";  // "CONVERGED", "DIVERGED", "MAX_ITERS", "PINNED"
  int best_iteration = -1;
  float best_residual = 0.0f;
  IntentGains solved_relative;  // relative to user baseline (E_c * e_best, g_best, b_best)
  IntentGains solved_absolute;  // E_user * E_c * e_best, g_user * g_best, b_user * b_best
  float e_center = 1.0f;
  bool anchor_clamped = false;  // E_c bisection did not straddle median L* 50
  float bounds_lo = 0.35355339f;
  float bounds_hi = 2.82842712f;
  std::vector<DinoIterationLog> iterations;
  std::vector<float> best_mu;
  std::vector<float> best_sigma;
  std::vector<float> tone_mass;
  int grid_w = 0;
  int grid_h = 0;
  int target_w = 0;
  int target_h = 0;
  int total_inferences = 0;
  double timing_views_ms = 0.0;
  double timing_render_ms = 0.0;
  double timing_infer_ms = 0.0;
  double timing_residual_ms = 0.0;
  double timing_search_ms = 0.0;
  double timing_total_ms = 0.0;
};

// Resamples from 4-channel image buffer (imgdata.image) to 3-channel u16 grid view straight from the crop
void resample_grid_rgb16(const uint16_t (*src)[4], int src_w, int src_h,
                         uint16_t* dst, int dst_w, int dst_h);

// Batched loss calculation across candidate matrices
void intent_loss(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                 const IntentGains* cand, int n_cand, const float* mu, const float* sigma, IntentLoss* out);

// Coordinate descent parameter search against a frozen intent field (mu, sigma)
float optimize_target_for_intent(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                                 int gw, int gh, const float* mu, const float* sigma,
                                 const IntentGains* initial, const float* e_bounds /* lo, hi, center or null */,
                                 unsigned pin_mask, IntentGains& out);

// Solves optimal (E, g, b) parameters against active DINOv3 intent field
bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const uint16_t (*image)[4], int w, int h,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out);

#endif  // DINOV3_AUTO_SOLVER_H
