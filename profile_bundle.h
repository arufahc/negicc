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

#ifndef PROFILE_BUNDLE_H
#define PROFILE_BUNDLE_H

#include <cstdint>
#include <string>
#include <vector>
#include <lcms2.h>

#include "neg_pipeline.h"
#include "dinov3_auto_solver.h"

// Default minimum target bracket index to consider (skips extreme highlight overexposure brackets, e.g. +2EV, +1.5EV)
static constexpr int kDefaultMinTargetIdx = 2;

struct BundleTarget {
  int index = 0;
  std::string name;
  std::string display_name;
  double iso = 100.0;
  std::string shutter = "1/8s";
  bool has_target_ev = false;
  double target_ev = 0.0;
  std::vector<float> patches_g;
  float tgt_min_g = 0.0f;
  float tgt_max_g = 0.0f;
  std::vector<uint8_t> icc_bytes;
  ParsedFilmProfile film_prof;

  // Frame enclosure metrics
  bool is_feasible = false;
  float e_min = 0.0f;
  float e_max = 0.0f;
  float e_center = 0.0f;
  float slack_ev = 0.0f;
  float fill_ratio = 0.0f;
  float tgt_capacity_ev = 0.0f;
};

struct ProfileBundle {
  std::string path;
  std::string camera_name;
  float crosstalk_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool has_crosstalk = false;
  bool has_film_base = false;
  int profile_film_base_rgb[3] = {1, 1, 1};
  double normalization_target = 55000.0;
  std::vector<BundleTarget> targets;
};

// Check by content if file is a JSON profile bundle (first non-space character is '{')
bool is_json_bundle(const std::string& resolved_path);

// Parse AToB0 pipeline tags from an open cmsHPROFILE into ParsedFilmProfile
bool parse_film_profile_pipeline(cmsHPROFILE hprof, ParsedFilmProfile& prof);

// Configure solver bounds and anchor for a specific target bracket
void set_target_bounds(SolverConfig& cfg, const BundleTarget& tgt);

// Load and parse a multi-bracket JSON profile bundle
bool load_profile_bundle(const std::string& resolved_path, ProfileBundle& bundle);

// Compute scene green percentiles [0.5%, 99.5%] and gamut volume enclosures for all targets
void compute_bundle_enclosures(ProfileBundle& bundle, const uint16_t (*image)[4],
                              int width, int height, float& g_lo, float& g_hi,
                              float& scene_ev, int& num_feasible);

// Target selection without DINOv3 (picks feasible target minimizing |fill_ratio - 0.85|)
int select_bundle_target_without_dino(const ProfileBundle& bundle, int min_target_idx = kDefaultMinTargetIdx);

// Tournament target selection with active DINOv3 fixed-point loop on best 2 finalists
int run_bundle_tournament_with_dino(ProfileBundle& bundle, DinoV3Engine& engine,
                                   const SolverConfig& base_cfg, const float* base_merged,
                                   const uint16_t (*image)[4], int width, int height,
                                   IntentGains user_gains, unsigned pin_mask, int max_iters,
                                   int min_target_idx, DinoSolveResult& winner_res);

#endif  // PROFILE_BUNDLE_H
