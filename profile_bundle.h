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
#include <functional>
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

  // For the automatic conversion (auto_convert.h): the patches' green range in double, the bracket's camera
  // exposure (shutter * ISO / 100, seconds), the tonal capacity of its grey wedge and the profcheck record.
  double g_min = 0.0;
  double g_max = 0.0;
  double exposure_s = 0.125;
  double tonal_capacity_ev = 0.0;
  std::string profcheck_output;

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
  std::string key;          // bundle key ("Portra400-A7R4-TriLED"): the export record's, else parsed from path
  std::string source_file;  // the station bundle a split bundle was exported from
  std::string camera_name;
  float crosstalk_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double crosstalk_matrix_d[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool has_crosstalk = false;
  bool has_film_base = false;
  int profile_film_base_rgb[3] = {1, 1, 1};
  double film_base_avg[3] = {0.0, 0.0, 0.0};   // unrounded film base (profile_film_base_rgb rounds it)
  std::string film_base_shutter = "1/8s";
  double film_base_iso = 100.0;
  double normalization_target = 55000.0;
  std::vector<BundleTarget> targets;
};

// Seconds of a shutter speed as profiles write it ("1/8s", "0.4s", "2"); throws std::invalid_argument when it
// cannot be read.
// py-ref: negicc_station/src/film_profiling.py @ 93be1ec (shutter_to_seconds)
double shutter_to_seconds(const std::string& shutter);

// Bundle and stock keys of a profile name: "profile_Portra160-A7R4-TriLED_20261003_190751.json" gives
// ("Portra160-A7R4-TriLED", "portra160"). Empty strings when the name has none.
// py-ref: negicc_station/src/intent_model.py @ 93be1ec (_parse_profile_stem)
void parse_profile_stem(const std::string& raw_name, std::string& bundle_key, std::string& stock_key);

// Check by content if file is a JSON profile bundle (first non-space character is '{')
bool is_json_bundle(const std::string& resolved_path);

// Parse AToB0 pipeline tags from an open cmsHPROFILE into ParsedFilmProfile
bool parse_film_profile_pipeline(cmsHPROFILE hprof, ParsedFilmProfile& prof);

// Configure solver bounds and anchor for a specific target bracket
void set_target_bounds(SolverConfig& cfg, const BundleTarget& tgt);

// Reads a file a split bundle names (a target's "icc_profile") into out; false when it cannot.
using BundleFileReader = std::function<bool(const std::string& name, std::vector<uint8_t>& out)>;

// Load and parse a multi-bracket JSON profile bundle: an all-in-one bundle from negicc-station (ICC profiles in
// base64), or a split one (profiles/<Key>/bundle.json, its directory, with the ICC profiles in files next to it;
// negicc-station tools/export_negicc_data.py).
bool load_profile_bundle(const std::string& resolved_path, ProfileBundle& bundle);

// Parse a JSON profile bundle held in memory (one embedded in neg_process); name is reported as bundle.path and
// read_file supplies the ICC profiles of a split bundle.
bool load_profile_bundle_from_memory(const char* data, size_t size, const std::string& name,
                                     const BundleFileReader& read_file, ProfileBundle& bundle);

// Compute scene green percentiles [0.5%, 99.5%] and gamut volume enclosures for all targets
void compute_bundle_enclosures(ProfileBundle& bundle, const uint16_t (*image)[4],
                              int width, int height, float& g_lo, float& g_hi,
                              float& scene_ev, int& num_feasible);

// Target selection without DINOv3 (picks feasible target minimizing |fill_ratio - 0.85|)
int select_bundle_target_without_dino(const ProfileBundle& bundle, int min_target_idx = kDefaultMinTargetIdx);

#endif  // PROFILE_BUNDLE_H
