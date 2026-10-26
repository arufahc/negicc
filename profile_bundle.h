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
#include "cfa_pipeline.h"
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

  // For the conversion parameter search (multi_target_optimizer.h): the bracket's camera exposure
  // (shutter * ISO / 100, seconds), the tonal capacity of its grey wedge and the profcheck record.
  float exposure_s = 0.125f;
  float tonal_capacity_ev = 0.0f;
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

// The colour space a bundle's brackets were measured in, which is the space a scan is decoded to for it.
// py-ref: negicc_station/src/film_profiling.py @ 813166c (FilmProfile.input_space)
enum class InputSpace {
  // Sensor RGB (narrowband TriLED light): black subtracted, no white balance, then the bundle's crosstalk matrix.
  kCameraRawRgb,
  // Linear Rec.2020 (white D65 light): LibRaw's white-light decode (neg_process.cc decode_white_light), with no
  // crosstalk matrix.
  kRec2020Linear,
};

// "camera_raw_rgb" or "rec2020_linear", as the bundles write it.
const char* to_string(InputSpace space);

// TODO: Hold each bracket's highlight ceiling here (now parsed from the profcheck report by
// profile_highlight_ceiling in multi_target_optimizer.cc) once the profile bundle files are backfilled with it and
// the profiling code writes it.
struct ProfileBundle {
  std::string path;
  std::string key;          // bundle key ("Portra400-A7R4-TriLED"): the export record's, else parsed from path
  std::string source_file;  // the station bundle a split bundle was exported from
  std::string camera_name;
  InputSpace input_space = InputSpace::kCameraRawRgb;
  // The crosstalk correction of a camera_raw_rgb bundle; identity (has_crosstalk false) for rec2020_linear.
  float crosstalk_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double crosstalk_matrix_d[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool has_crosstalk = false;
  bool has_film_base = false;
  int profile_film_base_rgb[3] = {1, 1, 1};
  float film_base_avg[3] = {0.0f, 0.0f, 0.0f};    // unrounded film base (profile_film_base_rgb rounds it)
  double film_base_avg_d[3] = {0.0, 0.0, 0.0};    // double precision for exact station parity
  std::string film_base_shutter = "1/8s";
  float film_base_iso = 100.0f;
  double normalization_target = 55000.0;
  std::vector<BundleTarget> targets;
};

// Seconds of a shutter speed as the profiles record it, a string ("1/8s", "0.4s", "2"); throws
// std::invalid_argument when it cannot be read.
// py-ref: negicc_station/src/film_profiling.py @ 93be1ec (shutter_to_seconds)
float shutter_to_seconds(const std::string& shutter);
double shutter_to_seconds_d(const std::string& shutter);

// Reads (ExposureTime in seconds, ISO) in double precision from a TIFF-based RAW's EXIF IFD (matching
// auto_exposure.read_exif_exposure). Returns true when ExposureTime was read.
bool read_exif_exposure_d(const std::string& filepath, double& shutter_s, double& iso);

// Computes negicc-station's merged crosstalk matrix (film_profiling.compute_merged_crosstalk) in double precision
// and rounds to float32 when out_f32 is non-null. When film_base_from_scan is true (e.g. rebate sampled from the
// scan itself), exposure_ratio is 1.0; otherwise exposure_ratio is exposure_base / exposure_scan.
void compute_station_merged_matrix(const ProfileBundle& bundle, double scan_shutter_s, double scan_iso,
                                   const double* custom_film_base_rgb, bool film_base_from_scan,
                                   float* out_f32, double* out_f64 = nullptr, double* out_row_scale_f64 = nullptr);

// Bundle and stock keys of a profile name: "profile_Portra160-A7R4-TriLED_20261003_190751.json" gives
// ("Portra160-A7R4-TriLED", "portra160"). Empty strings when the name has none.
// py-ref: negicc_station/src/intent_model.py @ 93be1ec (_parse_profile_stem)
void parse_profile_stem(const std::string& raw_name, std::string& bundle_key, std::string& stock_key);

// Check by content if file is a JSON profile bundle (first non-space character is '{')
bool is_json_bundle(const std::string& resolved_path);

// Parse AToB0 pipeline tags from an open cmsHPROFILE into ParsedFilmProfile
bool parse_film_profile_pipeline(cmsHPROFILE hprof, ParsedFilmProfile& prof);

// Apply C1 power-8 FIT3 highlight shoulder to linear sRGB in [0, 1] after pcs_to_linear_srgb
Rgb apply_fit3_highlight_shoulder_rgb(Rgb lin);

// Configure solver bounds and anchor for a specific target bracket
void set_target_bounds(SolverConfig& cfg, const BundleTarget& tgt);

// Reads a file a split bundle names (a target's "icc_profile") into out; false when it cannot.
using BundleFileReader = std::function<bool(const std::string& name, std::vector<uint8_t>& out)>;

// Load and parse a multi-bracket JSON profile bundle: an all-in-one bundle from negicc-station (ICC profiles in
// base64), or a split one (profiles/<Key>/bundle.json, its directory, with the ICC profiles in files next to it;
// negicc-station tools/export_negicc_data.py). Throws when the bundle cannot be read or has no usable bracket, when
// its input_space is neither camera_raw_rgb nor rec2020_linear (no input_space is camera_raw_rgb: bundles from
// before the field), or when a rec2020_linear bundle carries a crosstalk correction matrix.
void load_profile_bundle(const std::string& resolved_path, ProfileBundle& bundle);

// Parse a JSON profile bundle held in memory (one embedded in neg_process); name is reported as bundle.path and
// read_file supplies the ICC profiles of a split bundle. Throws as load_profile_bundle.
void load_profile_bundle_from_memory(const char* data, size_t size, const std::string& name,
                                     const BundleFileReader& read_file, ProfileBundle& bundle);

// Compute scene green percentiles [0.5%, 99.5%] and gamut volume enclosures for all targets
void compute_bundle_enclosures(ProfileBundle& bundle, const uint16_t (*image)[4],
                              int width, int height, float& g_lo, float& g_hi,
                              float& scene_ev, int& num_feasible);
void compute_bundle_enclosures(ProfileBundle& bundle, const negicc::RgbView& view,
                              float& g_lo, float& g_hi, float& scene_ev, int& num_feasible);

// Target selection without DINOv3 (picks feasible target minimizing |fill_ratio - 0.85|)
int select_bundle_target_without_dino(const ProfileBundle& bundle, int min_target_idx = kDefaultMinTargetIdx);

#endif  // PROFILE_BUNDLE_H
