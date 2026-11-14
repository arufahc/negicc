// Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
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
#include <array>
#include <string>
#include <vector>
#include <lcms2.h>

#include "color_pipeline.h"
#include "rgb_view.h"

struct SolverConfig;  // dinov3_auto_solver.h

// --------------------------------------------------------------------------------------------------------------------
// The film profile as the host holds it (the parsed tables and the pointers into them), the output encodings and the
// shared pipeline's configuration built from a profile. Every pixel stage lives once, in color_pipeline.h (namespace
// negicc); these are what the host needs around it.
// --------------------------------------------------------------------------------------------------------------------

using Rgb = negicc::Rgb;
using Lab = negicc::Lab;

// neg_process's output encodings: the two the shared pipeline encodes, and two that bypass its encoding stage.
enum OutputEncoding : int {
  ENCODE_SRGB = negicc::ENCODE_SRGB,                // sRGB transfer curve, with the highlight shoulder
  ENCODE_SRGB_LINEAR = negicc::ENCODE_SRGB_LINEAR,  // linear-light sRGB primaries
  ENCODE_CUSTOM = 2,                                // PCS XYZ through Little-CMS into an output ICC profile
  ENCODE_NONE = 3,                                  // the corrected camera RGB, no film profile applied
  ENCODE_DENSITY = 4                                // density above the film base as code values (np_density.h)
};

// The film profile's AToB0 pipeline (negicc::ProfileData) with host defaults: no profile, identity matrix.
struct ProfileData : negicc::ProfileData {
  ProfileData()
      : negicc::ProfileData{0, {nullptr, nullptr, nullptr}, {0, 0, 0}, {nullptr, nullptr, nullptr}, {0, 0, 0}, 0,
                            {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}, nullptr, {0, 0, 0}} {}
};

// A parsed film profile: the tables ProfileData points into.
struct ParsedFilmProfile {
  ProfileData data;
  std::vector<float> in_trc[3];
  std::vector<float> out_trc[3];
  std::vector<float> clut;

  // Points data at this object's tables (after a copy or a move).
  void rebind_pointers() {
    for (int c = 0; c < 3; ++c) {
      data.in_trc[c] = in_trc[c].empty() ? nullptr : in_trc[c].data();
      data.out_trc[c] = out_trc[c].empty() ? nullptr : out_trc[c].data();
    }
    data.clut = clut.empty() ? nullptr : clut.data();
  }
};

// The shared pipeline's configuration for a profile, knee and shoulder. Built once per conversion or search
// iteration, never per pixel: set_shoulder() solves the shoulder's knee.
inline negicc::PipelineConfig make_pipeline_config(const ProfileData& prof, float knee, float shoulder_x_hi = 1.0f,
                                                   int encoding = ENCODE_SRGB) {
  negicc::PipelineConfig cfg{};
  cfg.profile = prof;
  cfg.knee = knee;
  cfg.encoding = encoding;
  negicc::set_shoulder(cfg, shoulder_x_hi);
  return cfg;
}

// --------------------------------------------------------------------------------------------------------------------
// The profile bundle: a film stock's brackets, each with its film profile.
// --------------------------------------------------------------------------------------------------------------------

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
  // The chart's grey steps gs0, gs1, ... as the profile read them (R, G, B after the crosstalk correction, at this
  // target's exposure), and the position of the mid grey gs10 among them (-1 without one): what a print's slopes are
  // fitted on (np_density.h density_frame_print). Empty when a step lacks a channel.
  std::vector<std::array<double, 3>> grey_rgb;
  int grey_mid = -1;
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
// A density measure a scan's densities are written in (-P): Status M, SMPTE RP 180 printing density (the Cineon
// convention) or Academy Printing Density (SMPTE ST 2065-2, which ADX encodes).
enum DensityMeasure { kDensityStatusM = 0, kDensityRp180 = 1, kDensityApd = 2, kDensityMeasures = 3 };
// The -P value ("statusm", "rp180", "apd") and the bundle's density_matrices key of a measure.
const char* density_measure_name(DensityMeasure m);
const char* density_measure_key(DensityMeasure m);

// A bundle's matrix for one density measure: D_measure = m * D_light, both above the film base, rows R, G, B.
struct DensityMatrix {
  bool present = false;
  double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::string target_curves;          // what defines the measure, as the bundle names it
};

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
  // The density export (-P statusm, rp180, apd): the 3x3 matrices that turn the scanning light's channel densities
  // above the film base into a standard density measure, fitted when the bundle was built from the light's spectrum
  // (density_matrices), and the light's record with that spectrum (light_source), kept as the bundle writes it for
  // the output's metadata. A bundle from before the fields has none.
  DensityMatrix density_matrix[kDensityMeasures];
  std::string light_source_json;      // the "light_source" object, compact; empty when the bundle has none
  std::string density_fit_json;       // "density_matrices" without the matrices (the dye spectra and the fit's record)
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

// --------------------------------------------------------------------------------------------------------------------
// Profiles
// neg_process's profiles: resolving a profile path, and loading an ICC film profile, or a profile bundle JSON.
// --------------------------------------------------------------------------------------------------------------------

bool is_directory(const std::string& path);
bool is_regular_file(const std::string& path);
// The path of a profile argument: as given, else under profiles/ (and a few historical locations).
std::string resolve_profile_path(const std::string& input_path);
// Loads an ICC film profile (-p with a single ICC profile, for research).
bool load_film_profile(const std::string& raw_path, ParsedFilmProfile& prof);
// Loads the negicc-station profile bundle JSON a -p argument names (as given or under profiles/), or the bundle.json
// of a split-bundle directory (profiles/<Key>). False when the argument is not a bundle (an ICC profile).
bool load_bundle_arg(const std::string& arg, ProfileBundle& bundle);
// Reads an ICC profile file into a malloc'ed buffer (*prof_out, *size): 0 on success, -1 otherwise.
int read_profile(const std::string& prof_name, unsigned** prof_out, unsigned* size);

// --------------------------------------------------------------------------------------------------------------------
// Input matrix and shoulder
// neg_process's conversion: the input matrix (crosstalk correction, film base scale, exposure and gains) and the
// highlight shoulder's record.
// --------------------------------------------------------------------------------------------------------------------

// The soft knee of the highlight compression: fixed, as the station's Auto (multi_target_optimizer.cc AUTO_KNEE).
constexpr float kKnee = 0.95f;

// The highlight roll-off (color_pipeline.h's shoulder) as the sidecar records it (tone.shoulder*): measured for every
// bundle conversion (multi_target_optimizer.h measure_rolloff_x_hi), off with an ICC profile. Only the sRGB output
// applies it.
struct ShoulderInfo {
  float x_hi = 1.0f;                 // 1 is off
  std::string source = "off";        // off | rolloff
  bool has_q995 = false;
  float q995 = 0.0f;                 // the frame's 99.5% point of the pre-shoulder max channel
  bool has_knee = false;
  float knee = 1.0f;                 // the linear knee x_k
  bool active() const { return x_hi > negicc::kShoulderOff; }
};

// The input matrix: the crosstalk rows r_coef, g_coef and b_coef, each scaled by its channel's film base ratio
// (profile_film_base_rgb / film_base_rgb) and by global_exposure_comp, gain_g and gain_b as negicc-station applies
// them, into merged_matrix (3x3, row-major). Prints the rows.
void adjust_correction_matrix(const std::vector<float>& r_coef, const std::vector<float>& g_coef,
                              const std::vector<float>& b_coef, float global_exposure_comp, float gain_g,
                              float gain_b, const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb, float* merged_matrix);

#endif  // PROFILE_BUNDLE_H
