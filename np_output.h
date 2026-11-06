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

// neg_process's output: the 16-bit TIFF with its ICC profile, and the JSON sidecar beside it.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dinov3_auto_solver.h"
#include "film_base_detect.h"
#include "film_frame_detector.h"
#include "libraw/libraw.h"
#include "multi_target_optimizer.h"
#include "np_conversion.h"
#include "profile_bundle.h"
#include "white_film.h"

// Writes proc's image (or rgb3, 3 samples a pixel, in its place) as a 16-bit TIFF to output with the ICC profile
// attach_profile names ("srgb", "srgb-g10" or a file) or, when that is empty, embedded_icc. 0 on success.
int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc = {}, const uint16_t* rgb3 = nullptr);

// What the sidecar records.
// An input's size and CRC-32 (file_fingerprint), kept so a run hashes each input once: the sidecar loader's check and
// the sidecar writer share it.
struct FileFingerprint {
  bool ok = false;
  uintmax_t size = 0;
  uint32_t crc = 0;
};

struct Sidecar {
  std::string output_path;               // the TIFF; the sidecar is <output_path>.json
  int argc = 0;
  char** argv = nullptr;
  const LibRaw* proc = nullptr;          // the camera, ISO and shutter (nullptr: none recorded)
  std::vector<std::string> files;
  // Each file's fingerprint when this run already took it (the sidecar loader); null or short: taken here.
  const std::vector<FileFingerprint>* file_fingerprints = nullptr;
  int orig_w = 0, orig_h = 0;
  bool half_size = false;
  bool has_roi = false;
  const int* roi = nullptr;              // x, y, w, h when has_roi
  std::string roi_source;
  const FrameDetection* frame_det = nullptr;  // --roi auto's detection
  int rot_cw = 0;
  bool hflip = false, vflip = false;
  int out_w = 0, out_h = 0;
  // The film base: source "rebate" or "statistics" (read from the scan: film_base_detection), "bundle" or "none".
  bool film_base_applied = false;
  std::vector<double> film_base_rgb;
  std::vector<int> profile_film_base_rgb;
  std::string film_base_source;
  const negicc::FilmBaseDetection* film_base_detection = nullptr;
  std::string profile_path;              // empty: no profile
  bool is_bundle = false;
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  InputSpace input_space = InputSpace::kCameraRawRgb;
  const float* merged_matrix = nullptr;
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;  // absolute, on the crosstalk (matrix.*)
  const float* station_params = nullptr;  // with a bundle: E, g, b as -E, --gain_g, --gain_b take them
  unsigned pin_mask = 0;
  float knee = 0.95f;
  ShoulderInfo shoulder;
  const DinoSolveResult* dino = nullptr; // the single-profile search, when it ran
  std::string dino_model, dino_backend;
  const ParamSearchResult* auto_res = nullptr;  // the bundle search, when it ran
  const int* auto_frame_half = nullptr;
  std::string cached_auto_json_str;      // preserved "auto" JSON when reusing a sidecar
  std::string cached_dino_json_str;      // preserved "dino" JSON when reusing a sidecar
  std::string cached_intent_field_json_str;  // preserved "intent_field" JSON when reusing a sidecar
  std::string cached_input_curves_json_str;  // preserved "input_curves" JSON when reusing a sidecar
  double cached_dino_views_ms = 0.0, cached_dino_render_ms = 0.0;
  double cached_dino_infer_ms = 0.0, cached_dino_search_ms = 0.0;
  bool has_knobs = false;
  float brightness = 0.0f, tint = 0.0f, warmth = 0.0f;
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  std::string colorspace;
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;
  double t_auto_ms = 0.0;
  double t_analysis_ms = 0.0;  // Scan::t_analysis_ms
  // Fitted G/B input curves of the conversion's bracket (input_curves.h): source "fit" (--fit_curves on, with
  // curve_fit), "auto" (likewise) or "cli" (recorded curves given to --fit_curves); empty: none.
  std::string input_curves_source;
  const negicc::InputCurves* input_curves = nullptr;
  const CurveRefitResult* curve_fit = nullptr;
  // The curves the cli_equivalent replays when they differ from input_curves (re-anchored at the knobs' E, g and b);
  // null: input_curves.
  const negicc::InputCurves* cli_input_curves = nullptr;
  const WhiteFilm* white_film = nullptr;  // --white_model film: its conversion in place of the profile's
};

// Writes <output_path>.json: what neg_process read and did (schema negicc.neg_process.sidecar/1).
// The build's version (git describe), as the sidecar's tool.version records it.
const char* neg_process_version();

// A file's size and CRC-32 (zlib), as the sidecar records each input (inputs.raw_file_stats). False when unreadable.
bool file_fingerprint(const std::string& path, uintmax_t& size, uint32_t& crc);

bool write_json_sidecar(const Sidecar& sc);
