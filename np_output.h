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
#include "film_frame_detector.h"
#include "libraw/libraw.h"
#include "multi_target_optimizer.h"
#include "np_conversion.h"
#include "profile_bundle.h"

// Writes proc's image (or rgb3, 3 samples a pixel, in its place) as a 16-bit TIFF to output with the ICC profile
// attach_profile names ("srgb", "srgb-g10" or a file) or, when that is empty, embedded_icc. 0 on success.
int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc = {}, const uint16_t* rgb3 = nullptr);

// What the sidecar records.
struct Sidecar {
  std::string output_path;               // the TIFF; the sidecar is <output_path>.json
  int argc = 0;
  char** argv = nullptr;
  const LibRaw* proc = nullptr;          // the camera, ISO and shutter (nullptr: none recorded)
  std::vector<std::string> files;
  int orig_w = 0, orig_h = 0;
  bool half_size = false;
  bool has_roi = false;
  const int* roi = nullptr;              // x, y, w, h when has_roi
  std::string roi_source;
  const FrameDetection* frame_det = nullptr;  // --roi auto's detection
  int rot_cw = 0;
  bool hflip = false, vflip = false;
  int out_w = 0, out_h = 0;
  bool film_base_applied = false;
  std::vector<int> film_base_rgb, profile_film_base_rgb;
  std::string film_base_source;
  float std_density = 0.0f;
  bool has_std_density = false;
  std::string profile_path;              // empty: no profile
  bool is_bundle = false;
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  InputSpace input_space = InputSpace::kCameraRawRgb;
  const float* merged_matrix = nullptr;
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;
  unsigned pin_mask = 0;
  float knee = 0.95f;
  std::string knee_source;
  ShoulderInfo shoulder;
  const DinoSolveResult* dino = nullptr; // the single-profile search, when it ran
  std::string dino_model, dino_backend;
  const ParamSearchResult* auto_res = nullptr;  // the bundle search, when it ran
  const int* auto_frame_half = nullptr;
  std::string colorspace;
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;
  double t_auto_ms = 0.0;
};

// Writes <output_path>.json: what neg_process read and did (schema negicc.neg_process.sidecar/1).
bool write_json_sidecar(const Sidecar& sc);
