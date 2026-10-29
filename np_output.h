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

// Writes <output_path>.json: what neg_process read and did (schema negicc.neg_process.sidecar/1).
bool write_json_sidecar(
    const std::string& output_path,
    int ac, char* av[],
    const LibRaw* proc,
    const std::vector<std::string>& files,
    int orig_w, int orig_h,
    bool half_size,
    bool has_roi, const int* roi_coords,
    const std::string& roi_source,
    const FrameDetection* frame_det,
    int rot_cw, bool hflip, bool vflip,
    int out_w, int out_h,
    bool film_base_applied,
    const std::vector<int>& film_base_rgb,
    const std::vector<int>& profile_film_base_rgb,
    const std::string& film_base_source,
    float std_density,
    bool has_std_density,
    const std::string& film_profile_path,
    const float* merged_matrix,
    float exposure_comp, float gain_g, float gain_b,
    unsigned pin_mask,
    float knee, const std::string& knee_source,
    const ShoulderInfo& shoulder,
    bool has_dino, const DinoSolveResult& dino_res,
    const std::string& dino_model_path,
    const std::string& dino_backend,
    const std::string& colorspace_name,
    double t_raw_decode_ms,
    double t_frame_det_ms,
    double t_full_conversion_ms,
    double t_tiff_write_ms,
    bool is_bundle = false,
    const std::string& bundle_target_name = "",
    int bundle_target_idx = -1,
    const ParamSearchResult* auto_res = nullptr,
    const int* auto_frame_half = nullptr,
    double t_auto_ms = 0.0,
    InputSpace input_space = InputSpace::kCameraRawRgb);
