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

// neg_process's conversion: the input matrix (crosstalk correction, film base scale, exposure and gains) and the
// highlight shoulder's record.

#pragma once

#include <string>
#include <vector>

#include "color_pipeline.h"

// The highlight shoulder as the sidecar records it (tone.shoulder*).
struct ShoulderInfo {
  float x_hi = 1.0f;                 // the frozen parameter; 1 is off
  std::string source = "off";        // off | cli | auto
  bool has_q995 = false;
  float q995 = 0.0f;                 // measured before clamping (auto)
  bool has_knee = false;
  float knee = 1.0f;                 // the linear knee x_k
  bool has_guard = false;
  bool guard = false;
  bool active() const { return x_hi > negicc::kShoulderOff; }
};

// The input matrix: the crosstalk rows r_coef, g_coef and b_coef, each scaled by its channel's film base ratio
// (profile_film_base_rgb / film_base_rgb) and by global_exposure_comp, gain_g and gain_b as negicc-station applies
// them, into merged_matrix (3x3, row-major). Prints the rows.
void adjust_correction_matrix(const std::vector<float>& r_coef, const std::vector<float>& g_coef,
                              const std::vector<float>& b_coef, float global_exposure_comp, float gain_g,
                              float gain_b, const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb, float* merged_matrix);
