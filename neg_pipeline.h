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

#ifndef NEGICC_NEG_PIPELINE_H
#define NEGICC_NEG_PIPELINE_H

// neg_process's host-side types around the shared per-pixel pipeline. Every pixel stage (input matrix, knee, film
// profile, PCS to sRGB, shoulder, encoding, Lab scoring) lives once, in color_pipeline.h (namespace negicc), which
// negicc-station and a CUDA backend compile as well; this header only adds what neg_process needs on the host.

#include <stdint.h>
#include <vector>

#include "color_pipeline.h"

using Rgb = negicc::Rgb;
using Lab = negicc::Lab;

// neg_process's output encodings: the two the shared pipeline encodes, and two that bypass its encoding stage.
enum OutputEncoding : int {
  ENCODE_SRGB = negicc::ENCODE_SRGB,                // sRGB transfer curve, with the highlight shoulder
  ENCODE_SRGB_LINEAR = negicc::ENCODE_SRGB_LINEAR,  // linear-light sRGB primaries
  ENCODE_CUSTOM = 2,                                // PCS XYZ through Little-CMS into an output ICC profile
  ENCODE_NONE = 3                                   // the corrected camera RGB, no film profile applied
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

#endif  // NEGICC_NEG_PIPELINE_H
