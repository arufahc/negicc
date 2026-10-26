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

#ifndef COLORSPACE_CONVERSION_H
#define COLORSPACE_CONVERSION_H

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "neg_pipeline.h"

// Color space conversions the parameter searches score renders with, on top of the pixel pipeline of
// neg_pipeline.h.

// A camera pixel through the pipeline (matrix, knee, film profile) to linear sRGB D65 in [0, 1], before the output
// gamma encoding.
// py-ref: negicc_station/src/pipeline/color_pipeline.h @ 93be1ec (convert_pixel_linear)
static inline Rgb convert_pixel_linear(const uint16_t* in_px, const float* matrix, float knee,
                                       const ProfileData& prof) {
  Rgb v = load_u16(in_px);
  v = apply_matrix(matrix, v);
  v = apply_knee(v, knee);
  if (prof.has_profile) {
    v = apply_profile(prof, v);
  }
  return pcs_to_linear_srgb(v, prof.pcs_is_lab);
}

// Linear sRGB D65 to CIELAB D65 (OpenCV's constants, evaluated exactly).
static inline Lab linear_srgb_to_lab(Rgb lin) {
  const float r = lin.r, g = lin.g, b = lin.b;
  const float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
  const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
  const float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
  const float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
  const float L = (Y > 0.008856f) ? (116.0f * fy - 16.0f) : (903.3f * Y);
  return { L, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

// Linear sRGB D65 to CIE L*.
static inline float linear_srgb_to_L(Rgb lin) {
  const float Y = 0.212671f * lin.r + 0.715160f * lin.g + 0.072169f * lin.b;
  return (Y > 0.008856f) ? (116.0f * cbrtf(Y) - 16.0f) : (903.3f * Y);
}

// Whether a linear channel renders white-clipped: the linear equivalent of a 16-bit sRGB code value of 65400 or
// more, srgb_decode(65399.5 / 65535).
static inline bool linear_channel_white_clipped(float l) {
  return l >= 0.9953029f;
}

// ICC PCS L*a*b* (D50) to 8-bit sRGB code values in [0, 255], through Bradford D50 -> D65.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (_pcs_lab_to_srgb)
static inline void pcs_lab_to_srgb_u8(const float lab[3], float out[3]) {
  static const float kD50[3] = {0.9642f, 1.0f, 0.8249f};
  static const float kBradford[3][3] = {{0.9555766f, -0.0230393f, 0.0631636f},
                                        {-0.0282895f, 1.0099416f, 0.0210077f},
                                        {0.0122982f, -0.0204830f, 1.3299098f}};
  static const float kXyzToSrgb[3][3] = {{3.2406255f, -1.5372080f, -0.4986286f},
                                         {-0.9689307f, 1.8757561f, 0.0415175f},
                                         {0.0557101f, -0.2040211f, 1.0569959f}};
  const float fy = (lab[0] + 16.0f) / 116.0f;
  const float f[3] = {fy + lab[1] / 500.0f, fy, fy - lab[2] / 200.0f};
  float xyz[3], d65[3];
  for (int c = 0; c < 3; ++c) {
    const float f3 = f[c] * f[c] * f[c];
    xyz[c] = (f3 > 0.008856f ? f3 : (f[c] - 16.0f / 116.0f) / 7.787f) * kD50[c];
  }
  for (int r = 0; r < 3; ++r) d65[r] = kBradford[r][0] * xyz[0] + kBradford[r][1] * xyz[1] + kBradford[r][2] * xyz[2];
  for (int r = 0; r < 3; ++r) {
    const float lin = std::clamp(kXyzToSrgb[r][0] * d65[0] + kXyzToSrgb[r][1] * d65[1] + kXyzToSrgb[r][2] * d65[2],
                                 0.0f, 1.0f);
    const float enc = (lin <= 0.0031308f) ? lin * 12.92f : 1.055f * std::pow(lin, 1.0f / 2.4f) - 0.055f;
    out[r] = std::clamp(enc, 0.0f, 1.0f) * 255.0f;
  }
}

#endif  // COLORSPACE_CONVERSION_H
