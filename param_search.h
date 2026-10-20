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

#ifndef PARAM_SEARCH_H
#define PARAM_SEARCH_H

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "frame_crop.h"
#include "neg_pipeline.h"

// The measurements the parameter search makes on a frame: its statistics, previews, CIELAB and highlight
// clipping. The decisions made from them (auto_convert.h, and negicc-station's Python control loop) are separate.

// Returns linear sRGB D65 in [0, 1] before output gamma encoding (no post-correction gamma).
// py-ref: negicc_station/src/pipeline/color_pipeline.h @ 93be1ec (convert_pixel_linear)
static inline Rgb convert_pixel_linear(const uint16_t* in_px, const float* matrix, float knee,
                                       const ProfileData& prof) {
  Rgb v = load_u16(in_px);
  v = apply_matrix(matrix, v);
  v = apply_knee(v, knee);
  if (prof.has_profile) {
    v = apply_profile(prof, v);
  }
  return pcs_to_linear_srgb(v);
}

// Direct linear sRGB D65 -> CIELAB D65 (OpenCV's constants, evaluated exactly).
static inline Lab linear_srgb_to_lab(Rgb lin) {
  const float r = lin.r, g = lin.g, b = lin.b;
  const float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
  const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
  const float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
  const float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
  const float L = (Y > 0.008856f) ? (116.0f * fy - 16.0f) : (903.3f * Y);
  return { L, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

// Direct linear sRGB D65 -> CIE L*.
static inline float linear_srgb_to_L(Rgb lin) {
  const float Y = 0.212671f * lin.r + 0.715160f * lin.g + 0.072169f * lin.b;
  return (Y > 0.008856f) ? (116.0f * cbrtf(Y) - 16.0f) : (903.3f * Y);
}

// Linear-light equivalents of channel_white_clipped: srgb_decode(65399.5 / 65535).
static inline bool linear_channel_white_clipped(float l) {
  return l >= 0.9953029f;
}

// L* histogram of kLabBins bins over [0, 100].
constexpr int kLabBins = 1000;
static inline int lab_bin(float L) {
  int k = (int)(L * (kLabBins / 100.0f));
  return k < 0 ? 0 : (k > kLabBins - 1 ? kLabBins - 1 : k);
}

// Symmetric inner crop of a W x H image (crop_ratio of each side, centred).
struct InnerCrop {
  int x0 = 0, y0 = 0, w = 0, h = 0;
};

static inline InnerCrop compute_inner_crop(int W, int H, float crop_ratio) {
  if (W <= 0 || H <= 0) return {0, 0, 0, 0};
  if (crop_ratio >= 1.0f) return {0, 0, W, H};
  int x0 = (int)roundf(W * (1.0f - crop_ratio) * 0.5f);
  int y0 = (int)roundf(H * (1.0f - crop_ratio) * 0.5f);
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  const int max_x0 = (W - 1) / 2;
  const int max_y0 = (H - 1) / 2;
  if (x0 > max_x0) x0 = max_x0;
  if (y0 > max_y0) y0 = max_y0;
  return {x0, y0, W - 2 * x0, H - 2 * y0};
}

// Statistics of a frame's inner crop: the green 0.5 % / 99.5 % points, the log-average RGB and the crop averaged
// onto the DINOv3 patch grid (one pixel per patch).
// py-ref: negicc_station/src/image_capture.cpp @ 93be1ec (FrameView::raw_stats)
struct FrameStats {
  float g_lo = 0.0f, g_hi = 0.0f;
  double log_avg_rgb[3] = {0, 0, 0};  // geometric mean in [0, 1]
  int grid_w = 0, grid_h = 0;
  std::vector<uint16_t> grid;  // grid_w x grid_h RGB
};
void compute_frame_stats(const RgbImage& frame, float crop_ratio, int max_size, int patch_size, FrameStats& st);

// Renders img to 8-bit sRGB with one conversion matrix.
void render_srgb_u8(const RgbImage& img, const float* matrix, float knee, const ProfileData& prof, uint8_t* out);

// Per-channel 8-bit sRGB ceiling of a calibration bracket from its profcheck record: the brightest values its
// patches render to; 255 without a record.
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (profile_highlight_ceiling)
void profile_highlight_ceiling(const std::string& profcheck_output, double ceil[3]);

// Over crop of img, per candidate matrix (n * 9 floats): the % of pixels at or above ceil_rgb per channel, the %
// with any channel white-clipped and the median L*; out holds n * 5 floats.
// py-ref: negicc_station/src/pipeline/converter.cpp @ 93be1ec (Converter::highlight_clip)
// py-ref: negicc_station/src/pipeline/kernels_cpu.cpp @ 93be1ec (cpu_highlight_clip)
void highlight_clip(const RgbImage& img, const InnerCrop& crop, const float* matrices, int n, float knee,
                    const ProfileData& prof, const float ceil_rgb[3], float* out);

#endif  // PARAM_SEARCH_H
