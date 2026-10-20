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

#ifndef FRAME_CROP_H
#define FRAME_CROP_H

#include <cstdint>
#include <string>
#include <vector>

// Cropping for the parameter search (param_search.h): the half-size analysis decode, and the frame and its inner
// crop cut out of it in the orientation of the output.

// Interleaved 16-bit camera RGB.
struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint16_t> rgb;
};

// The half-size analysis decode of a RAW file: its black-subtracted Bayer mosaic with every 2x2 quad binned into
// one pixel (R, the mean of the greens, B). shutter_s and iso are the capture's.
// py-ref: negicc_station/src/raw_processor.cpp @ 93be1ec (load_raw_mosaic)
// py-ref: negicc_station/src/pipeline/kernels_cpu.cpp @ 93be1ec (cpu_preview_debayer)
bool load_half_size_mosaic(const std::string& path, RgbImage& out, double& shutter_s, double& iso);

// Moves rect (x, y, w, h) inside bw x bh, shrinking it only when it is larger.
// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
void clamp_rect(int rect[4], int bw, int bh);

// Crops (x, y, w, h) out of src, rotates it clockwise by rot_cw and flips it. The search renders its previews in
// the output's orientation (--rot, --hflip, --vflip), as the station's views do, because DINOv3 judges an image
// the way it will be seen; the statistics are taken on the same pixels.
// py-ref: negicc_station/src/image_capture.cpp @ 93be1ec (CapturedImage::view)
void cut_and_orient(const RgbImage& src, int x, int y, int w, int h, int rot_cw, bool hflip, bool vflip,
                    RgbImage& out);

// The centred inner crop of a w x h image: crop_ratio of each side, the margins rounded to whole pixels.
struct InnerCrop {
  int x0 = 0, y0 = 0, w = 0, h = 0;
};
InnerCrop compute_inner_crop(int w, int h, float crop_ratio);

#endif  // FRAME_CROP_H
