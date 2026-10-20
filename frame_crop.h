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

// Cropping for the automatic conversion: the half-size analysis decode, the film's bounding box and frame
// format, and oriented crops of it.

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

// The lit area of a half-size decode (x, y, w, h): the bounding rectangle of the largest region with green above
// 500 (the carrier blocks the light), or the whole image when there is none.
// py-ref: negicc_station/src/target_selection.py @ 93be1ec (detect_film_bbox)
void detect_film_bbox(const RgbImage& half, int bbox[4]);

// The film frame of a format inside the lit area: the largest centred rectangle of the format's aspect ("6x6",
// "35mm", "645", "6x7", "6x9"; "Free" or "" keeps the area). False for an unknown format.
// py-ref: negicc_station/src/film_frame.py @ 93be1ec (detect_frame, fit_aspect)
bool fit_film_format(const int area[4], const std::string& format, int frame[4]);

// Moves rect (x, y, w, h) inside bw x bh, shrinking it only when it is larger.
// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
void clamp_rect(int rect[4], int bw, int bh);

// Crops (x, y, w, h) out of src, rotates it clockwise by rot_cw and flips it, as the station's views.
void cut_and_orient(const RgbImage& src, int x, int y, int w, int h, int rot_cw, bool hflip, bool vflip,
                    RgbImage& out);

#endif  // FRAME_CROP_H
