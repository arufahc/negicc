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

// Non-RAW inputs of neg_process: TIFF (libtiff) and JPEG (libjpeg-turbo) files read into 16-bit samples, and their
// conversion to the linear Rec.2020 a rec2020_linear (d65) bundle converts (Little-CMS).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace negicc {

// A decoded TIFF or JPEG: width x height pixels of `channels` interleaved samples (1, grey, or 3, RGB; a TIFF's
// extra samples such as alpha are dropped), scaled to 0..65535 (8-bit samples times 257).
struct RgbImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  int bits = 0;                  // bits per sample in the file: 8 or 16
  std::vector<uint16_t> pixels;  // width * height * channels
  std::vector<uint8_t> icc;      // the file's embedded ICC profile, empty when it has none
};

// True when the file starts with a JPEG signature.
bool is_jpeg_file(const std::string& path);

// Reads an 8- or 16-bit unsigned-integer TIFF (strips or tiles, chunky or planar, any codec libtiff is built with:
// none, LZW, Deflate, PackBits, JPEG, ...), RGB or grey (min-is-black). False with err set otherwise.
bool read_tiff(const std::string& path, RgbImage& out, std::string& err);

// Reads a baseline or progressive JPEG as RGB (a grey JPEG is expanded to RGB), with its ICC profile (APP2).
bool read_jpeg(const std::string& path, RgbImage& out, std::string& err);

// Converts an RGB image in place to linear Rec.2020 (D65 white), relative colorimetric, through its embedded ICC
// profile; without one, 8-bit data is taken as sRGB and 16-bit data as already linear Rec.2020 (left as it is).
// Returns how the image was taken ("icc", "srgb" or "linear") in `source`.
bool to_linear_rec2020(RgbImage& img, std::string& source, std::string& err);

}  // namespace negicc
