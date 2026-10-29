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

// neg_process's inputs besides one RAW: the input modes, and loading an RGB TIFF or JPEG or three captures, one per
// light, into a LibRaw image as load_raw leaves a decoded RAW.

#pragma once

#include <string>
#include <vector>

#include "film_frame_detector.h"
#include "libraw/libraw.h"
#include "np_raw.h"

// The inputs neg_process reads besides one RAW (or the four of a pixel shift).
enum class InputMode {
  kRaw,        // a CFA RAW or DNG (load_raw)
  kLinearDng,  // a linear (demosaiced) DNG, e.g. a film scanner's: load_raw too, white light (d65) only
  kRgbFile,    // an RGB TIFF or JPEG (rgb_input.h): linear Rec.2020 (d65), or a TIFF of linear camera RGB (tri)
  kThreeShot,  // three captures, one under each of the red, green and blue lights: RAWs, DNGs or TIFFs (tri)
};

// True when LibRaw opens the file (a RAW or a DNG; the header only).
bool libraw_opens(const std::string& path);
// True when the file is a DNG with no CFA (its pixels already demosaiced; the header only).
bool is_linear_dng(const std::string& path);

// The non-RAW inputs (InputMode): an RGB TIFF or JPEG through its ICC profile to linear Rec.2020 (rgb_input.h), or
// with |camera_rgb| a TIFF's values as the linear camera RGB a tri bundle converts, or three captures, each giving the channel of its light: R of the first, G of the second, B of the third, in camera
// units (a RAW or DNG decoded to camera RGB as load_raw decodes it, a TIFF's samples as they are; a grey TIFF gives
// its one channel), each scaled to the first capture's exposure when all three record theirs. Returns a LibRaw holding
// the image as load_raw returns a decoded RAW, after cropping it to |roi| or, with |frame_det|, to the frame the
// detector finds on the image (as --roi auto), and running |analyse| on its half-size copy, as load_raw does.
LibRaw* load_rgb_input(const std::vector<std::string>& files, InputMode mode, bool camera_rgb, int qual,
                              bool crop, const int* roi,
                              const HalfSizeAnalysis& analyse, FrameDetection* frame_det,
                              const FrameDetectionParams& det_params, std::string& source);
