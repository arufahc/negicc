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

// The film base of a scan read from the scan itself, for Auto when no film base is given. neg_process and
// negicc-station both call detect_film_base, so the two pick the same base.
//
//  1. rebate: the frame detector (film_frame_detector.h, run on the RAW's mosaic) finds a flat patch of unexposed film
//     next to the frame (an inter-frame gap or the rebate) and validates it (the spread of its green density). The
//     base is the median per channel of the half-size decode over the inner part of that patch. The base is the
//     thinnest film on the scan, so a patch the scan's thinnest film (2.) outshines by more than kRebateMargin stops in
//     a channel is picture, not base, and is not taken.
//  2. statistics: else the thinnest film on the scan: the 99.5th percentile per channel of the film pixels of the
//     half-size decode reduced to kBaseWorkSize pixels, with bare light (clipped, or bright and less orange than the
//     film) and black holder masked out.
//
// The arithmetic follows negicc-station's auto_crop.py (downsample, classify) and the film_base_detect.py this replaced,
// as numpy 2 and OpenCV compute them, float32 where they are float32: on 59 frames it gives the Python's values bit for
// bit, so the base is the one Iteration 223 was evaluated with.
// The base is in the conversion's film base space: the decode through the crosstalk matrix (none for a white-light
// decode, already linear Rec.2020). The scan is its own base capture (the same exposure).

#pragma once

#include <string>

#include "cfa_pipeline.h"
#include "film_frame_detector.h"

namespace negicc {

constexpr double kBasePatchInset = 0.15;  // fraction of the patch's box dropped on each side (its edges blend)
constexpr int kBaseMinSamples = 32;       // pixels the inner box must hold
constexpr double kRebateMargin = 0.5;     // stops: the thinnest film may exceed a base patch by at most this
constexpr int kBaseWorkSize = 512;        // long side (px) of the image the thinnest film is read on

struct FilmBaseDetection {
  bool found = false;
  std::string source;                 // "rebate" or "statistics" when found
  double rgb[3] = {0.0, 0.0, 0.0};    // the film base: decode_rgb through the crosstalk matrix
  double decode_rgb[3] = {0.0, 0.0, 0.0};  // in the half-size decode's units
  float thinnest[3] = {0.0f, 0.0f, 0.0f};  // the thinnest film (2.), in the decode's units
  bool has_box = false;               // source rebate: the sampled patch in the half-size decode (x, y, w, h)
  int box[4] = {0, 0, 0, 0};
  // The frame detector's record (when it ran): its reason, its patch's green density spread, its frame confidence,
  // whether the bare light is clipped, and how far the thinnest film outshines the patch (stops, max over channels).
  bool has_detector = false;
  std::string reason;
  float std_d = 0.0f;
  float frame_conf = 0.0f;
  bool light_clipped = false;
  bool has_outshone = false;
  double thinnest_over_patch_stops = 0.0;
};

// The thinnest film on the half-size decode |half| (step 2. above), in its units.
void thinnest_film_base(const RgbView& half, float out[3]);

// The film base of the scan whose half-size decode (flat field applied) is |half|. |det| is the frame detector's
// result on the RAW's mosaic of sensor_w x sensor_h photosites (null when there is none: a white-light decode, an RGB
// file); |crosstalk| the 3x3 row-major crosstalk matrix (null for none). False (out.found false) when neither step gives
// a positive base; out still carries the detector's record.
bool detect_film_base(const RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                      const double* crosstalk, FilmBaseDetection& out);

}  // namespace negicc
