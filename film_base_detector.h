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

// The film base of a scan read from the scan itself, for Auto when no film base is given. neg_process and
// negicc-station both call detect_film_base, so the two pick the same base.
//
//  1. rebate: the frame detector (film_frame_detector.h, run on the RAW's mosaic) finds a flat patch of unexposed film
//     next to the frame (an inter-frame gap or the rebate) and validates it (the spread of its green density). The
//     base is the median per channel of the half-size decode over the inner part of that patch. The base is the
//     thinnest film on the scan, so a patch the scan's thinnest film (2.) outshines by more than kRebateMargin stops in
//     a channel is picture, not base, and is not taken. Nor is a patch the sensor clipped (more than
//     kBaseClipFraction of its pixels at kSensorClip of the white level in a channel): clipped film reads the clip,
//     not the base.
//  2. statistics: else the thinnest film on the scan: the 99.5th percentile per channel of the film pixels of the
//     half-size decode reduced to kBaseWorkSize pixels, with bare light and black holder masked out. On a decode in
//     sensor units (TriLED) bare light is what the sensor clipped in all three channels: light is brighter than the
//     base in every channel, while film near the clip clips first in its brightest channel, and the base is nearly
//     neutral in the camera's channels, so colour does not tell thin film from light. On a white-light decode
//     (Rec.2020, white-balanced) bare light is a part clipped in a channel (its maxima at a plateau) whose median blue
//     is above the film's 99.5th percentile of blue, with the bright unclipped light connected to it, or a neutral
//     part bluer than the orange film: film transmits no more blue than its base, so a flat stretch of film that
//     forms a plateau is not taken for light.
//
// The white-light arithmetic follows negicc-station's auto_crop.py (downsample, classify) and the film_base_detect.py
// this replaced, as numpy 2 and OpenCV compute them, float32 where they are float32.
// The base is in the conversion's film base space: the decode through the crosstalk matrix (none for a white-light
// decode, already linear Rec.2020). The scan is its own base capture (the same exposure).

#pragma once

#include <string>

#include "rgb_view.h"
#include "film_frame_detector.h"

namespace negicc {

constexpr double kBasePatchInset = 0.15;  // fraction of the patch's box dropped on each side (its edges blend)
constexpr int kBaseMinSamples = 32;       // pixels the inner box must hold
constexpr float kSensorClip = 0.98f;      // a sample at or above this fraction of the white level (before the flat
                                          // field, but after it on the LibRaw path with --flat) is clipped
constexpr double kBaseClipFraction = 0.001;  // a patch, tile or region with more clipped pixels is clipped
constexpr double kRebateMargin = 0.5;     // stops: the thinnest film may exceed a base patch by at most this
constexpr double kBasePercentile = 99.5;  // per-channel percentile of film pixels taken as the film base (the
                                          // statistics' step; also the print's floor on a base that is not a capture,
                                          // the rebate included: print_base_floor)
constexpr int kBaseWorkSize = 512;        // long side (px) of the image the thinnest film is read on
constexpr float kBaseMinFrameConf = 0.5f; // the detection's confidence below which its patch is not used (the frame
                                          // detection neg_process falls back from at the same level)

constexpr const char* kFilmBaseSourceRebate = "rebate";
constexpr const char* kFilmBaseSourceStatistics = "statistics";

struct FilmBaseDetection {
  bool found = false;
  std::string source;                 // kFilmBaseSourceRebate or kFilmBaseSourceStatistics when found
  bool is_rebate() const { return found && source == kFilmBaseSourceRebate; }
  double rgb[3] = {0.0, 0.0, 0.0};    // the film base: decode_rgb through the crosstalk matrix
  double decode_rgb[3] = {0.0, 0.0, 0.0};  // in the half-size decode's units
  float thinnest[3] = {0.0f, 0.0f, 0.0f};  // the thinnest film (2.), in the decode's units
  bool rebate_clipped = false;        // the frame detector's patch was refused because the sensor clipped it
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

// The thinnest film on the half-size decode |half| of a TriLED scan or capture (step 2. above), in its units. |white|
// is the clipping level of its samples before the flat field (black subtracted, in their units).
void thinnest_film_base(const RgbView& half, float white, float out[3]);

// The film base of the scan whose half-size decode (flat field applied) is |half|. |det| is the frame detector's
// result on the RAW's mosaic of sensor_w x sensor_h photosites (the CFA path's, or the white-light decode's run before
// it decodes, or a linear DNG's; null when there is none: an RGB file, three captures); its patch is used only when the
// detection succeeded with confidence at least kBaseMinFrameConf. |crosstalk| the 3x3 row-major crosstalk matrix (null
// for none: a white-light decode). |white| is the clipping level of half's samples before its flat field (black
// subtracted, in their units; unused on a white-light decode). False (out.found false) when neither step gives a
// positive base; out still carries the detector's record.
bool detect_film_base(const RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                      const double* crosstalk, float white, FilmBaseDetection& out);

}  // namespace negicc
