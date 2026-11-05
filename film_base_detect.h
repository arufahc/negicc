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
constexpr int kBaseMinSide = 16;          // px: the inner box's width and height must each reach this
constexpr float kBaseMinFrameConf = 0.5f; // the detection's confidence below which its patch is not used (the frame
                                          // detection neg_process falls back from at the same level)
// The plausibility of a reading (white light, with the base the bundle expects):
//  - the rebate patch and the thinnest film agree when every channel is within kBaseAgreeStops of the other's: then the
//    rebate is taken as it is (two independent readings of the same base);
//  - otherwise a reading is taken only inside the stock's ratio envelope: its R/G between 1 and kBaseRgMax times the
//    expected R/G, its B/G between kBaseBgMin times the expected B/G and the expected B/G. Base is never less orange
//    than the stock's, and a mask a little denser than the bundle's is still base; sky and bare light (neutral) and
//    picture fall outside.
constexpr double kBaseAgreeStops = 0.35;
constexpr double kBaseRgMax = 1.5;
constexpr double kBaseBgMin = 0.67;

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
  // The plausibility gate (with an expected base): whether the two readings agree (and by how much, stops, the
  // largest channel), each reading's R/G and B/G over the expected base's, and which readings it refused.
  bool has_chroma_check = false;
  bool has_agreement = false;
  bool readings_agree = false;
  double agreement_stops = 0.0;
  double rebate_rg = 0.0, rebate_bg = 0.0, statistics_rg = 0.0, statistics_bg = 0.0;
  bool rebate_refused = false, statistics_refused = false;
};

// The thinnest film on the half-size decode |half| (step 2. above), in its units.
void thinnest_film_base(const RgbView& half, float out[3]);

// The film base of the scan whose half-size decode (flat field applied) is |half|. |det| is the frame detector's
// result on the RAW's mosaic of sensor_w x sensor_h photosites (the CFA path's, or the white-light decode's run before
// it decodes; null when there is none: an RGB file, three captures, a linear DNG); its patch is used only when the
// detection succeeded with confidence at least kBaseMinFrameConf. |crosstalk| the 3x3 row-major crosstalk matrix (null
// for none). False (out.found false) when no step gives a positive, plausible base; out still carries the detector's
// record. |expected_rgb|: the base the stock's profile expects, in the same space (a white-light bundle's
// film_base_avg), or null for no plausibility gate (a crosstalk matrix, three captures, a custom illuminant): see
// kBaseAgreeStops. The rebate is preferred; a refused rebate falls back to the statistics, refused statistics to none,
// so the caller's base.
bool detect_film_base(const RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                      const double* crosstalk, FilmBaseDetection& out, const float* expected_rgb = nullptr);

// rgb's R/G and B/G over expected's (0 when either is not positive).
void film_base_ratios(const double rgb[3], const float expected[3], double& rg, double& bg);
// Whether those ratios lie in the stock's envelope (kBaseRgMax, kBaseBgMin).
bool film_base_in_envelope(double rg, double bg);

}  // namespace negicc
