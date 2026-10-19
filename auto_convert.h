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

#ifndef AUTO_CONVERT_H
#define AUTO_CONVERT_H

#include <string>
#include <vector>

#include "dinov3_engine.h"
#include "frame_analysis.h"
#include "profile_bundle.h"

// The automatic conversion of negicc-station (PerTargetConverter.convert with its defaults: intent exposure,
// tournament bracket selection, metered bootstrap, trust region, capacity filter, compromise bounds, highlight
// headroom and highlight rescue), run on the half-size analysis decode of one frame. It picks the calibration
// bracket of a profile bundle and the exposure E and gains g, b the frame converts with:
//
//   1. bound each bracket's exposure from the scene's green range (gamut volume enclosure);
//   2. meter each admissible bracket to L* 50 and bootstrap a DINOv3 intent field from the deepest one's render;
//   3. screen the brackets against that field, then run the fixed-point loop (render -> DINOv3 -> search E, g, b)
//      on the best two and keep the lowest self-consistency residual;
//   4. pull the highlights inside the bracket's calibrated range, and move to the next bracket when that clips
//      clearly less.
//
// The station's Python implementation stays the reference for research; this port follows it step by step (the
// py-ref comments name the functions) so both pick the same bracket and parameters.
// py-ref: negicc_station/src/per_target_conversion.py @ 93be1ec (PerTargetConverter.convert)

struct AutoConvertOptions {
  float crop_ratio = 0.90f;  // inner crop of the frame the statistics and the previews use
  // A film base measured on this capture (crosstalk-corrected RGB) instead of the bundle's; null for the bundle's.
  const double* film_base_rgb = nullptr;
  bool verbose = true;
};

struct AutoScreenRow {
  int target_idx = 0;
  double e = 0, g = 0, b = 0;  // reference units (see AutoConvertResult::e_norm)
  double intent_loss = 0;
  double lo = 0, hi = 0, e_met = 0;
};

struct AutoFixedPointRow {
  int target_idx = 0;
  double e = 0, g = 0, b = 0;  // reference units
  double final_err = 0;
  std::string status;
  int iters = 0;
};

struct AutoConvertResult {
  int target_idx = -1;
  std::string target_name;
  double e = 1.0, g = 1.0, b = 1.0;  // E in this frame's units, as the merged matrix below takes it
  double residual = 0.0;
  std::string status;
  float matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // input matrix of the chosen bracket (crosstalk, film base and
                                                  // exposure ratio, in reference units); candidate_matrix(matrix,
                                                  // e * e_norm, g, b) is the final conversion matrix
  double e_norm = 1.0;  // this frame's green input scale over the reference scan's (1/8 s, ISO 100)
  int bbox[4] = {0, 0, 0, 0};
  double g_lo = 0, g_hi = 0, scene_ev = 0;
  bool headroom_applied = false;
  double headroom_ev = 0.0;
  std::string headroom_reason;
  std::string rescue_branch;  // "N": no rescue needed, "R": rescue evaluated
  double rescue_clip = 0.0;   // white-clipped fraction of the first choice
  bool rescue_changed = false;
  std::vector<AutoScreenRow> screening;
  std::vector<AutoFixedPointRow> tournament;
  int inferences = 0;
  double timing_render_ms = 0, timing_infer_ms = 0, timing_search_ms = 0, timing_residual_ms = 0;
  double timing_total_ms = 0;
};

// Runs the automatic conversion on the half-size decode of one frame. bbox is the film in half-size coordinates;
// rot_cw, hflip and vflip orient it. scan_shutter_s and scan_iso are the capture's exposure.
bool run_auto_convert(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& half, const int bbox[4],
                      int rot_cw, bool hflip, bool vflip, double scan_shutter_s, double scan_iso,
                      const AutoConvertOptions& opts, AutoConvertResult& out);

#endif  // AUTO_CONVERT_H
