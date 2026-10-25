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

#ifndef MULTI_TARGET_OPTIMIZER_H
#define MULTI_TARGET_OPTIMIZER_H

#include <chrono>
#include <string>
#include <vector>

#include "cfa_pipeline.h"
#include "dinov3_engine.h"
#include "profile_bundle.h"

// Interleaved 16-bit camera RGB: width x height pixels, row-major, rgb holding width * height * 3 values.
struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint16_t> rgb;
};

// The conversion parameter search of negicc-station (PerTargetConverter.convert with its defaults). It does not
// convert anything: on the half-size analysis decode of one frame it picks the calibration bracket of a profile
// bundle and the exposure E and gains g, b that neg_process then converts the full image with.
//
//   1. Enclosure: bound each bracket's E so the scene's green range lands inside the green range the bracket's
//      patches were calibrated over.
//   2. Bootstrap: meter each admissible bracket so the scene's log-average renders at L* 50, render the bracket
//      metered deepest inside its bounds and let DINOv3 predict an intent field from that render.
//   3. Screening: search (E, g, b) of every metered bracket (every admissible one when none meters) against that
//      one intent field.
//   4. Tournament: run the fixed-point loop on the best two screened brackets and keep the lower residual (among
//      the converged ones, when any converged).
//   5. Highlight headroom and rescue: raise E until the highlights are inside the bracket's calibrated range, and
//      move to the next bracket when the chosen one clips clearly more.
//
// The intent field is DINOv3's prediction, from an 8-bit sRGB render of the frame, of how each patch of the image
// should look: a mean mu and a spread sigma of CIELAB per patch of its grid. How far a render is from a field is
// the mean squared z-score over the grid,
//
//   z2(render, field) = 1 / (3 N) * sum_i sum_{k in L, a, b} ((lab_i,k - mu_i,k) / sigma_i,k)^2,
//
// with lab_i the CIELAB of the render over patch i and N the number of patches.
//
// The station's Python implementation stays the reference for research; this port follows it step by step (the
// py-ref comments name the functions) so both pick the same bracket and parameters.
// py-ref: negicc_station/src/per_target_conversion.py @ 93be1ec (PerTargetConverter.convert)

// The parameters the search solves. With the input matrix M of a bracket (rows R, G, B; see
// ParamSearchResult::input_matrix) the conversion matrix is
//
//   diag(E, E g, E b) * M,
//
// so E is the global gain on the linear camera RGB after the crosstalk correction and the film base
// normalisation, ahead of the film profile, and g and b are the green and blue gains relative to red.
struct ConversionParams {
  float e = 1.0f, g = 1.0f, b = 1.0f;
};

// How the fixed-point loop of a bracket stopped.
enum class FixedPointStatus {
  kConverged,  // E, g, b moved by 0.02 or less in total, or the residual by 1.5 % or less, in the last step
  kDiverged,   // the residual grew by more than 25 % and more than 1.5 in one step
  kMaxIters,   // neither within ParamSearchOptions::max_steps steps
};
const char* to_string(FixedPointStatus status);

// What the highlight headroom did with the chosen E.
enum class HeadroomOutcome {
  kWithinRange,      // the highlights were already inside the bracket's calibrated range: E unchanged
  kNoCapacity,       // E is already at the bracket's upper bound: E unchanged
  kCapacityLimited,  // E raised to the bracket's upper bound, the highlights still exceed the range
  kPulledInside,     // E raised just enough to bring the highlights inside the range
};
const char* to_string(HeadroomOutcome outcome);
inline bool headroom_applied(HeadroomOutcome outcome) {
  return outcome == HeadroomOutcome::kCapacityLimited || outcome == HeadroomOutcome::kPulledInside;
}

// What the highlight rescue did. It is tried when the chosen bracket white-clips more than 0.03 % of the inner
// crop, on the next bracket of the bundle (target_idx + 1), and taken when that one passes every check.
enum class RescueOutcome {
  kNotNeeded,        // the chosen bracket does not clip
  kNotScreened,      // the next bracket was not screened
  kNotAdmissible,    // the next bracket's calibrated range is too narrow for the scene
  kClipsTooMuch,     // the next bracket clips more than half as much as the chosen one
  kShiftsLightness,  // the next bracket moves the median L* outside -4.5 .. +0.5
  kWorseResidual,    // the next bracket's residual is more than 10 % above the chosen one's
  kTaken,            // the next bracket replaces the chosen one
};
const char* to_string(RescueOutcome outcome);

struct ParamSearchOptions {
  float crop_ratio = 0.90f;  // the frame's centred inner crop the statistics and the previews use
  int max_steps = 8;         // the most steps of each bracket's fixed-point loop (neg_process's --dino_iters)
  // A film base measured on this capture (crosstalk-corrected RGB) instead of the bundle's; null for the bundle's.
  const float* film_base_rgb = nullptr;
  const double* film_base_rgb_d = nullptr;
  // True when the custom film base was sampled from the scan itself (e.g. rebate), so exposure_ratio is 1.0.
  bool film_base_from_scan = false;
  bool verbose = true;
};

// Step 3, one row per metered bracket: one coordinate descent of (E, g, b) against the bootstrap intent field.
// Every bracket is scored against the same field, from one DINOv3 inference, so the rows compare the brackets
// cheaply; the best two go on to the tournament.
struct ScreeningResult {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  ConversionParams params;  // the best (E, g, b), E in reference units (see ParamSearchResult::e_norm)
  // z2 of the render at params against the bootstrap field, plus the search's penalties on clipping and on E far
  // from the bracket's centre exposure.
  float intent_loss = 0;
  float e_lo = 0, e_hi = 0;  // the bounds E was searched within (reference units)
  float e_metered = 0;       // E that renders the scene's log-average at L* 50 (reference units)
};

// Step 4, one row per shortlisted bracket: its fixed-point loop. Each step renders the frame at the current
// (E, g, b), runs DINOv3 on the render (one inference), measures the residual and searches the next (E, g, b)
// against that render's own intent field. The row keeps the step with the lowest residual.
struct FixedPointResult {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  ConversionParams params;  // (E, g, b) of the best step, E in reference units
  // The self-consistency residual of that step: z2 of the render against the field DINOv3 predicts from the same
  // render. Low when the render already looks the way DINOv3 expects the image to look.
  float residual = 0;
  FixedPointStatus status = FixedPointStatus::kMaxIters;
  int steps = 0;  // render -> DINOv3 -> search steps run, one inference each
};

struct ParamSearchResult {
  int target_idx = -1;      // the chosen bracket, an index into ProfileBundle::targets
  std::string target_name;  // its name in the bundle ("Target 3")
  ConversionParams params;  // the chosen (E, g, b), E in this frame's units: diag(E, E g, E b) * M_frame
  float residual = 0.0f;    // the chosen bracket's FixedPointResult::residual
  FixedPointStatus status = FixedPointStatus::kMaxIters;  // and its FixedPointResult::status

  // The bracket's input matrix and the frame's exposure. With X the bundle's crosstalk matrix, f the film base
  // RGB, n the bundle's normalisation target and rho the exposure of the film base capture over this frame's
  // (shutter x ISO / 100):
  //
  //   M_frame[r][c] = X[r][c] * n / f[r] * rho            (the merged input matrix of this frame)
  //   e_norm        = M_frame[1][1] / M_ref[1][1]         (M_ref: the same for a 1/8 s, ISO 100 scan)
  //   input_matrix  = M_frame / e_norm                    (the matrix the search runs on, in reference units)
  //   row_scale[r]  = n / f[r] * rho / e_norm,  so  input_matrix[r][c] = row_scale[r] * X[r][c]
  //
  // The search's E is in reference units, so the same E means the same rendering whatever the scan's exposure;
  // params.e is E / e_norm. row_scale lets the result be written back as the bundle's crosstalk with -E,
  // --gain_g and --gain_b (neg_process's cli_equivalent).
  float input_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float e_norm = 1.0f;
  float row_scale[3] = {1.0f, 1.0f, 1.0f};

  // The scene's green range over the inner crop (its 0.5 % and 99.5 % points, 16-bit) and its span in stops.
  float g_lo = 0, g_hi = 0, scene_ev = 0;

  HeadroomOutcome headroom = HeadroomOutcome::kWithinRange;
  float headroom_ev = 0.0f;  // the stops E was raised by
  RescueOutcome rescue = RescueOutcome::kNotNeeded;
  float rescue_white_clip = 0.0f;  // the fraction of the inner crop the first choice white-clips

  std::vector<ScreeningResult> screening;
  std::vector<FixedPointResult> tournament;
  int inferences = 0;  // DINOv3 inferences of the whole search: the bootstrap plus every fixed-point step
  // Time spent rendering previews, in DINOv3, in the (E, g, b) searches, on residuals, and in the whole search.
  std::chrono::steady_clock::duration render_time{}, infer_time{}, search_time{}, residual_time{}, total_time{};
};

// Searches the conversion parameters of one frame. frame is the frame upright, already cut out of the half-size
// decode and oriented as the output (DINOv3 judges an image the way it will be seen); scan_shutter_s and scan_iso
// are the capture's exposure.
bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                              double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out);

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& frame,
                              float scan_shutter_s, float scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out);

#endif  // MULTI_TARGET_OPTIMIZER_H
