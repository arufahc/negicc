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
#include "input_curves.h"
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
//   5. The clip fallback: move to the next bracket when the chosen one clips clearly more (ClipFallbackOutcome).
//      (The highlight headroom that raised E first was removed, as negicc-station's deaf9cc removed it.)
//   6. The highlight pullback (formerly Part 1B, the enclosure-guided rescue): when a bright frame still has many
//      highlights near white (sRGB 242 and 235), pull E into the bracket's calibrated range or move to a darker
//      bracket whose range holds the frame.
//
// Every render carries the highlight roll-off (color_pipeline.h shoulder_x_hi_anchored): the fixed-point loop
// measures its x_hi at each iteration's incoming parameters, and the search measures it once more at its final
// parameters, after the clip fallback and the pullback have moved E (and possibly the bracket) past the loop that
// froze it, so the written image's shoulder matches its highlights. The intent field the result carries
// (ParamSearchResult::intent_mu) stays the one the chosen loop step predicted, at intent_params.
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
  // The highlight shoulder x_hi (color_pipeline.h) the render, the intent field and the loss use: per frame, frozen
  // by the fixed-point loop; 1 is off.
  float shoulder = 1.0f;
};

// How the fixed-point loop of a bracket stopped.
enum class FixedPointStatus {
  kConverged,  // E, g, b moved by 0.02 or less in total, or the residual by 1.5 % or less, in the last step
  kDiverged,   // the residual grew by more than 25 % and more than 1.5 in one step
  kMaxIters,   // neither within ParamSearchOptions::max_steps steps
};
const char* to_string(FixedPointStatus status);

// What the clip fallback (formerly the highlight rescue, Branch R) did. It is tried when the chosen bracket
// white-clips more than 0.03 % of the inner crop, on the next bracket of the bundle (target_idx + 1), and taken when
// that one passes every check.
enum class ClipFallbackOutcome {
  kNotNeeded,        // the chosen bracket does not clip
  kNotScreened,      // the next bracket was not screened
  kNotAdmissible,    // the next bracket's calibrated range is too narrow for the scene
  kClipsTooMuch,     // the next bracket clips more than half as much as the chosen one
  kShiftsLightness,  // the next bracket moves the median L* outside -4.5 .. +0.5
  kWorseResidual,    // the next bracket's residual is more than 10 % above the chosen one's
  kTaken,            // the next bracket replaces the chosen one
};
const char* to_string(ClipFallbackOutcome outcome);

struct ParamSearchOptions {
  float crop_ratio = 0.90f;  // the frame's centred inner crop the statistics and the previews use
  int max_steps = 8;         // the most steps of each bracket's fixed-point loop (neg_process's --dino_iters)
  // A film base measured on this capture (crosstalk-corrected RGB) instead of the bundle's; null for the bundle's.
  const float* film_base_rgb = nullptr;
  const double* film_base_rgb_d = nullptr;
  // True when the custom film base was sampled from the scan itself (e.g. rebate), so exposure_ratio is 1.0.
  bool film_base_from_scan = false;
  bool verbose = true;
  // The frame is not a detected film frame but the whole image (--roi auto fell back below its confidence): it may hold
  // rebate, bare light or holder. The curve fit then leaves out the grid cells at the film base's level or holder-dark
  // (fit_input_curves_after_search).
  bool fallback_crop = false;
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
  // The frozen shoulder of the best step (params.shoulder) and how it was found.
  float shoulder_q995 = 0.0f;   // the measured 99.5% point before clamping (auto only)
  float shoulder_knee = 1.0f;   // the linear knee x_k (1 when off)
  bool shoulder_guard = false;  // the intent loss with the shoulder is more than 2% worse than without
  // The intent field DINOv3 predicted from the best step's preview (the field that step's parameters were searched
  // against next and its residual was measured with): grid_h x grid_w cells, row-major, [L*, a*, b*] per cell.
  std::vector<float> mu, sigma;
  int grid_w = 0, grid_h = 0;
};

struct ParamSearchResult {
  int target_idx = -1;      // the chosen bracket, an index into ProfileBundle::targets
  std::string target_name;  // its name in the bundle ("Target 3")
  ConversionParams params;  // the chosen (E, g, b), E in this frame's units: diag(E, E g, E b) * M_frame
  float residual = 0.0f;    // the chosen bracket's FixedPointResult::residual
  // The chosen bracket's shoulder (params.shoulder is x_hi): measured q995, linear knee, and the guard.
  float shoulder_q995 = 0.0f;
  float shoulder_knee = 1.0f;
  bool shoulder_guard = false;
  FixedPointStatus status = FixedPointStatus::kMaxIters;  // and its FixedPointResult::status
  // The chosen step's intent field (FixedPointResult::mu, sigma, grid_w, grid_h), written to the sidecar's
  // intent_field, and that step's parameters (E in the search's reference units) and bracket.
  std::vector<float> intent_mu, intent_sigma;
  int intent_grid_w = 0, intent_grid_h = 0;
  ConversionParams intent_params;

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

  ClipFallbackOutcome clip_fallback = ClipFallbackOutcome::kNotNeeded;
  float clip_fallback_white_clip = 0.0f;  // the fraction of the inner crop the first choice white-clips
  bool highlight_pullback = false;        // the highlight pullback moved E or the bracket
  // The chosen bracket's calibrated range of E (its enclosure's e_min and e_max) in params.e's units; 0 when the
  // bracket has none (infeasible). The curve fit bounds E with it.
  float e_range_lo = 0.0f, e_range_hi = 0.0f;

  std::vector<ScreeningResult> screening;
  std::vector<FixedPointResult> tournament;
  int inferences = 0;  // DINOv3 inferences of the whole search: the bootstrap plus every fixed-point step
  // Time spent rendering previews, in DINOv3, in the (E, g, b) searches, on residuals, and in the whole search.
  std::chrono::steady_clock::duration render_time{}, infer_time{}, search_time{}, residual_time{}, total_time{};
};

// A bracket's exposure range on a frame, as the bundle search's Step 1 meters it (bracket_enclosure): the scene's
// green 0.5% and 99.5% points over the inner crop (crop_ratio), scaled by e_norm on white-light bundles, onto the
// bracket's calibrated range. In the station's E, which -E, Auto's result and the single-bracket search take, so the
// exposure given values default to and the single-bracket search centres on are the ones Auto meters for that bracket.
struct BracketExposure {
  bool valid = false;       // the bracket has patch readings
  bool feasible = false;    // e_min <= e_max
  float e_min = 0.0f, e_max = 0.0f;
  float e_center = 0.0f;    // sqrt(e_min * e_max)
  float search_lo = 0.0f, search_hi = 0.0f;  // the range the search would search E in
  // The metered exposure (Step 2): the E at which the scene's log-average renders at L* 50 through the bracket with
  // the bootstrap's g 1.10 and b 1.15, clamped to the search range: the E Auto's bootstrap starts that bracket from.
  // e_center when the bracket has no neutral L* 50 or the metering does not converge (has_metered false).
  bool has_metered = false;
  float e_metered = 0.0f;
};
// One BracketExposure per bracket of the bundle (out.size() == bundle.targets.size()). film_base_rgb: a film base read
// from this scan, as for measure_rolloff_x_hi; null for the bundle's.
void meter_bracket_exposures(const ProfileBundle& bundle, const negicc::RgbView& frame, double scan_shutter_s,
                             double scan_iso, float crop_ratio, const double* film_base_rgb,
                             std::vector<BracketExposure>& out);

// The highlight roll-off's x_hi (color_pipeline.h shoulder_x_hi_anchored) of frame, a whole frame at any size and
// orientation, converted through bracket target_idx at the search's E, g and b (ParamSearchResult::params, the
// station's): the 99.5% point of the pre-shoulder max channel over its inner crop, mapped to sRGB 242. The search's
// last step measures the same on its half-size frame; neg_process measures every bundle conversion's there too, so
// given values reproduce it. q995 gets the point. film_base_rgb: a film base read from this scan (film_base_detect.h),
// as ParamSearchOptions::film_base_rgb_d with film_base_from_scan; null for the bundle's.
// The same through a given profile (the bracket's, or a combined profile built from it).
float measure_rolloff_x_hi(const ProfileBundle& bundle, const negicc::ProfileData& prof, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995 = nullptr, const double* film_base_rgb = nullptr);
float measure_rolloff_x_hi(const ProfileBundle& bundle, int target_idx, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995 = nullptr, const double* film_base_rgb = nullptr);

struct KnobSolveResult {
  float e = 1.0f, g = 1.0f, b = 1.0f;
  bool limited = false;
  float grey_lab[3] = {50.0f, 0.0f, 0.0f};
};

// Solves the conversion gains (E, g, b) that move the grey that base=(base_e, base_g, base_b) renders as neutral L* 50
// through prof by brightness_stops (in stops of luminance), tint_a (CIELAB a*), and warmth_b (CIELAB b*), matching
// negicc-station's GreyPoint and ConversionSession._solve_knobs. grey_prof, when given, is the profile whose neutral
// L* 50 defines the grey (the bracket's profile before re-anchored curves changed it), rendered through prof.
bool solve_grey_knobs(const ProfileData& prof, float base_e, float base_g, float base_b, float brightness_stops,
                      float tint_a, float warmth_b, KnobSolveResult& out, const ProfileData* grey_prof = nullptr);
// The Lab through prof at gains egb of the grey that base renders as neutral L* 50 through grey_prof.
bool knob_grey_lab(const ProfileData& grey_prof, const ProfileData& prof, const float base[3], const float egb[3],
                   float lab_out[3]);
// How far, in stops of luminance, a grey rendered at L* grey_L lies from the L* brightness_stops asks for.
float knob_grey_miss_stops(float brightness_stops, float grey_L);

// Searches the conversion parameters of one frame. frame is the frame upright, already cut out of the half-size
// decode and oriented as the output (DINOv3 judges an image the way it will be seen); scan_shutter_s and scan_iso
// are the capture's exposure.
bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                              double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out);

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& frame,
                              float scan_shutter_s, float scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out);

// ---- Fitted G/B input curves (negicc-station designs/plan_fit_curves.md, input_curves.h) ----------------------

// Composes the G and B input tables of prof with curves, in place (its own tables, then rebound), so every copy of
// the bracket's profile (the search's renders, the grey point, the export) is the combined profile. No curves: no
// change.
void apply_input_curves(ParsedFilmProfile& prof, const negicc::InputCurves& curves);
// The same from base's (un-curved) G and B input tables into prof's: curves applied again at another anchor replace
// the earlier composition instead of composing on top of it. No curves: prof gets base's tables back.
void apply_input_curves(ParsedFilmProfile& prof, const ParsedFilmProfile& base, const negicc::InputCurves& curves);

// curves (anchored at from = {E, g, b}) re-anchored at to = {E, g, b} on frame, for the knobs, which move E, g and b
// after the curves were fitted or recorded: ref, u_med and u_lo moved by what the frame's film base and grid move
// between anchor_curves at from and at to, so with to == from the result is curves unchanged, bit for bit. frame,
// scan_shutter_s, scan_iso and opts (its film base and crop) as fit_input_curves_after_search took them; engine gives
// the grid's size only (it need not be loaded). Where the re-anchored curves leave the shape over the bracket's input
// table (curve_shape_ok at max_dev_stops), each channel's a and k are scaled down by the largest factor that keeps it
// (scale[0], scale[1]; 1: not needed). False (reported) on a bundle with a crosstalk matrix or a bad bracket.
bool reanchor_input_curves(const ProfileBundle& bundle, const DinoV3Engine& engine, int target_idx,
                           const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                           const ParamSearchOptions& opts, const negicc::InputCurves& curves, const float from[3],
                           const float to[3], float max_dev_stops, negicc::InputCurves& out, float scale[2]);

// The plausible range of Auto's grid median L* (the lightness the curve fit keeps): outside it the intent failed (a
// render nearly black or blown out, as when the frame holds rebate or holder), and the fit does not chase it.
constexpr float kIntentMinL = 12.0f;
constexpr float kIntentMaxL = 85.0f;

// What fit_input_curves_after_search did.
struct CurveRefitResult {
  negicc::CurveFitResult fit;     // the curves (with ref and u_med), g and b of the joint fit, its data losses
  ConversionParams params_before; // the search's parameters (E in this frame's units, as ParamSearchResult::params)
  ConversionParams params;        // the result: E re-metered on the combined profile, the fit's g and b (frame units)
  float target_median_L = 0.0f;   // Auto's grid median L* (the re-meter's target), on the export's render at Auto's
  float median_L = 0.0f;          // the grid median L* at params on the combined profile, on the export's render
  // The export's render: its encoding and roll-off (only the sRGB encoding applies one; 1 otherwise).
  int encoding = negicc::ENCODE_SRGB;  // OutputEncoding
  float x_hi_auto = 1.0f;         // the export's roll-off at Auto's parameters on the bracket's profile
  float x_hi_fit = 1.0f;          // the roll-off the last fit pass ran with (the first runs with x_hi_auto)
  int fit_passes = 0;             // fits run: one, and again while the result's roll-off moved (at most three)
  // The roll-off at params on the combined profile (and its q995): measured as the export measures it (only for the
  // sRGB encoding: rolloff_measured), so the export reuses it with no knobs instead of measuring it again.
  float x_hi = 1.0f, x_hi_q995 = 0.0f;
  bool rolloff_measured = false;
  int remeter_passes = 0;         // re-meter and roll-off alternations, over every re-meter (the shape rounds re-meter)
  bool remeter_converged = false; // the roll-off of the last pass equals the one its re-meter ran with
  double loss_export = 0.0;       // the intent loss at params on the combined profile, on the export's render
  // G's and B's a and k scaled by these to keep the shape at the export's anchor (1: not needed; 0: dropped), and the
  // smaller of the two.
  float shape_scale_ch[2] = {1.0f, 1.0f};
  float shape_scale = 1.0f;
  // Why the fit fell back to Auto's result with no curves, or null when it did not: "shape" (still past the shape
  // after three rounds of scaling), "clip" (E would leave [e_lo, e_hi] to keep Auto's median L*), or
  // "intent_lightness" (Auto's grid median L* outside [kIntentMinL, kIntentMaxL]: the intent failed, and E is re-metered
  // so the grid's median L* is the nearer end of that range instead of kept at Auto's).
  const char* fallback = nullptr;
  bool intent_lightness_failed = false;
  // On "intent_lightness": the L* E was re-metered to, whether the re-meter stopped at a bound of E before reaching
  // it, and whether E moved from Auto's.
  float fallback_target_L = 0.0f;
  bool fallback_clipped = false, fallback_e_moved = false;
  // Grid cells the fit left out on a fallback crop (ParamSearchOptions::fallback_crop), of the grid's cells.
  int cells_excluded = 0, cells_total = 0;
  // The bounds of E for the fit and the re-meter: the bracket's calibrated range widened to hold Auto's E.
  float e_lo = 0.0f, e_hi = 0.0f;
  // The search's intent field was predicted again at its final parameters (the pullback or the clip fallback moved
  // them after its preview), and the fit scored against that one.
  bool intent_repredicted = false;
  // The fit's options as it ran (the sidecar records them).
  negicc::CurveFitOptions options;
  // The loss's lightness and colour parts on the export's render: Auto's result on the bracket's profile, the fit's
  // result at its own E, and the result on the combined profile; and the time the fit, re-meter and checks took.
  negicc::LossParts parts_auto, parts_fit, parts_export;  // parts_fit: after the (last) fit, at fit_e
  double time_ms = 0.0;
  // The joint fit: the E it solved on its per-cell model, its grid renders for E, its steps, and whether it settled.
  float fit_e = 1.0f;
  int fit_renders = 0, fit_iterations = 0, fit_polish = 0, fit_start = 0;
  bool fit_converged = false;
};

// The station's ConversionSession.set_fit_curves on the bracket a finished search chose (res), for white-light
// bundles: on the same frame as the search,
//   1. the joint fit (negicc::fit_input_curves_joint): g, b and the four curve numbers by Levenberg-Marquardt against
//      the search's intent field on the frame's grid (predicted again at the search's final parameters when the
//      pullback or the clip fallback moved them after its preview), each candidate at the E that keeps Auto's grid
//      median L*, within the bracket's calibrated range of E widened to hold Auto's E; its curves anchored (ref,
//      u_med) on the film base (film_base_rgb_d in opts, else the brightest grid cell per channel) at its own E, g, b;
//   2. E re-metered on the real combined tables around the fit's E, so the grid's median L* at the fitted g and b
//      equals Auto's (the station's Iteration 230 F6L; no DINOv3 pass), each channel's curve scaled down where its
//      shape fails at the re-metered anchor; when the re-metered E leaves the bounds, or the shape still fails, Auto's
//      own result with no curves (CurveRefitResult::fallback);
//   3. the bracket's G and B input tables composed with the curves, anchored at the result, in place
//      (apply_input_curves on bundle.targets[res.target_idx]).
// Every step renders the grid as the export renders it (encoding: the export's OutputEncoding, ENCODE_SRGB or
// ENCODE_SRGB_LINEAR, which is what the fit models): its roll-off, measured as the export measures it, at Auto's
// parameters for the target and the first fit, alternated with the re-meter until it holds still at the result, and
// the fit run again (up to twice) when that roll-off differs from the one it ran with.
// False (reported) on a bundle with a crosstalk matrix, a search without an intent field, or a failed inference.
bool fit_input_curves_after_search(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                                   double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                                   const ParamSearchResult& res, const negicc::CurveFitOptions& fit_opts,
                                   int encoding, CurveRefitResult& out);

#endif  // MULTI_TARGET_OPTIMIZER_H
