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

#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <functional>
#include <memory>
#include <vector>

#include "rgb_view.h"
#include "dinov3_engine.h"
#include "input_curves.h"
#include "profile_bundle.h"

// Interleaved 16-bit camera RGB: width x height pixels, row-major, rgb holding width * height * 3 values.
struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint16_t> rgb;
};

// The conversion parameter search, ported from negicc-station's (PerTargetConverter.convert with its defaults). It
// does not convert anything: on the half-size analysis decode of one frame it picks the calibration bracket of a
// profile bundle and the exposure E and gains g, b that neg_process then converts the full image with.
//
//   1. Enclosure: each bracket's calibrated range of E (the scene's green range inside the green range its patches
//      were calibrated over) and its usable range, the calibrated range widened by the highlight and shadow
//      tolerance; every bracket is searched over its usable range.
//   2. Bootstrap: meter each admissible bracket so the scene's log-average renders at L* 50, render every metered
//      bracket at its metered E clamped to its usable range (when none meters, the bracket whose range the scene fills
//      best, at its e_center) and average the intent fields DINOv3 predicts from those renders: the bootstrap field.
//      Its L* is the frame's one lightness target.
//   3. Screening: search (E, g, b) of every metered bracket (every admissible one when none meters) against the
//      bootstrap field.
//   4. Tournament: run the fixed-point loop on the best two screened brackets and keep the lower residual, whichever
//      way each loop ended. The loops hold L* to the frame's lightness target.
//   5. The clip fallback: move to the next bracket when the chosen one white-clips (ClipFallbackOutcome).
//   6. The highlight pullback: when a bright frame still has many highlights near white (sRGB 242 and 235), nudge E
//      (at most 0.15 stop) toward the chosen bracket's calibrated range, or, when that bracket cannot hold the scene,
//      run Steps 2-4 again with penalties on E outside the bracket's calibrated range in place of the pull toward the
//      centre, and take the result if it clips at most 0.5 percentage point more and moves to a later bracket or a
//      higher E.
//
// Every render carries the highlight roll-off (color_pipeline.h shoulder_x_hi_anchored): the fixed-point loop
// measures its x_hi at each iteration's incoming parameters, and the search measures it once more at its final
// parameters, after the clip fallback and the pullback have moved E (and possibly the bracket) past the loop that
// froze it, so the written image's shoulder matches its highlights. The intent field the result carries
// (ParamSearchResult::intent_mu) is the one the chosen loop step was scored against, at intent_params: its L* is the
// frame's lightness target, its a* and b* the step's own prediction.
//
// The intent field is DINOv3's prediction, from an 8-bit sRGB render of the frame, of how each patch of the image
// should look: a mean mu and a spread sigma of CIELAB per patch of its grid. How far a render is from a field is
// the mean squared z-score over the grid,
//
//   z2(render, field) = 1 / (3 N) * sum_i sum_{k in L, a, b} ((lab_i,k - mu_i,k) / sigma_i,k)^2,
//
// with lab_i the CIELAB of the render over patch i and N the number of patches.

// The parameters the search solves. With the input matrix M the search runs on (rows R, G, B; input_matrix on
// TriLED, M_frame on white light, see ParamSearchResult::input_matrix) the conversion matrix is
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
  // E, g, b moved by 0.02 or less in total in the last step; or, on TriLED, the residual by 1.5 % or less (in the
  // pullback's search also not rising); or, on white light, the residual, already above the best step's, did not fall
  // again (the best step is kept). The 1.5 % rule does not apply on white light.
  kConverged,
  kDiverged,   // the residual grew by more than 25 % and more than 1.5 in one step
  kMaxIters,   // neither within ParamSearchOptions::max_steps steps
};
const char* to_string(FixedPointStatus status);

// What the clip fallback did. It is tried when the chosen bracket white-clips more than 0.03 % of the inner crop, on
// the next bracket of the bundle (target_idx + 1), and taken when that one passes every check: it must at least halve
// the clipping, keep the median L* within -4.5 .. +0.5 of the first choice's (the midtones may darken a little to save
// the highlights, but the frame's rendering must not change), and keep the residual within 10 % (DINOv3 must still
// find the render as plausible).
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

// The passes over the frame the bundle search renders: the preview of the inner crop and the clip statistics (the
// roll-off's 99.5 % point, white-clip shares). The defaults are negicc::cpu_render and negicc::cpu_highlight_clip; a
// caller that renders the same pipeline elsewhere (the station's GPU) overrides them. search_conversion_params,
// search_print_continuous, run_white_light_fit (the curve fit, its roll-off and its tone plan) and
// plan_auto_tone_stage send every pass of theirs through it (reanchor_input_curves renders nothing); the passes are
// serial, called from the caller's thread one at a time.
//
// What stays and what changes between calls:
// - bind_frame is called before the passes of each measurement with the frame they crop, and again, with the same
//   frame, by each roll-off measurement (measure_rolloff_x_hi). The frame's pixels and the view given to bind_frame
//   stay valid until the search returns.
// - RenderArgs::in_view and HighlightClipArgs::in_view point to views made for the call (crops of the bound frame): a
//   renderer reads them during the call and keeps no pointer to them.
// - The bracket's 3D table (cfg's clut) and output tables are the bundle's and stay for the whole search. The three
//   input tables (cfg's in_trc) may change between any two calls, in place at the same address (the curve fit's
//   re-meter passes) or freed and replaced (when the fit composes its curves into the bracket): a renderer reads them
//   on every call, or compares their contents, and never keys a copy by their address.
// - The search owns every output buffer: out is sized before each call and must be filled before it returns.
//
// A renderer may refuse a frame it cannot read (a CFA view with its private flat-field mapping, say) by throwing from
// bind_frame. A failed pass throws: the exception leaves the search, the fit, the tone plan or the roll-off measurement
// to the caller (none of them catches it), so a caller can tell a render failure from every other one and run the
// search again on the CPU. The bundle is then as it was given: run_white_light_fit restores the bracket's own input
// tables when a pass throws after it has composed its curves into them. The result structures (ParamSearchResult,
// WhiteLightFitResult, ToneStagePlan) are then partly filled, and a caller must not read them.
//
// neg_process --check_renderer runs a conversion with a pass-through renderer (PassThroughRenderer) and counts any
// pass that runs outside it (PassAudit), on any thread, and fails when a pass ran outside it or none ran at all; with a
// CUDA device it also compares each pass with the GPU's (CompareRenderer). Its picture must equal a conversion whose
// passes run on the CPU, bit for bit, which nothing checks for it. Run it, and compare its picture with such a
// conversion's (cmp), on a TriLED scan and on a white-light scan with -P srgb (the curve fit and the tone plan), before
// a change to the search, the curve fit, the tone plan or this interface is merged.
struct PreviewRenderer {
  virtual ~PreviewRenderer() = default;
  virtual void bind_frame(const negicc::RgbView& frame) { (void)frame; }
  virtual void render(const negicc::RenderArgs& a) { negicc::cpu_render(a); }
  virtual void highlight_clip(const negicc::HighlightClipArgs& a) { negicc::cpu_highlight_clip(a); }
};

// The check aid (neg_process --check_renderer): the passes the bundle search's entry points run through a renderer
// and outside it. AuditScope marks those entry points; a CPU pass inside one that does not come through
// PassThroughRenderer is counted as outside (negicc::cpu_pass_hook).
struct PassAudit {
  long binds = 0, renders = 0, clips = 0, outside = 0;
};
PassAudit& pass_audit();
void audit_cpu_pass(int kind);   // the hook neg_process sets in negicc::cpu_pass_hook
struct PassThroughRenderer : PreviewRenderer {
  void bind_frame(const negicc::RgbView& frame) override;
  void render(const negicc::RenderArgs& a) override;
  void highlight_clip(const negicc::HighlightClipArgs& a) override;
};

// The passes on a CUDA device (rgb_view.h gpu_render and gpu_highlight_clip): neg_process's on a USE_CUDA=1 build
// with a device whose kernels it was built for (gpu_device_available), unless --dino_backend cpu. The constructor
// throws without one.
// - The frame: bind_frame copies the bound frame to the device, each pixel as the view samples it (a CFA view's flat
//   field included), unless device_frame gives the address at which the GPU already reads it (a caller whose frame is
//   already in pinned, mapped memory). Binding the same view again (the same pointers, size, layout and flat field)
//   keeps the copy, so a caller that rewrites a frame in place, or frees it and allocates another at the same address,
//   calls release_frame before binding it. A pass's view, the bound frame or an upright crop inside it with the same
//   layout, is read there at its offset; any other view throws.
// - release_frame: drops the bound frame and frees its device copy (neg_process calls it after the last pass of the
//   search, before the full-resolution conversion).
// - The tables: each table a pass reads (the cLUT, the input and output tables, the tone table) is compared with the
//   copy on the device on every pass and copied again when its contents differ, so a table changed in place (the
//   curve fit's) or another one at the same address is never read stale.
// - The output: a render is written to device memory and copied into the search's buffer.
// A failed pass throws negicc::GpuError, which leaves the search to its caller.
struct GpuRenderer : PreviewRenderer {
  GpuRenderer();
  ~GpuRenderer() override;
  GpuRenderer(const GpuRenderer&) = delete;
  GpuRenderer& operator=(const GpuRenderer&) = delete;
  void bind_frame(const negicc::RgbView& frame) override;
  void render(const negicc::RenderArgs& a) override;
  void highlight_clip(const negicc::HighlightClipArgs& a) override;
  void release_frame();
  // For a frame the GPU reads in place: the device address of frame.r_ptr (an interleaved RGB view without a flat
  // field, whose pitch and stride hold on the device), or null to have the frame copied.
  std::function<const uint16_t*(const negicc::RgbView& frame)> device_frame;
  long binds = 0, renders = 0, clips = 0;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The CPU and the GPU passes compared (neg_process --check_renderer on a build with a device that runs its kernels):
// every pass runs on both from the same inputs and the search reads the CPU's, so the search is the CPU's and each GPU
// pass is measured against it. The CPU's side is PassThroughRenderer's, so the passes are counted as without a device.
// - A render: how many values differ and the largest difference; the largest and the mean change in L* per pixel,
//   each pixel read in the render's encoding (sRGB or linear sRGB; an 8-bit preview, what the network reads, a 16-bit
//   or a float render alike); and with
//   statistics, the largest difference of the white and black clip shares and of each histogram bin (as a share of
//   the crop's pixels), and of the six zone means of a* and b*.
// - A clip pass: the largest difference of its three exceedance shares and its white-clip share, its median L*, and
//   each bin of its max-channel histogram (as a share of the crop's pixels).
// - A value that is not finite on either side counts as differing and fails the comparison.
// within_tolerance(): every render's mean change in L* per pixel and every zone mean within kCompareMeanDL, every
// share (a clip share or a histogram bin's) within kCompareShare %, every median within one histogram bin
// (kCompareMedianL), and no value that is not finite.
struct CompareRenderer : PassThroughRenderer {
  static constexpr double kCompareMeanDL = 0.05;   // CIELAB: a render's mean change in L* per pixel; a zone mean
  static constexpr double kCompareShare = 0.01;    // %: each clip share, each histogram bin
  static constexpr double kCompareMedianL = 0.1;   // L*: the median, one bin of the clip pass's histogram
  void bind_frame(const negicc::RgbView& frame) override;
  void render(const negicc::RenderArgs& a) override;
  void highlight_clip(const negicc::HighlightClipArgs& a) override;
  bool within_tolerance() const;
  std::string summary() const;   // one line for the log
  struct Pass {
    bool render = false, stats = false;   // a render, with its statistics
    long long values = 0, differing = 0, nonfinite = 0;
    // max_share: the clip shares (a render's white and black clip); max_hist: a histogram bin, as a share of the
    // crop's pixels (%); max_zone: a render's zone means of a* and b*.
    double max_abs = 0.0, max_dl = 0.0, mean_dl = 0.0, max_share = 0.0, max_median_l = 0.0, max_hist = 0.0,
           max_zone = 0.0;
  };
  std::vector<Pass> passes;
  GpuRenderer gpu;
};

struct ParamSearchOptions {
  // The passes over the frame (PreviewRenderer); null: the CPU's, as neg_process runs.
  PreviewRenderer* renderer = nullptr;
  float crop_ratio = 0.90f;  // the frame's centred inner crop the statistics and the previews use
  int max_steps = 8;         // the most steps of each bracket's fixed-point loop (neg_process's --dino_iters)
  // A film base measured on this capture (crosstalk-corrected RGB) instead of the bundle's; null for the bundle's. Set
  // one of the two (neg_process sets film_base_rgb_d): the curve fit's roll-off reads film_base_rgb_d only.
  const float* film_base_rgb = nullptr;
  const double* film_base_rgb_d = nullptr;
  // True when the custom film base was sampled from the scan itself (e.g. rebate), so exposure_ratio is 1.0.
  bool film_base_from_scan = false;
  bool verbose = true;
  // The frame is not a detected film frame but the whole image (--roi auto fell back below its confidence, or a given
  // --roi is the whole image): it may hold rebate, bare light or holder. The curve fit and the knobs' re-anchoring of
  // the curves (reanchor_input_curves) then leave out the grid cells that are holder-dark, and (when
  // film_base_is_rebate is true) those at the film base's level (fallback_picture_cells). When the film base
  // came from the scan's statistics (the 99.5th percentile of the film itself), its level is the thinnest picture
  // highlights rather than unexposed rebate, so only holder-dark cells are left out.
  bool fallback_crop = false;
  bool film_base_is_rebate = false;
  // Pins (neg_process's --target, -E, --gain_g and --gain_b on a bundle): the search and the curve fit run
  // as without them, with the bracket restricted to pin_target (when >= 0) and E, g or b (pin_mask: PIN_E, PIN_G and
  // PIN_B of dinov3_auto_solver.h) held at pin_e, pin_g and pin_b (E in ParamSearchResult::params' units). A pinned
  // bracket leaves out the clip fallback, which only changes the bracket; a pinned E leaves out the highlight
  // pullback, which only moves E or the bracket. The curve fit (run_white_light_fit) holds the pinned g and b at the
  // search's and does not re-meter a pinned E.
  int pin_target = -1;
  unsigned pin_mask = 0;
  float pin_e = 1.0f, pin_g = 1.0f, pin_b = 1.0f;
  // neg_process's --probe_intent (a measurement aid): when probe_path is set, the search writes what the intent network
  // predicts from chosen renders to that JSON file and stops after its bootstrap, before screening
  // (ParamSearchResult::probed). Rendered: every metered bracket at its metered E with the bootstrap's gains (and at
  // the E its window clamps it to, where that differs), then every probe_params set (target index, E in the search's
  // units, g, b) at each exposure offset of probe_stops (stops), with the roll-off measured at each as the fixed-point
  // loop measures it. Each render's cells are recorded two ways: render_cells (the trained cells, as the residual reads
  // them and as the training targets are made) and optimiser_cells (what intent_loss scores at those parameters, before
  // the per-bracket correction). Each render's predicted is the network's own prediction from it, the mean of its mu
  // cells, with nothing held; the file's target_L is the mean L* of the frame's lightness target (the bootstrap field's
  // L*, which the search holds).
  std::string probe_path;
  std::vector<std::array<float, 4>> probe_params;
  std::vector<float> probe_stops = {0.0f};
};

// Step 3, one row per metered bracket: one coordinate descent of (E, g, b) against the bootstrap intent field (the
// mean of the fields predicted from the metered brackets' renders), each bracket aimed at that field minus its own
// correction (update_correction). No inference runs during the screening, so the rows compare the brackets cheaply;
// the best two go on to the tournament.
struct ScreeningResult {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  ConversionParams params;  // the best (E, g, b), E in the search's units (reference on TriLED, this frame's on white
                            // light; see ParamSearchResult::input_matrix)
  // The intent loss at params (intent_loss): z2 of the optimiser's cells, the patch-averaged grid rendered at params,
  // against the bootstrap field minus the bracket's correction, plus the search's penalties on clipping and on E far
  // from the bracket's centre exposure.
  float intent_loss = 0;
  float e_lo = 0, e_hi = 0;  // the bounds E was searched within (the search's units)
  float e_metered = 0;       // E that renders the scene's log-average at L* 50; e_center when no bracket metered
};

// Step 4, one row for each of the best two screened brackets: its fixed-point loop. Each step renders the frame at
// the current (E, g, b), runs DINOv3 on the render (one inference), replaces the predicted L* with the frame's
// lightness target, measures the residual and searches the next (E, g, b) against that field. The row keeps the step
// with the lowest residual.
struct FixedPointResult {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  ConversionParams params;  // (E, g, b) of the best step, E in the search's units
  // The residual of that step: z2 of the render against the field it was scored against (L* the frame's lightness
  // target, a* and b* predicted from the same render). Low when the render already looks the way DINOv3 expects.
  float residual = 0;
  FixedPointStatus status = FixedPointStatus::kMaxIters;
  int steps = 0;  // render -> DINOv3 -> search steps run, one inference each
  // The frozen shoulder of the best step (params.shoulder) and how it was found.
  float shoulder_q995 = 0.0f;   // the measured 99.5% point before clamping (auto only)
  float shoulder_knee = 1.0f;   // the linear knee x_k (1 when off)
  bool shoulder_guard = false;  // the L* part of the intent loss is more than 2% worse with the shoulder than without
  // The intent field of the best step: L* (and its sigma) the frame's lightness target, a* and b* (and their sigma) as
  // DINOv3 predicted them from that step's preview. Its residual was measured against this field, and the next
  // parameters were searched against it. grid_h x grid_w cells, row-major, [L*, a*, b*] per cell.
  std::vector<float> mu, sigma;
  // The neutral head's p per cell from the same preview (DinoV3InferenceResult::neutral_p); empty without one.
  std::vector<float> neutral_p;
  // The tone heads' input from the same preview: the backbone's pooled features and the tokens' spread (1920 values,
  // ToneHead); empty unless the engine gives the spread (DinoV3Engine::set_token_spread).
  std::vector<float> tone_features;
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
  // intent_field, and that step's parameters (E in the search's units).
  std::vector<float> intent_mu, intent_sigma;
  std::vector<float> intent_neutral_p;  // FixedPointResult::neutral_p of that step (empty without a neutral head)
  std::vector<float> intent_tone_features;  // FixedPointResult::tone_features of that step (may be empty)
  int intent_grid_w = 0, intent_grid_h = 0;
  ConversionParams intent_params;

  // The bracket's input matrix and the frame's exposure. With X the bundle's crosstalk matrix, f the film base
  // RGB, n the bundle's normalisation target and rho the exposure of the film base capture over this frame's
  // (shutter x ISO / 100):
  //
  //   M_frame[r][c] = X[r][c] * n / f[r] * rho            (the merged input matrix of this frame)
  //   e_norm        = M_frame[1][1] / M_ref[1][1]         (M_ref: the same for a 1/8 s, ISO 100 scan)
  //   input_matrix  = M_frame / e_norm                    (reference units)
  //   row_scale[r]  = n / f[r] * rho / e_norm,  so  input_matrix[r][c] = row_scale[r] * X[r][c]
  //
  // On TriLED the search runs on input_matrix: its E is in reference units, so the same E means the same rendering
  // whatever the scan's exposure, and params.e is E / e_norm. On white light (no crosstalk) it runs on M_frame and
  // scales the scene's green range by e_norm instead (bracket_enclosure), so its E is params.e. row_scale lets the
  // result be written back as the bundle's crosstalk with -E, --gain_g and --gain_b (neg_process's cli_equivalent).
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
  // DINOv3 inferences of the whole search: the bootstraps and every fixed-point step (the clip fallback's and the
  // pullback's included).
  int inferences = 0;
  bool probed = false;  // --probe_intent: the search wrote its probe file and stopped (ParamSearchOptions::probe_path)
  // Time spent rendering previews, in DINOv3, in the (E, g, b) searches, on residuals, and in the whole search.
  std::chrono::steady_clock::duration render_time{}, infer_time{}, search_time{}, residual_time{}, total_time{};
};

// The highlight roll-off's x_hi (color_pipeline.h shoulder_x_hi_anchored) of frame, a whole frame at any size and
// orientation, converted through prof (first overload: a bracket's profile, or a combined profile built from it) or
// bracket target_idx (second) at the search's E, g and b (ParamSearchResult::params, the station's): the 99.5% point of
// the pre-shoulder max channel over its inner crop, mapped to sRGB 242. The search's last step measures the same on its
// half-size frame; neg_process measures every bundle conversion's there too, so given values reproduce it. q995 gets
// the point. film_base_rgb: a film base read from this scan (film_base_detector.h), as
// ParamSearchOptions::film_base_rgb_d with film_base_from_scan; null for the bundle's.
float measure_rolloff_x_hi(const ProfileBundle& bundle, const negicc::ProfileData& prof, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995 = nullptr, const double* film_base_rgb = nullptr,
                           PreviewRenderer* renderer = nullptr);
float measure_rolloff_x_hi(const ProfileBundle& bundle, int target_idx, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995 = nullptr, const double* film_base_rgb = nullptr,
                           PreviewRenderer* renderer = nullptr);

struct KnobSolveResult {
  float e = 1.0f, g = 1.0f, b = 1.0f;
  bool limited = false;                     // the asked Lab was out of reach: e, g, b reach as far toward it as can be
  float grey_lab[3] = {50.0f, 0.0f, 0.0f};  // the Lab the grey renders at with e, g, b
  // When solve_grey_knobs returns false because no grey was found (GreyPoint's solve for a neutral L* 50 on the grey's
  // profile did not come within 0.25 of it): the dE76 that solve ended at. 0 otherwise.
  float grey_miss = 0.0f;
};

// Solves the conversion gains (E, g, b) that move the grey that base=(base_e, base_g, base_b) renders as neutral L* 50
// through prof by brightness_stops (in stops of luminance), tint_a (CIELAB a*), and warmth_b (CIELAB b*), matching
// negicc-station's GreyPoint and ConversionSession._solve_knobs. grey_prof, when given, is the profile whose neutral
// L* 50 defines the grey (the bracket's profile before re-anchored curves changed it), rendered through prof. False
// when base is not positive, or when no grey is found (out.grey_miss).
bool solve_grey_knobs(const ProfileData& prof, float base_e, float base_g, float base_b, float brightness_stops,
                      float tint_a, float warmth_b, KnobSolveResult& out, const ProfileData* grey_prof = nullptr);
struct PrintKnobResult {
  double a = 0.0;                           // the print's mid grey A after the knobs
  float g = 1.0f, b = 1.0f;                 // its gains (in the print's curves)
  bool limited = false;
  float grey_lab[3] = {50.0f, 0.0f, 0.0f};  // the grey's Lab through the print at (a, g, b)
};
// The knobs on a print, solved on the print's own profile (multi_target_optimizer.cc).
// print_at(a, g, b, prof) builds the print profile at mid grey a with gains g, b in its curves; A stays within
// [a_lo, a_hi], the range the print's curves are defined on (PrintContinuum::a_lo, a_hi).
bool solve_print_knobs(const std::function<void(double a, double g, double b, ParsedFilmProfile& prof)>& print_at,
                       double a0, float g0, float b0, double a_lo, double a_hi, float brightness_stops, float tint_a,
                       float warmth_b, PrintKnobResult& out);
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

// A stop of exposure in density, log10 2: the print search's unit of A.
inline const double kStopDensity = std::log10(2.0);

// The print's search (--print with Auto on a bundle with grey_balance_curves, either light): the print searched with
// its mid grey A as a continuous variable in place of the bracket (np_print.h PrintContinuum), E held at 1 in the
// search's reference units (the negative's own density above base, as the brackets were measured): A0 metered once (the
// A at which the frame's log-average cell prints at L* 50, as the scene's search meters a bracket); the lightness
// target from the mean field of the renders at A0 - boot_h, A0, A0 + boot_h (A0 alone when boot_h is 0), held; g and b
// screened at A0 on the optimiser's wide grids; then the fixed point over (A, g, b): each step renders the print at its
// A, measures the optimiser's correction, predicts a* b* (L* is the lightness target) and solves A (bracketed by steps
// of 0.1 APD downhill from the current A, then golden section to 0.001, each A with g and b by the optimiser, E
// pinned). The scene loop's thresholds stop it (A counted in stops, dA / log10 2); the lowest-residual step is kept. No
// tournament, no bracket fallback, no highlight pullback. ParamSearchResult::target_idx is the bracket whose A_mid is
// nearest A.
// The print's film base held no darker than the picture's thinnest film, for a base that is not a capture (the
// scan's rebate or statistics, or the bundle's): a base darker than the thinnest film would push thin film past the
// knee and cut the print's shadows. The factor β = max(1, the largest over the three channels of the inner crop's
// kBasePercentile point of (X v)_c / fb_c) (DESIGN.md 5.8), v a pixel of the frame, X the bundle's crosstalk matrix
// and fb the base (f in DESIGN.md) in the conversion's film base space (fb_c reads n0 at the print's E). The base the
// print takes is fb β, its colour kept; on a base already raised β is 1. Every 4th pixel each way.
float print_floor_factor(const ProfileBundle& bundle, const negicc::RgbView& frame, float crop_ratio,
                         const double fb[3]);
// The base a print takes when the film base is not a capture: found (the scan's rebate or statistics), or null for none
// found, when the bundle's base is taken, scaled once by the exposures (the scan's over its base capture's) as the
// conversion takes it. Writes the base before the floor (base_in) and after it (base_out, base_in times β) at the
// scan's exposure, in the conversion's film base space, and returns β.
float raise_base_to_print_floor(const ProfileBundle& bundle, const negicc::RgbView& frame, float crop_ratio,
                                const double* found, double scan_shutter_s, double scan_iso, double base_in[3],
                                double base_out[3]);

struct ContinuousPrint {
  // The print's input tables at A with the gains (r, g, b) in them, after the knee (null: none), as the conversion
  // renders the print (np_print.h PrintContinuum::trc_at).
  std::function<void(double a, const double gains[3], std::vector<float> trc[3])> trc_at;
  double a_lo = 0.0, a_hi = 0.0;     // the range A is searched in
  std::vector<double> a_mid;         // per bundle.targets
  double boot_h = kStopDensity / 2;  // half a stop of printing exposure (the bootstrap's spread)
};
struct ContinuousPrintResult {
  double a = 0.0, a0 = 0.0;
  int steps = 0;
  bool converged = false;
  float above_knee = 0.0f;           // the share of the inner crop above the knee at the bootstrap (the search's E,
                                     // g = b = 1)
  // At the final settings, per channel (the gains are in the print's curves, after the knee, so this is at E alone):
  // the share of the inner crop past the knee (input above 0.95, 0.054 density thinner than the base) and past the cut
  // (above 1.1227, where the knee's tanh reaches its cap of 0.9999: 0.126 density thinner), where the picture keeps no
  // detail.
  float past_knee[3] = {0.0f, 0.0f, 0.0f}, past_cut[3] = {0.0f, 0.0f, 0.0f};
  float target_mean_l = 0.0f;        // the mean L* of the lightness target it held
  float above_white[3] = {0, 0, 0};   // the share of the inner crop with the channel above 1.002 (linear) at the
                                      // result: cut
  float at_white[3] = {0, 0, 0};      // the share with the channel at 0.999 or more: the paper's white and above
  bool a0_at_range_end = false;      // A0's bisection ended at an end of the range
  float last_move[3] = {0, 0, 0};    // the last step's move: |dA| in stops, |dg|, |db|
  const char* stopped_by = "steps";   // "parameters" (the move within TOL_PARAM), "residual" (its change within
                                      // TOL_REL_ERR), "steps"
};
// One print material's search: print_base is a print profile of the bundle (its table and output curves; its input
// curves are rewritten at each A, with g and b in them for every render and optimiser pass, the matrix at E alone, as
// the conversion renders it), in its continuum; the search writes out and res. Each outlives the search.
struct PrintMaterialSearch {
  const ParsedFilmProfile* print_base = nullptr;
  const ContinuousPrint* in = nullptr;
  ParamSearchResult* out = nullptr;
  ContinuousPrintResult* res = nullptr;
};
// The print's search on one frame for each material in prints, in order (a picture print has one, a density output's
// --print one per material). The materials share the bundle's copy, the frame's crop, statistics and scale, and its
// shares at E; each has its own search.
bool search_print_continuous(const ProfileBundle& bundle, const std::vector<PrintMaterialSearch>& prints,
                             DinoV3Engine& engine, const negicc::RgbView& frame, double scan_shutter_s,
                             double scan_iso, const ParamSearchOptions& opts);
// The bracket that names a print at mid grey a: the one whose own A_mid is nearest a (the first of equals, in the
// bundle's order). The search names its result's this way, and a print the knobs moved is named by its moved A.
int nearest_bracket(const std::vector<double>& a_mid, double a);

bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const RgbImage& frame,
                              float scan_shutter_s, float scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out);

// ---- Fitted G/B input curves (input_curves.h) -------------------------------------------------------------------

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
// scan_shutter_s, scan_iso and opts (its film base and crop) as run_white_light_fit took them; engine gives
// the grid's size only (it need not be loaded). Where the re-anchored curves leave the shape over the bracket's input
// table (curve_shape_ok at max_dev_stops), each channel's a and k are scaled down by the largest factor that keeps it
// (scale[0], scale[1]; 1: not needed). False (reported) on a bundle with a crosstalk matrix or a bad bracket.
bool reanchor_input_curves(const ProfileBundle& bundle, const DinoV3Engine& engine, int target_idx,
                           const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                           const ParamSearchOptions& opts, const negicc::InputCurves& curves, const float from[3],
                           const float to[3], float max_dev_stops, negicc::InputCurves& out, float scale[2]);

// The plausible range of Auto's grid median L* (the lightness the curve fit keeps): outside it the intent failed (a
// render nearly black or blown out, as when the frame holds rebate or holder), and the fit does not chase it.
constexpr float kIntentMinL = 18.0f;
constexpr float kIntentMaxL = 85.0f;

// The tone stage's plan (tone_stage.h): the tone heads' predictions (dinov3_engine.h, on the features of
// one inference on the final render's preview) and that preview; the conversion solves the curve between the
// preview's descriptors and the predictions (TonePlanCurve) and applies it to the final image.
struct ToneStagePlan {
  bool planned = false;           // a curve was solved (by the conversion) and the conversion applies it
  const char* skipped = nullptr;  // why not: "off", "encoding" (not -P srgb), "features" (no pooled features)
  // Whose DINOv3 features the heads read: "final" (the final state's preview, one inference); "search" (the intent
  // preview) only if that inference failed.
  const char* features = "search";
  double pred_mean[11] = {}, pred_sd[11] = {};  // the heads' predictions (ToneHead targets, in their order)
  // The preview (8-bit RGB at DINOv3's size) the curve is solved on, kept for the conversion, which also caps the
  // highlight move on the final image where it would clip (tone_plan_for_image), solving it again at a smaller share.
  std::vector<uint8_t> preview;
};

// What run_white_light_fit did.
struct WhiteLightFitResult {
  // The joint fit of the last pass: its curves (with ref and u_med), g, b and E, its data losses, evaluations, grid
  // renders for E and steps, whether it settled, and its never-greener guard.
  negicc::JointCurveFit fit;
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
  // Why the fit fell back to Auto's result with no curves, or null when it did not: "cells" (fewer than 8 grid cells in
  // range, so no fit ran), "start" (Auto's start could not keep target_L within the E bounds, so no fit ran; the
  // re-meter still runs on Auto's values), "shape" (still past the shape after three rounds of scaling), "clip" (E
  // would leave [e_lo, e_hi] to keep Auto's median L*), or "intent_lightness" (Auto's grid median L* outside
  // [kIntentMinL, kIntentMaxL]: the intent failed, and E is re-metered so the grid's median L* is the nearer end of
  // that range instead of kept at Auto's).
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
  // them after its preview), and the fit scored against that one: its a* and b*, with L* held to the frame's
  // lightness target.
  bool intent_repredicted = false;
  // The fit's options as it ran (the sidecar records them).
  negicc::CurveFitOptions options;
  // The loss's lightness and colour parts on the export's render: Auto's result on the bracket's profile, the fit's
  // result at its own E, and the result on the combined profile; and the time the fit, re-meter and checks took.
  negicc::LossParts parts_auto, parts_fit, parts_export;  // parts_fit: after the (last) fit, at fit.e
  negicc::NeutralStats neutral_auto, neutral_export;
  // The fit had scene-neutral anchors (the neutral head's p with some evidence); without them the anchors hold Auto's
  // colour and only the intent field moves the overall a*.
  bool neutral_anchors = false;
  // Whose p the guard (fit.guard) weighted by: "search" (the search's chosen preview) or "refit" (predicted again at
  // the search's final parameters, intent_repredicted).
  const char* guard_p_source = "search";
  // The tone stage's plan.
  ToneStagePlan tone;
  double time_ms = 0.0;
};

// The tone stage's plan after a search with no curve fit (TriLED bundles, and white light with --fit_curves off): the
// heads' predictions and the preview of the search's own result, res.params on the bracket's profile at the roll-off
// the conversion measures there. Nothing of the search's result changes. False (reported) on a bad bracket; a plan
// that could not be made says why in out.skipped.
bool plan_auto_tone_stage(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                          double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                          const ParamSearchResult& res, int encoding, const ToneHead& tone_head, ToneStagePlan& out);

// The white-light fit on plain data: what run_white_light_fit does after the frame and its intent field are known,
// for callers with their own frame and search (negicc-station). Every pointer is the caller's and outlives the call.
struct WhiteLightFitData {
  const negicc::ProfileData* prof = nullptr;  // the bracket's own profile (un-curved tables)
  const float* merged = nullptr;              // the 3x3 input matrix before the gains, E in the frame's units
  const uint16_t* grid = nullptr;             // n cells of 16-bit camera-linear RGB (the frame's statistics grid)
  int n = 0, grid_w = 0;                      // cells, and the grid's width (the cells' 4 x 4 blocks)
  // The film base the conversion uses, in the grid's units; without one the brightest grid cell per channel.
  bool has_film_base = false;
  float film_base[3] = {0.0f, 0.0f, 0.0f};
  // The frame is the whole image, not a detected frame (cells at the film base's level or holder-dark are left
  // out), and whether its base was read from the rebate.
  bool fallback_crop = false, film_base_is_rebate = false;
  // The intent field (n x 3 each) and the neutral head's p per cell (n values, or empty when the head did not run),
  // of a preview at (e, g, b) on the bracket's profile.
  std::vector<float> mu, sigma, neutral_p;
  float e = 1.0f, g = 1.0f, b = 1.0f;         // the search's result
  float e_range_lo = 0.0f, e_range_hi = 0.0f; // the bracket's calibrated range of E (0: unknown)
  int pin_mask = 0;                           // PIN_E | PIN_G | PIN_B the caller holds
  bool verbose = false;
  int encoding = negicc::ENCODE_SRGB;         // the export's (the roll-off applies to sRGB only)
};
// The export's roll-off x_hi (and its 99.5% point) at (e, g, b) on the bracket's tables composed with curves
// (combined holds them composed; curves inactive: the bracket's own), measured as the caller's conversion measures it.
using WhiteLightRolloff = std::function<float(const negicc::ProfileData& combined, const negicc::InputCurves& curves,
                                              float e, float g, float b, float* q995)>;
// Steps 1 and 2 of run_white_light_fit below and its record: out.fit.curves anchored at out.params, which the caller
// composes into its profile (apply_input_curves, or compose_input_table per channel). out.params_before, out.tone and
// out.time_ms are the caller's. False (reported) when the data do not fit together.
bool fit_white_light_curves(const WhiteLightFitData& data, const negicc::CurveFitOptions& fit_opts,
                            const WhiteLightRolloff& rolloff, WhiteLightFitResult& out);

// The station's ConversionSession.set_fit_curves on the bracket a finished search chose (res), for white-light
// bundles, on the same frame as the search (the WhiteLightFit steps in multi_target_optimizer.cc):
//   0. prepare: the search's intent field (predicted again at the search's final parameters when the pullback or the
//      clip fallback moved them after its preview), the cells the fit uses (a fallback crop leaves out cells at the
//      film base's level or holder-dark), Auto's render and grid median L*, E's bounds (the bracket's calibrated range
//      widened to hold Auto's E) and the scene-neutral anchors (JointCurveModel::set_scene_neutral);
//   1. the joint fit (negicc::fit_input_curves_joint): g, b, a_G and a_B in one Levenberg-Marquardt solve on the
//      anchors' balance and the intent field's overall a* (the evidence shrinkage inside the residuals), each
//      candidate at the E that keeps Auto's grid median L*, and the never-greener guard checked on the result; its
//      curves anchored (ref, u_med) on the film base (the one opts gives, else the brightest grid cell per channel) at
//      its own E, g, b;
//   2. E re-metered on the real combined tables around the fit's E, so the grid's median L* at the fitted g and b
//      equals Auto's (no DINOv3 pass), each channel's curve scaled down where its shape fails at the re-metered
//      anchor; when the re-metered E leaves the bounds, or the shape still fails, Auto's own result with no curves
//      (WhiteLightFitResult::fallback). Steps 1 and 2 repeat (up to three fits) while the re-metered roll-off differs
//      from the one the fit ran with;
//   3. the bracket's G and B input tables composed with the curves, anchored at the result, in place
//      (apply_input_curves on bundle.targets[res.target_idx]);
//   4. record: the export's parameters, losses and neutral statistics, and with tone_head the tone stage's plan
//      (ToneStagePlan: the heads' predictions and the final render's preview) for the conversion to solve and apply
//      to the final image.
// Every step renders the grid as the export renders it (encoding: the export's OutputEncoding, ENCODE_SRGB or
// ENCODE_SRGB_LINEAR, which is what the fit models), with the roll-off measured as the export measures it.
// False (reported) on a bundle with a crosstalk matrix, a search without an intent field, or a failed inference.
bool run_white_light_fit(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                         double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                         const ParamSearchResult& res, const negicc::CurveFitOptions& fit_opts,
                         int encoding, const ToneHead* tone_head, WhiteLightFitResult& out);

#endif  // MULTI_TARGET_OPTIMIZER_H
