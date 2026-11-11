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

// neg_process's processing pipeline: the stages main runs in turn, the state they hand on, and the split between the
// scan as data (MosaicInput, RgbInput) and the compute on it (Backend: the CPU now; a CUDA backend can implement the
// same calls).
//
//   prepare   the profile (bundle, ICC profile or bracket), the parameter search mode, the crosstalk correction, the
//             DINOv3 engine
//   decode    the input as data: the zero-copy CFA path (a Sony Bayer RAW or a pixel shift: the mosaic, black
//             subtracted in place, with a half-size view over it) or LibRaw's decode, a TIFF, a JPEG or three
//             captures (an RGB image); the frame (--roi, --roi auto); and the bundle's parameter search on the
//             half-size view, as each path has it
//   begin_conversion
//             the film base the sidecar records, the pins (-E, --gain_g, --gain_b) and the input matrix to start from
//   render    the full-resolution image of the frame: the backend debayers or merges a mosaic; an RGB image is one
//             already
//   solve     the conversion parameters without the bundle search: the bracket, E, g and b (the single-profile
//             search, a --target bracket or given values), the knee and the shoulder
//   convert   the backend converts the image in place to the output encoding
//   write     the TIFF and the sidecar

#pragma once

#include <lcms2.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dinov3_auto_solver.h"
#include "dinov3_engine.h"
#include "film_base_detect.h"
#include "flat_field.h"
#include "flat_sample.h"
#include "libraw/libraw.h"
#include "multi_target_optimizer.h"
#include "neg_pipeline.h"
#include "np_options.h"
#include "np_output.h"
#include "np_raw.h"
#include "profile_bundle.h"

// ---- The white-light tone stage (np_pipeline.cc) ----
// The white-light tone stage: one monotone lightness curve on CIE L* (D65), set per frame from the tone heads
// (ToneHead, intent_head.h), applied to the final 16-bit sRGB image. Colour is not re-balanced: a* and b* are only
// scaled, by a factor chosen so the frame's mean chroma stays what it was before the curve.
//
//   f(L) = black + (white - black) * h(g(L / 100)),  g(x) = x^gamma,  h(x) = x^contrast / (x^contrast + (1 - x)^contrast)
//   a*, b* scaled by 1 + (chroma - 1) * smoothstep(3, 8, C*): near-neutral colours keep their balance, hue is kept.
//
// The four curve numbers are solved so the render's own descriptors move toward the heads' predictions: since f is
// monotone, the toned render's percentiles are f of the render's, so L2' = f(L2), L50' = f(L50), L98' = f(L98) and
// (L90 - L10)' = f(L90) - f(L10). Each target is the render's value moved a share of the way to the prediction:
// all the way for the shadow level L2 (kToneShadowMove), half way for the highlight level L98 and the tonal range
// L90 - L10 (kToneHighlightMove, kToneRangeMove), and not at all for the mid-tone L50, which the curve holds. A weak
// prior keeps the curve near the identity where the targets leave it free. The descriptors are measured on the
// network-size preview of the final render, as the heads' training targets were on lab scans.

// The shares of the way from the render's descriptor to the heads' prediction the curve moves.
constexpr double kToneShadowMove = 1.0;     // L2
constexpr double kToneHighlightMove = 0.5;  // L98
constexpr double kToneRangeMove = 0.5;      // L90 - L10 (the mid-tone L50 is held where it is)
// The curve's bounds (black, white, gamma, contrast) and the identity it starts from.
constexpr double kToneLo[4] = {-15.0, 80.0, 0.7, 0.7};
constexpr double kToneHi[4] = {15.0, 110.0, 1.45, 1.5};
// The prior's weight, and its scales: 5 L* for the black and white points, 0.1 in log for gamma and contrast.
constexpr double kTonePrior = 0.02;
// Above the render's L98 no target says where the curve goes, and quite different white points can fit the four targets
// about equally, so a small change of the render could switch between them. This weight holds the curve's rise above
// L98 to the input's, f(100) = f(L98) + (100 - L98) (slope 1 on average there), but never above white, f(100) <= 100,
// so a lifted L98 does not push the top past white into clipping. That makes the solution unique, and the solve starts
// from several white points and keeps the lowest cost, so it does not depend on which basin a single start falls into.
constexpr double kToneTopSlope = 0.07;
constexpr double kToneWhiteStarts[3] = {100.0, 110.0, 85.0};
// White clipping, at its two sources. A colour the curve lifts out of the sRGB gamut has its chroma scaled toward
// neutral at its L* (tone_lab_to_rgb_in_gamut) instead of its brightest channels clipped. Where the curve lifts L*
// itself to white on more than kToneClipAllowance more of the final image's pixels than the image had clipped, the
// highlight and range moves are scaled down to the largest of kToneClipShares that keeps it within the allowance,
// measured on every kToneClipSampleStep-th pixel each way of the final image. When none does, the clipping does not
// come from that move, and the moves stay as designed.
constexpr double kToneClipAllowance = 0.005;
constexpr double kToneClipShares[5] = {1.0, 0.75, 0.5, 0.25, 0.0};
constexpr int kToneClipSampleStep = 8;
// The chroma factor that keeps the mean chroma: refined this many times on the preview, within these bounds.
constexpr int kToneChromaRefinements = 2;
constexpr double kToneChromaLo = 0.5, kToneChromaHi = 1.6;

struct ToneCurve {
    double black = 0.0, white = 100.0, gamma = 1.0, contrast = 1.0, chroma = 1.0;
};

// The render's descriptors the curve is solved from: pixel L* percentiles and the mean chroma C*ab.
struct ToneDescriptors {
    double L2 = 0, L10 = 0, L25 = 0, L50 = 0, L75 = 0, L90 = 0, L98 = 0, Cmean = 0;
    double range() const { return L90 - L10; }
};

// The targets the curve is solved for (the render's descriptors moved toward the predictions).
struct ToneTargets {
    double L2 = 0, L50 = 0, L98 = 0, range = 0;
};

// What applying the stage to the full image did: the share of pixels clipped white (a channel at the maximum) and
// black (every channel 0), before and after.
struct ToneClip {
    double white_before = 0, white_after = 0, black_before = 0, black_after = 0;
};

// The descriptors of an 8-bit RGB image (n pixels, 3 bytes each); percentiles interpolate linearly between ranks.
ToneDescriptors tone_descriptors(const uint8_t* rgb, size_t n);

// The mean chroma of an 8-bit RGB image after the stage (toned in floating point, not re-quantised).
// (Summed in a fixed number of chunks, in parallel, then in order: the same result whatever the thread count.)
double tone_mean_chroma_after(const uint8_t* rgb, size_t n, const ToneCurve& q);

// The plan at a highlight share: the targets, the curve solved for them, and the chroma factor that keeps the
// preview's (8-bit RGB, n pixels) mean chroma at the render's.
ToneCurve tone_plan(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                    const uint8_t* preview, size_t n, double highlight_share, ToneTargets* targets);

// The plan's curve for a final image: the designed moves (designed, solved for designed_targets by tone_plan at
// share 1), or, where they would clip more than kToneClipAllowance beyond the image's own clipping, the largest share
// of the highlight and range moves that does not (see above). share gets the share kept.
ToneCurve tone_plan_for_image(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                              const uint8_t* preview, size_t n, const uint16_t* rgb, int w, int h, int stride,
                              const ToneCurve& designed, const ToneTargets& designed_targets, ToneTargets* targets,
                              double* share);

// rgb: interleaved 16-bit sRGB with `stride` uint16 per pixel (3 or 4), n pixels.
ToneClip apply_tone_stage(uint16_t* rgb, size_t n, int stride, const ToneCurve& q);
// The tone stage's plan as the conversion solves it after the curve fit (from the fit's ToneStagePlan: the heads'
// predictions and the final render's preview): the preview's descriptors, the targets, the curve between them and the
// chroma factor that keeps the mean chroma.
struct TonePlanCurve {
  ToneDescriptors render;         // the final render's preview before the stage
  ToneTargets targets;            // the descriptors the curve aims at
  ToneCurve curve;                // the curve and the chroma factor
  double chroma_after = 0.0;      // the preview's mean chroma with the curve and that factor (aim: render.Cmean)
};

// Cached Auto / DINOv3 metadata loaded from <output>.json when solving grey-point knobs.
struct CachedSidecar {
  bool is_auto = false;                  // true: cached from "auto"; false: cached from "dino"
  int target_idx = -1;
  std::string target_name;
  float base_e = 1.0f, base_g = 1.0f, base_b = 1.0f;
  std::string auto_json_str;             // raw "auto" JSON object from previous sidecar
  std::string dino_json_str;             // raw "dino" JSON object from previous sidecar
  std::string dino_backend;
  double t_dino_views_ms = 0.0, t_dino_render_ms = 0.0;
  double t_dino_infer_ms = 0.0, t_dino_search_ms = 0.0;
  // The previous run's fitted G/B input curves (replayed on its bracket, not refitted) and its intent field, carried
  // into the new sidecar so the knobs' output still records them.
  bool has_curves = false;
  negicc::InputCurves curves;
  std::string curves_source;
  std::string input_curves_json_str;
  std::string intent_field_json_str;
  std::string heads_json_str;
  // Its tint trim (input_curves.tint_trim) and tone curve (tone_stage.curve), when it applied them.
  bool has_tint_trim = false;
  TintTrimTables tint_trim;
  bool has_tone = false;
  ToneCurve tone_curve;
};

// What the profile and the options decide before the scan is read.
struct Setup {
  // Each input's size and CRC-32, taken once (by the sidecar loader when the knobs look for <output>.json) and
  // reused by the sidecar writer.
  std::vector<FileFingerprint> input_fingerprints;
  bool is_bundle = false;
  ProfileBundle bundle;
  std::string film_prof_arg;             // -p naming an ICC profile (not a bundle)
  bool has_profile = false;
  int manual_target_idx = -1;            // --target, -1 when not given
  bool exposure_given = false;           // -E: the exposure is not searched
  bool use_dino = false;                 // a parameter search runs
  bool bundle_search = false;            // the bundle's search (bracket, exposure and gains)
  bool fit_curves_active = false;        // the curve fit runs (--fit_curves auto or on, where it applies)
  bool has_cached_sidecar = false;       // <output>.json matched and supplies the base parameters
  CachedSidecar cached_sidecar;
  bool white_light = false;              // a rec2020_linear (d65) bundle
  bool has_crosstalk = false;
  bool camera_rgb_input = false;         // a TIFF of linear camera RGB (a TIFF with a camera_raw_rgb bundle)
  RawDecode raw_decode = RawDecode::kRec2020;
  std::vector<float> r_coeff, g_coeff, b_coeff;  // the crosstalk correction's rows
  negicc::FlatFieldPtr flat_field;
  DinoV3Engine engine;
  std::string dino_path;                 // the backbone that runs ([built-in] ...)
  std::string neutral_head;              // the neutral-patch head that runs ([built-in] ...), empty when none
  // The intent adapter, neutral head and tone heads that run ([built-in] <name> or the --dino path) and their files'
  // size and CRC-32, for the sidecar.
  std::string head_path[3];
  FileFingerprint head_fp[3];
  ToneHead tone_head;                    // the white-light tone heads (where the fit runs)
  // The tint trim and the tone curve the conversion renders with: this run's curve fit's, a previous run's (the knobs
  // on its sidecar) or given (--tint_trim theta:..., --tone_stage curve:...). Neither is searched or predicted again on
  // a replay. The trim belongs to bracket tint_trim_target_idx; tint_trim_base_prof is that bracket's profile without
  // it (curves composed, when there are curves), and tint_trim_composed says its tables carry it.
  bool has_tint_trim = false;
  TintTrimTables tint_trim;
  int tint_trim_target_idx = -1;
  bool tint_trim_composed = false;
  ParsedFilmProfile tint_trim_base_prof;
  // A replay: "knobs" (the knobs on a previous run's sidecar) or "cli" (recorded values given); empty otherwise. And
  // whether a recorded tint trim was left out because the conversion renders another bracket than the one it was made
  // on.
  std::string replay_source;
  bool tint_trim_other_bracket = false;
  bool has_tone = false;
  ToneCurve tone_curve;
  // Fitted G/B input curves the conversion's bracket renders through (input_curves.h): source "fit" (--fit_curves
  // on: curve_fit has the fit), "auto" (--fit_curves auto, likewise), "cli" (recorded curves given to --fit_curves) or
  // empty (none).
  std::string input_curves_source;
  negicc::InputCurves input_curves;
  WhiteLightFitResult curve_fit;
  TonePlanCurve tone_plan;  // the tone stage's plan solved from curve_fit.tone (when curve_fit.tone.planned)
  // The bracket the curves were composed into (-1: none) and its profile before that (the un-curved input tables),
  // from which the knobs compose the curves again at the E, g and b they move to (reanchor_input_curves).
  int curves_base_idx = -1;
  ParsedFilmProfile curves_base_prof;
};

// The scan as data for a backend: a CFA mosaic (one, or the four frames of a pixel shift), black subtracted, in place.
struct MosaicInput {
  const uint16_t* frames[4] = {nullptr, nullptr, nullptr, nullptr};
  int n_frames = 0;                      // 1, or 4 for a pixel shift
  size_t pitch = 0;                      // uint16_t elements per row
  int width = 0, height = 0;             // the visible sensor area
  unsigned filters = 0;                  // LibRaw's CFA pattern
  float white_level = 65535.0f;          // the clipping level above black
  negicc::FlatSampleArgs flat;           // the flat field at full size (empty without one)
};

// The scan as data for a backend: an RGB image, `channels` samples a pixel (3, or LibRaw's 4).
struct RgbInput {
  uint16_t* pixels = nullptr;
  int width = 0, height = 0, channels = 3;
};

// A region of the sensor (x, y, w, h) and its orientation: the frame to render.
struct Region {
  int x = 0, y = 0, w = 0, h = 0;
  int rot_cw = 0;
  bool hflip = false, vflip = false;
};

// The output encoding and its profiles.
struct OutputSpec {
  OutputEncoding encoding = ENCODE_NONE;
  std::string attach_profile;            // "srgb", "srgb-g10", an ICC file, or the -p profile
  cmsHPROFILE xyz = nullptr;             // ENCODE_CUSTOM: the XYZ input and the output profile
  cmsHPROFILE custom = nullptr;
};

// The compute on the scan's data.
class Backend {
 public:
  virtual ~Backend() = default;
  // The full-resolution RGB of the region of a mosaic, turned and flipped, into rgb_out (3 samples a pixel,
  // dst_w x dst_h: w x h, or h x w when turned by 90 or 270 degrees): the grain-aware debayer of one frame, or the
  // merge of a pixel shift's four, with the flat field.
  virtual void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, uint16_t* rgb_out) = 0;
  // Converts img in place to the output encoding through the shared pixel pipeline (color_pipeline.h) of pc: the
  // input matrix and knee, the film profile (tone curves and cLUT) and, for sRGB, the shoulder. ENCODE_NONE stops
  // after the knee; ENCODE_CUSTOM hands the profile's PCS XYZ to out.custom.
  virtual void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                       const RgbInput& img) = 0;
};

class CpuBackend : public Backend {
 public:
  void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, uint16_t* rgb_out) override;
  void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
               const RgbInput& img) override;
};

struct LibRawDeleter {
  void operator()(LibRaw* p) const;
};
using LibRawPtr = std::unique_ptr<LibRaw, LibRawDeleter>;

// The scan through the pipeline.
struct Scan {
  LibRawPtr proc;                        // the RAW (its metadata, and its image on the LibRaw path)
  LibRawPtr ps_extra[3];                 // a pixel shift's other three frames
  bool cfa = false;                      // the zero-copy CFA path
  MosaicInput mosaic;                    // on the CFA path
  Region cfa_region;                     // on the CFA path: the frame to render (sensor pixels)
  std::shared_ptr<const negicc::FlatTables> flat_full_tables;  // keeps mosaic.flat's tables alive
  std::vector<uint16_t> rgb3;            // on the CFA path: the rendered frame, 3 samples a pixel
  int orig_w = 0, orig_h = 0;            // the sensor size the sidecar records
  std::string input_source = "raw";      // what was read: "raw", "tiff:icc", "jpeg:srgb", "three:raw,raw,raw", ...

  FrameDetection frame_det;              // --roi auto
  int roi[4] = {0, 0, 0, 0};             // the frame in sensor pixels (--roi, or --roi auto's)
  bool has_roi = false;
  int rot_cw = 0;
  bool hflip = false, vflip = false;

  // The film base read from the scan (film_base_detect.h) for a bundle conversion, as negicc-station's Auto reads it
  // when no film base is given; not found: the bundle's.
  negicc::FilmBaseDetection film_base;
  const double* film_base_rgb() const { return film_base.found ? film_base.rgb : nullptr; }
  // The white-light decode's gains on the camera channels (ChannelBalance); 1, 1, 1: the mosaic was left as it was.
  float balance_gains[3] = {1, 1, 1};

  // Each bracket's exposure range (compute_bundle_enclosures) was measured on the half-size frame, in decode.
  bool has_enclosures = false;
  // Each bracket's exposure range in the station's E, as the bundle search meters it (meter_bracket_exposures).
  std::vector<BracketExposure> bracket_exposure;
  bool has_auto_res = false;             // the bundle search ran
  ParamSearchResult auto_res;
  int auto_frame_half[4] = {0, 0, 0, 0}; // the frame (x, y, w, h) in the half-size image the search saw
  // Where the fit runs: the curve fit's re-metered E and fitted g and b on the combined profile, which the
  // conversion takes in place of the search's (auto_res keeps the search's own).
  bool has_curve_params = false;
  ConversionParams curve_params;
  const ConversionParams& params() const { return has_curve_params ? curve_params : auto_res.params; }
  // The curves re-anchored at the knobs' E, g and b, which the bracket renders through after the knobs and the
  // cli_equivalent replays (Setup::input_curves stays the record at the knobs' base, as the sidecar keeps it).
  bool has_knob_curves = false;
  negicc::InputCurves knob_curves;

  bool has_knob_res = false;             // --brightness, --tint or --warmth solved
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;

  // The highlight roll-off of every bundle conversion, measured on the half-size frame as the bracket search measures
  // it: Auto's result and given values in decode, while that frame exists (render frees the mosaic; LibRaw's
  // half-size decode is analysed and gone); the single-bracket search in solve, at its final E, g, b, on
  // rolloff_frame (a copy of the frame, interleaved RGB, with the scan's exposure).
  bool has_rolloff = false;
  float rolloff_x_hi = 1.0f, rolloff_q995 = 0.0f;
  // The white-light tone stage as convert applied it to the final image (WhiteLightFitResult::tone has the plan).
  bool tone_applied = false;
  ToneClip tone_clip;
  ToneCurve tone_curve;            // the curve applied (the plan's, capped on this image; or the recorded one)
  ToneTargets tone_targets;        // with this run's plan: the targets it was solved for
  double tone_share = 1.0;         // and the share of the highlight and range moves kept
  std::vector<uint16_t> rolloff_frame;
  int rolloff_frame_w = 0, rolloff_frame_h = 0;
  double rolloff_shutter_s = 0.125, rolloff_iso = 100.0;

  double t_raw_0 = 0.0;                  // timings (ms) the sidecar records
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_auto_ms = 0.0;
  // The bracket analysis on the half-size frame outside Auto's time: the enclosures, the knobs, the roll-off and the
  // frame's copy for the single-bracket search (decode), and that search's roll-off and knobs (solve).
  double t_analysis_ms = 0.0;
  double t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;

  // The scan's exposure: EXIF (of exposure_file, the first input), else LibRaw's, else 1/8 s at ISO 100 (warned once).
  std::string exposure_file;
  bool has_exif_exp = false;
  double exif_shutter_d = 0.0, exif_iso_d = 0.0;
  bool warned_default_exposure = false;
  void scan_exposure(const LibRaw& r, double& shutter_out, double& iso_out);

  // The full-resolution image the conversion runs on: rgb3 on the CFA path, LibRaw's image otherwise.
  RgbInput image();
};

// The conversion parameters: what solve decides, convert applies and write records.
struct Conversion {
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;
  unsigned pin_mask = 0;
  float merged_matrix[9] = {0};
  float knee = 0.95f;
  ShoulderInfo shoulder;
  ParsedFilmProfile film_prof;
  std::vector<uint8_t> embedded_icc;     // the bracket's ICC profile
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  DinoSolveResult dino_res;              // the single-profile search
  // With a bundle: E, g and b on the station's scale (what -E, --gain_g and --gain_b take), for cli_equivalent.
  std::optional<std::array<float, 3>> station_params;
  // The film base the sidecar records. A bundle conversion takes the base read from the scan (Scan::film_base), else
  // the bundle's, through the station's merged matrix; the film base scale of the input matrix stays 1.
  std::vector<int> profile_film_base_rgb = {1, 1, 1};
  std::vector<int> sidecar_profile_film_base_rgb;
  std::vector<int> sidecar_film_base_rgb = {1, 1, 1};
  std::string film_base_source = "none";
  OutputSpec out;
};

int prepare(const Options& o, Setup& s);
int decode(const Options& o, Setup& s, Scan& scan);
int begin_conversion(const Options& o, const Setup& s, Conversion& c);
int render(const Options& o, const Setup& s, Scan& scan, Backend& backend);
int solve(const Options& o, Setup& s, Scan& scan, Conversion& c);
int convert(const Options& o, const Setup& s, Scan& scan, Conversion& c, Backend& backend);
int write(const Options& o, const Setup& s, Scan& scan, Conversion& c);
