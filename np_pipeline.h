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
#include "film_base_detector.h"
#include "flat_field.h"
#include "libraw/libraw.h"
#include "multi_target_optimizer.h"
#include "np_options.h"
#include "np_output.h"
#include "profile_bundle.h"
#include "tone_stage.h"

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
  // Its tone curve (tone_stage.curve), when it applied one.
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
  // The bundle's search with --target, -E, --gain_g or --gain_b held (white light): Auto with those pinned.
  bool pinned_search = false;
  bool fit_curves_active = false;        // the curve fit runs (--fit_curves auto or on, where it applies)
  bool tone_active = false;              // the tone stage runs (--tone_stage auto or on, where it applies)
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
  ToneHead tone_head;                    // the tone heads (where the tone stage runs)
  // The tone curve the conversion renders with: this run's tone stage's, a previous run's (the knobs on its sidecar) or
  // given (--tone_stage curve:...); not predicted again on a replay.
  // A replay: "knobs" (the knobs on a previous run's sidecar) or "cli" (recorded values given); empty otherwise.
  std::string replay_source;
  bool has_tone = false;
  ToneCurve tone_curve;
  // Fitted G/B input curves the conversion's bracket renders through (input_curves.h): source "fit" (--fit_curves
  // on: curve_fit has the fit), "auto" (--fit_curves auto, likewise), "cli" (recorded curves given to --fit_curves) or
  // empty (none).
  std::string input_curves_source;
  negicc::InputCurves input_curves;
  WhiteLightFitResult curve_fit;
  // This run's tone stage: the heads' plan (the curve fit's on its result, or plan_auto_tone_stage's on the search's
  // own result where no fit runs) and the curve solved from it (when tone_stage.planned). tone_stage_ran: a plan was
  // attempted this run (the sidecar then says what came of it).
  ToneStagePlan tone_stage;
  bool tone_stage_ran = false;
  TonePlanCurve tone_plan;
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

  // The film base read from the scan (film_base_detector.h) for a bundle conversion, as negicc-station's Auto reads it
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
  // The tone stage as convert applied it to the final image (Setup::tone_stage has the plan).
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
