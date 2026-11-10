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

#include "np_log.h"
#include "np_options.h"

#include <stdio.h>
#include <stdlib.h>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "argparse/argparse.hpp"
#include "embedded_data.h"
#include "libraw/libraw.h"
#include "pixel_shift_group.h"
#include "rgb_input.h"

// Four files of a 4-shot pixel shift: distinct, one group of four in shot order by the camera's group tags or, without
// them, by their file counters. False with the error reported otherwise.
static bool check_pixel_shift_files(const std::vector<std::string>& files) {
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i + 1; j < 4; ++j) {
      std::error_code ec;
      if (files[i] == files[j] ||
          (std::filesystem::exists(files[i], ec) && std::filesystem::exists(files[j], ec) &&
           std::filesystem::equivalent(files[i], files[j], ec))) {
        log_error("4-shot pixel shift requires 4 distinct RAW files (duplicate: %s)\n",
                files[i].c_str());
        return false;
      }
    }
  }
  // The camera's group tags decide when the files carry them: one group of four, in shot order. Without tags the
  // file counters must increase from frame to frame, modulo the counter's wrap (DCF numbers files 0001-9999, so a
  // Sony DSC09999 is followed by DSC00001): each step forward, by less than half the counter's range.
  PixelShiftTags group_tags[4];
  for (size_t i = 0; i < 4; ++i) {
    LibRaw probe;
    if (probe.open_file(files[i].c_str()) != LIBRAW_SUCCESS) continue;  // the load reports it
    const auto& sony = probe.imgdata.makernotes.sony;
    group_tags[i].group_id = sony.PixelShiftGroupID;
    group_tags[i].shots_in_group = sony.nShotsInPixelShiftGroup;
    group_tags[i].num_in_group = sony.numInPixelShiftGroup;
  }
  std::string group_err;
  const PixelShiftGroupCheck group = check_pixel_shift_group(group_tags, group_err);
  if (group == PixelShiftGroupCheck::kBad) {
    log_error("%s\n", group_err.c_str());
    return false;
  }
  int trailing_idx[4] = {-1, -1, -1, -1};
  int idx_mod = 1;
  bool all_have_idx = group == PixelShiftGroupCheck::kNoTags;
  for (size_t i = 0; all_have_idx && i < 4; ++i) {
    const std::string stem = std::filesystem::path(files[i]).stem().string();
    size_t p = stem.size();
    while (p > 0 && std::isdigit((unsigned char)stem[p - 1])) --p;
    if (p == stem.size() || stem.size() - p > 6) {
      all_have_idx = false;
      break;
    }
    trailing_idx[i] = std::stoi(stem.substr(p));
    int mod = 1;
    for (size_t d = p; d < stem.size() && mod < 10000; ++d) mod *= 10;
    idx_mod = std::max(idx_mod, mod);
  }
  bool in_order = true;
  for (int i = 0; all_have_idx && i < 3; ++i) {
    const int step = ((trailing_idx[i + 1] - trailing_idx[i]) % idx_mod + idx_mod) % idx_mod;
    if (step == 0 || 2 * step >= idx_mod) in_order = false;
  }
  if (all_have_idx && !in_order) {
    log_error("4-shot pixel shift files are out of frame order\n");
    return false;
  }
  return true;
}

// --roi: 'auto [h|v]', or x y w h [rot_cw [hflip [vflip]]]. False with the error reported.
static bool parse_roi(const std::vector<std::string>& roi_tokens, Options& o) {
  if (roi_tokens.empty()) {
    log_error("--roi requires arguments\n");
    return false;
  }
  if (roi_tokens[0] == "auto") {
    o.roi_auto = true;
    if (roi_tokens.size() > 1) {
      if (roi_tokens[1] == "h" || roi_tokens[1] == "H") {
        o.roi_axis_hint = 'h';
      } else if (roi_tokens[1] == "v" || roi_tokens[1] == "V") {
        o.roi_axis_hint = 'v';
      } else {
        log_error("Invalid --roi auto axis hint '%s' (must be 'h' or 'v')\n", roi_tokens[1].c_str());
        return false;
      }
    }
    if (roi_tokens.size() > 2) {
      log_error("Too many arguments for --roi auto\n");
      return false;
    }
    return true;
  }
  if (roi_tokens.size() < 4) {
    log_error("--roi requires at least 4 arguments: x y w h [rot_cw [hflip [vflip]]]\n");
    return false;
  }
  try {
    for (int i = 0; i < 4; ++i) o.roi[i] = std::stoi(roi_tokens[i]);
    if (roi_tokens.size() >= 5) o.rot_cw = std::stoi(roi_tokens[4]);
    if (roi_tokens.size() >= 6) o.hflip = (std::stoi(roi_tokens[5]) != 0);
    if (roi_tokens.size() >= 7) o.vflip = (std::stoi(roi_tokens[6]) != 0);
    o.has_roi = true;
  } catch (const std::exception& e) {
    log_error("Invalid integer argument for --roi\n");
    return false;
  }
  if (o.roi[0] < 0 || o.roi[1] < 0 || o.roi[2] <= 0 || o.roi[3] <= 0) {
    log_error("Invalid --roi [%d, %d, %d, %d]: x and y must be >= 0 and w and h must be > 0\n",
            o.roi[0], o.roi[1], o.roi[2], o.roi[3]);
    return false;
  }
  return true;
}

int parse_options(int ac, char* av[], Options& o) {
  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--list_profiles") == 0) {
      for (const std::string& key : list_embedded_bundles()) {
        // Keys are "<Stock>-<camera>-<Light>".
        std::string stock = key.substr(0, key.find('-'));
        for (char& ch : stock) ch = (char)std::tolower((unsigned char)ch);
        const std::string light = key.substr(key.find_last_of('-') + 1);
        log_info("--film_stock %s --light_source %s  (%s)\n", stock.c_str(),
               light == "TriLED" ? "tri" : light == "D65" ? "d65" : light.c_str(), key.c_str());
      }
      return -1;
    }
  }
  argparse::ArgumentParser parser("neg_process");
  parser.add_argument("-C", "--no_crop")
    .help("No cropping according to aspect ratio in RAW file.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-q", "--quality")
    .help("De-bayer quality (-1 for auto: grain-aware if Sony CFA, else bilinear 0; >=0 passes qual to LibRaw, e.g. 4 for DCB).")
    .scan<'i', int>()
    .default_value(-1);
  parser.add_argument("-r", "--r_coeff")
    .help("R (corrected) value is dot product of this 'r1 r2 r3' vector and 'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none).")
    .nargs(3)
    .default_value(std::vector<float>{1, 0, 0})
    .scan<'g', float>();
  parser.add_argument("-g", "--g_coeff")
    .help("G (corrected) value is dot product of this 'g1 g2 g3' vector and 'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none).")
    .nargs(3)
    .default_value(std::vector<float>{0, 1, 0})
    .scan<'g', float>();
  parser.add_argument("-b", "--b_coeff")
    .help("B (corrected) value is dot product of this 'b1 b2 b3' vector and 'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none).")
    .nargs(3)
    .default_value(std::vector<float>{0, 0, 1})
    .scan<'g', float>();
  parser.add_argument("-E", "--exposure_comp")
    .help("Exposure. With a bundle, E as negicc-station and the parameter search give it (the sidecar's "
          "cli_equivalent: auto.e, the search's own; with fitted curves, input_curves.remeter.e, re-metered on the "
          "combined profile; after the knobs, knobs.egb.e): "
          "relative to the bracket, the film base and the scan's exposure; with an ICC profile, a multiplier of the "
          "corrected RGB. Given, the parameter search keeps it and solves only the gains, on the bracket --target "
          "picks (or the bundle's default). Not given with given gains: with a bundle, the centre of the bracket's "
          "exposure range as the bundle search meters it; with an ICC profile, 1.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--no-metadata")
    .help("Disable automatic JSON metadata sidecar generation.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--gain_g")
    .help("Gain of the G channel relative to R (with a bundle, as negicc-station gives it: 1 renders the film base "
          "neutral). Defaults to 1.0. Given, it turns the parameter search off.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--gain_b")
    .help("Gain of the B channel relative to R (with a bundle, as negicc-station gives it: 1 renders the film base "
          "neutral). Defaults to 1.0. Given, it turns the parameter search off.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--brightness")
    .help("Brightness offset after the parameter search, in stops of the reference grey's luminance (as in "
          "negicc-station). Reuses the Auto or DINOv3 result in <output>.json when this build wrote it from the same input "
          "files (size and CRC-32) and the same arguments apart from -o and the knobs; otherwise searches again.")
    .scan<'g', float>()
    .default_value(0.0f);
  parser.add_argument("--tint")
    .help("Tint offset after the parameter search, in CIELAB a* units of the reference grey (negative: green, "
          "positive: magenta, as in negicc-station). Reuses <output>.json when it matches.")
    .scan<'g', float>()
    .default_value(0.0f);
  parser.add_argument("--warmth")
    .help("Warmth offset after the parameter search, in CIELAB b* units of the reference grey (negative: blue, "
          "positive: yellow, as in negicc-station). Reuses <output>.json when it matches.")
    .scan<'g', float>()
    .default_value(0.0f);
  // A bundle conversion reads the film base from the scan (film_base_detect.h: the rebate, else the thinnest film), as
  // negicc-station's Auto does; --film_base_rgb, which took it as numbers or from the rebate, is gone.
  // TODO: read the film base from an image (a capture of unexposed film, as negicc-station's Film Base tab does).
  parser.add_argument("--film_stock")
    .help("Film stock of the built-in profile bundle and intent adapter: ektar100, portra160 or portra400. Without it "
          "(-p with an ICC profile or a bundle whose name names no stock with an adapter), the parameter search falls "
          "back to the portra400 intent adapter.");
  parser.add_argument("--light_source")
    .help("Scanning light of the built-in profile bundle: tri (narrowband red, green and blue LED; sensor RGB and its "
          "crosstalk correction) or d65 (white; LibRaw's white-light decode to linear Rec.2020). Required for a RAW; "
          "otherwise it follows the input: d65 for a linear DNG or a JPEG, tri for three captures. A TIFF takes either: "
          "with d65 (the default) it is converted to linear Rec.2020 through its ICC profile (else 8 bits are sRGB and "
          "16 bits linear Rec.2020); with tri it is the scanning camera's linear RGB under the TriLED light (black "
          "subtracted, no white balance or colour matrix), converted with the bundle's crosstalk correction.");
  parser.add_argument("-p", "--film_profile")
    .help("Profile in place of the built-in bundle: a negicc-station profile bundle JSON or a split bundle's "
          "directory (e.g. to test parity with negicc-station), or a single ICC profile that applies to the corrected "
          "RGB values (for research).");
  parser.add_argument("-P", "--colorspace")
    .help("srgb, srgb-g10 or [ICC profile path]. If specified the corrected RGB will be converted using this as the output profile.");
  parser.add_argument("--roi")
    .help("ROI crop in sensor coordinates: x y w h [rot_cw [hflip [vflip]]], or 'auto [h|v]'. 'auto' reads the film "
          "base around a negative's frame; a positive (slide) scan has none, so it usually falls back to the full "
          "visible frame (geometry.roi_source auto_fallback).")
    .nargs(1, 7);
  parser.add_argument("--film_format")
    .help("Film format: 35mm, 645, 6x6, 6x7, 6x9, or auto (the frame detector tries every format). With --roi auto it "
          "is the frame to find (default 35mm); with a given frame, when given, it is the frame the film base is read "
          "around (the rebate; default: every format, as negicc-station).")
    .default_value(std::string("35mm"));
  parser.add_argument("--rot", "--rot_cw")
    .help("Clockwise rotation in degrees (0, 90, 180, 270).")
    .scan<'i', int>()
    .default_value(0);
  parser.add_argument("--hflip")
    .help("Horizontal flip.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--vflip")
    .help("Vertical flip.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-o", "--output")
    .required()
    .help("Output file location.");
  parser.add_argument("--dino")
    .help("Intent adapter for the parameter search in place of the film stock's built-in one: adapter.onnx or a "
          "directory holding it.");
  parser.add_argument("--list_profiles")
    .help("List the film stocks and light sources of the built-in profile bundles and exit.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--neutral_head")
    .help("Neutral-patch head of the white-light curve fit: on (default, the built-in "
          "models/neutral_head/neutral_head.bin), off, or the path of another neutral_head.bin. Its p(neutral) "
          "weights the fit's scene-neutral anchors (p squared, every cell); off, the fit has no anchors and keeps "
          "Auto's colour.")
    .default_value(std::string("on"));
  parser.add_argument("--offset_scan")
    .help("The white-light curve fit's last step: on (default) or off. On, the network judges whether the fitted "
          "picture reads too green or too magenta: a lattice of magenta-green offsets is scored against the intent "
          "field, the best few are rendered and scored again (up to 4 extra DINOv3 inferences), and the best one, "
          "moved one standard error toward magenta, is composed into the bracket's tables. Offsets that would make the "
          "picture greener than Auto are not candidates.")
    .default_value(std::string("on"));
  parser.add_argument("--tone_stage")
    .help("The white-light tone stage: on (default) or off. On, tone heads on DINOv3 features the fit already "
          "computed predict the frame's shadow level, highlight level and tonal range as a lab scan would have them, "
          "and one monotone L* curve moves the final sRGB image all the way to the predicted shadow level and half "
          "way to the highlight level and range, holding the mid-tones and the mean chroma. Runs where the curve fit "
          "runs, with -P srgb.")
    .default_value(std::string("on"));
  parser.add_argument("--dino_backend")
    .help("DINOv3 inference backend: auto, cpu (USE_CUDA=0 builds) or gpu (USE_CUDA=1 builds); defaults to auto.")
    .default_value(std::string("auto"));
  parser.add_argument("--dino_iters")
    .help("Maximum iterations for DINOv3 parameter solving: the fixed-point steps of each bracket in the parameter "
          "search (defaults to 8), else the solver's iterations (defaults to 4).")
    .scan<'i', int>()
    .default_value(4);
  parser.add_argument("--target", "--bundle_target")
    .help("Target index (0-based) or target name in JSON profile bundle.")
    .default_value(std::string(""));
  // Fitted G/B input curves (input_curves.h; negicc-station designs/plan_fit_curves.md).
  parser.add_argument("--fit_curves")
    .help("auto (default), off, on, or recorded curves: ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB (one value, comma-"
          "separated; ten numbers, with u_loG,u_loB appended, for a curve held below u_lo). auto: as on wherever on "
          "applies (a white-light bundle search with -P srgb or srgb-g10), and nothing elsewhere (TriLED, another or "
          "no -P, given parameters), so it can be passed for every scan. on: after the bundle search, fit g, b and G "
          "and B input curves jointly (Levenberg-Marquardt) to the search's intent field on the bracket it chose, each "
          "candidate at the E that keeps Auto's grid median L*, then re-meter E on that bracket's profile with its "
          "input tables composed with the curves (the combined profile), as negicc-station's Fit curves toggle does; "
          "needs the bundle search (no --target, -E, --gain_g or --gain_b) and -P srgb or srgb-g10. Recorded curves: "
          "convert through the --target bracket's combined profile with them at the given -E, --gain_g and --gain_b, "
          "without fitting or searching (a sidecar's input_curves; its cli_equivalent writes them in this form with "
          "the bracket, E, g and b); needs --target, -E, --gain_g and --gain_b. The form leads with the refs, which "
          "are positive, because the parser takes a value starting with '-' for an option. The older order "
          "aG,kG,aB,kB,refG,refB,u_medG,u_medB[,u_loG,u_loB] (no prefix) is still read when aG is not negative. "
          "White-light (d65) bundles and -P only.")
    .default_value(std::string("auto"));
  // TODO: --flat from an image file (a RAW of the empty gate, as negicc-station's Flat Field tab captures), not only a
  // flat field .npz measured from one.
  parser.add_argument("--flat")
    .help("Scan flat-field .npz file (measured on the empty gate), applied in sensor coordinates.");
  parser.add_argument("raw_files").nargs(1, 4);

  try {
    parser.parse_args(ac, av);
  } catch (const std::exception& err) {
    std::ostringstream help;
    help << parser;
    log_error("%s\n%s", err.what(), help.str().c_str());
    return 1;
  }
  o.argc = ac;
  o.argv = av;

  o.files = parser.get<std::vector<std::string>>("raw_files");
  const auto& files = o.files;
  if (files.size() == 2 || files.size() > 4) {
    log_error("Give one RAW, DNG, TIFF or JPEG file, three captures (one per light: red, green, blue) "
                    "or the four RAW files of a pixel shift capture\n");
    return 1;
  }
  o.input_mode = files.size() == 3 ? InputMode::kThreeShot
                 : files.size() == 1 && (negicc::is_jpeg_file(files[0]) || !libraw_opens(files[0]))
                     ? InputMode::kRgbFile
                 : files.size() == 1 && is_linear_dng(files[0]) ? InputMode::kLinearDng
                                                                : InputMode::kRaw;
  o.tiff_input = o.input_mode == InputMode::kRgbFile && !negicc::is_jpeg_file(files[0]);
  // The scanning light the input implies, for the built-in bundle when --light_source is not given: a linear DNG, a
  // JPEG or a TIFF is white light, three captures one per light are TriLED. A TIFF also takes tri (camera RGB).
  o.implied_light = o.input_mode == InputMode::kRaw ? nullptr
                    : o.input_mode == InputMode::kThreeShot ? "tri"
                                                            : "d65";
  if (o.input_mode != InputMode::kRaw && parser.is_used("--flat")) {
    log_error("--flat applies to a single CFA RAW or a pixel shift capture\n");
    return 1;
  }
  if (files.size() == 4 && !check_pixel_shift_files(files)) return 1;
  o.dino_iters = parser.get<int>("--dino_iters");
  o.dino_iters_given = parser.is_used("--dino_iters");
  if (o.dino_iters <= 0) {
    log_error("--dino_iters must be positive (got %d)\n", o.dino_iters);
    return 1;
  }
  o.r_coeff = parser.get<std::vector<float>>("--r_coeff");
  o.g_coeff = parser.get<std::vector<float>>("--g_coeff");
  o.b_coeff = parser.get<std::vector<float>>("--b_coeff");
  o.row_given[0] = parser.is_used("--r_coeff");
  o.row_given[1] = parser.is_used("--g_coeff");
  o.row_given[2] = parser.is_used("--b_coeff");
  o.exposure_comp = parser.get<float>("--exposure_comp");
  o.exposure_given = parser.is_used("-E") || parser.is_used("--exposure_comp");
  o.gain_g = parser.get<float>("--gain_g");
  o.gain_b = parser.get<float>("--gain_b");
  o.gain_g_given = parser.is_used("--gain_g");
  o.gain_b_given = parser.is_used("--gain_b");
  o.brightness = parser.get<float>("--brightness");
  o.tint = parser.get<float>("--tint");
  o.warmth = parser.get<float>("--warmth");
  o.knobs_given = parser.is_used("--brightness") || parser.is_used("--tint") || parser.is_used("--warmth");
  if (!std::isfinite(o.brightness) || !std::isfinite(o.tint) || !std::isfinite(o.warmth)) {
    log_error("--brightness, --tint and --warmth must be finite numbers\n");
    return 1;
  }

  if (parser.is_used("--roi") && !parse_roi(parser.get<std::vector<std::string>>("--roi"), o)) return 1;
  if (parser.is_used("--rot")) {
    o.rot_cw = parser.get<int>("--rot");
  } else if (parser.is_used("--rot_cw")) {
    o.rot_cw = parser.get<int>("--rot_cw");
  }
  if (parser.get<bool>("--hflip")) o.hflip = true;
  if (parser.get<bool>("--vflip")) o.vflip = true;
  if (o.rot_cw % 90 != 0) {
    log_error("Rotation must be a multiple of 90 degrees (got %d)\n", o.rot_cw);
    return 1;
  }

  // The film format of the frame detector: --roi auto's search (35mm unless given, as negicc-station and the recorded
  // parity cases), and, when given, the film base's frame detection with a given frame. Before, it was refused without
  // --roi auto, so the film base's detection with a given --roi could never be told the format; and auto (every
  // format) could not be asked for.
  o.film_format_given = parser.is_used("--film_format");
  const std::string film_format_arg = parser.get<std::string>("--film_format");
  o.film_format = film_format_from_name(film_format_arg);
  if (o.film_format == FILM_FORMAT_AUTO && film_format_arg != "auto") {
    log_error("Invalid --film_format '%s' (must be 35mm, 645, 6x6, 6x7, 6x9, or auto)\n", film_format_arg.c_str());
    return 1;
  }

  if (parser.is_used("--flat")) o.flat = parser.get<std::string>("--flat");
  if (parser.is_used("--film_stock")) o.film_stock = parser.get<std::string>("--film_stock");
  if (parser.is_used("--light_source")) o.light_source = parser.get<std::string>("--light_source");
  if (parser.is_used("--film_profile")) o.film_profile = parser.get<std::string>("--film_profile");
  o.target_given = parser.is_used("--target");
  if (o.target_given) o.target = parser.get<std::string>("--target");
  {
    // One value, so the flag never takes the RAW file that follows it (with nargs it did: "--fit_curves on in.ARW").
    // Recorded curves lead with "ref:" and the refs (refG,refB,u_medG,u_medB,aG,kG,aB,kB[,u_loG,u_loB]): the older
    // order led with aG, and a negative aG made the value start with '-', which argparse rejects as an unknown option
    // (with "--fit_curves=" too), so a sidecar's cli_equivalent did not parse. The older order is still read.
    std::string fcv = parser.get<std::string>("--fit_curves");
    const bool ref_first = fcv.compare(0, 4, "ref:") == 0;
    if (ref_first) fcv.erase(0, 4);
    std::vector<std::string> fc;
    for (size_t p0 = 0;;) {
      const size_t p1 = fcv.find(',', p0);
      fc.push_back(fcv.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0));
      if (p1 == std::string::npos) break;
      p0 = p1 + 1;
    }
    if (!ref_first && fc.size() == 1 && (fc[0] == "on" || fc[0] == "off" || fc[0] == "auto")) {
      o.fit_curves = fc[0] == "on";
      o.fit_curves_auto = fc[0] == "auto";
    } else if (fc.size() == 8 || fc.size() == 10) {
      // Ten: the curve held below u_lo (InputCurves::hold), with u_loG and u_loB.
      float v[10];
      for (size_t i = 0; i < fc.size(); ++i) {
        char* end = nullptr;
        v[i] = std::strtof(fc[i].c_str(), &end);
        if (fc[i].empty() || !end || *end != '\0' || !std::isfinite(v[i])) {
          log_error("--fit_curves: '%s' is not a number (give off, auto, on, or "
                    "ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB with u_loG,u_loB appended for a held curve)\n",
                    fc[i].c_str());
          return 1;
        }
      }
      o.input_curves_given = true;
      // Where each number sits: ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB, or the older aG,kG,aB,kB,refG,refB,u_medG,
      // u_medB; u_loG,u_loB last in both.
      const int at_ref = ref_first ? 0 : 4, at_u_med = ref_first ? 2 : 6, at_ak = ref_first ? 4 : 0;
      for (int i = 0; i < 2; ++i) {
        o.input_curves.a[i] = v[at_ak + i * 2];
        o.input_curves.k[i] = v[at_ak + i * 2 + 1];
        o.input_curves.ref[i] = v[at_ref + i];
        o.input_curves.u_med[i] = v[at_u_med + i];
        if (fc.size() == 10) o.input_curves.u_lo[i] = v[8 + i];
        o.input_curves.hold = fc.size() == 10;
        if (!(o.input_curves.ref[i] > 0.0f)) {
          log_error("--fit_curves: refG and refB must be positive\n");
          return 1;
        }
      }
    } else {
      log_error("--fit_curves takes off, auto, on, or recorded curves: ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB (eight "
                "comma-separated numbers), ten for a held curve (and u_loG,u_loB)\n");
      return 1;
    }
  }
  if (parser.is_used("--dino")) o.dino_adapter = parser.get<std::string>("--dino");
  o.dino_backend = parser.get<std::string>("--dino_backend");
  o.neutral_head = parser.get<std::string>("--neutral_head");
  auto on_off = [&](const char* name, bool& flag) {
    const std::string v = parser.get<std::string>(name);
    if (v != "on" && v != "off") {
      log_error("%s takes on or off, not '%s'\n", name, v.c_str());
      return false;
    }
    flag = v == "on";
    return true;
  };
  if (!on_off("--offset_scan", o.offset_scan) || !on_off("--tone_stage", o.tone_stage)) return 1;
  o.quality = parser.get<int>("--quality");
  o.no_crop = parser.get<bool>("--no_crop");
  o.output = parser.get<std::string>("--output");
  if (parser.is_used("--colorspace")) o.colorspace = parser.get<std::string>("--colorspace");
  o.no_metadata = parser.get<bool>("--no-metadata");
  return 0;
}
