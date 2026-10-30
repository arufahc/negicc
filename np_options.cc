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

#include "np_log.h"
#include "np_options.h"

#include <stdio.h>
#include <stdlib.h>

#include <cctype>
#include <cmath>
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
    .help("De-bayer quality (-1 for auto: grain-aware if Sony CFA, else bilinear; >=0 passes qual to LibRaw).")
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
    .help("Multiplier for RGB values. Given, the parameter search keeps it and solves only the gains, on the bracket "
          "--target picks (or the bundle's default).")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--shoulder")
    .help("Highlight shoulder: 'off', 'auto' (the parameter search measures it from the pre-shoulder render and the "
          "export applies it), or x_hi in [1, 1.5]: the 99.5% point of the render's max channel in linear sRGB, "
          "where the chart white is 1 (a linear value, not stops: 1.3 is +0.38 stop). 1 is off. The shoulder rolls "
          "the range above its knee into the ceiling at x_hi. It only applies to the sRGB output; srgb-g10 "
          "(scene-referred) and ICC outputs never apply it. Defaults to off.")
    .default_value(std::string("off"));
  parser.add_argument("--no-metadata")
    .help("Disable automatic JSON metadata sidecar generation.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--gain_g")
    .help("Additional gain multiplier for the G channel. Defaults to 1.0. Given, it turns the parameter search off.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--gain_b")
    .help("Additional gain multiplier for the B channel. Defaults to 1.0. Given, it turns the parameter search off.")
    .scan<'g', float>()
    .default_value(1.0f);
  // TODO: read the film base from an image (a capture of unexposed film, as negicc-station's Film Base tab does), in
  // place of the profile's; --film_base_rgb, which took it as numbers or from the rebate (--roi auto), is removed.
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
    .help("Film format for --roi auto: 35mm, 645, 6x6, 6x7, 6x9, or auto (each in turn; less accurate).")
    .default_value(std::string("auto"));
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
  parser.add_argument("--dino_backend")
    .help("DINOv3 inference backend: auto, cpu (USE_CUDA=0 builds) or gpu (USE_CUDA=1 builds); defaults to auto.")
    .default_value(std::string("auto"));
  parser.add_argument("--dino_iters")
    .help("Maximum iterations for DINOv3 parameter solving: the fixed-point steps of each bracket in the parameter "
          "search (defaults to 8), else the solver's iterations (defaults to 4).")
    .scan<'i', int>()
    .default_value(4);
  parser.add_argument("--enclosure_rescue")
    .help("Enable Part 1B of the parameter search, the enclosure-guided bracket rescue that re-screens the bracket "
          "pool when the chosen bracket still clips highlights (off by default).")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--target", "--bundle_target")
    .help("Target index (0-based) or target name in JSON profile bundle.")
    .default_value(std::string(""));
  // TODO: --flat from an image file (a RAW of the empty gate, as negicc-station's Flat Field tab captures), not only a
  // flat field .npz measured from one.
  parser.add_argument("--flat")
    .help("Scan flat-field .npz file (measured on the empty gate), applied in sensor coordinates.");
  parser.add_argument("--station_egb")
    .help("Conversion parameters E, g and b as negicc-station applies them (multi_target_optimizer.h): the bundle's "
          "merged matrix (crosstalk, film base and exposure ratio) scaled per row by E, E*g and E*b. Turns the "
          "parameter search off.")
    .nargs(3)
    .scan<'g', float>();
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
  const std::string shoulder_str = parser.get<std::string>("--shoulder");
  o.shoulder_auto = (shoulder_str == "auto");
  if (o.shoulder_auto) {
    o.shoulder.source = "auto";
  } else if (shoulder_str != "off") {
    char* end = nullptr;
    const float val = strtof(shoulder_str.c_str(), &end);
    if (!end || *end != '\0' || end == shoulder_str.c_str() || !std::isfinite(val) || val < 1.0f ||
        val > negicc::kShoulderXHiMax) {
      log_error("Invalid --shoulder argument '%s' (off, auto or x_hi in [1, %g])\n",
              shoulder_str.c_str(), (double)negicc::kShoulderXHiMax);
      return 1;
    }
    o.shoulder.x_hi = val;
    o.shoulder.source = val > negicc::kShoulderOff ? "cli" : "off";
  }
  o.gain_g = parser.get<float>("--gain_g");
  o.gain_b = parser.get<float>("--gain_b");
  o.gain_g_given = parser.is_used("--gain_g");
  o.gain_b_given = parser.is_used("--gain_b");

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

  const std::string film_format_arg = parser.get<std::string>("--film_format");
  o.film_format = film_format_from_name(film_format_arg);
  if (o.film_format == FILM_FORMAT_AUTO && film_format_arg != "auto") {
    log_error("Invalid --film_format '%s' (must be 35mm, 645, 6x6, 6x7, 6x9 or auto)\n",
            film_format_arg.c_str());
    return 1;
  }
  if (parser.is_used("--film_format") && !o.roi_auto) {
    log_error("--film_format requires --roi auto\n");
    return 1;
  }

  if (parser.is_used("--flat")) o.flat = parser.get<std::string>("--flat");
  if (parser.is_used("--film_stock")) o.film_stock = parser.get<std::string>("--film_stock");
  if (parser.is_used("--light_source")) o.light_source = parser.get<std::string>("--light_source");
  if (parser.is_used("--film_profile")) o.film_profile = parser.get<std::string>("--film_profile");
  o.target_given = parser.is_used("--target");
  if (o.target_given) o.target = parser.get<std::string>("--target");
  o.station_egb_given = parser.is_used("--station_egb");
  if (o.station_egb_given) {
    const auto v = parser.get<std::vector<float>>("--station_egb");
    for (int i = 0; i < 3; ++i) o.station_egb[i] = v[i];
  }
  if (parser.is_used("--dino")) o.dino_adapter = parser.get<std::string>("--dino");
  o.dino_backend = parser.get<std::string>("--dino_backend");
  o.enclosure_rescue = parser.get<bool>("--enclosure_rescue");
  o.quality = parser.get<int>("--quality");
  o.no_crop = parser.get<bool>("--no_crop");
  o.output = parser.get<std::string>("--output");
  if (parser.is_used("--colorspace")) o.colorspace = parser.get<std::string>("--colorspace");
  o.no_metadata = parser.get<bool>("--no-metadata");
  return 0;
}
