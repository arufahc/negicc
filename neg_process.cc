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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <lcms2.h>
#include <lcms2_plugin.h>
#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <netinet/in.h>

#include "argparse/argparse.hpp"
#include "libraw/libraw.h"
#include "dinov3_engine.h"
#include "neg_pipeline.h"
#include "cfa_pipeline.h"
#include "color_pipeline.h"
#include "flat_field.h"
#include "flat_sample.h"
#include "dinov3_auto_solver.h"
#include "embedded_data.h"
#include "film_frame_detector.h"
#include "pixel_shift_group.h"
#include "rgb_input.h"
#include "np_conversion.h"
#include "np_geometry.h"
#include "np_input.h"
#include "np_output.h"
#include "np_profile.h"
#include "np_raw.h"
#include "multi_target_optimizer.h"
#include "profile_bundle.h"


#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21))
#error This code requires LibRaw 0.21+ (vendored in 3rd_party/libraw)
#endif

// How neg_process is meant to be used, by capture (to move to the docs later):
//
// - A single RAW captured under the narrowband RGB LED (tri) or white (d65) light, e.g.
//     neg_process --film_stock portra400 --light_source tri --roi 1700 0 6686 6376 -o out.tif scan.ARW
//   A tri bundle (input_space camera_raw_rgb) converts the sensor RGB with its crosstalk correction; -r, -g and -b
//   override it with one measured for this capture. A d65 bundle (rec2020_linear) converts LibRaw's white-light
//   decode (the camera's daylight white balance and colour matrix, to linear Rec.2020) with no crosstalk
//   correction, as negicc-station does, so it takes no -r, -g or -b, and no pixel shift. -p overrides the bundle with a negicc-station bundle JSON
//   (e.g. to test parity with negicc-station) or a single ICC profile (for research).
// - Four RAW files of a Sony A7R IV 4-shot pixel shift capture, merged into one image, the same way.
// - A linear (demosaiced) DNG, e.g. a film scanner's, decoded by LibRaw as a RAW is: camera RGB and the bundle's
//   crosstalk correction for a tri bundle, the DNG's colour data to linear Rec.2020 for a d65 bundle. A fourth
//   sample per pixel (a scanner's infrared) is left out.
// - An RGB TIFF (8 or 16 bits) or JPEG, with a d65 bundle: converted to linear Rec.2020 through its embedded ICC
//   profile (rgb_input.h); without one, 8-bit data is taken as sRGB and 16-bit data as linear Rec.2020.
// - Three images, each captured under one of the red, green and blue lights (RAWs, DNGs or TIFFs, the same size):
//     neg_process --film_stock portra400 --light_source tri -o out.tif red.ARW green.ARW blue.ARW
//   R comes from the first, G from the second and B from the third (a grey TIFF gives its one channel), each scaled
//   to the first capture's exposure when all three record theirs. There is no crosstalk between the lights to
//   correct, so the correction is the identity (-r, -g and -b give another). --flat applies to RAWs only.
//
// The parameter search (the calibration bracket, the exposure and the gains, solved with DINOv3 intent) runs
// whenever a profile is given, unless --gain_g or --gain_b is; then the conversion takes the values as given. -E
// keeps the exposure as given and solves only the gains, on the bracket --target picks (or the bundle's default).
// TODO: the bracket search on the frame --roi auto finds; for now it needs the frame given with --roi.
//
// --roi auto [h|v] finds the frame to crop (film_frame_detector.h): 35mm, 645, 6x6, 6x7 or 6x9 film on bare light, in
// a strip holder or in a frame holder, tilted up to 5 degrees, partial neighbour frames beside it. --film_format gives
// the format (more accurate than trying each); h or v gives the strip direction. The detection reads the RAW's CFA
// mosaic in place before anything is decoded, and only the frame is decoded, e.g.
//     neg_process --film_stock portra400 --light_source tri --target 0 --roi auto --film_format 6x7 -o out.tif scan.ARW
int main(int ac, char *av[]) {
  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--list_profiles") == 0) {
      for (const std::string& key : list_embedded_bundles()) {
        // Keys are "<Stock>-<camera>-<Light>".
        std::string stock = key.substr(0, key.find('-'));
        for (char& ch : stock) ch = (char)std::tolower((unsigned char)ch);
        const std::string light = key.substr(key.find_last_of('-') + 1);
        printf("--film_stock %s --light_source %s  (%s)\n", stock.c_str(),
               light == "TriLED" ? "tri" : light == "D65" ? "d65" : light.c_str(), key.c_str());
      }
      return 0;
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
  parser.add_argument("--knee")
    .help("Soft knee highlight compression threshold in [0, 1], or 'auto' (the parameter search on an ICC profile or a "
          "--target bracket). Defaults to 0.95 (>=1.0 disables).")
    .default_value(std::string("0.95"));
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
  parser.add_argument("--profile_film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from profile, in the space the RAW is decoded to (sensor "
          "RGB, or linear Rec.2020 for a rec2020_linear bundle).")
    .nargs(3)
    .default_value(std::vector<int>{1, 1, 1})
    .scan<'i', int>();
  // TODO: read the film base from an image (a capture of unexposed film, as negicc-station's Film Base tab does), in
  // place of the profile's; --film_base_rgb, which took it as numbers or from the rebate (--roi auto), is removed.
  parser.add_argument("--film_stock")
    .help("Film stock of the built-in profile bundle and intent adapter: ektar100, portra160 or portra400. Without it "
          "(-p with an ICC profile or a bundle whose name names no stock with an adapter), the parameter search falls "
          "back to the portra400 intent adapter.");
  parser.add_argument("--light_source")
    .help("Scanning light of the built-in profile bundle: tri (narrowband red, green and blue LED; sensor RGB and its "
          "crosstalk correction) or d65 (white; LibRaw's white-light decode to linear Rec.2020). Required for a RAW; "
          "otherwise it follows the input: d65 for a linear DNG, a JPEG or a TIFF, tri for a TIFF with --input_rgb "
          "camera and for three captures.");
  parser.add_argument("--input_rgb")
    .help("What a TIFF's values are: rec2020 (converted through its ICC profile, else sRGB for 8 bits or taken as "
          "linear Rec.2020 for 16 bits; a d65 bundle) or camera (the scanning camera's linear RGB under the "
          "TriLED light, black subtracted, no white balance or colour matrix; a tri bundle and its crosstalk "
          "correction).")
    .default_value(std::string("rec2020"));
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
  // TODO: Expand --flat to accept a flat-field capture image directly in addition to a precomputed .npz file.
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
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  const auto files = parser.get<std::vector<std::string>>("raw_files");
  if (files.size() == 2 || files.size() > 4) {
    fprintf(stderr, "ERROR! Give one RAW, DNG, TIFF or JPEG file, three captures (one per light: red, green, blue) "
                    "or the four RAW files of a pixel shift capture\n");
    return 1;
  }
  const InputMode input_mode = files.size() == 3 ? InputMode::kThreeShot
                               : files.size() == 1 && (negicc::is_jpeg_file(files[0]) || !libraw_opens(files[0]))
                                   ? InputMode::kRgbFile
                               : files.size() == 1 && is_linear_dng(files[0]) ? InputMode::kLinearDng
                                                                              : InputMode::kRaw;
  const std::string input_rgb = parser.get<std::string>("--input_rgb");
  if (input_rgb != "rec2020" && input_rgb != "camera") {
    fprintf(stderr, "ERROR! --input_rgb is rec2020 or camera, not '%s'\n", input_rgb.c_str());
    return 1;
  }
  const bool camera_rgb_input = input_rgb == "camera";
  if (camera_rgb_input && (input_mode != InputMode::kRgbFile || negicc::is_jpeg_file(files[0]))) {
    fprintf(stderr, "ERROR! --input_rgb camera applies to a single TIFF (a JPEG's 8-bit sRGB is not linear camera "
                    "RGB)\n");
    return 1;
  }
  // The scanning light the input implies, for the built-in bundle when --light_source is not given: a linear DNG,
  // a JPEG or a TIFF (as Rec.2020) is white light, a TIFF of camera RGB or three captures one per light is TriLED.
  const char* implied_light = input_mode == InputMode::kRaw ? nullptr
                              : input_mode == InputMode::kThreeShot || camera_rgb_input ? "tri"
                                                                                        : "d65";
  if (input_mode != InputMode::kRaw && parser.is_used("--flat")) {
    fprintf(stderr, "ERROR! --flat applies to a single CFA RAW or a pixel shift capture\n");
    return 1;
  }
  if (files.size() == 4) {
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = i + 1; j < 4; ++j) {
        std::error_code ec;
        if (files[i] == files[j] ||
            (std::filesystem::exists(files[i], ec) && std::filesystem::exists(files[j], ec) &&
             std::filesystem::equivalent(files[i], files[j], ec))) {
          fprintf(stderr, "ERROR! 4-shot pixel shift requires 4 distinct RAW files (duplicate: %s)\n",
                  files[i].c_str());
          return 1;
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
      fprintf(stderr, "ERROR! %s\n", group_err.c_str());
      return 1;
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
      fprintf(stderr, "ERROR! 4-shot pixel shift files are out of frame order\n");
      return 1;
    }
  }
  if (parser.get<int>("--dino_iters") <= 0) {
    fprintf(stderr, "ERROR! --dino_iters must be positive (got %d)\n", parser.get<int>("--dino_iters"));
    return 1;
  }
  auto r_coeff = parser.get<std::vector<float>>("--r_coeff");
  auto g_coeff = parser.get<std::vector<float>>("--g_coeff");
  auto b_coeff = parser.get<std::vector<float>>("--b_coeff");
  float global_exposure_comp = parser.get<float>("--exposure_comp");
  const std::string knee_str = parser.get<std::string>("--knee");
  const bool is_knee_auto = (knee_str == "auto");
  float knee = 0.95f;
  if (!is_knee_auto) {
    char* end = nullptr;
    const float val = strtof(knee_str.c_str(), &end);
    if (!end || *end != '\0' || end == knee_str.c_str() || !std::isfinite(val) || val <= 0.0f) {
      fprintf(stderr, "ERROR! Invalid --knee argument '%s'\n", knee_str.c_str());
      return 1;
    }
    knee = val;
  }
  const std::string shoulder_str = parser.get<std::string>("--shoulder");
  const bool is_shoulder_auto = (shoulder_str == "auto");
  ShoulderInfo shoulder_info;
  if (is_shoulder_auto) {
    shoulder_info.source = "auto";
  } else if (shoulder_str != "off") {
    char* end = nullptr;
    const float val = strtof(shoulder_str.c_str(), &end);
    if (!end || *end != '\0' || end == shoulder_str.c_str() || !std::isfinite(val) || val < 1.0f ||
        val > negicc::kShoulderXHiMax) {
      fprintf(stderr, "ERROR! Invalid --shoulder argument '%s' (off, auto or x_hi in [1, %g])\n",
              shoulder_str.c_str(), (double)negicc::kShoulderXHiMax);
      return 1;
    }
    shoulder_info.x_hi = val;
    shoulder_info.source = val > negicc::kShoulderOff ? "cli" : "off";
  }
  float gain_g = parser.get<float>("--gain_g");
  float gain_b = parser.get<float>("--gain_b");

  int rot_cw = 0;
  bool hflip = false;
  bool vflip = false;
  int roi_coords[4] = {0, 0, 0, 0};
  bool has_roi = false;
  bool is_roi_auto = false;
  char roi_axis_hint = 0;

  if (parser.is_used("--roi")) {
    const auto roi_tokens = parser.get<std::vector<std::string>>("--roi");
    if (roi_tokens.empty()) {
      fprintf(stderr, "ERROR! --roi requires arguments\n");
      return 1;
    }
    if (roi_tokens[0] == "auto") {
      is_roi_auto = true;
      if (roi_tokens.size() > 1) {
        if (roi_tokens[1] == "h" || roi_tokens[1] == "H") {
          roi_axis_hint = 'h';
        } else if (roi_tokens[1] == "v" || roi_tokens[1] == "V") {
          roi_axis_hint = 'v';
        } else {
          fprintf(stderr, "ERROR! Invalid --roi auto axis hint '%s' (must be 'h' or 'v')\n", roi_tokens[1].c_str());
          return 1;
        }
      }
      if (roi_tokens.size() > 2) {
        fprintf(stderr, "ERROR! Too many arguments for --roi auto\n");
        return 1;
      }
    } else {
      if (roi_tokens.size() < 4) {
        fprintf(stderr, "ERROR! --roi requires at least 4 arguments: x y w h [rot_cw [hflip [vflip]]]\n");
        return 1;
      }
      try {
        roi_coords[0] = std::stoi(roi_tokens[0]);
        roi_coords[1] = std::stoi(roi_tokens[1]);
        roi_coords[2] = std::stoi(roi_tokens[2]);
        roi_coords[3] = std::stoi(roi_tokens[3]);
        if (roi_tokens.size() >= 5) rot_cw = std::stoi(roi_tokens[4]);
        if (roi_tokens.size() >= 6) hflip = (std::stoi(roi_tokens[5]) != 0);
        if (roi_tokens.size() >= 7) vflip = (std::stoi(roi_tokens[6]) != 0);
        has_roi = true;
      } catch (const std::exception& e) {
        fprintf(stderr, "ERROR! Invalid integer argument for --roi\n");
        return 1;
      }
      if (roi_coords[0] < 0 || roi_coords[1] < 0 || roi_coords[2] <= 0 || roi_coords[3] <= 0) {
        fprintf(stderr, "ERROR! Invalid --roi [%d, %d, %d, %d]: x and y must be >= 0 and w and h must be > 0\n",
                roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]);
        return 1;
      }
    }
  }

  if (parser.is_used("--rot")) {
    rot_cw = parser.get<int>("--rot");
  } else if (parser.is_used("--rot_cw")) {
    rot_cw = parser.get<int>("--rot_cw");
  }
  if (parser.get<bool>("--hflip")) hflip = true;
  if (parser.get<bool>("--vflip")) vflip = true;

  if (rot_cw % 90 != 0) {
    fprintf(stderr, "ERROR! Rotation must be a multiple of 90 degrees (got %d)\n", rot_cw);
    return 1;
  }

  const std::string film_format_arg = parser.get<std::string>("--film_format");
  const FilmFormat film_format = film_format_from_name(film_format_arg);
  if (film_format == FILM_FORMAT_AUTO && film_format_arg != "auto") {
    fprintf(stderr, "ERROR! Invalid --film_format '%s' (must be 35mm, 645, 6x6, 6x7, 6x9 or auto)\n",
            film_format_arg.c_str());
    return 1;
  }
  if (parser.is_used("--film_format") && !is_roi_auto) {
    fprintf(stderr, "ERROR! --film_format requires --roi auto\n");
    return 1;
  }


  // TODO: Expand --flat to take a flat-field capture image directly and compute the flat field on the fly.
  negicc::FlatFieldPtr flat_field;
  if (parser.is_used("--flat")) {
    try {
      flat_field = negicc::load_flat_field_file(parser.get<std::string>("--flat"));
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! %s\n", e.what());
      return 1;
    }
  }

  // The profile: -p, else the built-in bundle of --film_stock and --light_source.
  bool is_bundle = false;
  ProfileBundle bundle;
  std::string film_prof_arg;
  const std::string film_stock = parser.is_used("--film_stock") ? parser.get<std::string>("--film_stock") : "";
  if (parser.is_used("--light_source") && implied_light && parser.get<std::string>("--light_source") != implied_light) {
    fprintf(stderr, "ERROR! --light_source %s does not fit %s, which takes %s\n",
            parser.get<std::string>("--light_source").c_str(),
            input_mode == InputMode::kLinearDng ? "a linear DNG"
            : input_mode == InputMode::kThreeShot ? "three captures, one per light"
            : camera_rgb_input ? "a TIFF of linear camera RGB (--input_rgb camera)"
                               : "an RGB TIFF or JPEG converted to linear Rec.2020",
            implied_light);
    return 1;
  }
  if (parser.is_used("--light_source") && film_stock.empty()) {
    fprintf(stderr, "ERROR! --light_source requires --film_stock\n");
    return 1;
  }
  if (parser.is_used("--film_profile")) {
    film_prof_arg = parser.get<std::string>("--film_profile");
    try {
      is_bundle = load_bundle_arg(film_prof_arg, bundle);
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! %s\n", e.what());
      return 1;
    }
    if (!is_bundle && !is_regular_file(resolve_profile_path(film_prof_arg))) {
      fprintf(stderr, "ERROR! No profile bundle JSON or ICC profile '%s'\n", film_prof_arg.c_str());
      return 1;
    }
  } else if (!film_stock.empty()) {
    if (!parser.is_used("--light_source") && !implied_light) {
      fprintf(stderr, "ERROR! --film_stock requires --light_source (tri or d65)\n");
      return 1;
    }
    const std::string light = parser.is_used("--light_source") ? parser.get<std::string>("--light_source")
                                                                : std::string(implied_light);
    if (!parser.is_used("--light_source")) printf("Light source: %s, from the input\n", light.c_str());
    std::string error;
    const std::string key = embedded_bundle_key(film_stock, light, error);
    if (key.empty()) {
      fprintf(stderr, "ERROR! %s (--list_profiles lists them)\n", error.c_str());
      return 1;
    }
    try {
      if (!load_embedded_bundle(key, bundle)) {
        fprintf(stderr, "ERROR! Cannot load built-in profile bundle %s\n", key.c_str());
        return 1;
      }
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! %s\n", e.what());
      return 1;
    }
    is_bundle = true;
  }
  if (is_bundle) {
    printf("Profile bundle: %s (%s, %zu brackets)\n", bundle.key.c_str(), bundle.path.c_str(),
           bundle.targets.size());
  }
  const bool has_profile = is_bundle || !film_prof_arg.empty();

  int manual_target_idx = -1;
  if (parser.is_used("--target")) {
    if (!is_bundle) {
      fprintf(stderr, "ERROR! --target requires a JSON profile bundle\n");
      return 1;
    }
    const std::string tgt_str = parser.get<std::string>("--target");
    char* end = nullptr;
    long val = strtol(tgt_str.c_str(), &end, 10);
    if (end && *end == '\0' && val >= 0 && val < (long)bundle.targets.size()) {
      manual_target_idx = (int)val;
    } else {
      for (size_t i = 0; i < bundle.targets.size(); ++i) {
        if (bundle.targets[i].name == tgt_str || bundle.targets[i].display_name == tgt_str) {
          manual_target_idx = (int)i;
          break;
        }
      }
    }
    if (manual_target_idx < 0) {
      fprintf(stderr, "ERROR! Target '%s' not found in bundle\n", tgt_str.c_str());
      return 1;
    }
  }

  const bool station_egb_given = parser.is_used("--station_egb");
  if (station_egb_given && !is_bundle) {
    fprintf(stderr, "ERROR! --station_egb requires a JSON profile bundle\n");
    return 1;
  }
  float station_egb[3] = {1.0f, 1.0f, 1.0f};
  if (station_egb_given) {
    const auto v = parser.get<std::vector<float>>("--station_egb");
    station_egb[0] = v[0];
    station_egb[1] = v[1];
    station_egb[2] = v[2];
  }

  // The parameter search runs with a profile unless --gain_g, --gain_b or --station_egb is given. On a bundle it is
  // negicc-station's conversion parameter search (multi_target_optimizer.h), which picks the bracket, the exposure
  // and the gains; with --target or -E (and on a single ICC profile) it solves the gains, and the exposure unless -E
  // gives it, on one profile.
  const bool exposure_given = parser.is_used("-E") || parser.is_used("--exposure_comp") || station_egb_given;
  const bool use_dino = has_profile && !parser.is_used("--gain_g") && !parser.is_used("--gain_b") && !station_egb_given;
  const bool bundle_search = use_dino && is_bundle && manual_target_idx < 0 && !exposure_given;
  if (parser.is_used("--dino") && !use_dino) {
    fprintf(stderr, "ERROR! --dino requires the parameter search: a profile (--film_stock and --light_source, or "
                    "--film_profile) without --gain_g or --gain_b\n");
    return 1;
  }
  if (is_knee_auto && (!use_dino || bundle_search)) {
    fprintf(stderr, "ERROR! --knee auto requires the parameter search on an ICC profile or a --target bracket\n");
    return 1;
  }
  if (is_shoulder_auto && !use_dino) {
    fprintf(stderr, "ERROR! --shoulder auto requires the parameter search (it measures the shoulder from the render "
                    "at the searched parameters); give an x_hi instead\n");
    return 1;
  }
  if (bundle_search) {
    if (parser.is_used("--profile_film_base_rgb")) {
      fprintf(stderr, "ERROR! --profile_film_base_rgb cannot be used with the bracket search (the bundle gives the "
                      "film base); pick a bracket with --target to use it\n");
      return 1;
    }
  }

  // The crosstalk correction: the bundle's, with the rows -r, -g and -b give in its place (for the search too).
  const bool row_given[3] = {parser.is_used("--r_coeff"), parser.is_used("--g_coeff"), parser.is_used("--b_coeff")};
  std::vector<float>* rows[3] = {&r_coeff, &g_coeff, &b_coeff};
  // A rec2020_linear (white-light) bundle takes the white-light decode, which has no crosstalk to correct and, as in
  // negicc-station, decodes single-shot captures only.
  // py-ref: negicc_station/src/pipeline_api.py @ 813166c (prepare_image)
  const bool white_light = is_bundle && bundle.input_space == InputSpace::kRec2020Linear;
  if (white_light) {
    if (row_given[0] || row_given[1] || row_given[2]) {
      fprintf(stderr, "ERROR! -r, -g and -b cannot be used with the rec2020_linear profile bundle %s: its white-light "
                      "decode has no crosstalk to correct\n", bundle.key.c_str());
      return 1;
    }
    if (files.size() == 4) {
      fprintf(stderr, "ERROR! The rec2020_linear profile bundle %s converts single-shot captures only, not pixel "
                      "shift\n", bundle.key.c_str());
      return 1;
    }
    if (input_mode == InputMode::kThreeShot) {
      fprintf(stderr, "ERROR! The rec2020_linear profile bundle %s is for white light; three captures, one per "
                      "light, take a camera_raw_rgb (tri) bundle\n", bundle.key.c_str());
      return 1;
    }
  }
  // Three captures, one per light, have no crosstalk between the lights to correct: each channel comes from the
  // capture under its own light, so the correction is the identity unless -r, -g or -b give one.
  if (input_mode == InputMode::kThreeShot && is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) continue;
      for (int c = 0; c < 3; ++c) {
        bundle.crosstalk_matrix[r * 3 + c] = r == c ? 1.0f : 0.0f;
        bundle.crosstalk_matrix_d[r * 3 + c] = r == c ? 1.0 : 0.0;
      }
    }
    bundle.has_crosstalk = true;
  }
  if (is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) {
        for (int c = 0; c < 3; ++c) {
          bundle.crosstalk_matrix[r * 3 + c] = (*rows[r])[c];
          bundle.crosstalk_matrix_d[r * 3 + c] = (double)(*rows[r])[c];
        }
      } else if (bundle.has_crosstalk) {
        rows[r]->assign(bundle.crosstalk_matrix + r * 3, bundle.crosstalk_matrix + r * 3 + 3);
      }
    }
  }
  const bool has_crosstalk = row_given[0] || row_given[1] || row_given[2] || (is_bundle && bundle.has_crosstalk);
  // An RGB TIFF or JPEG is colour-managed RGB, converted to linear Rec.2020: not the camera RGB a crosstalk
  // correction (a camera_raw_rgb bundle) needs.
  if (input_mode == InputMode::kRgbFile && !camera_rgb_input && has_crosstalk) {
    fprintf(stderr, "ERROR! %s is an RGB TIFF or JPEG, converted to linear Rec.2020: give a rec2020_linear (d65) "
                    "profile bundle, not one with a crosstalk correction (or --input_rgb camera for a TIFF of linear "
                    "camera RGB)\n", files[0].c_str());
    return 1;
  }
  if (input_mode == InputMode::kRgbFile && camera_rgb_input && white_light) {
    fprintf(stderr, "ERROR! A TIFF of linear camera RGB (--input_rgb camera) takes a camera_raw_rgb (tri) bundle\n");
    return 1;
  }
  if (input_mode == InputMode::kLinearDng && is_bundle && !white_light) {
    fprintf(stderr, "ERROR! A linear DNG is converted to linear Rec.2020 with its own colour data: give a "
                    "rec2020_linear (d65) profile bundle\n");
    return 1;
  }
  const RawDecode raw_decode = white_light ? RawDecode::kWhiteLight
                               : has_crosstalk ? RawDecode::kCameraRgb
                                               : RawDecode::kRec2020;

  DinoV3Engine engine;
  std::string dino_path;
  const std::string dino_backend = parser.get<std::string>("--dino_backend");
  if (use_dino) {
    const EmbeddedFile* trt = find_embedded_backbone(true);
    const EmbeddedFile* onnx = find_embedded_backbone(false);
    const EmbeddedFile* backbone = trt ? trt : onnx;
    if (!backbone) {
      fprintf(stderr, "ERROR! This neg_process was built without the DINOv3 backbone of the parameter search (make "
                      "dinov3_model, or make dinov3_engine for USE_CUDA=1, then rebuild); give --gain_g or --gain_b "
                      "to convert without it\n");
      return 1;
    }
    dino_path = std::string("[built-in] ") + backbone->name;
    printf("Loading DINOv3 backbone %s (backend: %s)...\n", dino_path.c_str(), dino_backend.c_str());
    try {
      engine.load_memory(trt ? trt->data : nullptr, trt ? trt->size : 0, onnx ? onnx->data : nullptr,
                         onnx ? onnx->size : 0, dino_backend);
      // The file that runs: the .onnx when the engine was not loaded.
      if (trt && std::strcmp(engine.backend_used(), "gpu") != 0) {
        dino_path = std::string("[built-in] ") + onnx->name;
      }
      if (parser.is_used("--dino")) {
        std::string adapter = parser.get<std::string>("--dino");
        if (is_directory(adapter)) adapter = (std::filesystem::path(adapter) / "adapter.onnx").string();
        engine.load_adapter_file(adapter);
      } else {
        std::string adapter_dir;
        const std::string profile_name = !film_stock.empty() ? film_stock : is_bundle ? bundle.key : film_prof_arg;
        const EmbeddedFile* a = find_embedded_adapter(profile_name, adapter_dir);
        if (!a) {
          fprintf(stderr, "ERROR! No intent adapter for profile '%s'\n", profile_name.c_str());
          return 1;
        }
        engine.set_adapter(a->data, a->size, adapter_dir);
      }
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! %s\n", e.what());
      return 1;
    }
    printf("DINOv3 engine loaded [%s] adapter %s (memory: %.1f MB)\n", engine.backend_name(),
           engine.adapter_name().c_str(), (double)engine.memory_bytes() / (1024.0 * 1024.0));
  }

  double exif_shutter_d = 0.0, exif_iso_d = 0.0;
  const bool has_exif_exp = read_exif_exposure_d(files[0], exif_shutter_d, exif_iso_d);
  bool warned_default_exposure = false;
  auto get_scan_exposure = [&](const LibRaw& r, double& shutter_out, double& iso_out) {
    const bool have_shutter = (has_exif_exp && exif_shutter_d > 0.0) || r.imgdata.other.shutter > 0;
    const bool have_iso = (has_exif_exp && exif_iso_d > 0.0) || r.imgdata.other.iso_speed > 0;
    shutter_out = (has_exif_exp && exif_shutter_d > 0.0)
                      ? exif_shutter_d
                      : (r.imgdata.other.shutter > 0 ? (double)r.imgdata.other.shutter : 0.125);
    iso_out = (has_exif_exp && exif_iso_d > 0.0)
                  ? std::round(exif_iso_d)
                  : (r.imgdata.other.iso_speed > 0 ? std::round((double)r.imgdata.other.iso_speed) : 100.0);
    if ((!have_shutter || !have_iso) && !warned_default_exposure) {
      warned_default_exposure = true;
      fprintf(stderr, "Warning: Missing EXIF exposure metadata in %s; falling back to %.4g s, ISO %.0f\n",
              files[0].c_str(), shutter_out, iso_out);
    }
  };

  bool has_auto_res = false;
  ParamSearchResult auto_res;
  int auto_frame_half[4] = {0, 0, 0, 0};  // the frame (x, y, w, h) in the half-size image
  double t_auto_ms = 0.0;
  HalfSizeAnalysis analyse;
  if (bundle_search) {
    analyse = [&](const LibRaw& raw) -> bool {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(raw, scan_shutter_s, scan_iso);
      // --roi halved, or the whole image.
      const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
      int* bbox = auto_frame_half;
      bbox[0] = 0;
      bbox[1] = 0;
      bbox[2] = half_w;
      bbox[3] = half_h;
      if (has_roi) {
        bbox[0] = roi_coords[0] / 2;
        bbox[1] = roi_coords[1] / 2;
        bbox[2] = std::max(1, (roi_coords[0] + roi_coords[2]) / 2 - bbox[0]);
        bbox[3] = std::max(1, (roi_coords[1] + roi_coords[3]) / 2 - bbox[1]);
        clamp_rect(bbox, half_w, half_h);
      }
      // Step 3 & Step 4: zero-copy RgbView over the half-size image, cropped and oriented in place.
      negicc::RgbView half_view = negicc::RgbView::from_interleaved_rgb(
          &raw.imgdata.image[0][0], half_w, half_h, 4, (ptrdiff_t)half_w * 4);
      negicc::RgbView frame = half_view.crop_and_orient(bbox[0], bbox[1], bbox[2], bbox[3], rot_cw, hflip, vflip);
      ParamSearchOptions opts;
      if (parser.is_used("--dino_iters")) opts.max_steps = parser.get<int>("--dino_iters");
      opts.shoulder = shoulder_info.x_hi;
      opts.shoulder_auto = is_shoulder_auto;
      opts.enclosure_rescue = parser.get<bool>("--enclosure_rescue");
      if (!search_conversion_params(bundle, engine, frame, scan_shutter_s, scan_iso, opts, auto_res)) {
        fprintf(stderr, "ERROR! The conversion parameter search failed\n");
        return false;
      }
      has_auto_res = true;
      t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
      return true;
    };
  }

  FrameDetection frame_det;
  FrameDetectionParams det_params;
  det_params.format = film_format;
  det_params.axis_hint = roi_axis_hint;
  double t_frame_det_ms = 0.0;
  RawRegionFinder find_frame;
  if (is_roi_auto && files.size() != 4) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        fprintf(stderr, "ERROR! --roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        return false;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, frame_det);
      t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      if (frame_det.conf < kLowFrameConfidence) {
        roi_coords[0] = 0;
        roi_coords[1] = 0;
        roi_coords[2] = mosaic.width;
        roi_coords[3] = mosaic.height;
      } else {
        roi_coords[0] = frame_det.crop[0];
        roi_coords[1] = frame_det.crop[1];
        roi_coords[2] = frame_det.crop[2];
        roi_coords[3] = frame_det.crop[3];
      }
      has_roi = true;
      for (int i = 0; i < 4; ++i) roi[i] = roi_coords[i];
      return true;
    };
  }

  auto print_frame_detection = [&]() {
    const char* axis_name = frame_det.axis == 'h' ? "h" : (frame_det.axis == 'v' ? "v" : "undecided");
    printf("Film frame: axis=%s format=%s orientation=%s rotation=%.2f deg gaps=%d frames=%d conf=%.2f%s%s\n",
           axis_name, film_format_name(frame_det.format), frame_det.orientation.c_str(), frame_det.rotation_deg,
           frame_det.n_gaps, 1 + (int)frame_det.secondaries.size(), frame_det.conf,
           frame_det.reason.empty() ? "" : " reason: ", frame_det.reason.c_str());
    if (frame_det.n_gaps > 0) {
      printf("  Gaps (sensor px):");
      for (int g = 0; g < 2; ++g)
        if (frame_det.gaps[g][0] >= 0) printf(" [%d, %d]", frame_det.gaps[g][0], frame_det.gaps[g][1]);
      printf("\n");
    }
    printf("  Frame (sensor px): x=%d y=%d w=%d h=%d (frame centre %.1f, %.1f, size %.1f x %.1f, %.1f px/mm)\n",
           roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3], frame_det.primary.rect.cx,
           frame_det.primary.rect.cy, frame_det.primary.rect.w, frame_det.primary.rect.h, frame_det.px_per_mm);
    for (const FilmFrame& f : frame_det.secondaries)
      printf("  Frame %+d (sensor px): visible x=%d y=%d w=%d h=%d (%.0f%% of a frame)\n", f.index, f.visible[0],
             f.visible[1], f.visible[2], f.visible[3], 100.0f * f.visible_fraction);
  };

  const int qual_flag = parser.get<int>("--quality");
  const bool no_crop_flag = parser.get<bool>("--no_crop");

  // Determine whether the 5-step zero-copy CFA pipeline applies (Step 1..5).
  bool use_cfa_pipeline = false;
  negicc::InputFormat input_format = (files.size() == 4) ? negicc::InputFormat::SONY_PIXEL_SHIFT_4
                                                         : negicc::InputFormat::CFA_BAYER_OR_XTRANS;
  const double t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  LibRaw *unpacked_probe = nullptr;
  std::string input_source = "raw";  // what was read: "raw", "tiff:icc", "jpeg:srgb", "three:raw,raw,raw", ...
  std::unique_ptr<LibRaw> ps_extra_procs[3];
  float cfa_white_level = 65535.0f;
  int cfa_rx = 0, cfa_ry = 0, cfa_rw = 0, cfa_rh = 0;
  negicc::RgbView full_half_view;
  negicc::RgbView oriented_half_view;
  negicc::RgbView search_half_view;
  std::shared_ptr<const negicc::FlatTables> flat_full_tables;
  negicc::FlatSampleArgs flat_full{};

  if ((input_mode == InputMode::kRaw || input_mode == InputMode::kLinearDng) &&
      (files.size() == 4 || (raw_decode == RawDecode::kCameraRgb && qual_flag < 0))) {
    LibRaw* probe = new LibRaw();
    printf("Loading RAW file %s\n", files[0].c_str());
    int ret = probe->open_file(files[0].c_str());
    if (ret != LIBRAW_SUCCESS) {
      fprintf(stderr, "Cannot open %s: %s\n", files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    printf("Image size: %dx%d\n", probe->imgdata.sizes.iwidth, probe->imgdata.sizes.iheight);
    if ((ret = probe->unpack()) != LIBRAW_SUCCESS) {
      fprintf(stderr, "Cannot unpack %s: %s\n", files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    if (files.size() == 4) {
      if (!is_sony_a7rm4(probe)) {
        fprintf(stderr, "ERROR: 4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
                probe->imgdata.idata.make, probe->imgdata.idata.model);
        delete probe;
        return 1;
      }
      use_cfa_pipeline = true;
    } else if (is_sony_cfa_geometry(probe) && probe->imgdata.rawdata.raw_image) {
      use_cfa_pipeline = true;
    }
    if (use_cfa_pipeline) {
      proc = probe;
    } else {
      unpacked_probe = probe;
    }
  }

  double t_raw_decode_ms = 0.0;

  if (use_cfa_pipeline) {
    (void)input_format;
    if (is_roi_auto) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(*proc, mosaic)) {
        fprintf(stderr, "ERROR! --roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        delete proc;
        return 1;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, frame_det);
      t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
    }

    // Step 1: Per-channel black subtraction in-place on the unpacked CFA mosaic.
    if (!subtract_cfa_black_in_place(proc, cfa_white_level)) {
      fprintf(stderr, "ERROR! Invalid CFA mosaic geometry in %s\n", files[0].c_str());
      delete proc;
      return 1;
    }
    const int src_w = proc->imgdata.sizes.width;
    const int src_h = proc->imgdata.sizes.height;
    const size_t cfa_pitch = proc->imgdata.sizes.raw_pitch / 2;
    const unsigned int filters = proc->imgdata.idata.filters;
    const uint16_t* cfa_origin = proc->imgdata.rawdata.raw_image +
                                 (size_t)proc->imgdata.sizes.top_margin * cfa_pitch +
                                 proc->imgdata.sizes.left_margin;

    // For 4x pixel shift, unpack and black-subtract frames 1..3 in Step 1 and keep all 4 CFA mosaics in memory.
    if (files.size() == 4) {
      for (int mi = 1; mi < 4; ++mi) {
        ps_extra_procs[mi - 1] = std::make_unique<LibRaw>();
        LibRaw* fp = ps_extra_procs[mi - 1].get();
        printf("Loading RAW file %s\n", files[mi].c_str());
        int fret = fp->open_file(files[mi].c_str());
        if (fret != LIBRAW_SUCCESS || (fret = fp->unpack()) != LIBRAW_SUCCESS) {
          fprintf(stderr, "Cannot open/unpack %s: %s\n", files[mi].c_str(), libraw_strerror(fret));
          delete proc;
          return 1;
        }
        const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
        if (!is_sony_a7rm4(fp) || fp->imgdata.sizes.width != src_w || fp->imgdata.sizes.height != src_h ||
            fpitch != cfa_pitch || fp->imgdata.idata.filters != filters ||
            fp->imgdata.sizes.top_margin != proc->imgdata.sizes.top_margin ||
            fp->imgdata.sizes.left_margin != proc->imgdata.sizes.left_margin ||
            std::fabs(fp->imgdata.other.iso_speed - proc->imgdata.other.iso_speed) > 1e-3f ||
            std::fabs(fp->imgdata.other.shutter - proc->imgdata.other.shutter) > 1e-6f) {
          fprintf(stderr, "ERROR: 4-shot pixel shift merge is only available with matching Sony A7RM4 frames\n");
          delete proc;
          return 1;
        }
        float fw_white = 0.0f;
        if (!subtract_cfa_black_in_place(fp, fw_white)) {
          fprintf(stderr, "ERROR! Invalid CFA mosaic geometry in %s\n", files[mi].c_str());
          delete proc;
          return 1;
        }
      }
    }

    // Step 2: Downsizing via zero-copy RgbView over the CFA mosaic.
    const int half_w = src_w / 2;
    const int half_h = src_h / 2;
    std::shared_ptr<const negicc::FlatTables> flat_half_tables;
    negicc::FlatSampleArgs flat_half{};
    if (flat_field) {
      try {
        flat_half_tables = flat_field->tables(half_w, half_h, true);
        flat_half = flat_half_tables->host();
        flat_full_tables = flat_field->tables(src_w, src_h, false);
        flat_full = flat_full_tables->host();
      } catch (const std::exception& e) {
        fprintf(stderr, "ERROR! %s\n", e.what());
        delete proc;
        return 1;
      }
    }
    full_half_view = negicc::RgbView::from_cfa_half(
        cfa_origin, cfa_pitch, src_w, src_h, filters, nullptr, flat_half, flat_half_tables);

    // Step 3: Crop, rotate, flip & inline ROI search on the zero-copy view.
    if (is_roi_auto) {
      if (frame_det.conf < kLowFrameConfidence) {
        roi_coords[0] = 0;
        roi_coords[1] = 0;
        roi_coords[2] = src_w;
        roi_coords[3] = src_h;
      } else {
        roi_coords[0] = frame_det.crop[0];
        roi_coords[1] = frame_det.crop[1];
        roi_coords[2] = frame_det.crop[2];
        roi_coords[3] = frame_det.crop[3];
      }
      has_roi = true;
      print_frame_detection();

      if (frame_det.conf < kLowFrameConfidence) {
        printf("Warning: low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
      }
      if (use_dino && frame_det.axis == 'v' && rot_cw == 0) {
        printf("Warning: axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
      }
    }

    cfa_rx = 0;
    cfa_ry = 0;
    cfa_rw = src_w;
    cfa_rh = src_h;
    if (has_roi) {
      if (roi_coords[0] < 0 || roi_coords[1] < 0 || roi_coords[0] >= src_w || roi_coords[1] >= src_h ||
          roi_coords[2] <= 0 || roi_coords[3] <= 0) {
        fprintf(stderr, "ERROR! ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
                roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3], src_w, src_h);
        delete proc;
        return 1;
      }
      cfa_rx = roi_coords[0];
      cfa_ry = roi_coords[1];
      cfa_rw = std::min(roi_coords[2], src_w - cfa_rx);
      cfa_rh = std::min(roi_coords[3], src_h - cfa_ry);
    } else if (!no_crop_flag && !station_egb_given && files.size() == 1) {
      const auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
      const bool has_inset_crop = crop_ref.cwidth > 0 && crop_ref.cheight > 0 &&
                                  crop_ref.cleft != 0xffff && crop_ref.ctop != 0xffff;
      if (has_inset_crop && (crop_ref.cleft || crop_ref.ctop)) {
        cfa_rx = std::max(0, std::min((int)crop_ref.cleft, src_w - 1));
        cfa_ry = std::max(0, std::min((int)crop_ref.ctop, src_h - 1));
        cfa_rw = std::max(1, std::min((int)crop_ref.cwidth, src_w - cfa_rx));
        cfa_rh = std::max(1, std::min((int)crop_ref.cheight, src_h - cfa_ry));
      }
    }

    int out_half[4] = {
        cfa_rx / 2,
        cfa_ry / 2,
        std::max(1, (cfa_rx + cfa_rw) / 2 - cfa_rx / 2),
        std::max(1, (cfa_ry + cfa_rh) / 2 - cfa_ry / 2),
    };
    clamp_rect(out_half, half_w, half_h);
    for (int i = 0; i < 4; ++i) auto_frame_half[i] = out_half[i];
    search_half_view = full_half_view.crop_and_orient(
        auto_frame_half[0], auto_frame_half[1], auto_frame_half[2], auto_frame_half[3],
        rot_cw, hflip, vflip);
    oriented_half_view = search_half_view;

    // Step 4: Conversion parameter search inline on the zero-copy oriented RgbView.
    if (bundle_search) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(*proc, scan_shutter_s, scan_iso);
      ParamSearchOptions opts;
      if (parser.is_used("--dino_iters")) opts.max_steps = parser.get<int>("--dino_iters");
      opts.shoulder = shoulder_info.x_hi;
      opts.shoulder_auto = is_shoulder_auto;
      opts.enclosure_rescue = parser.get<bool>("--enclosure_rescue");
      if (!search_conversion_params(bundle, engine, search_half_view, scan_shutter_s, scan_iso, opts, auto_res)) {
        fprintf(stderr, "ERROR! The conversion parameter search failed\n");
        delete proc;
        return 1;
      }
      has_auto_res = true;
      t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    }
    t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0 - t_auto_ms - t_frame_det_ms;
  } else {
    if (input_mode == InputMode::kRgbFile || input_mode == InputMode::kThreeShot) {
      proc = load_rgb_input(files, input_mode, camera_rgb_input, qual_flag, !no_crop_flag && !has_roi && !is_roi_auto,
                            (!is_roi_auto && has_roi) ? roi_coords : nullptr, analyse,
                            is_roi_auto ? &frame_det : nullptr, det_params, input_source);
    } else {
      proc = load_raw(files[0], true,
                      qual_flag,
                      !no_crop_flag && !has_roi && !is_roi_auto,
                      raw_decode,
                      (!is_roi_auto && has_roi) ? roi_coords : nullptr,
                      analyse, find_frame, flat_field.get(),
                      unpacked_probe);
    }
    unpacked_probe = nullptr;
    if (!proc) {
      fprintf(stderr, "Cannot open %s\n", files[0].c_str());
      return 1;
    }
    if (is_roi_auto) {
      print_frame_detection();
      if (frame_det.conf < kLowFrameConfidence) {
        printf("Warning: low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
      }
      if (use_dino && frame_det.axis == 'v' && rot_cw == 0) {
        printf("Warning: axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
      }
    }
    if (rot_cw != 0 || hflip || vflip) {
      if (!extract_roi_and_transform(proc, 0, 0, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                     rot_cw, hflip, vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      printf("Transformed subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
    }
    t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0 - t_auto_ms - t_frame_det_ms;
  }

  const int orig_w = proc->imgdata.rawdata.sizes.width ? proc->imgdata.rawdata.sizes.width : proc->imgdata.sizes.width;
  const int orig_h = proc->imgdata.rawdata.sizes.height ? proc->imgdata.rawdata.sizes.height : proc->imgdata.sizes.height;

  printf("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  printf("Shutter Speed: %f\n", proc->imgdata.other.shutter);

  const auto orig_profile_film_base_rgb = parser.get<std::vector<int>>("--profile_film_base_rgb");
  std::vector<int> effective_profile_film_base_rgb = orig_profile_film_base_rgb;
  if (is_bundle && bundle.has_film_base && !parser.is_used("--profile_film_base_rgb")) {
    effective_profile_film_base_rgb = {bundle.profile_film_base_rgb[0], bundle.profile_film_base_rgb[1], bundle.profile_film_base_rgb[2]};
  }
  const std::vector<int> sidecar_profile_film_base_rgb = effective_profile_film_base_rgb;
  std::vector<int> effective_film_base_rgb = {1, 1, 1};
  std::vector<int> sidecar_film_base_rgb = {1, 1, 1};
  const bool film_base_applied = false;  // no film base of the capture (see the TODO at --profile_film_base_rgb)
  std::string film_base_source = "none";

  // No film base of the capture is given (TODO above: read it from an image), so none is applied: the film base scale
  // of the input matrix is 1.
  effective_profile_film_base_rgb = {1, 1, 1};

  if (bundle_search) {
    // The bundle's film base is folded into the solved exposure and gains (not applied as a film base scale).
    sidecar_film_base_rgb = sidecar_profile_film_base_rgb;
    film_base_source = "bundle";
  }

  unsigned pin_mask = 0;
  if (parser.is_used("-E") || parser.is_used("--exposure_comp")) pin_mask |= PIN_E;
  if (parser.is_used("--gain_g")) pin_mask |= PIN_G;
  if (parser.is_used("--gain_b")) pin_mask |= PIN_B;

  float merged_matrix[9];
  adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                           global_exposure_comp,
                           gain_g, gain_b,
                           effective_profile_film_base_rgb,
                           effective_film_base_rgb,
                           merged_matrix);

  // Step 5a: Debayer the cropped/oriented output RGB buffer (after Step 4's zero-copy search_conversion_params).
  // TODO: When `!bundle_search` (e.g. single-target `compute_bundle_enclosures` or `solve_dinov3_intent`), deferring
  // Step 5a full-resolution debayer until after `compute_bundle_enclosures` and `solve_dinov3_intent` have run on
  // `oriented_half_view` avoids running histogramming/solver passes over the full-resolution (e.g. 61 MP) buffer.
  // Currently kept before `compute_bundle_enclosures` / `solve_dinov3_intent` to preserve exact single-target parity
  // with master.
  std::vector<uint16_t> cfa_export_rgb;
  if (use_cfa_pipeline) {
    const double t_step5_debayer_0 = omp_get_wtime();
    const int src_w = proc->imgdata.sizes.width;
    const int src_h = proc->imgdata.sizes.height;
    const size_t cfa_pitch = proc->imgdata.sizes.raw_pitch / 2;
    const unsigned int filters = proc->imgdata.idata.filters;
    const uint16_t* cfa_origin = proc->imgdata.rawdata.raw_image +
                                 (size_t)proc->imgdata.sizes.top_margin * cfa_pitch +
                                 proc->imgdata.sizes.left_margin;
    int norm_rot = rot_cw;
    bool norm_hf = hflip, norm_vf = vflip;
    normalize_rotation(norm_rot, norm_hf, norm_vf);
    const bool swap_dims = (norm_rot == 90 || norm_rot == 270);
    int dst_w = 0, dst_h = 0;

    auto free_libraw_cfa = [&](LibRaw* r) {
      full_half_view = {};
      oriented_half_view = {};
      search_half_view = {};
      if (!r) return;
      if (r->imgdata.rawdata.raw_alloc && !r->imgdata.image) {
        r->imgdata.image = (ushort (*)[4])r->imgdata.rawdata.raw_alloc;
        r->imgdata.rawdata.raw_alloc = nullptr;
        r->imgdata.rawdata.raw_image = nullptr;
        r->free_image();
      }
      r->recycle_datastream();
    };

    dst_w = swap_dims ? cfa_rh : cfa_rw;
    dst_h = swap_dims ? cfa_rw : cfa_rh;
    cfa_export_rgb.resize((size_t)dst_w * dst_h * 3);
    if (files.size() == 1) {
      printf("Debayer quality: grain-aware\n");
      negicc::GrainAwareDebayerArgs ga{};
      ga.cfa_bayer = cfa_origin;
      ga.src_w = src_w;
      ga.src_h = src_h;
      ga.rx = cfa_rx;
      ga.ry = cfa_ry;
      ga.rw = cfa_rw;
      ga.rh = cfa_rh;
      ga.rot_cw = rot_cw;
      ga.hflip = hflip;
      ga.vflip = vflip;
      ga.dst_w = dst_w;
      ga.dst_h = dst_h;
      ga.filters = filters;
      ga.white_level = cfa_white_level;
      ga.eps = 4096.0f;
      ga.rgb_out = cfa_export_rgb.data();
      ga.flat = flat_full;
      ga.flat_x0 = cfa_rx;
      ga.flat_y0 = cfa_ry;
      ga.cfa_pitch = cfa_pitch;
      negicc::cpu_grain_aware_debayer(ga);
      free_libraw_cfa(proc);
    } else {
      printf("Merging 4 images...\n");
      const uint16_t* fcfa[3] = {nullptr, nullptr, nullptr};
      for (int i = 0; i < 3; ++i) {
        LibRaw* fp = ps_extra_procs[i].get();
        const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
        fcfa[i] = fp->imgdata.rawdata.raw_image +
                  (size_t)fp->imgdata.sizes.top_margin * fpitch +
                  fp->imgdata.sizes.left_margin;
      }
      negicc::PixelShiftMergeArgs pm{};
      pm.bayer_f0 = cfa_origin;
      pm.bayer_f1 = fcfa[0];
      pm.bayer_f2 = fcfa[1];
      pm.bayer_f3 = fcfa[2];
      pm.src_w = src_w;
      pm.src_h = src_h;
      pm.rx = cfa_rx;
      pm.ry = cfa_ry;
      pm.rw = cfa_rw;
      pm.rh = cfa_rh;
      pm.rot_cw = rot_cw;
      pm.hflip = hflip;
      pm.vflip = vflip;
      pm.dst_w = dst_w;
      pm.dst_h = dst_h;
      pm.filters = filters;
      pm.white_level = cfa_white_level;
      pm.rgb_out = cfa_export_rgb.data();
      pm.flat = flat_full;
      pm.flat_x0 = cfa_rx;
      pm.flat_y0 = cfa_ry;
      pm.cfa_pitch = cfa_pitch;
      negicc::cpu_pixel_shift_merge(pm);
      free_libraw_cfa(proc);
      for (int i = 0; i < 3; ++i) {
        ps_extra_procs[i].reset();
      }
    }
    proc->imgdata.sizes.iwidth = dst_w;
    proc->imgdata.sizes.iheight = dst_h;
    proc->imgdata.idata.colors = 3;
    proc->imgdata.params.output_bps = 16;
    printf("Processed image size: %dx%d\n", dst_w, dst_h);
    t_raw_decode_ms += (omp_get_wtime() - t_step5_debayer_0) * 1000.0;
  }

  ParsedFilmProfile film_prof;
  std::vector<uint8_t> embedded_icc;
  std::string bundle_target_name;
  int bundle_target_idx = -1;

  if (is_bundle && !bundle_search) {
    if (use_dino || manual_target_idx < 0 || (!station_egb_given && !(pin_mask & PIN_E))) {
      float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
      int num_feasible = 0;
      if (use_cfa_pipeline) {
        // TODO: Run `compute_bundle_enclosures` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
        // full-resolution `cfa_export_rgb` buffer once single-target enclosure baselines are updated.
        compute_bundle_enclosures(bundle,
                                  negicc::RgbView::from_interleaved_rgb(
                                      cfa_export_rgb.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                                  g_lo, g_hi, scene_ev, num_feasible);
      } else {
        compute_bundle_enclosures(bundle,
                                  (const uint16_t (*)[4])proc->imgdata.image,
                                  proc->imgdata.sizes.iwidth,
                                  proc->imgdata.sizes.iheight,
                                  g_lo, g_hi, scene_ev, num_feasible);
      }
    }
  } else if (!is_bundle) {
    if (!film_prof_arg.empty() && !load_film_profile(film_prof_arg, film_prof)) {
      fprintf(stderr, "ERROR! Cannot load film profile %s\n", film_prof_arg.c_str());
    }
  }

  DinoSolveResult dino_res;

  if (has_auto_res) {
    bundle_target_idx = auto_res.target_idx;
    bundle_target_name = auto_res.target_name;
    film_prof = bundle.targets[bundle_target_idx].film_prof;
    film_prof.rebind_pointers();
    // The solved conversion written as the bundle's crosstalk with an exposure and gains (the film base and
    // exposure ratio folded in), so the sidecar's cli_equivalent reproduces it with --target.
    // Note: `auto_res.row_scale` already folds `normalization_target / fb[r]` into `global_exposure_comp`, `gain_g`,
    // and `gain_b`, so pass `{1, 1, 1}` for the film-base arguments of `adjust_correction_matrix`.
    const std::vector<int> unit_fb = {1, 1, 1};
    global_exposure_comp = (float)(auto_res.row_scale[0] * auto_res.params.e * auto_res.e_norm);
    gain_g = (float)(auto_res.row_scale[1] / auto_res.row_scale[0] * auto_res.params.g);
    gain_b = (float)(auto_res.row_scale[2] / auto_res.row_scale[0] * auto_res.params.b);
    adjust_correction_matrix(r_coeff, g_coeff, b_coeff, global_exposure_comp, gain_g, gain_b,
                             unit_fb, unit_fb, merged_matrix);
    shoulder_info.x_hi = auto_res.params.shoulder;
    if (is_shoulder_auto) {
      shoulder_info.has_q995 = true;
      shoulder_info.q995 = auto_res.shoulder_q995;
    }
    if (shoulder_info.active()) {
      shoulder_info.has_knee = true;
      shoulder_info.knee = auto_res.shoulder_knee;
      shoulder_info.has_guard = true;
      shoulder_info.guard = auto_res.shoulder_guard;
    }
    printf("Auto: %s E %.4f g %.4f b %.4f (resid %.4f, %s), %d inferences\n", auto_res.target_name.c_str(),
           auto_res.params.e, auto_res.params.g, auto_res.params.b, auto_res.residual, to_string(auto_res.status),
           auto_res.inferences);
    auto ms = [](std::chrono::steady_clock::duration d) {
      return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    printf("Timings (ms): render %lld, infer %lld, residual %lld, search %lld, total %lld\n",
           ms(auto_res.render_time), ms(auto_res.infer_time), ms(auto_res.residual_time), ms(auto_res.search_time),
           ms(auto_res.total_time));
    printf("Solved (absolute): -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n\n", global_exposure_comp, gain_g,
           gain_b, bundle_target_idx);
  } else if (use_dino) {
    if (!is_bundle && !film_prof.data.has_profile) {
      fprintf(stderr, "ERROR! The parameter search requires a film profile to evaluate intent losses.\n");
      proc->free_image();
      delete proc;
      return 1;
    }

    SolverConfig cfg;
    cfg.knee = knee;
    cfg.knee_auto = is_knee_auto;
    cfg.shoulder = shoulder_info.x_hi;
    cfg.shoulder_auto = is_shoulder_auto;

    const int max_iters = parser.get<int>("--dino_iters");
    IntentGains user_gains{global_exposure_comp, gain_g, gain_b};

    if (is_bundle) {
      // -E without --target: the bracket the bundle picks without the search.
      bundle_target_idx = manual_target_idx >= 0 ? manual_target_idx
                                                 : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
      printf("Selected target: %s\n", bundle.targets[bundle_target_idx].display_name.c_str());
      const auto& tgt = bundle.targets[bundle_target_idx];
      bundle_target_name = tgt.name;
      film_prof = tgt.film_prof;
      film_prof.rebind_pointers();
      set_target_bounds(cfg, tgt);
    } else {
      cfg.prof = &film_prof.data;
    }

    // TODO: Run `solve_dinov3_intent` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
    // full-resolution `cfa_export_rgb` buffer once single-target solver baselines are updated.
    const bool solve_ok = use_cfa_pipeline
        ? solve_dinov3_intent(engine, cfg, merged_matrix,
                              negicc::RgbView::from_interleaved_rgb(
                                  cfa_export_rgb.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                              user_gains, pin_mask, max_iters, dino_res)
        : solve_dinov3_intent(engine, cfg, merged_matrix, (const uint16_t (*)[4])proc->imgdata.image,
                              proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                              user_gains, pin_mask, max_iters, dino_res);
    if (!solve_ok) {
      fprintf(stderr, "ERROR! DINOv3 intent solver failed.\n");
      proc->free_image();
      delete proc;
      return 1;
    }

    if (is_knee_auto && dino_res.dynamic_knee.active) {
      knee = dino_res.dynamic_knee.knee;
    }
    shoulder_info.x_hi = dino_res.shoulder.x_hi;
    if (is_shoulder_auto) {
      shoulder_info.has_q995 = true;
      shoulder_info.q995 = dino_res.shoulder.q995;
    }
    if (dino_res.shoulder.active) {
      shoulder_info.has_knee = true;
      shoulder_info.knee = dino_res.shoulder.knee;
      shoulder_info.has_guard = true;
      shoulder_info.guard = dino_res.shoulder.guard;
    }

    std::string pins_str = "none";
    if (pin_mask != 0) {
      pins_str = "";
      if (pin_mask & PIN_E) pins_str += "E ";
      if (pin_mask & PIN_G) pins_str += "g ";
      if (pin_mask & PIN_B) pins_str += "b";
      if (!pins_str.empty() && pins_str.back() == ' ') pins_str.pop_back();
    }
    printf("\nDINOv3 solver: grid %dx%d (%dx%d), backend %s, E_c %.3f, bounds [%.3f, %.3f], max iters %d, pins %s\n",
           dino_res.grid_w, dino_res.grid_h, dino_res.target_w, dino_res.target_h,
           engine.backend_name(), dino_res.e_center, dino_res.bounds_lo, dino_res.bounds_hi,
           max_iters, pins_str.c_str());
    if (dino_res.anchor_clamped) {
      printf("Warning: ANCHOR_CLAMPED (median L* 50 not reached within E_c 2^[-4, 4])\n");
    }
    printf(" it      E      g      b   resid    L_L    L_a    L_b  clip%%  P_clip  P_anch       J\n");
    for (const auto& it : dino_res.iterations) {
      if (it.residual < 0.0f) {
        printf("  %c  %5.3f  %5.3f  %5.3f       -  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
               it.tag, it.e, it.g, it.b, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
               it.penalty_clip, it.penalty_anchor, it.objective);
      } else {
        printf("  %c  %5.3f  %5.3f  %5.3f  %6.3f  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
               it.tag, it.e, it.g, it.b, it.residual, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
               it.penalty_clip, it.penalty_anchor, it.objective);
      }
    }
    printf("Status: %s after %d inferences (best it %d, resid %.3f)\n",
           dino_res.status.c_str(), dino_res.total_inferences, dino_res.best_iteration, dino_res.best_residual);
    printf("Solved (relative): E %.3f g %.3f b %.3f\n",
           dino_res.solved_relative.e, dino_res.solved_relative.g, dino_res.solved_relative.b);
    printf("Solved (absolute): -E %.9g --gain_g %.9g --gain_b %.9g\n",
           dino_res.solved_absolute.e, dino_res.solved_absolute.g, dino_res.solved_absolute.b);
    if (dino_res.solved_relative.g < 0.5f || dino_res.solved_relative.g > 2.0f ||
        dino_res.solved_relative.b < 0.5f || dino_res.solved_relative.b > 2.0f) {
      printf("Warning: GAIN_RANGE (relative g or b outside [0.5, 2.0])\n");
    }
    if (dino_res.best_iteration >= 0 && (size_t)dino_res.best_iteration + 1 < dino_res.iterations.size()) {
      const auto& bit = dino_res.iterations[dino_res.best_iteration + 1];
      float log2_ratio = (bit.e > 0.0f) ? std::log2(bit.e) : 0.0f;
      printf("Penalties: clip %.3f (%.2f%% %s 2.50%%), anchor %.3f (log2(E/E_c) %+.3f)\n",
             bit.penalty_clip, bit.clip_pct, bit.clip_pct <= 2.5f ? "<=" : ">", bit.penalty_anchor, log2_ratio);
    }
    printf("Timings (ms): views %.1f, render %.1f, infer %.1f, residual %.1f, search %.1f\n\n",
           dino_res.timing_views_ms, dino_res.timing_render_ms, dino_res.timing_infer_ms,
           dino_res.timing_residual_ms, dino_res.timing_search_ms);

    global_exposure_comp = dino_res.solved_absolute.e;
    gain_g = dino_res.solved_absolute.g;
    gain_b = dino_res.solved_absolute.b;

    adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                             global_exposure_comp,
                             gain_g, gain_b,
                             effective_profile_film_base_rgb,
                             effective_film_base_rgb,
                             merged_matrix);
  } else if (is_bundle) {
    if (manual_target_idx >= 0) {
      bundle_target_idx = manual_target_idx;
    } else {
      bundle_target_idx = select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
    }
    printf("Selected target: %s\n", bundle.targets[bundle_target_idx].display_name.c_str());
    bundle_target_name = bundle.targets[bundle_target_idx].name;
    film_prof = bundle.targets[bundle_target_idx].film_prof;
    film_prof.rebind_pointers();
    if (station_egb_given) {
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(*proc, scan_shutter_s, scan_iso);
      float base_matrix[9];
      double row_scale[3];
      compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, nullptr, false,
                                    base_matrix, nullptr, row_scale);
      negicc::candidate_matrix(base_matrix, station_egb[0], station_egb[1], station_egb[2], merged_matrix);
      global_exposure_comp = (float)(row_scale[0] * (double)station_egb[0]);
      gain_g = (float)(row_scale[1] / row_scale[0] * (double)station_egb[1]);
      gain_b = (float)(row_scale[2] / row_scale[0] * (double)station_egb[2]);
    } else if (!(pin_mask & PIN_E) && bundle.targets[bundle_target_idx].e_center > 0.0f) {
      global_exposure_comp = bundle.targets[bundle_target_idx].e_center;
      adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                               global_exposure_comp,
                               gain_g, gain_b,
                               effective_profile_film_base_rgb,
                               effective_film_base_rgb,
                               merged_matrix);
    }
  }

  if (is_bundle && bundle_target_idx >= 0) {
    embedded_icc = bundle.targets[bundle_target_idx].icc_bytes;
  }

  OutputEncoding encoding = ENCODE_NONE;
  std::string colorspace_str;
  std::string attach_profile;
  cmsHPROFILE hXYZ = nullptr;
  cmsHPROFILE hCustomOut = nullptr;

  if (parser.is_used("--colorspace")) {
    colorspace_str = parser.get<std::string>("--colorspace");
    if (colorspace_str == "srgb") {
      encoding = ENCODE_SRGB;
      attach_profile = "srgb";
      printf("Using elle sRGB TRC as output profile\n");
    } else if (colorspace_str == "srgb-g10" || colorspace_str == "srgb-linear") {
      encoding = ENCODE_SRGB_LINEAR;
      attach_profile = "srgb-g10";
      printf("Using elle sRGB Linear (Gamma = 1.0) as output profile\n");
    } else {
      std::string resolved_custom = resolve_profile_path(colorspace_str);
      hCustomOut = cmsOpenProfileFromFile(resolved_custom.c_str(), "r");
      if (!hCustomOut) {
        fprintf(stderr, "ERROR! Cannot read output ICC profile: %s\n", colorspace_str.c_str());
        delete proc;
        return 1;
      }
      hXYZ = cmsCreateXYZProfile();
      encoding = ENCODE_CUSTOM;
      attach_profile = resolved_custom;
      printf("Reading output ICC profile: %s\n", resolved_custom.c_str());
    }
  } else if (has_profile) {
    encoding = ENCODE_NONE;
    if (!is_bundle) {
      attach_profile = film_prof_arg;
    }
  }

  const int width = proc->imgdata.sizes.iwidth;
  const int height = proc->imgdata.sizes.iheight;
  const double t_start = omp_get_wtime();

  // The shared pipeline (color_pipeline.h) with the frozen shoulder: the stage every consumer runs. Only the sRGB
  // encoding applies it.
  const negicc::PipelineConfig export_pc = make_pipeline_config(film_prof.data, knee, shoulder_info.x_hi, encoding);
  if (!cfa_export_rgb.empty()) {
    uint16_t* rgb3 = cfa_export_rgb.data();
    if (encoding == ENCODE_SRGB || encoding == ENCODE_SRGB_LINEAR) {
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          uint16_t* px = &rgb3[((size_t)j * width + i) * 3];
          const negicc::Rgb v = negicc::convert_pixel(export_pc, merged_matrix, px);
          px[0] = quantize_u16(v.r);
          px[1] = quantize_u16(v.g);
          px[2] = quantize_u16(v.b);
        }
      }
    } else if (encoding == ENCODE_CUSTOM) {
      #pragma omp parallel
      {
        cmsHTRANSFORM thread_xform = cmsCreateTransform(
            hXYZ, TYPE_XYZ_FLT, hCustomOut, TYPE_RGB_16, INTENT_PERCEPTUAL, 0);
        std::vector<float> xyz_row(width * 3);

        #pragma omp for schedule(static)
        for (int j = 0; j < height; ++j) {
          for (int i = 0; i < width; ++i) {
            const uint16_t* px = &rgb3[((size_t)j * width + i) * 3];
            Rgb v = process_pixel(px, merged_matrix, knee, film_prof.data);
            xyz_row[i * 3 + 0] = v.r;
            xyz_row[i * 3 + 1] = v.g;
            xyz_row[i * 3 + 2] = v.b;
          }
          cmsDoTransform(thread_xform, xyz_row.data(), &rgb3[(size_t)j * width * 3], width);
        }
        cmsDeleteTransform(thread_xform);
      }
      cmsCloseProfile(hXYZ);
      cmsCloseProfile(hCustomOut);
    } else {
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          uint16_t* px = &rgb3[((size_t)j * width + i) * 3];
          Rgb v = load_u16(px);
          v = apply_matrix(merged_matrix, v);
          v = apply_knee(v, knee);
          px[0] = quantize_u16(v.r);
          px[1] = quantize_u16(v.g);
          px[2] = quantize_u16(v.b);
        }
      }
    }
  } else if (encoding == ENCODE_SRGB || encoding == ENCODE_SRGB_LINEAR) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        ushort* px = proc->imgdata.image[j * width + i];
        const negicc::Rgb v = negicc::convert_pixel(export_pc, merged_matrix, px);
        px[0] = quantize_u16(v.r);
        px[1] = quantize_u16(v.g);
        px[2] = quantize_u16(v.b);
      }
    }
  } else if (encoding == ENCODE_CUSTOM) {
    #pragma omp parallel
    {
      cmsHTRANSFORM thread_xform = cmsCreateTransform(
          hXYZ, TYPE_XYZ_FLT, hCustomOut, TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
      std::vector<float> xyz_row(width * 3);

      #pragma omp for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          const ushort* px = proc->imgdata.image[j * width + i];
          Rgb v = process_pixel(px, merged_matrix, knee, film_prof.data);
          xyz_row[i * 3 + 0] = v.r;
          xyz_row[i * 3 + 1] = v.g;
          xyz_row[i * 3 + 2] = v.b;
        }
        cmsDoTransform(thread_xform, xyz_row.data(), proc->imgdata.image + j * width, width);
      }
      cmsDeleteTransform(thread_xform);
    }
    cmsCloseProfile(hXYZ);
    cmsCloseProfile(hCustomOut);
  } else {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        ushort* px = proc->imgdata.image[j * width + i];
        Rgb v = load_u16(px);
        v = apply_matrix(merged_matrix, v);
        v = apply_knee(v, knee);
        px[0] = quantize_u16(v.r);
        px[1] = quantize_u16(v.g);
        px[2] = quantize_u16(v.b);
      }
    }
  }

  const double t_end = omp_get_wtime();
  const double t_full_conversion_ms = (t_end - t_start) * 1000.0;
  printf("Processing time: %f\n", t_end - t_start);

  const auto output = parser.get<std::string>("--output");
  printf("Writing TIFF '%s'\n", output.c_str());
  const double t_tiff_0 = omp_get_wtime();
  int ret = write_tiff(proc, attach_profile, output,
                       is_bundle ? embedded_icc : std::vector<uint8_t>{},
                       cfa_export_rgb.empty() ? nullptr : cfa_export_rgb.data());
  const double t_tiff_write_ms = (omp_get_wtime() - t_tiff_0) * 1000.0;

  if (ret != 0 || parser.get<bool>("--no-metadata")) {
    unlink((output + ".json").c_str());
  } else {
    const std::string knee_source = is_knee_auto ? "auto" : (parser.is_used("--knee") ? "cli" : "default");
    std::string cs_name = "camera-linear";
    if (encoding == ENCODE_SRGB) {
      cs_name = "srgb";
    } else if (encoding == ENCODE_SRGB_LINEAR) {
      cs_name = "srgb-linear";
    } else if (encoding == ENCODE_CUSTOM) {
      cs_name = attach_profile;
    } else if (encoding == ENCODE_NONE) {
      cs_name = attach_profile.empty() ? "camera-linear" : attach_profile;
    }

    std::string roi_source = "none";
    if (is_roi_auto) {
      roi_source = (frame_det.conf < kLowFrameConfidence) ? "auto_fallback" : "auto";
    } else if (has_roi) {
      roi_source = "cli";
    }

    write_json_sidecar(output, ac, av, proc, files,
                       orig_w, orig_h,
                       false,
                       has_roi, roi_coords, roi_source,
                       is_roi_auto ? &frame_det : nullptr,
                       rot_cw, hflip, vflip,
                       width, height,
                       film_base_applied, sidecar_film_base_rgb, sidecar_profile_film_base_rgb,
                       film_base_source,
                       frame_det.base_std_d, frame_det.has_base,
                       is_bundle ? bundle.path : film_prof_arg,
                       merged_matrix,
                       global_exposure_comp, gain_g, gain_b,
                       pin_mask,
                       knee, knee_source, shoulder_info,
                       use_dino && !has_auto_res, dino_res,
                       dino_path, use_dino ? std::string(engine.backend_used()) : dino_backend,
                       cs_name,
                       t_raw_decode_ms, t_frame_det_ms, t_full_conversion_ms, t_tiff_write_ms,
                       is_bundle, bundle_target_name, bundle_target_idx,
                       has_auto_res ? &auto_res : nullptr, auto_frame_half, t_auto_ms, bundle.input_space);
  }

  proc->free_image();
  delete proc;
  return ret;
}
