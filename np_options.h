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

// neg_process's command line: the options, parsed and checked on their own (what needs the profile is checked by
// prepare, np_pipeline.h).

#pragma once

#include <string>
#include <vector>

#include "film_frame_detector.h"
#include "input_curves.h"
#include "np_conversion.h"
#include "np_input.h"

struct Options {
  int argc = 0;
  char** argv = nullptr;                       // the command line, for the sidecar

  // Input
  std::vector<std::string> files;              // one RAW, DNG, TIFF or JPEG, three captures, or a pixel shift's four
  InputMode input_mode = InputMode::kRaw;
  bool tiff_input = false;                     // one TIFF: camera RGB with a tri bundle, else Rec.2020
  const char* implied_light = nullptr;         // the light the input implies ("tri", "d65"), nullptr for a RAW
  int quality = -1;                            // --quality
  bool no_crop = false;                        // --no_crop
  std::string flat;                            // --flat, empty when not given

  // Profile
  std::string film_stock, light_source, film_profile;  // --film_stock, --light_source, -p (empty when not given)
  std::string target;                          // --target, empty when not given
  bool target_given = false;

  // Conversion parameters
  std::vector<float> r_coeff, g_coeff, b_coeff;  // -r, -g, -b
  bool row_given[3] = {false, false, false};
  float exposure_comp = 1.0f;                  // -E
  bool exposure_given = false;                 // -E (the search keeps it)
  float gain_g = 1.0f, gain_b = 1.0f;          // --gain_g, --gain_b (given: no search)
  bool gain_g_given = false, gain_b_given = false;
  float brightness = 0.0f;                     // --brightness (stops of the grey's luminance)
  float tint = 0.0f;                           // --tint (a* of the grey)
  float warmth = 0.0f;                         // --warmth (b* of the grey)
  bool knobs_given = false;                    // --brightness, --tint or --warmth given

  // Geometry
  int roi[4] = {0, 0, 0, 0};                   // --roi x y w h (sensor pixels)
  bool has_roi = false, roi_auto = false;      // --roi given as numbers / as 'auto'
  char roi_axis_hint = 0;                      // --roi auto h|v
  FilmFormat film_format = FILM_35MM;          // --film_format
  bool film_format_given = false;              // --film_format was given (else 35mm for --roi auto)
  int rot_cw = 0;                              // --rot_cw (or the --roi's fifth value)
  bool hflip = false, vflip = false;

  // Parameter search
  // --dino: the intent adapter (an adapter.onnx or its directory) and the neutral and tone heads' files that replace
  // the built-in ones; empty: the built-in one.
  std::string dino_adapter, neutral_head_file, tone_head_file;
  bool offset_scan = true;                     // --offset_scan on|off: the network's magenta-green offset (white light)
  bool tone_stage = true;                      // --tone_stage on|off: the learned lightness curve (white light, srgb)
  // Recorded values to replay (a sidecar's cli_equivalent), with given values and no search:
  // --offset_scan theta:dG,dB,kG,kB,pivG,pivB,diagG,diagB,baseG,baseB (OffsetTables' G and B numbers) and
  // --tone_stage curve:black,white,gamma,contrast,chroma (ToneCurve).
  bool offset_replay_given = false;
  double offset_replay[10] = {};
  bool tone_replay_given = false;
  double tone_replay[5] = {};
  std::string dino_backend = "auto";
  int dino_iters = 4;
  bool dino_iters_given = false;

  // Fitted G/B input curves (input_curves.h)
  bool fit_curves = false;                     // --fit_curves on
  // --fit_curves auto, the default: the curves on every white-light bundle search with -P srgb or srgb-g10, nothing
  // elsewhere (TriLED, another or no -P, given parameters). --fit_curves off turns it off.
  bool fit_curves_auto = true;
  bool input_curves_given = false;             // --input_curves (replayed, not fitted)
  negicc::InputCurves input_curves;

  // Output
  std::string output;                          // -o
  std::string colorspace;                      // -P, empty when not given
  bool no_metadata = false;
};

// Parses and checks the command line into o: 0 to convert, 1 on an error (reported), -1 when there is nothing more to
// do (--list_profiles, which it prints).
int parse_options(int ac, char* av[], Options& o);
