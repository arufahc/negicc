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

// neg_process's command line: the options, parsed and checked on their own (what needs the profile is checked by
// prepare, np_pipeline.h).

#pragma once

#include <string>
#include <vector>

#include "film_frame_detector.h"
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

  // Geometry
  int roi[4] = {0, 0, 0, 0};                   // --roi x y w h (sensor pixels)
  bool has_roi = false, roi_auto = false;      // --roi given as numbers / as 'auto'
  char roi_axis_hint = 0;                      // --roi auto h|v
  FilmFormat film_format = FILM_FORMAT_AUTO;   // --film_format
  int rot_cw = 0;                              // --rot_cw (or the --roi's fifth value)
  bool hflip = false, vflip = false;

  // Parameter search
  std::string dino_adapter;                    // --dino, empty when not given
  std::string dino_backend = "auto";
  int dino_iters = 4;
  bool dino_iters_given = false;

  // Output
  std::string output;                          // -o
  std::string colorspace;                      // -P, empty when not given
  bool no_metadata = false;
};

// Parses and checks the command line into o: 0 to convert, 1 on an error (reported), -1 when there is nothing more to
// do (--list_profiles, which it prints).
int parse_options(int ac, char* av[], Options& o);
