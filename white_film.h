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

// The white-light film model (--white_model film): a colour negative scanned under white light, converted on the
// camera's own channels by a film model instead of a film profile. Steps: docs/white_light_film_model.md, section 4.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cfa_pipeline.h"
#include "dinov3_engine.h"
#include "film_base_detect.h"
#include "nlohmann/json.hpp"

struct WhiteFilmParams {
  std::array<double, 3> p = {1.0, 1.0, 1.0};  // exponents
  std::array<double, 3> w = {1.0, 1.0, 1.0};  // balance, w_R = 1
  double e = 1.0;                             // exposure
};

// One round of the search: the a*b* and L* losses (mean squared z-scores against the intent field) and the slope of
// the rendered L* against the field's.
struct WhiteFilmRound {
  double ab = 0.0, L = 0.0, slope = 0.0;
};

struct WhiteFilm {
  double base[3] = {0.0, 0.0, 0.0};  // the film base in the camera's channels
  WhiteFilmParams params;
  // The search's factors on pG and pB (the frame's own, or --white_slopes), its contrast step and its slope step's
  // factors on pG and pB; not recorded when --white_params gives the parameters.
  std::array<double, 2> ratio_factors = {1.0, 1.0};
  double contrast_scale = 1.0;
  std::array<double, 2> slope_factors = {1.0, 1.0};
  std::vector<WhiteFilmRound> log;
  double time_ms = 0.0;
};

// Steps 2 and 4 to 9 on the half-size decode in camera channels (step 1: RawDecode::kWhiteCamera). half: the whole
// image, in which fb's patch lies; frame: the frame, cropped and oriented. The parameters are searched with engine
// unless given (params: pR,pG,pB,wG,wB,e; slopes: fG,fB in place of the frame's own ratio factors; empty when not
// given). False, with the reason on stderr, when the intent model fails.
bool solve_white_film(const negicc::RgbView& half, const negicc::RgbView& frame, const negicc::FilmBaseDetection& fb,
                      DinoV3Engine& engine, const std::vector<double>& params, const std::vector<double>& slopes,
                      WhiteFilm& wf);

// Step 3 at full size, in place: n pixels of `channels` samples (camera channels in, 16-bit sRGB out).
void render_white_film(const WhiteFilm& wf, uint16_t* px, size_t n, int channels);

// The sidecar's white_film block, and the arguments that replay the conversion (--white_params).
nlohmann::ordered_json white_film_json(const WhiteFilm& wf);
std::vector<std::string> white_film_cli(const WhiteFilm& wf);
