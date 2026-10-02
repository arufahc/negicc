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

#pragma once

#include <cstdint>
#include <string>

struct FrameDetection {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;                    // image[] coordinates (half-size if -H)
  char axis = 'h';              // 'h' or 'v'
  int n_gaps = 0;
  int n_segments = 0;
  int selected = 0;
  float aspect = 1.0f;
  float conf = 0.0f;
  std::string format = "-";     // "3:2", "1:1", ... or "-"
  std::string reason = "";      // "" or comma-separated codes
  int gap[2][2] = {{-1, -1}, {-1, -1}}; // bounding gaps along axis in image[] coords
  bool has_base = false;
  int base_rgb[3] = {0, 0, 0};
  float base_std_d = 0.0f;
};

// LibRaw-free film frame and rebate detector for camera-scanned negatives.
// image: decoded 4-channel uint16_t buffer (stride 4, RGB in channels 0, 1, 2)
// filters: LibRaw CFA filter pattern (9 for X-Trans, else Bayer)
// axis_hint: 'h' for horizontal strip, 'v' for vertical strip, or 0 / 'a' for auto-detection
bool detect_film_frame(const uint16_t (*image)[4], int width, int height, unsigned filters, bool half_size,
                       char axis_hint, FrameDetection& out);
