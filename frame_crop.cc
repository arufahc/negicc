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

#include "frame_crop.h"

#include <algorithm>
#include <cmath>

namespace {

// py-ref: negicc_station/src/pipeline/kernels_cpu.cpp @ 93be1ec (map_inverse_geometry_cpu)
inline void map_inverse_geometry(int x_dst, int y_dst, int dst_w, int dst_h, int rw, int rh, int rot_cw,
                                 bool hflip, bool vflip, int& x_rel, int& y_rel) {
  rot_cw = ((rot_cw % 360) + 360) % 360;
  if (rot_cw == 180) {
    hflip = !hflip;
    vflip = !vflip;
    rot_cw = 0;
  }
  const int x1 = hflip ? (dst_w - 1 - x_dst) : x_dst;
  const int y1 = vflip ? (dst_h - 1 - y_dst) : y_dst;
  if (rot_cw == 0) {
    x_rel = x1;
    y_rel = y1;
  } else if (rot_cw == 90) {
    x_rel = y1;
    y_rel = rh - 1 - x1;
  } else {
    x_rel = rw - 1 - y1;
    y_rel = x1;
  }
}

}  // namespace

// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
void clamp_rect(int r[4], int bw, int bh) {
  r[2] = std::min(r[2], bw);
  r[3] = std::min(r[3], bh);
  r[0] = std::min(std::max(0, r[0]), bw - r[2]);
  r[1] = std::min(std::max(0, r[1]), bh - r[3]);
}

void cut_and_orient(const HalfSizeImage& src, int x, int y, int w, int h, int rot_cw, bool hflip, bool vflip,
                    RgbImage& out) {
  const int rot = ((rot_cw % 360) + 360) % 360;
  const bool swap = (rot == 90 || rot == 270);
  out.width = swap ? h : w;
  out.height = swap ? w : h;
  out.rgb.resize((size_t)out.width * out.height * 3);
#pragma omp parallel for schedule(static)
  for (int y2 = 0; y2 < out.height; ++y2) {
    for (int x2 = 0; x2 < out.width; ++x2) {
      int xr = 0, yr = 0;
      map_inverse_geometry(x2, y2, out.width, out.height, w, h, rot, hflip, vflip, xr, yr);
      const uint16_t* s = src.px[(size_t)(y + yr) * src.width + (x + xr)];
      uint16_t* d = &out.rgb[((size_t)y2 * out.width + x2) * 3];
      d[0] = s[0];
      d[1] = (uint16_t)(((uint32_t)s[1] + s[3]) >> 1);
      d[2] = s[2];
    }
  }
}

InnerCrop compute_inner_crop(int w, int h, float crop_ratio) {
  if (w <= 0 || h <= 0) return {0, 0, 0, 0};
  if (crop_ratio >= 1.0f) return {0, 0, w, h};
  const int x0 = std::clamp(static_cast<int>(roundf(w * (1.0f - crop_ratio) * 0.5f)), 0, (w - 1) / 2);
  const int y0 = std::clamp(static_cast<int>(roundf(h * (1.0f - crop_ratio) * 0.5f)), 0, (h - 1) / 2);
  return {x0, y0, w - 2 * x0, h - 2 * y0};
}
