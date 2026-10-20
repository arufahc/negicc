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

#include <libraw/libraw.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>

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

inline int bayer_color(int r, int c, unsigned filters) { return (filters >> (((r << 1 & 14) + (c & 1)) << 1)) & 3; }

inline uint16_t px_reflect(const uint16_t* m, int r, int c, int h, int w) {
  int rr = (r < 0) ? -r : (r >= h ? 2 * (h - 1) - r : r);
  int cc = (c < 0) ? -c : (c >= w ? 2 * (w - 1) - c : c);
  rr = std::max(0, std::min(h - 1, rr));
  cc = std::max(0, std::min(w - 1, cc));
  return m[(size_t)rr * w + cc];
}

}  // namespace

// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
void clamp_rect(int r[4], int bw, int bh) {
  r[2] = std::min(r[2], bw);
  r[3] = std::min(r[3], bh);
  r[0] = std::min(std::max(0, r[0]), bw - r[2]);
  r[1] = std::min(std::max(0, r[1]), bh - r[3]);
}

bool load_half_size_mosaic(const std::string& path, RgbImage& out, float& shutter_s, float& iso) {
  LibRaw proc;
  int ret;
  if ((ret = proc.open_file(path.c_str())) != LIBRAW_SUCCESS) {
    std::cerr << "Cannot open " << path << ": " << libraw_strerror(ret) << std::endl;
    return false;
  }
  if ((ret = proc.unpack()) != LIBRAW_SUCCESS) {
    std::cerr << "Cannot unpack " << path << ": " << libraw_strerror(ret) << std::endl;
    return false;
  }
  if (!proc.imgdata.rawdata.raw_image || proc.imgdata.idata.filters <= 1000 || proc.imgdata.idata.colors != 3) {
    std::cerr << "Only 3-color Bayer RAW files are supported by the parameter search: " << path << std::endl;
    return false;
  }
  const int w = proc.imgdata.sizes.width, h = proc.imgdata.sizes.height;
  const int top = proc.imgdata.sizes.top_margin, left = proc.imgdata.sizes.left_margin;
  const int pitch = proc.imgdata.sizes.raw_pitch / 2;
  if (w < 2 || h < 2 || top + h > proc.imgdata.sizes.raw_height || left + w > pitch) {
    std::cerr << "Unexpected RAW geometry in " << path << std::endl;
    return false;
  }
  const unsigned filters = proc.imgdata.idata.filters;
  const int black = proc.imgdata.color.black;
  int cblk[4];
  for (int i = 0; i < 4; ++i) cblk[i] = black + (int)proc.imgdata.color.cblack[i];
  const uint16_t* raw = proc.imgdata.rawdata.raw_image;

  // Black-subtracted visible mosaic.
  std::vector<uint16_t> mosaic((size_t)w * h);
#pragma omp parallel for schedule(static)
  for (int r = 0; r < h; ++r) {
    const uint16_t* src = &raw[(size_t)(r + top) * pitch + left];
    const int blk[2] = {cblk[bayer_color(r, 0, filters)], cblk[bayer_color(r, 1, filters)]};
    for (int c = 0; c < w; ++c) {
      const int v = src[c] - blk[c & 1];
      mosaic[(size_t)r * w + c] = v > 0 ? (uint16_t)v : 0;
    }
  }

  // One pixel per 2x2 quad: R, (G1 + G2) >> 1, B.
  out.width = w / 2;
  out.height = h / 2;
  out.rgb.assign((size_t)out.width * out.height * 3, 0);
#pragma omp parallel for schedule(static)
  for (int y2 = 0; y2 < out.height; ++y2) {
    for (int x2 = 0; x2 < out.width; ++x2) {
      const int r = y2 * 2, c = x2 * 2;
      uint16_t R = 0, B = 0;
      uint32_t g_sum = 0;
      for (int k = 0; k < 4; ++k) {
        const int rr = r + (k >> 1), cc = c + (k & 1);
        const uint16_t s = px_reflect(mosaic.data(), rr, cc, h, w);
        const int col = bayer_color(rr, cc, filters);
        if (col & 1) g_sum += s;
        else if (col == 0) R = s;
        else B = s;
      }
      uint16_t* o = &out.rgb[((size_t)y2 * out.width + x2) * 3];
      o[0] = R;
      o[1] = (uint16_t)(g_sum >> 1);
      o[2] = B;
    }
  }

  shutter_s = proc.imgdata.other.shutter > 0 ? proc.imgdata.other.shutter : 0.125f;
  iso = proc.imgdata.other.iso_speed > 0 ? std::round(proc.imgdata.other.iso_speed) : 100.0f;
  return true;
}

void cut_and_orient(const RgbImage& src, int x, int y, int w, int h, int rot_cw, bool hflip, bool vflip,
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
      const uint16_t* s = &src.rgb[((size_t)(y + yr) * src.width + (x + xr)) * 3];
      uint16_t* d = &out.rgb[((size_t)y2 * out.width + x2) * 3];
      d[0] = s[0];
      d[1] = s[1];
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
