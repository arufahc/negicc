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
#include <cstdio>

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

bool fit_film_format(const int area[4], const std::string& format, int frame[4]) {
  static const struct { const char* name; double l, s; } kFormats[] = {
      {"6x6", 56.0, 56.0}, {"35mm", 36.0, 24.0}, {"645", 56.0, 41.5}, {"6x7", 69.5, 56.0}, {"6x9", 84.0, 56.0}};
  std::copy(area, area + 4, frame);
  if (format.empty() || format == "Free") return true;
  double aspect = 0.0;
  for (const auto& f : kFormats)
    if (format == f.name) aspect = f.l / f.s;
  if (aspect <= 0.0) return false;
  const int x = area[0], y = area[1], w = area[2], h = area[3];
  if (h > w) aspect = 1.0 / aspect;  // in the orientation of the area
  int fw, fh;
  if ((double)w / std::max(1, h) > aspect) {
    fw = (int)std::nearbyint(h * aspect);
    fh = h;
  } else {
    fw = w;
    fh = (int)std::nearbyint(w / aspect);
  }
  // Python's // floors.
  auto floordiv2 = [](int v) { return (v >= 0) ? v / 2 : -((-v + 1) / 2); };
  frame[0] = x + floordiv2(w - fw);
  frame[1] = y + floordiv2(h - fh);
  frame[2] = fw;
  frame[3] = fh;
  return true;
}

bool load_half_size_mosaic(const std::string& path, RgbImage& out, double& shutter_s, double& iso) {
  LibRaw proc;
  int ret;
  if ((ret = proc.open_file(path.c_str())) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot open %s: %s\n", path.c_str(), libraw_strerror(ret));
    return false;
  }
  if ((ret = proc.unpack()) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot unpack %s: %s\n", path.c_str(), libraw_strerror(ret));
    return false;
  }
  if (!proc.imgdata.rawdata.raw_image || proc.imgdata.idata.filters <= 1000 || proc.imgdata.idata.colors != 3) {
    fprintf(stderr, "Only 3-color Bayer RAW files are supported by the parameter search: %s\n", path.c_str());
    return false;
  }
  const int w = proc.imgdata.sizes.width, h = proc.imgdata.sizes.height;
  const int top = proc.imgdata.sizes.top_margin, left = proc.imgdata.sizes.left_margin;
  const int pitch = proc.imgdata.sizes.raw_pitch / 2;
  if (w < 2 || h < 2 || top + h > proc.imgdata.sizes.raw_height || left + w > pitch) {
    fprintf(stderr, "Unexpected RAW geometry in %s\n", path.c_str());
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

  // EXIF exposure time is a rational; LibRaw keeps it as a float. Snap 1/n speeds back to their exact value.
  shutter_s = proc.imgdata.other.shutter > 0 ? (double)proc.imgdata.other.shutter : 0.125;
  if (shutter_s < 1.0) {
    const double n = std::round(1.0 / shutter_s);
    if (n > 0 && std::fabs(1.0 / n - shutter_s) <= 1e-6 * shutter_s) shutter_s = 1.0 / n;
  }
  iso = proc.imgdata.other.iso_speed > 0 ? std::round((double)proc.imgdata.other.iso_speed) : 100.0;
  return true;
}

void detect_film_bbox(const RgbImage& half, int bbox[4]) {
  // The largest 8-connected region of green > 500. The reference takes the largest external contour by its
  // polygon area; for the film frame, which dwarfs any other lit region, the largest region by pixel count is the
  // same one and so is its bounding rectangle.
  const int w = half.width, h = half.height;
  std::vector<int> label((size_t)w * h, -1);
  std::vector<int> stack;
  size_t best_count = 0;
  int best[4] = {0, 0, w, h};
  int next_label = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = (size_t)y * w + x;
      if (label[i] >= 0 || half.rgb[i * 3 + 1] <= 500) continue;
      const int lab = next_label++;
      size_t count = 0;
      int x0 = x, x1 = x, y0 = y, y1 = y;
      label[i] = lab;
      stack.assign(1, (int)i);
      while (!stack.empty()) {
        const int p = stack.back();
        stack.pop_back();
        const int px = p % w, py = p / w;
        count++;
        x0 = std::min(x0, px);
        x1 = std::max(x1, px);
        y0 = std::min(y0, py);
        y1 = std::max(y1, py);
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = px + dx, ny = py + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const size_t j = (size_t)ny * w + nx;
            if (label[j] >= 0 || half.rgb[j * 3 + 1] <= 500) continue;
            label[j] = lab;
            stack.push_back((int)j);
          }
        }
      }
      if (count > best_count) {
        best_count = count;
        best[0] = x0;
        best[1] = y0;
        best[2] = x1 - x0 + 1;
        best[3] = y1 - y0 + 1;
      }
    }
  }
  std::copy(best, best + 4, bbox);
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
