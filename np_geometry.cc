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

#include "np_geometry.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

void normalize_rotation(int& rot_cw, bool& hflip, bool& vflip) {
  rot_cw = ((rot_cw % 360) + 360) % 360;
  if (rot_cw == 180) {
    hflip = !hflip;
    vflip = !vflip;
    rot_cw = 0;
  }
}

// Where pixel (x, y) of a rw x rh region lands, (dx, dy), when the region is rotated clockwise by rot_cw (0, 90 or
// 270, see normalize_rotation) and then flipped. The result is dst_w x dst_h: rw x rh, or rh x rw when rotated.
static inline void oriented_pos(int x, int y, int rw, int rh, int dst_w, int dst_h, int rot_cw, bool hflip,
                                bool vflip, int& dx, int& dy) {
  const int x1 = (rot_cw == 0) ? x : ((rot_cw == 90) ? (rh - 1 - y) : y);
  const int y1 = (rot_cw == 0) ? y : ((rot_cw == 90) ? x : (rw - 1 - x));
  dx = hflip ? (dst_w - 1 - x1) : x1;
  dy = vflip ? (dst_h - 1 - y1) : y1;
}

bool extract_roi_and_transform(LibRaw* proc, int rx, int ry, int rw, int rh,
                               int rot_cw, bool hflip, bool vflip) {
  if (!proc || !proc->imgdata.image) return false;

  const int cur_w = proc->imgdata.sizes.iwidth;
  const int cur_h = proc->imgdata.sizes.iheight;
  rx = std::max(0, std::min(rx, cur_w - 1));
  ry = std::max(0, std::min(ry, cur_h - 1));
  rw = std::max(1, std::min(rw, cur_w - rx));
  rh = std::max(1, std::min(rh, cur_h - ry));

  normalize_rotation(rot_cw, hflip, vflip);
  const bool swap_dims = (rot_cw == 90 || rot_cw == 270);
  const int dst_w = swap_dims ? rh : rw;
  const int dst_h = swap_dims ? rw : rh;

  if (rx == 0 && ry == 0 && rw == cur_w && rh == cur_h && rot_cw == 0 && !hflip && !vflip) {
    return true;
  }

  ushort (*new_image)[4] = (ushort (*)[4]) malloc((size_t)dst_w * dst_h * sizeof(*new_image));
  if (!new_image) {
    fprintf(stderr, "ERROR! Failed to allocate memory for transformed ROI (%dx%d)\n", dst_w, dst_h);
    return false;
  }

  const ushort (*src_image)[4] = proc->imgdata.image;

  if (rot_cw == 0 && !hflip && !vflip) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < rh; ++y) {
      memcpy(&new_image[(size_t)y * dst_w],
             &src_image[(size_t)(ry + y) * cur_w + rx],
             (size_t)rw * sizeof(*new_image));
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < rh; ++y) {
      for (int x = 0; x < rw; ++x) {
        int dx = 0, dy = 0;
        oriented_pos(x, y, rw, rh, dst_w, dst_h, rot_cw, hflip, vflip, dx, dy);

        const ushort* s = src_image[(size_t)(ry + y) * cur_w + (rx + x)];
        ushort* d = new_image[(size_t)dy * dst_w + dx];
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = s[3];
      }
    }
  }

  proc->free_image();
  proc->imgdata.image = new_image;
  proc->imgdata.sizes.iwidth = dst_w;
  proc->imgdata.sizes.iheight = dst_h;
  return true;
}

// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
void clamp_rect(int r[4], int bw, int bh) {
  r[2] = std::min(r[2], bw);
  r[3] = std::min(r[3], bh);
  r[0] = std::min(std::max(0, r[0]), bw - r[2]);
  r[1] = std::min(std::max(0, r[1]), bh - r[3]);
}

void detect_frame_or_fallback(const FilmImage& mosaic, const FrameDetectionParams& params,
                                     FrameDetection& det) {
  if (detect_film_frame(mosaic, params, det)) return;
  const std::string why = det.reason;
  det = FrameDetection();
  det.ok = false;
  det.conf = 0.0f;
  det.reason = why.empty() ? "no film frame found" : why;
  fprintf(stderr, "Warning: film frame detection failed (%s); falling back to the full visible frame\n",
          det.reason.c_str());
}
