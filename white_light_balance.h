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

// The film-base balance of the white-light decode: gains on the camera's own channels, applied to the mosaic before
// the camera matrix. Header-only, so that neg_process (np_raw.cc, load_raw) and negicc-station (its CapturedImage
// decode) compile the same code and balance a scan the same way.

#ifndef NEGICC_WHITE_LIGHT_BALANCE_H
#define NEGICC_WHITE_LIGHT_BALANCE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "libraw/libraw.h"

namespace negicc {

// The gains for the camera channels that put the film base |base| (linear Rec.2020, as the white-light decode of proc
// gives it) on the profile's film base |profile_base|: K = (M^-1 profile_base) / (M^-1 base) per channel, M the camera
// matrix the decode applies (rec2020_rgb x rgb_cam), divided by its largest so no channel is raised and nothing new
// clips. k is left as it is when either base has a camera channel that is not positive.
//
// Why before the matrix: a light of another colour than the profile's gives the base another colour in the camera's
// own channels. Corrected after the matrix has mixed them, the matrix's negative terms drive dense areas below zero
// (clamped: blue shadows and skies with a warm light). And the profile's patches were read as Rec.2020 divided by
// the calibration scan's base, which is M with its rows divided by that base: dividing by another scan's base scales
// the rows by another vector, while after these gains every scan's base is the profile's. For the calibration scan
// itself K is 1.
inline void film_base_balance_gains(const LibRaw& proc, const double base[3], const int profile_base[3], float k[3]) {
  // M = rec2020_rgb x rgb_cam, as LibRaw's convert_to_rgb builds it for output_color 8.
  double m[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      m[i][j] = 0;
      for (int n = 0; n < 3; ++n) m[i][j] += LibRaw_constants::rec2020_rgb[i][n] * proc.imgdata.color.rgb_cam[n][j];
    }
  // adj = det(M) M^-1 (the adjugate). The determinant cancels in the ratio of the two bases; it only gives the sign.
  double adj[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      adj[j][i] = m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3] -
                  m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3];
  const double det = m[0][0] * adj[0][0] + m[0][1] * adj[1][0] + m[0][2] * adj[2][0];
  double gain[3];
  for (int c = 0; c < 3; ++c) {
    double cam = 0, cam_profile = 0;  // the two bases in camera channel c, each times det(M)
    for (int j = 0; j < 3; ++j) {
      cam += adj[c][j] * base[j];
      cam_profile += adj[c][j] * profile_base[j];
    }
    if (!(cam / det > 0 && cam_profile / det > 0)) return;
    gain[c] = cam_profile / cam;
  }
  const double largest = std::max({gain[0], gain[1], gain[2]});
  for (int c = 0; c < 3; ++c) k[c] = (float)(gain[c] / largest);
}

// Multiplies the white-light mosaic (black-subtracted and white-balanced in place, as np_raw.cc's
// prepare_white_light_mosaic leaves it) by k, per camera channel. A CFA mosaic holds one sample a
// photosite, of the photosite's colour; a linear DNG holds three or four a pixel, R, G and B first.
inline void scale_white_light_mosaic(LibRaw* proc, const float k[3]) {
  const auto& S = proc->imgdata.sizes;
  const auto& R = proc->imgdata.rawdata;
  const int samples = R.raw_image ? 1 : R.color4_image ? 4 : 3;
  uint16_t* raw = R.raw_image ? R.raw_image : samples == 4 ? &R.color4_image[0][0] : &R.color3_image[0][0];
  const size_t pitch = S.raw_pitch / 2 / samples;  // pixels per row of the raw buffer
  float tile[36];  // the gain of each photosite of a 6x6 CFA tile (Bayer repeats every 2, X-Trans every 6)
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) {
      const int colour = proc->COLOR(r, c);
      tile[r * 6 + c] = k[colour == 3 ? 1 : colour];  // 3 is the second green
    }
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < S.height; ++y) {
    uint16_t* p = raw + ((size_t)(y + S.top_margin) * pitch + S.left_margin) * samples;
    for (int x = 0; x < S.width; ++x, p += samples) {
      if (samples == 1) {
        p[0] = (uint16_t)std::nearbyint(p[0] * tile[(y % 6) * 6 + x % 6]);
      } else {
        for (int c = 0; c < 3; ++c) p[c] = (uint16_t)std::nearbyint(p[c] * k[c]);
      }
    }
  }
}

}  // namespace negicc

#endif  // NEGICC_WHITE_LIGHT_BALANCE_H
