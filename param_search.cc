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

#include "param_search.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <regex>

namespace {

constexpr double SCENE_TAIL = 0.005;

}  // namespace

// ---- Frame statistics --------------------------------------------------------------------------------------

namespace {

double u16_percentile_linear(const std::vector<uint64_t>& hist, uint64_t total, double q) {
  if (total == 0) return 0.0;
  const double pos = q * (double)(total - 1);
  const uint64_t idx_lo = (uint64_t)std::floor(pos);
  const uint64_t idx_hi = (uint64_t)std::ceil(pos);
  const double frac = pos - (double)idx_lo;
  uint64_t cum = 0;
  int v_lo = -1, v_hi = -1;
  for (int v = 0; v < 65536; ++v) {
    cum += hist[v];
    if (v_lo < 0 && cum > idx_lo) v_lo = v;
    if (v_hi < 0 && cum > idx_hi) {
      v_hi = v;
      break;
    }
  }
  if (v_lo < 0) v_lo = 65535;
  if (v_hi < 0) v_hi = 65535;
  return (1.0 - frac) * (double)v_lo + frac * (double)v_hi;
}

}  // namespace

void compute_frame_stats(const RgbImage& frame, float crop_ratio, int max_size, int patch_size, FrameStats& st) {
  const int W = frame.width;
  const InnerCrop crop = compute_inner_crop(frame.width, frame.height, crop_ratio);
  // compute_aspect_preserved_shape of color_pipeline.h (round half away from zero)
  const int p = std::max(1, patch_size);
  const double scale = (double)std::max(max_size, p) / (double)std::max(std::max(crop.w, 1), std::max(crop.h, 1));
  const int gw = std::max(1, (int)std::round((double)std::max(crop.w, 1) * scale / (double)p));
  const int gh = std::max(1, (int)std::round((double)std::max(crop.h, 1) * scale / (double)p));
  const int dino_w = gw * p, dino_h = gh * p;
  const uint16_t* src = frame.rgb.data();

  std::vector<double> log_lut(65536);
  for (int v = 0; v < 65536; ++v) log_lut[v] = std::log(std::max((double)v, 1.0));

  std::vector<uint64_t> g_hist(65536, 0);
  std::vector<double> row_logs((size_t)std::max(0, crop.h) * 3, 0.0);
#pragma omp parallel
  {
    std::vector<uint64_t> loc(65536, 0);
#pragma omp for schedule(static)
    for (int ry = 0; ry < crop.h; ++ry) {
      const uint16_t* row = src + ((size_t)(crop.y0 + ry) * W + crop.x0) * 3;
      double lr = 0.0, lg = 0.0, lb = 0.0;
      for (int rx = 0; rx < crop.w; ++rx, row += 3) {
        loc[row[1]]++;
        lr += log_lut[row[0]];
        lg += log_lut[row[1]];
        lb += log_lut[row[2]];
      }
      row_logs[(size_t)ry * 3] = lr;
      row_logs[(size_t)ry * 3 + 1] = lg;
      row_logs[(size_t)ry * 3 + 2] = lb;
    }
#pragma omp critical
    for (int v = 0; v < 65536; ++v) g_hist[v] += loc[v];
  }
  const uint64_t npx = (uint64_t)crop.w * (uint64_t)crop.h;
  if (npx > 0) {
    double tot[3] = {0, 0, 0};
    for (int ry = 0; ry < crop.h; ++ry)
      for (int c = 0; c < 3; ++c) tot[c] += row_logs[(size_t)ry * 3 + c];
    for (int c = 0; c < 3; ++c) st.log_avg_rgb[c] = std::exp(tot[c] * (1.0 / (double)npx)) / 65535.0;
    const double p_lo = u16_percentile_linear(g_hist, npx, SCENE_TAIL);
    const double p_hi = u16_percentile_linear(g_hist, npx, 1.0 - SCENE_TAIL);
    st.g_lo = (float)std::max(1.0, p_lo);
    st.g_hi = (float)std::max((double)st.g_lo + 1.0, p_hi);
  }

  // Inner crop resampled to the DINOv3 input size in double (area average, bilinear when enlarging), then each
  // patch averaged into the grid.
  std::vector<double> dino((size_t)dino_w * dino_h * 3, 0.0);
  const double sx = (double)crop.w / (double)dino_w;
  const double sy = (double)crop.h / (double)dino_h;
  if (sx >= 1.0 && sy >= 1.0) {
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      const double y0 = dy * sy, y1 = (dy + 1) * sy;
      const int ys = (int)std::floor(y0), ye = std::min(crop.h, (int)std::ceil(y1));
      for (int dx = 0; dx < dino_w; ++dx) {
        const double x0 = dx * sx, x1 = (dx + 1) * sx;
        const int xs = (int)std::floor(x0), xe = std::min(crop.w, (int)std::ceil(x1));
        double acc[3] = {0, 0, 0}, wsum = 0.0;
        for (int yy = ys; yy < ye; ++yy) {
          const double wy = std::min(y1, (double)(yy + 1)) - std::max(y0, (double)yy);
          if (wy <= 0.0) continue;
          const uint16_t* row = src + ((size_t)(crop.y0 + yy) * W + crop.x0) * 3;
          for (int xx = xs; xx < xe; ++xx) {
            const double wx = std::min(x1, (double)(xx + 1)) - std::max(x0, (double)xx);
            if (wx <= 0.0) continue;
            const double w = wx * wy;
            acc[0] += row[(size_t)xx * 3] * w;
            acc[1] += row[(size_t)xx * 3 + 1] * w;
            acc[2] += row[(size_t)xx * 3 + 2] * w;
            wsum += w;
          }
        }
        const double inv = (wsum > 0.0) ? (1.0 / wsum) : 0.0;
        for (int c = 0; c < 3; ++c) dino[((size_t)dy * dino_w + dx) * 3 + c] = acc[c] * inv;
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dino_h; ++dy) {
      const double fy = ((double)dy + 0.5) * sy - 0.5;
      const int y0 = std::clamp((int)std::floor(fy), 0, crop.h - 1);
      const int y1 = std::clamp(y0 + 1, 0, crop.h - 1);
      const double wy1 = fy - (double)y0, wy0 = 1.0 - wy1;
      const uint16_t* row0 = src + ((size_t)(crop.y0 + y0) * W + crop.x0) * 3;
      const uint16_t* row1 = src + ((size_t)(crop.y0 + y1) * W + crop.x0) * 3;
      for (int dx = 0; dx < dino_w; ++dx) {
        const double fx = ((double)dx + 0.5) * sx - 0.5;
        const int x0 = std::clamp((int)std::floor(fx), 0, crop.w - 1);
        const int x1 = std::clamp(x0 + 1, 0, crop.w - 1);
        const double wx1 = fx - (double)x0, wx0 = 1.0 - wx1;
        for (int c = 0; c < 3; ++c) {
          const double top = row0[(size_t)x0 * 3 + c] * wx0 + row0[(size_t)x1 * 3 + c] * wx1;
          const double bot = row1[(size_t)x0 * 3 + c] * wx0 + row1[(size_t)x1 * 3 + c] * wx1;
          dino[((size_t)dy * dino_w + dx) * 3 + c] = top * wy0 + bot * wy1;
        }
      }
    }
  }
  st.grid_w = gw;
  st.grid_h = gh;
  st.grid.assign((size_t)gw * gh * 3, 0);
  const double inv_patch = 1.0 / (double)(p * p);
  for (int gy = 0; gy < gh; ++gy) {
    for (int gx = 0; gx < gw; ++gx) {
      double acc[3] = {0, 0, 0};
      for (int py = 0; py < p; ++py) {
        const double* drow = &dino[((size_t)(gy * p + py) * dino_w + gx * p) * 3];
        for (int px = 0; px < p; ++px, drow += 3) {
          acc[0] += drow[0];
          acc[1] += drow[1];
          acc[2] += drow[2];
        }
      }
      for (int c = 0; c < 3; ++c) {
        st.grid[((size_t)gy * gw + gx) * 3 + c] = (uint16_t)std::clamp(std::round(acc[c] * inv_patch), 0.0, 65535.0);
      }
    }
  }
}

// ---- Highlight ceiling and clipping ------------------------------------------------------------------------

namespace {

// sRGB code values (0-255) of PCS L*a*b* (D50).
// py-ref: negicc_station/src/optimizer.py @ 93be1ec (_pcs_lab_to_srgb)
void pcs_lab_to_srgb(const double lab[3], double out[3]) {
  static const double kD50[3] = {0.9642, 1.0, 0.8249};
  static const double kB[3][3] = {{0.9555766, -0.0230393, 0.0631636},
                                  {-0.0282895, 1.0099416, 0.0210077},
                                  {0.0122982, -0.0204830, 1.3299098}};
  static const double kM[3][3] = {{3.2406255, -1.5372080, -0.4986286},
                                  {-0.9689307, 1.8757561, 0.0415175},
                                  {0.0557101, -0.2040211, 1.0569959}};
  const double fy = (lab[0] + 16.0) / 116.0;
  const double f[3] = {fy + lab[1] / 500.0, fy, fy - lab[2] / 200.0};
  double xyz[3], d65[3];
  for (int c = 0; c < 3; ++c) {
    const double f3 = f[c] * f[c] * f[c];
    xyz[c] = (f3 > 0.008856 ? f3 : (f[c] - 16.0 / 116.0) / 7.787) * kD50[c];
  }
  for (int r = 0; r < 3; ++r) d65[r] = kB[r][0] * xyz[0] + kB[r][1] * xyz[1] + kB[r][2] * xyz[2];
  for (int r = 0; r < 3; ++r) {
    const double lin = std::clamp(kM[r][0] * d65[0] + kM[r][1] * d65[1] + kM[r][2] * d65[2], 0.0, 1.0);
    const double enc = (lin <= 0.0031308) ? lin * 12.92 : 1.055 * std::pow(lin, 1.0 / 2.4) - 0.055;
    out[r] = std::clamp(enc, 0.0, 1.0) * 255.0;
  }
}

}  // namespace

void profile_highlight_ceiling(const std::string& profcheck_output, double ceil[3]) {
  static const std::regex kRow(R"(\]\s+(\S+):\s+\S+\s+\S+\s+\S+\s+->\s+(\S+)\s+(\S+)\s+(\S+)\s+should be)");
  ceil[0] = ceil[1] = ceil[2] = -std::numeric_limits<double>::infinity();
  bool any = false;
  for (auto it = std::sregex_iterator(profcheck_output.begin(), profcheck_output.end(), kRow);
       it != std::sregex_iterator(); ++it) {
    double lab[3], rgb[3];
    try {
      for (int c = 0; c < 3; ++c) lab[c] = std::stod((*it)[2 + c].str());
    } catch (const std::exception&) {
      continue;
    }
    pcs_lab_to_srgb(lab, rgb);
    for (int c = 0; c < 3; ++c) ceil[c] = std::max(ceil[c], rgb[c]);
    any = true;
  }
  if (!any) ceil[0] = ceil[1] = ceil[2] = 255.0;
}

namespace {

double percentile_from_hist_L(const std::vector<uint64_t>& hist, double tot, double q) {
  if (tot <= 0.0) return 0.0;
  const double target = tot * q;
  double cum = 0.0;
  int k = 0;
  for (; k < kLabBins; ++k) {
    cum += (double)hist[k];
    if (cum >= target) break;
  }
  return (std::min(k, kLabBins - 1) + 0.5) * (100.0 / kLabBins);
}
}  // namespace

void highlight_clip(const RgbImage& img, const InnerCrop& crop_in, const float* matrices, int n, float knee,
                    const ProfileData& prof, const float ceil_rgb[3], float* out) {
  if (n <= 0) return;
  const InnerCrop crop = (crop_in.w > 0 && crop_in.h > 0) ? crop_in : InnerCrop{0, 0, img.width, img.height};
  const double tot = (double)crop.w * crop.h;
  if (tot <= 0) {
    std::memset(out, 0, (size_t)n * 5 * sizeof(float));
    return;
  }
  float ceil_lin[3];
  for (int k = 0; k < 3; ++k) {
    const float c_u8 = (ceil_rgb[k] > 1.0001f) ? ceil_rgb[k] : (ceil_rgb[k] * 255.0f);
    const float thr_u8 = std::min(std::floor(c_u8) + 1.0f, 255.0f);
    ceil_lin[k] = srgb_decode(std::max(0.0f, (thr_u8 - 0.5f) / 255.0f));
  }
  for (int c = 0; c < n; ++c) {
    const float* m = matrices + (size_t)c * 9;
    uint64_t exc[3] = {0, 0, 0}, wc = 0;
    std::vector<uint64_t> hist(kLabBins, 0);
#pragma omp parallel
    {
      uint64_t l_exc[3] = {0, 0, 0}, l_wc = 0;
      std::vector<uint64_t> l_hist(kLabBins, 0);
#pragma omp for schedule(static)
      for (int y = crop.y0; y < crop.y0 + crop.h; ++y) {
        const uint16_t* row = &img.rgb[((size_t)y * img.width + crop.x0) * 3];
        for (int x = 0; x < crop.w; ++x, row += 3) {
          const Rgb lin = convert_pixel_linear(row, m, knee, prof);
          if (lin.r >= ceil_lin[0]) l_exc[0]++;
          if (lin.g >= ceil_lin[1]) l_exc[1]++;
          if (lin.b >= ceil_lin[2]) l_exc[2]++;
          if (linear_channel_white_clipped(lin.r) || linear_channel_white_clipped(lin.g) ||
              linear_channel_white_clipped(lin.b)) {
            l_wc++;
          }
          l_hist[lab_bin(linear_srgb_to_L(lin))]++;
        }
      }
#pragma omp critical
      {
        for (int k = 0; k < 3; ++k) exc[k] += l_exc[k];
        wc += l_wc;
        for (int k = 0; k < kLabBins; ++k) hist[k] += l_hist[k];
      }
    }
    float* o = out + (size_t)c * 5;
    for (int k = 0; k < 3; ++k) o[k] = (float)((double)exc[k] / tot * 100.0);
    o[3] = (float)((double)wc / tot * 100.0);
    o[4] = (float)percentile_from_hist_L(hist, tot, 0.50);
  }
}

void render_srgb_u8(const RgbImage& img, const float* m, float knee, const ProfileData& prof, uint8_t* out) {
  const long long n = (long long)img.width * img.height;
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < n; ++i) {
    const Rgb v = convert_pixel_srgb(&img.rgb[(size_t)i * 3], m, knee, false, 1.0f, prof);
    out[i * 3] = quantize_u8(v.r);
    out[i * 3 + 1] = quantize_u8(v.g);
    out[i * 3 + 2] = quantize_u8(v.b);
  }
}
