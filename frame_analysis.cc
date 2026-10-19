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

#include "frame_analysis.h"

#include <libraw/libraw.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <regex>

namespace {

constexpr double SCENE_TAIL = 0.005;

// ---- Geometry ----------------------------------------------------------------------------------------------

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

// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
}  // namespace

void clamp_rect(int r[4], int bw, int bh) {
  r[2] = std::min(r[2], bw);
  r[3] = std::min(r[3], bh);
  r[0] = std::min(std::max(0, r[0]), bw - r[2]);
  r[1] = std::min(std::max(0, r[1]), bh - r[3]);
}

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
    fprintf(stderr, "Only 3-color Bayer RAW files are supported by --auto: %s\n", path.c_str());
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
