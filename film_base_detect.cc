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

#include "film_base_detect.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace negicc {
namespace {

// auto_crop.py's thresholds.
constexpr float kBlackFraction = 0.002f;    // darker than this fraction of the p99.9 luminance: black (holder)
constexpr double kClipLevel = 15000.0;      // a channel at or above this: clipped (bare light)
constexpr double kLightChromaGap = 0.8;     // stops of log2(R/B) between film (orange mask) and bare light
constexpr double kLightHoleArea = 0.005;    // a light part inside the scan at most this fraction: sprocket hole
constexpr double kLightOfRef = 0.65;        // bare light luminance at least this fraction of the p99.9 luminance
constexpr double kBasePercentile = 99.5;    // per-channel percentile of film pixels taken as the film base

// numpy's percentile (method linear) of float32 values with a Python float q: the interpolation runs in float32.
float percentile_f32(std::vector<float>& a, double q) {
  const size_t n = a.size();
  if (n == 0) return std::nanf("");
  const double vi = (double)(n - 1) * (q / 100.0);
  size_t prev, next;
  double gamma;
  if (vi >= (double)(n - 1)) {
    prev = next = n - 1;
    gamma = vi + 1.0;  // numpy's previous index is -1 there; the two values are equal
  } else {
    prev = (size_t)std::floor(vi);
    next = prev + 1;
    gamma = vi - (double)prev;
  }
  std::nth_element(a.begin(), a.begin() + prev, a.end());
  const float lo = a[prev];
  float hi = lo;
  if (next != prev) hi = *std::min_element(a.begin() + prev + 1, a.end());
  const float diff = hi - lo;
  if (gamma >= 0.5) return hi - diff * (float)(1.0 - gamma);
  return lo + diff * (float)gamma;
}

// numpy's median of float32 values (the mean of the middle two in float32).
float median_f32(std::vector<float>& a) {
  const size_t n = a.size();
  if (n == 0) return std::nanf("");
  const size_t h = n / 2;
  std::nth_element(a.begin(), a.begin() + h, a.end());
  const float upper = a[h];
  if (n % 2) return upper;
  const float lower = *std::max_element(a.begin(), a.begin() + h);
  const float s = lower + upper;
  return (float)((double)s / 2.0);
}

// numpy's median of uint16 values (float64).
double median_u16(std::vector<uint16_t>& a) {
  const size_t n = a.size();
  const size_t h = n / 2;
  std::nth_element(a.begin(), a.begin() + h, a.end());
  const double upper = (double)a[h];
  if (n % 2) return upper;
  const double lower = (double)*std::max_element(a.begin(), a.begin() + h);
  return (lower + upper) / 2.0;
}

// OpenCV's INTER_AREA reduction of a 3-channel uint16 image (resize.cpp computeResizeAreaTab and ResizeArea_Invoker,
// float weights, rounded to uint16), reading the source rows from |src| as they are needed.
struct AreaTab {
  int di, si;
  float alpha;
};

std::vector<AreaTab> area_tab(int ssize, int dsize, int cn, double scale) {
  std::vector<AreaTab> tab;
  for (int dx = 0; dx < dsize; dx++) {
    const double fsx1 = dx * scale;
    const double fsx2 = fsx1 + scale;
    const double cell_width = std::min(scale, ssize - fsx1);
    int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);
    sx2 = std::min(sx2, ssize - 1);
    sx1 = std::min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.push_back({dx * cn, (sx1 - 1) * cn, (float)((sx1 - fsx1) / cell_width)});
    for (int sx = sx1; sx < sx2; sx++) tab.push_back({dx * cn, sx * cn, (float)(1.0 / cell_width)});
    if (fsx2 - sx2 > 1e-3)
      tab.push_back({dx * cn, sx2 * cn, (float)(std::min(std::min(fsx2 - sx2, 1.0), cell_width) / cell_width)});
  }
  return tab;
}

uint16_t saturate_u16(float v) {
  const long r = std::lrint(v);
  return (uint16_t)std::max(0L, std::min(65535L, r));
}

// auto_crop.downsample: the long side at most kBaseWorkSize, as float32.
std::vector<float> downsample(const RgbView& src, int& ow, int& oh) {
  const int W = src.width, H = src.height;
  const double s = std::min(1.0, kBaseWorkSize / (double)std::max(H, W));
  std::vector<uint16_t> row((size_t)W * 3);
  auto load_row = [&](int y) {
    for (int x = 0; x < W; ++x) src.sample(x, y, &row[(size_t)x * 3]);
  };
  std::vector<float> out;
  if (s >= 1.0) {
    ow = W;
    oh = H;
    out.resize((size_t)W * H * 3);
    for (int y = 0; y < H; ++y) {
      load_row(y);
      for (size_t i = 0; i < row.size(); ++i) out[(size_t)y * W * 3 + i] = (float)row[i];
    }
    return out;
  }
  ow = std::max(1, (int)std::nearbyint(W * s));
  oh = std::max(1, (int)std::nearbyint(H * s));
  const double scale_x = 1.0 / ((double)ow / W), scale_y = 1.0 / ((double)oh / H);
  const std::vector<AreaTab> xtab = area_tab(W, ow, 3, scale_x);
  const std::vector<AreaTab> ytab = area_tab(H, oh, 1, scale_y);
  const int dw = ow * 3;
  std::vector<float> buf(dw), sum(dw, 0.0f);
  std::vector<uint16_t> dst((size_t)dw * oh);
  int prev_dy = ytab.empty() ? 0 : ytab[0].di;
  int loaded = -1;
  for (const AreaTab& yt : ytab) {
    const float beta = yt.alpha;
    if (yt.si != loaded) {
      load_row(yt.si);
      loaded = yt.si;
    }
    std::fill(buf.begin(), buf.end(), 0.0f);
    for (const AreaTab& xt : xtab) {
      const float alpha = xt.alpha;
      const float t0 = buf[xt.di] + row[xt.si] * alpha;
      const float t1 = buf[xt.di + 1] + row[xt.si + 1] * alpha;
      const float t2 = buf[xt.di + 2] + row[xt.si + 2] * alpha;
      buf[xt.di] = t0;
      buf[xt.di + 1] = t1;
      buf[xt.di + 2] = t2;
    }
    if (yt.di != prev_dy) {
      uint16_t* d = &dst[(size_t)prev_dy * dw];
      for (int dx = 0; dx < dw; dx++) {
        d[dx] = saturate_u16(sum[dx]);
        sum[dx] = beta * buf[dx];
      }
      prev_dy = yt.di;
    } else {
      for (int dx = 0; dx < dw; dx++) sum[dx] += beta * buf[dx];
    }
  }
  uint16_t* d = &dst[(size_t)prev_dy * dw];
  for (int dx = 0; dx < dw; dx++) d[dx] = saturate_u16(sum[dx]);
  out.assign(dst.begin(), dst.end());
  return out;
}

// cv2.connectedComponents (8-connected) of mask: labels (0 outside the mask) and the number of labels with 0.
int label_components(const std::vector<uint8_t>& mask, int w, int h, std::vector<int>& labels) {
  labels.assign(mask.size(), 0);
  int n = 1;
  std::vector<int> stack;
  for (int i = 0; i < w * h; ++i) {
    if (!mask[i] || labels[i]) continue;
    labels[i] = n;
    stack.push_back(i);
    while (!stack.empty()) {
      const int p = stack.back();
      stack.pop_back();
      const int px = p % w, py = p / w;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int qx = px + dx, qy = py + dy;
          if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
          const int q = qy * w + qx;
          if (mask[q] && !labels[q]) {
            labels[q] = n;
            stack.push_back(q);
          }
        }
      }
    }
    ++n;
  }
  return n;
}

// The labels on the image's border.
std::vector<uint8_t> border_labels(const std::vector<int>& labels, int w, int h, int n) {
  std::vector<uint8_t> edge(n, 0);
  for (int x = 0; x < w; ++x) edge[labels[x]] = edge[labels[(size_t)(h - 1) * w + x]] = 1;
  for (int y = 0; y < h; ++y) edge[labels[(size_t)y * w]] = edge[labels[(size_t)y * w + w - 1]] = 1;
  return edge;
}

// auto_crop._light_parts: the 8-connected parts of mask that can be bare light (reaching the border, or as small as a
// sprocket hole) and are as bright as light through air.
std::vector<uint8_t> light_parts(const std::vector<uint8_t>& mask, const std::vector<float>& lum, double ref, int w,
                                 int h) {
  std::vector<int> labels;
  const int n = label_components(mask, w, h, labels);
  if (n <= 1) return mask;
  const std::vector<uint8_t> edge = border_labels(labels, w, h, n);
  std::vector<long> area(n, 0);
  std::vector<double> lum_sum(n, 0.0);
  for (size_t i = 0; i < labels.size(); ++i) {
    area[labels[i]]++;
    lum_sum[labels[i]] += (double)lum[i];
  }
  std::vector<uint8_t> keep(n, 0);
  const double hole = kLightHoleArea * (double)mask.size();
  for (int l = 1; l < n; ++l) {
    const double mean = lum_sum[l] / (double)std::max(area[l], 1L);
    keep[l] = (edge[l] || (double)area[l] <= hole) && mean >= kLightOfRef * ref;
  }
  std::vector<uint8_t> out(mask.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = keep[labels[i]];
  return out;
}

// auto_crop._otsu on float32 values, with numpy's histogram (128 equal bins over [min, max]):
// (threshold, low-class mean, high-class mean, low fraction).
void otsu(const std::vector<float>& values, float& t, double& m_lo, double& m_hi, double& f_lo) {
  constexpr int kBins = 128;
  float first = 0.0f, last = 1.0f;
  if (!values.empty()) {
    first = *std::min_element(values.begin(), values.end());
    last = *std::max_element(values.begin(), values.end());
  }
  if (first == last) {
    first = first - 0.5f;
    last = last + 0.5f;
  }
  float edges[kBins + 1];
  const float step = (last - first) / (float)kBins;
  for (int i = 0; i <= kBins; ++i) edges[i] = (float)i * step + first;
  edges[kBins] = last;
  const float denom = last - first;
  long hist[kBins] = {0};
  for (const float v : values) {
    const float f = ((v - first) / denom) * (float)kBins;
    long i = (long)f;
    if (i == kBins) i -= 1;
    if (v < edges[i]) i -= 1;
    if (v >= edges[i + 1] && i != kBins - 1) i += 1;
    hist[i]++;
  }
  double s0[kBins];
  long w0[kBins];
  double s = 0.0;
  long w = 0;
  for (int i = 0; i < kBins; ++i) {
    const float centre = (edges[i] + edges[i + 1]) / 2.0f;
    s += (double)hist[i] * (double)centre;
    w += hist[i];
    s0[i] = s;
    w0[i] = w;
  }
  int best = 0;
  double best_between = 0.0;
  for (int i = 0; i < kBins - 1; ++i) {
    const double m0 = s0[i] / (double)std::max(w0[i], 1L);
    const long w1 = w0[kBins - 1] - w0[i];
    const double m1 = (s0[kBins - 1] - s0[i]) / (double)std::max(w1, 1L);
    const double d = m0 - m1;
    const double between = (double)(w0[i] * w1) * (d * d);
    if (i == 0 || between > best_between) {
      best = i;
      best_between = between;
    }
  }
  t = edges[best + 1];
  m_lo = s0[best] / (double)std::max(w0[best], 1L);
  m_hi = (s0[kBins - 1] - s0[best]) / (double)std::max(w0[kBins - 1] - w0[best], 1L);
  f_lo = (double)w0[best] / (double)w0[kBins - 1];
}

void corrected(const double v[3], const double* m, double out[3]) {
  for (int i = 0; i < 3; ++i) {
    if (!m) {
      out[i] = v[i];
      continue;
    }
    const double a = v[0] * m[i * 3 + 0];
    const double b = v[1] * m[i * 3 + 1];
    const double c = v[2] * m[i * 3 + 2];
    out[i] = (a + b) + c;
  }
}

}  // namespace

void thinnest_film_base(const RgbView& half, float out[3]) {
  int w = 0, h = 0;
  std::vector<float> v = downsample(half, w, h);
  const size_t n = (size_t)w * h;
  for (float& x : v) x = std::max(x, 1.0f);
  std::vector<float> lum(n);
  for (size_t i = 0; i < n; ++i) {
    const float s = (v[i * 3] + v[i * 3 + 1]) + v[i * 3 + 2];
    lum[i] = (float)((double)s / 3.0);
  }
  std::vector<float> tmp(lum);
  const double ref = (double)percentile_f32(tmp, 99.9);
  const float black_level = (float)(kBlackFraction * ref);
  std::vector<uint8_t> black(n), light(n);
  const float clip = (float)kClipLevel;
  bool any_light = false;
  for (size_t i = 0; i < n; ++i) {
    black[i] = lum[i] < black_level;
    light[i] = v[i * 3] >= clip || v[i * 3 + 1] >= clip || v[i * 3 + 2] >= clip;
    any_light |= light[i] != 0;
  }
  if (any_light) light = light_parts(light, lum, ref, w, h);

  std::vector<float> q(n);
  for (size_t i = 0; i < n; ++i) q[i] = std::log2(v[i * 3] / v[i * 3 + 2]);
  size_t n_light = 0, n_cand = 0;
  std::vector<uint8_t> cand(n);
  for (size_t i = 0; i < n; ++i) {
    n_light += light[i];
    cand[i] = !black[i] && !light[i];
    n_cand += cand[i];
  }
  if ((double)n_light / (double)n < 0.005 && n_cand > 64) {
    // Bare light that did not clip: bright, and less orange than the film (the orange mask absorbs blue).
    tmp.clear();
    for (size_t i = 0; i < n; ++i)
      if (cand[i]) tmp.push_back(lum[i]);
    const float med = median_f32(tmp);
    std::vector<uint8_t> bright(n);
    std::vector<float> qb;
    for (size_t i = 0; i < n; ++i) {
      bright[i] = cand[i] && lum[i] >= med;
      if (bright[i]) qb.push_back(q[i]);
    }
    float t;
    double m_lo, m_hi, f_lo;
    otsu(qb, t, m_lo, m_hi, f_lo);
    std::vector<uint8_t> lo(n);
    std::vector<float> b_lo, b_hi;
    for (size_t i = 0; i < n; ++i) {
      lo[i] = bright[i] && q[i] < t;
      if (lo[i]) b_lo.push_back(v[i * 3 + 2]);
      else if (bright[i]) b_hi.push_back(v[i * 3 + 2]);
    }
    if (m_hi - m_lo >= kLightChromaGap && 0.01 <= f_lo && f_lo <= 0.99 && !b_lo.empty() && !b_hi.empty() &&
        median_f32(b_lo) > percentile_f32(b_hi, kBasePercentile)) {
      const std::vector<uint8_t> parts = light_parts(lo, lum, ref, w, h);
      for (size_t i = 0; i < n; ++i) light[i] |= parts[i];
    }
  } else if (any_light) {
    // Bare light partly clipped: its unclipped parts connected to the clipped ones, reaching the border.
    const float light_level = (float)(kLightOfRef * ref);
    std::vector<uint8_t> unclipped(n);
    for (size_t i = 0; i < n; ++i) unclipped[i] = lum[i] >= light_level && !black[i];
    std::vector<int> labels;
    const int nl = label_components(unclipped, w, h, labels);
    if (nl > 1) {
      const std::vector<uint8_t> edge = border_labels(labels, w, h, nl);
      std::vector<uint8_t> keep(nl, 0);
      for (size_t i = 0; i < n; ++i)
        if (light[i]) keep[labels[i]] = 1;
      for (int l = 0; l < nl; ++l) keep[l] = keep[l] && edge[l];
      keep[0] = 0;
      for (size_t i = 0; i < n; ++i) light[i] |= keep[labels[i]];
    }
  }
  // Pixels where light and film blend count as light: dilated by a 5x5 square.
  std::vector<uint8_t> grown(n, 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t any = 0;
      for (int dy = -2; dy <= 2 && !any; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -2; dx <= 2; ++dx) {
          const int xx = x + dx;
          if (xx >= 0 && xx < w && light[(size_t)yy * w + xx]) {
            any = 1;
            break;
          }
        }
      }
      grown[(size_t)y * w + x] = any && !black[(size_t)y * w + x];
    }
  }
  std::vector<uint8_t> film(n);
  size_t n_film = 0;
  for (size_t i = 0; i < n; ++i) n_film += (film[i] = !black[i] && !grown[i]);
  if (n_film < 16)
    for (size_t i = 0; i < n; ++i) film[i] = !black[i];
  for (int c = 0; c < 3; ++c) {
    tmp.clear();
    for (size_t i = 0; i < n; ++i)
      if (film[i]) tmp.push_back(v[i * 3 + c]);
    out[c] = tmp.empty() ? 0.0f : percentile_f32(tmp, kBasePercentile);
  }
}

bool detect_film_base(const RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                      const double* crosstalk, FilmBaseDetection& out) {
  out = FilmBaseDetection{};
  const int W = half.width, H = half.height;
  bool has_box = false;
  int box[4] = {0, 0, 0, 0};
  if (det) {
    out.has_detector = true;
    out.reason = det->reason;
    out.std_d = det->base_std_d;
    out.frame_conf = det->conf;
    out.light_clipped = det->light_clipped;
    if (det->ok && det->has_base && sensor_w > 0 && sensor_h > 0) {
      const double sx = W / (double)sensor_w, sy = H / (double)sensor_h;
      if (std::fabs(sx - sy) > 0.02 * std::max(sx, sy)) {
        out.reason += ",decode_not_sensor_aligned";  // the decode is not the sensor's area scaled (turned)
      } else {
        const int x = det->base_box[0], y = det->base_box[1], w = det->base_box[2], h = det->base_box[3];
        const double x0 = x + kBasePatchInset * w, x1 = x + (1.0 - kBasePatchInset) * w;
        const double y0 = y + kBasePatchInset * h, y1 = y + (1.0 - kBasePatchInset) * h;
        int b[4] = {(int)std::ceil(x0 * sx), (int)std::ceil(y0 * sy), (int)std::floor(x1 * sx), (int)std::floor(y1 * sy)};
        b[0] = std::max(0, b[0]);
        b[1] = std::max(0, b[1]);
        b[2] = std::min(W, b[2]);
        b[3] = std::min(H, b[3]);
        if ((long)(b[2] - b[0]) * (b[3] - b[1]) < kBaseMinSamples) {
          out.reason += ",patch_too_small";
        } else {
          has_box = true;
          box[0] = b[0];
          box[1] = b[1];
          box[2] = b[2] - b[0];
          box[3] = b[3] - b[1];
        }
      }
    }
  }
  thinnest_film_base(half, out.thinnest);
  const double thinnest[3] = {out.thinnest[0], out.thinnest[1], out.thinnest[2]};
  if (has_box) {
    std::vector<uint16_t> ch[3];
    for (auto& c : ch) c.reserve((size_t)box[2] * box[3]);
    uint16_t px[3];
    for (int y = box[1]; y < box[1] + box[3]; ++y) {
      for (int x = box[0]; x < box[0] + box[2]; ++x) {
        half.sample(x, y, px);
        for (int c = 0; c < 3; ++c) ch[c].push_back(px[c]);
      }
    }
    double dec[3], rgb[3];
    for (int c = 0; c < 3; ++c) dec[c] = median_u16(ch[c]);
    corrected(dec, crosstalk, rgb);
    double outshone = -INFINITY;
    for (int c = 0; c < 3; ++c) outshone = std::max(outshone, std::log2(thinnest[c] / std::max(dec[c], 1.0)));
    out.has_outshone = true;
    out.thinnest_over_patch_stops = outshone;
    if (std::min({dec[0], dec[1], dec[2]}) > 0 && std::min({rgb[0], rgb[1], rgb[2]}) > 0 && outshone <= kRebateMargin) {
      out.found = true;
      out.source = "rebate";
      std::copy(dec, dec + 3, out.decode_rgb);
      std::copy(rgb, rgb + 3, out.rgb);
      out.has_box = true;
      std::copy(box, box + 4, out.box);
      return true;
    }
  }
  double rgb[3];
  corrected(thinnest, crosstalk, rgb);
  if (std::min({thinnest[0], thinnest[1], thinnest[2]}) <= 1.0 || std::min({rgb[0], rgb[1], rgb[2]}) <= 0) return false;
  out.found = true;
  out.source = "statistics";
  std::copy(thinnest, thinnest + 3, out.decode_rgb);
  std::copy(rgb, rgb + 3, out.rgb);
  return true;
}

}  // namespace negicc
