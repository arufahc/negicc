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

#include "negicc_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace negicc {
namespace {

// auto_crop.py's thresholds.
constexpr float kBlackFraction = 0.002f;    // darker than this fraction of the p99.9 luminance: black (holder)
constexpr double kLightChromaGap = 0.8;     // stops of log2(R/B) between film (orange mask) and bare light
constexpr double kLightHoleArea = 0.005;    // a light part inside the scan at most this fraction: sprocket hole
constexpr double kLightOfRef = 0.65;        // bare light luminance at least this fraction of the p99.9 luminance
constexpr int kCarrierPad = 3;              // dark-carrier dilation radius (px on a 1024-px-wide work image)

// numpy's percentile (method linear, Hyndman and Fan's 1996 definition 7) of float32 values with a Python float q: the
// interpolation runs in float32.
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

// numpy's median of uint16 values (float64); NaN for none, as numpy's.
double median_u16(std::vector<uint16_t>& a) {
  const size_t n = a.size();
  if (n == 0) return std::nan("");
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
  std::vector<uint16_t> dst((size_t)dw * oh);
  // The ytab entries of each output row (they are contiguous). The output rows are then made on all threads: each
  // is a sum over its own source rows only, read and added as one thread would, in the same order.
  // Measured: on a 192-thread machine a read of a 61-megapixel scan with a flat field (two work images) takes 370 ms
  // on one thread and gains up to 96 threads (about 20 ms), then loses at 192; on a Jetson Orin Nano it gains up to
  // all 6 cores (628 to 153 ms). It takes the program's thread count, as every region does (AGENTS.md, "Parallel
  // regions").
  std::vector<std::pair<size_t, size_t>> groups;
  for (size_t i = 0; i < ytab.size();) {
    size_t j = i + 1;
    while (j < ytab.size() && ytab[j].di == ytab[i].di) ++j;
    groups.push_back({i, j});
    i = j;
  }
  #pragma omp parallel
  {
    std::vector<uint16_t> trow((size_t)W * 3);
    std::vector<float> buf(dw), sum(dw);
    // The source row in trow. An output row's last source row is usually the next output row's first, and a thread's
    // output rows are consecutive, so it is kept across them and read once.
    int loaded = -1;
    #pragma omp for schedule(static)
    for (long gi = 0; gi < (long)groups.size(); ++gi) {
      for (size_t e = groups[gi].first; e < groups[gi].second; ++e) {
        const AreaTab& yt = ytab[e];
        const float beta = yt.alpha;
        if (yt.si != loaded) {
          for (int x = 0; x < W; ++x) src.sample(x, yt.si, &trow[(size_t)x * 3]);
          loaded = yt.si;
        }
        std::fill(buf.begin(), buf.end(), 0.0f);
        for (const AreaTab& xt : xtab) {
          const float alpha = xt.alpha;
          const float t0 = buf[xt.di] + trow[xt.si] * alpha;
          const float t1 = buf[xt.di + 1] + trow[xt.si + 1] * alpha;
          const float t2 = buf[xt.di + 2] + trow[xt.si + 2] * alpha;
          buf[xt.di] = t0;
          buf[xt.di + 1] = t1;
          buf[xt.di + 2] = t2;
        }
        if (e == groups[gi].first) {
          // The first output row's sum starts from 0.0f plus its first product; every later row's from its first
          // product alone. (The two differ only in the sign of a zero.)
          if (gi == 0) {
            for (int dx = 0; dx < dw; dx++) {
              sum[dx] = 0.0f;
              sum[dx] += beta * buf[dx];
            }
          } else {
            for (int dx = 0; dx < dw; dx++) sum[dx] = beta * buf[dx];
          }
        } else {
          for (int dx = 0; dx < dw; dx++) sum[dx] += beta * buf[dx];
        }
      }
      uint16_t* d = &dst[(size_t)ytab[groups[gi].first].di * dw];
      for (int dx = 0; dx < dw; dx++) d[dx] = saturate_u16(sum[dx]);
    }
  }
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

// The labels on the image's border, or (when given) bordering the dark carrier (lum < 0.01 * ref, widened by
// kCarrierPad * w / 1024 px across the blended edge between light and carrier, as in film_frame_detector.cc).
std::vector<uint8_t> border_labels(const std::vector<int>& labels, int w, int h, int n,
                                   const std::vector<uint8_t>* near_carrier = nullptr) {
  std::vector<uint8_t> edge(n, 0);
  for (int x = 0; x < w; ++x) edge[labels[x]] = edge[labels[(size_t)(h - 1) * w + x]] = 1;
  for (int y = 0; y < h; ++y) edge[labels[(size_t)y * w]] = edge[labels[(size_t)y * w + w - 1]] = 1;
  for (size_t i = 0; i < labels.size(); ++i)
    if (near_carrier && (*near_carrier)[i]) edge[labels[i]] = 1;
  return edge;
}

// auto_crop._light_parts: the 8-connected parts of mask that can be bare light (reaching the border or dark carrier,
// or as small as a sprocket hole) and are as bright as light through air.
std::vector<uint8_t> light_parts(const std::vector<uint8_t>& mask, const std::vector<float>& lum, double ref, int w,
                                 int h, const std::vector<uint8_t>* near_carrier = nullptr) {
  std::vector<int> labels;
  const int n = label_components(mask, w, h, labels);
  if (n <= 1) return mask;
  const std::vector<uint8_t> edge = border_labels(labels, w, h, n, near_carrier);
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

// The mask dilated by a 5x5 square: where light and film blend.
std::vector<uint8_t> grow5(const std::vector<uint8_t>& mask, int w, int h) {
  std::vector<uint8_t> grown(mask.size(), 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t any = 0;
      for (int dy = -2; dy <= 2 && !any; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -2; dx <= 2; ++dx) {
          const int xx = x + dx;
          if (xx >= 0 && xx < w && mask[(size_t)yy * w + xx]) {
            any = 1;
            break;
          }
        }
      }
      grown[(size_t)y * w + x] = any;
    }
  }
  return grown;
}

// The parts of light (8-connected) whose median blue is above the kBasePercentile percentile of blue over the film
// outside them (not black, not within light's 5x5 growth): film transmits no more blue than its base, which its
// orange mask darkens in blue, and bare light transmits more.
std::vector<uint8_t> bluer_than_film(const std::vector<uint8_t>& light, const std::vector<float>& v,
                                     const std::vector<uint8_t>& black, int w, int h) {
  const std::vector<uint8_t> grown = grow5(light, w, h);
  std::vector<float> b_film;
  for (size_t i = 0; i < light.size(); ++i)
    if (!black[i] && !grown[i]) b_film.push_back(v[i * 3 + 2]);
  std::vector<uint8_t> out(light.size(), 0);
  if (b_film.empty()) return out;
  const float b_ref = percentile_f32(b_film, kBasePercentile);
  std::vector<int> labels;
  const int n = label_components(light, w, h, labels);
  std::vector<std::vector<float>> b_part(n);
  for (size_t i = 0; i < light.size(); ++i)
    if (labels[i]) b_part[labels[i]].push_back(v[i * 3 + 2]);
  std::vector<uint8_t> keep(n, 0);
  for (int l = 1; l < n; ++l) keep[l] = median_f32(b_part[l]) > b_ref;
  for (size_t i = 0; i < out.size(); ++i) out[i] = keep[labels[i]];
  return out;
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

// The same samples without the view's flat field: the photosites as the sensor recorded them (pixel x, y of the view
// keeps its place), for the clip.
RgbView without_flat(const RgbView& half) {
  RgbView v = half;
  v.flat = FlatSampleArgs{};
  return v;
}

// A sample of the decode before the flat field clipped in every channel (light, step 2.).
bool clipped_all(const float* px, float clip) { return px[0] >= clip && px[1] >= clip && px[2] >= clip; }

// The share of the box's samples (before the flat field) clipped in some channel (a rebate patch, step 1.).
double clipped_share(const RgbView& sensor, const int box[4], float clip) {
  long clipped = 0;
  uint16_t px[3];
  for (int y = box[1]; y < box[1] + box[3]; ++y) {
    for (int x = box[0]; x < box[0] + box[2]; ++x) {
      sensor.sample(x, y, px);
      clipped += (px[0] >= clip || px[1] >= clip || px[2] >= clip) ? 1 : 0;
    }
  }
  return (double)clipped / std::max(1.0, (double)box[2] * box[3]);
}

}  // namespace

static void thinnest_film_base_impl(const RgbView& half, bool white_light, float white, float out[3],
                                    bool* at_clip = nullptr) {
  if (!half.valid()) {  // no image to read (a size not positive, or a channel without samples): no film
    out[0] = out[1] = out[2] = 0.0f;
    if (at_clip) *at_clip = false;
    return;
  }
  int w = 0, h = 0;
  std::vector<float> v = downsample(half, w, h);
  const size_t n = (size_t)w * h;
  // Sensor units (TriLED): the clip is read on the work image of the samples before the flat field, which divides
  // each pixel by its own gain.
  std::vector<float> sensor;
  if (!white_light && half.flat.gain) {
    int sw = 0, sh = 0;
    sensor = downsample(without_flat(half), sw, sh);
  }
  const float clip = kSensorClip * white;
  const float* sv = white_light ? nullptr : (sensor.empty() ? v.data() : sensor.data());
  for (float& x : v) x = std::max(x, 1.0f);
  std::vector<float> lum(n);
  for (size_t i = 0; i < n; ++i) {
    const float s = (v[i * 3] + v[i * 3 + 1]) + v[i * 3 + 2];
    lum[i] = (float)((double)s / 3.0);
  }
  std::vector<float> tmp(lum);
  const double ref = (double)percentile_f32(tmp, 99.9);
  const float black_level = (float)(kBlackFraction * ref);
  std::vector<uint8_t> near_carrier;
  const std::vector<uint8_t>* carrier = nullptr;
  if (white_light) {
    // On a white-light scan, bare light seen through a holder's aperture borders the dark carrier rather than the image
    // border. Keep only dark components (lum < 0.01 * ref) that reach the image border, hold black holder pixels, and
    // cover >= 1% of the work image before dilating by kCarrierPad * w / 1024 px, so dense interior picture highlights
    // on a pre-cropped scan are not mistaken for a carrier.
    std::vector<uint8_t> dark(n, 0);
    const float dark_level = (float)(0.01 * ref);
    bool any_black = false;
    for (size_t i = 0; i < n; ++i) {
      dark[i] = lum[i] < dark_level;
      any_black |= lum[i] < black_level;
    }
    if (any_black) {
      std::vector<int> dlabels;
      const int nd = label_components(dark, w, h, dlabels);
      if (nd > 1) {
        const std::vector<uint8_t> dedge = border_labels(dlabels, w, h, nd);
        std::vector<long> darea(nd, 0);
        std::vector<uint8_t> has_blk(nd, 0);
        for (size_t i = 0; i < n; ++i) {
          darea[dlabels[i]]++;
          if (lum[i] < black_level) has_blk[dlabels[i]] = 1;
        }
        std::vector<uint8_t> dkeep(nd, 0);
        bool any_carrier = false;
        for (int l = 1; l < nd; ++l) {
          if (dedge[l] && has_blk[l] && (double)darea[l] >= 0.01 * (double)n) {
            dkeep[l] = 1;
            any_carrier = true;
          }
        }
        if (any_carrier) {
          near_carrier.resize(n);
          for (size_t i = 0; i < n; ++i) near_carrier[i] = dkeep[dlabels[i]];
          dilate(near_carrier, w, h, std::max(1, (kCarrierPad * w + 512) / 1024));
          carrier = &near_carrier;
        }
      }
    }
  }
  std::vector<uint8_t> black(n), light(n);
  for (size_t i = 0; i < n; ++i) black[i] = lum[i] < black_level;
  if (!white_light) {
    // Sensor units: bare light is what the sensor clipped in all three channels (the file comment, 2.), in parts that
    // can be bare light.
    bool any_light = false;
    for (size_t i = 0; i < n; ++i) {
      light[i] = white > 0.0f && clipped_all(&sv[i * 3], clip);
      any_light |= light[i] != 0;
    }
    if (any_light) light = light_parts(light, lum, ref, w, h, carrier);
  } else {
    // A white-light decode is in 16-bit linear Rec.2020 (white-balanced by pre_mul, so orange film routinely exceeds a
    // 14-bit sensor's clip in Red). As in detect_film_frame, a channel is clipped when its work-image maxima sit at a
    // plateau (>= 0.05% of pixels within 0.5% of the channel maximum, at >= 13000, about 20% of 65535). A flat stretch
    // of film can form that plateau too, so a clipped part is light only when it is bluer than the film
    // (bluer_than_film).
    float clip_c[3];
    for (int c = 0; c < 3; ++c) {
      float mx = 0.0f;
      for (size_t i = 0; i < n; ++i) mx = std::max(mx, v[i * 3 + c]);
      size_t cnt = 0;
      for (size_t i = 0; i < n; ++i) cnt += v[i * 3 + c] >= 0.995f * mx;
      const bool plateau = cnt >= std::max<size_t>(4, n / 2000) && mx >= 13000.0f;
      clip_c[c] = plateau ? 0.98f * mx : 65000.0f;
    }
    bool any_light = false;
    for (size_t i = 0; i < n; ++i) {
      light[i] = v[i * 3] >= clip_c[0] || v[i * 3 + 1] >= clip_c[1] || v[i * 3 + 2] >= clip_c[2];
      any_light |= light[i] != 0;
    }
    if (any_light) {
      light = light_parts(light, lum, ref, w, h, carrier);
      light = bluer_than_film(light, v, black, w, h);
      // Bare light partly clipped: its unclipped parts connected to the clipped ones, reaching the border or carrier.
      const float light_level = (float)(kLightOfRef * ref);
      std::vector<uint8_t> unclipped(n);
      for (size_t i = 0; i < n; ++i) unclipped[i] = lum[i] >= light_level && !black[i];
      std::vector<int> labels;
      const int nl = label_components(unclipped, w, h, labels);
      if (nl > 1) {
        const std::vector<uint8_t> edge = border_labels(labels, w, h, nl, carrier);
        std::vector<uint8_t> keep(nl, 0);
        for (size_t i = 0; i < n; ++i)
          if (light[i]) keep[labels[i]] = 1;
        for (int l = 0; l < nl; ++l) keep[l] = keep[l] && edge[l];
        keep[0] = 0;
        for (size_t i = 0; i < n; ++i) light[i] |= keep[labels[i]];
      }
    }
    std::vector<float> q(n);
    for (size_t i = 0; i < n; ++i) q[i] = std::log2(v[i * 3] / v[i * 3 + 2]);
    // Bare light that did not clip: on the border, the holder's aperture or the sprocket holes. When the film is
    // strongly orange, mask connected components of neutral bright pixels (reaching the border or holder, or
    // hole-sized) whose blue channel exceeds the orange film's 99.5th percentile.
    std::vector<float> q_cand;
    for (size_t i = 0; i < n; ++i)
      if (!black[i] && !light[i]) q_cand.push_back(q[i]);
    if (q_cand.size() > 64) {
      const float q_med = median_f32(q_cand);
      if (q_med >= 1.2f) {
        const float q_cut = std::min(1.0f, q_med - (float)kLightChromaGap);
        std::vector<float> b_orange, lum_orange;
        for (size_t i = 0; i < n; ++i) {
          if (!black[i] && !light[i] && q[i] >= q_cut) {
            b_orange.push_back(v[i * 3 + 2]);
            lum_orange.push_back(lum[i]);
          }
        }
        if (b_orange.size() > 64) {
          const float b_ref = percentile_f32(b_orange, kBasePercentile);
          const double ref_orange = (double)percentile_f32(lum_orange, 99.9);
          std::vector<uint8_t> neutral(n, 0);
          bool any_neutral = false;
          for (size_t i = 0; i < n; ++i) {
            if (!black[i] && !light[i] && q[i] < q_cut && v[i * 3 + 2] > b_ref) {
              neutral[i] = 1;
              any_neutral = true;
            }
          }
          if (any_neutral) {
            const std::vector<uint8_t> parts = light_parts(neutral, lum, ref_orange, w, h, carrier);
            for (size_t i = 0; i < n; ++i) light[i] |= parts[i];
          }
        }
      }
    }
  }
  // Pixels where light and film blend count as light.
  std::vector<uint8_t> grown = grow5(light, w, h);
  for (size_t i = 0; i < n; ++i) grown[i] = grown[i] && !black[i];
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
  // Sensor units: whether the thinnest film, read on the same pixels before the flat field, is at the clip in a channel
  // (more than 100 - kBasePercentile % of the film clipped there): a base read from it is the clip, not the base.
  if (at_clip) {
    *at_clip = false;
    for (int c = 0; c < 3 && sv && white > 0.0f && !*at_clip; ++c) {
      tmp.clear();
      for (size_t i = 0; i < n; ++i)
        if (film[i]) tmp.push_back(sv[i * 3 + c]);
      *at_clip = !tmp.empty() && percentile_f32(tmp, kBasePercentile) >= clip;
    }
  }
}

void thinnest_film_base(const RgbView& half, float white, float out[3], bool* at_clip) {
  thinnest_film_base_impl(half, false, white, out, at_clip);
}

// Appends a reason to a comma-separated list (no comma before the first).
static void add_reason(std::string& reason, const char* why) {
  if (!reason.empty()) reason += ',';
  reason += why;
}

// Whether the three values of a base, read through the crosstalk matrix, are finite (a matrix whose products overflow
// gives no base).
static bool finite3(const double v[3]) { return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]); }

bool detect_film_base(const RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                      const double* crosstalk, float white, FilmBaseDetection& out) {
  out = FilmBaseDetection{};
  if (!half.valid()) return false;  // no image to read
  const int W = half.width, H = half.height;
  bool has_box = false;
  int box[4] = {0, 0, 0, 0};
  if (det) {
    out.has_detector = true;
    out.reason = det->reason;
    out.std_d = det->base_std_d;
    out.frame_conf = det->conf;
    out.light_clipped = det->light_clipped;
    // A failed or unconfident detection (where neg_process takes the whole image instead of its frame) does not place
    // its patch: on a frame it did not find, the patch may lie on picture.
    if (det->ok && det->has_base && det->conf < kBaseMinFrameConf) add_reason(out.reason, "frame_conf_low");
    if (det->ok && det->has_base && det->conf >= kBaseMinFrameConf && sensor_w > 0 && sensor_h > 0) {
      const double sx = W / (double)sensor_w, sy = H / (double)sensor_h;
      if (std::fabs(sx - sy) > 0.02 * std::max(sx, sy)) {
        add_reason(out.reason, "decode_not_sensor_aligned");  // the decode is not the sensor's area scaled (turned)
      } else {
        const int x = det->base_box[0], y = det->base_box[1], w = det->base_box[2], h = det->base_box[3];
        const double x0 = x + kBasePatchInset * w, x1 = x + (1.0 - kBasePatchInset) * w;
        const double y0 = y + kBasePatchInset * h, y1 = y + (1.0 - kBasePatchInset) * h;
        // Each edge is held within the decode before it becomes an int (a box far outside it, as only a detection the
        // detector did not make can give, would not fit an int); a box that leaves the decode comes out empty below.
        auto edge = [](double v, int n) { return (int)std::min(std::max(v, 0.0), (double)n); };
        const int b[4] = {edge(std::ceil(x0 * sx), W), edge(std::ceil(y0 * sy), H), edge(std::floor(x1 * sx), W),
                          edge(std::floor(y1 * sy), H)};
        if (b[2] <= b[0] || b[3] <= b[1] || (long)(b[2] - b[0]) * (b[3] - b[1]) < kBaseMinSamples) {
          add_reason(out.reason, "patch_too_small");
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
  const bool white_light = crosstalk == nullptr;
  thinnest_film_base_impl(half, white_light, white, out.thinnest, &out.thinnest_clipped);
  if (out.thinnest_clipped) add_reason(out.reason, "thinnest_clipped");
  const double thinnest[3] = {out.thinnest[0], out.thinnest[1], out.thinnest[2]};
  // A patch the sensor clipped reads the clip, not the base (sensor units only: a white-light decode has no white).
  if (has_box && !white_light && white > 0.0f &&
      clipped_share(without_flat(half), box, kSensorClip * white) > kBaseClipFraction) {
    has_box = false;
    out.rebate_clipped = true;
    add_reason(out.reason, "rebate_clipped");
  }
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
    if (std::min({dec[0], dec[1], dec[2]}) > 0 && std::min({rgb[0], rgb[1], rgb[2]}) > 0 && finite3(rgb) &&
        outshone <= kRebateMargin) {
      out.found = true;
      out.source = kFilmBaseSourceRebate;
      std::copy(dec, dec + 3, out.decode_rgb);
      std::copy(rgb, rgb + 3, out.rgb);
      out.has_box = true;
      std::copy(box, box + 4, out.box);
      return true;
    }
  }
  double rgb[3];
  corrected(thinnest, crosstalk, rgb);
  if (std::min({thinnest[0], thinnest[1], thinnest[2]}) <= 1.0 || std::min({rgb[0], rgb[1], rgb[2]}) <= 0 ||
      !finite3(rgb)) {
    return false;
  }
  out.found = true;
  out.source = kFilmBaseSourceStatistics;
  std::copy(thinnest, thinnest + 3, out.decode_rgb);
  std::copy(rgb, rgb + 3, out.rgb);
  return true;
}

}  // namespace negicc
