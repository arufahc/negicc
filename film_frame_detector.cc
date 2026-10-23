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

// Film frame detector (see film_frame_detector.h for the interface).
//
// Sizes used below
//   image   W x H        the input's pixel grid (FilmImage::width x height); never copied
//   lattice nx x ny      samples per channel (FilmImage::samples_x/y): W / step x H / step
//   work    ww x wh      box-averaged lattice, long side <= params.work_size (1024): ~1024 x 683 for 3:2
//   canvas  cw x ch      the work image turned upright by the film's tilt; equals the work size when there is no
//                        tilt and is at most ~9% wider and taller at 5 degrees
//   search  sw x sh      the canvas reduced to <= 320 px on the long side for the rectangle search
//
// Pipeline
//  1. downsample           image -> work: per channel box mean and box maximum (the maximum finds clipped light).
//                          The only pass over the whole input; large images read every k-th sample of each box.
//  2. estimate_rotation    work: projection profiles of near-axis edges of log luminance; the tilt in
//                          +-max_rotation_deg that makes the strip and frame edges the sharpest (0.02 degree steps).
//  3. build_canvas         work -> canvas: the work image turned by -tilt (bilinear), so the film is axis-aligned;
//                          canvas pixels outside the image are invalid.
//  4. classify             canvas: every pixel is LIGHT (bare light, sprocket holes), DARK (holder, carrier, very
//                          dense film), BASE (unexposed film: rebate, gaps between frames) or IMAGE (picture). Levels
//                          are relative to the film base and the bare light found in the same image, so the lamp's
//                          colour and the exposure cancel; space decides what levels cannot (bare light lies outside
//                          the film and reaches the border, a carrier is large, flat and reaches the border, holes
//                          are small). After negicc's earlier density / chroma classes and negicc-station's
//                          auto_crop.
//  5. detect_one_format    for one film format:
//     a. rectangle search  non-image pixels on long lines (line gate) feed a search for the largest rectangle of the
//                          frame's aspect with few non-image pixels inside (integral image), for both orientations;
//                          strip-direction evidence (film running off two opposite borders with non-film lines beside
//                          it) or params.axis_hint picks the orientation.
//     b. strip model       in the "view" (the canvas, transposed when the strip runs along y, so the strip always
//                          runs along view x): the frame's transverse edges from row profiles, px per mm from the
//                          format, the strip cut into frames at gaps that are flat and at the base level, each frame
//                          end typed as a film gap, a hard limit (holder, light, strip end) or open (image border),
//                          the format's frame length used only where an end is open. The primary is the complete
//                          frame nearest the image centre (else the most visible); the neighbours are walked at the
//                          measured pitch. Then the strip's edges and ends (film base), the sprocket holes, and
//                          candidate base patches (gaps and rebate next to the primary).
//     c. refine_edge_full  each edge of the primary moved to the steepest log-luminance step within a few pixels, on
//                          the full-resolution image (bilinear samples, 48 profiles per edge).
//  6. detect_film_frame    the format (or the best of all formats for AUTO), the crop (largest axis-aligned rectangle
//                          inside the tilted frame), and the flattest base patch (film_sample_patch, full resolution).
//
// Memory: besides the input (read in place) the detector allocates work-image sized buffers only: about 30 float or
// byte planes of the canvas size, transiently, for a 1024 px work image about 20 MB at the peak (classify). Each
// allocation is marked "Allocates:" below.
//
// Units: the work image holds sample / white (1.0 is a clipped photosite). Densities and colour ratios are log2
// ("stops") unless named otherwise.

#include "film_frame_detector.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

constexpr float kPi = 3.14159265358979f;

// Classes of canvas pixels (classify). INVALID is outside the image.
enum PixelClass : uint8_t { CL_INVALID = 0, CL_IMAGE = 1, CL_BASE = 2, CL_LIGHT = 3, CL_DARK = 4 };

// Film format geometry in mm. along: the frame's length along the strip; across: its height across the strip;
// film_width: the strip; gap_*: the gap between frames; len_tol: tolerated deviation of a measured frame length
// from along before the nominal length is used; sprockets: perforated (35mm).
struct FormatGeom {
  float along, across, film_width, gap_nominal, gap_min, gap_max, len_tol;
  bool sprockets;
};

FormatGeom format_geom(FilmFormat f) {
  switch (f) {
    case FILM_645: return {41.5f, 56.0f, 61.5f, 5.0f, 1.5f, 10.0f, 0.06f, false};
    case FILM_6X6: return {56.0f, 56.0f, 61.5f, 5.0f, 1.5f, 10.0f, 0.05f, false};
    case FILM_6X7: return {69.5f, 56.0f, 61.5f, 5.0f, 1.5f, 10.0f, 0.06f, false};
    case FILM_6X9: return {84.0f, 56.0f, 61.5f, 4.5f, 1.5f, 10.0f, 0.05f, false};
    case FILM_35MM:
    default: return {36.0f, 24.0f, 35.0f, 2.0f, 0.8f, 5.0f, 0.04f, true};
  }
}

// 35mm perforation (KS): width along the strip, height across, pitch (mm).
constexpr float kHoleW = 1.98f, kHoleH = 2.79f, kHolePitch = 4.75f;

// p-th quantile (0..1). Takes a copy, partially sorted.
float percentile(std::vector<float> v, double p) {
  if (v.empty()) return 0.0f;
  const size_t k = (size_t)std::clamp(p * (double)(v.size() - 1), 0.0, (double)(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Gaussian smoothing of a 1-D profile in place (edges replicated). Allocates: the kernel and one profile copy.
void smooth1d(std::vector<float>& v, float sigma) {
  if (sigma <= 0.3f || v.size() < 3) return;
  const int r = std::max(1, (int)std::ceil(2.5f * sigma));
  std::vector<float> k(2 * r + 1);
  float s = 0.0f;
  for (int i = -r; i <= r; ++i) s += (k[i + r] = std::exp(-0.5f * i * i / (sigma * sigma)));
  for (float& x : k) x /= s;
  const int n = (int)v.size();
  std::vector<float> o(n);
  for (int i = 0; i < n; ++i) {
    float a = 0.0f;
    for (int j = -r; j <= r; ++j) a += k[j + r] * v[std::clamp(i + j, 0, n - 1)];
    o[i] = a;
  }
  v.swap(o);
}

std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += v[i];
  }
  return s;
}

// A float plane of the work, canvas or search size.
struct Plane {
  int w = 0, h = 0;
  std::vector<float> d;
  void init(int W, int H, float v = 0.0f) { w = W; h = H; d.assign((size_t)W * H, v); }
  float& at(int x, int y) { return d[(size_t)y * w + x]; }
  float at(int x, int y) const { return d[(size_t)y * w + x]; }
};

// ---- Input access (in place) ----------------------------------------------------------------------------------

// Sample (i, j) of channel c, black subtracted, in sample units (not normalised), >= 0.
inline float sample_at(const FilmImage& im, int c, int i, int j) {
  const FilmChannel& ch = im.ch[c];
  const float v = (float)ch.data[(ptrdiff_t)i * ch.pitch + (ptrdiff_t)j * ch.stride] - ch.black;
  return v > 0.0f ? v : 0.0f;
}

// Channel c at image position (x, y) (pixel centres at +0.5), bilinear between its lattice samples, normalised.
float channel_bilinear(const FilmImage& im, int c, float x, float y) {
  const FilmChannel& ch = im.ch[c];
  const int nx = im.samples_x(), ny = im.samples_y();
  const float u = (x - 0.5f - (float)ch.x0) / (float)im.step;
  const float v = (y - 0.5f - (float)ch.y0) / (float)im.step;
  const int i0 = std::clamp((int)std::floor(u), 0, nx - 1), j0 = std::clamp((int)std::floor(v), 0, ny - 1);
  const int i1 = std::min(i0 + 1, nx - 1), j1 = std::min(j0 + 1, ny - 1);
  const float fu = std::clamp(u - (float)i0, 0.0f, 1.0f), fv = std::clamp(v - (float)j0, 0.0f, 1.0f);
  const float a = sample_at(im, c, i0, j0) * (1 - fu) + sample_at(im, c, i1, j0) * fu;
  const float b = sample_at(im, c, i0, j1) * (1 - fu) + sample_at(im, c, i1, j1) * fu;
  return (a * (1 - fv) + b * fv) / im.white;
}

float luminance_bilinear(const FilmImage& im, float x, float y) {
  return (channel_bilinear(im, 0, x, y) + channel_bilinear(im, 1, x, y) + channel_bilinear(im, 2, x, y)) *
         (1.0f / 3.0f);
}

// ---- 1. Downsample: image -> work -----------------------------------------------------------------------------

// The work image and its mapping to image pixels: image x = off_x + u * sx for work coordinate u (pixel u covers
// [u, u + 1)), likewise y.
struct WorkImage {
  int w = 0, h = 0;
  float sx = 1.0f, sy = 1.0f;    // image pixels per work pixel
  float off_x = 0.0f, off_y = 0.0f;
  Plane r, g, b;                 // box means, normalised
  Plane mr, mg, mb;              // box maxima, normalised
};

// The work image's long side is min(work_size, the image's long side in pixels), whatever the lattice step, so a
// CFA mosaic gives the same work image size (and the same pixel-sized constants downstream) as the RGB image of the
// same size. The usual case, a lattice at least that large, box-averages it; a smaller lattice (an image under
// ~2 x work_size pixels read as a Bayer mosaic) is interpolated up instead (upsample_lattice), by at most 2x.
// Allocates: the six work planes (6 x ww x wh floats) and, per thread, row accumulators of ww elements.
void upsample_lattice(const FilmImage& im, WorkImage& wi);

void downsample(const FilmImage& im, int work_size, WorkImage& wi) {
  const int nx = im.samples_x(), ny = im.samples_y();
  const int target = std::min({work_size, std::max(im.width, im.height), 2 * std::max(nx, ny)});
  const float s = (float)target / (float)std::max(nx, ny);
  wi.w = std::max(8, (int)std::lround(nx * s));
  wi.h = std::max(8, (int)std::lround(ny * s));
  wi.sx = (float)im.step * (float)nx / (float)wi.w;
  wi.sy = (float)im.step * (float)ny / (float)wi.h;
  // A lattice sample's centre is at x0 + i * step + 0.5; the work grid starts at the mean channel origin.
  const float mx0 = (im.ch[0].x0 + im.ch[1].x0 + im.ch[2].x0) / 3.0f;
  const float my0 = (im.ch[0].y0 + im.ch[1].y0 + im.ch[2].y0) / 3.0f;
  wi.off_x = mx0 + 0.5f - 0.5f * (float)im.step;
  wi.off_y = my0 + 0.5f - 0.5f * (float)im.step;
  for (Plane* p : {&wi.r, &wi.g, &wi.b, &wi.mr, &wi.mg, &wi.mb}) p->init(wi.w, wi.h);
  if (s > 1.0f) {
    upsample_lattice(im, wi);
    return;
  }
  // Large inputs: read every k-th sample of each box along both axes (a box still averages >= 6 x 6 samples).
  const int k = std::max(1, (int)((float)std::max(nx, ny) / (2.5f * (float)std::max(wi.w, wi.h))));
  const float inv_white = 1.0f / im.white;
  #pragma omp parallel
  {
    std::vector<double> acc[3];
    std::vector<float> mx[3];
    for (int c = 0; c < 3; ++c) {
      acc[c].resize(wi.w);
      mx[c].resize(wi.w);
    }
    std::vector<int> cnt(wi.w);
    #pragma omp for schedule(static)
    for (int py = 0; py < wi.h; ++py) {
      const int j0 = (int)((int64_t)py * ny / wi.h);
      const int j1 = std::max(j0 + 1, (int)((int64_t)(py + 1) * ny / wi.h));
      for (int c = 0; c < 3; ++c) {
        std::fill(acc[c].begin(), acc[c].end(), 0.0);
        std::fill(mx[c].begin(), mx[c].end(), 0.0f);
      }
      std::fill(cnt.begin(), cnt.end(), 0);
      for (int j = j0; j < j1; j += k) {
        for (int i = 0; i < nx; i += k) {
          const int px = std::min(wi.w - 1, (int)((int64_t)i * wi.w / nx));
          for (int c = 0; c < 3; ++c) {
            const float v = sample_at(im, c, i, j);
            acc[c][px] += v;
            mx[c][px] = std::max(mx[c][px], v);
          }
          cnt[px]++;
        }
      }
      for (int px = 0; px < wi.w; ++px) {
        const float m = inv_white / (float)std::max(1, cnt[px]);
        wi.r.at(px, py) = (float)(acc[0][px] * m);
        wi.g.at(px, py) = (float)(acc[1][px] * m);
        wi.b.at(px, py) = (float)(acc[2][px] * m);
        wi.mr.at(px, py) = mx[0][px] * inv_white;
        wi.mg.at(px, py) = mx[1][px] * inv_white;
        wi.mb.at(px, py) = mx[2][px] * inv_white;
      }
    }
  }
}

// Work pixel (u, v) takes each channel bilinearly at its centre on the lattice; its "box maximum" is the largest of
// those four lattice samples. Writes the work planes downsample allocated.
void upsample_lattice(const FilmImage& im, WorkImage& wi) {
  const int nx = im.samples_x(), ny = im.samples_y();
  const float inv_white = 1.0f / im.white;
  Plane* mean[3] = {&wi.r, &wi.g, &wi.b};
  Plane* maxp[3] = {&wi.mr, &wi.mg, &wi.mb};
  #pragma omp parallel for schedule(static)
  for (int py = 0; py < wi.h; ++py) {
    const float lv = (py + 0.5f) * ny / wi.h - 0.5f;
    const int j0 = std::clamp((int)std::floor(lv), 0, ny - 1), j1 = std::min(j0 + 1, ny - 1);
    const float fv = std::clamp(lv - (float)j0, 0.0f, 1.0f);
    for (int px = 0; px < wi.w; ++px) {
      const float lu = (px + 0.5f) * nx / wi.w - 0.5f;
      const int i0 = std::clamp((int)std::floor(lu), 0, nx - 1), i1 = std::min(i0 + 1, nx - 1);
      const float fu = std::clamp(lu - (float)i0, 0.0f, 1.0f);
      for (int c = 0; c < 3; ++c) {
        const float a = sample_at(im, c, i0, j0), b = sample_at(im, c, i1, j0);
        const float d = sample_at(im, c, i0, j1), e = sample_at(im, c, i1, j1);
        mean[c]->at(px, py) = ((a * (1 - fu) + b * fu) * (1 - fv) + (d * (1 - fu) + e * fu) * fv) * inv_white;
        maxp[c]->at(px, py) = std::max(std::max(a, b), std::max(d, e)) * inv_white;
      }
    }
  }
}

// ---- 2. Rotation ----------------------------------------------------------------------------------------------

// Tilt of the film in degrees (FilmRect convention). Edge points of log luminance whose gradient is within 12
// degrees of an axis are projected across their edge direction at each candidate angle; straight strip and frame
// edges collapse into sharp peaks, so the angle maximising the sum of squared projection bins wins (0.2 degree
// steps, then 0.02, then a parabola). |confidence| is how much the best angle stands out from the median.
// Allocates: a work-size log luminance plane, the edge point lists and one projection histogram.
float estimate_rotation(const WorkImage& wi, float max_deg, float* confidence) {
  const int W = wi.w, H = wi.h;
  Plane L;
  L.init(W, H);
  for (size_t i = 0; i < L.d.size(); ++i)
    L.d[i] = std::log2(std::max((wi.r.d[i] + wi.g.d[i] + wi.b.d[i]) * (1.0f / 3.0f), 1e-5f));
  struct Pt { float x, y, w; };
  std::vector<Pt> ph, pv;  // edges running along x (horizontal) and along y
  std::vector<float> mags;
  const float tan_lim = std::tan(12.0f * kPi / 180.0f);
  const int rs = (W * H > 400000) ? 2 : 1;
  for (int y = 1; y < H - 1; y += rs) {
    for (int x = 1; x < W - 1; x += rs) {
      const float gx = (L.at(x + 1, y - 1) + 2 * L.at(x + 1, y) + L.at(x + 1, y + 1) - L.at(x - 1, y - 1) -
                        2 * L.at(x - 1, y) - L.at(x - 1, y + 1)) * 0.125f;
      const float gy = (L.at(x - 1, y + 1) + 2 * L.at(x, y + 1) + L.at(x + 1, y + 1) - L.at(x - 1, y - 1) -
                        2 * L.at(x, y - 1) - L.at(x + 1, y - 1)) * 0.125f;
      const float m = std::sqrt(gx * gx + gy * gy);
      if (m < 0.08f) continue;
      mags.push_back(m);
      if (std::fabs(gx) <= tan_lim * std::fabs(gy)) ph.push_back({(float)x, (float)y, m});
      else if (std::fabs(gy) <= tan_lim * std::fabs(gx)) pv.push_back({(float)x, (float)y, m});
    }
  }
  if (mags.size() < 50) {
    *confidence = 0.0f;
    return 0.0f;
  }
  const float thr = std::max(0.12f, percentile(mags, 0.6));
  auto keep_strong = [&](std::vector<Pt>& v) {
    std::vector<Pt> o;
    for (const Pt& p : v)
      if (p.w >= thr) o.push_back({p.x, p.y, std::min(p.w, 4 * thr)});
    v.swap(o);
  };
  keep_strong(ph);
  keep_strong(pv);
  const float cx = W * 0.5f, cy = H * 0.5f;
  const int nb = (int)std::ceil(std::sqrt((double)W * W + (double)H * H)) + 4;
  std::vector<float> hist(nb);
  auto score = [&](float deg) {
    const float th = deg * kPi / 180.0f, c = std::cos(th), s = std::sin(th);
    double tot = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
      std::fill(hist.begin(), hist.end(), 0.0f);
      for (const Pt& p : pass ? pv : ph) {
        const float X = p.x - cx, Y = p.y - cy;
        const float t = (pass ? (X * c + Y * s) : (-X * s + Y * c)) + nb * 0.5f;
        const int i = (int)t;
        const float f = t - (float)i;
        if (i < 0 || i + 1 >= nb) continue;
        hist[i] += p.w * (1 - f);
        hist[i + 1] += p.w * f;
      }
      for (float v : hist) tot += (double)v * v;
    }
    return tot;
  };
  float best = 0.0f;
  double bs = -1.0;
  std::vector<float> coarse;
  for (float a = -max_deg; a <= max_deg + 1e-4f; a += 0.2f) {
    const double s = score(a);
    coarse.push_back((float)s);
    if (s > bs) {
      bs = s;
      best = a;
    }
  }
  const float c0 = best;
  for (float a = c0 - 0.24f; a <= c0 + 0.24f + 1e-4f; a += 0.02f) {
    const double s = score(a);
    if (s > bs) {
      bs = s;
      best = a;
    }
  }
  const double s0 = score(best - 0.02f), s1 = score(best + 0.02f);
  const double den = s0 - 2 * bs + s1;
  if (den < 0) best += (float)(0.02 * 0.5 * (s0 - s1) / den);
  const double med = percentile(coarse, 0.5);
  *confidence = (float)(med > 0 ? std::min(1.0, (bs / med - 1.0) / 2.0) : 0.0);
  return best;
}

// ---- 3. Canvas: the work image turned upright -----------------------------------------------------------------

struct Canvas {
  int w = 0, h = 0;
  float theta = 0.0f;              // the tilt, degrees
  float ct = 1.0f, st = 0.0f;      // its cos and sin
  float cxw = 0.0f, cyw = 0.0f;    // work image centre
  float cxc = 0.0f, cyc = 0.0f;    // canvas centre
  Plane r, g, b;                   // normalised colour
  std::vector<uint8_t> valid;      // inside the image
  std::vector<uint8_t> clip;       // the work pixel's box held a clipped sample
  // canvas -> work coordinates
  void to_work(float X, float Y, float& x, float& y) const {
    const float dx = X - cxc, dy = Y - cyc;
    x = cxw + ct * dx - st * dy;
    y = cyw + st * dx + ct * dy;
  }
};

// Allocates: three canvas float planes and two canvas byte masks.
void build_canvas(const WorkImage& wi, float theta, const float clip_level[3], Canvas& cv) {
  const float th = theta * kPi / 180.0f, c = std::cos(th), s = std::sin(th);
  cv.theta = theta;
  cv.ct = c;
  cv.st = s;
  cv.w = (int)std::ceil(std::fabs(c) * wi.w + std::fabs(s) * wi.h);
  cv.h = (int)std::ceil(std::fabs(s) * wi.w + std::fabs(c) * wi.h);
  cv.cxw = wi.w * 0.5f;
  cv.cyw = wi.h * 0.5f;
  cv.cxc = cv.w * 0.5f;
  cv.cyc = cv.h * 0.5f;
  cv.r.init(cv.w, cv.h);
  cv.g.init(cv.w, cv.h);
  cv.b.init(cv.w, cv.h);
  cv.valid.assign((size_t)cv.w * cv.h, 0);
  cv.clip.assign((size_t)cv.w * cv.h, 0);
  const bool upright = std::fabs(theta) < 1e-6f;
  #pragma omp parallel for schedule(static)
  for (int Y = 0; Y < cv.h; ++Y) {
    for (int X = 0; X < cv.w; ++X) {
      float x, y;
      if (upright) {
        x = X + 0.5f;
        y = Y + 0.5f;
      } else {
        cv.to_work(X + 0.5f, Y + 0.5f, x, y);
      }
      if (x < 0.5f || y < 0.5f || x > wi.w - 0.5f || y > wi.h - 0.5f) continue;
      const float fx = x - 0.5f, fy = y - 0.5f;
      const int x0 = std::min((int)fx, wi.w - 2), y0 = std::min((int)fy, wi.h - 2);
      const float ax = fx - x0, ay = fy - y0;
      auto bil = [&](const Plane& p) {
        return (p.at(x0, y0) * (1 - ax) + p.at(x0 + 1, y0) * ax) * (1 - ay) +
               (p.at(x0, y0 + 1) * (1 - ax) + p.at(x0 + 1, y0 + 1) * ax) * ay;
      };
      const size_t i = (size_t)Y * cv.w + X;
      cv.r.d[i] = bil(wi.r);
      cv.g.d[i] = bil(wi.g);
      cv.b.d[i] = bil(wi.b);
      cv.valid[i] = 1;
      const int nx = std::clamp((int)std::lround(fx), 0, wi.w - 1), ny = std::clamp((int)std::lround(fy), 0, wi.h - 1);
      cv.clip[i] = (wi.mr.at(nx, ny) >= clip_level[0] || wi.mg.at(nx, ny) >= clip_level[1] ||
                    wi.mb.at(nx, ny) >= clip_level[2]) ? 1 : 0;
    }
  }
}

// ---- 4. Classification ----------------------------------------------------------------------------------------

// 8-connected components of a canvas mask, with per component area, sum of luminance and whether it touches the
// "outside" (the canvas border or a pixel where inside[] is 0).
struct Labels {
  std::vector<int> lab;          // component per pixel, -1 outside the mask
  std::vector<int> area;
  std::vector<double> sum;
  std::vector<uint8_t> touch;
};

// Allocates: a canvas int label plane and a flood-fill stack.
void components(const std::vector<uint8_t>& mask, const std::vector<uint8_t>& inside, const Plane& lum, int W, int H,
                Labels& L) {
  L.lab.assign((size_t)W * H, -1);
  L.area.clear();
  L.sum.clear();
  L.touch.clear();
  std::vector<int> stack;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t i = (size_t)y * W + x;
      if (!mask[i] || L.lab[i] >= 0) continue;
      const int id = (int)L.area.size();
      L.area.push_back(0);
      L.sum.push_back(0.0);
      L.touch.push_back(0);
      L.lab[i] = id;
      stack.push_back((int)i);
      while (!stack.empty()) {
        const int j = stack.back();
        stack.pop_back();
        const int jx = j % W, jy = j / W;
        L.area[id]++;
        L.sum[id] += lum.d[j];
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int nx = jx + dx, ny = jy + dy;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) {
              L.touch[id] = 1;
              continue;
            }
            const size_t k = (size_t)ny * W + nx;
            if (!inside[k]) {
              L.touch[id] = 1;
              continue;
            }
            if (mask[k] && L.lab[k] < 0) {
              L.lab[k] = id;
              stack.push_back((int)k);
            }
          }
        }
      }
    }
  }
}

// Square dilation by r pixels (separable distance scans). Allocates: two canvas byte masks.
void dilate(std::vector<uint8_t>& m, int W, int H, int r) {
  std::vector<uint8_t> t(m.size(), 0), o(m.size(), 0);
  for (int y = 0; y < H; ++y) {
    int last = -1000000;
    for (int x = 0; x < W; ++x) {
      if (m[(size_t)y * W + x]) last = x;
      if (x - last <= r) t[(size_t)y * W + x] = 1;
    }
    last = 1000000;
    for (int x = W - 1; x >= 0; --x) {
      if (m[(size_t)y * W + x]) last = x;
      if (last - x <= r) t[(size_t)y * W + x] = 1;
    }
  }
  for (int x = 0; x < W; ++x) {
    int last = -1000000;
    for (int y = 0; y < H; ++y) {
      if (t[(size_t)y * W + x]) last = y;
      if (y - last <= r) o[(size_t)y * W + x] = 1;
    }
    last = 1000000;
    for (int y = H - 1; y >= 0; --y) {
      if (t[(size_t)y * W + x]) last = y;
      if (last - y <= r) o[(size_t)y * W + x] = 1;
    }
  }
  m.swap(o);
}

// Histogram of log2 values in 0.05 stop bins over [-16, 0.6), lightly smoothed, with mode finding.
struct ModeHist {
  static constexpr float lo = -16.0f, bw = 0.05f;
  static constexpr int nb = 332;
  std::vector<double> h, hs;  // counts, smoothed counts
  double tot = 0.0;
  // Every step-th pixel i with use(i).
  template <class F>
  void build(const std::vector<float>& v, F use, size_t step) {
    h.assign(nb, 0.0);
    tot = 0.0;
    for (size_t i = 0; i < v.size(); i += step) {
      if (!use(i)) continue;
      const int b = (int)((v[i] - lo) / bw);
      if (b < 0 || b >= nb) continue;
      h[b] += 1.0;
      tot += 1.0;
    }
    hs.assign(nb, 0.0);
    for (int b = 0; b < nb; ++b) {
      double a = 0.0, ws = 0.0;
      for (int k = -4; k <= 4; ++k) {
        if (b + k < 0 || b + k >= nb) continue;
        const double w = std::exp(-0.5 * k * k / (1.5 * 1.5));
        a += w * h[b + k];
        ws += w;
      }
      hs[b] = a / ws;
    }
  }
  float center(int b) const { return lo + (b + 0.5f) * bw; }
  double mass(int c) const {
    double a = 0.0;
    for (int k = -4; k <= 4; ++k)
      if (c + k >= 0 && c + k < nb) a += h[c + k];
    return tot > 0 ? a / tot : 0.0;
  }
  std::vector<int> peaks(double min_mass) const {
    std::vector<int> p;
    for (int b = 1; b < nb - 1; ++b)
      if (hs[b] >= hs[b - 1] && hs[b] > hs[b + 1] && mass(b) >= min_mass) p.push_back(b);
    return p;
  }
  // The brightest mode (top), the next substantial mode at least 0.35 stop below it (nxt) and the valley between
  // them (val). True when they are separated: a clear valley, or at least 0.5 stop apart.
  bool find_light_split(int& top, int& nxt, int& val) const {
    if (tot < 100) return false;
    const std::vector<int> p = peaks(0.002);
    if (p.size() < 2) return false;
    top = p.back();
    nxt = -1;
    for (int k = (int)p.size() - 2; k >= 0; --k) {
      if ((top - p[k]) * bw >= 0.35f && mass(p[k]) >= 0.02) {
        nxt = p[k];
        break;
      }
    }
    if (nxt < 0) return false;
    val = nxt;
    for (int b = nxt; b <= top; ++b)
      if (hs[b] < hs[val]) val = b;
    return hs[val] <= 0.35 * std::min(hs[top], hs[nxt]) || (top - nxt) * bw >= 0.5f;
  }
  // The brightest mode with at least min_mass within range stops of the brightest populated bin, or -1.
  int top_mode(double min_mass, float range) const {
    int hi = -1;
    for (int b = nb - 1; b >= 0; --b) {
      if (h[b] > 0.0005 * tot) {
        hi = b;
        break;
      }
    }
    if (hi < 0) return -1;
    const std::vector<int> p = peaks(min_mass);
    for (int k = (int)p.size() - 1; k >= 0; --k)
      if ((hi - p[k]) * bw <= range) return p[k];
    return -1;
  }
};

struct ClassInfo {
  std::vector<uint8_t> cls;      // PixelClass per canvas pixel
  std::vector<float> logl;       // log2 luminance per canvas pixel
  float base[3] = {0.0f, 0.0f, 0.0f};  // film base colour (normalised)
  bool base_valid = false;
  bool light_clipped = false;
};

// Classifies every canvas pixel (see the file comment, step 4). Levels: ref is the 99.9th percentile luminance and
// refc the per channel 99.9th percentiles (bare light when some is visible, else the thinnest film); m(i), the
// minimum over channels of value / refc, is how close a pixel is to that reference in every channel (log2).
// Allocates (transiently, canvas sized): ~12 float planes and ~12 byte masks, and component label planes.
void classify(const Canvas& cv, FilmPolarity pol, ClassInfo& ci) {
  const int W = cv.w, H = cv.h;
  const size_t N = (size_t)W * H;
  Plane lum;
  lum.init(W, H);
  size_t nvalid = 0;
  for (size_t i = 0; i < N; ++i) {
    if (!cv.valid[i]) continue;
    nvalid++;
    lum.d[i] = (cv.r.d[i] + cv.g.d[i] + cv.b.d[i]) * (1.0f / 3.0f);
  }
  // Percentiles from every step-th valid pixel (at most 200k).
  const size_t step = std::max<size_t>(1, nvalid / 200000);
  std::vector<float> lv, rv[3];
  for (size_t i = 0, k = 0; i < N; ++i) {
    if (!cv.valid[i]) continue;
    if ((k++ % step) == 0) {
      lv.push_back(lum.d[i]);
      rv[0].push_back(cv.r.d[i]);
      rv[1].push_back(cv.g.d[i]);
      rv[2].push_back(cv.b.d[i]);
    }
  }
  const float ref = std::max(1e-6f, percentile(lv, 0.999));
  float refc[3];
  for (int c = 0; c < 3; ++c) refc[c] = std::max(1e-6f, percentile(rv[c], 0.999));
  const float black_frac = (pol == FILM_POSITIVE) ? 0.0002f : 0.002f;

  // Black (no light at all) and clipped light.
  std::vector<uint8_t> black(N, 0), light(N, 0);
  size_t nclip = 0;
  for (size_t i = 0; i < N; ++i) {
    if (!cv.valid[i]) continue;
    if (lum.d[i] < black_frac * ref) black[i] = 1;
    else if (cv.clip[i]) {
      light[i] = 1;
      nclip++;
    }
  }
  // What counts as outside for light: the image border, or the dark carrier (light seen through a holder's aperture
  // borders the holder), widened by 3 px across the blended edge between light and carrier.
  std::vector<uint8_t> inside(N, 0);
  {
    std::vector<uint8_t> outside(N, 0);
    for (size_t i = 0; i < N; ++i) outside[i] = (!cv.valid[i] || lum.d[i] < 0.01f * ref) ? 1 : 0;
    dilate(outside, W, H, 3);
    for (size_t i = 0; i < N; ++i) inside[i] = outside[i] ? 0 : 1;
  }
  // Keeps the parts of mask that can be bare light: reaching the outside or hole-sized (<= 0.5% of the image), and
  // on average at least lum_min bright.
  auto keep_light_parts = [&](std::vector<uint8_t>& m, float lum_min) {
    Labels L;
    components(m, inside, lum, W, H, L);
    std::vector<uint8_t> keep(L.area.size(), 0);
    for (size_t k = 0; k < L.area.size(); ++k) {
      const double mean = L.sum[k] / std::max(1, L.area[k]);
      keep[k] = (L.touch[k] || L.area[k] <= 0.005 * (double)nvalid) && mean >= lum_min;
    }
    for (size_t i = 0; i < N; ++i) m[i] = (m[i] && keep[L.lab[i]]) ? 1 : 0;
  };
  if (nclip) {
    keep_light_parts(light, 0.65f * ref);
    ci.light_clipped = true;
  }
  size_t nlight = 0;
  for (size_t i = 0; i < N; ++i) nlight += light[i];
  // m(i): log2 of the minimum over channels of value / refc.
  std::vector<float> mall(N, -100.0f);
  for (size_t i = 0; i < N; ++i) {
    if (!cv.valid[i] || black[i]) continue;
    const float m = std::min({cv.r.d[i] / refc[0], cv.g.d[i] / refc[1], cv.b.d[i] / refc[2]});
    mall[i] = std::log2(std::max(m, 1e-6f));
  }
  // Local flatness: Sobel magnitude of log2 luminance, stops per canvas pixel.
  std::vector<float> grad(N, 0.0f);
  ci.logl.assign(N, -20.0f);
  {
    std::vector<float>& ll = ci.logl;
    for (size_t i = 0; i < N; ++i) ll[i] = std::log2(std::max(lum.d[i], 1e-6f));
    #pragma omp parallel for schedule(static)
    for (int y = 1; y < H - 1; ++y) {
      for (int x = 1; x < W - 1; ++x) {
        const size_t i = (size_t)y * W + x;
        const float gx = (ll[i - W + 1] + 2 * ll[i + 1] + ll[i + W + 1] - ll[i - W - 1] - 2 * ll[i - 1] -
                          ll[i + W - 1]) * 0.125f;
        const float gy = (ll[i + W - 1] + 2 * ll[i + W] + ll[i + W + 1] - ll[i - W - 1] - 2 * ll[i - W] -
                          ll[i - W + 1]) * 0.125f;
        grad[i] = std::sqrt(gx * gx + gy * gy);
      }
    }
  }
  if (pol == FILM_POSITIVE) {
    // Slide film: highlights are nearly as bright as bare light, so space decides: bright parts that reach the
    // image border (only the border: a slide's own rebate is dark and borders every frame) and are flat, or
    // hole-sized and flat inside.
    std::vector<uint8_t> cand(N, 0);
    for (size_t i = 0; i < N; ++i) cand[i] = (cv.valid[i] && !black[i] && !light[i] && mall[i] >= -0.6f) ? 1 : 0;
    Labels L;
    components(cand, cv.valid, lum, W, H, L);
    std::vector<double> gsum(L.area.size(), 0.0), gflat(L.area.size(), 0.0);
    for (size_t i = 0; i < N; ++i) {
      if (L.lab[i] < 0) continue;
      gsum[L.lab[i]] += grad[i];
      if (grad[i] < 0.1f) gflat[L.lab[i]] += 1.0;
    }
    std::vector<uint8_t> keep(L.area.size(), 0);
    for (size_t k = 0; k < L.area.size(); ++k) {
      const double mean = L.sum[k] / std::max(1, L.area[k]);
      const bool small = L.area[k] <= 0.005 * (double)nvalid;
      const bool flat = small ? gflat[k] >= 0.5 * L.area[k] : gsum[k] / L.area[k] < 0.06;
      keep[k] = (L.touch[k] || small) && L.area[k] >= 4 && flat && mean >= 0.5 * ref;
    }
    for (size_t i = 0; i < N; ++i)
      if (L.lab[i] >= 0 && keep[L.lab[i]]) light[i] = 1;
  } else if (nlight < 0.005 * nvalid) {
    // Unclipped bare light: the brightest mode of m, separated from the next mode (the film base). It must border
    // flat film base rather than picture, and not lie between two film areas (bare light lies outside the film;
    // a gap between two frames lies between them).
    ModeHist mh;
    mh.build(mall, [&](size_t i) { return cv.valid[i] && !black[i] && !light[i]; }, step);
    int top = -1, nxt = -1, val = -1;
    if (mh.find_light_split(top, nxt, val)) {
      const bool valley = mh.hs[val] <= 0.35 * std::min(mh.hs[top], mh.hs[nxt]);
      const float mnext = mh.center(nxt);
      const float t = valley ? mh.center(val) : 0.5f * (mh.center(top) + mnext);
      std::vector<uint8_t> cand(N, 0);
      for (size_t i = 0; i < N; ++i) cand[i] = (cv.valid[i] && !black[i] && !light[i] && mall[i] >= t) ? 1 : 0;
      // The ring 3..5 px around the candidate: flat at the next mode's level (film base) or not (picture).
      std::vector<uint8_t> d1 = cand, d2 = cand;
      dilate(d1, W, H, 2);
      dilate(d2, W, H, 5);
      size_t flat = 0, other = 0;
      for (size_t i = 0; i < N; ++i) {
        if (!d2[i] || d1[i] || !cv.valid[i] || black[i]) continue;
        if (std::fabs(mall[i] - mnext) <= 0.2f && grad[i] < 0.06f) flat++;
        else if (mall[i] >= mnext - 3.0f) other++;  // picture; the holder (far darker) does not count
      }
      // Between film: rays along x and y from every 3rd pixel of the large candidate parts; film found for 6 px in a
      // row on both sides (within a fifth of the canvas) means the part lies between film.
      bool outside_film = true;
      {
        Labels Lc;
        components(cand, inside, lum, W, H, Lc);
        const int R = std::max(4, (int)(0.2f * std::min(W, H)));
        auto ray = [&](int x, int y, int dx, int dy) {
          int run = 0;
          for (int k = 1; k <= R; ++k) {
            const int xx = x + dx * k, yy = y + dy * k;
            if (xx < 0 || yy < 0 || xx >= W || yy >= H) return false;
            const size_t j = (size_t)yy * W + xx;
            if (cand[j]) {
              run = 0;
              continue;
            }
            if (!cv.valid[j] || lum.d[j] < 0.01f * ref) return false;
            const bool film = !black[j] && mall[j] <= mnext + 0.3f;
            if (film) {
              if (++run >= 6) return true;
            } else {
              run = 0;
            }
          }
          return false;
        };
        size_t tot = 0, between = 0;
        for (int y = 0; y < H; y += 3) {
          for (int x = 0; x < W; x += 3) {
            const size_t i = (size_t)y * W + x;
            if (!cand[i] || Lc.area[Lc.lab[i]] <= 0.005 * (double)nvalid) continue;
            tot++;
            if ((ray(x, y, -1, 0) && ray(x, y, 1, 0)) || (ray(x, y, 0, -1) && ray(x, y, 0, 1))) between++;
          }
        }
        outside_film = tot == 0 || between < 0.25 * tot;
      }
      if (outside_film && flat + other > 0 && flat >= (valley ? 0.45 : 0.6) * (double)(flat + other)) {
        keep_light_parts(cand, 0.5f * ref);
        for (size_t i = 0; i < N; ++i) light[i] |= cand[i];
      }
    }
  } else if (ci.light_clipped) {
    // Partly clipped light (a vignetted light table): grow into the unclipped bright light connected to it that
    // reaches the border.
    std::vector<uint8_t> bright(N, 0);
    for (size_t i = 0; i < N; ++i) bright[i] = (cv.valid[i] && !black[i] && mall[i] >= -0.4f) ? 1 : 0;
    Labels L;
    components(bright, cv.valid, lum, W, H, L);
    std::vector<uint8_t> hit(L.area.size(), 0);
    for (size_t i = 0; i < N; ++i)
      if (light[i] && L.lab[i] >= 0) hit[L.lab[i]] = 1;
    for (size_t i = 0; i < N; ++i)
      if (L.lab[i] >= 0 && hit[L.lab[i]] && L.touch[L.lab[i]]) light[i] = 1;
  }
  // The blended edge of light counts as light.
  dilate(light, W, H, 2);
  for (size_t i = 0; i < N; ++i)
    if (black[i] || !cv.valid[i]) light[i] = 0;

  // Film base. A first estimate from the film's percentiles (the thinnest film for a negative, the densest for a
  // slide); for a negative then the brightest mode of m that forms large connected regions (rebate bands, gaps),
  // not only small blobs (unclipped holes, blended edges), walking down from the brightest mode.
  {
    std::vector<float> fr[3];
    for (size_t i = 0, k = 0; i < N; ++i) {
      if (!cv.valid[i] || black[i] || light[i]) continue;
      if ((k++ % step) == 0) {
        fr[0].push_back(cv.r.d[i]);
        fr[1].push_back(cv.g.d[i]);
        fr[2].push_back(cv.b.d[i]);
      }
    }
    if (fr[0].size() < 16) {
      for (int c = 0; c < 3; ++c) fr[c].clear();
      for (size_t i = 0; i < N; ++i) {
        if (!cv.valid[i] || black[i]) continue;
        fr[0].push_back(cv.r.d[i]);
        fr[1].push_back(cv.g.d[i]);
        fr[2].push_back(cv.b.d[i]);
      }
    }
    for (int c = 0; c < 3; ++c)
      ci.base[c] = std::max(1e-6f, percentile(fr[c], pol == FILM_POSITIVE ? 0.005 : 0.995));
    ci.base_valid = !fr[0].empty();
  }
  if (pol == FILM_NEGATIVE) {
    ModeHist mh;
    mh.build(mall, [&](size_t i) { return cv.valid[i] && !black[i] && !light[i]; }, step);
    int pk = -1;
    const std::vector<int> pks = mh.peaks(0.004);
    int hi = -1;
    for (int b = ModeHist::nb - 1; b >= 0; --b) {
      if (mh.h[b] > 0.0005 * mh.tot) {
        hi = b;
        break;
      }
    }
    std::vector<uint8_t> mm(N, 0);
    for (int k = (int)pks.size() - 1; k >= 0 && hi >= 0; --k) {
      if ((hi - pks[k]) * ModeHist::bw > 2.5f) break;
      const float mc = mh.center(pks[k]);
      for (size_t i = 0; i < N; ++i)
        mm[i] = (cv.valid[i] && !black[i] && !light[i] && std::fabs(mall[i] - mc) <= 0.15f) ? 1 : 0;
      Labels L;
      components(mm, cv.valid, lum, W, H, L);
      double big = 0.0;
      for (size_t c = 0; c < L.area.size(); ++c)
        if (L.area[c] >= 0.0015 * (double)nvalid) big += L.area[c];
      if (big >= 0.003 * (double)nvalid) {
        pk = pks[k];
        break;
      }
    }
    if (pk < 0) pk = mh.top_mode(0.004, 2.5f);
    if (pk >= 0) {
      const float mc = mh.center(pk);
      std::vector<float> cr[3];
      for (size_t i = 0, k = 0; i < N; ++i) {
        if (!cv.valid[i] || black[i] || light[i] || std::fabs(mall[i] - mc) > 0.12f) continue;
        if ((k++ % step) == 0) {
          cr[0].push_back(cv.r.d[i]);
          cr[1].push_back(cv.g.d[i]);
          cr[2].push_back(cv.b.d[i]);
        }
      }
      if (cr[0].size() >= 16)
        for (int c = 0; c < 3; ++c) ci.base[c] = std::max(1e-6f, percentile(cr[c], 0.5));
    }
  }
  const float base_lum = (ci.base[0] + ci.base[1] + ci.base[2]) / 3.0f;
  const float base_q = std::log2(ci.base[0] / ci.base[2]);  // how orange the base is (log2 R / B)
  const float base_margin = 0.30f;                          // stops below the base still counted as base

  // Unclipped sprocket holes and light patches: small parts brighter than the base by 0.35 stop in every channel
  // (no part of a negative is thinner than its own base).
  if (pol == FILM_NEGATIVE && ci.base_valid) {
    std::vector<uint8_t> br(N, 0);
    const float k = std::exp2(0.35f);
    for (size_t i = 0; i < N; ++i)
      br[i] = (cv.valid[i] && !black[i] && !light[i] && cv.r.d[i] > k * ci.base[0] && cv.g.d[i] > k * ci.base[1] &&
               cv.b.d[i] > k * ci.base[2]) ? 1 : 0;
    Labels L;
    components(br, cv.valid, lum, W, H, L);
    std::vector<uint8_t> add(N, 0);
    bool any = false;
    for (size_t i = 0; i < N; ++i) {
      if (L.lab[i] < 0) continue;
      const int a = L.area[L.lab[i]];
      if (a >= 4 && (a <= 0.005 * (double)nvalid || L.touch[L.lab[i]])) {
        add[i] = 1;
        any = true;
      }
    }
    if (any) {
      dilate(add, W, H, 1);
      for (size_t i = 0; i < N; ++i)
        if (add[i] && !black[i] && cv.valid[i]) light[i] = 1;
    }
  }
  // Holder / carrier: large (>= 1% of the image), flat parts that reach the border, darker than 2% of the base
  // (negative) or 1% of the light (slide); flare can keep them above black.
  std::vector<uint8_t> holder(N, 0);
  {
    const float lim = pol == FILM_NEGATIVE ? 0.02f * base_lum : 0.01f * ref;
    std::vector<uint8_t> dc(N, 0);
    for (size_t i = 0; i < N; ++i) dc[i] = (cv.valid[i] && !light[i] && lum.d[i] < lim) ? 1 : 0;
    Labels L;
    components(dc, cv.valid, lum, W, H, L);
    std::vector<double> gsum(L.area.size(), 0.0);
    for (size_t i = 0; i < N; ++i)
      if (L.lab[i] >= 0) gsum[L.lab[i]] += grad[i];
    std::vector<uint8_t> keep(L.area.size(), 0);
    for (size_t k = 0; k < L.area.size(); ++k)
      keep[k] = L.touch[k] && L.area[k] >= 0.01 * (double)nvalid && gsum[k] / L.area[k] < 0.08;
    for (size_t i = 0; i < N; ++i) holder[i] = (L.lab[i] >= 0 && keep[L.lab[i]]) ? 1 : 0;
  }
  // Slide: its unexposed film is the densest mode; within 0.7 stop of it is DARK.
  float slide_dark = -100.0f;
  if (pol == FILM_POSITIVE) {
    std::vector<float> rel(N, -100.0f);
    for (size_t i = 0; i < N; ++i)
      if (cv.valid[i]) rel[i] = std::log2(std::max(lum.d[i], 1e-7f) / ref);
    ModeHist mh;
    mh.build(rel, [&](size_t i) { return cv.valid[i] && !light[i]; }, step);
    const std::vector<int> pk = mh.peaks(0.02);
    slide_dark = pk.empty() ? std::log2(0.006f) : mh.center(pk.front()) + 0.7f;
  }
  ci.cls.assign(N, CL_INVALID);
  for (size_t i = 0; i < N; ++i) {
    if (!cv.valid[i]) continue;
    if (light[i]) {
      ci.cls[i] = CL_LIGHT;
      continue;
    }
    if (black[i]) {
      ci.cls[i] = CL_DARK;
      continue;
    }
    const float l = lum.d[i];
    if (pol == FILM_NEGATIVE) {
      // Dark: the holder, very dense, or dense and less orange than the base (flare on a carrier is less orange
      // than film, which keeps its mask; relative to the measured base, so the lamp's colour cancels).
      const float q = std::log2(std::max(cv.r.d[i], 1e-7f) / std::max(cv.b.d[i], 1e-7f));
      const bool dark = holder[i] || l < 0.006f * base_lum || (l < 0.02f * base_lum && q < base_q - 0.10f) ||
                        (l < 0.06f * base_lum && q < base_q - 0.6f);
      if (dark) {
        ci.cls[i] = CL_DARK;
        continue;
      }
      const float k = std::exp2(-base_margin);
      const bool thin = cv.r.d[i] >= ci.base[0] * k && cv.g.d[i] >= ci.base[1] * k && cv.b.d[i] >= ci.base[2] * k;
      ci.cls[i] = thin ? CL_BASE : CL_IMAGE;
    } else {
      const bool dense = std::log2(std::max(l, 1e-7f) / ref) <= slide_dark;
      ci.cls[i] = (dense || holder[i]) ? CL_DARK : CL_IMAGE;
    }
  }
}

// ---- 5a. Rectangle search -------------------------------------------------------------------------------------

// Keeps m[i] where at least frac of the horizontal or the vertical window (window_frac of the short side, edges
// replicated) around i is set: rebate bands, gaps and carrier edges are lines, thin blobs in a picture are not.
// Allocates: two search-size float planes and a prefix sum line.
void line_gate(const std::vector<float>& m, int W, int H, float window_frac, float frac, std::vector<float>& out) {
  const int n = std::max(3, (int)std::lround(window_frac * std::min(W, H)) | 1);
  const int r = n / 2;
  std::vector<float> ax((size_t)W * H), ay((size_t)W * H);
  std::vector<double> cs(std::max(W, H) + 1);
  for (int y = 0; y < H; ++y) {
    cs[0] = 0.0;
    for (int x = 0; x < W; ++x) cs[x + 1] = cs[x] + m[(size_t)y * W + x];
    for (int x = 0; x < W; ++x) {
      const int a = x - r, b = x + r + 1;
      double s = cs[std::clamp(b, 0, W)] - cs[std::clamp(a, 0, W)];
      if (a < 0) s += (double)(-a) * m[(size_t)y * W];
      if (b > W) s += (double)(b - W) * m[(size_t)y * W + W - 1];
      ax[(size_t)y * W + x] = (float)(s / n);
    }
  }
  for (int x = 0; x < W; ++x) {
    cs[0] = 0.0;
    for (int y = 0; y < H; ++y) cs[y + 1] = cs[y] + m[(size_t)y * W + x];
    for (int y = 0; y < H; ++y) {
      const int a = y - r, b = y + r + 1;
      double s = cs[std::clamp(b, 0, H)] - cs[std::clamp(a, 0, H)];
      if (a < 0) s += (double)(-a) * m[x];
      if (b > H) s += (double)(b - H) * m[(size_t)(H - 1) * W + x];
      ay[(size_t)y * W + x] = (float)(s / n);
    }
  }
  out.resize((size_t)W * H);
  for (size_t i = 0; i < out.size(); ++i) out[i] = (ax[i] >= frac || ay[i] >= frac) ? m[i] : 0.0f;
}

// Prefix sums of the gated non-image map: I the integral image ((sw + 1) x (sh + 1)), down the running sum down
// each column ((sh + 1) x sw), across along each row (sh x (sw + 1)). Allocates those three double planes.
struct SearchMap {
  int w = 0, h = 0;
  std::vector<double> I, down, across;
};

void build_search(const std::vector<float>& m, int W, int H, SearchMap& S) {
  S.w = W;
  S.h = H;
  S.I.assign((size_t)(W + 1) * (H + 1), 0.0);
  S.down.assign((size_t)(H + 1) * W, 0.0);
  S.across.assign((size_t)H * (W + 1), 0.0);
  for (int y = 0; y < H; ++y) {
    double row = 0.0;
    for (int x = 0; x < W; ++x) {
      const float v = m[(size_t)y * W + x];
      row += v;
      S.I[(size_t)(y + 1) * (W + 1) + x + 1] = S.I[(size_t)y * (W + 1) + x + 1] + row;
      S.down[(size_t)(y + 1) * W + x] = S.down[(size_t)y * W + x] + v;
      S.across[(size_t)y * (W + 1) + x + 1] = S.across[(size_t)y * (W + 1) + x] + v;
    }
  }
}

struct SearchRes {
  double score = -1e300;
  int x = 0, y = 0, w = 0, h = 0;
  // Ties keep the smallest (h, y, x), the order a single thread visits them in.
  bool better(const SearchRes& o) const {
    if (score != o.score) return score > o.score;
    if (h != o.h) return h < o.h;
    if (y != o.y) return y < o.y;
    return x < o.x;
  }
};

// Best rectangle of the given aspect (w / h) over heights h_lo..h_hi in steps of h_step: maximises
//   area - 2 * non_image - 0.002 * area * (distance of its centre from the map centre, normalised)
// so an edge row or column is taken while less than half of it is non-image, and no rectangle may hold a full
// non-image line (>= 90% of its side) more than 2 px inside its edges (the gap between two frames, a rebate band).
// Heights run in parallel. Allocates per thread: two int prefix planes of the search size.
SearchRes search_heights(const SearchMap& S, float aspect, int h_lo, int h_hi, int h_step) {
  const int W = S.w, H = S.h;
  const double penalty = 1.0, centre_bias = 0.002, cross_frac = 0.9;
  const int cm = 2;
  SearchRes best;
  const int n_heights = h_hi >= h_lo ? (h_hi - h_lo) / h_step + 1 : 0;
  #pragma omp parallel
  {
    SearchRes local;
    std::vector<int> colc, rowc;
    #pragma omp for schedule(dynamic)
    for (int hi = 0; hi < n_heights; ++hi) {
      const int h = h_lo + hi * h_step;
      const int w = (int)std::lround(h * aspect);
      if (h < 3 || w < 3 || w > W || h > H) continue;
      const int ny = H + 1 - h, nx = W + 1 - w;
      // colc[y][x + 1]: full non-image columns among columns < x + 1, over rows [y, y + h)
      colc.assign((size_t)ny * (W + 1), 0);
      for (int y = 0; y < ny; ++y) {
        int* c = &colc[(size_t)y * (W + 1)];
        for (int x = 0; x < W; ++x) {
          const double v = S.down[(size_t)(y + h) * W + x] - S.down[(size_t)y * W + x];
          c[x + 1] = c[x] + (v >= cross_frac * h ? 1 : 0);
        }
      }
      // rowc[y + 1][x]: full non-image rows among rows < y + 1, over columns [x, x + w)
      rowc.assign((size_t)(H + 1) * nx, 0);
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < nx; ++x) {
          const double v = S.across[(size_t)y * (W + 1) + x + w] - S.across[(size_t)y * (W + 1) + x];
          rowc[(size_t)(y + 1) * nx + x] = rowc[(size_t)y * nx + x] + (v >= cross_frac * w ? 1 : 0);
        }
      }
      const double area = (double)w * h;
      for (int y = 0; y < ny; ++y) {
        const double dy = (y + h * 0.5 - H * 0.5) / H;
        const int* c = &colc[(size_t)y * (W + 1)];
        for (int x = 0; x < nx; ++x) {
          const double cnt = S.I[(size_t)(y + h) * (W + 1) + x + w] - S.I[(size_t)y * (W + 1) + x + w] -
                             S.I[(size_t)(y + h) * (W + 1) + x] + S.I[(size_t)y * (W + 1) + x];
          const double dx = (x + w * 0.5 - W * 0.5) / W;
          const double sc = area - (1.0 + penalty) * cnt - centre_bias * area * std::sqrt(dx * dx + dy * dy);
          if (sc <= local.score) continue;
          if (w - 2 * cm > 0 && c[x + w - cm] - c[x + cm] > 0) continue;
          if (h - 2 * cm > 0 && rowc[(size_t)(y + h - cm) * nx + x] - rowc[(size_t)(y + cm) * nx + x] > 0) continue;
          local.score = sc;
          local.x = x;
          local.y = y;
          local.w = w;
          local.h = h;
        }
      }
    }
    #pragma omp critical
    {
      if (local.better(best)) best = local;
    }
  }
  return best;
}

// Coarse heights (sh / 96 steps, from 12% of the short side) then +-1 step at single pixels.
SearchRes best_rect(const SearchMap& S, float aspect) {
  const int hmin = std::max(4, (int)(0.12f * std::min(S.w, S.h)));
  const int step = std::max(1, S.h / 96);
  SearchRes r = search_heights(S, aspect, hmin, S.h, step);
  if (r.h > 0) {
    const SearchRes f = search_heights(S, aspect, std::max(hmin, r.h - step), std::min(S.h, r.h + step), 1);
    if (f.score >= r.score) r = f;
  }
  return r;
}

// ---- 5b. Strip model ------------------------------------------------------------------------------------------

struct Rect {
  float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
  float w() const { return x1 - x0; }
  float h() const { return y1 - y0; }
};

// The canvas classes seen with the strip along x: transposed when the strip runs along canvas y.
struct View {
  const std::vector<uint8_t>* cls;
  int W, H;  // canvas size
  bool tr;
  int w() const { return tr ? H : W; }
  int h() const { return tr ? W : H; }
  uint8_t at(int x, int y) const { return tr ? (*cls)[(size_t)x * W + y] : (*cls)[(size_t)y * W + x]; }
};

// Fraction of pixels with pred(class) in column x over rows [y0, y1), or in row y over columns [x0, x1).
template <class F>
float col_frac(const View& v, int x, int y0, int y1, F pred) {
  if (x < 0 || x >= v.w()) return 0.0f;
  y0 = std::clamp(y0, 0, v.h());
  y1 = std::clamp(y1, 0, v.h());
  if (y1 <= y0) return 0.0f;
  int n = 0;
  for (int y = y0; y < y1; ++y) n += pred(v.at(x, y));
  return (float)n / (y1 - y0);
}

template <class F>
float row_frac(const View& v, int y, int x0, int x1, F pred) {
  if (y < 0 || y >= v.h()) return 0.0f;
  x0 = std::clamp(x0, 0, v.w());
  x1 = std::clamp(x1, 0, v.w());
  if (x1 <= x0) return 0.0f;
  int n = 0;
  for (int x = x0; x < x1; ++x) n += pred(v.at(x, y));
  return (float)n / (x1 - x0);
}

inline bool is_image(uint8_t c) { return c == CL_IMAGE; }
inline bool is_valid(uint8_t c) { return c != CL_INVALID; }
inline bool is_hard(uint8_t c) { return c == CL_LIGHT || c == CL_DARK; }
inline bool is_gap(uint8_t c) { return c == CL_BASE || c == CL_LIGHT || c == CL_DARK; }

// Edge position near t0 in a profile p of picture fractions: the position within +-delta where the mean over k
// samples inside (above t when rising) minus the mean outside is largest; beyond the profile counts as non-image.
// strength: that difference (1: a clean edge).
float refine_edge_profile(const std::vector<float>& p, float t0, int delta, int k, bool rising, float* strength) {
  const int n = (int)p.size();
  std::vector<double> cs(n + 1, 0.0);
  for (int i = 0; i < n; ++i) cs[i + 1] = cs[i] + p[i];
  auto mean = [&](int a, int b) {
    a = std::clamp(a, 0, n);
    b = std::clamp(b, 0, n);
    return b <= a ? -1.0 : (cs[b] - cs[a]) / (b - a);
  };
  const int c = (int)std::lround(t0);
  double best = -1e9;
  int bi = c;
  for (int t = c - delta; t <= c + delta; ++t) {
    if (t < 0 || t > n) continue;
    const double in = rising ? mean(t, t + k) : mean(t - k, t);
    double outside = rising ? mean(t - k, t) : mean(t, t + k);
    if (in < 0) continue;
    if (outside < 0) outside = 0.0;
    const double J = in - outside - 0.002 * std::abs(t - t0) / std::max(1, delta);
    if (J > best) {
      best = J;
      bi = t;
    }
  }
  if (strength) *strength = (float)best;
  return (float)bi;
}

// Maps view rectangles to image FilmRects: view -> canvas (transpose) -> work (turn by the tilt) -> image.
struct Geometry {
  const Canvas* cv;
  const WorkImage* wi;
  bool tr;
  FilmRect to_image(const Rect& vr) const {
    Rect r = vr;
    if (tr) {
      r.x0 = vr.y0;
      r.x1 = vr.y1;
      r.y0 = vr.x0;
      r.y1 = vr.x1;
    }
    float wx, wy;
    cv->to_work(0.5f * (r.x0 + r.x1), 0.5f * (r.y0 + r.y1), wx, wy);
    FilmRect o;
    o.cx = wi->off_x + wx * wi->sx;
    o.cy = wi->off_y + wy * wi->sy;
    o.w = r.w() * wi->sx;  // sx and sy agree to a fraction of a pixel (the downsample keeps the aspect)
    o.h = r.h() * wi->sy;
    o.angle_deg = cv->theta;
    return o;
  }
};

// Clips a quad to the image: the bounding box (x, y, w, h) of the part inside and its area.
void clip_quad(const float xy[8], int W, int H, int box[4], double& area) {
  std::vector<std::pair<double, double>> poly;
  for (int i = 0; i < 4; ++i) poly.push_back({xy[2 * i], xy[2 * i + 1]});
  auto clip = [&](int axis, double v, bool keep_less) {
    std::vector<std::pair<double, double>> o;
    const size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
      const auto a = poly[i], b = poly[(i + 1) % n];
      const double av = axis ? a.second : a.first, bv = axis ? b.second : b.first;
      const bool ain = keep_less ? av <= v : av >= v, bin = keep_less ? bv <= v : bv >= v;
      if (ain) o.push_back(a);
      if (ain != bin) {
        const double t = (v - av) / (bv - av);
        o.push_back({a.first + t * (b.first - a.first), a.second + t * (b.second - a.second)});
      }
    }
    poly.swap(o);
  };
  clip(0, 0.0, false);
  if (!poly.empty()) clip(0, (double)W, true);
  if (!poly.empty()) clip(1, 0.0, false);
  if (!poly.empty()) clip(1, (double)H, true);
  area = 0.0;
  box[0] = box[1] = box[2] = box[3] = 0;
  if (poly.size() < 3) return;
  double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
  for (size_t i = 0; i < poly.size(); ++i) {
    const auto a = poly[i], b = poly[(i + 1) % poly.size()];
    area += a.first * b.second - b.first * a.second;
    x0 = std::min(x0, a.first);
    y0 = std::min(y0, a.second);
    x1 = std::max(x1, a.first);
    y1 = std::max(y1, a.second);
  }
  area = std::fabs(area) * 0.5;
  box[0] = (int)std::floor(x0);
  box[1] = (int)std::floor(y0);
  box[2] = (int)std::ceil(x1) - box[0];
  box[3] = (int)std::ceil(y1) - box[1];
}

FilmFrame make_frame(const FilmRect& r, float nominal_area, int index, int W, int H) {
  FilmFrame f;
  f.rect = r;
  float xy[8];
  r.corners(xy);
  double area;
  clip_quad(xy, W, H, f.visible, area);
  f.visible_fraction = (float)(area / std::max(1.0f, nominal_area));
  f.complete = area >= 0.985 * (double)r.w * r.h;
  f.index = index;
  return f;
}

// Half sizes (p, q) of the largest axis-aligned rectangle with r's aspect centred inside r.
void inscribed_half(const FilmRect& r, float& p, float& q) {
  const float th = r.angle_deg * kPi / 180.0f, c = std::fabs(std::cos(th)), s = std::fabs(std::sin(th));
  const float a = 0.5f * r.w, b = 0.5f * r.h;
  const float k = std::min(a / std::max(1e-6f, a * c + b * s), b / std::max(1e-6f, a * s + b * c));
  p = a * k;
  q = b * k;
}

// 5c. Sub-pixel refinement of one edge of r on the full-resolution image: 0 left, 1 right, 2 top, 3 bottom (in r's
// own frame). 48 profiles across the edge, spread over its middle 70%, of log2 luminance sampled every image pixel
// within +-range, are averaged; the edge moves to the steepest step (parabolic sub-sample) when that step stands
// out (>= 0.04 stop per pixel and 3x the median slope). Returns the shift along the edge's normal, 0 for none.
// Allocates: profiles of 2 * range + 1 floats.
float refine_edge_full(const FilmImage& im, const FilmRect& r, int edge, float range) {
  const float th = r.angle_deg * kPi / 180.0f, c = std::cos(th), s = std::sin(th);
  float nx, ny, ex, ey, len, off;
  if (edge < 2) {
    nx = c;
    ny = s;
    ex = -s;
    ey = c;
    len = r.h;
    off = (edge == 0 ? -0.5f : 0.5f) * r.w;
  } else {
    nx = -s;
    ny = c;
    ex = c;
    ey = s;
    len = r.w;
    off = (edge == 2 ? -0.5f : 0.5f) * r.h;
  }
  const int nsamp = 48;
  const int nprof = (int)std::ceil(2 * range) + 1;
  if (nprof < 5) return 0.0f;
  std::vector<double> prof(nprof, 0.0);
  std::vector<float> p(nprof);
  int used = 0;
  for (int k = 0; k < nsamp; ++k) {
    const float t = (-0.35f + 0.7f * (k + 0.5f) / nsamp) * len;
    const float bx = r.cx + nx * off + ex * t, by = r.cy + ny * off + ey * t;
    bool ok = true;
    for (int j = 0; j < nprof; ++j) {
      const float d = -range + j * (2 * range) / (nprof - 1);
      const float px = bx + nx * d, py = by + ny * d;
      if (px < 0 || py < 0 || px >= im.width || py >= im.height) {
        ok = false;
        break;
      }
      p[j] = std::log2(std::max(luminance_bilinear(im, px, py), 1e-6f));
    }
    if (!ok) continue;
    for (int j = 0; j < nprof; ++j) prof[j] += p[j];
    used++;
  }
  if (used < nsamp / 3) return 0.0f;
  std::vector<float> pf(nprof);
  for (int j = 0; j < nprof; ++j) pf[j] = (float)(prof[j] / used);
  smooth1d(pf, std::max(0.7f, range / 12.0f));
  std::vector<float> d(nprof, 0.0f), ad;
  for (int j = 1; j < nprof - 1; ++j) {
    d[j] = 0.5f * (pf[j + 1] - pf[j - 1]);
    ad.push_back(std::fabs(d[j]));
  }
  int bj = -1;
  float bv = 0.0f;
  for (int j = 1; j < nprof - 1; ++j) {
    const float wgt = 1.0f - 0.3f * std::fabs((float)j - (nprof - 1) * 0.5f) / ((nprof - 1) * 0.5f);
    if (std::fabs(d[j]) * wgt > bv) {
      bv = std::fabs(d[j]) * wgt;
      bj = j;
    }
  }
  const float med = percentile(ad, 0.5);
  if (bj < 0 || std::fabs(d[bj]) < 0.04f || std::fabs(d[bj]) < 3.0f * med) return 0.0f;
  const float y0 = std::fabs(d[bj - 1]), y1 = std::fabs(d[bj]), y2 = std::fabs(d[bj + 1]);
  const float den = y0 - 2 * y1 + y2;
  const float sub = den < 0 ? 0.5f * (y0 - y2) / den : 0.0f;
  return -range + (bj + std::clamp(sub, -0.5f, 0.5f)) * (2 * range) / (nprof - 1);
}

void shift_edge(FilmRect& r, int edge, float sh) {
  const float th = r.angle_deg * kPi / 180.0f, c = std::cos(th), s = std::sin(th);
  if (edge < 2) {
    r.w -= (edge == 0 ? 1.0f : -1.0f) * sh;
    r.cx += c * sh * 0.5f;
    r.cy += s * sh * 0.5f;
  } else {
    r.h -= (edge == 2 ? 1.0f : -1.0f) * sh;
    r.cx += -s * sh * 0.5f;
    r.cy += c * sh * 0.5f;
  }
}

// The result for one format, before the base patch is sampled.
struct FormatResult {
  FrameDetection det;
  std::vector<FilmRect> base_patches;  // candidate film base patches, image coordinates
  double score = 0.0;                  // rectangle search score per search pixel (compares formats)
};

// Steps 5a-c for one format. Allocates: the search map (sw x sh floats, gated copy, three double prefix planes)
// and view-length profiles.
bool detect_one_format(const FilmImage& im, const WorkImage& wi, const Canvas& cv, const ClassInfo& ci,
                       const FrameDetectionParams& P, FilmFormat fmt, FormatResult& res) {
  const FormatGeom fg = format_geom(fmt);
  const int W = cv.w, H = cv.h;
  const bool positive = P.polarity == FILM_POSITIVE;
  // A canvas class on the film: picture or base, and on a positive also its dense (dark) parts.
  auto film = [positive](uint8_t c) { return c == CL_IMAGE || c == CL_BASE || (positive && c == CL_DARK); };
  std::vector<std::string> reasons;
  FrameDetection& out = res.det;

  // ---- 5a. The non-image map at <= 320 px (box fraction of non-image canvas pixels), gated, as prefix sums.
  const int f = std::max(1, (int)std::ceil(std::max(W, H) / 320.0));
  const int sw = W / f, sh = H / f;
  std::vector<float> nim((size_t)sw * sh, 0.0f);
  for (int y = 0; y < sh; ++y) {
    for (int x = 0; x < sw; ++x) {
      float a = 0.0f;
      for (int j = 0; j < f; ++j)
        for (int i = 0; i < f; ++i) a += ci.cls[(size_t)(y * f + j) * W + x * f + i] != CL_IMAGE ? 1.0f : 0.0f;
      nim[(size_t)y * sw + x] = a / (f * f);
    }
  }
  std::vector<float> gated;
  line_gate(nim, sw, sh, 0.3f, 0.85f, gated);
  SearchMap S;
  build_search(gated, sw, sh, S);

  const float aspect = fg.along / fg.across;  // the frame's w / h with the strip along x
  const bool square = std::fabs(fg.along - fg.across) < 1e-3f;
  SearchRes rh = best_rect(S, aspect), rv;
  if (!square) rv = best_rect(S, 1.0f / aspect);

  // Strip direction from the image itself: a strip along x runs off the left and/or right border, the top and
  // bottom borders see no film, and some rows (beyond the strip) hold no film at all.
  int axis_ev = 0;  // 1 along x, -1 along y
  {
    auto no_film_lines = [&](bool rows) {
      const int n = rows ? H : W, m = rows ? W : H;
      int cnt = 0, tot = 0;
      for (int a = 0; a < n; ++a) {
        int nf = 0, fi = 0;
        for (int b = 0; b < m; ++b) {
          const uint8_t c = rows ? ci.cls[(size_t)a * W + b] : ci.cls[(size_t)b * W + a];
          if (c == CL_INVALID) continue;
          (film(c) ? fi : nf)++;
        }
        if (fi + nf < 0.5f * m) continue;
        tot++;
        if (nf >= 0.9f * (fi + nf)) cnt++;
      }
      return tot ? (float)cnt / tot : 0.0f;
    };
    // Sides (0..2) whose first valid pixels inward from the border are film for >= 20% of the border.
    auto film_sides = [&](bool left_right) {
      int sides = 0;
      for (int side = 0; side < 2; ++side) {
        const int n = left_right ? H : W, m = left_right ? W : H;
        int fi = 0, tot = 0;
        for (int a = 0; a < n; ++a) {
          for (int k = 0; k < m; ++k) {
            const int b = side ? m - 1 - k : k;
            const uint8_t c = left_right ? ci.cls[(size_t)a * W + b] : ci.cls[(size_t)b * W + a];
            if (c == CL_INVALID) continue;
            tot++;
            fi += film(c) ? 1 : 0;
            break;
          }
        }
        if (tot && fi >= 0.2f * tot) sides++;
      }
      return sides;
    };
    const int t_lr = film_sides(true), t_tb = film_sides(false);
    if (t_lr >= 1 && t_tb == 0 && no_film_lines(true) > 0.03f) axis_ev = 1;
    else if (t_tb >= 1 && t_lr == 0 && no_film_lines(false) > 0.03f) axis_ev = -1;
  }
  bool tr;
  if (P.axis_hint == 'h' || P.axis_hint == 'H') tr = false;
  else if (P.axis_hint == 'v' || P.axis_hint == 'V') tr = true;
  else if (axis_ev != 0) tr = axis_ev < 0;
  else tr = !square && rv.score > rh.score;
  if (tr && !square && rv.h <= 0) tr = false;
  if (!tr && !square && rh.h <= 0) tr = true;
  // The frame's w / h on the canvas swaps with the strip direction; a square frame is the same either way.
  const SearchRes r0 = (tr && !square) ? rv : rh;
  if (r0.h <= 0) return false;
  res.score = r0.score / ((double)sw * sh);

  // ---- 5b. The strip model, in the view (strip along view x).
  const View v{&ci.cls, W, H, tr};
  const int VW = v.w(), VH = v.h();
  Rect fr;  // the frame estimate, view coordinates
  {
    const float x0 = (float)(r0.x * f), y0 = (float)(r0.y * f), x1 = (float)((r0.x + r0.w) * f),
                y1 = (float)((r0.y + r0.h) * f);
    if (tr) {
      fr.x0 = y0;
      fr.x1 = y1;
      fr.y0 = x0;
      fr.y1 = x1;
    } else {
      fr.x0 = x0;
      fr.x1 = x1;
      fr.y0 = y0;
      fr.y1 = y1;
    }
  }
  // Allocates: view-height and view-length profiles.
  auto refine_tb = [&](Rect& r) {
    const int xa = (int)(r.x0 + 0.15f * r.w()), xb = (int)(r.x1 - 0.15f * r.w());
    std::vector<float> rp(VH);
    for (int y = 0; y < VH; ++y) rp[y] = row_frac(v, y, xa, xb, is_image);
    const int delta = std::max(3, (int)(0.10f * r.h())), k = std::max(2, (int)(0.02f * r.h()));
    float st0, st1;
    r.y0 = refine_edge_profile(rp, r.y0, delta, k, true, &st0);
    r.y1 = refine_edge_profile(rp, r.y1, delta, k, false, &st1);
    return std::min(st0, st1);
  };
  auto refine_lr = [&](Rect& r, float* sl, float* sr) {
    const int ya = (int)(r.y0 + 0.12f * r.h()), yb = (int)(r.y1 - 0.12f * r.h());
    std::vector<float> cp(VW);
    for (int x = 0; x < VW; ++x) cp[x] = col_frac(v, x, ya, yb, is_image);
    const int delta = std::max(3, (int)(0.10f * r.w())), k = std::max(2, (int)(0.02f * r.w()));
    r.x0 = refine_edge_profile(cp, r.x0, delta, k, true, sl);
    r.x1 = refine_edge_profile(cp, r.x1, delta, k, false, sr);
  };
  // A partial frame cannot hold the full aspect, so the search rectangle can be shorter across the strip than the
  // frame: grow the band to the run of picture rows through its centre first.
  {
    const int xa = (int)(fr.x0 + 0.1f * fr.w()), xb = (int)(fr.x1 - 0.1f * fr.w());
    const int yc = (int)(0.5f * (fr.y0 + fr.y1));
    const int max_miss = std::max(2, (int)(0.01f * VH));
    auto run_end = [&](int dir) {
      int y = yc, last = yc, miss = 0;
      while (y >= 0 && y < VH) {
        if (row_frac(v, y, xa, xb, is_image) >= 0.45f) {
          last = y;
          miss = 0;
        } else if (++miss > max_miss) {
          break;
        }
        y += dir;
      }
      return last;
    };
    fr.y0 = std::min(fr.y0, (float)run_end(-1));
    fr.y1 = std::max(fr.y1, (float)run_end(+1) + 1);
  }
  const float st_tb = refine_tb(fr);
  const float across_px = fr.h();
  float ppm = across_px / fg.across;     // canvas pixels per mm
  const float L_exp = across_px * aspect;  // nominal frame length, canvas pixels

  // Picture fraction per view column over the band's middle 76% (cp, smoothed cps), and the fraction of gap-like
  // pixels (base, light, dark) over the whole band (cnon).
  const int ya = (int)(fr.y0 + 0.12f * fr.h()), yb = (int)(fr.y1 - 0.12f * fr.h());
  std::vector<float> cp(VW), cnon(VW), cvalid(VW);
  for (int x = 0; x < VW; ++x) {
    cp[x] = col_frac(v, x, ya, yb, is_image);
    cnon[x] = col_frac(v, x, (int)fr.y0, (int)fr.y1, is_gap);
    cvalid[x] = col_frac(v, x, (int)fr.y0, (int)fr.y1, is_valid);
  }
  std::vector<float> cps = cp;
  smooth1d(cps, std::max(0.6f, 0.15f * ppm));

  // Segments: runs of picture columns; a run ends at >= 0.6 mm of columns under 30% picture.
  struct Seg { int a, b; };
  std::vector<Seg> segs;
  {
    const int gap_min_px = std::max(1, (int)std::lround(0.6f * ppm));
    int x = 0;
    while (x < VW) {
      while (x < VW && cps[x] < 0.5f) ++x;
      if (x >= VW) break;
      int a = x, low = 0, b = x;
      while (x < VW) {
        if (cps[x] < 0.3f) {
          if (++low >= gap_min_px) break;
        } else {
          low = 0;
          b = x + 1;
        }
        ++x;
      }
      if (b - a >= std::max(3, (int)(0.12f * L_exp))) segs.push_back({a, b});
    }
  }
  // A real gap between frames is non-image and either light / holder or flat film base at the base level;
  // anything else (a thin band in a picture) is merged back when the merged run still fits one frame.
  {
    const float base_ll = std::log2(std::max(1e-6f, (ci.base[0] + ci.base[1] + ci.base[2]) / 3.0f));
    auto ll_at = [&](int x, int y) { return tr ? ci.logl[(size_t)x * W + y] : ci.logl[(size_t)y * W + x]; };
    auto real_gap = [&](int a, int b) {
      const int y0 = (int)(fr.y0 + 0.1f * fr.h()), y1 = (int)(fr.y1 - 0.1f * fr.h());
      int a2 = a + 1, b2 = b - 1;
      if (b2 <= a2) {
        a2 = a;
        b2 = b;
      }
      std::vector<float> vals;
      int non = 0, tot = 0;
      for (int x = a2; x < b2; ++x) {
        for (int y = std::max(0, y0); y < std::min(VH, y1); ++y) {
          const uint8_t c = v.at(x, y);
          if (c == CL_INVALID) continue;
          tot++;
          if (c != CL_IMAGE) non++;
          if (c == CL_BASE) vals.push_back(ll_at(x, y));
        }
      }
      if (tot == 0 || non < 0.85f * tot) return false;
      if (vals.size() < 0.5f * tot) return true;
      return std::fabs(percentile(vals, 0.5) - base_ll) <= 0.25f &&
             percentile(vals, 0.9) - percentile(vals, 0.1) <= 0.3f;
    };
    bool merged = true;
    while (merged && segs.size() > 1) {
      merged = false;
      for (size_t i = 0; i + 1 < segs.size(); ++i) {
        if (segs[i + 1].b - segs[i].a <= L_exp * (1.0f + fg.len_tol + 0.02f) && !real_gap(segs[i].b, segs[i + 1].a)) {
          segs[i].b = segs[i + 1].b;
          segs.erase(segs.begin() + i + 1);
          merged = true;
          break;
        }
      }
    }
  }
  if (segs.empty()) segs.push_back({(int)fr.x0, (int)fr.x1});

  // What bounds a frame end going outward from edge_x: 0 open (the image border, or picture right away), 1 a film
  // gap (base, then film), 2 a hard limit (holder or light: the end of the visible film).
  auto bound_type = [&](int edge_x, int dir) {
    const int maxd = std::max(2, (int)std::lround(fg.gap_max * ppm));
    int run = 0, hard = 0, base = 0;
    for (int d = 0; d < maxd; ++d) {
      const int x = edge_x + dir * d;
      if (x < 0 || x >= VW || cvalid[x] < 0.9f) break;
      if (cnon[x] >= 0.85f) {
        ++run;
        if (col_frac(v, x, (int)fr.y0, (int)fr.y1, is_hard) >= 0.6f) hard++;
        else base++;
        if (run >= 3 && hard >= 2 && hard > base) return 2;
      } else if (cps[x] >= 0.5f) {
        return run >= 2 ? 1 : 0;
      }
    }
    if (run >= 2) return hard > base ? 2 : 1;
    return 0;
  };
  // Each segment becomes a frame candidate. Its measured length is kept when both ends are bounded and it is near
  // the nominal length (cameras differ) or when a hard limit cuts it; otherwise the nominal length is laid from the
  // bounded end (or around the centre when neither end is bounded).
  struct Cand { Rect r; int lt, rt; float vis; bool complete; };
  std::vector<Cand> cands;
  const float tol = fg.len_tol;
  for (const Seg& sg : segs) {
    Rect r = fr;
    r.x0 = (float)sg.a;
    r.x1 = (float)sg.b;
    float sl = 0.0f, sr = 0.0f;
    refine_lr(r, &sl, &sr);
    const int lt = bound_type((int)r.x0 - 1, -1), rt = bound_type((int)r.x1, +1);
    const bool lg = lt > 0, rg = rt > 0;
    const float len = r.w();
    if (len > L_exp * (1 + tol) || len < L_exp * (1 - tol)) {
      if (lg && rg && len > L_exp * (1 - 2 * tol) && len < L_exp * (1 + 2 * tol)) {
        // camera-specific frame length: keep
      } else if (lg && rg && len < L_exp && (lt == 2 || rt == 2)) {
        // cut by the holder aperture or the strip end: keep the visible extent
      } else if (lg && rg && len < L_exp) {
        // bounded by gaps but short (a shadow band at one end): lay the frame from the cleaner edge
        if (sl >= sr) r.x1 = r.x0 + L_exp;
        else r.x0 = r.x1 - L_exp;
      } else if (lg && !rg) {
        r.x1 = r.x0 + L_exp;
      } else if (rg && !lg) {
        r.x0 = r.x1 - L_exp;
      } else if (len > L_exp * (1 + tol)) {
        if (lg && rg && len > 1.6f * L_exp) {
          r.x1 = r.x0 + L_exp;  // frames with no visible gap: the first; the others are found as neighbours
        } else if (lg && rg) {
          if (sl >= sr) r.x1 = r.x0 + L_exp;
          else r.x0 = r.x1 - L_exp;
        } else {
          const float c = 0.5f * (r.x0 + r.x1);
          r.x0 = c - 0.5f * L_exp;
          r.x1 = c + 0.5f * L_exp;
        }
      } else {
        const float c = 0.5f * (r.x0 + r.x1);  // truncated by the image on both ends
        r.x0 = c - 0.5f * L_exp;
        r.x1 = c + 0.5f * L_exp;
      }
    }
    float vis = 0.0f;
    for (int x = (int)std::floor(r.x0); x < (int)std::ceil(r.x1); ++x) vis += (x >= 0 && x < VW) ? cvalid[x] : 0.0f;
    vis /= std::max(1.0f, L_exp);
    const bool complete = r.x0 >= 0 && r.x1 <= VW && vis >= 0.97f * (r.w() / L_exp);
    cands.push_back({r, lt, rt, vis, complete});
  }
  // The primary: the complete frame nearest the image centre, else the most visible frame.
  const float centre = tr ? cv.cyc : cv.cxc;
  int pi = 0;
  double pbest = -1e30;
  for (size_t i = 0; i < cands.size(); ++i) {
    const Cand& c = cands[i];
    const double key = c.complete ? 1e6 - std::fabs(0.5f * (c.r.x0 + c.r.x1) - centre) : c.vis * 1e3;
    if (key > pbest) {
      pbest = key;
      pi = (int)i;
    }
  }
  const Rect pr = cands[pi].r;

  // The neighbours: from the primary outward at its length, the next frame starting where picture resumes after a
  // gap (or the nominal gap when none is seen); stops at a blank frame, the strip end, the holder or the border.
  std::vector<std::pair<int, Rect>> sec;
  {
    const float L = pr.w();
    for (int dir = -1; dir <= 1; dir += 2) {
      Rect cur = pr;
      for (int idx = 1; idx < 40; ++idx) {
        const int edge = dir > 0 ? (int)std::ceil(cur.x1) : (int)std::floor(cur.x0) - 1;
        const int maxd = (int)std::lround(fg.gap_max * ppm) + 2;
        int found = -1;
        for (int d = std::max(1, (int)(fg.gap_min * 0.5f * ppm)); d < maxd; ++d) {
          const int x = edge + dir * d;
          if (x < 0 || x >= VW) break;
          if (cps[x] >= 0.5f) {
            found = d;
            break;
          }
        }
        const float gap = found > 0 ? (float)found : fg.gap_nominal * ppm;
        Rect nr = cur;
        if (dir > 0) {
          nr.x0 = cur.x1 + gap;
          nr.x1 = nr.x0 + L;
        } else {
          nr.x1 = cur.x0 - gap;
          nr.x0 = nr.x1 - L;
        }
        if (nr.x1 < 0 || nr.x0 > VW) break;
        const int xa2 = std::max(0, (int)nr.x0), xb2 = std::min(VW, (int)nr.x1);
        if (xb2 - xa2 < std::max(2.0f, 0.03f * L)) break;
        float img = 0.0f, valid = 0.0f;
        for (int x = xa2; x < xb2; ++x) {
          img += cp[x];
          valid += cvalid[x];
        }
        if (valid < 1 || img / valid < 0.35f) break;
        if (found > 0) {
          Rect t = nr;
          float sl, sr;
          refine_lr(t, &sl, &sr);
          if (dir > 0) {
            nr.x0 = t.x0;
            nr.x1 = nr.x0 + L;
          } else {
            nr.x1 = t.x1;
            nr.x0 = nr.x1 - L;
          }
        }
        if (valid / L >= 0.03f) sec.push_back({dir * idx, nr});
        cur = nr;
      }
    }
  }

  // The strip's edges (s0, s1): from the frame band outward while rows over the frame are film; its ends (e0, e1):
  // from the frame outward while columns over the strip are film.
  const int fx0 = std::max(0, (int)pr.x0), fx1 = std::min(VW, (int)pr.x1);
  int s0 = (int)pr.y0, s1 = (int)pr.y1;
  bool edge0 = false, edge1 = false;
  {
    const int lim = (int)std::lround((fg.film_width - fg.across) * 0.5f * ppm * 1.6f) + 3;
    for (int y = (int)pr.y0 - 1; y >= std::max(0, (int)pr.y0 - lim); --y) {
      if (row_frac(v, y, fx0, fx1, is_valid) < 0.6f) break;
      if (row_frac(v, y, fx0, fx1, film) < 0.25f) {
        edge0 = true;
        break;
      }
      s0 = y;
    }
    for (int y = (int)pr.y1; y < std::min(VH, (int)pr.y1 + lim); ++y) {
      if (row_frac(v, y, fx0, fx1, is_valid) < 0.6f) break;
      if (row_frac(v, y, fx0, fx1, film) < 0.25f) {
        edge1 = true;
        break;
      }
      s1 = y + 1;
    }
  }
  int e0 = fx0, e1 = fx1;
  bool end0 = false, end1 = false;
  {
    const int ra = std::max(0, s0), rb = std::min(VH, s1);
    for (int x = fx0 - 1; x >= 0; --x) {
      if (col_frac(v, x, (int)fr.y0, (int)fr.y1, is_valid) < 0.6f) break;
      if (col_frac(v, x, ra, rb, film) < 0.4f) {
        end0 = true;
        break;
      }
      e0 = x;
    }
    for (int x = fx1; x < VW; ++x) {
      if (col_frac(v, x, (int)fr.y0, (int)fr.y1, is_valid) < 0.6f) break;
      if (col_frac(v, x, ra, rb, film) < 0.4f) {
        end1 = true;
        break;
      }
      e1 = x + 1;
    }
  }
  if (end0 || end1) reasons.push_back("strip_end_visible");
  if (edge0 && edge1) {
    // Both strip edges seen: average px per mm from the strip width, unless it disagrees (wrong format, cut film).
    const float ppm2 = (s1 - s0) / fg.film_width;
    if (ppm2 / ppm > 0.92f && ppm2 / ppm < 1.08f) ppm = 0.5f * (ppm + ppm2);
    else reasons.push_back("strip_width_mismatch");
  } else {
    reasons.push_back("strip_edges_partial");
  }

  // 35mm perforations: in each rebate band the longest run of rows where light covers 20..80% of the strip's
  // length, then light runs along it of a hole's width, accepted when their median pitch is 4.75 mm (+25% / -20%).
  std::vector<Rect> holes;
  if (fg.sprockets) {
    const float hw = kHoleW * ppm, hh = kHoleH * ppm;
    auto light = [](uint8_t c) { return c == CL_LIGHT; };
    for (int side = 0; side < 2; ++side) {
      int ya2 = side == 0 ? std::max(0, (int)(pr.y0 - 6.0f * ppm)) : (int)pr.y1;
      int yb2 = side == 0 ? (int)pr.y0 : std::min(VH, (int)(pr.y1 + 6.0f * ppm));
      if (side == 0 && edge0) ya2 = std::max(ya2, s0);
      if (side == 1 && edge1) yb2 = std::min(yb2, s1);
      if (yb2 - ya2 < hh * 0.5f) continue;
      int hy0 = -1, hy1 = -1, cur = -1;
      for (int y = ya2; y <= yb2; ++y) {
        const float lr = y < yb2 ? row_frac(v, y, e0, e1, light) : 0.0f;
        const bool in = y < yb2 && lr >= 0.2f && lr <= 0.8f;
        if (in && cur < 0) cur = y;
        if (!in && cur >= 0) {
          if (y - cur > hy1 - hy0) {
            hy0 = cur;
            hy1 = y;
          }
          cur = -1;
        }
      }
      if (hy0 < 0 || hy1 - hy0 < 0.4f * hh || hy1 - hy0 > 1.8f * hh) continue;
      std::vector<Rect> found;
      int x = 0;
      while (x < VW) {
        while (x < VW && col_frac(v, x, hy0, hy1, light) < 0.5f) ++x;
        const int a = x;
        while (x < VW && col_frac(v, x, hy0, hy1, light) >= 0.5f) ++x;
        if (x > a && x - a >= 0.5f * hw && x - a <= 1.7f * hw) {
          Rect h;
          h.x0 = (float)a;
          h.x1 = (float)x;
          h.y0 = (float)hy0;
          h.y1 = (float)hy1;
          found.push_back(h);
        }
      }
      if (found.size() >= 2) {
        std::vector<float> pitches;
        for (size_t i = 1; i < found.size(); ++i) pitches.push_back(found[i].x0 - found[i - 1].x0);
        const float pm = percentile(pitches, 0.5);
        if (pm > 0.8f * kHolePitch * ppm && pm < 1.25f * kHolePitch * ppm) {
          holes.insert(holes.end(), found.begin(), found.end());
        }
      }
    }
    if (holes.empty()) reasons.push_back("no_sprockets");
  }

  // Film base patches next to the primary: each adjacent inter-frame gap (bounded by film on both sides), over the
  // middle 60% of the band, and the rebate above and below the frame, each as long as its columns or rows stay >= 90%
  // base, shrunk by a quarter of their width on each side.
  {
    auto base = [](uint8_t c) { return c == CL_BASE; };
    const Geometry G{&cv, &wi, tr};
    const int by0 = (int)(pr.y0 + 0.2f * pr.h()), by1 = (int)(pr.y1 - 0.2f * pr.h());
    for (int dir = -1; dir <= 1; dir += 2) {
      int x = dir > 0 ? (int)std::ceil(pr.x1) : (int)std::floor(pr.x0) - 1;
      const int maxd = (int)std::lround(fg.gap_max * ppm) + 2;
      int a = -1, b = -1;
      for (int d = 0; d < maxd; ++d, x += dir) {
        if (x < 0 || x >= VW) break;
        const bool in = col_frac(v, x, by0, by1, base) >= 0.9f;
        if (in) {
          if (a < 0) a = x;
          b = x;
        } else if (a >= 0) {
          break;
        }
      }
      if (a < 0 || std::abs(b - a) + 1 < 3) continue;
      Rect g;
      const int lo = std::min(a, b), hi = std::max(a, b) + 1;
      const float q = 0.25f * (hi - lo);
      g.x0 = lo + q;
      g.x1 = hi - q;
      g.y0 = (float)by0;
      g.y1 = (float)by1;
      res.base_patches.push_back(G.to_image(g));
    }
    const int bx0 = (int)(pr.x0 + 0.2f * pr.w()), bx1 = (int)(pr.x1 - 0.2f * pr.w());
    for (int dir = -1; dir <= 1; dir += 2) {
      int y = dir > 0 ? (int)std::ceil(pr.y1) : (int)std::floor(pr.y0) - 1;
      const int maxd = (int)std::lround((fg.film_width - fg.across) * 0.5f * ppm) + 2;
      int a = -1, b = -1;
      for (int d = 0; d < maxd; ++d, y += dir) {
        if (y < 0 || y >= VH) break;
        const bool in = row_frac(v, y, bx0, bx1, base) >= 0.9f;
        if (in) {
          if (a < 0) a = y;
          b = y;
        } else if (a >= 0) {
          break;
        }
      }
      if (a < 0 || std::abs(b - a) + 1 < 3) continue;
      Rect g;
      const int lo = std::min(a, b), hi = std::max(a, b) + 1;
      const float q = 0.25f * (hi - lo);
      g.y0 = lo + q;
      g.y1 = hi - q;
      g.x0 = (float)bx0;
      g.x1 = (float)bx1;
      res.base_patches.push_back(G.to_image(g));
    }
  }

  // Strip direction: known for a non-square frame, or from neighbours, both strip edges, holes or the image.
  char axis = tr ? 'v' : 'h';
  if (square && sec.empty() && !(edge0 && edge1) && holes.empty() && axis_ev == 0 && !P.axis_hint) axis = '?';

  // ---- Map to image coordinates.
  const Geometry G{&cv, &wi, tr};
  const float nominal_area = L_exp * across_px * wi.sx * wi.sy;
  FilmRect prim = G.to_image(pr);
  // ---- 5c. Sub-pixel edges of the primary on the full-resolution image, within +-2.5 work pixels.
  if (P.refine) {
    const float range = std::max(3.0f, 2.5f * 0.5f * (wi.sx + wi.sy));
    for (int e = 0; e < 4; ++e) {
      const float s = refine_edge_full(im, prim, e, range);
      if (s != 0.0f) shift_edge(prim, e, s);
    }
  }
  out.primary = make_frame(prim, nominal_area, 0, im.width, im.height);
  out.secondaries.clear();
  for (const auto& s : sec)
    out.secondaries.push_back(make_frame(G.to_image(s.second), nominal_area, s.first, im.width, im.height));
  std::sort(out.secondaries.begin(), out.secondaries.end(),
            [](const FilmFrame& a, const FilmFrame& b) { return a.index < b.index; });
  {
    Rect fb;
    fb.x0 = (float)e0;
    fb.x1 = (float)e1;
    fb.y0 = (float)s0;
    fb.y1 = (float)s1;
    out.film_base = G.to_image(fb);
    out.has_film_base = (s1 - s0) > across_px * 1.02f || (e1 - e0) > pr.w() * 1.02f;
  }
  out.sprockets.clear();
  for (const Rect& h : holes) out.sprockets.push_back(G.to_image(h));
  // The film gaps at the primary's ends: from its edge outward over the gap-like columns.
  for (int side = 0; side < 2; ++side) {
    if ((side == 0 ? cands[pi].lt : cands[pi].rt) != 1) continue;
    const int dir = side == 0 ? -1 : 1;
    const int start = side == 0 ? (int)std::floor(pr.x0) - 1 : (int)std::ceil(pr.x1);
    const int maxd = std::max(2, (int)std::lround(fg.gap_max * ppm));
    int far = start;
    for (int d = 0; d < maxd; ++d) {
      const int x = start + dir * d;
      if (x < 0 || x >= VW || cnon[x] < 0.85f) break;
      far = x;
    }
    Rect g;
    g.x0 = (float)std::min(start, far);
    g.x1 = (float)std::max(start, far) + 1;
    g.y0 = pr.y0;
    g.y1 = pr.y1;
    const FilmRect gi = G.to_image(g);
    const float c = tr ? gi.cy : gi.cx, half = 0.5f * (tr ? gi.h : gi.w);
    // Clamped to the image: the work grid and a tilted band can reach past the border, and a negative start would
    // read as the {-1, -1} sentinel.
    const int lim = tr ? im.height : im.width;
    out.gaps[side][0] = std::clamp((int)std::lround(c - half), 0, lim);
    out.gaps[side][1] = std::clamp((int)std::lround(c + half), 0, lim);
    out.n_gaps++;
  }
  out.axis = axis;
  out.rotation_deg = cv.theta;
  out.px_per_mm = ppm * 0.5f * (wi.sx + wi.sy);
  out.format = fmt;
  out.orientation = square ? "square" : (out.primary.rect.w > out.primary.rect.h ? "landscape" : "portrait");
  float conf = 1.0f;
  if (!cands[pi].complete) {
    conf *= 0.7f;
    reasons.push_back("primary_partial");
  }
  if (!(cands[pi].lt > 0 && cands[pi].rt > 0)) {
    conf *= 0.85f;
    reasons.push_back("frame_ends_unbounded");
  }
  if (st_tb < 0.4f) {
    conf *= 0.8f;
    reasons.push_back("weak_transverse_edges");
  }
  out.conf = conf;
  out.reason = join(reasons);
  out.ok = true;
  return true;
}

}  // namespace


int FilmImage::samples_x() const {
  if (step <= 0) return 0;
  int n = width;
  for (const FilmChannel& c : ch) n = std::min(n, (width - c.x0) / step);
  return std::max(0, n);
}

int FilmImage::samples_y() const {
  if (step <= 0) return 0;
  int n = height;
  for (const FilmChannel& c : ch) n = std::min(n, (height - c.y0) / step);
  return std::max(0, n);
}

FilmImage film_image_from_interleaved(const uint16_t* pixels, int width, int height, int channels,
                                      ptrdiff_t row_stride, float white) {
  FilmImage im;
  im.width = width;
  im.height = height;
  im.step = 1;
  im.white = white;
  if (!row_stride) row_stride = (ptrdiff_t)width * channels;
  for (int c = 0; c < 3; ++c) {
    im.ch[c].data = pixels + c;
    im.ch[c].pitch = channels;
    im.ch[c].stride = row_stride;
  }
  return im;
}

FilmImage film_image_from_planes(const uint16_t* r, const uint16_t* g, const uint16_t* b, int width, int height,
                                 ptrdiff_t row_stride, float white) {
  FilmImage im;
  im.width = width;
  im.height = height;
  im.step = 1;
  im.white = white;
  const uint16_t* p[3] = {r, g, b};
  for (int c = 0; c < 3; ++c) {
    im.ch[c].data = p[c];
    im.ch[c].pitch = 1;
    im.ch[c].stride = row_stride ? row_stride : width;
  }
  return im;
}

bool film_image_from_cfa(const uint16_t* mosaic, int width, int height, ptrdiff_t row_stride, int period,
                         const std::function<int(int row, int col)>& color, const float black[4], float white,
                         FilmImage& out) {
  out = FilmImage();
  if (!mosaic || period <= 0 || width < period || height < period) return false;
  out.width = width;
  out.height = height;
  out.step = period;
  out.white = white;
  bool found[3] = {false, false, false};
  // Two passes for G: a site of colour 1 is preferred to the second green (3) of a Bayer quad.
  for (int pass = 0; pass < 2; ++pass) {
    for (int row = 0; row < period; ++row) {
      for (int col = 0; col < period; ++col) {
        const int k = color(row, col);
        const int c = k == 3 ? 1 : k;
        if (c < 0 || c > 2 || found[c] || (c == 1 && (k == 3) != (pass == 1))) continue;
        found[c] = true;
        FilmChannel& ch = out.ch[c];
        ch.data = mosaic + (ptrdiff_t)row * row_stride + col;
        ch.pitch = period;
        ch.stride = row_stride * period;
        ch.x0 = col;
        ch.y0 = row;
        ch.black = black ? black[k] : 0.0f;
      }
    }
  }
  return found[0] && found[1] && found[2];
}

const char* film_format_name(FilmFormat f) {
  switch (f) {
    case FILM_35MM: return "35mm";
    case FILM_645: return "645";
    case FILM_6X6: return "6x6";
    case FILM_6X7: return "6x7";
    case FILM_6X9: return "6x9";
    default: return "auto";
  }
}

FilmFormat film_format_from_name(const std::string& s) {
  if (s == "35mm" || s == "135") return FILM_35MM;
  if (s == "645") return FILM_645;
  if (s == "6x6") return FILM_6X6;
  if (s == "6x7") return FILM_6X7;
  if (s == "6x9") return FILM_6X9;
  return FILM_FORMAT_AUTO;
}

void FilmRect::corners(float xy[8]) const {
  const float th = angle_deg * kPi / 180.0f, c = std::cos(th), s = std::sin(th);
  const float ux[4] = {-0.5f, 0.5f, 0.5f, -0.5f}, uy[4] = {-0.5f, -0.5f, 0.5f, 0.5f};
  for (int i = 0; i < 4; ++i) {
    const float X = ux[i] * w, Y = uy[i] * h;
    xy[2 * i] = cx + c * X - s * Y;
    xy[2 * i + 1] = cy + s * X + c * Y;
  }
}

bool film_sample_patch(const FilmImage& im, const FilmRect& patch, float rgb[3], float* std_d) {
  const int nx = im.samples_x(), ny = im.samples_y();
  float xy[8];
  patch.corners(xy);
  float bx0 = xy[0], by0 = xy[1], bx1 = xy[0], by1 = xy[1];
  for (int k = 1; k < 4; ++k) {
    bx0 = std::min(bx0, xy[2 * k]);
    bx1 = std::max(bx1, xy[2 * k]);
    by0 = std::min(by0, xy[2 * k + 1]);
    by1 = std::max(by1, xy[2 * k + 1]);
  }
  const float th = patch.angle_deg * kPi / 180.0f, c = std::cos(th), s = std::sin(th);
  std::vector<float> v[3];
  for (int ch = 0; ch < 3; ++ch) {
    const FilmChannel& fc = im.ch[ch];
    // Lattice samples whose photosite centre (x0 + i * step + 0.5) lies in the patch's bounding box ...
    const int i0 = std::max(0, (int)std::ceil((bx0 - 0.5f - fc.x0) / im.step));
    const int i1 = std::min(nx - 1, (int)std::floor((bx1 - 0.5f - fc.x0) / im.step));
    const int j0 = std::max(0, (int)std::ceil((by0 - 0.5f - fc.y0) / im.step));
    const int j1 = std::min(ny - 1, (int)std::floor((by1 - 0.5f - fc.y0) / im.step));
    for (int j = j0; j <= j1; ++j) {
      for (int i = i0; i <= i1; ++i) {
        // ... and inside the turned patch.
        const float dx = fc.x0 + i * im.step + 0.5f - patch.cx, dy = fc.y0 + j * im.step + 0.5f - patch.cy;
        if (std::fabs(c * dx + s * dy) > 0.5f * patch.w || std::fabs(-s * dx + c * dy) > 0.5f * patch.h) continue;
        v[ch].push_back(sample_at(im, ch, i, j));
      }
    }
  }
  if (v[0].size() < 32 || v[1].size() < 32 || v[2].size() < 32) return false;
  if (std_d) {
    double sum = 0.0, sum2 = 0.0;
    for (float g : v[1]) {
      const double d = std::log10(std::max(g, 1.0f));
      sum += d;
      sum2 += d * d;
    }
    const double m = sum / v[1].size();
    *std_d = (float)std::sqrt(std::max(0.0, sum2 / v[1].size() - m * m));
  }
  for (int ch = 0; ch < 3; ++ch) rgb[ch] = percentile(v[ch], 0.5);
  return true;
}

bool detect_film_frame(const FilmImage& im, const FrameDetectionParams& P, FrameDetection& out) {
  out = FrameDetection();
  if (im.samples_x() < 16 || im.samples_y() < 16 || im.white <= 0.0f) return false;
  for (const FilmChannel& c : im.ch)
    if (!c.data) return false;

  // 1. Work image: ww x wh, long side <= work_size.
  WorkImage wi;
  downsample(im, P.work_size, wi);

  // Clipped light: a channel whose box maxima sit at a plateau (>= 0.05% of the work pixels within 0.5% of the
  // channel's maximum, as the previous detector); 2% below that plateau counts as clipped.
  float clip_level[3];
  {
    const Plane* mp[3] = {&wi.mr, &wi.mg, &wi.mb};
    for (int c = 0; c < 3; ++c) {
      const float mx = *std::max_element(mp[c]->d.begin(), mp[c]->d.end());
      size_t cnt = 0;
      for (float v : mp[c]->d) cnt += v >= 0.995f * mx;
      const bool plateau = cnt >= std::max<size_t>(4, mp[c]->d.size() / 2000) && mx > 0.05f;
      clip_level[c] = plateau ? 0.98f * mx : 1e9f;
    }
  }
  // 2. Tilt, ignored below 0.03 degree or when no direction stands out.
  float rot_conf = 0.0f;
  float theta = estimate_rotation(wi, P.max_rotation_deg, &rot_conf);
  if (std::fabs(theta) < 0.03f || rot_conf < 0.05f) theta = 0.0f;

  // 3, 4. Canvas and classes.
  Canvas cv;
  build_canvas(wi, theta, clip_level, cv);
  ClassInfo ci;
  classify(cv, P.polarity, ci);

  // 5. The format given, or the best of all: the rectangle score, discounted by the confidence and by a strip width
  // that disagrees with the format.
  std::vector<FilmFormat> fmts;
  if (P.format == FILM_FORMAT_AUTO) fmts = {FILM_35MM, FILM_645, FILM_6X6, FILM_6X7, FILM_6X9};
  else fmts = {P.format};
  FormatResult best;
  double best_key = -1e30;
  bool ok = false;
  for (FilmFormat fm : fmts) {
    FormatResult r;
    if (!detect_one_format(im, wi, cv, ci, P, fm, r)) continue;
    double key = r.score;
    if (P.format == FILM_FORMAT_AUTO) {
      key *= r.det.conf;
      if (r.det.reason.find("strip_width_mismatch") != std::string::npos) key *= 0.7;
    }
    if (key > best_key) {
      best_key = key;
      best = std::move(r);
      ok = true;
    }
  }
  if (!ok) return false;
  out = std::move(best.det);

  // 6. The crop: the largest axis-aligned rectangle of the frame's aspect inside the tilted primary, inset and
  // aligned (x and y rounded up, the far edges down), within the image.
  {
    float p, q;
    inscribed_half(out.primary.rect, p, q);
    p *= 1.0f - 2.0f * P.crop_inset;
    q *= 1.0f - 2.0f * P.crop_inset;
    const int a = std::max(1, P.crop_align);
    auto up = [a](float v) { return (int)std::ceil(v / a) * a; };
    auto down = [a](float v) { return (int)std::floor(v / a) * a; };
    const int x0 = std::max(0, up(out.primary.rect.cx - p)), y0 = std::max(0, up(out.primary.rect.cy - q));
    const int x1 = std::min(down((float)im.width), down(out.primary.rect.cx + p));
    const int y1 = std::min(down((float)im.height), down(out.primary.rect.cy + q));
    out.crop[0] = x0;
    out.crop[1] = y0;
    out.crop[2] = std::max(0, x1 - x0);
    out.crop[3] = std::max(0, y1 - y0);
    if (out.crop[2] < 16 || out.crop[3] < 16) return false;
  }
  // The base patch: the flattest candidate on the full-resolution image.
  bool any_patch = false;
  for (const FilmRect& r : best.base_patches) {
    float rgb[3], sd;
    if (!film_sample_patch(im, r, rgb, &sd)) continue;
    if (!any_patch || sd < out.base_std_d) {
      any_patch = true;
      out.base_patch = r;
      for (int c = 0; c < 3; ++c) out.base_rgb[c] = rgb[c];
      out.base_std_d = sd;
    }
  }
  if (any_patch) {
    float xy[8];
    out.base_patch.corners(xy);
    double area;
    clip_quad(xy, im.width, im.height, out.base_box, area);
  }
  out.has_base = any_patch && out.base_std_d <= 0.025f;
  out.light_clipped = ci.light_clipped;
  if (theta != 0.0f) out.reason += out.reason.empty() ? "rotated" : ",rotated";
  return true;
}
