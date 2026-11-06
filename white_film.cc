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

#include "white_film.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>

#include "color_pipeline.h"
#include "np_log.h"

namespace {

// Fitted on the IT8 calibration readings (Portra 400, ILCE-7RM4, white LED): the starting exponents, and the matrix
// from x to XYZ D50 (its scale is the middle bracket's exposure; the search's exposure takes it).
// tools/white_film/fit_calibration.py; report Iteration 1.
// The report is docs/white_light_film_model.md; constants below without a source: its section 4, table of constants.
constexpr double kPCal[3] = {2.331477, 1.904209, 1.707575};
constexpr double kM[9] = {1.065838, 0.257458, 0.208911, 0.367323, 1.357380, -0.167243, -0.008611, -0.072967, 1.350631};
// XYZ D50 to linear sRGB D65 (Bradford).
constexpr double kXyzD50ToSrgb[9] = {3.1338561, -1.6168667, -0.4906146, -0.9787684, 1.9161415, 0.0334540,
                                     0.0719453, -0.2289914, 1.4052427};
// The tone curve, per channel: log-logistic with contrast kToneC, mid-grey 0.18 to kToneGrey, white at infinity.
constexpr double kToneC = 1.4, kToneGrey = 0.18;
// The weights of a frame's own density-range ratios against the calibration's (report Iteration 3).
constexpr double kShrink[2] = {0.5, 0.6};
// Lightness offset: the intent optimum is this many stops brighter than the renders rated best (report Iteration 2).
constexpr double kDarken = 0.7;

constexpr float kFloor = 0.5f;         // camera values (16-bit counts) below this are taken as this
constexpr double kPictureStops = 0.1;  // a cell holds picture when a channel is this far below the base
constexpr int kPreviewSide = 1280;     // long side of the analysis image
constexpr int kCells = 96;             // cells a side for the lightness statistics and the density ratios
constexpr double kStartL = 47.0;       // each search starts at the exposure that puts the median L* here
constexpr double kSlopeTarget = 0.9;   // the contrast step aims at this slope of rendered L* on the field's

using Vec3 = std::array<double, 3>;

struct Image {
  int w = 0, h = 0;
  std::vector<float> px;  // 3 a pixel
};

// The w x h image px(x, y, v) reads, averaged into ow x oh cells (ow <= w, oh <= h): cell (cx, cy) takes the pixels
// with x * ow / w == cx and y * oh / h == cy.
template <class F>
Image area_mean(int w, int h, int ow, int oh, F px) {
  Image out{ow, oh, std::vector<float>((size_t)ow * oh * 3)};
  auto first = [](int c, int n, int on) { return (int)(((long long)c * n + on - 1) / on); };
  #pragma omp parallel for schedule(static)
  for (int cy = 0; cy < oh; ++cy) {
    const int y0 = first(cy, h, oh), y1 = first(cy + 1, h, oh);
    for (int cx = 0; cx < ow; ++cx) {
      const int x0 = first(cx, w, ow), x1 = first(cx + 1, w, ow);
      double s[3] = {0.0, 0.0, 0.0};
      float v[3];
      for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
          px(x, y, v);
          for (int c = 0; c < 3; ++c) s[c] += v[c];
        }
      for (int c = 0; c < 3; ++c) out.px[((size_t)cy * ow + cx) * 3 + c] = (float)(s[c] / ((y1 - y0) * (x1 - x0)));
    }
  }
  return out;
}

// The rectangle (x0, y0, w, h) of img averaged into ow x oh cells.
Image shrink(const Image& img, int x0, int y0, int w, int h, int ow, int oh) {
  return area_mean(w, h, ow, oh, [&](int x, int y, float v[3]) {
    std::copy_n(&img.px[((size_t)(y + y0) * img.w + x + x0) * 3], 3, v);
  });
}

// The q-quantile of v, interpolated linearly (numpy's default).
double quantile(std::vector<float> v, double q) {
  const double pos = q * (double)(v.size() - 1);
  const size_t i = (size_t)pos;
  std::nth_element(v.begin(), v.begin() + i, v.end());
  if (i + 1 >= v.size()) return v[i];
  const double next = *std::min_element(v.begin() + i + 1, v.end());
  return v[i] + (pos - (double)i) * (next - v[i]);
}

std::vector<float> channel(const std::vector<float>& v3, int c) {
  std::vector<float> v(v3.size() / 3);
  for (size_t i = 0; i < v.size(); ++i) v[i] = v3[i * 3 + c];
  return v;
}

// Step 3, the film model, in three parts. The stops below the base per channel (negative: brighter than the base).
void stops(const double base[3], const float v[3], float d[3]) {
  for (int c = 0; c < 3; ++c) d[c] = (float)std::log2(base[c] / std::max(v[c], kFloor));
}

// The film model to XYZ: x = w (2^(p d) - 1), with brighter than the base as no film (d = 0); XYZ = M x.
void film_xyz(const WhiteFilmParams& q, const float d[3], double xyz[3]) {
  double x[3];
  for (int c = 0; c < 3; ++c) x[c] = q.w[c] * (std::exp2(q.p[c] * std::max(d[c], 0.0f)) - 1.0);
  for (int r = 0; r < 3; ++r) xyz[r] = kM[3 * r] * x[0] + kM[3 * r + 1] * x[1] + kM[3 * r + 2] * x[2];
}

// XYZ at exposure e to linear sRGB through the tone curve, in [0, 1).
void display(const double xyz[3], double e, float out[3]) {
  static const double k = std::pow(0.18, kToneC) * (1.0 - kToneGrey) / kToneGrey;
  for (int r = 0; r < 3; ++r) {
    const double rgb = e * (kXyzD50ToSrgb[3 * r] * xyz[0] + kXyzD50ToSrgb[3 * r + 1] * xyz[1] +
                            kXyzD50ToSrgb[3 * r + 2] * xyz[2]);
    const double s = std::pow(std::max(rgb, 0.0), kToneC);
    out[r] = (float)(s / (s + k));
  }
}

void film_display(const double base[3], const WhiteFilmParams& q, const float v[3], float out[3]) {
  float d[3];
  double xyz[3];
  stops(base, v, d);
  film_xyz(q, d, xyz);
  display(xyz, q.e, out);
}

// The cells of an image that hold picture: each one's stops below the base, and its index.
struct Cells {
  std::vector<float> d;  // 3 a cell
  std::vector<int> idx;
};

Cells picture_cells(const Image& img, const double base[3]) {
  Cells c;
  for (int i = 0; i < img.w * img.h; ++i) {
    float d[3];
    stops(base, &img.px[(size_t)i * 3], d);
    if (std::max({d[0], d[1], d[2]}) <= kPictureStops) continue;
    c.d.insert(c.d.end(), d, d + 3);
    c.idx.push_back(i);
  }
  return c;
}

std::vector<double> cell_xyz(const Cells& c, const WhiteFilmParams& q) {
  std::vector<double> xyz(c.d.size());
  for (size_t i = 0; i < c.idx.size(); ++i) film_xyz(q, &c.d[i * 3], &xyz[i * 3]);
  return xyz;
}

// The cells' CIELAB at exposure e, through the 8-bit render as the preview has it.
std::vector<float> xyz_lab(const std::vector<double>& xyz, double e) {
  std::vector<float> lab(xyz.size());
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < xyz.size() / 3; ++i) {
    float o[3];
    display(&xyz[i * 3], e, o);
    auto u8 = [](float v) { return negicc::quantize_u8(negicc::srgb_encode(v)) / 255.0f; };
    const negicc::Lab l = negicc::srgb_to_lab({u8(o[0]), u8(o[1]), u8(o[2])});
    lab[i * 3] = l.L;
    lab[i * 3 + 1] = l.a;
    lab[i * 3 + 2] = l.b;
  }
  return lab;
}

double median_L(const Cells& c, const WhiteFilmParams& q) {
  return quantile(channel(xyz_lab(cell_xyz(c, q), q.e), 0), 0.5);
}

// The exposure at which the median L* of the cells is target at q's p and w (bisection in stops
// around the exposure that puts their median Y at 0.18).
double meter(const Cells& c, const WhiteFilmParams& q, double target) {
  const std::vector<double> xyz = cell_xyz(c, q);
  std::vector<float> y(xyz.size() / 3);
  for (size_t i = 0; i < y.size(); ++i) y[i] = (float)xyz[i * 3 + 1];
  const double e0 = 0.18 / std::max(quantile(y, 0.5), 1e-9);
  double lo = -12.0, hi = 12.0;
  for (int i = 0; i < 28; ++i) {
    const double mid = 0.5 * (lo + hi);
    (quantile(channel(xyz_lab(xyz, e0 * std::exp2(mid)), 0), 0.5) < target ? lo : hi) = mid;
  }
  return e0 * std::exp2(0.5 * (lo + hi));
}

// An intent field (mu, sigma) on the picture cells of its grid.
struct Field {
  Cells grid;
  std::vector<float> mu, sigma;  // 3 a cell

  // The mean squared z-score of the cells' Lab at q over channels c0 to c1 - 1.
  double z2(const WhiteFilmParams& q, int c0, int c1) const {
    const std::vector<float> lab = xyz_lab(cell_xyz(grid, q), q.e);
    double s = 0.0;
    for (size_t i = 0; i < lab.size(); i += 3)
      for (int c = c0; c < c1; ++c) s += std::pow((lab[i + c] - mu[i + c]) / sigma[i + c], 2);
    return s / (double)(lab.size() / 3 * (c1 - c0));
  }
};

// The intent field the model predicts from the 8-bit render of img at q.
bool predict(DinoV3Engine& engine, const Image& img, const double base[3], const WhiteFilmParams& q, Field& f) {
  std::vector<uint8_t> rgb((size_t)img.w * img.h * 3);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < (size_t)img.w * img.h; ++i) {
    float o[3];
    film_display(base, q, &img.px[i * 3], o);
    for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = negicc::quantize_u8(negicc::srgb_encode(o[c]));
  }
  DinoV3InferenceResult res;
  try {
    res = engine.infer(rgb.data(), img.w, img.h, img.w * 3);
  } catch (const std::exception& ex) {
    log_error("The intent model failed: %s\n", ex.what());
    return false;
  }
  f.grid = picture_cells(shrink(img, 0, 0, img.w, img.h, res.grid_w, res.grid_h), base);
  for (int i : f.grid.idx) {
    f.mu.insert(f.mu.end(), &res.mu[i * 3], &res.mu[i * 3] + 3);
    f.sigma.insert(f.sigma.end(), &res.sigma[i * 3], &res.sigma[i * 3] + 3);
  }
  return true;
}

// Coordinate descent on v[1] and v[2], each multiplied by 2^+-step, the step halved down to min_step; a trial is taken
// when it lowers loss by more than tol. Returns the stops moved in all.
template <class Loss>
double descend(Vec3& v, double step, double min_step, double tol, Loss loss) {
  double best = loss(v), moved = 0.0;
  while (step > min_step) {
    bool took = false;
    for (int i = 1; i <= 2; ++i)
      for (double s : {step, -step}) {
        Vec3 t = v;
        t[i] *= std::exp2(s);
        const double l = loss(t);
        if (l < best - tol) {
          v = t;
          best = l;
          took = true;
          moved += step;
        }
      }
    if (!took) step /= 2;
  }
  return moved;
}

// Step 6, the exposure and balance search, up to iters rounds, from the exposure that puts the median L* of the picture
// cells at kStartL: (a) the balance wG, wB by coordinate descent on the a*b* z-scores, the exposure re-metered after
// every trial to hold the median L*; (b) the exposure on the L* z-scores in 1/16 stop steps, within +-2 stops in the
// first round of the first search and +-0.5 after. Each round predicts the field from its render. Stops when the
// balance moved less than 0.03 stop and the exposure less than 0.07.
bool search(DinoV3Engine& engine, const Image& img, const double base[3], const Cells& small, int iters,
            WhiteFilmParams& q, std::vector<WhiteFilmRound>& log) {
  q.e = meter(small, q, kStartL);
  for (int it = 0; it < iters; ++it) {
    Field f;
    if (!predict(engine, img, base, q, f)) return false;
    const double target_L = median_L(small, q);
    auto trial = [&](const Vec3& w) {
      WhiteFilmParams t = q;
      t.w = w;
      t.e = meter(small, t, target_L);
      return t;
    };
    const double moved = descend(q.w, 0.25, 0.01, 1e-6, [&](const Vec3& w) { return f.z2(trial(w), 1, 3); });
    if (moved > 0.0) q = trial(q.w);  // the exposure the balance found was tried with
    const double loss_ab = f.z2(q, 1, 3);

    const double span = it == 0 && log.empty() ? 2.0 : 0.5, e = q.e;
    const int n = (int)(span * 16) + 1;
    double loss_L = HUGE_VAL, ds = 0.0;
    for (int k = 0; k < n; ++k) {
      const double s = -span + 2.0 * span * k / (n - 1);
      q.e = e * std::exp2(s);
      const double l = f.z2(q, 0, 1);
      if (l < loss_L) {
        loss_L = l;
        ds = s;
      }
    }
    q.e = e * std::exp2(ds);

    // The least-squares slope of the rendered L* on the field's, which the contrast step reads.
    const std::vector<float> L = channel(xyz_lab(cell_xyz(f.grid, q), q.e), 0), M = channel(f.mu, 0);
    double mm = 0.0, ml = 0.0, sml = 0.0, smm = 0.0;
    for (size_t i = 0; i < L.size(); ++i) {
      mm += M[i] / L.size();
      ml += L[i] / L.size();
    }
    for (size_t i = 0; i < L.size(); ++i) {
      sml += (M[i] - mm) * (L[i] - ml);
      smm += (M[i] - mm) * (M[i] - mm);
    }
    log.push_back({loss_ab, loss_L, sml / smm});
    log_info("White film search: round %zu: wG %.4f wB %.4f e %.6g (step %+.4f), loss ab %.4f L %.4f, slope %.4f\n",
             log.size(), q.w[1], q.w[2], q.e, ds, loss_ab, loss_L, sml / smm);
    if (moved < 0.03 && std::fabs(ds) < 0.07) break;
  }
  return true;
}

// Step 5, exponent ratios: the factors on pG and pB from the frame's density ranges (2nd to 98th percentile of the
// stops below the base per channel over the picture cells, all channels above 2, of the frame's centre 92%): R's range
// over G's and over B's, against the calibration's, clipped to 0.7-1.4 and shrunk toward 1. A setup whose green or blue
// density scale differs from the calibration's shows here; one frame alone is noisy, hence the shrinkage. Fewer than
// 500 cells: 1.
std::array<double, 2> ratio_factors(const Image& img, const double base[3]) {
  const int x0 = (int)(0.04 * img.w), x1 = (int)(0.96 * img.w), y0 = (int)(0.04 * img.h), y1 = (int)(0.96 * img.h);
  const Image cells = shrink(img, x0, y0, x1 - x0, y1 - y0, kCells, kCells);
  std::vector<float> d3;
  for (int i = 0; i < kCells * kCells; ++i) {
    const float* v = &cells.px[(size_t)i * 3];
    float d[3];
    stops(base, v, d);
    if (std::min({v[0], v[1], v[2]}) > 2.0f && std::max({d[0], d[1], d[2]}) > kPictureStops)
      d3.insert(d3.end(), d, d + 3);
  }
  if (d3.size() < 500 * 3) return {1.0, 1.0};
  double r[3];
  for (int c = 0; c < 3; ++c) r[c] = quantile(channel(d3, c), 0.98) - quantile(channel(d3, c), 0.02);
  return {std::pow(std::clamp(r[0] / r[1] / (kPCal[1] / kPCal[0]), 0.7, 1.4), kShrink[0]),
          std::pow(std::clamp(r[0] / r[2] / (kPCal[2] / kPCal[0]), 0.7, 1.4), kShrink[1])};
}

// The exponents p * f, each channel's balance moved with its exponent so that its value at the stops dm is unchanged.
WhiteFilmParams pivoted(const WhiteFilmParams& q, const Vec3& f, const Vec3& dm) {
  WhiteFilmParams r = q;
  for (int c = 0; c < 3; ++c) {
    r.p[c] = q.p[c] * f[c];
    r.w[c] = q.w[c] * (std::exp2(q.p[c] * dm[c]) - 1.0) / (std::exp2(r.p[c] * dm[c]) - 1.0);
  }
  return r;
}

// Step 8, slope: the factors f on the G and B exponents fitted to the intent field, pivoted at each channel's median
// stops over the picture cells so that the fit judges slope alone. Three rounds of: the pivoted render at the exposure
// that holds the median L* of q, its field, and coordinate descent over fG, fB (0.08 stop down to 0.008, each within
// 0.6-1.7) on the a*b* z-scores. Then f is clipped to 0.75-1.33 and taken half-way (one frame is weak evidence) and q
// pivoted by it.
bool slope_step(DinoV3Engine& engine, const Image& img, const double base[3], const Cells& small, WhiteFilmParams& q,
                Vec3& f) {
  const Vec3 dm = {quantile(channel(small.d, 0), 0.5), quantile(channel(small.d, 1), 0.5),
                   quantile(channel(small.d, 2), 0.5)};
  const double target_L = median_L(small, q);
  f = {1.0, 1.0, 1.0};
  for (int round = 0; round < 3; ++round) {
    WhiteFilmParams start = pivoted(q, f, dm);
    start.e = meter(small, start, target_L);
    Field fld;
    if (!predict(engine, img, base, start, fld)) return false;
    descend(f, 0.08, 0.008, 1e-5, [&](const Vec3& t) {
      if (!(t[1] > 0.6 && t[1] < 1.7 && t[2] > 0.6 && t[2] < 1.7)) return HUGE_VAL;
      WhiteFilmParams r = pivoted(q, t, dm);
      r.e = start.e;
      return fld.z2(r, 1, 3);
    });
  }
  for (int c = 1; c < 3; ++c) f[c] = std::sqrt(std::clamp(f[c], 0.75, 1.33));
  q = pivoted(q, f, dm);  // w_R stays 1: f_R is 1
  return true;
}

// Step 2, film base: the median of the base patch (fb's box in the half-size image); without one, the 99.9th
// percentile of the frame.
void film_base(const negicc::RgbView& half, const negicc::RgbView& frame, const negicc::FilmBaseDetection& fb,
               double base[3]) {
  const bool patch = fb.found && fb.has_box;
  const negicc::RgbView& v = patch ? half : frame;
  const int x0 = patch ? fb.box[0] : 0, y0 = patch ? fb.box[1] : 0;
  const int w = patch ? fb.box[2] : frame.width, h = patch ? fb.box[3] : frame.height;
  for (int c = 0; c < 3; ++c) {
    std::vector<float> values((size_t)w * h);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        uint16_t px[3];
        v.sample(x0 + x, y0 + y, px);
        values[(size_t)y * w + x] = px[c];
      }
    base[c] = quantile(std::move(values), patch ? 0.5 : 0.999);
  }
}

}  // namespace

bool solve_white_film(const negicc::RgbView& half, const negicc::RgbView& frame, const negicc::FilmBaseDetection& fb,
                      DinoV3Engine& engine, const std::vector<double>& params, const std::vector<double>& slopes,
                      WhiteFilm& wf) {
  const double t0 = omp_get_wtime();
  film_base(half, frame, fb, wf.base);
  log_info("White film: base %.1f %.1f %.1f (%s)\n", wf.base[0], wf.base[1], wf.base[2],
           fb.found && fb.has_box ? "base patch" : "99.9th percentile of the frame");
  WhiteFilmParams& q = wf.params;
  if (params.size() == 6) {
    q = {{params[0], params[1], params[2]}, {1.0, params[3], params[4]}, params[5]};
    return true;
  }
  // The analysis image: the frame area-averaged to kPreviewSide on its long side; its kCells x kCells picture cells.
  const double s = std::min(1.0, (double)kPreviewSide / std::max(frame.width, frame.height));
  const Image img = area_mean(frame.width, frame.height, (int)std::lround(frame.width * s),
                              (int)std::lround(frame.height * s), [&](int x, int y, float v[3]) {
                                uint16_t px[3];
                                frame.sample(x, y, px);
                                std::copy_n(px, 3, v);
                              });
  const Cells small = picture_cells(shrink(img, 0, 0, img.w, img.h, kCells, kCells), wf.base);

  const std::array<double, 2> f =
      slopes.empty() ? ratio_factors(img, wf.base) : std::array<double, 2>{slopes[0], slopes[1]};
  const Vec3 p0 = {kPCal[0], kPCal[1] * f[0], kPCal[2] * f[1]};
  q.p = p0;  // step 4, starting values (with step 5's ratios)
  q.w = {1.0, 1.0, 1.0};
  if (!search(engine, img, wf.base, small, 4, q, wf.log)) return false;
  // Step 7, contrast: the exponents scaled half-way toward what brings the slope of rendered L* on the field's to
  // kSlopeTarget, then the search again from the balance found.
  wf.contrast_scale = std::clamp(1.0 + 0.5 * (kSlopeTarget / wf.log.back().slope - 1.0), 0.75, 1.2);
  for (int c = 0; c < 3; ++c) q.p[c] = p0[c] * wf.contrast_scale;
  if (!search(engine, img, wf.base, small, 2, q, wf.log)) return false;
  Vec3 sf;
  if (!slope_step(engine, img, wf.base, small, q, sf)) return false;
  if (!search(engine, img, wf.base, small, 2, q, wf.log)) return false;
  q.e *= std::exp2(-kDarken);  // step 9, lightness offset

  wf.ratio_factors = f;
  wf.slope_factors = {sf[1], sf[2]};
  wf.time_ms = (omp_get_wtime() - t0) * 1000.0;
  log_info("White film: p %.4f %.4f %.4f, w 1 %.4f %.4f, e %.6g (ratio factors %.4f %.4f, contrast %.4f, slope "
           "factors %.4f %.4f)\n", q.p[0], q.p[1], q.p[2], q.w[1], q.w[2], q.e, f[0], f[1], wf.contrast_scale, sf[1],
           sf[2]);
  return true;
}

void render_white_film(const WhiteFilm& wf, uint16_t* px, size_t n, int channels) {
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    uint16_t* p = px + i * channels;
    const float v[3] = {(float)p[0], (float)p[1], (float)p[2]};
    float o[3];
    film_display(wf.base, wf.params, v, o);
    for (int c = 0; c < 3; ++c) p[c] = negicc::quantize_u16(negicc::srgb_encode(o[c]));
  }
}

nlohmann::ordered_json white_film_json(const WhiteFilm& wf) {
  using J = nlohmann::ordered_json;
  const bool searched = !wf.log.empty();
  J rounds = J::array();
  for (const WhiteFilmRound& r : wf.log) rounds.push_back({{"ab", r.ab}, {"L", r.L}, {"slope", r.slope}});
  return {{"p", wf.params.p}, {"w", wf.params.w}, {"e", wf.params.e},
          {"ratio_factors", searched ? J(wf.ratio_factors) : J(nullptr)},
          {"contrast_scale", searched ? J(wf.contrast_scale) : J(nullptr)},
          {"slope_factors", searched ? J(wf.slope_factors) : J(nullptr)},
          {"base_rgb", {wf.base[0], wf.base[1], wf.base[2]}}, {"search", rounds}, {"time_ms", wf.time_ms}};
}

std::vector<std::string> white_film_cli(const WhiteFilm& wf) {
  const WhiteFilmParams& q = wf.params;
  std::string v;
  for (double x : {q.p[0], q.p[1], q.p[2], q.w[1], q.w[2], q.e}) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%s%.17g", v.empty() ? "" : ",", x);
    v += buf;
  }
  return {"--white_model", "film", "--white_params", v};
}
