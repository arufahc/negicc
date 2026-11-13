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


#include "np_film_base.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "film_base_detector.h"
#include "libraw/libraw.h"
#include "np_input.h"
#include "np_log.h"
#include "profile_bundle.h"
#include "rgb_view.h"

namespace {

double median_of(std::vector<float>& v) {
  if (v.empty()) return 0.0;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) {
    const float lo = *std::max_element(v.begin(), v.begin() + mid);
    m = 0.5 * (m + lo);
  }
  return m;
}

// A region of the half-size view: each channel's median, the spread of the green density and the clipped share.
struct RegionStats {
  double med[3] = {0.0, 0.0, 0.0};
  double spread = 0.0;
  double clipped = 0.0;
  size_t n = 0;
};

// view: the capture with the flat field, which the levels are read from. raw: the same mosaic without it, which the
// clipping is counted on (a photosite clips at the white level as recorded, whatever the flat field then makes of it).
RegionStats region_stats(const negicc::RgbView& view, const negicc::RgbView& raw, int x0, int y0, int w, int h,
                         float clip) {
  RegionStats st;
  // At most about 4096 samples a region, on a regular lattice.
  const int step = std::max(1, (int)std::ceil(std::sqrt((double)w * h / 4096.0)));
  std::vector<float> ch[3];
  size_t clipped = 0;
  for (int y = y0; y < y0 + h; y += step) {
    for (int x = x0; x < x0 + w; x += step) {
      uint16_t px[3], rpx[3];
      view.sample(x, y, px);
      raw.sample(x, y, rpx);
      if (rpx[0] >= clip || rpx[1] >= clip || rpx[2] >= clip) ++clipped;
      for (int c = 0; c < 3; ++c) ch[c].push_back((float)px[c]);
    }
  }
  st.n = ch[0].size();
  if (!st.n) return st;
  st.clipped = (double)clipped / (double)st.n;
  std::vector<float> green = ch[1];
  for (int c = 0; c < 3; ++c) st.med[c] = median_of(ch[c]);
  if (st.med[1] > 0.0) {
    std::vector<float> dev(green.size());
    for (size_t i = 0; i < green.size(); ++i) {
      const double g = std::max(1.0, (double)green[i]);
      dev[i] = (float)std::fabs(std::log10(g / st.med[1]));
    }
    st.spread = 1.4826 * median_of(dev);
  }
  return st;
}

// Whether a tile's colour is the film base's: its medians through the crosstalk matrix, G/R and B/R against the
// expected base's (np_film_base.h step 3).
bool base_coloured(const double med[3], const double* crosstalk, const double* expected) {
  if (!expected || !(expected[0] > 0.0) || !(expected[1] > 0.0) || !(expected[2] > 0.0)) return true;
  double v[3];
  for (int r = 0; r < 3; ++r) {
    v[r] = 0.0;
    for (int c = 0; c < 3; ++c) v[r] += (crosstalk ? crosstalk[r * 3 + c] : (r == c ? 1.0 : 0.0)) * med[c];
  }
  if (!(v[0] > 0.0) || !(v[1] > 0.0) || !(v[2] > 0.0)) return false;
  const double dg = std::log10((v[1] / v[0]) / (expected[1] / expected[0]));
  const double db = std::log10((v[2] / v[0]) / (expected[2] / expected[0]));
  return std::fabs(dg) <= kBaseColourTol && std::fabs(db) <= kBaseColourTol;
}

// The tiles of the window (x0, y0, w, h) of the half-size view: true with the base and the kept tiles' record when a
// group of at least kBaseMinTiles agrees.
bool base_from_tiles(const negicc::RgbView& view, const negicc::RgbView& raw, int x0, int y0, int w, int h, float clip,
                     const double* crosstalk, const double* expected, FilmBaseCapture& out) {
  struct Tile { int x, y, w, h; int i, j; RegionStats st; };
  std::vector<Tile> tiles;
  // Square tiles, kBaseTileGrid along the longer side and never under 8 pixels, so that a rebate strip beside a
  // picture, far narrower than the frame, holds whole tiles.
  const int side = std::max(8, std::max(w, h) / kBaseTileGrid);
  const int gx = std::max(1, w / side), gy = std::max(1, h / side);
  for (int j = 0; j < gy; ++j) {
    for (int i = 0; i < gx; ++i) {
      Tile t;
      t.x = x0 + (int)((long long)w * i / gx);
      t.y = y0 + (int)((long long)h * j / gy);
      t.w = x0 + (int)((long long)w * (i + 1) / gx) - t.x;
      t.h = y0 + (int)((long long)h * (j + 1) / gy) - t.y;
      t.i = i;
      t.j = j;
      if (t.w < 4 || t.h < 4) continue;
      tiles.push_back(t);
    }
  }
  #pragma omp parallel for schedule(dynamic, 16)
  for (size_t k = 0; k < tiles.size(); ++k) {
    tiles[k].st = region_stats(view, raw, tiles[k].x, tiles[k].y, tiles[k].w, tiles[k].h, clip);
  }
  // The tiles that must agree: kBaseMinTiles, and at least one tile in kBaseMinShare of them.
  const int min_tiles = std::max(kBaseMinTiles, (int)(tiles.size() / kBaseMinShare));
  out.tiles_total = (int)tiles.size();
  out.tiles_kept = 0;
  // Unclipped tiles with light in every channel; then the smooth ones among them, by the median tile's spread.
  std::vector<const Tile*> lit;
  std::vector<float> spreads;
  for (const Tile& t : tiles) {
    if (t.st.n && t.st.clipped <= kBaseClipFraction && t.st.med[0] > 0.0 && t.st.med[1] > 0.0 && t.st.med[2] > 0.0) {
      lit.push_back(&t);
      spreads.push_back((float)t.st.spread);
    }
  }
  if ((int)lit.size() < min_tiles) return false;
  const double spread_limit = std::max(kBaseTileMaxSpread, kBaseSpreadFactor * median_of(spreads));
  std::vector<const Tile*> smooth;
  for (const Tile* t : lit) {
    if (t->st.spread <= spread_limit && base_coloured(t->st.med, crosstalk, expected)) smooth.push_back(t);
  }
  if ((int)smooth.size() < min_tiles) return false;
  // The brightest group: from the brightest tile down, the first with kBaseMinTiles tiles within the margin of it.
  std::sort(smooth.begin(), smooth.end(), [](const Tile* a, const Tile* b) { return a->st.med[1] > b->st.med[1]; });
  std::vector<const Tile*> kept;
  for (size_t i = 0; i < smooth.size() && kept.empty(); ++i) {
    const double ref = smooth[i]->st.med[1];
    std::vector<const Tile*> group;
    for (const Tile* t : smooth) {
      if (std::fabs(std::log10(t->st.med[1] / ref)) <= kBaseTileMargin) group.push_back(t);
    }
    if ((int)group.size() >= min_tiles) kept = group;
  }
  if (kept.empty()) return false;
  // The group was found from its brightest tile; it is taken again around its own median, so that the base is the
  // level of that film and not of its brighter half.
  {
    std::vector<float> greens;
    for (const Tile* t : kept) greens.push_back((float)t->st.med[1]);
    const double ref = median_of(greens);
    std::vector<const Tile*> group;
    for (const Tile* t : smooth) {
      if (std::fabs(std::log10(t->st.med[1] / ref)) <= kBaseTileMargin) group.push_back(t);
    }
    if ((int)group.size() >= min_tiles) kept = group;
  }
  // Film at the base's level can lie in several places (the rebate on either side of a picture, an unexposed shadow
  // in it); the largest piece of touching tiles is one stretch of film under one part of the light, and is taken
  // alone when it is large enough.
  {
    std::vector<int> label((size_t)gx * gy, -1);
    for (const Tile* t : kept) label[(size_t)t->j * gx + t->i] = 0;
    int pieces = 0, best = 0, best_n = 0;
    std::vector<int> stack;
    for (const Tile* t : kept) {
      const int start = t->j * gx + t->i;
      if (label[start] != 0) continue;
      const int id = ++pieces;
      int n = 0;
      label[start] = id;
      stack.push_back(start);
      while (!stack.empty()) {
        const int at = stack.back();
        stack.pop_back();
        ++n;
        const int ai = at % gx, aj = at / gx;
        const int next[4][2] = {{ai - 1, aj}, {ai + 1, aj}, {ai, aj - 1}, {ai, aj + 1}};
        for (const auto& nb : next) {
          if (nb[0] < 0 || nb[0] >= gx || nb[1] < 0 || nb[1] >= gy) continue;
          const int q = nb[1] * gx + nb[0];
          if (label[q] == 0) {
            label[q] = id;
            stack.push_back(q);
          }
        }
      }
      if (n > best_n) {
        best_n = n;
        best = id;
      }
    }
    if (pieces > 1 && best_n >= min_tiles) {
      std::vector<const Tile*> piece;
      for (const Tile* t : kept) {
        if (label[(size_t)t->j * gx + t->i] == best) piece.push_back(t);
      }
      kept = piece;
    }
  }
  std::vector<float> ch[3];
  int bx0 = kept[0]->x, by0 = kept[0]->y, bx1 = bx0, by1 = by0;
  double lo = 1e30, hi = -1e30;
  for (const Tile* t : kept) {
    for (int c = 0; c < 3; ++c) ch[c].push_back((float)t->st.med[c]);
    bx0 = std::min(bx0, t->x);
    by0 = std::min(by0, t->y);
    bx1 = std::max(bx1, t->x + t->w);
    by1 = std::max(by1, t->y + t->h);
    const double d = -std::log10(t->st.med[1]);
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  for (int c = 0; c < 3; ++c) out.decode_rgb[c] = median_of(ch[c]);
  out.tiles_kept = (int)kept.size();
  out.spread_density = hi - lo;
  out.roi[0] = bx0 * 2;
  out.roi[1] = by0 * 2;
  out.roi[2] = (bx1 - bx0) * 2;
  out.roi[3] = (by1 - by0) * 2;
  return true;
}

}  // namespace

bool read_film_base_capture(const std::string& path, const int* roi, const negicc::FlatFieldPtr& flat,
                            const double* crosstalk, const double* expected_base, double scan_shutter_s,
                            double scan_iso, FilmBaseCapture& out, std::string& err) {
  out = FilmBaseCapture{};
  if (!is_regular_file(path)) {
    err = "no such file: " + path;
    return false;
  }
  std::unique_ptr<LibRaw> raw(new LibRaw());
  const int ret = open_and_unpack_raw_file(raw.get(), path);
  if (ret != LIBRAW_SUCCESS) {
    err = "cannot open or unpack " + path + ": " + libraw_strerror(ret);
    return false;
  }
  if (!is_sony_cfa_geometry(raw.get()) || !raw->imgdata.rawdata.raw_image) {
    err = path + " is not a Sony CFA RAW (the film base capture is read as the scan's mosaic is)";
    return false;
  }
  float white_level = 65535.0f;
  if (!subtract_cfa_black_in_place(raw.get(), white_level)) {
    err = "invalid CFA mosaic geometry in " + path;
    return false;
  }
  const int src_w = raw->imgdata.sizes.width, src_h = raw->imgdata.sizes.height;
  const size_t pitch = raw->imgdata.sizes.raw_pitch / 2;
  const unsigned filters = raw->imgdata.idata.filters;
  const uint16_t* origin = raw->imgdata.rawdata.raw_image + (size_t)raw->imgdata.sizes.top_margin * pitch +
                           raw->imgdata.sizes.left_margin;
  const int half_w = src_w / 2, half_h = src_h / 2;
  std::shared_ptr<const negicc::FlatTables> flat_tables;
  negicc::FlatSampleArgs flat_args{};
  if (flat) {
    try {
      flat_tables = flat->tables(half_w, half_h, true);
      flat_args = flat_tables->host();
    } catch (const std::exception& e) {
      err = std::string("the flat field does not fit ") + path + ": " + e.what();
      return false;
    }
  }
  const negicc::RgbView view =
      negicc::RgbView::from_cfa_half(origin, pitch, src_w, src_h, filters, nullptr, flat_args, flat_tables);
  // The same mosaic without the flat field, for the clipping (region_stats).
  const negicc::RgbView raw_view = negicc::RgbView::from_cfa_half(origin, pitch, src_w, src_h, filters, nullptr);
  const float clip = 0.98f * white_level;

  if (roi) {
    if (roi[0] < 0 || roi[1] < 0 || roi[2] <= 0 || roi[3] <= 0 || (long long)roi[0] + roi[2] > src_w ||
        (long long)roi[1] + roi[3] > src_h) {
      err = "the film base region " + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," +
            std::to_string(roi[2]) + "," + std::to_string(roi[3]) + " is off the capture's sensor (" +
            std::to_string(src_w) + "x" + std::to_string(src_h) + ")";
      return false;
    }
    const int hx = roi[0] / 2, hy = roi[1] / 2;
    const int hw = std::max(1, std::min(half_w - hx, (roi[0] + roi[2] + 1) / 2 - hx));
    const int hh = std::max(1, std::min(half_h - hy, (roi[1] + roi[3] + 1) / 2 - hy));
    const RegionStats st = region_stats(view, raw_view, hx, hy, hw, hh, clip);
    if (!st.n || st.med[0] <= 0.0 || st.med[1] <= 0.0 || st.med[2] <= 0.0) {
      err = "the film base region of " + path + " holds no light";
      return false;
    }
    if (st.clipped > kBaseClipFraction) {
      err = "the film base region of " + path + " is clipped (" + std::to_string(100.0 * st.clipped) +
            "% of its pixels): capture the film base at a shorter exposure";
      return false;
    }
    for (int c = 0; c < 3; ++c) out.decode_rgb[c] = st.med[c];
    for (int k = 0; k < 4; ++k) out.roi[k] = roi[k];
    out.roi_source = "given";
    out.spread_density = st.spread;
    if (st.spread > kBaseTileMaxSpread) {
      log_warn("the film base region of %s is not smooth (green density spread %.4f)\n", path.c_str(), st.spread);
    }
  } else if (base_from_tiles(view, raw_view, 0, 0, half_w, half_h, clip, crosstalk, expected_base, out)) {
    out.roi_source = "tiles";
  } else {
    // The thinnest film on the capture, as a TriLED scan's base is read when it shows no rebate; on the mosaic as
    // recorded it must not be clipped.
    float thin[3] = {0.0f, 0.0f, 0.0f}, thin_raw[3] = {0.0f, 0.0f, 0.0f};
    negicc::thinnest_film_base(view, thin);
    negicc::thinnest_film_base(raw_view, thin_raw);
    if (!(thin[0] > 1.0f) || !(thin[1] > 1.0f) || !(thin[2] > 1.0f)) {
      err = "no usable film base on " + path + " (no smooth film)";
      return false;
    }
    if (thin_raw[0] >= clip || thin_raw[1] >= clip || thin_raw[2] >= clip) {
      err = "the film base capture " + path + " is clipped: capture the film base at a shorter exposure";
      return false;
    }
    log_warn("the film base capture %s has no smooth region; its thinnest film is taken\n", path.c_str());
    for (int c = 0; c < 3; ++c) out.decode_rgb[c] = thin[c];
    out.roi[0] = out.roi[1] = 0;
    out.roi[2] = src_w;
    out.roi[3] = src_h;
    out.roi_source = "statistics";
    out.tiles_kept = 0;
  }

  // The capture's exposure: EXIF, else LibRaw's.
  // As the scan's (Scan::scan_exposure): a capture that records none is taken at 1/8 s and ISO 100, with a warning.
  double shutter = 0.0, iso = 0.0;
  const bool has_exif = read_exif_exposure_d(path, shutter, iso);
  if (!has_exif || !(shutter > 0.0)) shutter = raw->imgdata.other.shutter > 0 ? (double)raw->imgdata.other.shutter : 0.0;
  if (!has_exif || !(iso > 0.0)) {
    iso = raw->imgdata.other.iso_speed > 0 ? std::round((double)raw->imgdata.other.iso_speed) : 0.0;
  }
  if (!(shutter > 0.0) || !(iso > 0.0)) {
    if (!(shutter > 0.0)) shutter = 0.125;
    if (!(iso > 0.0)) iso = 100.0;
    log_warn("the film base capture %s records no exposure; taking %.4g s at ISO %.0f to scale it to the scan's\n",
             path.c_str(), shutter, iso);
  }
  out.shutter_s = shutter;
  out.iso = iso;
  out.exposure_scale = (scan_shutter_s * scan_iso) / (shutter * iso);
  for (int r = 0; r < 3; ++r) {
    double v = 0.0;
    for (int c = 0; c < 3; ++c) v += (crosstalk ? crosstalk[r * 3 + c] : (r == c ? 1.0 : 0.0)) * out.decode_rgb[c];
    out.rgb[r] = v * out.exposure_scale;
  }
  if (!(out.rgb[0] > 0.0) || !(out.rgb[1] > 0.0) || !(out.rgb[2] > 0.0)) {
    err = "the film base of " + path + " is not positive after the crosstalk correction";
    return false;
  }
  return true;
}
