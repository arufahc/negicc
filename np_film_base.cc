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

RegionStats region_stats(const negicc::RgbView& view, int x0, int y0, int w, int h, float clip) {
  RegionStats st;
  // At most about 4096 samples a region, on a regular lattice.
  const int step = std::max(1, (int)std::ceil(std::sqrt((double)w * h / 4096.0)));
  std::vector<float> ch[3];
  size_t clipped = 0;
  for (int y = y0; y < y0 + h; y += step) {
    for (int x = x0; x < x0 + w; x += step) {
      uint16_t px[3];
      view.sample(x, y, px);
      if (px[0] >= clip || px[1] >= clip || px[2] >= clip) ++clipped;
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

// The tiles of the window (x0, y0, w, h) of the half-size view: true with the base and the kept tiles' record when at
// least kBaseMinTiles agree.
bool base_from_tiles(const negicc::RgbView& view, int x0, int y0, int w, int h, float clip, FilmBaseCapture& out) {
  struct Tile { int x, y, w, h; RegionStats st; };
  std::vector<Tile> tiles;
  // kBaseTileGrid tiles each way, fewer on a small capture so that a tile stays at least 8 pixels wide and high.
  const int gx = std::max(1, std::min(kBaseTileGrid, w / 8)), gy = std::max(1, std::min(kBaseTileGrid, h / 8));
  for (int j = 0; j < gy; ++j) {
    for (int i = 0; i < gx; ++i) {
      Tile t;
      t.x = x0 + (int)((long long)w * i / gx);
      t.y = y0 + (int)((long long)h * j / gy);
      t.w = x0 + (int)((long long)w * (i + 1) / gx) - t.x;
      t.h = y0 + (int)((long long)h * (j + 1) / gy) - t.y;
      if (t.w < 4 || t.h < 4) continue;
      t.st = region_stats(view, t.x, t.y, t.w, t.h, clip);
      tiles.push_back(t);
    }
  }
  out.tiles_total = (int)tiles.size();
  out.tiles_kept = 0;
  std::vector<const Tile*> smooth;
  for (const Tile& t : tiles) {
    if (t.st.n && t.st.clipped <= kBaseClipFraction && t.st.spread <= kBaseTileMaxSpread && t.st.med[1] > 0.0) {
      smooth.push_back(&t);
    }
  }
  if ((int)smooth.size() < kBaseMinTiles) return false;
  std::vector<float> greens;
  for (const Tile* t : smooth) greens.push_back((float)t->st.med[1]);
  std::sort(greens.begin(), greens.end());
  const double g90 = greens[std::min(greens.size() - 1, (size_t)std::floor(0.9 * (double)(greens.size() - 1) + 0.5))];
  std::vector<const Tile*> kept;
  for (const Tile* t : smooth) {
    if (std::fabs(std::log10(t->st.med[1] / g90)) <= kBaseTileMargin) kept.push_back(t);
  }
  if ((int)kept.size() < kBaseMinTiles) return false;
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
                            const double* crosstalk, double scan_shutter_s, double scan_iso, FilmBaseCapture& out,
                            std::string& err) {
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
  const float clip = 0.98f * white_level;

  if (roi) {
    if (roi[0] < 0 || roi[1] < 0 || roi[2] <= 0 || roi[3] <= 0 || roi[0] + roi[2] > src_w || roi[1] + roi[3] > src_h) {
      err = "the film base region " + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," +
            std::to_string(roi[2]) + "," + std::to_string(roi[3]) + " is off the capture's sensor (" +
            std::to_string(src_w) + "x" + std::to_string(src_h) + ")";
      return false;
    }
    const int hx = roi[0] / 2, hy = roi[1] / 2;
    const int hw = std::max(1, std::min(half_w - hx, (roi[0] + roi[2] + 1) / 2 - hx));
    const int hh = std::max(1, std::min(half_h - hy, (roi[1] + roi[3] + 1) / 2 - hy));
    const RegionStats st = region_stats(view, hx, hy, hw, hh, clip);
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
  } else if (base_from_tiles(view, half_w / 4, half_h / 4, half_w / 2, half_h / 2, clip, out)) {
    out.roi_source = "tiles";
  } else if (base_from_tiles(view, 0, 0, half_w, half_h, clip, out)) {
    out.roi_source = "tiles_whole_frame";
  } else {
    // The thinnest film on the capture, as a scan's base is read when it shows no rebate.
    negicc::FilmBaseDetection det;
    if (!negicc::detect_film_base(view, nullptr, 0, 0, nullptr, det) || !det.found) {
      err = "no usable film base on " + path + " (clipped, or no smooth film)";
      return false;
    }
    if (det.decode_rgb[0] >= clip || det.decode_rgb[1] >= clip || det.decode_rgb[2] >= clip) {
      err = "the film base capture " + path + " is clipped: capture the film base at a shorter exposure";
      return false;
    }
    log_warn("the film base capture %s has no smooth region; its thinnest film is taken\n", path.c_str());
    for (int c = 0; c < 3; ++c) out.decode_rgb[c] = det.decode_rgb[c];
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
