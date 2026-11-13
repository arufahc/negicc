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


// The film base read from a capture of unexposed film (--film_base <file>[:x,y,w,h]), in place of the base read from
// the scan itself (film_base_detector.h).
//
// The capture is decoded as the scan's half-size view is: a Sony CFA RAW, black subtracted, its 2x2 cells as RGB,
// with the flat field. The base is read over a region of it, taken through the crosstalk matrix and scaled to the
// scan's exposure, which gives it in the conversion's film base space, as a base read from the scan is.
//
// The region, when none is given, is found on the capture, which is seldom perfectly clean (dust, scratches, the
// holder's edge, a sprocket hole, part of a picture):
//   1. the central half of the frame (each way) is split into kBaseTileGrid x kBaseTileGrid tiles; each tile's median
//      per channel and the spread of its green density (1.4826 x the median absolute deviation) are taken;
//   2. tiles with clipped pixels are dropped (the base is the brightest film there is, so a clipped tile reads low),
//      and tiles whose spread is above kBaseTileMaxSpread (dust, scratches, edges);
//   3. of the rest, the tiles within kBaseTileMargin (density) of the 90th-percentile tile's green are kept: the base
//      is the thinnest film, so darker tiles are picture or shadow;
//   4. the base is the median, per channel, of the kept tiles' medians.
// With fewer than kBaseMinTiles kept the whole frame is tried the same way, then the thinnest film on the capture
// (film_base_detector.h's statistics); with nothing usable the capture is refused.

#pragma once

#include <string>

#include "flat_field.h"

constexpr int kBaseTileGrid = 16;             // tiles each way (fewer on a capture too small for 8-pixel tiles)
constexpr int kBaseMinTiles = 8;              // tiles that must agree
constexpr double kBaseTileMaxSpread = 0.03;   // density: a tile's green spread above which it is not smooth
constexpr double kBaseTileMargin = 0.02;      // density: how far from the 90th-percentile tile a kept tile may lie
constexpr double kBaseClipFraction = 0.001;   // a tile (or a given region) with more clipped pixels is clipped

struct FilmBaseCapture {
  double rgb[3] = {0.0, 0.0, 0.0};         // the film base: through the crosstalk matrix, at the scan's exposure
  double decode_rgb[3] = {0.0, 0.0, 0.0};  // as the capture's half-size decode gives it, at the capture's exposure
  int roi[4] = {0, 0, 0, 0};               // the region read (x, y, w, h in sensor pixels): the given one, or the kept
                                           // tiles' bounding box
  std::string roi_source;                  // "given", "tiles", "tiles_whole_frame" or "statistics"
  int tiles_kept = 0, tiles_total = 0;     // with the tiles
  double spread_density = 0.0;             // the spread of the kept tiles' green medians (a given region: of its pixels)
  double shutter_s = 0.0, iso = 0.0;       // the capture's exposure
  double exposure_scale = 1.0;             // scan exposure / capture exposure, which rgb carries
};

// Reads the film base from the capture at path. roi: the region in sensor pixels, or null to find it. flat: the scan's
// flat field (null for none). crosstalk: the 3x3 row-major crosstalk matrix (null for none). scan_shutter_s, scan_iso:
// the scan's exposure. False with err on a file that cannot be read, a capture that is not a Sony CFA RAW, a region
// off the sensor or clipped, or a capture with no usable film.
bool read_film_base_capture(const std::string& path, const int* roi, const negicc::FlatFieldPtr& flat,
                            const double* crosstalk, double scan_shutter_s, double scan_iso, FilmBaseCapture& out,
                            std::string& err);
