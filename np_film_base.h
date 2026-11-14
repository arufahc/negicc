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
//   1. the whole frame is split into square tiles, kBaseTileGrid along its longer side, small enough that a strip of
//      rebate beside a picture holds whole tiles; each tile's median per channel and the spread of its green density
//      (1.4826 x the median absolute deviation) are taken;
//   2. tiles with clipped photosites are dropped (clipping is counted on the mosaic as recorded, before the flat
//      field, which moves the levels), and tiles whose spread stands out: above kBaseTileMaxSpread and above
//      kBaseSpreadFactor times the median tile's (dust, scratches, edges; a grainy film is uneven in every tile alike,
//      which the median allows for);
//   3. tiles whose colour is not the film base's are dropped: through the crosstalk matrix, G/R and B/R must lie
//      within kBaseColourTol (density) of the bundle's film base. Bare light beside the film is brighter than the
//      base and not orange;
//   4. of the rest, the brightest group is kept: going down from the brightest tile, the first whose green has enough
//      tiles within kBaseTileMargin (density) of it (kBaseMinTiles, and one tile in kBaseMinShare). The base is the
//      thinnest film, so darker tiles are picture or shadow, whatever share of the frame they cover. The group is
//      then taken again around its own median;
//   5. of that group, the largest piece of touching tiles is kept when it is large enough: one stretch of film (a
//      rebate strip) and not film of the same level elsewhere in the frame;
//   6. the base is the median, per channel, of the kept tiles' medians.
// With no such group the thinnest film on the capture is taken (film_base_detector.h's statistics); with nothing
// usable, or a clipped result, the capture is refused.
//
// The thresholds of steps 2 to 4 were set on simulated captures. The tile size and steps 4 and 5 were set on seven
// captures of Portra 400 frames with a rebate strip at one edge (2026-10-04), where the search agrees with the
// station's hand-marked base to 0.005 density on five; on two it reads the strip on the other side of the picture,
// which a region given by hand reads 0.01 to 0.03 density brighter than the marked one.

#pragma once

#include <string>

#include "flat_field.h"

constexpr int kBaseTileGrid = 96;             // square tiles along the longer side (fewer when they would be under 8 pixels)
constexpr int kBaseMinTiles = 8;              // tiles that must agree, and at least ...
constexpr int kBaseMinShare = 200;            // ... one tile in this many
constexpr double kBaseTileMaxSpread = 0.03;   // density: a tile's green spread up to which it is smooth whatever the rest
constexpr double kBaseSpreadFactor = 1.5;     // ... and up to this many times the median tile's spread
constexpr double kBaseTileMargin = 0.02;      // density: how far from the group's brightest tile a kept tile may lie
constexpr double kBaseColourTol = 0.25;       // density: how far a tile's G/R and B/R may lie from the bundle's base
constexpr double kBaseClipFraction = 0.001;   // a tile (or a given region) with more clipped pixels is clipped

struct FilmBaseCapture {
  double rgb[3] = {0.0, 0.0, 0.0};         // the film base: through the crosstalk matrix, at the scan's exposure
  double decode_rgb[3] = {0.0, 0.0, 0.0};  // as the capture's half-size decode gives it, at the capture's exposure
  int roi[4] = {0, 0, 0, 0};               // the region read (x, y, w, h in sensor pixels): the given one, or the kept
                                           // tiles' bounding box
  std::string roi_source;                  // "given", "tiles" or "statistics"
  int tiles_kept = 0, tiles_total = 0;     // with the tiles
  double spread_density = 0.0;             // the spread of the kept tiles' green medians (a given region: of its pixels)
  double shutter_s = 0.0, iso = 0.0;       // the capture's exposure
  double exposure_scale = 1.0;             // scan exposure / capture exposure, which rgb carries
};

// Reads the film base from the capture at path. roi: the region in sensor pixels, or null to find it. flat: the scan's
// flat field (null for none). crosstalk: the 3x3 row-major crosstalk matrix (null for none). expected_base: the
// bundle's film base through the crosstalk matrix, whose colour the tiles are held to (null: no colour test).
// scan_shutter_s, scan_iso: the scan's exposure. False with err on a file that cannot be read, a capture that is not a
// Sony CFA RAW, a region off the sensor or clipped, or a capture with no usable film.
bool read_film_base_capture(const std::string& path, const int* roi, const negicc::FlatFieldPtr& flat,
                            const double* crosstalk, const double* expected_base, double scan_shutter_s,
                            double scan_iso, FilmBaseCapture& out, std::string& err);
