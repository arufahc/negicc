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

// neg_process's RAW decoding with LibRaw: the camera checks, the mosaic as the frame detector reads it, the black
// subtraction of the zero-copy CFA path, and load_raw (LibRaw's decode: camera RGB, Rec.2020 or the white-light
// decode of a d65 bundle; also a linear DNG).

#pragma once

#include <functional>
#include <string>

#include "film_frame_detector.h"
#include "flat_field.h"
#include "libraw/libraw.h"

// 4-shot pixel shift merge is only available with Sony A7RM4 camera.
bool is_sony_a7rm4(LibRaw* proc);
// Checks if the camera has Sony Bayer RGGB CFA geometry.
bool is_sony_cfa_geometry(LibRaw* proc);

// Called with a LibRaw whose imgdata.image holds the half-size image of its RAW (read_half_size): iwidth x iheight
// pixels, row-major, LibRaw's 4 channels each (R, G, B in the first three); a point (x, y) of the RAW is (x / 2, y / 2)
// there. False fails the load.
using HalfSizeAnalysis = std::function<bool(const LibRaw& proc)>;

// Called by load_raw with the unpacked RAW in proc (imgdata.rawdata.raw_image holds the CFA mosaic, margins
// included), before the half-size analysis and the decode. Sets roi to the region to decode (x, y, w, h in the
// visible sensor area, imgdata.sizes.width x height), as --roi would; false fails the load.
using RawRegionFinder = std::function<bool(LibRaw& proc, int roi[4])>;

// How load_raw decodes a RAW to RGB.
enum class RawDecode {
  kRec2020,     // LibRaw to linear Rec.2020 with unit channel multipliers (no profile bundle, no -r -g -b)
  kCameraRgb,   // sensor RGB, unit channel multipliers: a camera_raw_rgb (TriLED) bundle or -r -g -b
  kWhiteLight,  // a rec2020_linear (D65) bundle: prepare_white_light_mosaic, then decode_white_light
};

// The unpacked CFA mosaic of proc as the frame detector reads it, in place (no copy): rawdata.raw_image from its
// first visible photosite (top_margin, left_margin), raw_pitch / 2 elements per row; the R, G and B lattices take one
// photosite of their colour per 2x2 (Bayer) or 6x6 (X-Trans) cell, each with its black level (black + cblack[colour])
// subtracted; maximum - black is the clipping point. A monochrome RAW is its one plane three times. False when proc
// holds no mosaic (e.g. a linear DNG) or when its margins put the visible area outside the raw buffer.
//
// When LibRaw holds the black level as a cblack[6..] pattern (cblack[4] rows x cblack[5] columns, indexed by the
// visible row and column as prepare_white_light_mosaic applies it), each lattice's black also takes the pattern's mean
// over the pattern entries its photosites land on: the rows y0 + j * step and columns x0 + i * step, modulo the
// pattern size. When the pattern period divides the CFA period (the usual case) that is the single entry at the
// lattice's phase, exactly the black of every photosite of the lattice; otherwise it is the mean over the lattice.
bool film_image_from_raw(LibRaw& proc, FilmImage& img);

// Step 1 of the zero-copy CFA path: subtracts each channel's black in place on the unpacked CFA mosaic, and gives the
// white level above black.
bool subtract_cfa_black_in_place(LibRaw* proc, float& out_white_level);

// Load a RAW file and decode it into linear values.
//
// If |debayer| is true, interpolation is performed to generate missing pixels
// from the color filter array (usually a bayer pattern). Otherwise, the linear
// RGB values will have missing pixels and will not be scaled to 16-bit. If the
// sensor produces 14-bit files, then a scale factor of 4 needs to be applied,
// this is needed only when merging pixel-shift images.
//
// |qual| chooses the debayer algorithm used. If qual < 0 (default), auto
// selection is performed: grain-aware demosaicing is used when Sony CFA geometry
// is detected, otherwise falling back to bilinear (qual = 0). Explicit qual >= 0
// forces LibRaw's built-in demosaic algorithms (0 is bilinear).
//
// When |crop| is false, the entire RAW file is used, disregarding aspect ratio
// and cropbox specified in the RAW metadata.
//
// |decode| picks the colour path of a debayered decode (RawDecode); kWhiteLight decodes with LibRaw's bilinear
// demosaic (or qual when it is >= 0), never grain-aware.
//
// |analyse|, when set, is called with the half-size image of the RAW before it is decoded: LibRaw's half-size readout
// of the mosaic (read_half_size), or for kWhiteLight its half-size white-light decode, in the same space as the
// output.
//
// |find_region|, when set, is called on the unpacked mosaic first and gives the region to decode in place of roi
// (--roi auto: the detected frame), so that the analysis and the decode see only that region.
LibRaw* load_raw(const std::string& fn, bool debayer, int qual, bool crop,
                 RawDecode decode = RawDecode::kRec2020, const int* roi = nullptr,
                 const HalfSizeAnalysis& analyse = {}, const RawRegionFinder& find_region = {},
                 const negicc::FlatField* flat = nullptr,
                 LibRaw* unpacked_proc = nullptr);
