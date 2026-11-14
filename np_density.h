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


// The density output (-P statusm, rp180, apd): the negative's density above the film base in a standard measure, as
// code values in a 10-bit DPX or a 16-bit TIFF.
//
// A pixel, after the film base's merged matrix (crosstalk correction and film base scale; the film base lands on
// base_level), is a transmittance relative to bare film in each of the scanning light's channels:
//
//     T = v / base_level,   D_light = -log10(max(T, kDensityMinT)),   D = M * D_light,   code = round(offset + scale * D)
//
// M is the bundle's 3x3 matrix for the measure (profile_bundle.h). The two encodings are the Cineon one (10 bits:
// offset 95, 500 codes per unit density) and the 16-bit one (offset 1520, 8000 codes per unit density), so RP 180
// density in a DPX is a Cineon file. Academy Printing Density is written as ADX (SMPTE ST 2065-3): the same offsets
// and scales with the standard's channel factors on the density, code = offset + scale * k * D with k = 1.00, 0.92,
// 0.95 for R, G, B. Status M and RP 180 take no factors (k = 1).
//
// Both files carry what is needed to read and recompute them, as JSON (schema negicc.density/1): the measure, the
// encoding, the film base, the matrix applied and the light's record with its spectrum. A TIFF has it in its
// ImageDescription tag, a DPX in its user-defined data section.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "profile_bundle.h"

constexpr double kDensityMinT = 1e-4;   // the transmittance floor before the log (4.0 density)

struct DensityEncoding {
  int bits = 16;
  int offset = 1520;
  int scale = 8000;
  double k[3] = {1.0, 1.0, 1.0};      // the channel factors on the density (ADX: 1.00, 0.92, 0.95)
  int max_code() const { return (1 << bits) - 1; }
};
// The channel factors of ADX (SMPTE ST 2065-3:2020, equations 1 and 2).
constexpr double kAdxChannelFactor[3] = {1.00, 0.92, 0.95};
inline DensityEncoding density_encoding(bool dpx, DensityMeasure measure) {
  DensityEncoding e;
  if (dpx) {
    e.bits = 10;
    e.offset = 95;
    e.scale = 500;
  }
  if (measure == kDensityApd) {
    for (int c = 0; c < 3; ++c) e.k[c] = kAdxChannelFactor[c];
  }
  return e;
}

// Pixels whose code was clamped, per channel (R, G, B): below 0 (film much thinner than the base) and above the
// largest code (film denser than the encoding reaches).
// floor: pixels whose transmittance in a light channel was at or below kDensityMinT, so that the density of that
// channel was held at its floor before the matrix (R, G, B of the light).
struct DensityClip {
  uint64_t low[3] = {0, 0, 0};
  uint64_t high[3] = {0, 0, 0};
  uint64_t floor[3] = {0, 0, 0};
  uint64_t pixels = 0;
};

// Converts the image in place (width x height, `channels` 16-bit samples a pixel, the first three R, G, B) to density
// codes: merged the film base's merged matrix (color_pipeline.h apply_matrix on the sample / 65535), base_level the
// value the film base lands on (the bundle's normalization target / 65535), m the measure's matrix.
void density_encode_in_place(uint16_t* pixels, int width, int height, int channels, const float merged[9],
                             double base_level, const double m[9], const DensityEncoding& enc, DensityClip& clip);

// What the file's metadata records beyond the bundle's own (ProfileBundle::light_source_json, density_fit_json).
struct DensityRecord {
  DensityMeasure measure = kDensityStatusM;
  DensityEncoding encoding;
  std::string container;              // "dpx" or "tiff"
  std::string bundle_key;             // the profile bundle
  double film_base_rgb[3] = {0, 0, 0};
  std::string film_base_source;
  const DensityMatrix* matrix = nullptr;
  const ProfileBundle* bundle = nullptr;
  const DensityClip* clip = nullptr;  // null before the conversion
  // The balance the picture conversion's parameter search found for this frame (E, g, b as -E, --gain_g and --gain_b
  // take them, and the bracket it chose). It is not applied to the densities; the file records the colour part
  // (g and b) as offsets a grading tool can add (density_metadata_json: suggested_balance).
  bool has_balance = false;
  double balance_e = 1.0, balance_g = 1.0, balance_b = 1.0;
  std::string balance_target;
  std::string tool_version;
};
// The metadata as JSON text (schema negicc.density/1); indent < 0 for one line.
std::string density_metadata_json(const DensityRecord& rec, int indent = -1);

// The colour balance the picture conversion found for the frame, as offsets on the measure's density (zero sum).
// False, and zeros, when the record has no balance.
bool density_balance_offset(const DensityRecord& rec, double offset[3]);

// A print material's slopes for one measure, from the bundle (density_matrices.print_slopes; negicc-station
// src/print_slopes.py): the 3x3 from the measure's density to the print material's printing density, and the scale
// on each channel of that density, about a reference grey, that makes this film's grey scale print grey:
//
//     print density = to_print * about + slopes * (to_print * (D - about))
//
// They are a property of the film and the print material, measured on the profile's chart. A density file is written
// without them; its metadata records them (print_slopes) and the CLF beside it applies them (density_print_clf).
struct PrintSlopes {
  std::string id;                     // "kodak_portra_endura", "kodak_2383"
  std::string name;
  double to_print[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double slopes[3] = {1, 1, 1};
  double about[3] = {0, 0, 0};
};
std::vector<PrintSlopes> density_print_slopes(const ProfileBundle& bundle, DensityMeasure measure);

// A Common LUT Format (CLF, ST 2136-1) transform for one density file and one print material: the file's code values
// (0-1) to the print, as sRGB. print_clf is the print material alone (prints/<id>.clf, written by negicc-station:
// its printing density, kPrintGreyDensity exposed as mid grey, to sRGB). In front of its nodes goes one Matrix for the
// frame, with the balance (density_balance_offset) and the film's slopes:
//
//     printing density = kPrintGreyDensity + slopes * (to_print * (D + balance - grey))
//
// grey is the frame's mid grey (density_frame_grey), which so prints as the print's mid grey; without one, the film's
// reference grey (PrintSlopes::about). This is the only place the balance, the exposure and the slopes are applied.
// Empty when print_clf has no Matrix node to go in front of.
constexpr double kPrintGreyDensity = 0.70;
// The frame's mid grey as the measure's density: the chart's mid grey on the bracket the picture conversion's search
// chose for the frame (the bundle's print_slopes.bracket_grey_density), denser by log10(E) in every light channel for
// the exposure E it found. False, and zeros, when the search gave none or the bundle has no table.
bool density_frame_grey(const DensityRecord& rec, double grey[3]);
std::string density_print_clf(const DensityRecord& rec, const PrintSlopes& print, const std::string& print_clf);

struct DensityFileInfo {
  std::string make, model;            // the camera
  std::string software;               // "neg_process <version>"
  long long timestamp = 0;            // the capture's time (seconds since the epoch), 0 for now
};

// Writes density codes (width x height, `channels` samples a pixel, the first three R, G, B) to a 16-bit RGB TIFF with
// description (the metadata JSON) in its ImageDescription tag, or to a 10-bit RGB DPX (three samples in each 32-bit
// word, big-endian; the transfer and colorimetric codes and the element's description name the measure) with it in the
// user-defined data. 0 on success.
int write_density_tiff(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                       const std::string& description, const DensityFileInfo& info);
int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, DensityMeasure measure,
                      const DensityFileInfo& info);
