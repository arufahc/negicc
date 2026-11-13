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
// M is the bundle's 3x3 matrix for the measure (profile_bundle.h). The two encodings are the Cineon / ADX10 one
// (10 bits: offset 95, 500 codes per unit density) and the ADX16 one (16 bits: offset 1520, 8000 codes per unit
// density), used for every measure, so RP 180 density in a DPX is a Cineon file and APD in a TIFF is ADX16.
//
// Both files carry what is needed to read and recompute them, as JSON (schema negicc.density/1): the measure, the
// encoding, the film base, the matrix applied and the light's record with its spectrum. A TIFF has it in its
// ImageDescription tag, a DPX in its user-defined data section.

#pragma once

#include <cstdint>
#include <string>

#include "profile_bundle.h"

constexpr double kDensityMinT = 1e-4;   // the transmittance floor before the log (4.0 density)

struct DensityEncoding {
  int bits = 16;
  int offset = 1520;
  int scale = 8000;
  int max_code() const { return (1 << bits) - 1; }
};
inline DensityEncoding density_encoding(bool dpx) {
  DensityEncoding e;
  if (dpx) {
    e.bits = 10;
    e.offset = 95;
    e.scale = 500;
  }
  return e;
}

// Pixels whose code was clamped, per channel (R, G, B): below 0 (film much thinner than the base) and above the
// largest code (film denser than the encoding reaches).
struct DensityClip {
  uint64_t low[3] = {0, 0, 0};
  uint64_t high[3] = {0, 0, 0};
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
  std::string tool_version;
};
// The metadata as JSON text (schema negicc.density/1); indent < 0 for one line.
std::string density_metadata_json(const DensityRecord& rec, int indent = -1);

struct DensityFileInfo {
  std::string make, model;            // the camera
  std::string software;               // "neg_process <version>"
  long long timestamp = 0;            // the capture's time (seconds since the epoch), 0 for now
};

// Writes density codes (width x height, `channels` samples a pixel, the first three R, G, B) to a 16-bit RGB TIFF with
// description (the metadata JSON) in its ImageDescription tag, or to a 10-bit RGB DPX (three samples in each 32-bit
// word, big-endian, transfer and colorimetric "printing density") with it in the user-defined data. 0 on success.
int write_density_tiff(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                       const std::string& description, const DensityFileInfo& info);
int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, const DensityFileInfo& info);
