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

#include <array>
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
  // The metadata's "print" block, as JSON text: how the frame prints on each print material
  // (DensityPrints::metadata_json). Empty: none.
  std::string print_json;
};
// The metadata as JSON text (schema negicc.density/1); indent < 0 for one line.
std::string density_metadata_json(const DensityRecord& rec, int indent = -1);

// The colour balance the picture conversion found for the frame, as offsets on the measure's density with their mean
// taken out (zero sum), and that mean. The conversion multiplies the light's channels by E, E g and E b, so the whole
// of what g and b do to the density is offset + mean in every channel: the zero-sum part is the colour, the mean is
// level. False, and zeros, when the record has no balance.
bool density_balance_offset(const DensityRecord& rec, double offset[3], double* mean = nullptr);

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


// ---- The print of a density file ------------------------------------------------------------------------------------
//
// Beside a printing density file goes one CLF per print material that prints it: density -> the print material's
// printing density -> its curves and dyes -> sRGB.
//
// A print material (a paper or a print film) and what is computed with it when a density file is written: the slopes
// that make the film's grey steps print grey, and the print's curves and colour table as CLF nodes.
//
// A print material is a data file, prints/<id>.json (schema negicc.print_material/1, written by negicc-station
// src/print_slopes.py: print_material), built into neg_process. It holds the curve of each layer, the dye spectra and
// base, the observer under the viewing light, and the 3x3 from each printing density measure to the density the
// material reads itself. It holds nothing of a film stock, and a film's bundle holds nothing of a print material.
//
// The model, for a log exposure of each layer:
//     dye amounts   a_c = curve_c(logE_c)                                   (linear interpolation, held at the ends)
//     transmittance t(w) = 10^-(sum_c a_c dye_c(w) + base(w))
//     XYZ           = sum_w t(w) observer(w)
// and a printing density pd is exposed as  logE = aim - (pd - grey_density),  aim the exposure that prints an 18 % grey.
//
// py-ref: negicc_station/src/print_slopes.py (PrintModel, fit_slopes, bracket_slopes, print_clf)

struct PrintMaterial {
  std::string id, name, viewing_illuminant;
  double grey_density = 0.70;                 // the printing density exposed as the print's mid grey
  // The 3x3 from a measure's density to the density this material reads, for [0] RP 180 and [1] Academy Printing
  // Density: one per film model by its name (a bundle's density_matrices.film_dye_spectra) and "any" for other films.
  struct ToPrint {
    std::string film;
    std::array<double, 9> m;
  };
  std::vector<ToPrint> to_print[2];
  // The matrix for a film model, else the one for any film; null when the material has none for the measure.
  const ToPrint* to_print_for(int measure, const std::string& film) const {
    const ToPrint* any = nullptr;
    for (const ToPrint& t : to_print[measure]) {
      if (t.film == film) return &t;
      if (t.film == "any") any = &t;
    }
    return any;
  }
  std::vector<double> log_exposure;           // ascending
  std::vector<double> curves[3];              // dye amount of each layer along log_exposure
  std::vector<double> dye[3], base;           // spectral densities on the wavelength grid
  std::vector<double> observer[3];            // X, Y, Z weights on the grid (viewing light x observer)
  double white[3] = {1, 1, 1};
  double aim[3] = {0, 0, 0};
  double to_srgb[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // XYZ under the viewing light -> linear sRGB
  double judged_lo = 12.0, judged_hi = 92.0;  // the printed lightness between which a grey step is judged
  double slope_lo = 0.6, slope_hi = 1.5;
  // The material's own grey scale, as slopes (own_slopes): what the fit gives the neutral scale of the films the
  // material was made for. A frame's slopes are the fit's divided by these, so the print keeps the stock's look and
  // only the film's share is corrected. 1, 1, 1 for a material that is printed neutral.
  double own_slopes[3] = {1.0, 1.0, 1.0};
  int min_patches = 12;                       // the fewest judged steps for a fit
};

// Reads a print material's JSON. False with why in err.
bool load_print_material(const std::string& json_text, PrintMaterial& out, std::string& err);

// CIELAB (L*, a*, b*) of the print of a log exposure per layer, under the material's viewing light.
void print_lab(const PrintMaterial& m, const double log_e[3], double lab[3]);

// The slopes on the red and the blue printing density (green stays 1), about the row grey_at, that make grey steps
// print neutral: the steps (rows of the material's printing density) are printed with the row grey_at as an 18 % grey
// and one printer-light setting; the slopes and the red and blue lights minimise the steps' a* and b*, over the steps
// that print between judged_lo and judged_hi before the slopes. ok is false, and the slopes 1, when fewer than
// min_patches steps are judged.
struct PrintSlopeFit {
  bool ok = false;
  double slopes[3] = {1.0, 1.0, 1.0};
  bool at_bound = false;                            // a slope ended on slope_lo or slope_hi
  int judged = 0;
  // Mean C*ab of the judged steps printed as a CLF prints them, with no printer light of the fit's own: without the
  // slopes and with them.
  double chroma_before = 0.0, chroma_after = 0.0;
  double slopes_neutral[3] = {1.0, 1.0, 1.0};       // the fit's own, before the division by the material's own_slopes
};
PrintSlopeFit fit_print_slopes(const PrintMaterial& m, const std::vector<std::array<double, 3>>& print_density,
                               int grey_at);

// The material as CLF nodes, to follow a Matrix that gives its printing density: Matrix (density -> log exposure) /
// Range / LUT1D (the curve of each layer) / Matrix / LUT3D (dye amounts -> linear sRGB, cube points a side) /
// Exponent (the sRGB curve). The text ends before </ProcessList>.
std::string print_clf_nodes(const PrintMaterial& m, int curve_points = 4096, int cube = 45);

// How one frame prints on one print material: computed when the file is written, from the bundle's chart readings
// and the print material's data. Neither is stored in the other.
//
//   bracket   the exposure bracket the picture conversion's search gave the frame; without a search, the middle one.
//   grey      the frame's mid grey as the measure's density: the chart's mid grey (gs10) on that bracket, denser by
//             log10(E) in every light channel for the exposure E the search found. With the balance (the gains g and
//             b, level included) it prints as the print's mid grey.
//   slopes    fitted on that bracket's grey steps, as the print material reads them, about its mid grey, so that they
//             print neutral (fit_print_slopes). 1 when the bracket has too few usable steps.
//
//     printing density = grey_density + slopes * (to_print * (D + balance - grey))
struct FramePrint {
  std::string id, name;               // the print material
  double to_print[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::string to_print_for;           // the film model the matrix is for, or "any"
  std::string bracket;
  bool searched = false;              // the bracket and exposure come from the search
  double exposure = 1.0;
  double grey[3] = {0, 0, 0};
  double slopes[3] = {1, 1, 1};       // as applied: the fit's, less the print material's own grey scale
  double slopes_neutral[3] = {1, 1, 1};
  double own_slopes[3] = {1, 1, 1};   // the print material's
  bool fitted = false;
  bool at_bound = false;              // a slope ended on the fit's bound: clipped, not found
  int steps = 0, steps_judged = 0;
  double chroma_before = 0.0, chroma_after = 0.0;   // of the judged steps as the CLF prints them: no slopes, slopes
  double grey_density = 0.70;
};
// False, with why in err, when the bundle has no grey steps for the bracket or the material has no matrix for the
// measure.
bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err);

// A Common LUT Format (CLF, ST 2136-1) transform for one density file and one print material: the file's code values
// (0-1) to the print, as sRGB. Its first node is a Matrix for the frame, with the balance (density_balance_offset),
// the exposure and the slopes of FramePrint; nodes is the print material's own (print_clf_nodes). This is the only
// place the balance, the exposure and the slopes are applied.
std::string frame_print_clf(const DensityRecord& rec, const FramePrint& print, const std::string& nodes);

// The prints of one density file: for every print material built into neg_process (prints/<id>.json), how the frame
// prints on it and the CLF to write, named after the whole output as the sidecar is (<output>.<print>.clf: a CLF fits
// one file, its code scale, measure and channel factors). Empty for Status M, which is not a printing density.
struct DensityPrints {
  std::vector<PrintMaterial> materials;
  std::vector<FramePrint> frames;
  std::vector<std::string> paths;
  std::string metadata_json;          // the "print" block of the file's metadata; empty with no print
};
DensityPrints density_prints(const DensityRecord& rec, const std::string& output);
// Writes the CLFs. 0 on success; a CLF that cannot be written removes those written before it and returns 1.
int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints);
void remove_print_clfs(const DensityPrints& prints);
