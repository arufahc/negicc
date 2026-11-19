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
// offset 95, 500 codes per unit density) and the 16-bit one (offset 1520, 8000 codes per unit density), the offsets and
// scales of ADX10 and ADX16 (SMPTE ST 2065-3:2020 4.3.3 equation 2 and 4.3.2 equation 1), so RP 180 density in a DPX
// is a Cineon file. Academy Printing Density is written as ADX (SMPTE ST 2065-3): the same offsets and scales with the
// standard's channel factors on the density, code = offset + scale * k * D with k = 1.00, 0.92, 0.95 for R, G, B.
// Status M and RP 180 take no factors (k = 1).
//
// Both files carry what is needed to read and recompute them, as JSON (schema negicc.density/1): the measure, the
// encoding, the film base, the matrix applied and the light's record with its spectrum. A TIFF has it in its
// ImageDescription tag, a DPX in its user-defined data section.

#pragma once

#include <array>
#include <cstdint>
#include <map>
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
  // A TriLED frame with --print: each material's own print search (np_pipeline; search_print_continuous), by material
  // id: its mid grey A, E (the -E units), g, b and the nearest bracket. Its CLF prints the frame as the picture
  // conversion would print it on that material. Empty: the CLFs take the picture conversion's search (white light).
  struct PrintSearch {
    double a = 0.0, e = 1.0, g = 1.0, b = 1.0;
    std::string bracket;
    bool converged = false;
    int steps = 0;
    double last_move[3] = {0, 0, 0};    // |dA| in stops, |dg|, |db| of the search's last step
    double above_knee = 0.0;            // the crop's share above the knee at the bootstrap
    double past_knee[3] = {0, 0, 0}, past_cut[3] = {0, 0, 0};   // at E (the matrix at E alone), per channel (shares)
    std::string stopped_by;             // parameters, residual or steps
    double a0 = 0.0, wish_l = 0.0;      // the A its bootstrap started from; the mean L* of the lightness wish it held
    double above_white_pct[3] = {0, 0, 0}, at_white_pct[3] = {0, 0, 0};   // at the result, per channel (percent)
    // --brightness, --tint, --warmth on this print (solve_print_knobs): a, g and b above are the moved ones; these the
    // search's, the knobs as given (brightness in stops, tint a*, warmth b*), and the print's grey where they put it
    bool knobs = false, knobs_limited = false;
    double knob_base_a = 0.0, knob_base_g = 1.0, knob_base_b = 1.0;
    double knob_given[3] = {0, 0, 0};
    double knob_grey_lab[3] = {0, 0, 0};
  };
  std::map<std::string, PrintSearch> print_search;
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
// Beside a density file goes, for each print material --print names, a CLF that prints it: density -> the print
// material's printing density -> its curves and dyes -> sRGB.
//
// A print material (a paper or a print film) and what is computed with it when a density file is written: one
// matrix for the frame, and the print's curves and colour table as CLF nodes.
//
// A print material is a print profile of spektrafilm (https://github.com/andreavolpato/spektrafilm), a JSON file read
// as published: those under 3rd_party/spektrafilm/profiles/ are built into neg_process, and --print takes any other
// by its path. The profile gives the curve of each layer, the dye spectra and base, and the layers' spectral
// sensitivity; the rest is computed when it is loaded (load_spektrafilm_print), the print matrix for the film of the
// bundle in use. A profile holds nothing of a film stock, and a film's bundle holds nothing of a print material.
//
// The model, for a log exposure of each layer:
//     dye amounts   a_c = curve_c(logE_c)                                   (linear interpolation, held at the ends)
//     transmittance t(w) = 10^-(sum_c a_c dye_c(w) + base(w))
//     XYZ           = sum_w t(w) observer(w)
// and a printing density pd is exposed as  logE = aim - (pd - grey_density),  aim the exposure that prints an 18 % grey.
//
// py-ref: negicc_station/src/print_emulation.py (PrintModel, frame_grey, print_clf)

struct PrintMaterial {
  std::string id, name, viewing_illuminant;
  // The printing density exposed as the print's mid grey: a stated value (DESIGN.md), which cancels inside the CLF.
  double grey_density = 0.70;
  // The 3x3 from a measure's density to the density this material reads, indexed by DensityMeasure (Status M, RP 180,
  // Academy Printing Density), for the film model named (the bundle's the material was loaded for).
  struct ToPrint {
    std::string film;
    std::array<double, 9> m;
  };
  std::vector<ToPrint> to_print[kDensityMeasures];
  // The matrix for a film model; null when the material has none for the measure and the film.
  const ToPrint* to_print_for(int measure, const std::string& film) const {
    for (const ToPrint& t : to_print[measure]) {
      if (t.film == film) return &t;
    }
    return nullptr;
  }
  std::vector<double> log_exposure;           // ascending
  std::vector<double> curves[3];              // dye amount of each layer along log_exposure
  std::vector<double> dye[3], base;           // spectral densities on the wavelength grid
  std::vector<double> observer[3];            // X, Y, Z weights on the grid (viewing light x observer)
  double white[3] = {1, 1, 1};
  double aim[3] = {0, 0, 0};
  double to_srgb[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // XYZ under the viewing light -> linear sRGB
};

// A print material from its spektrafilm print profile (a JSON with info and data), for the film of one bundle.
// Computed here:
//
//   the print's own density   read with its layers' sensitivity under a 3200 K blackbody printer lamp
//   the observer              the CIE 1931 2 degree observer (np_print.cc's CIE tables) under D50 for a paper, D55 for a print
//                             film (info.support), adapted to D65 with CAT02 for sRGB
//   aim                       the exposure of each layer that prints an 18 % grey
//   the print matrix          Q L inv(M) for each measure of the bundle, computed as Q inv(F) when the measure has
//                             no light_density_factors: F its film_to_measure (dye amounts to the measure's density),
//                             L the bundle's light_to_film, M = F L diag(alpha) the measure's full matrix, and Q the
//                             3x3 from the film's dye amounts to the print's own density, least squares on fixed
//                             sample colours of the bundle's film_model (every point of its neutral scale between
//                             log exposure -1.5 and 1.0, each with 27 colour variations). P M = Q L for every
//                             measure, so the bracket's uncalibrated APD slopes apply once
//
// The profile is checked first: info.stage "printing", info.support paper or film (it picks the viewing light),
// log_exposure ascending with one row of density_curves per value, each layer's curve rising with exposure (by more
// than 0.5 over the profile, never more than 0.02 below its highest value so far) and each layer's log_sensitivity
// peaking within 6 (log10) of the most sensitive layer's. --print loads a file given to it at option time, without a
// bundle, so every rule here stops a wrong file before any frame is converted.
//
// id names the material when the profile's info.stock is not a string. False with why in err; a bundle without a
// film_model gives a material without matrices, which frame_print refuses.
// py-ref: negicc_station/src/print_emulation.py (spektrafilm_print, print_matrix, film_to_print_density, PrintModel)
bool load_spektrafilm_print(const std::string& json_text, const std::string& id, const ProfileBundle& bundle,
                            PrintMaterial& out, std::string& err);

// The material as CLF nodes, to follow a Matrix that gives its printing density: Matrix (density -> log exposure) /
// Range / LUT1D (the curve of each layer) / Matrix / LUT3D (dye amounts -> linear sRGB, cube points a side) /
// Exponent (the sRGB curve). The text ends before </ProcessList>.
std::string print_clf_nodes(const PrintMaterial& m, int curve_points = 4096, int cube = 45);

// How one frame prints on one print material. Everything is looked up and multiplied; nothing is fitted when a file
// is written. The bundle gives, for the exposure bracket the picture conversion's search chose (without a search, the
// middle one), what the film's chart says of that bracket (DensityBracket), and the print material gives its matrix:
//
//   grey      the frame's mid grey as the measure's density: the bracket's mid grey, denser by log10(E) in every
//             light channel for the exposure E the search found. With the balance (the gains g and b, level
//             included) it prints as the print's mid grey. A narrow-band bundle with the film's grey balance curve
//             takes the curve's grey at the bracket's mid grey APD (curve_grey), as its print ICCs do.
//   slopes    they give the film's grey scale equal contrast in the three channels: from the grey balance curve
//             (curve_slopes), else the bracket's own.
//
//     printing density = grey_density + slopes * (to_print * (D + balance - grey))
//
struct FramePrint {
  std::string id, name;               // the print material
  double to_print[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::string to_print_for;           // the film model the matrix is for
  std::string bracket;
  bool searched = false;              // the bracket and exposure come from the search
  double exposure = 1.0;
  double grey[3] = {0, 0, 0};
  double slopes[3] = {1, 1, 1};
  bool from_curve = false;            // grey and slopes are the film's grey balance curve's (curve_grey, curve_slopes)
  // The material's own print search (DensityRecord::print_search): the mid grey A the grey and slopes are taken at,
  // and its gains, which the CLF's balance applies in place of the picture conversion's.
  bool print_searched = false;
  double a = NAN, gain_g = 1.0, gain_b = 1.0;
  bool converged = false;
  int steps = 0;
  double last_move[3] = {0, 0, 0}, above_knee = 0.0;
  double past_knee[3] = {0, 0, 0}, past_cut[3] = {0, 0, 0};   // at E (the matrix at E alone), per channel (shares)
  std::string stopped_by;
  double a0 = 0.0, wish_l = 0.0, above_white_pct[3] = {0, 0, 0}, at_white_pct[3] = {0, 0, 0};   // (PrintSearch)
  bool knobs = false, knobs_limited = false;   // the knobs moved a, gain_g and gain_b from these (PrintSearch)
  double knob_base_a = NAN, knob_base_g = 1.0, knob_base_b = 1.0, knob_grey_lab[3] = {0, 0, 0};
  double knob_given[3] = {0, 0, 0};            // brightness, tint, warmth as given
  double grey_density = 0.70;
};
// False, with why in err, when the bundle has nothing for the bracket or the material has no matrix for the measure.
bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err);

// A Common LUT Format (CLF, ST 2136-1) transform for one density file and one print material: the file's code values
// (0-1) to the print, as sRGB. Its first node is a Matrix for the frame, with the balance (density_balance_offset),
// the exposure and the slopes of FramePrint; nodes is the print material's own (print_clf_nodes). This is the only
// place the balance, the exposure and the slopes are applied.
std::string frame_print_clf(const DensityRecord& rec, const FramePrint& print, const std::string& nodes);

// The prints of one density file: for each print material asked for (--print: ids of builtin_print_materials, "all",
// or the path of a spektrafilm print profile; none gives no print), how the frame prints on it and the CLF to write,
// named after the whole output as the sidecar is (<output>.<print>.clf: a CLF fits one file, its code scale, measure
// and channel factors).
// The print materials built into neg_process (3rd_party/spektrafilm/profiles/<id>.json), in the order of their ids,
// with their matrices for the film of bundle; without a bundle, or with one that has no film_model, they have no
// matrix (enough to list them). One that cannot be read is left out, with a warning.
std::vector<PrintMaterial> builtin_print_materials(const ProfileBundle* bundle = nullptr);
// Whether a --print value names a file (a spektrafilm print profile: a path, or a name ending in .json) and not a
// built-in print material.
bool is_print_file(const std::string& name);

// The text of a spektrafilm print profile given as a file, and its stem (the name without folder and .json), the id of
// a profile without info.stock. False with "cannot be read" in err for a file that cannot be opened or read.
bool read_print_file(const std::string& path, std::string& text, std::string& stem, std::string& err);

struct DensityPrints {
  std::vector<PrintMaterial> materials;
  std::vector<FramePrint> frames;
  std::vector<std::string> paths;
  std::string metadata_json;          // the "print" block of the file's metadata; empty with no print
  std::string error;                  // a material asked for that has no CLF, and why: the run fails
};
DensityPrints density_prints(const DensityRecord& rec, const std::string& output,
                             const std::vector<std::string>& print_ids);
// Writes the CLFs. 0 on success; a CLF that cannot be written removes those written before it and returns 1.
int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints);
void remove_print_clfs(const DensityPrints& prints);

// ---- The print ICC: a frame printed as the picture conversion renders it ---------------------------------------------
//
// With --print and a picture output on a profile bundle, the print's own search runs on the print profile
// (search_print_continuous) and the conversion renders that profile at the search's A: the film printed on the print
// material, shaped as the chart profiles are (ICC v2, RGB -> PCS XYZ D50, A2B0 lut16: 4096-entry input curves, a 45^3
// table, identity output curves), so the conversion and the output run it as they run them. The bundle carries one
// print ICC per print material (print_profiles, built by the profiling tool when it saved the bundle);
// its table serves every bracket, and each bracket's three input curves are computed here:
//
//     curve_c(x) = Paper_c((G_c(-log10(unknee(x) / n0)) - A_mid) (1 + t_c))
//
//   G_c       the film's grey balance curve (GreyBalanceCurves): light density in channel c -> APD of the film's grey
//   A_mid     the bracket's mid grey as APD: the green APD of its gs10 (bracket_mid_apd)
//   t_c       the bracket's tilt relative to the curve (DensityBracket::tilt; 0 without grey_relative)
//   Paper_c   0.75 curve_c(aim_c - eta_c alpha_i) / top_c + 0.25 (1 - i / (kPaperAlphaPoints - 1)) on
//             alpha_i = kPaperAlphaStart + kPaperAlphaStep i (-1.5 .. 1.8), eta = P (1, 1, 1) with P the material's
//             print matrix for APD, top_c the channel's largest value; held strictly decreasing on [0, 1]
//             (paper_curve)
//   unknee    the inverse of the soft knee at kKnee the conversion applies before the profile
//
// and the table, at node a: alpha_c = Paper_c^-1(a_c), printing density 0.70 + Q L (psi alpha) with
// psi = M^-1 (1, 1, 1) and Q L = P M, the material's dyes, XYZ under its viewing light adapted to D50 (Bradford; none
// for a paper). A print material without a bundle ICC (a --print file), or whose bundle ICC no longer matches what is
// computed here, gets its table computed here (print_table).
// The print ICC's copyright tag, as spektrafilm's licence asks of a derivative (its author, its source, the licence,
// and that and by whom it was modified); the print CLF's description says the same. The profile-building tools write
// the same text into the bundle's print ICCs.
constexpr const char* kPrintIccCopyright =
    "Tables derived by negicc from spektrafilm's print profile (Andrea Volpato, "
    "https://github.com/andreavolpato/spektrafilm), modified: its curves resampled and its dye spectra computed into "
    "colours. CC BY-SA 4.0, https://github.com/andreavolpato/spektrafilm/blob/"
    "3bb2c2d2801ff68b92019cf1dbcbb133d60832bc/SPEKTRAFILM_LICENSE.txt";
constexpr int kPrintIccPoints = 4096;
// The largest XYZ a lut16 (PCS XYZ) holds: code FFFFh as a u1Fixed15Number (ICC.1:2022 4.8 and 6.3.4.2 Table 11).
constexpr double kPcsMax = 1.0 + 32767.0 / 32768.0;
constexpr int kPrintIccCube = 45;
// Paper_c's grid: -1.5 to 1.8 in steps of 0.005, 661 points.
constexpr double kPaperAlphaStart = -1.5, kPaperAlphaStep = 0.005;
constexpr int kPaperAlphaPoints = 661;
using PrintCurves = std::array<std::vector<double>, 3>;

// The bracket's mid grey as green APD: its grey_light_density through the bundle's APD matrix.
double bracket_mid_apd(const ProfileBundle& bundle, const DensityBracket& bracket);
// curve_grey and curve_slopes at a mid grey A (green APD) in place of a bracket's.
void curve_grey_at(const ProfileBundle& bundle, double a, double light[3]);
void curve_slopes_at(const ProfileBundle& bundle, double a, double slopes[3]);
// The print's bracket term at a mid grey A, the one rule for the picture and the CLF: each bracket has its own tilt,
// and between two brackets' A_mid the tilt is interpolated linearly in A, held beyond the end ones, so at a bracket's
// own A_mid it is that bracket's tilt; a bundle with one bracket has that bracket's tilt. a_mid and tilt per bracket,
// in any order; zeros without brackets.
void print_tilt_at(const std::vector<double>& a_mid, const std::vector<std::array<double, 3>>& tilt, double a,
                   double t[3]);
// The same from a bundle's density brackets.
void print_tilt_at(const ProfileBundle& bundle, double a, double t[3]);
// The light densities of the grey balance curve's grey with the bracket's mid grey APD: the bracket's mid grey as the
// curve has it. A print CLF of a narrow-band bundle takes it as the frame's mid grey (frame_print).
void curve_grey(const ProfileBundle& bundle, const DensityBracket& bracket, double light[3]);
// The slopes a print CLF takes for the bracket from the curve: with n(g) the light densities of the curve's grey at
// green APD g, k_c = sum_j M_cj dn_j/dg at the mid grey (dn_j/dg = 1 / G_j', G_j' a central difference of one step
// of the curve's table), slopes = (k_green / k_red, 1, k_green / k_blue).
void curve_slopes(const ProfileBundle& bundle, const DensityBracket& bracket, double slopes[3]);
// Paper_c on kPaperAlphaStart + kPaperAlphaStep i.
PrintCurves paper_curve(const PrintMaterial& m, const double p_apd[9]);
// A bracket's input curves on x = i / (kPrintIccPoints - 1), held non-decreasing (a 1e-9 ramp). gains (r, g, b): the
// print's gains as a density offset after the knee, dL_c = -log10(unknee(x) g_c / n0) (null: none).
PrintCurves bracket_print_curves(const GreyBalanceCurves& g, const PrintCurves& paper, double a_mid, double n0,
                                 const double tilt[3], const double gains[3] = nullptr);
// The table: kPrintIccCube^3 XYZ D50, red slowest.
std::vector<double> print_table(const PrintMaterial& m, const double p_apd[9], const double m_apd[9],
                                const PrintCurves& paper);
// An ICC v2.4 input profile with tags desc, cprt, wtpt and A2B0 (the curves, the table, identity output curves),
// dated date (year, month, day, hour, minute, second), so the same inputs give the same bytes.
std::vector<uint8_t> write_print_icc(const PrintCurves& curves, const std::vector<double>& table,
                                     const std::string& description, const std::string& copyright,
                                     const std::array<int, 6>& date);
// The date a bundle's print ICCs carry: its source file's timestamp (profile_<Key>_<YYYYMMDD>_<HHMMSS>), else
// 2026-01-01 00:00:00.
std::array<int, 6> bundle_date(const ProfileBundle& bundle);
// The SHA-256 of bytes, as 64 lowercase hex digits.
std::string sha256_hex(const std::string& bytes);
// Multiplies the X, Y and Z of an ICC's A2B0 table (a lut16's) by k[0], k[1] and k[2] in place, each code rounded and
// held at 65535: the print relative to its paper's white in the profile an output embeds. False when it has none.
bool scale_print_icc_table(std::vector<uint8_t>& icc, const double k[3]);
// Writes xyz into an ICC's wtpt tag in place (the media white: the paper's, for a print made relative to its paper).
// False when it has none.
bool set_icc_wtpt(std::vector<uint8_t>& icc, const double xyz[3]);
// The paper's white under the material's viewing light, as the print CLF's XYZ has it: the colour of the print where
// every layer holds its least dye (each curve's lowest value: the base and the layers' fog), the lightest it gets.
void print_paper_xyz(const PrintMaterial& m, double xyz[3]);
// Writes curves into an ICC's A2B0 input tables in place (a lut16 with as many entries). False when it has none.
bool set_print_icc_curves(std::vector<uint8_t>& icc, const PrintCurves& curves);

// The print at a continuous mid grey A (np_pipeline: the print search): the input curves above with A for
// A_mid and the brackets' tilt at A (print_tilt_at: linear in A between the two neighbouring brackets, held beyond
// the end ones). At A = A_mid of a bracket the profile is that bracket's to the bit (set_curves_at writes the 16-bit
// codes an ICC holds, as code / 65535).
struct PrintContinuum {
  GreyBalanceCurves g;
  PrintCurves paper;
  double n0 = 1.0;
  std::vector<double> a_mid;                  // per bundle.targets, in its order
  std::vector<std::array<double, 3>> tilt;    // per bundle.targets
  void tilt_at(double a, double t[3]) const;  // print_tilt_at over these brackets
  // gains (r, g, b): the print's gains in its curves, after the knee (null: none).
  PrintCurves curves_at(double a, const double gains[3] = nullptr) const;
  void set_curves_at(double a, ParsedFilmProfile& prof, const double gains[3] = nullptr) const;
  // The tables set_curves_at writes into prof.in_trc.
  void trc_at(double a, const double gains[3], std::vector<float> trc[3]) const;
  double a_lo() const;                        // the range A is searched in: the curve's green APD range
  double a_hi() const;
};

// The print profile of every bracket of a bundle on one print material (a built-in id or a spektrafilm file).
struct BracketPrints {
  PrintContinuum continuum;                    // the print at a continuous A
  std::string id, name;
  std::string file, file_sha256;               // a material given as a file: its path as given and its SHA-256
  bool from_bundle = false;                    // the table is the bundle's print ICC's, else computed here
  std::vector<std::vector<uint8_t>> icc;       // per bundle.targets, in its order
};
bool bracket_print_profiles(const ProfileBundle& bundle, const std::string& print, BracketPrints& out,
                            std::string& err);
