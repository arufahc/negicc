// Copyright 2024-2026 Alpha Lam <arufa.hc@gmail.com>
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

// neg_process, the program: converts film negative RAW scans with the profiles and intent adapters built into it.
// In this order: the declarations of the print, the film base, the options, the output and the pipeline (each needs
// only those before it); then their code, in the same order; then main. Its inputs (np_input.h, np_input.cc) and
// the library negicc-station also links (libnegicc_dinov3.a) are apart.

#include <lcms2.h>
#include <netinet/in.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "argparse/argparse.hpp"
#include "color_pipeline.h"
#include "dinov3.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "embedded_data.h"
#include "film_base_detector.h"
#include "film_frame_detector.h"
#include "flat_field.h"
#include "input_curves.h"
#include "libraw/libraw.h"
#include "multi_target_optimizer.h"
#include "nlohmann/json.hpp"
#include "np_input.h"
#include "np_log.h"
#include "profile_bundle.h"
#include "rgb_view.h"
#include "tiff_head.h"
#include "tone_stage.h"

// ====================================================================================================================
// Print: declarations
// ====================================================================================================================

// The density output (-P statusm, rp180, apd): the negative's density above the film base in a standard measure, as
// code values in a 10-bit DPX or a 16-bit TIFF.
//
// A pixel, after the film base's merged matrix (crosstalk correction and film base scale; the film base lands on
// base_level), is a transmittance relative to bare film in each of the scanning light's channels:
//
//     T = v / base_level,  D_light = -log10(max(T, kDensityMinT)),  D = M * D_light,  code = round(offset + scale * D)
//
// M is the bundle's 3x3 matrix for the measure (profile_bundle.h). The two encodings are the Cineon one (10 bits:
// offset 95, 500 codes per unit density) and the 16-bit one (offset 1520, 8000 codes per unit density), so RP 180
// density in a DPX is a Cineon file. Academy Printing Density is written as ADX: ADX10 and ADX16 have these offsets
// and scales (SMPTE ST 2065-3:2020 4.3.3 equation 2 and 4.3.2 equation 1), with the standard's channel factors on the
// density, code = offset + scale * k * D with k = 1.00, 0.92, 0.95 for R, G, B.
// Status M and RP 180 take no factors (k = 1).
//
// Both files carry what is needed to read and recompute them, as JSON (schema negicc.density/1): the measure, the
// encoding, the film base, the matrix applied and the light's record with its spectrum. A TIFF has it in its
// ImageDescription tag, a DPX in its user-defined data section.

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
  // A density run with --print, on either light: each material's own print search (the pipeline code below;
  // search_print_continuous), by material id: its mid grey A, E (the -E units), g, b and the nearest bracket. Its CLF
  // prints the frame as the picture conversion would print it on that material. Empty (a material without a print
  // search): its CLF takes the scene's search.
  struct PrintSearch {
    double a = 0.0, e = 1.0, g = 1.0, b = 1.0;
    std::string bracket;
    bool converged = false;
    int steps = 0;
    double last_move[3] = {0, 0, 0};    // |dA| in stops, |dg|, |db| of the search's last step
    double above_knee = 0.0;            // the crop's share above the knee at the bootstrap
    double past_knee[3] = {0, 0, 0}, past_cut[3] = {0, 0, 0};   // at E (the matrix at E alone), per channel (shares)
    std::string stopped_by;             // parameters, residual or steps
    double a0 = 0.0, target_l = 0.0;    // the A its bootstrap started from; the mean L* of the lightness target it held
    double above_white[3] = {0, 0, 0}, at_white[3] = {0, 0, 0};   // at the result, per channel (shares)
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
  long long timestamp = 0;            // the capture's time (seconds since the epoch); 0 or less: no date written
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
//     dye amounts   delta_c = curve_c(logE_c)                    (linear interpolation, held at the ends)
//     transmittance tau(lambda) = 10^-(sum_c delta_c dye_c(lambda) + base(lambda))
//     XYZ           = sum_lambda tau(lambda) observer(lambda)
// and a printing density pd is exposed as logE = aim - (pd - grey_density), aim the exposure that prints an 18 % grey.

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
  // Why a measure has no matrix when the bundle's film_model gives none (not finite); empty otherwise.
  std::string no_print_matrix[kDensityMeasures];
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
//   the observer              the CIE 1931 2 degree observer (the print code's CIE tables) under D50 for a paper, D55
//                             for a print film (info.support), adapted to D65 with CAT02 for sRGB
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
// film_model gives a material without matrices, which frame_print refuses, and one whose film_model gives a matrix
// that is not finite a material without matrices and no_print_matrix saying why.
bool load_spektrafilm_print(const std::string& json_text, const std::string& id, const ProfileBundle& bundle,
                            PrintMaterial& out, std::string& err);

// The material as CLF nodes, to follow a Matrix that gives its printing density: Matrix (density -> log exposure) /
// Range / LUT1D (the curve of each layer) / Matrix / LUT3D (dye amounts -> linear sRGB, cube points a side) /
// Exponent (the sRGB curve). The text ends before </ProcessList>; empty when a value is not a finite number.
std::string print_clf_nodes(const PrintMaterial& m, int curve_points = 4096, int cube = 45);

// How one frame prints on one print material. Everything is looked up and multiplied; nothing is fitted when a file
// is written. The bundle gives, for the exposure bracket the picture conversion's search chose (without a search, the
// middle one), what the film's chart says of that bracket (DensityBracket), and the print material gives its matrix:
//
//   grey      the frame's mid grey as the measure's density: the bracket's mid grey, denser by log10(E) in every
//             light channel for the exposure E the search found. With the balance (the gains g and b, level
//             included) it prints as the print's mid grey. A bundle with the film's grey balance curve, under
//             either light, takes the curve's grey at the bracket's mid grey APD (curve_grey), as its print ICCs do.
//   slopes    they give the film's grey scale equal contrast in the three channels: from the grey balance curve
//             (curve_slopes), else the bracket's own (read only for a bundle without the curve).
//
//     printing density = grey_density + slopes * (to_print * (D + balance - grey))
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
  double a0 = 0.0, target_l = 0.0, above_white[3] = {0, 0, 0}, at_white[3] = {0, 0, 0};   // (PrintSearch)
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
// The prints of one density file: for each print material asked for (--print: ids of builtin_print_materials, "all",
// or the path of a spektrafilm print profile; none gives no print), how the frame prints on it and the CLF to write,
// named after the whole output as the sidecar is (<output>.<material>.clf: a CLF fits one file, its code scale,
// measure and channel factors).
DensityPrints density_prints(const DensityRecord& rec, const std::string& output,
                             const std::vector<std::string>& print_ids);
// Writes the CLFs. 0 on success; a CLF that cannot be written removes those written before it and returns 1.
int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints);
void remove_print_clfs(const DensityPrints& prints);

// ---- The print ICC: a frame printed as the picture conversion renders it --------------------------------------------
//
// With --print and a picture output on a profile bundle, the print's own search runs on the print profile
// (search_print_continuous) and the conversion renders that profile at the search's A: the film printed on the print
// material, shaped as the chart profiles are (ICC v2, RGB -> PCS XYZ D50, A2B0 lut16: 4096-entry input curves, a 45^3
// table, identity output curves), so the conversion and the output run it as they run them. The bundle carries one
// print ICC per print material (print_profiles, built by the profiling tool when it saved the bundle);
// its table serves every bracket, and each bracket's three input curves are computed here:
//
//     in_c(x) = Paper_c((G_c(-log10(unknee(x) / n0)) - A_mid) (1 + t_c))
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
// and that and by whom it was modified). The profile-building tools write the same text into the bundle's print ICCs.
// The print CLF's description (frame_print_clf in the print code below) gives the same attribution in its own words,
// with the licence at spektrafilm's main branch rather than at this pinned commit.
constexpr const char* kPrintIccCopyright =
    "Tables derived by negicc from spektrafilm's print profile (Andrea Volpato, "
    "https://github.com/andreavolpato/spektrafilm), modified: its curves resampled and its dye spectra computed into "
    "colours. CC BY-SA 4.0, https://github.com/andreavolpato/spektrafilm/blob/"
    "3bb2c2d2801ff68b92019cf1dbcbb133d60832bc/SPEKTRAFILM_LICENSE.txt";
constexpr int kPrintIccPoints = 4096;
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
// curve has it. A print CLF of a bundle with grey_balance_curves, under either light, takes it as the frame's mid grey
// (frame_print).
void curve_grey(const ProfileBundle& bundle, const DensityBracket& bracket, double light[3]);
// The slopes a print CLF takes for the bracket from the curve (DESIGN.md 5.9): with n(A) the light densities of the
// curve's grey at green APD A, gamma_c = sum_j M_cj dn_j/dA at the mid grey (dn_j/dA = 1 / G_j', G_j' a central
// difference of one step of the curve's table), slopes = (gamma_green / gamma_red, 1, gamma_green / gamma_blue).
void curve_slopes(const ProfileBundle& bundle, const DensityBracket& bracket, double slopes[3]);
// Paper_c on kPaperAlphaStart + kPaperAlphaStep i.
PrintCurves paper_curve(const PrintMaterial& m, const double p_apd[9]);
// A bracket's input curves on x = i / (kPrintIccPoints - 1), held non-decreasing (a 1e-9 ramp). gains (r, g, b): the
// print's gains as a density offset after the knee, dL_c = -log10(unknee(x) g_c / n0) (null: none).
PrintCurves bracket_print_curves(const GreyBalanceCurves& g, const PrintCurves& paper, double a_mid, double n0,
                                 const double tilt[3], const double gains[3] = nullptr);
// The table: kPrintIccCube^3 XYZ D50, red slowest; empty when a value is not a finite number.
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

// The print at a continuous mid grey A (the pipeline code below: the print search): the input curves above with A for
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

// ====================================================================================================================
// Film base: declarations
// ====================================================================================================================

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
//      (the median absolute deviation times kMadToSigma) are taken;
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

// The exposure of a scan or a film base capture whose file records none (Scan::scan_exposure,
// read_film_base_capture): 1/8 s at ISO 100.
constexpr double kDefaultScanShutterS = 0.125, kDefaultScanIso = 100.0;

constexpr int kBaseTileGrid = 96;             // square tiles along the longer side (fewer when they would be under 8
                                              // pixels)
constexpr int kBaseMinTiles = 8;              // tiles that must agree, and at least ...
constexpr int kBaseMinShare = 200;            // ... one tile in this many
constexpr double kBaseTileMaxSpread = 0.03;   // density: a tile's green spread up to which it is smooth whatever the
                                              // rest
constexpr double kBaseSpreadFactor = 1.5;     // ... and up to this many times the median tile's spread
constexpr double kBaseTileMargin = 0.02;      // density: how far from the group's brightest tile a kept tile may lie
constexpr double kBaseColourTol = 0.25;       // density: how far a tile's G/R and B/R may lie from the bundle's base

struct FilmBaseCapture {
  double rgb[3] = {0.0, 0.0, 0.0};         // the film base: through the crosstalk matrix, at the scan's exposure
  double decode_rgb[3] = {0.0, 0.0, 0.0};  // as the capture's half-size decode gives it, at the capture's exposure
  int roi[4] = {0, 0, 0, 0};               // the region read (x, y, w, h in sensor pixels): the given one, or the kept
                                           // tiles' bounding box
  std::string roi_source;                  // "given", "tiles" or "statistics"
  int tiles_kept = 0, tiles_total = 0;     // with the tiles
  double spread_density = 0.0;             // the spread of the kept tiles' green medians (a given region: of its
                                           // pixels)
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

// ====================================================================================================================
// Options: declarations
// ====================================================================================================================

// neg_process's command line: the options, parsed and checked on their own (what needs the profile is checked by
// prepare, the pipeline declarations below).

struct Options {
  int argc = 0;
  char** argv = nullptr;                       // the command line, for the sidecar

  // Input
  std::vector<std::string> files;              // one RAW, DNG, TIFF or JPEG, three captures, or a pixel shift's four
  InputMode input_mode = InputMode::kRaw;
  bool tiff_input = false;                     // one TIFF: camera RGB with a tri bundle, else Rec.2020
  const char* implied_light = nullptr;         // the light the input implies ("tri", "d65"), nullptr for a RAW
  int quality = -1;                            // --quality
  bool no_crop = false;                        // --no_crop
  std::string flat;                            // --flat, empty when not given

  // Profile
  std::string film_stock, light_source, film_profile;  // --film_stock, --light_source, -p (empty when not given)
  std::string target;                          // --target, empty when not given
  bool target_given = false;

  // Conversion parameters
  std::vector<float> r_coeff, g_coeff, b_coeff;  // -r, -g, -b
  bool row_given[3] = {false, false, false};
  float exposure_comp = 1.0f;                  // -E
  bool exposure_given = false;                 // -E (the search keeps it)
  float gain_g = 1.0f, gain_b = 1.0f;          // --gain_g, --gain_b (the search holds them)
  bool gain_g_given = false, gain_b_given = false;
  float brightness = 0.0f;                     // --brightness (stops of the grey's luminance)
  float tint = 0.0f;                           // --tint (a* of the grey)
  float warmth = 0.0f;                         // --warmth (b* of the grey)
  bool knobs_given = false;                    // --brightness, --tint or --warmth given

  // Geometry
  int roi[4] = {0, 0, 0, 0};                   // --roi x y w h (sensor pixels)
  bool has_roi = false, roi_auto = false;      // --roi given as numbers / as 'auto'
  char roi_axis_hint = 0;                      // --roi auto h|v
  FilmFormat film_format = FILM_35MM;          // --film_format
  bool film_format_given = false;              // --film_format was given (else 35mm for --roi auto)
  int rot_cw = 0;                              // --rot_cw (or the --roi's fifth value)
  bool hflip = false, vflip = false;

  // Parameter search
  // --dino: the intent adapter (an adapter.onnx or its directory) and the neutral and tone heads' files that replace
  // the built-in ones; empty: the built-in one. dino_arg is --dino's value as given, which the replay line repeats.
  std::string dino_arg, dino_adapter, neutral_head_file, tone_head_file;
  // The tone stage (tone_stage.h). --tone_stage auto (the default) runs it after every bundle search with -P srgb, on
  // white light and TriLED alike, whether or not the curve fit runs; on does the same and is an error where it does
  // not apply; off turns it off. curve:black,white,gamma,contrast (ToneCurve) replays a recorded curve (a sidecar's
  // cli_equivalent), with given values and no search. highlights_first and least_total run it as auto does, with
  // their own rule where no share of the moves holds both ends (tone_fallback; auto and on keep the shadows first).
  enum ToneStageMode { kToneStageAuto, kToneStageOn, kToneStageOff };
  ToneStageMode tone_stage = kToneStageAuto;
  ToneClipFallback tone_fallback = kToneFallbackShadowsFirst;
  bool tone_replay_given = false;
  double tone_replay[4] = {};
  std::string dino_backend = "auto";
  int dino_iters = 4;
  bool check_renderer = false;   // --check_renderer: the bundle search's passes through a counting renderer
  // --probe_intent FILE, --probe_params "t,E,g,b;...", --probe_stops s [s ...]: a measurement aid
  // (ParamSearchOptions::probe_path). The search writes the network's predictions for chosen renders to
  // FILE and neg_process stops without converting (the print's search writes it and the conversion goes on).
  std::string probe_intent;
  std::vector<std::array<float, 4>> probe_params;
  std::vector<float> probe_stops = {0.0f};
  bool dino_iters_given = false;

  // Fitted G/B input curves (input_curves.h). --fit_curves auto (the default) fits them on every white-light bundle
  // search with -P srgb or srgb-g10, given parameters or not, and nowhere else (TriLED, whose curves are the measured
  // ones; another or no -P); on does the same and is an error where it does not apply; off turns the fit off. The tone
  // stage has its own flag.
  enum FitCurvesMode { kFitCurvesAuto, kFitCurvesOn, kFitCurvesOff };
  FitCurvesMode fit_curves = kFitCurvesAuto;
  bool input_curves_given = false;             // recorded curves given to --fit_curves (replayed, not fitted)
  negicc::InputCurves input_curves;

  // The film base (--film_base): auto (the default: a base capture's values kept in <output>.json by a previous run,
  // else read from the scan), or a capture of unexposed film, with the region to read it over (x y w h in sensor
  // pixels) when given after a colon; without one the region is found on the capture (the film base declarations
  // above).
  std::string film_base_file;                  // empty: auto, or scan
  bool film_base_scan = false;                 // --film_base scan: read from the scan, whatever <output>.json holds
  bool film_base_roi_given = false;
  int film_base_roi[4] = {0, 0, 0, 0};

  // Output
  std::string output;                          // -o
  std::string colorspace;                      // -P, empty when not given
  // -P statusm, rp180 or apd: the output is the negative's density above the film base in that measure, not a
  // picture. -o's extension gives the container: .dpx (10 bits, code = 95 + 500 D) or .tif / .tiff (16 bits,
  // code = 1520 + 8000 D).
  bool density = false;
  DensityMeasure density_measure = kDensityStatusM;
  bool output_dpx = false;
  // --print: the print materials (ids of the built-in ones, "all", or the path of a spektrafilm print profile): with a
  // picture output the first is the print the conversion renders; with a density output a CLF is written beside the
  // file for each.
  std::vector<std::string> prints;
  bool no_metadata = false;
};

// Parses and checks the command line into o: 0 to convert, 1 on an error (reported), -1 when there is nothing more to
// do (--list_profiles, which it prints after loading each built-in bundle; one that cannot be read is an error).
int parse_options(int ac, char* av[], Options& o);

// ====================================================================================================================
// Output: declarations
// ====================================================================================================================

// neg_process's output: the 16-bit TIFF with its ICC profile, and the JSON sidecar beside it.

// Writes proc's image (or rgb3, 3 samples a pixel, in its place) as a 16-bit TIFF to output with the ICC profile
// attach_profile names ("srgb", "srgb-g10" or a file) or, when that is empty, embedded_icc. 0 on success.
int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc = {}, const uint16_t* rgb3 = nullptr);

// What the sidecar records.
// An input's size and CRC-32 (file_fingerprint), kept so a run hashes each input once: the sidecar loader's check and
// the sidecar writer share it.
// The tone stage's types (tone_stage.h), which the sidecar refers to.
struct ToneCurve;
struct ToneTargets;
struct ToneClip;
struct TonePlanCurve;

struct FileFingerprint {
  bool ok = false;
  uintmax_t size = 0;
  uint32_t crc = 0;
};

struct Sidecar {
  std::string output_path;               // the TIFF; the sidecar is <output_path>.json
  std::string search_renderer = "cpu";   // where the search's passes ran; the auto block records it when not the CPU
  int argc = 0;
  char** argv = nullptr;
  const LibRaw* proc = nullptr;          // the camera, ISO and shutter (nullptr: none recorded)
  std::vector<std::string> files;
  // Each file's fingerprint when this run already took it (the sidecar loader); null or short: taken here.
  const std::vector<FileFingerprint>* file_fingerprints = nullptr;
  int orig_w = 0, orig_h = 0;
  bool half_size = false;
  bool has_roi = false;
  const int* roi = nullptr;              // x, y, w, h when has_roi
  std::string roi_source;
  const FrameDetection* frame_det = nullptr;  // --roi auto's detection
  std::string film_format;               // the format the base's detection looked for, given or not; or empty
  int rot_cw = 0;
  bool hflip = false, vflip = false;
  int out_w = 0, out_h = 0;
  // The film base: source "rebate" or "statistics" (read from the scan: film_base_detection), "bundle" or "none".
  bool film_base_applied = false;
  std::vector<double> film_base_rgb;
  std::vector<int> profile_film_base_rgb;
  std::string film_base_source;
  // A print on a base that is not a capture (Scan::print_floor_*): the base as found, its source and the factor.
  bool print_floor_applied = false;
  double print_floor_found_rgb[3] = {0, 0, 0};
  std::string print_floor_found_source;
  float print_floor_k = 1.0f;
  double knob_base_a = NAN;   // a print's A before the knobs (knobs.base.a)
  // The frame the floor read (the picture's): "roi" (the given --roi, or --roi auto's crop), "detected" (no --roi: the
  // frame detector's crop, at or above kBaseMinFrameConf), "inset_crop" (no --roi and no confident detection: the RAW's
  // inset crop, the search's frame) or "whole_scan" (the same with no inset crop, or a --roi auto that fell back to the
  // whole image). The detection's confidence only where the detection decided the
  // frame (a --roi auto that fell back; no --roi, with a detection that maps onto the scan), else -1.
  std::string print_floor_frame;
  float print_floor_frame_conf = -1.0f;
  const negicc::FilmBaseDetection* film_base_detection = nullptr;
  // The white-light decode's gains on the camera channels before the camera matrix; null: not a white-light decode.
  const float* balance_gains = nullptr;
  std::string profile_path;              // empty: no profile
  bool is_bundle = false;
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  InputSpace input_space = InputSpace::kCameraRawRgb;
  const float* merged_matrix = nullptr;
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;  // absolute, on the crosstalk (matrix.*)
  const float* station_params = nullptr;  // with a bundle: E, g, b as -E, --gain_g, --gain_b take them
  unsigned pin_mask = 0;
  float knee = 0.95f;
  ShoulderInfo shoulder;
  const DinoSolveResult* dino = nullptr; // the single-profile search, when it ran
  std::string dino_model, dino_backend, dino_arg;
  std::string dino_backend_arg;   // --dino_backend when given other than auto (where the search's passes run)
  bool dino_iters_given = false;
  int dino_iters = 4;
  const ParamSearchResult* auto_res = nullptr;  // the bundle search, when it ran
  const int* auto_frame_half = nullptr;
  std::string cached_auto_json_str;      // preserved "auto" JSON when reusing a sidecar
  std::string shadow_json_str;           // the search's shadow alternative and the choice (auto.shadow_alternative)
  std::string cached_dino_json_str;      // preserved "dino" JSON when reusing a sidecar
  std::string cached_intent_field_json_str;  // preserved "intent_field" JSON when reusing a sidecar
  std::string cached_input_curves_json_str;  // preserved "input_curves" JSON when reusing a sidecar
  std::string cached_print_search_json_str;  // preserved "profile.print.search" JSON when reusing a sidecar
  double cached_dino_views_ms = 0.0, cached_dino_render_ms = 0.0;
  double cached_dino_infer_ms = 0.0, cached_dino_search_ms = 0.0;
  bool has_knobs = false;
  float brightness = 0.0f, tint = 0.0f, warmth = 0.0f;
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  std::string colorspace;
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;
  double t_auto_ms = 0.0;
  double t_analysis_ms = 0.0;  // Scan::t_analysis_ms
  // Fitted G/B input curves of the conversion's bracket (input_curves.h): source "fit" (--fit_curves on, with
  // curve_fit), "auto" (likewise) or "cli" (recorded curves given to --fit_curves); empty: none.
  std::string input_curves_source;
  const negicc::InputCurves* input_curves = nullptr;
  const WhiteLightFitResult* curve_fit = nullptr;
  // The models the search ran with: the intent adapter, the neutral head and the tone heads ("[built-in] <name>" or
  // the path --dino gave, with the file's size and CRC-32; empty: not loaded). A run that reused a previous sidecar
  // keeps that run's record (cached_heads_json_str).
  std::string head_path[3];
  FileFingerprint head_fp[3];
  std::string cached_heads_json_str;
  // The tone curve the conversion rendered with (curve_fit's, or replayed; null: none), whether it was applied to the
  // final image, and the clipping it measured.
  // A replay ("knobs" or "cli", else empty).
  std::string replay_source;
  const ToneCurve* tone_curve = nullptr;
  const ToneTargets* tone_targets = nullptr;  // this run's plan, as applied: its targets and highlight share
  // This run's tone stage, when it was planned (after the curve fit, or after the search where no fit runs): the
  // heads' plan and the curve the conversion solved from it.
  const ToneStagePlan* tone_stage_plan = nullptr;
  const TonePlanCurve* tone_plan = nullptr;
  double tone_share = 1.0;
  double tone_shadow_share = 1.0;
  // The rule that chose the curve where no share of the moves held both ends ("shadows_first", "highlights_first" or
  // "least_total"); null where a share held.
  const char* tone_fallback = nullptr;
  bool tone_applied = false;
  const ToneClip* tone_clip = nullptr;
  // The curves the cli_equivalent replays when they differ from input_curves (re-anchored at the knobs' E, g and b);
  // null: input_curves.
  const negicc::InputCurves* cli_input_curves = nullptr;
  // A film base read from a capture (--film_base; film_base_source "capture", or "sidecar" when a previous run's is
  // taken): the capture's record as JSON (film_base.capture). Empty: none.
  std::string film_base_capture_json;
  // The density output's record (the print declarations above, schema negicc.density/1) as JSON, also in the file.
  // Empty: none.
  std::string density_json;
  std::string used_json;                 // what the conversion used (conversion_used in the pipeline code below)
  bool density_fit_off = false;      // a white-light density run without --print, with --fit_curves off: its replay
                                     // keeps it
  // A density run's --print materials as given (built-in ids, 'all', or files): its replay gives them again, so the
  // base's floor (raised only with --print) and the CLFs are made again. Empty: no --print.
  std::vector<std::string> density_prints;
  std::string print_id, print_name;  // --print with a picture output: the print ICC that replaced the chart profiles
  std::string print_file, print_file_sha256;  // a material given as a file: its path as given and its SHA-256
  bool print_from_bundle = false;
  // The print's mid grey A (NaN without a print), the chosen bracket's A_mid, and the search's A0 (the A at which
  // the frame's log-average cell prints at L* 50).
  double print_a = NAN, print_bracket_a_mid = NAN, print_a0 = NAN;
  double print_tilt[3] = {NAN, NAN, NAN};   // the brackets' tilt at the A the print used (print_tilt_at)
  // The paper's white the print is relative to (PCS D50).
  double print_paper_xyz[3] = {negicc::kPcsWhite[0], negicc::kPcsWhite[1], negicc::kPcsWhite[2]};
  bool print_searched = false, print_converged = false;
  int print_steps = 0;
  float print_target_l = 0.0f, print_above_knee = 0.0f;
  // ContinuousPrintResult's shares: past_knee and past_cut at E (the matrix at E alone), above_white and at_white at
  // the result.
  float print_past_knee[3] = {0, 0, 0}, print_past_cut[3] = {0, 0, 0};
  float print_above_white[3] = {0, 0, 0}, print_at_white[3] = {0, 0, 0};
  bool print_a0_at_end = false;
  float print_last_move[3] = {0, 0, 0};
  std::string print_stopped_by;
};

// Writes <output_path>.json: what neg_process read and did (schema negicc.neg_process.sidecar/1).
// The build's version (git describe), as the sidecar's tool.version records it.
const char* neg_process_version();

// A file's size and CRC-32 (zlib), as the sidecar records each input (inputs.raw_file_stats). False when unreadable.
bool file_fingerprint(const std::string& path, uintmax_t& size, uint32_t& crc);
// The same for bytes in memory (a built-in model).
FileFingerprint memory_fingerprint(const void* data, size_t size);

// s with its bytes that are not UTF-8 written as U+FFFD, as the sidecar's writer writes them (a camera model or a
// path need not be UTF-8). The log shows a string as the sidecar records it, and a previous run's sidecar is compared
// with this run's strings as they would be recorded. Valid UTF-8 comes back unchanged.
std::string utf8_replaced(const std::string& s);

bool write_json_sidecar(const Sidecar& sc);

// ====================================================================================================================
// Pipeline: declarations
// ====================================================================================================================

// neg_process's processing pipeline: the stages main runs in turn, the state they hand on, and the split between the
// scan as data (MosaicInput, RgbInput) and the compute on it (Backend: the CPU's, and a CUDA device's for the
// full-size steps).
//
//   prepare   the profile (bundle, ICC profile or bracket), the parameter search mode, the crosstalk correction, the
//             DINOv3 engine
//   decode    the input as data: the zero-copy CFA path (a Sony Bayer RAW or a pixel shift: the mosaic, black
//             subtracted in place, with a half-size view over it) or LibRaw's decode, a TIFF, a JPEG or three
//             captures (an RGB image); the frame (--roi, --roi auto); and the bundle's parameter search on the
//             half-size view, as each path has it
//   begin_conversion
//             the film base the sidecar records, the pins (-E, --gain_g, --gain_b) and the input matrix to start from
//   render    the full-resolution image of the frame: the backend debayers or merges a mosaic; an RGB image is one
//             already
//   solve     the conversion parameters the bundle search did not give: the bracket, E, g and b (an ICC profile's
//             single-profile search, or a bundle's four given values), the knee and the shoulder
//   convert   the backend converts the image in place to the output encoding
//   write     the TIFF and the sidecar

// Cached Auto / DINOv3 metadata loaded from <output>.json when solving grey-point knobs.
struct CachedSidecar {
  bool is_auto = false;                  // true: cached from "auto"; false: cached from "dino"
  int target_idx = -1;
  std::string target_name;
  float base_e = 1.0f, base_g = 1.0f, base_b = 1.0f;
  double print_a = NAN;                  // a print's mid grey A (profile.print.a); NaN: none
  std::string print_search_json_str;     // the print's search that found it (profile.print.search); empty: none
  std::string auto_json_str;             // raw "auto" JSON object from previous sidecar
  std::string dino_json_str;             // raw "dino" JSON object from previous sidecar
  std::string dino_backend;
  double t_dino_views_ms = 0.0, t_dino_render_ms = 0.0;
  double t_dino_infer_ms = 0.0, t_dino_search_ms = 0.0;
  // The previous run's fitted G/B input curves (replayed on its bracket, not refitted) and its intent field, carried
  // into the new sidecar so the knobs' output still records them.
  bool has_curves = false;
  negicc::InputCurves curves;
  std::string curves_source;
  std::string input_curves_json_str;
  std::string intent_field_json_str;
  std::string heads_json_str;
  // Its tone curve (tone_stage.curve), when it applied one.
  bool has_tone = false;
  ToneCurve tone_curve;
};

// What the profile and the options decide before the scan is read.
struct Setup {
  // The renderer the search's passes run through: the GPU's, or --check_renderer's comparison with it (null: the CPU's,
  // or --check_renderer's pass-through renderer without a device). Made only where a search runs.
  std::unique_ptr<PreviewRenderer> renderer;
  // Each input's size and CRC-32, taken once (by the sidecar loader when the knobs look for <output>.json) and
  // reused by the sidecar writer.
  std::vector<FileFingerprint> input_fingerprints;
  bool is_bundle = false;
  ProfileBundle bundle;
  std::string film_prof_arg;             // -p naming an ICC profile (not a bundle)
  bool has_profile = false;
  int manual_target_idx = -1;            // --target, -1 when not given
  bool exposure_given = false;           // -E: the exposure is not searched
  bool use_dino = false;                 // a parameter search runs
  bool bundle_search = false;            // the bundle's search (bracket, exposure and gains)
  // The bundle's search with --target, -E, --gain_g or --gain_b held (either light): Auto with those pinned.
  bool pinned_search = false;
  bool fit_curves_active = false;        // the curve fit runs (--fit_curves auto or on, where it applies)
  bool tone_active = false;              // the tone stage runs (--tone_stage auto or on, where it applies)
  bool has_cached_sidecar = false;       // <output>.json matched and supplies the base parameters
  CachedSidecar cached_sidecar;
  bool white_light = false;              // a rec2020_linear (d65) bundle
  bool density = false;                  // -P statusm, rp180 or apd: the output is density (the print declarations
                                         // above)
  // --print with a picture output on a profile bundle: the print material, and each bracket's print profile
  // (bracket_print_profiles in the print declarations above, in bundle.targets' order). Auto searches the print itself
  // (search_print_continuous); the conversion renders the print at its mid grey A (print_a: the search's, or the
  // reused run's). Empty without.
  std::string print_id, print_name;
  std::string print_file, print_file_sha256;  // a material given as a file: the path as given and the file's SHA-256
  bool print_from_bundle = false;        // the print ICC's table is the bundle's (else computed by neg_process)
  std::vector<ParsedFilmProfile> print_profs;
  std::vector<std::vector<uint8_t>> print_iccs;
  PrintContinuum print_cont;          // the print at a continuous mid grey (the print declarations above)
  ContinuousPrint print_in;           // the search's view of it
  double print_a = NAN;               // the print's mid grey A: the search's or a reused run's (NaN until then)
  // The lightest node of the print's table: the paper's white.
  double print_paper_xyz[3] = {negicc::kPcsWhite[0], negicc::kPcsWhite[1], negicc::kPcsWhite[2]};
  ContinuousPrintResult print_res;    // the print's search
  std::map<std::string, DensityRecord::PrintSearch> density_print_search;   // a density run's --print, per material
  bool print_searched = false;
  bool has_crosstalk = false;
  bool camera_rgb_input = false;         // a TIFF of linear camera RGB (a TIFF with a camera_raw_rgb bundle)
  RawDecode raw_decode = RawDecode::kRec2020;
  std::vector<float> r_coeff, g_coeff, b_coeff;  // the crosstalk correction's rows
  // Where each row comes from: "bundle", "given" (-r, -g, -b) or "identity" (three captures, one per light; an ICC
  // profile's row not given; a bundle's correction off); empty: no correction.
  std::string crosstalk_row_source[3];
  // Why a camera_raw_rgb bundle's correction is off (decode: the scan states a camera the bundle does not list); empty
  // when it is applied.
  std::string crosstalk_off;
  negicc::FlatFieldPtr flat_field;
  DinoV3Engine engine;
  std::string dino_path;                 // the backbone that runs ([built-in] ...)
  std::string neutral_head;              // the neutral-patch head that runs ([built-in] ...), empty when none
  // The intent adapter, neutral head and tone heads that run ([built-in] <name> or the --dino path) and their files'
  // size and CRC-32, for the sidecar.
  std::string head_path[3];
  FileFingerprint head_fp[3];
  ToneHead tone_head;                    // the tone heads (where the tone stage runs)
  // The tone curve the conversion renders with: this run's tone stage's, a previous run's (the knobs on its sidecar) or
  // given (--tone_stage curve:...); not predicted again on a replay.
  // A replay: "knobs" (the knobs on a previous run's sidecar) or "cli" (recorded values given); empty otherwise.
  std::string replay_source;
  bool has_tone = false;
  ToneCurve tone_curve;
  // Fitted G/B input curves the conversion's bracket renders through (input_curves.h): source "fit" (--fit_curves
  // on: curve_fit has the fit), "auto" (--fit_curves auto, likewise), "cli" (recorded curves given to --fit_curves) or
  // empty (none).
  std::string input_curves_source;
  negicc::InputCurves input_curves;
  WhiteLightFitResult curve_fit;
  // This run's tone stage: the heads' plan (the curve fit's on its result, or plan_auto_tone_stage's on the search's
  // own result where no fit runs) and the curve solved from it (when tone_stage.planned). tone_stage_ran: a plan was
  // attempted this run (the sidecar then says what came of it).
  ToneStagePlan tone_stage;
  bool tone_stage_ran = false;
  TonePlanCurve tone_plan;
  // The bracket the curves were composed into (-1: none) and its profile before that (the un-curved input tables),
  // from which the knobs compose the curves again at the E, g and b they move to (reanchor_input_curves).
  int curves_base_idx = -1;
  ParsedFilmProfile curves_base_prof;
};

// The scan as data for a backend: a CFA mosaic (one, or the four frames of a pixel shift), black subtracted, in place.
struct MosaicInput {
  const uint16_t* frames[4] = {nullptr, nullptr, nullptr, nullptr};
  int n_frames = 0;                      // 1, or 4 for a pixel shift
  size_t pitch = 0;                      // uint16_t elements per row
  int width = 0, height = 0;             // the visible sensor area
  unsigned filters = 0;                  // LibRaw's CFA pattern
  float white_level = 65535.0f;          // the clipping level above black
  negicc::FlatSampleArgs flat;           // the flat field at full size (empty without one)
  const negicc::FlatTables* flat_tables = nullptr;   // the tables flat points into (for a copy on a device)
};

// The scan as data for a backend: an RGB image, `channels` samples a pixel (3, or LibRaw's 4).
struct RgbInput {
  uint16_t* pixels = nullptr;
  int width = 0, height = 0, channels = 3;
};

// A region of the sensor (x, y, w, h) and its orientation: the frame to render.
struct Region {
  int x = 0, y = 0, w = 0, h = 0;
  int rot_cw = 0;
  bool hflip = false, vflip = false;
};

// The output encoding and its profiles.
struct OutputSpec {
  OutputEncoding encoding = ENCODE_NONE;
  std::string attach_profile;            // "srgb", "srgb-g10", an ICC file, or the -p profile
  cmsHPROFILE xyz = nullptr;             // ENCODE_CUSTOM: the XYZ input and the output profile
  cmsHPROFILE custom = nullptr;
};
// The density output's conversion (the print declarations above): the measure's matrix, the encoding and the clamped
// pixels.
struct DensityOutput {
  double base_level = 1.0;               // what the film base lands on after the film base's merged matrix
  float merged[9] = {0};                 // that matrix (crosstalk correction x film base scale), E = g = b = 1
  DensityEncoding encoding;
  DensityClip clip;
};

// The compute on the scan's data.
class Backend {
 public:
  virtual ~Backend() = default;
  // The full-resolution RGB of the region of a mosaic, turned and flipped, into rgb_out (3 samples a pixel,
  // dst_w x dst_h: w x h, or h x w when turned by 90 or 270 degrees): the grain-aware debayer of one frame, or the
  // merge of a pixel shift's four, with the flat field. The backend sizes rgb_out, at the point where it first needs
  // it. frame_done(k) frees in.frames[k]: a backend that copies a frame elsewhere calls it as soon as that frame is
  // copied and reads it no more, so that the two copies are not held longer than the copying takes; the caller calls
  // it for every frame after the render in any case.
  virtual void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h,
                      std::vector<uint16_t>& rgb_out, const std::function<void(int)>& frame_done) = 0;
  // Converts img in place to the output encoding through the shared pixel pipeline (color_pipeline.h) of pc: the
  // input matrix and knee, the film profile (tone curves and cLUT) and, for sRGB, the shoulder. ENCODE_NONE stops
  // after the knee; ENCODE_CUSTOM hands the profile's PCS XYZ to out.custom.
  virtual void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                       const RgbInput& img) = 0;
  // Where the last render and the last convert ran: "cpu" or "gpu" (the record's "full_size").
  const char* render_ran = "cpu";
  const char* convert_ran = "cpu";
};

class CpuBackend : public Backend {
 public:
  void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, std::vector<uint16_t>& rgb_out,
              const std::function<void(int)>& frame_done) override;
  void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
               const RgbInput& img) override;
};

// The same on a CUDA device: the debayer or the merge by gpu_grain_aware_debayer and gpu_pixel_shift_merge, and the
// conversion to sRGB or linear sRGB by the search's GPU render (GpuRenderer). The mosaics, the flat field's tables and
// the picture are copied to the device and the result copied back.
//
// Against the CPU's picture: the debayer and the merge are equal in every value. The conversion computes the same
// convert_pixel, but the device rounds powf, tanhf and cbrtf differently in the last place (DESIGN.md 4.2), so a
// value of the exported picture can differ from the CPU's by one 16-bit code.
//
// Another output encoding, or an image that is not three samples a pixel, keeps the CPU's conversion (a custom
// profile's last step is Little CMS), and so does every step when no device runs this build's kernels: the device is
// asked for only where a step is about to use it.
class GpuBackend : public CpuBackend {
 public:
  void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, std::vector<uint16_t>& rgb_out,
              const std::function<void(int)>& frame_done) override;
  void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
               const RgbInput& img) override;
};

// The backend of the full-size steps: the device's (which is the CPU's without a device that runs this build's
// kernels), unless --dino_backend cpu, which keeps everything on the CPU, or --check_renderer, whose picture is the
// CPU's. The same rule as the search's passes.
std::unique_ptr<Backend> make_backend(const Options& o);

struct LibRawDeleter {
  void operator()(LibRaw* p) const;
};
using LibRawPtr = std::unique_ptr<LibRaw, LibRawDeleter>;

// The search's shadow alternative (ParamSearchResult::shadow) through the stages after the search: one candidate's
// state from them (the curve fit, the tone plan and the roll-off), with its bracket's profile before (prof0) and after
// (prof) the fit composed curves into it.
struct ShadowCandidate {
  ParamSearchResult auto_res;
  bool has_curve_params = false;
  ConversionParams curve_params;
  bool has_rolloff = false;
  float rolloff_x_hi = 1.0f, rolloff_q995 = 0.0f;
  WhiteLightFitResult curve_fit;
  ToneStagePlan tone_stage;
  bool tone_stage_ran = false;
  TonePlanCurve tone_plan;
  bool has_tone = false;
  ToneCurve tone_curve;
  negicc::InputCurves input_curves;
  std::string input_curves_source;
  int curves_base_idx = -1;
  ParsedFilmProfile curves_base_prof;
  // The knobs' state (--brightness, --tint, --warmth): their solve on this candidate, and the curves re-anchored at it.
  bool has_knob_res = false;
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  bool has_knob_curves = false;
  negicc::InputCurves knob_curves;
  double knob_base_a = NAN;
  bool has_plain = false;  // and the conversion without them (Scan::has_plain)
  float plain_x_hi = 1.0f, plain_q995 = 0.0f;
  bool has_plain_prof = false;
  ParsedFilmProfile plain_prof;
  int prof_idx = -1;
  ParsedFilmProfile prof0, prof;
  ParsedFilmProfile result_prof0;  // in the alternative's: the result's bracket's profile before the stages
};

// The tone stage's clip check of the kept candidate, made in solve; convert takes it instead of checking again.
struct ToneCheck {
  bool valid = false;
  double white_before = 0, black_before = 0;
  ToneCurve curve;
  ToneTargets targets;
  double share = 1.0, shadow_share = 1.0;
  bool fell_back = false;  // no share held both ends (Options::tone_fallback chose the curve)
  // Made on the conversion without the knobs: convert takes its curve and counts the shares before the stage on the
  // knobbed image it converts, as a replay does.
  bool without_knobs = false;
};

// The choice between the search's result and its shadow alternative (solve). The shares are of the tone stage's clip
// check sample: white before, black before, white after and black after the stage.
struct ShadowChoice {
  bool checked = false;            // both went through the clip check
  bool kept_alternative = false;   // the alternative was taken
  double result[4] = {0, 0, 0, 0}, alternative[4] = {0, 0, 0, 0};
  double stages_ms = 0.0;          // the alternative's stages after the search
  double check_ms = 0.0;           // the alternative's solve and both candidates' clip checks
  const char* not_checked = nullptr;  // why an alternative was not checked
};

// The scan through the pipeline.
struct Scan {
  LibRawPtr proc;                        // the RAW (its metadata, and its image on the LibRaw path)
  LibRawPtr ps_extra[3];                 // a pixel shift's other three frames
  bool cfa = false;                      // the zero-copy CFA path
  MosaicInput mosaic;                    // on the CFA path
  Region cfa_region;                     // on the CFA path: the frame to render (sensor pixels)
  std::shared_ptr<const negicc::FlatTables> flat_full_tables;  // keeps mosaic.flat's tables alive
  std::vector<uint16_t> rgb3;            // on the CFA path: the rendered frame, 3 samples a pixel
  int orig_w = 0, orig_h = 0;            // the sensor size the sidecar records
  std::string input_source = "raw";      // what was read: "raw", "tiff:icc", "jpeg:srgb", "three:raw,raw,raw", ...
  // The camera make and model of the first input as LibRaw reads them (empty: an RGB TIFF or JPEG, or a RAW that
  // states none). camera_compared: a camera_raw_rgb bundle with a camera_models list and a stated model, compared
  // (decode); camera_listed: the list holds the model. Not compared: a white-light bundle, a bundle without the list,
  // or an input that states no model.
  std::string camera_make, camera_model;
  bool camera_compared = false, camera_listed = false;
  // The first input's inset crop (inset_crop), which a single RAW, or three captures, is cut to without --roi or
  // --no_crop.
  bool has_inset = false;
  int inset[4] = {0, 0, 0, 0};

  FrameDetection frame_det;              // --roi auto
  // The format the frame detection that placed the film base's patch looked for (FILM_FORMAT_AUTO: every format, or
  // no detection ran); cli_equivalent gives it as --film_format whenever the detection used one, given or --roi auto's,
  // so that a replay with the explicit --roi reads the same base.
  FilmFormat base_det_format = FILM_FORMAT_AUTO;
  int roi[4] = {0, 0, 0, 0};             // the frame in sensor pixels (--roi, or --roi auto's)
  bool has_roi = false;
  int rot_cw = 0;
  bool hflip = false, vflip = false;

  // The film base read from the scan (film_base_detector.h) for a bundle conversion, as negicc-station's Auto reads it
  // when no film base is given; not found: the bundle's.
  negicc::FilmBaseDetection film_base;
  const double* film_base_rgb() const { return film_base.found ? film_base.rgb : nullptr; }
  // The film base given in place of the scan's own (--film_base): read from a capture of unexposed film
  // (the film base declarations above; film_base.source "capture"), or, with --film_base auto, the base a previous run
  // read from a capture and kept in <output>.json (source "sidecar", with that run's capture record as JSON).
  // has_given_base: film_base holds it and the scan's is not read.
  bool has_given_base = false;
  negicc::FilmBaseDetection given_base;
  // A print on a base that is not a capture (print_floor_factor): the base as found, its source ("rebate",
  // "statistics" or "bundle") and the factor it was raised by (film_base holds the raised base; 1: not raised).
  bool print_floor_applied = false;
  double print_floor_found_rgb[3] = {0, 0, 0};
  std::string print_floor_found_source;
  float print_floor_k = 1.0f;
  // The frame the floor read (the picture's): "roi" (the given --roi, or --roi auto's crop), "detected" (no --roi: the
  // frame detector's crop, at or above kBaseMinFrameConf), "inset_crop" (no --roi and no confident detection: the RAW's
  // inset crop, the search's frame) or "whole_scan" (the same with no inset crop, or a --roi auto that fell back to the
  // whole image). The detection's confidence only where the detection decided the
  // frame (a --roi auto that fell back; no --roi, with a detection that maps onto the scan), else -1.
  std::string print_floor_frame;
  float print_floor_frame_conf = -1.0f;
  double knob_base_a = NAN;   // a print's A before the knobs (recorded as knobs.base.a)
  bool has_base_capture = false;
  FilmBaseCapture base_capture;
  std::string base_capture_json;         // the capture record for the sidecar (film_base.capture)
  // The white-light decode's gains on the camera channels (ChannelBalance); 1, 1, 1: the mosaic was left as it was.
  float balance_gains[3] = {1, 1, 1};

  bool has_auto_res = false;             // the bundle search ran
  ParamSearchResult auto_res;
  int auto_frame_half[4] = {0, 0, 0, 0}; // the frame (x, y, w, h) in the half-size image the search saw
  bool frame_is_whole = false;           // that frame is the whole half-size image
  // Where the fit runs: the curve fit's re-metered E and fitted g and b on the combined profile, which the
  // conversion takes in place of the search's (auto_res keeps the search's own).
  bool has_curve_params = false;
  ConversionParams curve_params;
  const ConversionParams& params() const { return has_curve_params ? curve_params : auto_res.params; }
  // The curves re-anchored at the knobs' E, g and b, which the bracket renders through after the knobs and the
  // cli_equivalent replays (Setup::input_curves stays the record at the knobs' base, as the sidecar keeps it).
  bool has_knob_curves = false;
  negicc::InputCurves knob_curves;

  bool has_knob_res = false;             // --brightness, --tint or --warmth solved
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  // A fresh run's conversion without the knobs, which its tone stage check is made on (solve): the roll-off at the
  // knobs' base and the bracket's profile before the knobs re-anchored its curves (a print: its A before the knobs,
  // knob_base_a). Set by the knobs where the tone stage runs and no record is replayed.
  bool has_plain = false;
  float plain_x_hi = 1.0f, plain_q995 = 0.0f;
  bool has_plain_prof = false;
  ParsedFilmProfile plain_prof;

  // The highlight roll-off of every bundle conversion, measured on the half-size frame as the bracket search measures
  // it: Auto's result and given values in decode, while that frame exists (render frees the mosaic; LibRaw's
  // half-size decode is analysed and gone).
  bool has_rolloff = false;
  float rolloff_x_hi = 1.0f, rolloff_q995 = 0.0f;
  // The tone stage as convert applied it to the final image (Setup::tone_stage has the plan).
  bool tone_applied = false;
  ToneClip tone_clip;
  ToneCurve tone_curve;            // the curve applied (the plan's, capped on this image; or the recorded one)
  ToneTargets tone_targets;        // with this run's plan: the targets it was solved for
  double tone_share = 1.0;         // and the share of the highlight and range moves kept
  double tone_shadow_share = 1.0;  // and of the shadow move
  bool tone_fell_back = false;     // and whether no share held both ends (Options::tone_fallback chose the curve)
  // The search's shadow alternative (ParamSearchResult::shadow, as the search returned it), the alternative after its
  // stages (null: none, or already chosen in solve), the choice, and the kept candidate's clip check.
  ShadowAlternative shadow_search;
  std::shared_ptr<ShadowCandidate> shadow_candidate;
  ShadowChoice shadow_choice;
  ToneCheck tone_check;

  double t_raw_0 = 0.0;                  // timings (ms) the sidecar records
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_auto_ms = 0.0;
  // The bracket analysis on the half-size frame outside Auto's time: the knobs and the roll-off.
  double t_analysis_ms = 0.0;
  double t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;
  // Where the full-size steps ran, "cpu" or "gpu": the debayer or the merge of a mosaic (the CPU's for any other
  // decode), and the conversion to the output encoding.
  const char* full_size_decode = "cpu";
  const char* full_size_conversion = "cpu";

  // The scan's exposure: EXIF (of exposure_file, the first input), else LibRaw's, else 1/8 s at ISO 100 (warned once).
  std::string exposure_file;
  bool has_exif_exp = false;
  double exif_shutter_d = 0.0, exif_iso_d = 0.0;
  bool warned_default_exposure = false;
  void scan_exposure(const LibRaw& r, double& shutter_out, double& iso_out);

  // The full-resolution image the conversion runs on: rgb3 on the CFA path, LibRaw's image otherwise.
  RgbInput image();
};

// The conversion parameters: what solve decides, convert applies and write records.
struct Conversion {
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;
  unsigned pin_mask = 0;
  float merged_matrix[9] = {0};
  float knee = 0.95f;
  ShoulderInfo shoulder;
  ParsedFilmProfile film_prof;
  std::vector<uint8_t> embedded_icc;     // the bracket's ICC profile
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  DinoSolveResult dino_res;              // the single-profile search
  // With a bundle: E, g and b on the station's scale (what -E, --gain_g and --gain_b take), for cli_equivalent.
  std::optional<std::array<float, 3>> station_params;
  // The film base the sidecar records. A bundle conversion takes the base read from the scan (Scan::film_base), else
  // the bundle's, through the station's merged matrix; the film base scale of the input matrix stays 1.
  std::vector<int> profile_film_base_rgb = {1, 1, 1};
  std::vector<int> sidecar_profile_film_base_rgb;
  std::vector<int> sidecar_film_base_rgb = {1, 1, 1};
  std::string film_base_source = "none";
  OutputSpec out;
  DensityOutput density;                 // with Setup::density
};

int prepare(const Options& o, Setup& s);
int decode(const Options& o, Setup& s, Scan& scan);
// After the last pass of the search (after decode, or where --probe_intent stops): logs the GPU's passes and frees its
// frame copy, and fails when --check_renderer counts a pass outside the renderer or its comparison with the GPU is
// outside the tolerance, before anything is written.
int finish_search_passes(const Options& o, Setup& s);
int begin_conversion(const Options& o, const Setup& s, Conversion& c);
int render(const Options& o, const Setup& s, Scan& scan, Backend& backend);
int solve(const Options& o, Setup& s, Scan& scan, Conversion& c);
int convert(const Options& o, const Setup& s, Scan& scan, Conversion& c, Backend& backend);
int write(const Options& o, const Setup& s, Scan& scan, Conversion& c);

// ====================================================================================================================
// Print: code
// The density output's encoding and files, the print materials (built in and spektrafilm files), a frame's print, its
// print CLF and the print ICC profiles: what the print declarations above give.
// ====================================================================================================================

// ---- BEGIN CIE TABLES (written by the companion profiling tools, not in this repository) ----
// The CIE 1931 2 degree standard observer and the relative spectral power of CIE illuminants D50 and D55, on
// 380-780 nm at 1 nm, as colour-science 0.4.7 gives them (D50 and D55 interpolated from the CIE's 5 nm tables).
// Three z_bar values at 650-652 nm are interpolation residue of the order of 1e-22, one of them negative, where the
// CIE's table is 0. Used for the colour of a print. export_cie_header.py of the companion profiling tools writes
// and checks this block from a CSV whose SHA-256 begins b69d080c7b908b1e; nothing in this repository does.
constexpr int kCieSamples = 401;
constexpr double kCieFirstNm = 380.0;   // 1 nm steps

// x_bar
constexpr double kCieXBar[kCieSamples] = {
    0.001368, 0.00150205, 0.001642328, 0.001802382, 0.001995757, 0.002236, 0.002535385, 0.002892603,
    0.003300829, 0.003753236, 0.004243, 0.004762389, 0.005330048, 0.005978712, 0.006741117, 0.00765,
    0.008751373, 0.01002888, 0.0114217, 0.01286901, 0.01431, 0.01570443, 0.01714744, 0.01878122,
    0.02074801, 0.02319, 0.02620736, 0.02978248, 0.03388092, 0.03846824, 0.04351, 0.0489956,
    0.0550226, 0.0617188, 0.069212, 0.07763, 0.08695811, 0.09717672, 0.1084063, 0.1207672,
    0.13438, 0.1493582, 0.1653957, 0.1819831, 0.198611, 0.21477, 0.2301868, 0.2448797,
    0.2587773, 0.2718079, 0.2839, 0.2949438, 0.3048965, 0.3137873, 0.3216454, 0.3285,
    0.3343513, 0.3392101, 0.3431213, 0.3461296, 0.34828, 0.3495999, 0.3501474, 0.350013,
    0.349287, 0.34806, 0.3463733, 0.3442624, 0.3418088, 0.3390941, 0.3362, 0.3331977,
    0.3300411, 0.3266357, 0.3228868, 0.3187, 0.3140251, 0.308884, 0.3032904, 0.2972579,
    0.2908, 0.2839701, 0.2767214, 0.2689178, 0.2604227, 0.2511, 0.2408475, 0.2298512,
    0.2184072, 0.2068115, 0.19536, 0.1842136, 0.1733273, 0.1626881, 0.1522833, 0.1421,
    0.1321786, 0.1225696, 0.1132752, 0.1042979, 0.09564, 0.08729955, 0.07930804, 0.07171776,
    0.06458099, 0.05795001, 0.05186211, 0.04628152, 0.04115088, 0.03641283, 0.03201, 0.0279172,
    0.0241444, 0.020687, 0.0175404, 0.0147, 0.01216179, 0.00991996, 0.00796724, 0.006296346,
    0.0049, 0.003777173, 0.00294532, 0.00242488, 0.002236293, 0.0024, 0.00292552, 0.00383656,
    0.00517484, 0.00698208, 0.0093, 0.01214949, 0.01553588, 0.01947752, 0.02399277, 0.0291,
    0.03481485, 0.04112016, 0.04798504, 0.05537861, 0.06327, 0.07163501, 0.08046224, 0.08973996,
    0.09945645, 0.1096, 0.1201674, 0.1311145, 0.1423679, 0.1538542, 0.1655, 0.1772571,
    0.18914, 0.2011694, 0.2133658, 0.2257499, 0.2383209, 0.2510668, 0.2639922, 0.2771017,
    0.2904, 0.3038912, 0.3175726, 0.3314384, 0.3454828, 0.3597, 0.3740839, 0.3886396,
    0.4033784, 0.4183115, 0.4334499, 0.4487953, 0.464336, 0.480064, 0.4959713, 0.5120501,
    0.5282959, 0.5446916, 0.5612094, 0.5778215, 0.5945, 0.6112209, 0.6279758, 0.6447602,
    0.6615697, 0.6784, 0.6952392, 0.7120586, 0.7288284, 0.7455188, 0.7621, 0.7785432,
    0.7948256, 0.8109264, 0.8268248, 0.8425, 0.8579325, 0.8730816, 0.8878944, 0.9023181,
    0.9163, 0.9297995, 0.9427984, 0.9552776, 0.9672179, 0.9786, 0.9893856, 0.9995488,
    1.0090892, 1.0180064, 1.0263, 1.0339827, 1.040986, 1.047188, 1.0524667, 1.0567,
    1.0597944, 1.0617992, 1.0628068, 1.0629096, 1.0622, 1.0607352, 1.0584436, 1.0552244,
    1.0509768, 1.0456, 1.0390369, 1.0313608, 1.0226662, 1.0130477, 1.0026, 0.9913675,
    0.9793314, 0.9664916, 0.9528479, 0.9384, 0.923194, 0.907244, 0.890502, 0.87292,
    0.8544499, 0.835084, 0.814946, 0.794186, 0.772954, 0.7514, 0.7295836, 0.7075888,
    0.6856022, 0.6638104, 0.6424, 0.6215149, 0.6011138, 0.5811052, 0.5613977, 0.5419,
    0.5225995, 0.5035464, 0.4847436, 0.4661939, 0.4479, 0.4298613, 0.412098, 0.394644,
    0.3775333, 0.3608, 0.3444563, 0.3285168, 0.3130192, 0.2980011, 0.2835, 0.2695448,
    0.2561184, 0.2431896, 0.2307272, 0.2187, 0.2070971, 0.1959232, 0.1851708, 0.1748323,
    0.1649, 0.1553667, 0.14623, 0.13749, 0.1291467, 0.1212, 0.1136397, 0.106465,
    0.09969044, 0.09333061, 0.0874, 0.08190096, 0.07680428, 0.07207712, 0.06768664, 0.0636,
    0.05980685, 0.05628216, 0.05297104, 0.04981861, 0.04677, 0.04378405, 0.04087536, 0.03807264,
    0.03540461, 0.0329, 0.03056419, 0.02838056, 0.02634484, 0.02445275, 0.0227, 0.02108429,
    0.01959988, 0.01823732, 0.01698717, 0.01584, 0.01479064, 0.01383132, 0.01294868, 0.0121292,
    0.01135916, 0.01062935, 0.009938846, 0.009288422, 0.008678854, 0.008110916, 0.007582388, 0.007088746,
    0.006627313, 0.006195408, 0.005790346, 0.005409826, 0.005052583, 0.004717512, 0.004403507, 0.004109457,
    0.003833913, 0.003575748, 0.003334342, 0.003109075, 0.002899327, 0.002704348, 0.00252302, 0.002354168,
    0.002196616, 0.00204919, 0.00191096, 0.001781438, 0.00166011, 0.001546459, 0.001439971, 0.001340042,
    0.001246275, 0.001158471, 0.00107643, 0.0009999493, 0.0009287358, 0.0008624332, 0.0008007503, 0.000743396,
    0.0006900786, 0.0006405156, 0.0005945021, 0.0005518646, 0.000512429, 0.0004760213, 0.0004424536, 0.0004115117,
    0.0003829814, 0.0003566491, 0.0003323011, 0.0003097586, 0.0002888871, 0.0002695394, 0.0002515682, 0.0002348261,
    0.000219171, 0.0002045258, 0.0001908405, 0.0001780654, 0.0001661505, 0.0001550236, 0.0001446219, 0.0001349098,
    0.000125852, 0.000117413, 0.0001095515, 0.0001022245, 9.539445e-05, 8.90239e-05, 8.307527e-05, 7.751269e-05,
    7.231304e-05, 6.745778e-05, 6.292844e-05, 5.870652e-05, 5.477028e-05, 5.109918e-05, 4.767654e-05, 4.448567e-05,
    4.150994e-05};

// y_bar
constexpr double kCieYBar[kCieSamples] = {
    3.9e-05, 4.28264e-05, 4.69146e-05, 5.15896e-05, 5.71764e-05, 6.4e-05, 7.234421e-05, 8.221224e-05,
    9.350816e-05, 0.0001061361, 0.00012, 0.000134984, 0.000151492, 0.000170208, 0.000191816, 0.000217,
    0.0002469067, 0.00028124, 0.00031852, 0.0003572667, 0.000396, 0.0004337147, 0.000473024, 0.000517876,
    0.0005722187, 0.00064, 0.00072456, 0.0008255, 0.00094116, 0.00106988, 0.00121, 0.001362091,
    0.001530752, 0.001720368, 0.001935323, 0.00218, 0.0024548, 0.002764, 0.0031178, 0.0035264,
    0.004, 0.00454624, 0.00515932, 0.00582928, 0.00654616, 0.0073, 0.008086507, 0.00890872,
    0.00976768, 0.01066443, 0.0116, 0.01257317, 0.01358272, 0.01462968, 0.01571509, 0.01684,
    0.01800736, 0.01921448, 0.02045392, 0.02171824, 0.023, 0.02429461, 0.02561024, 0.02695857,
    0.02835125, 0.0298, 0.03131083, 0.03288368, 0.03452112, 0.03622571, 0.038, 0.03984667,
    0.041768, 0.043766, 0.04584267, 0.048, 0.05024368, 0.05257304, 0.05498056, 0.05745872,
    0.06, 0.06260197, 0.06527752, 0.06804208, 0.07091109, 0.0739, 0.077016, 0.0802664,
    0.0836668, 0.0872328, 0.09098, 0.09491755, 0.09904584, 0.1033674, 0.1078846, 0.1126,
    0.117532, 0.1226744, 0.1279928, 0.1334528, 0.13902, 0.1446764, 0.1504693, 0.1564619,
    0.1627177, 0.1693, 0.1762431, 0.1835581, 0.1912735, 0.199418, 0.20802, 0.2171199,
    0.2267345, 0.2368571, 0.2474812, 0.2586, 0.2701849, 0.2822939, 0.2950505, 0.308578,
    0.323, 0.3384021, 0.3546858, 0.3716986, 0.3892875, 0.4073, 0.4256299, 0.4443096,
    0.4633944, 0.4829395, 0.503, 0.5235693, 0.544512, 0.56569, 0.5869653, 0.6082,
    0.6293456, 0.6503068, 0.6708752, 0.6908424, 0.71, 0.7281852, 0.7454636, 0.7619694,
    0.7778368, 0.7932, 0.8081104, 0.8224962, 0.8363068, 0.8494916, 0.862, 0.8738108,
    0.8849624, 0.8954936, 0.9054432, 0.9148501, 0.9237348, 0.9320924, 0.9399226, 0.9472252,
    0.954, 0.9602561, 0.9660074, 0.9712606, 0.9760225, 0.9803, 0.9840924, 0.9874182,
    0.9903128, 0.9928116, 0.9949501, 0.9967108, 0.9980983, 0.999112, 0.9997482, 1,
    0.9998567, 0.9993046, 0.9983255, 0.9968987, 0.995, 0.9926005, 0.9897426, 0.9864444,
    0.9827241, 0.9786, 0.9740837, 0.9691712, 0.9638568, 0.9581349, 0.952, 0.9454504,
    0.9384992, 0.9311628, 0.9234576, 0.9154, 0.9070064, 0.8982772, 0.8892048, 0.8797816,
    0.87, 0.8598613, 0.849392, 0.838622, 0.8275813, 0.8163, 0.8047947, 0.793082,
    0.781192, 0.7691547, 0.757, 0.7447541, 0.7324224, 0.7200036, 0.7074965, 0.6949,
    0.6822192, 0.6694716, 0.6566744, 0.6438448, 0.631, 0.6181555, 0.6053144, 0.5924756,
    0.5796379, 0.5668, 0.5539611, 0.5411372, 0.5283528, 0.5156323, 0.503, 0.4904688,
    0.4780304, 0.4656776, 0.4534032, 0.4412, 0.42908, 0.417036, 0.405032, 0.393032,
    0.381, 0.3689184, 0.3568272, 0.3447768, 0.3328176, 0.321, 0.3093381, 0.2978504,
    0.2865936, 0.2756245, 0.265, 0.2547632, 0.2448896, 0.2353344, 0.2260528, 0.217,
    0.2081616, 0.1995488, 0.1911552, 0.1829744, 0.175, 0.1672235, 0.1596464, 0.1522776,
    0.1451259, 0.1382, 0.1315003, 0.1250248, 0.1187792, 0.1127691, 0.107, 0.1014762,
    0.09618864, 0.09112296, 0.08626485, 0.0816, 0.07712064, 0.07282552, 0.06871008, 0.06476976,
    0.061, 0.05739621, 0.05395504, 0.05067376, 0.04754965, 0.04458, 0.04175872, 0.03908496,
    0.03656384, 0.03420048, 0.032, 0.02996261, 0.02807664, 0.02632936, 0.02470805, 0.0232,
    0.02180077, 0.02050112, 0.01928108, 0.01812069, 0.017, 0.01590379, 0.01483718, 0.01381068,
    0.01283478, 0.01192, 0.01106831, 0.01027339, 0.009533311, 0.008846157, 0.00821, 0.007623781,
    0.007085424, 0.006591476, 0.006138485, 0.005723, 0.005343059, 0.004995796, 0.004676404, 0.004380075,
    0.004102, 0.003838453, 0.003589099, 0.003354219, 0.003134093, 0.002929, 0.002738139, 0.002559876,
    0.002393244, 0.002237275, 0.002091, 0.001953587, 0.00182458, 0.00170358, 0.001590187, 0.001484,
    0.001384496, 0.001291268, 0.001204092, 0.001122744, 0.001047, 0.0009765896, 0.0009111088, 0.0008501332,
    0.0007932384, 0.00074, 0.0006900827, 0.00064331, 0.000599496, 0.0005584547, 0.00052, 0.0004839136,
    0.0004500528, 0.0004183452, 0.0003887184, 0.0003611, 0.0003353835, 0.0003114404, 0.0002891656, 0.0002684539,
    0.0002492, 0.0002313019, 0.0002146856, 0.0001992884, 0.0001850475, 0.0001719, 0.0001597781, 0.0001486044,
    0.0001383016, 0.0001287925, 0.00012, 0.0001118595, 0.0001043224, 9.73356e-05, 9.084587e-05, 8.48e-05,
    7.914667e-05, 7.3858e-05, 6.8916e-05, 6.430267e-05, 6e-05, 5.598187e-05, 5.22256e-05, 4.87184e-05,
    4.544747e-05, 4.24e-05, 3.956104e-05, 3.691512e-05, 3.444868e-05, 3.214816e-05, 3e-05, 2.799125e-05,
    2.611356e-05, 2.436024e-05, 2.272461e-05, 2.12e-05, 1.977855e-05, 1.845285e-05, 1.721687e-05, 1.606459e-05,
    1.499e-05};

// z_bar
constexpr double kCieZBar[kCieSamples] = {
    0.006450001, 0.007083216, 0.007745488, 0.008501152, 0.009414544, 0.01054999, 0.0119658, 0.01365587,
    0.01558805, 0.01773015, 0.02005001, 0.02251136, 0.02520288, 0.02827972, 0.03189704, 0.03621,
    0.04143771, 0.04750372, 0.05411988, 0.06099803, 0.06785001, 0.07448632, 0.08136156, 0.08915364,
    0.09854048, 0.1102, 0.1246133, 0.1417017, 0.1613035, 0.1832568, 0.2074, 0.2336921,
    0.2626114, 0.2947746, 0.3307985, 0.3713, 0.4162091, 0.4654642, 0.5196948, 0.5795303,
    0.6456, 0.7184838, 0.7967133, 0.8778459, 0.959439, 1.0390501, 1.1153673, 1.1884971,
    1.2581233, 1.3239296, 1.3856, 1.4426352, 1.4948035, 1.5421903, 1.5848807, 1.62296,
    1.6564048, 1.6852959, 1.7098745, 1.7303821, 1.74706, 1.7600446, 1.7696233, 1.7762637,
    1.7804334, 1.7826, 1.7829682, 1.7816998, 1.7791982, 1.7758671, 1.77211, 1.7682589,
    1.764039, 1.7589438, 1.7524663, 1.7441, 1.7335595, 1.7208581, 1.7059369, 1.6887372,
    1.6692, 1.6475287, 1.6234127, 1.5960223, 1.564528, 1.5281, 1.4861114, 1.4395215,
    1.3898799, 1.3387362, 1.28764, 1.2374223, 1.1878243, 1.1387611, 1.090148, 1.0419,
    0.9941976, 0.9473473, 0.9014531, 0.8566193, 0.8129501, 0.7705173, 0.7294448, 0.6899136,
    0.6521049, 0.6162, 0.5823286, 0.5504162, 0.5203376, 0.4919673, 0.46518, 0.4399246,
    0.4161836, 0.3938822, 0.3729459, 0.3533, 0.3348578, 0.3175521, 0.3013375, 0.2861686,
    0.272, 0.2588171, 0.2464838, 0.2347718, 0.2234533, 0.2123, 0.2011692, 0.1901196,
    0.1792254, 0.1685608, 0.1582, 0.1481383, 0.1383758, 0.1289942, 0.1200751, 0.1117,
    0.1039048, 0.09666748, 0.08998272, 0.08384531, 0.07824999, 0.07320899, 0.06867816, 0.06456784,
    0.06078835, 0.05725001, 0.05390435, 0.05074664, 0.04775276, 0.04489859, 0.04216, 0.03950728,
    0.03693564, 0.03445836, 0.03208872, 0.02984, 0.02771181, 0.02569444, 0.02378716, 0.02198925,
    0.0203, 0.01871805, 0.01724036, 0.01586364, 0.01458461, 0.0134, 0.01230723, 0.01130188,
    0.01037792, 0.009529306, 0.008749999, 0.0080352, 0.0073816, 0.0067854, 0.0062428, 0.005749999,
    0.0053036, 0.0048998, 0.0045342, 0.0042024, 0.0039, 0.0036232, 0.0033706, 0.0031414,
    0.0029348, 0.002749999, 0.0025852, 0.0024386, 0.0023094, 0.0021968, 0.0021, 0.002017733,
    0.0019482, 0.0018898, 0.001840933, 0.0018, 0.001766267, 0.0017378, 0.0017112, 0.001683067,
    0.001650001, 0.001610133, 0.0015644, 0.0015136, 0.001458533, 0.0014, 0.001336667, 0.00127,
    0.001205, 0.001146667, 0.0011, 0.0010688, 0.0010494, 0.0010356, 0.0010212, 0.001,
    0.00096864, 0.00092992, 0.00088688, 0.00084256, 0.0008, 0.00076096, 0.00072368, 0.00068592,
    0.00064544, 0.0006, 0.0005478667, 0.0004916, 0.0004354, 0.0003834667, 0.00034, 0.0003072533,
    0.00028316, 0.00026544, 0.0002518133, 0.00024, 0.0002295467, 0.00022064, 0.00021196, 0.0002021867,
    0.00019, 0.0001742133, 0.00015564, 0.00013596, 0.0001168533, 0.0001, 8.613333e-05, 7.46e-05,
    6.5e-05, 5.693333e-05, 4.999999e-05, 4.416e-05, 3.948e-05, 3.572e-05, 3.264e-05, 3e-05,
    2.765333e-05, 2.556e-05, 2.364e-05, 2.181333e-05, 2e-05, 1.813333e-05, 1.62e-05, 1.42e-05,
    1.213333e-05, 1e-05, 7.733333e-06, 5.4e-06, 3.2e-06, 1.333333e-06, 2.1175824e-22, -3.1763736e-22,
    5.2939559e-23, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0};

// D50
constexpr double kCieD50[kCieSamples] = {
    24.488, 25.0262, 25.5644, 26.1026, 26.6408, 27.179, 27.7174, 28.2558,
    28.7942, 29.3326, 29.871, 31.8146, 33.7582, 35.7018, 37.6454, 39.589,
    41.5328, 43.4766, 45.4204, 47.3642, 49.308, 50.0284, 50.7488, 51.4692,
    52.1896, 52.91, 53.6306, 54.3512, 55.0718, 55.7924, 56.513, 56.865,
    57.217, 57.569, 57.921, 58.273, 58.6252, 58.9774, 59.3296, 59.6818,
    60.034, 59.8124, 59.5908, 59.3692, 59.1476, 58.926, 58.7044, 58.4828,
    58.2612, 58.0396, 57.818, 59.5186, 61.2192, 62.9198, 64.6204, 66.321,
    68.0218, 69.7226, 71.4234, 73.1242, 74.825, 76.0672, 77.3094, 78.5516,
    79.7938, 81.036, 82.2782, 83.5204, 84.7626, 86.0048, 87.247, 87.5836,
    87.9202, 88.2568, 88.5934, 88.93, 89.2664, 89.6028, 89.9392, 90.2756,
    90.612, 90.6876, 90.7632, 90.8388, 90.9144, 90.99, 91.0656, 91.1412,
    91.2168, 91.2924, 91.368, 91.742, 92.116, 92.49, 92.864, 93.238,
    93.6122, 93.9864, 94.3606, 94.7348, 95.109, 94.7944, 94.4798, 94.1652,
    93.8506, 93.536, 93.2214, 92.9068, 92.5922, 92.2776, 91.963, 92.339,
    92.715, 93.091, 93.467, 93.843, 94.2192, 94.5954, 94.9716, 95.3478,
    95.724, 95.813, 95.902, 95.991, 96.08, 96.169, 96.2578, 96.3466,
    96.4354, 96.5242, 96.613, 96.6646, 96.7162, 96.7678, 96.8194, 96.871,
    96.9226, 96.9742, 97.0258, 97.0774, 97.129, 97.626, 98.123, 98.62,
    99.117, 99.614, 100.111, 100.608, 101.105, 101.602, 102.099, 101.965,
    101.83, 101.696, 101.561, 101.427, 101.293, 101.158, 101.024, 100.889,
    100.755, 100.911, 101.067, 101.224, 101.38, 101.536, 101.692, 101.848,
    102.005, 102.161, 102.317, 102.085, 101.854, 101.622, 101.391, 101.159,
    100.927, 100.695, 100.464, 100.232, 100, 99.7736, 99.5472, 99.3208,
    99.0944, 98.868, 98.6414, 98.4148, 98.1882, 97.9616, 97.735, 97.8534,
    97.9718, 98.0902, 98.2086, 98.327, 98.4452, 98.5634, 98.6816, 98.7998,
    98.918, 98.376, 97.834, 97.292, 96.75, 96.208, 95.6662, 95.1244,
    94.5826, 94.0408, 93.499, 93.9178, 94.3366, 94.7554, 95.1742, 95.593,
    96.012, 96.431, 96.85, 97.269, 97.688, 97.846, 98.004, 98.162,
    98.32, 98.478, 98.6362, 98.7944, 98.9526, 99.1108, 99.269, 99.2462,
    99.2234, 99.2006, 99.1778, 99.155, 99.1324, 99.1098, 99.0872, 99.0646,
    99.042, 98.71, 98.378, 98.046, 97.714, 97.382, 97.05, 96.718,
    96.386, 96.054, 95.722, 96.0356, 96.3492, 96.6628, 96.9764, 97.29,
    97.6034, 97.9168, 98.2302, 98.5436, 98.857, 98.538, 98.219, 97.9,
    97.581, 97.262, 96.943, 96.624, 96.305, 95.986, 95.667, 95.9194,
    96.1718, 96.4242, 96.6766, 96.929, 97.1812, 97.4334, 97.6856, 97.9378,
    98.19, 98.6714, 99.1528, 99.6342, 100.116, 100.597, 101.078, 101.559,
    102.041, 102.522, 103.003, 102.616, 102.229, 101.842, 101.455, 101.068,
    100.681, 100.294, 99.907, 99.52, 99.133, 97.9578, 96.7826, 95.6074,
    94.4322, 93.257, 92.0818, 90.9066, 89.7314, 88.5562, 87.381, 87.8032,
    88.2254, 88.6476, 89.0698, 89.492, 89.9144, 90.3368, 90.7592, 91.1816,
    91.604, 91.7324, 91.8608, 91.9892, 92.1176, 92.246, 92.3746, 92.5032,
    92.6318, 92.7604, 92.889, 91.2856, 89.6822, 88.0788, 86.4754, 84.872,
    83.2684, 81.6648, 80.0612, 78.4576, 76.854, 77.8198, 78.7856, 79.7514,
    80.7172, 81.683, 82.6486, 83.6142, 84.5798, 85.5454, 86.511, 87.118,
    87.725, 88.332, 88.939, 89.546, 90.1528, 90.7596, 91.3664, 91.9732,
    92.58, 91.145, 89.71, 88.275, 86.84, 85.405, 83.97, 82.535,
    81.1, 79.665, 78.23, 76.1762, 74.1224, 72.0686, 70.0148, 67.961,
    65.9072, 63.8534, 61.7996, 59.7458, 57.692, 60.215, 62.738, 65.261,
    67.784, 70.307, 72.8302, 75.3534, 77.8766, 80.3998, 82.923, 82.4582,
    81.9934, 81.5286, 81.0638, 80.599, 80.134, 79.669, 79.204, 78.739,
    78.274};

// D55
constexpr double kCieD55[kCieSamples] = {
    32.584, 33.1342, 33.6844, 34.2346, 34.7848, 35.335, 35.8854, 36.4358,
    36.9862, 37.5366, 38.087, 40.3732, 42.6594, 44.9456, 47.2318, 49.518,
    51.8042, 54.0904, 56.3766, 58.6628, 60.949, 61.7094, 62.4698, 63.2302,
    63.9906, 64.751, 65.5116, 66.2722, 67.0328, 67.7934, 68.554, 68.8562,
    69.1584, 69.4606, 69.7628, 70.065, 70.3674, 70.6698, 70.9722, 71.2746,
    71.577, 71.2108, 70.8446, 70.4784, 70.1122, 69.746, 69.3796, 69.0132,
    68.6468, 68.2804, 67.914, 69.6832, 71.4524, 73.2216, 74.9908, 76.76,
    78.529, 80.298, 82.067, 83.836, 85.605, 86.8438, 88.0826, 89.3214,
    90.5602, 91.799, 93.0378, 94.2766, 95.5154, 96.7542, 97.993, 98.24,
    98.487, 98.734, 98.981, 99.228, 99.475, 99.722, 99.969, 100.216,
    100.463, 100.408, 100.353, 100.298, 100.243, 100.188, 100.133, 100.078,
    100.023, 99.968, 99.913, 100.196, 100.478, 100.761, 101.043, 101.326,
    101.609, 101.891, 102.174, 102.456, 102.739, 102.273, 101.807, 101.341,
    100.875, 100.409, 99.9428, 99.4766, 99.0104, 98.5442, 98.078, 98.3382,
    98.5984, 98.8586, 99.1188, 99.379, 99.6392, 99.8994, 100.16, 100.42,
    100.68, 100.682, 100.683, 100.685, 100.686, 100.688, 100.689, 100.691,
    100.692, 100.694, 100.695, 100.624, 100.553, 100.483, 100.412, 100.341,
    100.27, 100.199, 100.129, 100.058, 99.987, 100.409, 100.831, 101.254,
    101.676, 102.098, 102.52, 102.943, 103.365, 103.788, 104.21, 103.999,
    103.788, 103.578, 103.367, 103.156, 102.945, 102.734, 102.524, 102.313,
    102.102, 102.189, 102.275, 102.362, 102.448, 102.535, 102.622, 102.708,
    102.795, 102.881, 102.968, 102.671, 102.374, 102.078, 101.781, 101.484,
    101.187, 100.89, 100.594, 100.297, 100, 99.7216, 99.4432, 99.1648,
    98.8864, 98.608, 98.3296, 98.0512, 97.7728, 97.4944, 97.216, 97.2692,
    97.3224, 97.3756, 97.4288, 97.482, 97.5354, 97.5888, 97.6422, 97.6956,
    97.749, 97.1172, 96.4854, 95.8536, 95.2218, 94.59, 93.9584, 93.3268,
    92.6952, 92.0636, 91.432, 91.7308, 92.0296, 92.3284, 92.6272, 92.926,
    93.2246, 93.5232, 93.8218, 94.1204, 94.419, 94.4912, 94.5634, 94.6356,
    94.7078, 94.78, 94.852, 94.924, 94.996, 95.068, 95.14, 95.048,
    94.956, 94.864, 94.772, 94.68, 94.588, 94.496, 94.404, 94.312,
    94.22, 93.8428, 93.4656, 93.0884, 92.7112, 92.334, 91.9568, 91.5796,
    91.2024, 90.8252, 90.448, 90.6362, 90.8244, 91.0126, 91.2008, 91.389,
    91.5772, 91.7654, 91.9536, 92.1418, 92.33, 91.9824, 91.6348, 91.2872,
    90.9396, 90.592, 90.2444, 89.8968, 89.5492, 89.2016, 88.854, 89.0004,
    89.1468, 89.2932, 89.4396, 89.586, 89.7322, 89.8784, 90.0246, 90.1708,
    90.317, 90.6802, 91.0434, 91.4066, 91.7698, 92.133, 92.4964, 92.8598,
    93.2232, 93.5866, 93.95, 93.5506, 93.1512, 92.7518, 92.3524, 91.953,
    91.5536, 91.1542, 90.7548, 90.3554, 89.956, 88.9282, 87.9004, 86.8726,
    85.8448, 84.817, 83.789, 82.761, 81.733, 80.705, 79.677, 79.9932,
    80.3094, 80.6256, 80.9418, 81.258, 81.5744, 81.8908, 82.2072, 82.5236,
    82.84, 83.0404, 83.2408, 83.4412, 83.6416, 83.842, 84.0424, 84.2428,
    84.4432, 84.6436, 84.844, 83.383, 81.922, 80.461, 79, 77.539,
    76.0782, 74.6174, 73.1566, 71.6958, 70.235, 71.1416, 72.0482, 72.9548,
    73.8614, 74.768, 75.6746, 76.5812, 77.4878, 78.3944, 79.301, 79.8702,
    80.4394, 81.0086, 81.5778, 82.147, 82.7162, 83.2854, 83.8546, 84.4238,
    84.993, 83.6818, 82.3706, 81.0594, 79.7482, 78.437, 77.1256, 75.8142,
    74.5028, 73.1914, 71.88, 69.9714, 68.0628, 66.1542, 64.2456, 62.337,
    60.4282, 58.5194, 56.6106, 54.7018, 52.793, 55.1064, 57.4198, 59.7332,
    62.0466, 64.36, 66.6734, 68.9868, 71.3002, 73.6136, 75.927, 75.516,
    75.105, 74.694, 74.283, 73.872, 73.4612, 73.0504, 72.6396, 72.2288,
    71.818};
// ---- END CIE TABLES ----
// The block above is written by the companion tools with its own grid constants: they are the film model's
// (profile_bundle.h).
static_assert(kCieSamples == kSpectralSamples && kCieFirstNm == kSpectralFirstNm, "the CIE tables' grid");

void density_encode_in_place(uint16_t* pixels, int width, int height, int channels, const float merged[9],
                             double base_level, const double m[9], const DensityEncoding& enc, DensityClip& clip) {
  const double inv_base = 1.0 / base_level;
  const double offset = enc.offset, scale = enc.scale;
  const long max_code = enc.max_code();
  uint64_t low[3] = {0, 0, 0}, high[3] = {0, 0, 0}, floor[3] = {0, 0, 0};
  #pragma omp parallel for schedule(static) reduction(+ : low[:3], high[:3], floor[:3])
  for (int j = 0; j < height; ++j) {
    uint16_t* row = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      uint16_t* px = row + (size_t)i * channels;
      const double r = px[0] / 65535.0, g = px[1] / 65535.0, b = px[2] / 65535.0;
      double d[3];
      for (int c = 0; c < 3; ++c) {
        const double v = (double)merged[c * 3] * r + (double)merged[c * 3 + 1] * g + (double)merged[c * 3 + 2] * b;
        const double t = v * inv_base;
        if (!(t > kDensityMinT)) ++floor[c];
        d[c] = -std::log10(std::max(t, kDensityMinT));
      }
      for (int c = 0; c < 3; ++c) {
        const double dens = m[c * 3] * d[0] + m[c * 3 + 1] * d[1] + m[c * 3 + 2] * d[2];
        long code = std::lround(offset + scale * enc.k[c] * dens);
        if (code < 0) {
          code = 0;
          ++low[c];
        } else if (code > max_code) {
          code = max_code;
          ++high[c];
        }
        px[c] = (uint16_t)code;
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    clip.low[c] = low[c];
    clip.high[c] = high[c];
    clip.floor[c] = floor[c];
  }
  clip.pixels = (uint64_t)width * (uint64_t)height;
}

std::string density_metadata_json(const DensityRecord& rec, int indent) {
  using nlohmann::ordered_json;
  ordered_json j;
  j["schema"] = "negicc.density/1";
  j["density"] = density_measure_name(rec.measure);
  if (rec.matrix && !rec.matrix->target_curves.empty()) j["target_curves"] = rec.matrix->target_curves;
  j["encoding"] = {{"container", rec.container},
                   {"bits", rec.encoding.bits},
                   {"offset", rec.encoding.offset},
                   {"scale", rec.encoding.scale},
                   {"channel_factors", {rec.encoding.k[0], rec.encoding.k[1], rec.encoding.k[2]}},
                   {"standard", rec.measure == kDensityApd
                                    ? "ADX, SMPTE ST 2065-3 (the channel factors applied)"
                                    : (rec.measure == kDensityRp180 ? "Cineon scaling" : "none")},
                   {"formula",
                    "code = offset + scale * k * D, D the density above the film base, k the channel factor"}};
  j["profile_bundle"] = rec.bundle_key;
  j["film_base"] = {{"rgb", {rec.film_base_rgb[0], rec.film_base_rgb[1], rec.film_base_rgb[2]}},
                    {"source", rec.film_base_source}};
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    j["density_matrix"] = {{m[0], m[1], m[2]}, {m[3], m[4], m[5]}, {m[6], m[7], m[8]}};
  }
  if (rec.bundle) {
    if (!rec.bundle->density_fit_json.empty()) {
      j["density_fit"] = ordered_json::parse(rec.bundle->density_fit_json, nullptr, false);
    }
    if (!rec.bundle->light_source_json.empty()) {
      j["light_source"] = ordered_json::parse(rec.bundle->light_source_json, nullptr, false);
      // The bundle's three LED spectra (led_spectra) are left out when they are the all-on spectrum cut at its band
      // splits: the spectrum and the splits, both recorded, give them back. Spectra measured one LED at a time stay.
      ordered_json& light = j["light_source"];
      if (light.is_object() && light.contains("led_spectra") && light["led_spectra"].contains("band_splits_nm")) {
        light.erase("led_spectra");
      }
    }
  }
  if (rec.clip) {
    const DensityClip& c = *rec.clip;
    j["clipped_pixels"] = {{"below_code_0", {c.low[0], c.low[1], c.low[2]}},
                           {"above_max_code", {c.high[0], c.high[1], c.high[2]}},
                           {"at_density_floor", {c.floor[0], c.floor[1], c.floor[2]}},
                           {"pixels", c.pixels}};
  }
  // The picture conversion multiplies the light's green and blue channels by g and b to balance the frame; in
  // density that is an offset per channel. Through the measure's matrix, and with their mean taken out so that the
  // file's overall level stays, the offsets are the colour balance alone. The exposure E is recorded and left to
  // the grade: it is relative to the picture conversion's film profile, which a density file does not use.
  double measure[3], mean = 0.0;
  if (density_balance_offset(rec, measure, &mean)) {
    j["suggested_balance"] = {
        {"from", "the picture conversion's parameter search on this frame; not applied to the densities"},
        {"exposure", rec.balance_e},
        {"gain_g", rec.balance_g},
        {"gain_b", rec.balance_b},
        {"bracket", rec.balance_target},
        {"density_offset", {measure[0], measure[1], measure[2]}},
        {"code_offset",
         {measure[0] * rec.encoding.scale * rec.encoding.k[0], measure[1] * rec.encoding.scale * rec.encoding.k[1],
          measure[2] * rec.encoding.scale * rec.encoding.k[2]}},
        {"density_offset_mean_removed", mean},
        {"use", "D + density_offset (code + code_offset) has the colour balance the picture conversion gives this "
                "frame; the offsets sum to zero, so the overall level is unchanged and the exposure is left to the "
                "grade. The gains also shift the level, by density_offset_mean_removed in every channel; the print "
                "CLF beside the file, which sets the exposure, applies density_offset + density_offset_mean_removed, "
                "except for a print material with its own print search, whose CLF applies the offsets of that "
                "search's gains (print.materials.<id>.balance)"}};
  }
  // How the frame prints on each print material (below), computed for this frame and not applied to it.
  if (!rec.print_json.empty()) j["print"] = ordered_json::parse(rec.print_json, nullptr, false);
  j["tool"] = rec.tool_version;
  // The bundle's key and a print file's name need not be UTF-8: a byte that is not is written as U+FFFD, as the
  // sidecar's writer writes it.
  return j.dump(indent, ' ', false, ordered_json::error_handler_t::replace);
}

bool density_balance_offset(const DensityRecord& rec, double offset[3], double* mean_out) {
  offset[0] = offset[1] = offset[2] = 0.0;
  if (mean_out) *mean_out = 0.0;
  if (!(rec.has_balance && rec.balance_e > 0.0 && rec.balance_g > 0.0 && rec.balance_b > 0.0)) return false;
  const double light[3] = {0.0, -std::log10(rec.balance_g), -std::log10(rec.balance_b)};
  for (int c = 0; c < 3; ++c) offset[c] = light[c];
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    for (int c = 0; c < 3; ++c) offset[c] = m[c * 3] * light[0] + m[c * 3 + 1] * light[1] + m[c * 3 + 2] * light[2];
  }
  const double mean = (offset[0] + offset[1] + offset[2]) / 3.0;
  for (int c = 0; c < 3; ++c) offset[c] -= mean;
  if (mean_out) *mean_out = mean;
  return true;
}

namespace {

// The capture's time as the TIFF and DPX date fields write it, "YYYY:MM:DD HH:MM:SS"; empty when the input records
// none, so that the file is the same on every run (its replay included) and claims no time it does not know, and when
// the time cannot be converted or its year takes more than the field's four digits.
std::string date_time(long long timestamp, char sep) {
  if (timestamp <= 0) return "";
  time_t t = (time_t)timestamp;
  struct tm tm_buf;
  if (!localtime_r(&t, &tm_buf)) return "";
  char buf[80];
  const int n = snprintf(buf, sizeof buf, "%04d:%02d:%02d%c%02d:%02d:%02d", tm_buf.tm_year + 1900, tm_buf.tm_mon + 1,
                         tm_buf.tm_mday, sep, tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
  if (n != (int)sizeof "YYYY:MM:DD HH:MM:SS" - 1) return "";
  return buf;
}

// Ends a write: on a failed one (a full disk, say) the file is removed, unless the output is a device or a pipe, so
// that nothing is left half written.
int finish_write(const std::string& path, bool ok) {
  if (ok) return 0;
  log_error("Failed to write %s\n", path.c_str());
  std::error_code ec;
  if (std::filesystem::is_regular_file(path, ec)) remove(path.c_str());
  return -1;
}

// ---- TIFF (little-endian, one strip) --------------------------------------------------------------------------------
struct TiffEntry {
  uint16_t tag, type;
  uint32_t count, value;
};

void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back((uint8_t)(v & 0xff));
  b.push_back((uint8_t)(v >> 8));
}
void put32(std::vector<uint8_t>& b, uint32_t v) {
  for (int k = 0; k < 4; ++k) b.push_back((uint8_t)((v >> (8 * k)) & 0xff));
}

// ---- DPX (big-endian) -----------------------------------------------------------------------------------------------
void be16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v >> 8);
  p[1] = (uint8_t)(v & 0xff);
}
void be32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)((v >> 16) & 0xff);
  p[2] = (uint8_t)((v >> 8) & 0xff);
  p[3] = (uint8_t)(v & 0xff);
}
void be_float(uint8_t* p, float f) {
  uint32_t v;
  memcpy(&v, &f, 4);
  be32(p, v);
}
void put_str(uint8_t* p, size_t field, const std::string& s) {
  memcpy(p, s.data(), std::min(field - 1, s.size()));
}

}  // namespace

int write_density_tiff(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                       const std::string& description, const DensityFileInfo& info) {
  const uint64_t strip_bytes = (uint64_t)width * (uint64_t)height * 6;
  // The values that do not fit an entry, laid out after the IFD.
  const std::string date = date_time(info.timestamp, ' ');
  std::vector<TiffEntry> entries;
  std::vector<uint8_t> extra;
  const int n_entries = 17 + (date.empty() ? 0 : 1) + (info.make.empty() ? 0 : 1) + (info.model.empty() ? 0 : 1);
  const uint32_t ifd_offset = 8;
  const uint32_t extra_offset = ifd_offset + 2 + (uint32_t)n_entries * 12 + 4;
  auto add_extra = [&](const void* data, size_t n) {
    const uint32_t at = extra_offset + (uint32_t)extra.size();
    const uint8_t* p = (const uint8_t*)data;
    extra.insert(extra.end(), p, p + n);
    if (extra.size() % 2) extra.push_back(0);
    return at;
  };
  // An ASCII value with its terminating NUL; four bytes or fewer are stored in the entry itself (TIFF 6.0).
  auto ascii = [&](uint16_t tag, const std::string& s) {
    const uint32_t n = (uint32_t)s.size() + 1;
    if (n <= 4) {
      uint32_t v = 0;
      for (uint32_t k = 0; k < n; ++k) v |= (uint32_t)(uint8_t)s[k] << (8 * k);   // the bytes in file order
      entries.push_back({tag, 2, n, v});
      return;
    }
    entries.push_back({tag, 2, n, add_extra(s.c_str(), n)});
  };
  const uint16_t bps[3] = {16, 16, 16};
  const uint32_t res[2] = {300, 1};
  entries.push_back({254, 4, 1, 0});                       // NewSubfileType
  entries.push_back({256, 4, 1, (uint32_t)width});         // ImageWidth
  entries.push_back({257, 4, 1, (uint32_t)height});        // ImageLength
  {
    std::vector<uint8_t> b;
    for (uint16_t v : bps) put16(b, v);
    entries.push_back({258, 3, 3, add_extra(b.data(), b.size())});  // BitsPerSample
  }
  entries.push_back({259, 3, 1, 1});                       // Compression: none
  entries.push_back({262, 3, 1, 2});                       // PhotometricInterpretation: RGB
  ascii(270, description);                                 // ImageDescription
  if (!info.make.empty()) ascii(271, info.make);
  if (!info.model.empty()) ascii(272, info.model);
  const size_t strip_entry = entries.size();
  entries.push_back({273, 4, 1, 0});                       // StripOffsets (filled below)
  entries.push_back({274, 3, 1, 1});                       // Orientation: top left
  entries.push_back({277, 3, 1, 3});                       // SamplesPerPixel
  entries.push_back({278, 4, 1, (uint32_t)height});        // RowsPerStrip
  entries.push_back({279, 4, 1, (uint32_t)strip_bytes});   // StripByteCounts
  {
    std::vector<uint8_t> b;
    put32(b, res[0]);
    put32(b, res[1]);
    const uint32_t at = add_extra(b.data(), b.size());
    entries.push_back({282, 5, 1, at});                    // XResolution
    entries.push_back({283, 5, 1, at});                    // YResolution
  }
  entries.push_back({284, 3, 1, 1});                       // PlanarConfiguration: chunky
  entries.push_back({296, 3, 1, 2});                       // ResolutionUnit: inch
  ascii(305, info.software);                               // Software
  if (!date.empty()) ascii(306, date);                     // DateTime: the capture's, when the input records one
  if ((int)entries.size() != n_entries) {
    log_error("Internal error: the TIFF directory has %d entries, expected %d\n", (int)entries.size(), n_entries);
    return -1;
  }
  // A classic TIFF addresses its data with 32-bit offsets: the strip, after the header and the values laid out after
  // the IFD (the description among them), must end within them.
  if ((uint64_t)extra_offset + extra.size() + strip_bytes > 0xffffff00ull) {
    log_error("The image is too large for a TIFF file (%dx%d, %zu bytes of metadata)\n", width, height,
              description.size());
    return -1;
  }
  const uint32_t data_offset = extra_offset + (uint32_t)extra.size();
  entries[strip_entry].value = data_offset;

  std::vector<uint8_t> head;
  head.push_back('I');
  head.push_back('I');
  put16(head, 42);
  put32(head, ifd_offset);
  put16(head, (uint16_t)entries.size());
  for (const TiffEntry& e : entries) {
    put16(head, e.tag);
    put16(head, e.type);
    put32(head, e.count);
    if (e.type == 3 && e.count == 1) {
      put16(head, (uint16_t)e.value);
      put16(head, 0);
    } else {
      put32(head, e.value);
    }
  }
  put32(head, 0);  // no further directory
  head.insert(head.end(), extra.begin(), extra.end());

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    log_error("Cannot open %s for writing\n", path.c_str());
    return -1;
  }
  bool ok = fwrite(head.data(), 1, head.size(), fp) == head.size();
  // The samples little-endian, as the header says ("II"), whatever the host's byte order; three per pixel.
  std::vector<uint8_t> row((size_t)width * 6);
  for (int j = 0; ok && j < height; ++j) {
    const uint16_t* src = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      for (int c = 0; c < 3; ++c) {
        const uint16_t v = src[(size_t)i * channels + c];
        row[(size_t)i * 6 + 2 * c] = (uint8_t)(v & 0xff);
        row[(size_t)i * 6 + 2 * c + 1] = (uint8_t)(v >> 8);
      }
    }
    ok = fwrite(row.data(), 1, row.size(), fp) == row.size();
  }
  ok = (fclose(fp) == 0) && ok;
  return finish_write(path, ok);
}

int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, DensityMeasure measure,
                      const DensityFileInfo& info) {
  if (enc.bits != 10) {
    log_error("Internal error: a DPX holds 10-bit density codes\n");
    return -1;
  }
  // The layout is SMPTE ST 268:2014's (DPX, header version V2.0): the file (768 bytes, 5.1), image (640, 5.2) and
  // image source (256, 5.3, the orientation information below) headers make the generic 1664, the film (256, 6.1) and
  // television (128, 6.2) headers the industry 384, so the user-defined data starts at 2048 (clause 7; at most 1 MiB,
  // 4.1). An undefined integer or real field holds all ones (4.2). Descriptor 50 is R, G, B (Table 1); transfer and
  // colorimetric codes 0 and 1 are user defined and printing density (Tables 5A and 5B); packing 1 is filled to 32-bit
  // words, method A (Table 3B).
  // The user-defined data: a 32-byte identification, then the metadata JSON; the image starts on a multiple of 8.
  const uint32_t kHeader = 2048, kUserId = 32, kMaxUser = 1024 * 1024;
  uint32_t user_size = kUserId + (uint32_t)description.size() + 1;
  user_size = (user_size + 7u) & ~7u;
  if (user_size > kMaxUser) {
    log_error("The density metadata (%u bytes) does not fit a DPX file's user data (1 MiB)\n", user_size);
    return -1;
  }
  const uint64_t image_bytes = (uint64_t)width * (uint64_t)height * 4;
  const uint32_t image_offset = kHeader + user_size;
  if (image_bytes + image_offset > 0xffffff00ull) {
    log_error("The image is too large for a DPX file (%dx%d)\n", width, height);
    return -1;
  }
  std::vector<uint8_t> h(image_offset, 0);
  uint8_t* p = h.data();
  // File information (768 bytes).
  memcpy(p, "SDPX", 4);
  be32(p + 4, image_offset);
  put_str(p + 8, 8, "V2.0");
  be32(p + 16, (uint32_t)(image_offset + image_bytes));
  be32(p + 20, 1);                 // a new frame
  be32(p + 24, 1664);              // generic header: file, image and orientation information
  be32(p + 28, 384);               // industry header: film and television
  be32(p + 32, user_size);
  {
    const size_t slash = path.find_last_of("/\\");
    put_str(p + 36, 100, slash == std::string::npos ? path : path.substr(slash + 1));
  }
  put_str(p + 136, 24, date_time(info.timestamp, ':'));   // the capture's time, as the TIFF's DateTime (none: blank)
  put_str(p + 160, 100, info.software);
  be32(p + 660, 0xffffffffu);      // not encrypted
  // Image information (640 bytes).
  uint8_t* im = p + 768;
  be16(im, 0);                     // left to right, top to bottom
  be16(im + 2, 1);                 // one image element
  be32(im + 4, (uint32_t)width);
  be32(im + 8, (uint32_t)height);
  uint8_t* el = im + 12;
  be32(el, 0);                     // unsigned
  be32(el + 4, (uint32_t)enc.offset);                                      // reference low: the film base
  be_float(el + 8, 0.0f);
  be32(el + 12, (uint32_t)enc.max_code());                                 // reference high
  be_float(el + 16, (float)(enc.max_code() - enc.offset) / (float)enc.scale);
  el[20] = 50;                     // descriptor: R, G, B
  // The transfer characteristic and the colorimetric specification: 1, printing density, for RP 180 (the Cineon
  // convention) and for Academy Printing Density (the code tools know; the element's description says ADX); 0, user
  // defined, for Status M, which is not a printing density.
  const uint8_t kind = measure == kDensityStatusM ? 0 : 1;
  el[21] = kind;
  el[22] = kind;
  el[23] = 10;                     // bits
  be16(el + 24, 1);                // packing: filled to 32-bit words, method A
  be16(el + 26, 0);                // no run-length encoding
  be32(el + 28, image_offset);
  be32(el + 32, 0);
  be32(el + 36, 0);
  put_str(el + 40, 32, measure == kDensityStatusM ? "Status M density above base"
                           : measure == kDensityRp180 ? "RP 180 printing density"
                                                      : "ADX10, SMPTE ST 2065-3");
  for (int k = 1; k < 8; ++k) memset(im + 12 + k * 72, 0, 72);
  // Orientation information (256 bytes): undefined but for the original size and the input device.
  uint8_t* orient = p + 1408;
  be32(orient, 0);
  be32(orient + 4, 0);
  memset(orient + 8, 0xff, 8);     // centre: undefined
  be32(orient + 16, (uint32_t)width);
  be32(orient + 20, (uint32_t)height);
  put_str(orient + 148, 32, info.model.empty() ? info.make : info.model);
  memset(orient + 212, 0xff, 8);   // border validity
  be32(orient + 220, 1);           // pixel aspect ratio 1:1
  be32(orient + 224, 1);
  // Film information (256 bytes) and television information (128 bytes): undefined.
  uint8_t* film = p + 1664;
  memset(film + 48, 0xff, 20);     // frame position, sequence length, held count, frame rate, shutter angle
  uint8_t* tv = p + 1920;
  memset(tv, 0xff, 52);
  // User-defined data.
  put_str(p + kHeader, kUserId, "negicc.density/1");
  memcpy(p + kHeader + kUserId, description.data(), description.size());

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    log_error("Cannot open %s for writing\n", path.c_str());
    return -1;
  }
  bool ok = fwrite(h.data(), 1, h.size(), fp) == h.size();
  std::vector<uint8_t> row((size_t)width * 4);
  for (int j = 0; ok && j < height; ++j) {
    const uint16_t* src = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      const uint32_t r = src[(size_t)i * channels] & 0x3ffu, g = src[(size_t)i * channels + 1] & 0x3ffu,
                     b = src[(size_t)i * channels + 2] & 0x3ffu;
      be32(&row[(size_t)i * 4], (r << 22) | (g << 12) | (b << 2));
    }
    ok = fwrite(row.data(), 4, (size_t)width, fp) == (size_t)width;
  }
  ok = (fclose(fp) == 0) && ok;
  return finish_write(path, ok);
}


// ---- The print of a density file (the print declarations above) -----------------------------------------------------

namespace {

bool read_vector(const nlohmann::json& j, std::vector<double>& out) {
  if (!j.is_array()) return false;
  out.clear();
  for (const auto& v : j) {
    if (!v.is_number()) return false;
    out.push_back(v.get<double>());
    if (!std::isfinite(out.back())) return false;
  }
  return true;
}

// numpy.interp: linear between the points, the end values outside them.
double interp(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  const size_t hi = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
  const double x0 = xs[hi - 1], x1 = xs[hi];
  return ys[hi - 1] + (ys[hi] - ys[hi - 1]) * (x - x0) / (x1 - x0);
}

void amounts_to_xyz(const PrintMaterial& m, const double amount[3], double xyz[3]) {
  xyz[0] = xyz[1] = xyz[2] = 0.0;
  const size_t n = m.base.size();
  for (size_t w = 0; w < n; ++w) {
    const double t =
        std::pow(10.0, -(amount[0] * m.dye[0][w] + amount[1] * m.dye[1][w] + amount[2] * m.dye[2][w] + m.base[w]));
    xyz[0] += t * m.observer[0][w];
    xyz[1] += t * m.observer[1][w];
    xyz[2] += t * m.observer[2][w];
  }
}

}  // namespace

namespace {

// A column of spektrafilm data (numbers and nulls) on its own wavelengths w, resampled to 380-780 nm at 1 nm. Nulls
// before the first and after the last number take the nearest number (hold) or `missing`; a null between numbers is
// an error (density_matrices.py hold_gaps).
bool resample_column(const nlohmann::json& rows, int column, const std::vector<double>& w, bool hold, double missing,
                     bool as_power_of_ten, std::vector<double>& out) {
  if (!rows.is_array() || rows.size() != w.size()) return false;
  std::vector<double> v(w.size(), std::nan(""));
  for (size_t i = 0; i < w.size(); ++i) {
    const nlohmann::json& cell = column < 0 ? rows[i] : (rows[i].is_array() && (int)rows[i].size() > column
                                                             ? rows[i][column] : nlohmann::json());
    if (cell.is_number() && std::isfinite(cell.get<double>())) v[i] = cell.get<double>();
  }
  size_t first = 0, last = w.size();
  while (first < w.size() && std::isnan(v[first])) ++first;
  while (last > first && std::isnan(v[last - 1])) --last;
  if (first == w.size()) return false;
  if (hold) {
    for (size_t i = first; i < last; ++i) {
      if (std::isnan(v[i])) return false;
    }
    for (size_t i = 0; i < first; ++i) v[i] = v[first];
    for (size_t i = last; i < w.size(); ++i) v[i] = v[last - 1];
  }
  for (double& x : v) x = std::isnan(x) ? missing : as_power_of_ten ? std::pow(10.0, x) : x;
  out.resize(kCieSamples);
  for (int i = 0; i < kCieSamples; ++i) out[i] = interp(w, v, kCieFirstNm + i);
  return true;
}

void multiply3(const double a[9], const double b[9], double out[9]) {
  double r[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  std::copy(r, r + 9, out);
}

bool invert3(const double m[9], double out[9]) {
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (!std::isfinite(det) || std::fabs(det) < 1e-12) return false;
  const double r[9] = {m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                       m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                       m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]};
  for (int k = 0; k < 9; ++k) out[k] = r[k] / det;
  return true;
}

// XYZ of the print for a log exposure of each layer.
void exposure_to_xyz(const PrintMaterial& m, const double log_e[3], double xyz[3]) {
  double amount[3];
  for (int c = 0; c < 3; ++c) amount[c] = interp(m.log_exposure, m.curves[c], log_e[c]);
  amounts_to_xyz(m, amount, xyz);
}

// Q: the 3x3 from a film's dye amounts to the density above the film base that response (rows R, G, B, 401 samples)
// reads, least squares with no offset on the film model's fixed sample colours. Every density read is the logarithm of
// a transmittance 10^-D weighted by the response, which must be a finite number above 0: a density for which a double
// holds 10^-D as 0 or infinity (a dye density of -13822, say) has none. False with why in why.
bool film_to_print_density(const FilmModel& film, const std::vector<double> response[3], double q[9],
                           std::string& why) {
  const double kSpread = 0.15 * std::sqrt(1.5);   // each dye amount times exp(kSpread z), z = -1, 0, 1
  double sum[3] = {0, 0, 0}, base_density[3];
  for (int c = 0; c < 3; ++c) {
    double t = 0.0;
    for (int w = 0; w < kCieSamples; ++w) {
      sum[c] += response[c][w];
      t += std::pow(10.0, -film.base[w]) * response[c][w];
    }
    if (!(sum[c] > 0.0)) return why = "the print's layer " + std::to_string(c) + " has no response", false;
    base_density[c] = -std::log10(t / sum[c]);
    if (!std::isfinite(base_density[c])) {
      return why = "the density of film_model.base_density that the print's layer " + std::to_string(c) +
                   " reads is not a finite number", false;
    }
  }
  double ata[9] = {0}, atd[9] = {0};            // sums of a a^T and of a d^T over the samples
  std::vector<double> transmittance(kCieSamples);
  for (size_t i = 0; i < film.log_exposure.size(); ++i) {
    if (film.log_exposure[i] < -1.5 || film.log_exposure[i] > 1.0) continue;
    for (int v = 0; v < 27; ++v) {
      const int z[3] = {v / 9 - 1, v / 3 % 3 - 1, v % 3 - 1};
      double a[3], d[3];
      for (int c = 0; c < 3; ++c) a[c] = std::max(0.0, film.neutral[c][i] * std::exp(kSpread * z[c]));
      for (int w = 0; w < kCieSamples; ++w) {
        transmittance[w] = std::pow(
            10.0, -(a[0] * film.dye[0][w] + a[1] * film.dye[1][w] + a[2] * film.dye[2][w] + film.base[w]));
      }
      for (int c = 0; c < 3; ++c) {
        double t = 0.0;
        for (int w = 0; w < kCieSamples; ++w) t += transmittance[w] * response[c][w];
        d[c] = -std::log10(t / sum[c]) - base_density[c];
        if (!std::isfinite(d[c])) {
          char at[32];
          snprintf(at, sizeof at, "%.9g", film.log_exposure[i]);
          return why = std::string("the density that the print's layer ") + std::to_string(c) + " reads of " +
                       "film_model's neutral at log exposure " + at + " (its dye_density, neutral_dye_amounts and " +
                       "base_density) is not a finite number", false;
        }
      }
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          ata[r * 3 + c] += a[r] * a[c];
          atd[r * 3 + c] += a[r] * d[c];
        }
      }
    }
  }
  double inverse[9], solved[9];
  if (!invert3(ata, inverse)) {
    return why = "film_model's neutral_dye_amounts between log exposure -1.5 and 1.0 do not determine the matrix (the "
                 "least squares is singular or not finite)", false;
  }
  multiply3(inverse, atd, solved);              // solved[k][c]: the weight of amount k in density c
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) q[r * 3 + c] = solved[c * 3 + r];
  }
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(q[k])) return why = "the matrix from film_model's dye amounts is not finite", false;
  }
  return true;
}

}  // namespace

namespace {

// Whether a parsed JSON is a spektrafilm profile of a print material neg_process can print with: info.stage
// "printing", info.support paper or film (it picks the viewing light), log_exposure ascending with one row of
// density_curves per value, each layer's curve rising with exposure, and each layer's log_sensitivity peaking within 6
// (log10) of the most sensitive layer's. False with why in err. Only load_spektrafilm_print calls it, so one reader
// holds every rule.
bool check_spektrafilm_print(const nlohmann::json& j, std::string& err) {
  if (!j.is_object() || !j.contains("data") || !j["data"].is_object())
    return err = "not a spektrafilm profile (no data)", false;
  if (!j.contains("info") || !j["info"].is_object()) return err = "a spektrafilm profile without info", false;
  const nlohmann::json& info = j["info"];
  // A print material: spektrafilm also has camera films (stage "filming"), which would compute without complaint.
  const std::string stage = info.contains("stage") && info["stage"].is_string() ? info["stage"].get<std::string>() : "";
  if (stage != "printing") {
    return err = stage.empty() ? "a spektrafilm profile that does not say it is a print material (info.stage)"
                               : "a spektrafilm profile of a " + stage + " stock, not of a print material", false;
  }
  // The viewing light comes from the support: D50 for a paper, D55 for a print film.
  const std::string support =
      info.contains("support") && info["support"].is_string() ? info["support"].get<std::string>() : "";
  if (support != "paper" && support != "film") {
    return err = "a spektrafilm print profile whose info.support is '" + support +
                 "', not paper or film: the viewing light is chosen by it", false;
  }
  // The exposure axis the curves are judged against: ascending, one value per row of density_curves.
  const nlohmann::json& d = j["data"];
  if (!d.contains("density_curves") || !d["density_curves"].is_array() || d["density_curves"].size() < 2) {
    return err = "a spektrafilm print profile without density_curves", false;
  }
  std::vector<double> log_exposure;
  if (!d.contains("log_exposure") || !read_vector(d["log_exposure"], log_exposure) ||
      log_exposure.size() != d["density_curves"].size()) {
    return err = "log_exposure of the spektrafilm print profile is missing or not one value per row of "
                 "density_curves", false;
  }
  if (!std::is_sorted(log_exposure.begin(), log_exposure.end())) {
    return err = "log_exposure of the spektrafilm print profile is not in ascending order", false;
  }
  // Each layer's curve rises with exposure: by kMinCurveRise over the profile, and never more than kMaxCurveDip below
  // its highest value so far, which does not depend on how finely the curve is sampled. In spektrafilm's eight
  // published print profiles the largest step down is 0.00093 and the largest fall below the running maximum 0.0025,
  // where the print is at its minimum density.
  constexpr double kMinCurveRise = 0.5, kMaxCurveDip = 0.02;
  for (int c = 0; c < 3; ++c) {
    double first = 0.0, last = 0.0, highest = -INFINITY;
    for (size_t i = 0; i < d["density_curves"].size(); ++i) {
      const nlohmann::json& row = d["density_curves"][i];
      if (!row.is_array() || row.size() != 3 || !row[c].is_number() || !std::isfinite(row[c].get<double>())) {
        return err = "density_curves of the spektrafilm print profile has a missing value", false;
      }
      const double v = row[c].get<double>();
      if (i == 0) first = v;
      if (v < highest - kMaxCurveDip) {
        return err = "layer " + std::to_string(c) + "'s curve of the spektrafilm print profile falls with "
                     "exposure", false;
      }
      highest = std::max(highest, v);
      last = v;
    }
    if (!(last - first > kMinCurveRise)) {
      return err = "layer " + std::to_string(c) +
                   "'s curve of the spektrafilm print profile does not rise with exposure", false;
    }
  }
  // Each layer is sensitive somewhere: its peak log sensitivity within kMaxSensitivityGap (log10) of the most
  // sensitive layer's, so no layer of the print is blind to the printer's light. A row is read as the load reads it
  // (resample_column): the layer's column, whatever follows it.
  constexpr double kMaxSensitivityGap = 6.0;
  if (!d.contains("log_sensitivity") || !d["log_sensitivity"].is_array()) {
    return err = "a spektrafilm print profile without log_sensitivity", false;
  }
  double peak[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (const nlohmann::json& row : d["log_sensitivity"]) {
    for (int c = 0; row.is_array() && c < (int)std::min<size_t>(3, row.size()); ++c) {
      if (row[c].is_number() && std::isfinite(row[c].get<double>())) peak[c] = std::max(peak[c], row[c].get<double>());
    }
  }
  const double best = std::max(peak[0], std::max(peak[1], peak[2]));
  for (int c = 0; c < 3; ++c) {
    if (!(peak[c] > best - kMaxSensitivityGap)) {
      return err = "layer " + std::to_string(c) + " of the spektrafilm print profile has no spectral "
                   "sensitivity", false;
    }
  }
  return true;
}

// info.<key> when it is a string, else fallback: spektrafilm writes nulls in info (target_print is null in all eight
// published print profiles), and json::value throws on a key that is there with another type.
std::string info_string(const nlohmann::json& info, const char* key, const std::string& fallback) {
  return info.contains(key) && info[key].is_string() ? info[key].get<std::string>() : fallback;
}

}  // namespace

bool load_spektrafilm_print(const std::string& json_text, const std::string& id, const ProfileBundle& bundle,
                            PrintMaterial& out, std::string& err) {
  const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
  if (j.is_discarded()) return err = "not a spektrafilm profile (not JSON)", false;
  if (!check_spektrafilm_print(j, err)) return false;
  const nlohmann::json& d = j["data"];
  const nlohmann::json& info = j["info"];
  PrintMaterial m;
  m.id = info_string(info, "stock", id);
  m.name = info_string(info, "name", m.id);
  m.viewing_illuminant = info["support"] == "paper" ? "D50" : "D55";
  std::vector<double> w, sensitivity[3];
  bool ok = d.contains("wavelengths") && read_vector(d["wavelengths"], w) && w.size() >= 2 &&
            std::is_sorted(w.begin(), w.end()) && d.contains("log_sensitivity") && d.contains("channel_density") &&
            d.contains("base_density") && read_vector(d["log_exposure"], m.log_exposure) &&
            resample_column(d["base_density"], -1, w, true, 0.0, false, m.base);
  for (int c = 0; ok && c < 3; ++c) {
    // A layer's sensitivity where the profile has none is zero; its dye spectrum is held at the nearest value.
    ok = resample_column(d["log_sensitivity"], c, w, false, 0.0, true, sensitivity[c]) &&
         resample_column(d["channel_density"], c, w, true, 0.0, false, m.dye[c]);
    m.curves[c].resize(m.log_exposure.size());   // the check has read every value as a finite number
    for (size_t i = 0; i < m.log_exposure.size(); ++i) m.curves[c][i] = d["density_curves"][i][c].get<double>();
  }
  if (!ok) {
    err = "a field of the spektrafilm print profile is missing or malformed";
    return false;
  }
  // What the print's own density is read with: each layer's sensitivity under the printer lamp, a 3200 K blackbody.
  // Planck's law in relative units, with the second radiation constant c2 = 1.4388e-2 m K that ITS-90 fixes.
  std::vector<double> response[3];
  for (int c = 0; c < 3; ++c) {
    response[c].resize(kCieSamples);
    for (int i = 0; i < kCieSamples; ++i) {
      const double metres = (kCieFirstNm + i) * 1e-9;
      const double lamp = 1.0 / (std::pow(metres, 5) * (std::exp(1.4388e-2 / (metres * 3200.0)) - 1.0));
      response[c][i] = sensitivity[c][i] * lamp;
    }
    // 10^log_sensitivity times the lamp, and their sum, as doubles: a value above what a double holds has none.
    double sum = 0.0;
    for (int i = 0; i < kCieSamples; ++i) sum += response[c][i];
    if (!std::isfinite(sum)) {
      err = "layer " + std::to_string(c) + "'s log_sensitivity of the spektrafilm print profile gives a response "
            "under the printer lamp that is not a finite number";
      return false;
    }
  }
  // The observer under the viewing light, Y of the light 1; the print's white is the light itself.
  const double* light = m.viewing_illuminant == "D50" ? kCieD50 : kCieD55;
  const double* cmf[3] = {kCieXBar, kCieYBar, kCieZBar};
  double light_y = 0.0;
  for (int i = 0; i < kCieSamples; ++i) light_y += light[i] * kCieYBar[i];
  for (int c = 0; c < 3; ++c) {
    m.observer[c].resize(kCieSamples);
    m.white[c] = 0.0;
    for (int i = 0; i < kCieSamples; ++i) {
      m.observer[c][i] = light[i] * cmf[c][i] / light_y;
      m.white[c] += m.observer[c][i];
    }
  }
  // XYZ under the viewing light to linear sRGB, adapted to D65 with CAT02. cat02: the CIECAM02 chromatic adaptation
  // matrix (CIE 159:2004). xyz_to_srgb: XYZ D65 -> linear sRGB as Bruce Lindbloom tabulates it (brucelindbloom.com,
  // computed with ASTM E308-01's D65 white 0.95047, 1, 1.08883).
  const double cat02[9] = {0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.0030, 0.0136, 0.9834};
  const double xyz_to_srgb[9] = {3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108,
                                 0.0415560, 0.0556434,  -0.2040259, 1.0572252};
  // D65 as the matrix has it (the XYZ it takes to sRGB 1, 1, 1), so the viewing light's white lands on sRGB white
  // exactly.
  double srgb_to_xyz[9], d65[3];
  invert3(xyz_to_srgb, srgb_to_xyz);
  for (int r = 0; r < 3; ++r) d65[r] = srgb_to_xyz[r * 3] + srgb_to_xyz[r * 3 + 1] + srgb_to_xyz[r * 3 + 2];
  double cat_inverse[9], scaled[9];
  invert3(cat02, cat_inverse);
  for (int r = 0; r < 3; ++r) {
    const double to = cat02[r * 3] * d65[0] + cat02[r * 3 + 1] * d65[1] + cat02[r * 3 + 2] * d65[2];
    const double from = cat02[r * 3] * m.white[0] + cat02[r * 3 + 1] * m.white[1] + cat02[r * 3 + 2] * m.white[2];
    for (int c = 0; c < 3; ++c) scaled[r * 3 + c] = cat02[r * 3 + c] * to / from;
  }
  multiply3(cat_inverse, scaled, scaled);
  multiply3(xyz_to_srgb, scaled, m.to_srgb);
  // aim: the exposure of each layer at which the print is an 18 % grey under its light. Newton's method on the three
  // exposures, with the Jacobian by differences.
  double e[3] = {0.0, 0.0, 0.0};
  for (int iteration = 0; iteration < 50; ++iteration) {
    double xyz[3], f[3], jac[9], inverse[9];
    exposure_to_xyz(m, e, xyz);
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      f[c] = xyz[c] - 0.18 * m.white[c];
      worst = std::max(worst, std::fabs(f[c]));
    }
    if (worst < 1e-13) break;
    for (int k = 0; k < 3; ++k) {
      double moved[3] = {e[0], e[1], e[2]}, xyz2[3];
      moved[k] += 1e-4;
      exposure_to_xyz(m, moved, xyz2);
      for (int c = 0; c < 3; ++c) jac[c * 3 + k] = (xyz2[c] - xyz[c]) / 1e-4;
    }
    if (!invert3(jac, inverse)) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
    for (int k = 0; k < 3; ++k) e[k] -= inverse[k * 3] * f[0] + inverse[k * 3 + 1] * f[1] + inverse[k * 3 + 2] * f[2];
    // A step that is not a finite number has no exposure to read the curves at.
    if (!std::isfinite(e[0]) || !std::isfinite(e[1]) || !std::isfinite(e[2])) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
  }
  double check[3];
  exposure_to_xyz(m, e, check);
  for (int c = 0; c < 3; ++c) {
    if (!(std::fabs(check[c] - 0.18 * m.white[c]) < 1e-6)) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
    m.aim[c] = e[c];
  }
  // The print matrix of each measure, for this bundle's film: Q inv(F), with any light_density_factors of the measure
  // undone on the light's channels (Q inv(F) + Q L diag(1 / alpha - 1) inv(L) inv(F) = Q L inv(M)) so that P M = Q L
  // for every measure and the bracket's uncalibrated APD slopes apply once.
  auto print_matrix = [](const double q[9], const DensityMatrix& dm, double p[9]) {
    double inv_f[9];
    if (!dm.present || !invert3(dm.film_to_measure, inv_f)) return false;
    multiply3(q, inv_f, p);
    const double* alpha = dm.light_density_factors;
    if (alpha[0] == 1.0 && alpha[1] == 1.0 && alpha[2] == 1.0) return true;
    double inv_l[9];
    if (!invert3(dm.light_to_film, inv_l)) return false;
    double ql[9], inv_lf[9], scaled[9], corr[9];
    multiply3(q, dm.light_to_film, ql);
    multiply3(inv_l, inv_f, inv_lf);
    for (int r = 0; r < 3; ++r) {
      const double d = 1.0 / alpha[r] - 1.0;
      for (int c = 0; c < 3; ++c) scaled[r * 3 + c] = d * inv_lf[r * 3 + c];
    }
    multiply3(ql, scaled, corr);
    for (int k = 0; k < 9; ++k) p[k] += corr[k];
    return true;
  };
  // A film_model that gives no finite matrix leaves the material without one, and why, which a print reports.
  double q[9];
  std::string why;
  if (bundle.film_model.present && !film_to_print_density(bundle.film_model, response, q, why)) {
    for (std::string& no : m.no_print_matrix) {
      no = "the profile bundle's film_model gives no print matrix on " + m.name + ": " + why;
    }
  } else if (bundle.film_model.present) {
    for (int k = 0; k < kDensityMeasures; ++k) {
      PrintMaterial::ToPrint t;
      if (!print_matrix(q, bundle.density_matrix[k], t.m.data())) continue;
      if (!std::all_of(t.m.begin(), t.m.end(), [](double v) { return std::isfinite(v); })) {
        m.no_print_matrix[k] = "the print matrix on " + m.name + " for " + density_measure_key((DensityMeasure)k) +
                            " from the profile bundle's film_model and density matrix is not finite";
        continue;
      }
      t.film = bundle.film_model.name;
      m.to_print[k].push_back(t);
    }
  }
  out = std::move(m);
  return true;
}

std::string print_clf_nodes(const PrintMaterial& m, int curve_points, int cube) {
  char buf[256];
  std::string s;
  s.reserve((size_t)cube * cube * cube * 40 + (size_t)curve_points * 40 + 4096);
  // log E = aim - (D - grey)
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %d %d %d %.10g\n", i == 0 ? -1 : 0, i == 1 ? -1 : 0, i == 2 ? -1 : 0,
             m.aim[i] + m.grey_density);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  // Every value written is a finite number, or there is no CLF (an empty text): the exposure range, the curves' scale
  // and the cube.
  const double lo = m.log_exposure.front(), hi = m.log_exposure.back();
  if (!std::isfinite(hi - lo)) return {};
  snprintf(buf, sizeof buf,
           "  <Range inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <minInValue>%.10g</minInValue>\n"
           "    <maxInValue>%.10g</maxInValue>\n    <minOutValue>0</minOutValue>\n    <maxOutValue>1</maxOutValue>\n"
           "  </Range>\n",
           lo, hi);
  s += buf;
  snprintf(buf, sizeof buf, "  <LUT1D inBitDepth=\"32f\" outBitDepth=\"32f\" interpolation=\"linear\">\n"
                            "    <Array dim=\"%d 3\">\n", curve_points);
  s += buf;
  double top = 0.0;
  for (int c = 0; c < 3; ++c) top = std::max(top, *std::max_element(m.curves[c].begin(), m.curves[c].end()));
  top = std::ceil(top * 10.0) / 10.0;
  if (!std::isfinite(top) || !std::isfinite(1.0 / top)) return {};
  for (int i = 0; i < curve_points; ++i) {
    const double x = lo + (hi - lo) * i / (curve_points - 1);
    snprintf(buf, sizeof buf, "      %.6g %.6g %.6g\n", interp(m.log_exposure, m.curves[0], x),
             interp(m.log_exposure, m.curves[1], x), interp(m.log_exposure, m.curves[2], x));
    s += buf;
  }
  s += "    </Array>\n  </LUT1D>\n";
  snprintf(buf, sizeof buf, "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 3\">\n"
                            "      %.10g 0 0\n      0 %.10g 0\n      0 0 %.10g\n    </Array>\n  </Matrix>\n",
           1.0 / top, 1.0 / top, 1.0 / top);
  s += buf;
  snprintf(buf, sizeof buf, "  <LUT3D inBitDepth=\"32f\" outBitDepth=\"32f\" interpolation=\"tetrahedral\">\n"
                            "    <Array dim=\"%d %d %d 3\">\n", cube, cube, cube);
  s += buf;
  // The cube: dye amounts 0..top on each axis, blue changing fastest, to linear sRGB, relative to the paper's white:
  // each of X, Y and Z scaled by the viewing light's over the paper's (the ICC's media-relative colorimetry), so the
  // paper renders at the display's white, as the print ICC's picture does (the pipeline code below: the print's white).
  double paper[3], rel[3];
  print_paper_xyz(m, paper);
  for (int c = 0; c < 3; ++c) rel[c] = m.white[c] / paper[c];
  const size_t points = (size_t)cube * cube * cube;
  std::vector<double> rgb(points * 3);
  #pragma omp parallel for schedule(static)
  for (long long idx = 0; idx < (long long)points; ++idx) {
    const int b = (int)(idx % cube), g = (int)((idx / cube) % cube), r = (int)(idx / ((long long)cube * cube));
    const double amount[3] = {top * r / (cube - 1), top * g / (cube - 1), top * b / (cube - 1)};
    double xyz[3];
    amounts_to_xyz(m, amount, xyz);
    for (int c = 0; c < 3; ++c) xyz[c] *= rel[c];
    for (int c = 0; c < 3; ++c) {
      rgb[idx * 3 + c] = m.to_srgb[c * 3] * xyz[0] + m.to_srgb[c * 3 + 1] * xyz[1] + m.to_srgb[c * 3 + 2] * xyz[2];
    }
  }
  if (!std::all_of(rgb.begin(), rgb.end(), [](double v) { return std::isfinite(v); })) return {};
  for (size_t idx = 0; idx < points; ++idx) {
    snprintf(buf, sizeof buf, "      %.6g %.6g %.6g\n", rgb[idx * 3], rgb[idx * 3 + 1], rgb[idx * 3 + 2]);
    s += buf;
  }
  s += "    </Array>\n  </LUT3D>\n";
  // monCurveRev with exponent 2.4 and offset 0.055: the sRGB encoding (IEC 61966-2-1) as the Academy's Common LUT
  // Format v3.0 Exponent element writes it.
  s += "  <Exponent inBitDepth=\"32f\" outBitDepth=\"32f\" style=\"monCurveRev\">\n"
       "    <ExponentParams exponent=\"2.4\" offset=\"0.055\" />\n  </Exponent>\n";
  return s;
}

bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err) {
  if (!rec.bundle || !rec.matrix) return err = "no bundle", false;
  const ProfileBundle& bundle = *rec.bundle;
  // The film model the bundle's density matrices were fitted on names the material's matrix for this film.
  const nlohmann::json fit_record = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit_record.is_object() ? fit_record.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* to_print = material.to_print_for((int)rec.measure, film);
  if (!to_print) {
    const std::string& why = material.no_print_matrix[(int)rec.measure];
    return err = why.empty() ? "no print matrix for this measure and this film" : why, false;
  }
  if (bundle.targets.empty()) return err = "the bundle has no brackets", false;
  FramePrint p;
  p.id = material.id;
  p.name = material.name;
  p.grey_density = material.grey_density;
  for (int k = 0; k < 9; ++k) p.to_print[k] = to_print->m[k];
  p.to_print_for = to_print->film;
  // The frame's bracket and exposure: the search's, else the middle bracket at its own exposure. A material with its
  // own print search (a density run with --print, on either light) takes that search's A, E and gains (below).
  p.searched = rec.has_balance && rec.balance_e > 0.0;
  p.bracket = p.searched ? rec.balance_target : bundle.targets[bundle.targets.size() / 2].name;
  p.exposure = p.searched ? rec.balance_e : 1.0;
  const auto own = rec.print_search.find(material.id);
  const bool by_print = own != rec.print_search.end() && bundle.grey_balance.present &&
                        bundle.density_matrix[kDensityApd].present;
  if (by_print) {
    p.searched = true;
    p.print_searched = true;
    p.bracket = own->second.bracket;
    p.exposure = own->second.e;
    p.a = own->second.a;
    p.gain_g = own->second.g;
    p.gain_b = own->second.b;
    p.converged = own->second.converged;
    p.steps = own->second.steps;
    for (int c = 0; c < 3; ++c) p.last_move[c] = own->second.last_move[c];
    p.above_knee = own->second.above_knee;
    for (int c = 0; c < 3; ++c) p.past_knee[c] = own->second.past_knee[c], p.past_cut[c] = own->second.past_cut[c];
    p.stopped_by = own->second.stopped_by;
    p.a0 = own->second.a0;
    p.target_l = own->second.target_l;
    for (int c = 0; c < 3; ++c) {
      p.above_white[c] = own->second.above_white[c];
      p.at_white[c] = own->second.at_white[c];
    }
    p.knobs = own->second.knobs;
    p.knobs_limited = own->second.knobs_limited;
    p.knob_base_a = own->second.knob_base_a;
    p.knob_base_g = own->second.knob_base_g;
    p.knob_base_b = own->second.knob_base_b;
    for (int c = 0; c < 3; ++c) p.knob_grey_lab[c] = own->second.knob_grey_lab[c];
    for (int c = 0; c < 3; ++c) p.knob_given[c] = own->second.knob_given[c];
  }
  const DensityBracket* bracket = nullptr;
  for (const DensityBracket& b : bundle.density_brackets) {
    if (b.name == p.bracket) bracket = &b;
  }
  if (!bracket) {
    return err = "the bundle has no density_matrices.brackets entry for " + p.bracket +
                 ": rebuild it with the profile-building tools", false;
  }
  // The exposure E multiplies the light's channels before the bracket's profile reads them: the film that the
  // profile reads as its mid grey is denser by log10(E) in every light channel, through the measure's matrix.
  // A bundle with the film's grey balance curve: the mid grey and the slopes are the curve's, as in its
  // print ICCs.
  const double* m = rec.matrix->m;
  const double shift = std::log10(p.exposure);
  double light[3] = {bracket->grey_light[0], bracket->grey_light[1], bracket->grey_light[2]};
  for (int c = 0; c < 3; ++c) p.slopes[c] = bracket->slopes[c];
  if (by_print) {
    // The print at its own search's mid grey A, as the picture conversion renders it: the curve's grey and tangent at
    // A, and the brackets' tilt at A (print_tilt_at: linear in A between the two brackets around it).
    curve_grey_at(bundle, p.a, light);
    curve_slopes_at(bundle, p.a, p.slopes);
    double t[3];
    print_tilt_at(bundle, p.a, t);
    for (int c = 0; c < 3; ++c) p.slopes[c] *= 1.0 + t[c];
    p.from_curve = true;
  } else if (bundle.grey_balance.present && bundle.density_matrix[kDensityApd].present) {
    curve_grey(bundle, *bracket, light);
    curve_slopes(bundle, *bracket, p.slopes);
    // The bracket's tilt relative to the curve, as its print ICC's curves take it: (G_c - A_mid)(1 + t_c) has the
    // slope (1 + t_c) times the curve's at the mid grey.
    for (int c = 0; c < 3; ++c) p.slopes[c] *= 1.0 + bracket->tilt[c];
    p.from_curve = true;
  } else if (!bracket->has_slopes) {
    return err = "the bundle has neither grey_balance_curves nor slopes for " + p.bracket +
                 ": rebuild it with the profile-building tools", false;
  }
  for (int c = 0; c < 3; ++c) {
    p.grey[c] = 0.0;
    for (int k = 0; k < 3; ++k) p.grey[c] += m[c * 3 + k] * (light[k] + shift);
  }
  out = p;
  return true;
}

// The density offsets bal the print CLF applies for the frame's colour balance, level included (the zero-sum offsets
// plus the mean taken out of them): from the material's own print search's gains when it has one, else from the
// picture conversion's search. False when there is no balance (bal is then zero).
static bool print_balance(const DensityRecord& rec, const FramePrint& print, double bal[3]) {
  DensityRecord own = rec;   // the material's own print search: its gains in place of the picture conversion's
  if (print.print_searched) {
    own.has_balance = true;
    own.balance_e = print.exposure;
    own.balance_g = print.gain_g;
    own.balance_b = print.gain_b;
  }
  double mean = 0.0;
  const bool balanced = density_balance_offset(own, bal, &mean);
  for (int c = 0; c < 3; ++c) bal[c] += mean;
  return balanced;
}

std::string frame_print_clf(const DensityRecord& rec, const FramePrint& print, const std::string& nodes) {
  // x: the file's code / max code.  D_j = (x_j * max - offset) / (scale * k_j);  with the balance bal and the frame's
  // mid grey,  pd_i = grey_density + s_i * sum_j P_ij (D_j + bal_j - grey_j).
  //
  // bal is the whole of what the search's gains g and b do to the density, level included (the zero-sum offsets and
  // the mean taken out of them). The conversion renders as the chart's mid grey the film that is denser than it by
  // log10 E, log10 E g and log10 E b in the light's channels; with grey carrying log10 E and bal the rest, that film
  // comes out at grey_density in all three channels.
  const DensityEncoding& e = rec.encoding;
  const double max_code = (double)e.max_code();
  double bal[3];
  const bool balanced = print_balance(rec, print, bal);
  double a[9], b[3];
  for (int i = 0; i < 3; ++i) {
    double shift = 0.0;
    for (int j = 0; j < 3; ++j) {
      const double p = print.slopes[i] * print.to_print[i * 3 + j];
      a[i * 3 + j] = p * max_code / (e.scale * e.k[j]);
      shift += p * (bal[j] - print.grey[j] - e.offset / (e.scale * e.k[j]));
    }
    b[i] = print.grey_density + shift;
  }
  // Names come from a bundle and a print material file: escaped, and the text built without a fixed buffer.
  auto xml = [](const std::string& text) {
    std::string out;
    for (char ch : text) {
      switch (ch) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        default: out += ch;
      }
    }
    return out;
  };
  auto num = [](const char* format, double a, double b, double c) {
    char text[96];
    snprintf(text, sizeof text, format, a, b, c);
    return std::string(text);
  };
  char buf[256];
  const std::string print_name = xml(print.name);
  std::string s = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  s += "<ProcessList id=\"negicc.print." + xml(print.id) + "\" name=\"" + print_name + "\" compCLFversion=\"3.0\">\n";
  snprintf(buf, sizeof buf, "%s density in %d bits, code = %d + %d k D", density_measure_name(rec.measure), e.bits,
           e.offset, e.scale);
  s += "  <Description>This frame (" + xml(rec.bundle_key) + ", " + buf + ") printed on " + print_name +
       ", as sRGB. The first Matrix takes the file's code values to the print's printing density with the frame's "
       "colour balance, density offsets " + num("%+.4f %+.4f %+.4f", bal[0], bal[1], bal[2]) +
       (balanced ? "" : " (no balance was found: none applied)") + ", and slopes " +
       num("%.4f %.4f %.4f", print.slopes[0], print.slopes[1], print.slopes[2]) +
       " about the frame's mid grey, density " + num("%.4f %.4f %.4f", print.grey[0], print.grey[1], print.grey[2]) +
       ". " +
       (print.print_searched
            ? "The mid grey is the film's grey balance curve's grey at the print's own mid grey A " +
                  num("%.4f", print.a, 0.0, 0.0) +
                  ", which this material's own print search found on this frame, with that search's exposure and "
                  "gains (the nearest bracket: " + xml(print.bracket) +
                  "); balanced, it prints as the print's mid grey. "
                  "The slopes are that curve's contrast there times one plus the brackets' tilt at A (grey_relative.b, "
                  "linear in A between the brackets): they give the film's grey scale equal contrast in the three "
                  "channels"
            : print.from_curve
                  ? "The mid grey is the film's grey balance curve's grey at the green density of the chart's mid grey "
                    "on the bracket " + xml(print.bracket) +
                        (print.searched ? " and at the exposure the picture conversion's search gave this frame"
                                        : " (no search: the middle bracket)") +
                        "; balanced, it prints as the print's mid grey. The slopes are that curve's contrast there "
                        "times one plus the bracket's tilt (grey_relative.b): they give the film's grey scale equal "
                        "contrast in the three channels"
                  : "The mid grey is the chart's on the bracket " + xml(print.bracket) +
                        (print.searched ? " and at the exposure the picture conversion's search gave this frame"
                                        : " (no search: the middle bracket)") +
                        "; balanced, it prints as the print's mid grey. The slopes are the bracket's in the film's "
                        "profile: they give the film's grey scale equal contrast in the three channels") +
       ". The density file holds none of this. Add a constant to the three offsets of the first Matrix to print "
       "lighter (+) or darker (-), 0.30 a stop.</Description>\n";
  s += "  <Description>The 1D and 3D tables are derived by neg_process from the print profile of this material in "
       "spektrafilm (Andrea Volpato, https://github.com/andreavolpato/spektrafilm): its curve per layer resampled, "
       "and its dye spectra computed into colours. CC BY-SA 4.0, "
       "https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt. The matrices are computed "
       "by neg_process and are not spektrafilm's data. A model, not compared with a real print.</Description>\n";
  s += "  <InputDescriptor>density code values</InputDescriptor>\n  <OutputDescriptor>sRGB</OutputDescriptor>\n";
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %.10g %.10g %.10g %.10g\n", a[i * 3], a[i * 3 + 1], a[i * 3 + 2], b[i]);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  s += nodes;
  s += "</ProcessList>\n";
  return s;
}

bool is_print_file(const std::string& name) {
  return name.find_first_of("/\\") != std::string::npos ||
         (name.size() > 5 && name.substr(name.size() - 5) == ".json");
}

std::vector<PrintMaterial> builtin_print_materials(const ProfileBundle* bundle) {
  static const ProfileBundle no_bundle;
  const std::string folder = "3rd_party/spektrafilm/profiles/";
  std::vector<PrintMaterial> out;
  for (const EmbeddedFile& file : list_embedded_files()) {
    const std::string name = file.name;
    if (name.rfind(folder, 0) != 0 || name.size() < folder.size() + 6 || name.substr(name.size() - 5) != ".json") {
      continue;
    }
    PrintMaterial material;
    std::string err;
    if (!load_spektrafilm_print(std::string((const char*)file.data, file.size),
                                name.substr(folder.size(), name.size() - folder.size() - 5),
                                bundle ? *bundle : no_bundle, material, err)) {
      log_warn("Print profile %s cannot be read: %s\n", name.c_str(), err.c_str());
      continue;
    }
    out.push_back(std::move(material));
  }
  return out;
}

bool read_print_file(const std::string& path, std::string& text, std::string& stem, std::string& err) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return err = "cannot be read", false;
  std::stringstream buffer;
  buffer << file.rdbuf();
  if (file.bad()) return err = "cannot be read", false;
  text = buffer.str();
  const size_t slash = path.find_last_of("/\\");
  stem = slash == std::string::npos ? path : path.substr(slash + 1);
  if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".json") stem.resize(stem.size() - 5);
  return true;
}

DensityPrints density_prints(const DensityRecord& rec, const std::string& output,
                             const std::vector<std::string>& print_ids) {
  DensityPrints out;
  if (print_ids.empty()) return out;
  // The materials asked for: built-in ones by id (or all of them), and spektrafilm print profiles by path, whose
  // matrices are computed here for the bundle's film.
  const bool all = std::find(print_ids.begin(), print_ids.end(), "all") != print_ids.end();
  std::vector<PrintMaterial> materials;
  for (PrintMaterial& material : builtin_print_materials(rec.bundle)) {
    if (all || std::find(print_ids.begin(), print_ids.end(), material.id) != print_ids.end()) {
      materials.push_back(std::move(material));
    }
  }
  for (const std::string& id : print_ids) {
    if (!is_print_file(id)) continue;
    std::string text, stem, err;
    PrintMaterial material;
    if (!read_print_file(id, text, stem, err) || !rec.bundle ||
        !load_spektrafilm_print(text, stem, *rec.bundle, material, err)) {
      out.error = "no print CLF for " + id + ": " + (err.empty() ? "no profile bundle" : err);
      return out;
    }
    materials.push_back(std::move(material));
  }
  if (!materials.empty() && (!rec.bundle || !rec.bundle->film_model.present)) {
    out.error = "no print CLF: the profile bundle has no film_model, which a print's matrix is computed from: rebuild "
                "it with the profile-building tools";
    return out;
  }
  for (PrintMaterial& material : materials) {
    FramePrint print;
    std::string err;
    // Two materials of one id would share a CLF name: the first is kept.
    const std::string path = output + "." + material.id + ".clf";
    if (std::find(out.paths.begin(), out.paths.end(), path) != out.paths.end()) {
      log_warn("--print names the print material %s twice: its CLF is written once, for the first (built-in "
               "materials before files)\n", material.id.c_str());
      continue;
    }
    if (!frame_print(rec, material, print, err)) {
      out.error = "no print CLF for " + material.id + ": " + err;
      return out;
    }
    out.paths.push_back(output + "." + material.id + ".clf");
    out.materials.push_back(std::move(material));
    out.frames.push_back(print);
  }
  if (out.frames.empty()) return out;
  using nlohmann::ordered_json;
  ordered_json j;
  const std::vector<FramePrint>* prints = &out.frames;
  {
    // How the frame prints on each print material: computed for this frame, applied by the CLF beside the file.
    ordered_json materials = ordered_json::object();
    for (const FramePrint& p : *prints) {
      const double* t = p.to_print;
      ordered_json one = {
          {"print", p.name},
          {"to_print_density", {{t[0], t[1], t[2]}, {t[3], t[4], t[5]}, {t[6], t[7], t[8]}}},
          {"to_print_density_for", p.to_print_for},
          {"bracket", p.bracket},
          {"bracket_from", p.print_searched ? "the print's own search on this frame, on this material (the bracket "
                                              "nearest its mid grey A, named only: the tilt is interpolated at A)"
                           : p.searched ? "the picture conversion's search on this frame"
                                        : "the middle bracket (no search)"},
          {"exposure", p.exposure},
          {"frame_grey_density", {p.grey[0], p.grey[1], p.grey[2]}},
          {"slopes", {p.slopes[0], p.slopes[1], p.slopes[2]}},
          {"grey_and_slopes_from", p.print_searched ? "the film's grey balance curve at the print's own mid grey A, "
                                                      "the slopes times one plus the brackets' tilt at A"
                                   : p.from_curve ? "the film's grey balance curve at the bracket's mid grey, the "
                                                    "slopes times one plus the bracket's tilt"
                                                  : "the bracket's entry in the profile set"}};
      if (p.print_searched) {
        one["mid_grey_a"] = p.a;
        one["gains"] = {p.gain_g, p.gain_b};
        double bal[3];
        print_balance(rec, p, bal);
        one["balance"] = {bal[0], bal[1], bal[2]};   // the formula's balance for this material, as its CLF applies it
        one["search"] = {{"a0", p.a0}, {"target_L", p.target_l},
                         {"converged", p.converged}, {"stopped_by", p.stopped_by}, {"steps", p.steps},
                         {"last_move", {p.last_move[0], p.last_move[1], p.last_move[2]}},
                         {"above_knee_pct", 100.0 * p.above_knee},
                         {"past_knee_pct", {100.0 * p.past_knee[0], 100.0 * p.past_knee[1], 100.0 * p.past_knee[2]}},
                         {"past_cut_pct", {100.0 * p.past_cut[0], 100.0 * p.past_cut[1], 100.0 * p.past_cut[2]}},
                         {"above_white_pct",
                          {100.0 * p.above_white[0], 100.0 * p.above_white[1], 100.0 * p.above_white[2]}},
                         {"at_white_pct", {100.0 * p.at_white[0], 100.0 * p.at_white[1], 100.0 * p.at_white[2]}}};
        if (p.knobs) {
          one["knobs"] = {{"what", "--brightness, --tint and --warmth on this print: mid_grey_a and gains are the "
                                   "moved ones (A, g and b solved on the print's profile, E held), from the search's"},
                          {"brightness", p.knob_given[0]}, {"tint", p.knob_given[1]}, {"warmth", p.knob_given[2]},
                          {"search", {{"a", p.knob_base_a}, {"g", p.knob_base_g}, {"b", p.knob_base_b}}},
                          {"grey_lab", {p.knob_grey_lab[0], p.knob_grey_lab[1], p.knob_grey_lab[2]}},
                          {"limited", p.knobs_limited}};
        }
      }
      materials[p.id] = one;
    }
    j["print"] = {
        {"what", "how this frame prints on each print material: the frame's mid grey lands on the print's mid grey, "
                 "and the slopes give the film's grey scale equal contrast in the three channels. With the print's own "
                 "search on a material, under either light, the mid grey A, the exposure and the gains are that "
                 "search's, and the mid grey and the slopes are the film's grey balance curve's at A "
                 "(density_matrices.grey_balance_curves). Otherwise the bracket and the exposure are the picture "
                 "conversion's search's; the mid grey and the slopes are the film's grey balance curve's at the "
                 "bracket's mid grey, or the bracket's in the profile set (density_matrices.brackets) "
                 "(grey_and_slopes_from). The curve's slopes are multiplied by one plus the brackets' tilt "
                 "(grey_relative.b) at the mid grey"},
        {"formula", "print density = 0.70 + slopes * (to_print_density @ (D + balance - frame_grey_density)), "
                    "balance = the material's balance where it has its own print search (materials.<id>.balance, "
                    "from its gains), else suggested_balance.density_offset + density_offset_mean_removed"},
        {"applied", "no: the densities are as read. The CLF beside the file for each print material "
                    "(<output>.<print>.clf) applies the balance, the exposure and the slopes; apply them once, there "
                    "or by hand, not both"},
        {"materials", materials}};
  }
  // A print file's name (a material's id) need not be UTF-8: written as the sidecar's writer writes it.
  out.metadata_json = j["print"].dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
  return out;
}

void remove_print_clfs(const DensityPrints& prints) {
  for (const std::string& path : prints.paths) remove(path.c_str());
}

int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints) {
  // Every material's nodes before any file is written, so a material with none writes nothing.
  std::vector<std::string> nodes;
  for (size_t k = 0; k < prints.frames.size(); ++k) {
    nodes.push_back(print_clf_nodes(prints.materials[k]));
    if (nodes.back().empty()) {
      log_error("No print CLF for %s: its log exposure range, its curves or its colours are not finite numbers\n",
                prints.materials[k].id.c_str());
      return 1;
    }
  }
  for (size_t k = 0; k < prints.frames.size(); ++k) {
    const FramePrint& print = prints.frames[k];
    const std::string clf = frame_print_clf(rec, print, nodes[k]);
    FILE* f = fopen(prints.paths[k].c_str(), "wb");
    const bool ok = f && fwrite(clf.data(), 1, clf.size(), f) == clf.size();
    if ((f && fclose(f) != 0) || !ok) {
      log_error("Cannot write '%s'\n", prints.paths[k].c_str());
      remove_print_clfs(prints);
      return 1;
    }
    log_info("Wrote '%s': this file as a print on %s, sRGB (bracket %s, exposure %.3f, slopes %.3f %.3f %.3f)\n",
             prints.paths[k].c_str(), print.name.c_str(), print.bracket.c_str(), print.exposure, print.slopes[0],
             print.slopes[1], print.slopes[2]);
  }
  return 0;
}

// ---- The print ICC --------------------------------------------------------------------------------------------------

namespace {

// The value of a table on start + step i at x, linear, the end values outside it (numpy.interp on a uniform grid).
double on_grid(const std::vector<double>& ys, double start, double step, double x) {
  const double t = (x - start) / step;
  if (t <= 0.0) return ys.front();
  if (t >= (double)(ys.size() - 1)) return ys.back();
  const size_t i = (size_t)t;
  return ys[i] + (t - (double)i) * (ys[i + 1] - ys[i]);
}

std::vector<double> grey_grid(const GreyBalanceCurves& g) {
  std::vector<double> x(g.apd[0].size());
  for (size_t i = 0; i < x.size(); ++i) x[i] = g.start + g.step * (double)i;
  return x;
}

// The print matrix for APD of a material, for the film of the bundle's density matrices; null when it has none.
const double* apd_print_matrix(const ProfileBundle& bundle, const PrintMaterial& m) {
  const nlohmann::json fit = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit.is_object() ? fit.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* t = m.to_print_for(kDensityApd, film);
  return t ? t->m.data() : nullptr;
}

void put_u16(std::vector<uint8_t>& b, double v01) {
  const long q = std::lround(std::min(std::max(v01, 0.0), 1.0) * 65535.0);
  b.push_back((uint8_t)(q >> 8));
  b.push_back((uint8_t)(q & 255));
}
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) b.push_back((uint8_t)(v >> s));
}
void put_s15(std::vector<uint8_t>& b, double v) { put_u32(b, (uint32_t)(int32_t)std::lround(v * 65536.0)); }
uint32_t get_u32(const std::vector<uint8_t>& b, size_t o) {
  return (uint32_t)b[o] << 24 | (uint32_t)b[o + 1] << 16 | (uint32_t)b[o + 2] << 8 | b[o + 3];
}
// The offset of an ICC's A2B0 lut16 tag, or 0.
size_t a2b0_lut16(const std::vector<uint8_t>& icc) {
  if (icc.size() < 132) return 0;
  const uint32_t n = get_u32(icc, 128);
  for (uint32_t i = 0; i < n && 144 + 12 * i <= icc.size(); ++i) {
    if (std::memcmp(icc.data() + 132 + 12 * i, "A2B0", 4) != 0) continue;
    const size_t off = get_u32(icc, 136 + 12 * i);
    if (off + 52 > icc.size() || std::memcmp(icc.data() + off, "mft2", 4) != 0 || icc[off + 8] != 3) return 0;
    return off;
  }
  return 0;
}

constexpr double kPrintKnee = kKnee;  // the conversion's soft knee (profile_bundle.h), in double

}  // namespace

double bracket_mid_apd(const ProfileBundle& bundle, const DensityBracket& bracket) {
  const double* m = bundle.density_matrix[kDensityApd].m;
  return m[3] * bracket.grey_light[0] + m[4] * bracket.grey_light[1] + m[5] * bracket.grey_light[2];
}

void curve_grey_at(const ProfileBundle& bundle, double a, double light[3]) {
  const GreyBalanceCurves& g = bundle.grey_balance;
  const std::vector<double> x = grey_grid(g);
  for (int c = 0; c < 3; ++c) light[c] = interp(g.apd[c], x, a);
}

void curve_grey(const ProfileBundle& bundle, const DensityBracket& bracket, double light[3]) {
  curve_grey_at(bundle, bracket_mid_apd(bundle, bracket), light);
}

void print_tilt_at(const std::vector<double>& a_mid, const std::vector<std::array<double, 3>>& tilt, double a,
                   double t[3]) {
  for (int c = 0; c < 3; ++c) t[c] = 0.0;
  if (a_mid.empty() || a_mid.size() != tilt.size()) return;
  std::vector<size_t> o(a_mid.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = i;
  // Stable, so brackets at the same A_mid keep the bundle's order.
  std::stable_sort(o.begin(), o.end(), [&](size_t x, size_t y) { return a_mid[x] < a_mid[y]; });
  if (o.size() == 1) {   // one bracket: its tilt
    for (int c = 0; c < 3; ++c) t[c] = tilt[o[0]][c];
    return;
  }
  size_t k = 0;   // the segment o[k], o[k + 1]
  while (k + 2 < o.size() && a > a_mid[o[k + 1]]) ++k;
  const size_t i = o[k], j = o[k + 1];
  // Two brackets at the same A_mid: the tilt of the one first in the bundle (no division by zero).
  const double f = a_mid[j] > a_mid[i] ? std::clamp((a - a_mid[i]) / (a_mid[j] - a_mid[i]), 0.0, 1.0) : 0.0;
  for (int c = 0; c < 3; ++c) t[c] = tilt[i][c] + f * (tilt[j][c] - tilt[i][c]);
}

void print_tilt_at(const ProfileBundle& bundle, double a, double t[3]) {
  // Over the brackets of bundle.targets, as the picture's print (bracket_print_profiles' continuum).
  std::vector<double> am;
  std::vector<std::array<double, 3>> tl;
  for (const BundleTarget& target : bundle.targets)
    for (const DensityBracket& b : bundle.density_brackets) {
      if (b.name != target.name) continue;
      am.push_back(bracket_mid_apd(bundle, b));
      tl.push_back({b.tilt[0], b.tilt[1], b.tilt[2]});
      break;
    }
  print_tilt_at(am, tl, a, t);
}

void curve_slopes(const ProfileBundle& bundle, const DensityBracket& bracket, double slopes[3]) {
  curve_slopes_at(bundle, bracket_mid_apd(bundle, bracket), slopes);
}

void curve_slopes_at(const ProfileBundle& bundle, double a, double slopes[3]) {
  const GreyBalanceCurves& g = bundle.grey_balance;
  double n[3], inv[3], k[3] = {0, 0, 0};
  curve_grey_at(bundle, a, n);
  for (int c = 0; c < 3; ++c) {
    const double d = (on_grid(g.apd[c], g.start, g.step, n[c] + g.step) -
                      on_grid(g.apd[c], g.start, g.step, n[c] - g.step)) / (2.0 * g.step);
    inv[c] = 1.0 / d;
  }
  const double* m = bundle.density_matrix[kDensityApd].m;
  for (int c = 0; c < 3; ++c) k[c] = m[c * 3] * inv[0] + m[c * 3 + 1] * inv[1] + m[c * 3 + 2] * inv[2];
  slopes[0] = k[1] / k[0];
  slopes[1] = 1.0;
  slopes[2] = k[1] / k[2];
}

PrintCurves paper_curve(const PrintMaterial& m, const double p_apd[9]) {
  PrintCurves out;
  for (int c = 0; c < 3; ++c) {
    const double eta = p_apd[c * 3] + p_apd[c * 3 + 1] + p_apd[c * 3 + 2];
    out[c].resize(kPaperAlphaPoints);
    double run = INFINITY;
    for (int i = 0; i < kPaperAlphaPoints; ++i) {
      const double alpha = kPaperAlphaStart + kPaperAlphaStep * i;
      run = std::min(run, std::max(0.0, interp(m.log_exposure, m.curves[c], m.aim[c] - eta * alpha)));
      out[c][i] = run;
    }
    // The 0.25 linear ramp makes each Paper_c strictly decreasing, so print_table can invert it.
    const double top = out[c][0];
    for (int i = 0; i < kPaperAlphaPoints; ++i)
      out[c][i] = 0.75 * (out[c][i] / top) + 0.25 * (1.0 - (double)i / (kPaperAlphaPoints - 1));
  }
  return out;
}

PrintCurves bracket_print_curves(const GreyBalanceCurves& g, const PrintCurves& paper, double a_mid, double n0,
                                 const double tilt[3], const double gains[3]) {
  // The knee's inverse caps atanh's argument at 0.998, and the light density floors u g_c at 1e-7 before the log, so
  // both stay finite; the 1e-9 i ramp keeps each curve strictly increasing.
  const double d = 1.0 - kPrintKnee;
  PrintCurves out;
  for (int c = 0; c < 3; ++c) {
    out[c].resize(kPrintIccPoints);
    double run = -INFINITY;
    for (int i = 0; i < kPrintIccPoints; ++i) {
      const double v = (double)i / (kPrintIccPoints - 1);
      const double u = v <= kPrintKnee
                           ? v
                           : kPrintKnee + d * std::atanh(std::min(std::max((v - kPrintKnee) / d, 0.0), 0.998));
      const double dl = -std::log10(std::max(u * (gains ? gains[c] : 1.0), 1e-7) / n0);
      const double alpha = (on_grid(g.apd[c], g.start, g.step, dl) - a_mid) * (1.0 + tilt[c]);
      run = std::max(run, on_grid(paper[c], kPaperAlphaStart, kPaperAlphaStep, alpha));
      out[c][i] = run + 1e-9 * i;
    }
  }
  return out;
}

std::vector<double> print_table(const PrintMaterial& m, const double p_apd[9], const double m_apd[9],
                                const PrintCurves& paper) {
  // Q L = P M and w = M^-1 (1, 1, 1).
  double ql[9], inv[9], w[3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      ql[r * 3 + c] = p_apd[r * 3] * m_apd[c] + p_apd[r * 3 + 1] * m_apd[3 + c] + p_apd[r * 3 + 2] * m_apd[6 + c];
  {
    const double* a = m_apd;
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    inv[0] = (a[4] * a[8] - a[5] * a[7]) / det;  inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) / det;  inv[3] = (a[5] * a[6] - a[3] * a[8]) / det;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) / det;  inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) / det;  inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  }
  for (int r = 0; r < 3; ++r) w[r] = inv[r * 3] + inv[r * 3 + 1] + inv[r * 3 + 2];
  // The PCS: the identity for a paper (D50), Bradford from the light's white to D50 for a print film. B: the linearised
  // Bradford matrix of ICC.1:2022 Annex E.3 (equation E.1). D50 is the PCS white as an ICC profile stores it
  // (kPcsWhite), which the table's wtpt and the header's illuminant carry too.
  double pcs[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  if (m.viewing_illuminant != "D50") {
    static const double B[9] = {0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296};
    const double* d50 = negicc::kPcsWhite;
    double src[3] = {m.white[0] / m.white[1], 1.0, m.white[2] / m.white[1]}, gain[3];
    for (int r = 0; r < 3; ++r) {
      const double to = B[r * 3] * d50[0] + B[r * 3 + 1] * d50[1] + B[r * 3 + 2] * d50[2];
      const double from = B[r * 3] * src[0] + B[r * 3 + 1] * src[1] + B[r * 3 + 2] * src[2];
      gain[r] = to / from;
    }
    // inv(B) diag(gain) B, with inv(B) computed exactly rather than taken from the rounded table above.
    double bi[9];
    {
      const double* a = B;
      const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                         a[2] * (a[3] * a[7] - a[4] * a[6]);
      bi[0] = (a[4] * a[8] - a[5] * a[7]) / det;  bi[1] = (a[2] * a[7] - a[1] * a[8]) / det;
      bi[2] = (a[1] * a[5] - a[2] * a[4]) / det;  bi[3] = (a[5] * a[6] - a[3] * a[8]) / det;
      bi[4] = (a[0] * a[8] - a[2] * a[6]) / det;  bi[5] = (a[2] * a[3] - a[0] * a[5]) / det;
      bi[6] = (a[3] * a[7] - a[4] * a[6]) / det;  bi[7] = (a[1] * a[6] - a[0] * a[7]) / det;
      bi[8] = (a[0] * a[4] - a[1] * a[3]) / det;
    }
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        double v = 0.0;
        for (int k = 0; k < 3; ++k) v += bi[r * 3 + k] * gain[k] * B[k * 3 + c];
        pcs[r * 3 + c] = v;
      }
  }
  // The nodes' alpha: Paper_c inverted (it is decreasing, so read reversed).
  std::vector<double> alpha_grid(kPaperAlphaPoints), node_alpha[3];
  for (int i = 0; i < kPaperAlphaPoints; ++i)
    alpha_grid[kPaperAlphaPoints - 1 - i] = kPaperAlphaStart + kPaperAlphaStep * i;
  for (int c = 0; c < 3; ++c) {
    const std::vector<double> rev(paper[c].rbegin(), paper[c].rend());
    for (int i = 0; i < kPrintIccCube; ++i)
      node_alpha[c].push_back(interp(rev, alpha_grid, (double)i / (kPrintIccCube - 1)));
  }
  std::vector<double> out((size_t)kPrintIccCube * kPrintIccCube * kPrintIccCube * 3);
  size_t o = 0;
  for (int i = 0; i < kPrintIccCube; ++i)
    for (int j = 0; j < kPrintIccCube; ++j)
      for (int k = 0; k < kPrintIccCube; ++k) {
        const double wa[3] = {node_alpha[0][i] * w[0], node_alpha[1][j] * w[1], node_alpha[2][k] * w[2]};
        double amount[3], xyz[3];
        for (int c = 0; c < 3; ++c) {
          const double pd = m.grey_density + ql[c * 3] * wa[0] + ql[c * 3 + 1] * wa[1] + ql[c * 3 + 2] * wa[2];
          if (!std::isfinite(pd)) return {};
          amount[c] = interp(m.log_exposure, m.curves[c], m.aim[c] - (pd - m.grey_density));
        }
        amounts_to_xyz(m, amount, xyz);
        for (int c = 0; c < 3; ++c) {
          out[o] = pcs[c * 3] * xyz[0] + pcs[c * 3 + 1] * xyz[1] + pcs[c * 3 + 2] * xyz[2];
          if (!std::isfinite(out[o++])) return {};
        }
      }
  return out;
}

std::array<int, 6> bundle_date(const ProfileBundle& bundle) {
  // The bundle's source file name (export.source_file) ends in _YYYYMMDD_HHMMSS (profile_<Key>_<date>_<time>.json),
  // the date the bundle's own print ICCs carry; 2026-01-01 00:00:00 when it does not.
  std::string stem = bundle.source_file;
  const size_t slash = stem.find_last_of("/\\");
  if (slash != std::string::npos) stem = stem.substr(slash + 1);
  if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".json") stem.resize(stem.size() - 5);
  std::array<int, 6> d = {2026, 1, 1, 0, 0, 0};
  if (stem.size() >= 16 && stem[stem.size() - 16] == '_' && stem[stem.size() - 7] == '_') {
    const std::string t = stem.substr(stem.size() - 15, 8) + stem.substr(stem.size() - 6);
    if (std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; })) {
      d = {std::stoi(t.substr(0, 4)), std::stoi(t.substr(4, 2)), std::stoi(t.substr(6, 2)), std::stoi(t.substr(8, 2)),
           std::stoi(t.substr(10, 2)), std::stoi(t.substr(12, 2))};
    }
  }
  return d;
}

// A textDescriptionType tag (ICC v2; ICC.1:2001-04 6.5.17) holding description as its ASCII text, with empty Unicode
// and ScriptCode parts.
static std::vector<uint8_t> icc_desc_tag(const std::string& description) {
  // The ASCII part holds 7-bit text (ICC.1:2001-04 6.5.17): each character outside it (a UTF-8 material name's) is
  // written as one '?', as the station's write_icc writes it (Python's encode("ascii", "replace")).
  std::string ascii;
  for (const char ch : description) {
    const unsigned char u = (unsigned char)ch;
    if (u < 0x80) ascii += ch;
    else if (u >= 0xc0) ascii += '?';   // a character's first byte; its continuation bytes (0x80 to 0xbf) are dropped
  }
  std::vector<uint8_t> desc = {'d', 'e', 's', 'c', 0, 0, 0, 0};
  put_u32(desc, (uint32_t)ascii.size() + 1);
  desc.insert(desc.end(), ascii.begin(), ascii.end());
  desc.insert(desc.end(), 1 + 8 + 3 + 67, 0);
  return desc;
}

// A profile from its 128-byte header and its tags in order (each tag's data on a 4-byte boundary); the header's
// size field is set.
static std::vector<uint8_t> icc_assemble(std::vector<uint8_t> icc,
                                         const std::vector<std::pair<std::string, std::vector<uint8_t>>>& tags) {
  std::vector<uint8_t> table_bytes, body;
  const size_t start = 128 + 4 + 12 * tags.size();
  for (const auto& t : tags) {
    table_bytes.insert(table_bytes.end(), t.first.begin(), t.first.begin() + 4);
    put_u32(table_bytes, (uint32_t)(start + body.size()));
    put_u32(table_bytes, (uint32_t)t.second.size());
    body.insert(body.end(), t.second.begin(), t.second.end());
    while (body.size() % 4) body.push_back(0);
  }
  icc.resize(128);
  put_u32(icc, (uint32_t)tags.size());
  icc.insert(icc.end(), table_bytes.begin(), table_bytes.end());
  icc.insert(icc.end(), body.begin(), body.end());
  const uint32_t size = (uint32_t)icc.size();
  for (int k = 0; k < 4; ++k) icc[k] = (uint8_t)(size >> (24 - 8 * k));
  return icc;
}

// Replaces the profile's desc tag by one holding description; false when it has none or its tag table is not whole.
static bool set_icc_description(std::vector<uint8_t>& icc, const std::string& description) {
  if (icc.size() < 132) return false;
  const uint32_t n = get_u32(icc, 128);
  if (132 + 12 * (size_t)n > icc.size()) return false;
  std::vector<std::pair<std::string, std::vector<uint8_t>>> tags;
  bool found = false;
  for (uint32_t i = 0; i < n; ++i) {
    const std::string sig((const char*)icc.data() + 132 + 12 * i, 4);
    const size_t off = get_u32(icc, 136 + 12 * i), size = get_u32(icc, 140 + 12 * i);
    // Inside the file and after the tag table, without overflowing.
    if (off < 132 + 12 * (size_t)n || off > icc.size() || size > icc.size() - off) return false;
    if (sig == "desc") {
      tags.push_back({sig, icc_desc_tag(description)});
      found = true;
    } else {
      tags.push_back({sig, std::vector<uint8_t>(icc.begin() + off, icc.begin() + off + size)});
    }
  }
  if (!found) return false;
  icc = icc_assemble(std::vector<uint8_t>(icc.begin(), icc.begin() + 128), tags);
  return true;
}

// The layout is ICC.1:2001-04's (version 2.4.0, 02400000h, 6.1.3): the 128-byte header (6.1), the tag table, and the
// tags A2B0 as a lut16Type (mft2, 6.5.7), cprt as a textType (6.5.18) and wtpt as an XYZType (6.5.26).
std::vector<uint8_t> write_print_icc(const PrintCurves& curves, const std::vector<double>& table,
                                     const std::string& description, const std::string& copyright,
                                     const std::array<int, 6>& date) {
  const int n = (int)curves[0].size();
  std::vector<uint8_t> a2b0 = {'m', 'f', 't', '2', 0, 0, 0, 0, 3, 3, (uint8_t)kPrintIccCube, 0};
  for (int i = 0; i < 9; ++i) put_s15(a2b0, i % 4 == 0 ? 1.0 : 0.0);
  a2b0.push_back((uint8_t)(n >> 8));
  a2b0.push_back((uint8_t)(n & 255));
  a2b0.push_back(0);
  a2b0.push_back(2);
  for (int c = 0; c < 3; ++c)
    for (double v : curves[c]) put_u16(a2b0, v);
  for (double v : table) put_u16(a2b0, v / negicc::kPcsScale);
  for (int c = 0; c < 3; ++c) {
    put_u16(a2b0, 0.0);
    put_u16(a2b0, 1.0);
  }
  std::vector<uint8_t> cprt = {'t', 'e', 'x', 't', 0, 0, 0, 0};
  cprt.insert(cprt.end(), copyright.begin(), copyright.end());
  cprt.push_back(0);
  std::vector<uint8_t> wtpt = {'X', 'Y', 'Z', ' ', 0, 0, 0, 0};
  for (double v : negicc::kPcsWhite) put_s15(wtpt, v);
  std::vector<uint8_t> icc;
  put_u32(icc, 0);   // the size: icc_assemble sets it
  put_u32(icc, 0);
  put_u32(icc, 0x02400000);
  for (const char* s : {"scnr", "RGB ", "XYZ "}) icc.insert(icc.end(), s, s + 4);
  for (int v : date) {
    icc.push_back((uint8_t)(v >> 8));
    icc.push_back((uint8_t)(v & 255));
  }
  icc.insert(icc.end(), {'a', 'c', 's', 'p'});
  icc.insert(icc.end(), 24, 0);
  put_u32(icc, 0);
  for (double v : negicc::kPcsWhite) put_s15(icc, v);
  icc.insert(icc.end(), 48, 0);
  return icc_assemble(std::move(icc), {{"desc", icc_desc_tag(description)}, {"cprt", std::move(cprt)},
                                       {"wtpt", std::move(wtpt)}, {"A2B0", std::move(a2b0)}});
}

bool set_icc_wtpt(std::vector<uint8_t>& icc, const double xyz[3]) {
  if (icc.size() < 132) return false;
  const uint32_t n = get_u32(icc, 128);
  for (uint32_t i = 0; i < n && 144 + 12 * i <= icc.size(); ++i) {
    if (std::memcmp(icc.data() + 132 + 12 * i, "wtpt", 4) != 0) continue;
    const size_t off = get_u32(icc, 136 + 12 * i);
    if (off + 20 > icc.size() || std::memcmp(icc.data() + off, "XYZ ", 4) != 0) return false;
    for (int c = 0; c < 3; ++c) {
      const int32_t v = (int32_t)std::lround(xyz[c] * 65536.0);
      const uint32_t u = (uint32_t)v;
      icc[off + 8 + 4 * c] = (uint8_t)(u >> 24);
      icc[off + 9 + 4 * c] = (uint8_t)(u >> 16);
      icc[off + 10 + 4 * c] = (uint8_t)(u >> 8);
      icc[off + 11 + 4 * c] = (uint8_t)u;
    }
    return true;
  }
  return false;
}

void print_paper_xyz(const PrintMaterial& m, double xyz[3]) {
  double least[3];
  for (int c = 0; c < 3; ++c) least[c] = *std::min_element(m.curves[c].begin(), m.curves[c].end());
  amounts_to_xyz(m, least, xyz);
}

bool scale_print_icc_table(std::vector<uint8_t>& icc, const double k[3]) {
  const size_t off = a2b0_lut16(icc);
  if (!off) return false;
  const size_t n_in = (size_t)icc[off + 48] << 8 | icc[off + 49];
  const size_t grid = icc[off + 10];
  const size_t at = off + 52 + 6 * n_in;
  const size_t count = grid * grid * grid * 3;
  if (at + 2 * count > icc.size()) return false;
  for (size_t i = 0; i < count; ++i) {
    const size_t p = at + 2 * i;
    const long code = (long)icc[p] << 8 | icc[p + 1];
    const long q = std::min(65535L, std::lround(code * k[i % 3]));
    icc[p] = (uint8_t)(q >> 8);
    icc[p + 1] = (uint8_t)(q & 0xff);
  }
  return true;
}

bool set_print_icc_curves(std::vector<uint8_t>& icc, const PrintCurves& curves) {
  const size_t off = a2b0_lut16(icc);
  if (!off) return false;
  const size_t n = (size_t)icc[off + 48] << 8 | icc[off + 49];
  if (n != curves[0].size() || off + 52 + 6 * n > icc.size()) return false;
  std::vector<uint8_t> codes;
  for (int c = 0; c < 3; ++c)
    for (double v : curves[c]) put_u16(codes, v);
  std::copy(codes.begin(), codes.end(), icc.begin() + (long)(off + 52));
  return true;
}

void PrintContinuum::tilt_at(double a, double t[3]) const { print_tilt_at(a_mid, tilt, a, t); }

PrintCurves PrintContinuum::curves_at(double a, const double gains[3]) const {
  static const double zero[3] = {0.0, 0.0, 0.0};
  if (a_mid.empty()) return bracket_print_curves(g, paper, a, n0, zero, gains);
  double t[3];
  tilt_at(a, t);
  return bracket_print_curves(g, paper, a, n0, t, gains);
}

void PrintContinuum::trc_at(double a, const double gains[3], std::vector<float> trc[3]) const {
  const PrintCurves cv = curves_at(a, gains);
  for (int c = 0; c < 3; ++c) {
    trc[c].resize(cv[c].size());
    for (size_t i = 0; i < cv[c].size(); ++i)
      trc[c][i] = (float)std::lround(std::min(std::max(cv[c][i], 0.0), 1.0) * 65535.0) / 65535.0f;
  }
}

void PrintContinuum::set_curves_at(double a, ParsedFilmProfile& prof, const double gains[3]) const {
  trc_at(a, gains, prof.in_trc);
  for (int c = 0; c < 3; ++c) prof.data.in_trc_size[c] = (int)prof.in_trc[c].size();
  prof.rebind_pointers();
}

double PrintContinuum::a_lo() const { return *std::min_element(g.apd[1].begin(), g.apd[1].end()); }
double PrintContinuum::a_hi() const { return *std::max_element(g.apd[1].begin(), g.apd[1].end()); }

bool bracket_print_profiles(const ProfileBundle& bundle, const std::string& print, BracketPrints& out,
                            std::string& err) {
  if (!bundle.grey_balance.present) {
    return err = "the profile bundle has no grey_balance_curves (a print ICC is computed from them): rebuild it with "
                 "the profile-building tools", false;
  }
  if (!bundle.density_matrix[kDensityApd].present || !bundle.film_model.present) {
    return err = "the profile bundle has no APD matrix or film_model: rebuild it with the profile-building "
                 "tools", false;
  }
  // The material: a built-in one by id, or a spektrafilm print profile by path.
  PrintMaterial material;
  bool found = false;
  std::string file_text;
  if (is_print_file(print)) {
    std::string stem;
    if (!read_print_file(print, file_text, stem, err)) return false;
    if (!load_spektrafilm_print(file_text, stem, bundle, material, err)) return false;
    found = true;
  } else {
    const EmbeddedFile* f = find_embedded_file("3rd_party/spektrafilm/profiles/" + print + ".json");
    if (f && !load_spektrafilm_print(std::string((const char*)f->data, f->size), print, bundle, material, err)) {
      return false;
    }
    found = f != nullptr;
  }
  if (!found) return err = "no such print material", false;
  const double* p = apd_print_matrix(bundle, material);
  if (!p) {
    const std::string& why = material.no_print_matrix[kDensityApd];
    return err = why.empty() ? "no print matrix for the bundle's film" : why, false;
  }
  // Paper_c divides by its top, the print's density at the film's brightest: one not above 0 gives no curve.
  const PrintCurves paper = paper_curve(material, p);
  for (int c = 0; c < 3; ++c) {
    if (!std::all_of(paper[c].begin(), paper[c].end(), [](double v) { return std::isfinite(v); })) {
      return err = "the print's curve of layer " + std::to_string(c) + " through the film's print matrix is not "
                   "finite (its density at the film's brightest is not above 0)", false;
    }
  }
  const double n0 = bundle.normalization_target / 65535.0;
  auto bracket_of = [&](const std::string& name) -> const DensityBracket* {
    for (const DensityBracket& b : bundle.density_brackets)
      if (b.name == name) return &b;
    return nullptr;
  };
  // The table: the bundle's print ICC's when it is the file the bundle records, computed from the spektrafilm profile
  // built in here, and its own curves are those computed here for its bracket (the same film, print material and math,
  // to one 16-bit step); else computed here, with a warning saying why.
  const std::string description = bundle.key + " printed on " + material.name + " (spektrafilm)";
  std::vector<uint8_t> base;
  out = BracketPrints{};
  out.id = material.id;
  out.name = material.name;
  if (is_print_file(print)) {
    out.file = print;
    out.file_sha256 = sha256_hex(file_text);
  }
  for (const BundlePrint& bp : bundle.prints) {
    if (is_print_file(print) || bp.id != print) continue;
    // Read now, when the print asks for it (BundlePrint).
    std::vector<uint8_t> icc;
    if (!bp.read_icc || !bp.read_icc(icc, err)) return false;
    const DensityBracket* ref = bracket_of(bp.reference_bracket);
    const size_t off = a2b0_lut16(icc);
    if (!ref || !off) {
      log_warn("The bundle's print ICC for %s %s: its table is computed here\n", print.c_str(),
               !ref ? ("names a reference bracket the bundle has no density bracket for (" + bp.reference_bracket +
                       ")").c_str()
                    : "has no lut16 A2B0 table");
      break;
    }
    const EmbeddedFile* sf = find_embedded_file("3rd_party/spektrafilm/profiles/" + print + ".json");
    const std::string icc_sha = sha256_hex(std::string(icc.begin(), icc.end()));
    const std::string sf_sha = sf ? sha256_hex(std::string((const char*)sf->data, sf->size)) : "";
    if (bp.icc_in_file ? bp.icc_sha256 != icc_sha : !bp.icc_sha256.empty() && bp.icc_sha256 != icc_sha) {
      log_warn("The bundle's print ICC for %s is not the file the bundle records (its SHA-256 %s): its table is "
               "computed here\n", print.c_str(), bp.icc_sha256.empty() ? "is not recorded" : "differs");
      break;
    }
    if (bp.spektrafilm_sha256.empty() || bp.spektrafilm_sha256 != sf_sha) {
      log_warn("The bundle's print ICC for %s was computed from %s spektrafilm profile than the one built in: its "
               "table is computed here\n", print.c_str(), bp.spektrafilm_sha256.empty() ? "an unrecorded" : "another");
      break;
    }
    const PrintCurves mine =
        bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *ref), n0, ref->tilt);
    const size_t n = (size_t)icc[off + 48] << 8 | icc[off + 49];
    // The three input curves must lie inside the file: a truncated ICC is not read past its end.
    if (off + 52 + 6 * n > icc.size()) {
      log_warn("The bundle's print ICC for %s is cut short (%zu bytes): its table is computed here\n", print.c_str(),
               icc.size());
      break;
    }
    if (n != mine[0].size()) {
      log_warn("The bundle's print ICC for %s has %zu input curve entries, %zu computed now: its table is computed "
               "here\n", print.c_str(), n, mine[0].size());
      break;
    }
    long worst = 0;
    for (int c = 0; c < 3 && worst <= 1; ++c)
      for (size_t i = 0; i < n; ++i) {
        const size_t at = off + 52 + 2 * (c * n + i);
        const long code = (long)icc[at] << 8 | icc[at + 1];
        worst = std::max(worst, std::labs(code - std::lround(std::min(std::max(mine[c][i], 0.0), 1.0) * 65535.0)));
      }
    if (worst <= 1) {
      // Described as the table computed here is: the material, with no bracket (each bracket's curves and then the
      // print's A replace the file's).
      if (!set_icc_description(icc, description)) {
        log_warn("The bundle's print ICC for %s has no description tag to replace: its table is computed here\n",
                 print.c_str());
        break;
      }
      base = std::move(icc);
      out.from_bundle = true;
    } else {
      log_warn("The bundle's print ICC for %s does not match its film and print material as computed now (curves "
               "differ by %ld): its table is computed here\n", print.c_str(), worst);
    }
    break;
  }
  if (base.empty()) {
    const DensityBracket* mid =
        bracket_of(bundle.targets.empty() ? "" : bundle.targets[bundle.targets.size() / 2].name);
    if (!mid) return err = "the profile bundle has no density_matrices.brackets entry for its middle bracket", false;
    const std::vector<double> table = print_table(material, p, bundle.density_matrix[kDensityApd].m, paper);
    if (table.empty()) {
      return err = "the print's colours computed from the film's print matrix and the print material are not "
                   "finite", false;
    }
    base = write_print_icc(bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *mid), n0,
                                                mid->tilt),
                           table, description, kPrintIccCopyright, bundle_date(bundle));
  }
  for (const BundleTarget& t : bundle.targets) {
    const DensityBracket* b = bracket_of(t.name);
    if (!b) return err = "the profile bundle has no density_matrices.brackets entry for " + t.name, false;
    std::vector<uint8_t> icc = base;
    if (!set_print_icc_curves(icc, bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *b), n0,
                                                        b->tilt))) {
      return err = "the print ICC has no lut16 A2B0 to write curves into", false;
    }
    out.icc.push_back(std::move(icc));
    out.continuum.a_mid.push_back(bracket_mid_apd(bundle, *b));
    out.continuum.tilt.push_back({b->tilt[0], b->tilt[1], b->tilt[2]});
  }
  out.continuum.g = bundle.grey_balance;
  out.continuum.paper = paper;
  out.continuum.n0 = n0;
  return true;
}

// FIPS 180-4 SHA-256: k, the constants of 4.2.2; h, the initial hash value of 5.3.3.
std::string sha256_hex(const std::string& bytes) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string m = bytes;
  const uint64_t bits = (uint64_t)bytes.size() * 8;
  m.push_back((char)0x80);
  while (m.size() % 64 != 56) m.push_back(0);
  for (int i = 7; i >= 0; --i) m.push_back((char)(bits >> (8 * i)));
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (size_t block = 0; block < m.size(); block += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t)(uint8_t)m[block + 4 * i] << 24 | (uint32_t)(uint8_t)m[block + 4 * i + 1] << 16 |
             (uint32_t)(uint8_t)m[block + 4 * i + 2] << 8 | (uint32_t)(uint8_t)m[block + 4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  char out[65];
  for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
  return std::string(out, 64);
}

// ====================================================================================================================
// Film base: code
// The film base read from a capture of unexposed film (read_film_base_capture).
// ====================================================================================================================

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
    st.spread = kMadToSigma * median_of(dev);   // the MAD as a normal standard deviation
  }
  return st;
}

// Whether a tile's colour is the film base's: its medians through the crosstalk matrix, G/R and B/R against the
// expected base's (step 3 in the film base declarations above).
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
    if (t.st.n && t.st.clipped <= negicc::kBaseClipFraction && t.st.med[0] > 0.0 && t.st.med[1] > 0.0 &&
        t.st.med[2] > 0.0) {
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
  const float clip = negicc::kSensorClip * white_level;

  if (roi) {
    if (roi[0] < 0 || roi[1] < 0 || roi[2] <= 0 || roi[3] <= 0 || (long long)roi[0] + roi[2] > src_w ||
        (long long)roi[1] + roi[3] > src_h) {
      err = "the film base region " + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," +
            std::to_string(roi[2]) + "," + std::to_string(roi[3]) + " is off the capture's sensor (" +
            std::to_string(src_w) + "x" + std::to_string(src_h) + ")";
      return false;
    }
    // The half-size pixels (2x2 photosites) the region touches. An odd sensor's last column or row is in none.
    const int hx = roi[0] / 2, hy = roi[1] / 2;
    if (hx >= half_w || hy >= half_h) {
      err = "the film base region " + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," +
            std::to_string(roi[2]) + "," + std::to_string(roi[3]) + " lies on the capture's last column or row (" +
            std::to_string(src_w) + "x" + std::to_string(src_h) + "), which the half-size decode (2x2 photosites "
            "to a pixel) does not hold";
      return false;
    }
    const int hw = std::max(1, std::min(half_w - hx, (roi[0] + roi[2] + 1) / 2 - hx));
    const int hh = std::max(1, std::min(half_h - hy, (roi[1] + roi[3] + 1) / 2 - hy));
    const RegionStats st = region_stats(view, raw_view, hx, hy, hw, hh, clip);
    if (!st.n || st.med[0] <= 0.0 || st.med[1] <= 0.0 || st.med[2] <= 0.0) {
      err = "the film base region of " + path + " holds no light";
      return false;
    }
    if (st.clipped > negicc::kBaseClipFraction) {
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
    // recorded (the same pixels before the flat field) it must not be clipped.
    float thin[3] = {0.0f, 0.0f, 0.0f};
    bool thin_clipped = false;
    negicc::thinnest_film_base(view, white_level, thin, &thin_clipped);
    if (!(thin[0] > 1.0f) || !(thin[1] > 1.0f) || !(thin[2] > 1.0f)) {
      err = "no usable film base on " + path + " (no smooth film)";
      return false;
    }
    if (thin_clipped) {
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
  if (!has_exif || !(shutter > 0.0))
    shutter = raw->imgdata.other.shutter > 0 ? (double)raw->imgdata.other.shutter : 0.0;
  if (!has_exif || !(iso > 0.0)) {
    iso = raw->imgdata.other.iso_speed > 0 ? std::round((double)raw->imgdata.other.iso_speed) : 0.0;
  }
  if (!(shutter > 0.0) || !(iso > 0.0)) {
    if (!(shutter > 0.0)) shutter = kDefaultScanShutterS;
    if (!(iso > 0.0)) iso = kDefaultScanIso;
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

// ====================================================================================================================
// Options: code
// The command line, parsed and checked (parse_options).
// ====================================================================================================================

// Four files of a 4-shot pixel shift: distinct, one group of four in shot order by the camera's group tags or, without
// them, by their file counters. False with the error reported otherwise.
static bool check_pixel_shift_files(const std::vector<std::string>& files) {
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i + 1; j < 4; ++j) {
      std::error_code ec;
      if (files[i] == files[j] ||
          (std::filesystem::exists(files[i], ec) && std::filesystem::exists(files[j], ec) &&
           std::filesystem::equivalent(files[i], files[j], ec))) {
        log_error("4-shot pixel shift requires 4 distinct RAW files (duplicate: %s)\n",
                files[i].c_str());
        return false;
      }
    }
  }
  // The camera's group tags decide when the files carry them: one group of four, in shot order. Without tags the
  // file counters must increase from frame to frame, modulo the counter's wrap (DCF numbers files 0001-9999, so a
  // Sony DSC09999 is followed by DSC00001): each step forward, by less than half the counter's range.
  PixelShiftTags group_tags[4];
  for (size_t i = 0; i < 4; ++i) {
    LibRaw probe;
    if (probe.open_file(files[i].c_str()) != LIBRAW_SUCCESS) continue;  // the load reports it
    const auto& sony = probe.imgdata.makernotes.sony;
    group_tags[i].group_id = sony.PixelShiftGroupID;
    group_tags[i].shots_in_group = sony.nShotsInPixelShiftGroup;
    group_tags[i].num_in_group = sony.numInPixelShiftGroup;
  }
  std::string group_err;
  const PixelShiftGroupCheck group = check_pixel_shift_group(group_tags, group_err);
  if (group == PixelShiftGroupCheck::kBad) {
    log_error("%s\n", group_err.c_str());
    return false;
  }
  int trailing_idx[4] = {-1, -1, -1, -1};
  int idx_mod = 1;
  bool all_have_idx = group == PixelShiftGroupCheck::kNoTags;
  for (size_t i = 0; all_have_idx && i < 4; ++i) {
    const std::string stem = std::filesystem::path(files[i]).stem().string();
    size_t p = stem.size();
    while (p > 0 && std::isdigit((unsigned char)stem[p - 1])) --p;
    if (p == stem.size() || stem.size() - p > 6) {
      all_have_idx = false;
      break;
    }
    trailing_idx[i] = std::stoi(stem.substr(p));
    int mod = 1;
    for (size_t d = p; d < stem.size() && mod < 10000; ++d) mod *= 10;
    idx_mod = std::max(idx_mod, mod);
  }
  bool in_order = true;
  for (int i = 0; all_have_idx && i < 3; ++i) {
    const int step = ((trailing_idx[i + 1] - trailing_idx[i]) % idx_mod + idx_mod) % idx_mod;
    if (step == 0 || 2 * step >= idx_mod) in_order = false;
  }
  if (all_have_idx && !in_order) {
    log_error("4-shot pixel shift files are out of frame order\n");
    return false;
  }
  return true;
}

// --roi: 'auto [h|v]', or x y w h [rot_cw [hflip [vflip]]]. False with the error reported.
// The items of a comma-separated value, empty ones included ("1,,2" has three).
static std::vector<std::string> split_csv(const std::string& v) {
  std::vector<std::string> items;
  for (size_t p0 = 0;;) {
    const size_t p1 = v.find(',', p0);
    items.push_back(v.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0));
    if (p1 == std::string::npos) return items;
    p0 = p1 + 1;
  }
}

// One item of a recorded-numbers value: the whole item a finite number (strtof for float, strtod for double).
template <typename T>
static bool parse_number(const std::string& item, T& x) {
  char* end = nullptr;
  if constexpr (std::is_same_v<T, float>) x = std::strtof(item.c_str(), &end);
  else x = std::strtod(item.c_str(), &end);
  return !item.empty() && end && *end == '\0' && std::isfinite(x);
}

// One integer item: the whole token a decimal integer in int's range.
static bool parse_int(const std::string& item, int& x) {
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(item.c_str(), &end, 10);
  if (item.empty() || !end || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return false;
  x = (int)v;
  return true;
}

static bool parse_roi(const std::vector<std::string>& roi_tokens, Options& o) {
  if (roi_tokens.empty()) {
    log_error("--roi requires arguments\n");
    return false;
  }
  if (roi_tokens[0] == "auto") {
    o.roi_auto = true;
    if (roi_tokens.size() > 1) {
      if (roi_tokens[1] == "h" || roi_tokens[1] == "H") {
        o.roi_axis_hint = 'h';
      } else if (roi_tokens[1] == "v" || roi_tokens[1] == "V") {
        o.roi_axis_hint = 'v';
      } else {
        log_error("Invalid --roi auto axis hint '%s' (must be 'h' or 'v')\n", roi_tokens[1].c_str());
        return false;
      }
    }
    if (roi_tokens.size() > 2) {
      log_error("Too many arguments for --roi auto\n");
      return false;
    }
    return true;
  }
  if (roi_tokens.size() < 4) {
    log_error("--roi requires at least 4 arguments: x y w h [rot_cw [hflip [vflip]]]\n");
    return false;
  }
  // Each value a whole integer ("100.5" and "100px" are refused), and the flips 0 or 1.
  int v[7] = {0, 0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < roi_tokens.size() && i < 7; ++i) {
    if (!parse_int(roi_tokens[i], v[i])) {
      log_error("Invalid --roi value '%s': each must be an integer\n", roi_tokens[i].c_str());
      return false;
    }
    if (i >= 5 && v[i] != 0 && v[i] != 1) {
      log_error("Invalid --roi %s '%s': it must be 0 or 1\n", i == 5 ? "hflip" : "vflip", roi_tokens[i].c_str());
      return false;
    }
  }
  for (int i = 0; i < 4; ++i) o.roi[i] = v[i];
  if (roi_tokens.size() >= 5) o.rot_cw = v[4];
  if (roi_tokens.size() >= 6) o.hflip = v[5] != 0;
  if (roi_tokens.size() >= 7) o.vflip = v[6] != 0;
  o.has_roi = true;
  if (o.roi[0] < 0 || o.roi[1] < 0 || o.roi[2] <= 0 || o.roi[3] <= 0) {
    log_error("Invalid --roi [%d, %d, %d, %d]: x and y must be >= 0 and w and h must be > 0\n",
            o.roi[0], o.roi[1], o.roi[2], o.roi[3]);
    return false;
  }
  return true;
}

int parse_options(int ac, char* av[], Options& o) {
  for (int i = 1; i < ac; ++i) {
    // --print alone lists the print materials built in, and does nothing else. Given with a conversion's other
    // arguments and no material (at the end of the command line or before another flag), it is refused here so that
    // the next flag is not taken as the material's name.
    if (strcmp(av[i], "--print") == 0 && ac == 2) {
      for (const PrintMaterial& m : builtin_print_materials()) {
        log_info("--print %s  (%s)\n", m.id.c_str(), m.name.c_str());
      }
      return -1;
    }
    if (strcmp(av[i], "--print") == 0 && (i + 1 == ac || av[i + 1][0] == '-')) {
      log_error("--print needs a print material (a built-in id, 'all' or a spektrafilm profile's path); --print alone "
                "lists them\n");
      return 1;
    }
    if (strcmp(av[i], "--list_profiles") == 0) {
      for (const std::string& key : list_embedded_bundles()) {
        // Every bundle listed is one that loads: the build runs this after linking, so a bundle the parser rejects
        // (profile_bundle.cc) fails the build instead of a conversion.
        try {
          ProfileBundle bundle;
          load_embedded_bundle(key, bundle);
        } catch (const std::exception& e) {
          log_error("Built-in profile bundle %s cannot be read: %s\n", key.c_str(), e.what());
          return 1;
        }
        // Keys are "<Stock>-<camera>-<Light>".
        std::string stock = key.substr(0, key.find('-'));
        for (char& ch : stock) ch = (char)std::tolower((unsigned char)ch);
        const std::string light = key.substr(key.find_last_of('-') + 1);
        log_info("--film_stock %s --light_source %s  (%s)\n", stock.c_str(),
               light == "TriLED" ? "tri" : light == "D65" ? "d65" : light.c_str(), key.c_str());
      }
      return -1;
    }
  }
  argparse::ArgumentParser parser("neg_process", neg_process_version());
  parser.add_argument("-C", "--no_crop")
    .help("No cropping according to aspect ratio in RAW file.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-q", "--quality")
    .help("De-bayer quality (-1 for auto: grain-aware if Sony CFA, else bilinear 0; >=0 passes qual to LibRaw, e.g. 4 "
          "for DCB).")
    .scan<'i', int>()
    .default_value(-1);
  parser.add_argument("-r", "--r_coeff")
    .help("R (corrected) value is dot product of the vector r1 r2 r3, three separate numbers (-r 1 0 0), and "
          "'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none, and neither does "
          "a RAW that states a camera model the bundle does not list).")
    .nargs(3)
    .default_value(std::vector<float>{1, 0, 0})
    .scan<'g', float>();
  parser.add_argument("-g", "--g_coeff")
    .help("G (corrected) value is dot product of the vector g1 g2 g3, three separate numbers (-g 0 1 0), and "
          "'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none, and neither does "
          "a RAW that states a camera model the bundle does not list).")
    .nargs(3)
    .default_value(std::vector<float>{0, 1, 0})
    .scan<'g', float>();
  parser.add_argument("-b", "--b_coeff")
    .help("B (corrected) value is dot product of the vector b1 b2 b3, three separate numbers (-b 0 0 1), and "
          "'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none, and neither does "
          "a RAW that states a camera model the bundle does not list).")
    .nargs(3)
    .default_value(std::vector<float>{0, 0, 1})
    .scan<'g', float>();
  parser.add_argument("-E", "--exposure_comp")
    .help("Exposure. With a bundle, E as the parameter search gives it (the sidecar's "
          "cli_equivalent: auto.e, the search's own; with fitted curves, input_curves.remeter.e, re-metered on the "
          "combined profile; after the knobs, knobs.egb.e): "
          "relative to the bracket, the film base and the scan's exposure; with an ICC profile, a multiplier of the "
          "corrected RGB. Given with a bundle (either light), the bundle search holds it and solves the rest (the "
          "bracket unless --target gives it, the gains unless given), and the curve fit (white light) and the tone "
          "stage run on the result as after an unpinned search, without re-metering E. Given with an ICC profile, "
          "the single-bracket solver keeps it and solves only the gains; not given there, 1. --brightness moves the "
          "result, a given E too.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--no-metadata")
    .help("Disable automatic JSON metadata sidecar generation.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--gain_g")
    .help("Gain of the G channel relative to R (with a bundle, 1 renders the film base neutral). Defaults to 1.0. "
          "Given with a bundle (either light), the bundle search and, on white light, the curve fit hold it and solve "
          "the rest; with -E, --target and --gain_b given too there is nothing to search, and the search runs only for "
          "the curve fit (white light) and the tone stage (--tone_stage off, with --fit_curves off on white light, "
          "converts with the given values alone). Given with an ICC profile, it turns the "
          "parameter search off. --tint and --warmth move the result, a given gain too.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--gain_b")
    .help("Gain of the B channel relative to R (with a bundle, 1 renders the film base "
          "neutral). Defaults to 1.0. Given, as --gain_g.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--brightness")
    .help("Brightness offset after the parameter search and the curve fit, in stops of the reference grey's "
          "luminance. It moves the result, whether searched or given (-E sets what it starts "
          "from, not the final exposure), and the tone curve is applied on top unchanged; on a print (--print) it "
          "moves the print's mid grey A while E is held (a density output takes it only with --print, for its CLFs). "
          "Reuses the Auto or DINOv3 result in <output>.json when this build wrote it from the same input files (size "
          "and CRC-32) and the same arguments apart from -o and the knobs; otherwise searches again.")
    .scan<'g', float>()
    .default_value(0.0f);
  parser.add_argument("--tint")
    .help("Tint offset after the parameter search, in CIELAB a* units of the reference grey (negative: green, "
          "positive: magenta). Reuses <output>.json when it matches.")
    .scan<'g', float>()
    .default_value(0.0f);
  parser.add_argument("--warmth")
    .help("Warmth offset after the parameter search, in CIELAB b* units of the reference grey (negative: blue, "
          "positive: yellow). Reuses <output>.json when it matches.")
    .scan<'g', float>()
    .default_value(0.0f);
  // A bundle conversion reads the film base from the scan (film_base_detector.h: the rebate, else the thinnest film),
  // as negicc-station's Auto does, or from a capture of unexposed film (--film_base, below; the film base declarations
  // above); --film_base_rgb, which took it as numbers or from the rebate, is gone.
  parser.add_argument("--film_stock")
    .help("Film stock of the built-in profile bundle and intent adapter: ektar100, portra160 or portra400. Without it "
          "(-p with an ICC profile or a bundle whose name names no stock with an adapter), the parameter search falls "
          "back to the portra400 intent adapter.");
  parser.add_argument("--light_source")
    .help("Scanning light of the built-in profile bundle: tri (narrowband red, green and blue LED; sensor RGB and its "
          "crosstalk correction) or d65 (white; LibRaw's white-light decode to linear Rec.2020). Required for a RAW; "
          "otherwise it follows the input: d65 for a linear DNG or a JPEG, tri for three captures. A TIFF takes "
          "either: with d65 (the default) it is converted to linear Rec.2020 through its ICC profile (else 8 bits are "
          "sRGB and 16 bits linear Rec.2020); with tri it is the scanning camera's linear RGB under the TriLED light "
          "(black subtracted, no white balance or colour matrix), converted with the bundle's crosstalk correction.");
  parser.add_argument("-p", "--film_profile")
    .help("Profile in place of the built-in bundle: a profile bundle JSON with its ICC profiles inside (base64), or "
          "a split bundle's directory (a bundle.json with its ICC profiles beside it, as in profiles/), or a single "
          "ICC profile that applies to the corrected RGB values (for research).");
  parser.add_argument("-P", "--colorspace")
    .help("srgb, srgb-g10 or [ICC profile path]: the corrected RGB is converted using this as the output profile "
          "(an RGB profile). "
          "statusm, rp180 or apd: the output is the negative's density above the film base (Status M, SMPTE RP 180 "
          "printing density, or Academy Printing Density), not a picture; -o's extension picks the file: .dpx (10 "
          "bits; rp180 gives a Cineon file, apd ADX10) or .tif/.tiff (16 bits; apd gives ADX16). apd is written "
          "with the channel factors of SMPTE ST 2065-3. Density takes a "
          "bundle whose density matrices were built in, TriLED or white light, and none of the exposure, gain, "
          "curve, tone or look options; the colour balance the search finds is recorded in the file, not applied. "
          "On white light without --print the curve fit runs (--fit_curves off turns it off); it changes no pixel and "
          "gives the file's suggested balance. With --print no fit runs.");
  parser.add_argument("--print")
    .help("Print the frame on a print material: one built in, or the path of a spektrafilm print profile (a .json "
          "file). With -P srgb (or another picture output) the output is the print: Auto searches "
          "the print itself (its mid grey A as a continuous printing exposure, E held, then g and b; DESIGN.md, \"The "
          "print's search\") and the conversion renders it there; a print is made by its own search, so --target, -E, "
          "--gain_g and --gain_b are refused with it, and re-running reproduces it. It needs the bundle's "
          "grey_balance_curves. With a density "
          "output (-P statusm, rp180 or apd) the density file is written, and beside it goes a CLF per "
          "material that renders it as that print, as sRGB (<output>.<material>.clf); give it once per print material, "
          "or 'all' for those built in. Each material's print is searched as for a picture output, and its CLF "
          "carries that print's A, g and b. On a white-light bundle the print takes no curve fit. Without a name, "
          "--print lists the print materials built in. Without --print the brackets' chart profiles render the scene.")
    .append();
  parser.add_argument("--roi")
    .help("ROI crop in sensor coordinates: x y w h [rot_cw [hflip [vflip]]], or 'auto [h|v]'. 'auto' reads the film "
          "base around a negative's frame; a positive (slide) scan has none, so it usually falls back to the full "
          "visible frame (geometry.roi_source auto_fallback).")
    .nargs(1, 7);
  parser.add_argument("--film_format")
    .help("Film format: 35mm, 645, 6x6, 6x7, 6x9, or auto (the frame detector tries every format). With --roi auto it "
          "is the frame to find (default 35mm); with a given frame, when given, it is the frame the film base is read "
          "around (the rebate; default: every format). The sidecar's cli_equivalent gives the format the "
          "film base was read with whenever its detection used one, given or --roi auto's 35mm, as it gives the frame, "
          "so that its explicit --roi reads the same base.")
    .default_value(std::string("35mm"));
  parser.add_argument("--rot", "--rot_cw")
    .help("Clockwise rotation in degrees (0, 90, 180, 270).")
    .scan<'i', int>()
    .default_value(0);
  parser.add_argument("--hflip")
    .help("Horizontal flip.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--vflip")
    .help("Vertical flip.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-o", "--output")
    .required()
    .help("Output file location.");
  parser.add_argument("--dino")
    .help("Models in place of the built-in ones, as a comma-separated list of <head>:<path> entries, head one of "
          "intent (the intent adapter: adapter.onnx or a directory holding it), neutral (a neutral_head.bin) or tone "
          "(a tone_head.bin); an entry without a head is the intent adapter. Heads not named stay built in. Paths "
          "cannot contain commas; everything after the head's colon is the path.");
  parser.add_argument("--list_profiles")
    .help("List the film stocks and light sources of the built-in profile bundles and exit.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--tone_stage")
    .help("auto (default), on, off, highlights_first, least_total, or a recorded curve: "
          "curve:black,white,gamma,contrast. auto: the tone stage "
          "runs on every bundle conversion with -P srgb that is not a print, on white-light and TriLED bundles alike, "
          "whether or not the curve fit runs and with or without given --target, -E and gains (with all four given "
          "the search runs for it), and nowhere else (another or no -P, a print, a single ICC profile), so it can be "
          "passed for every scan. on: the same, and an error where it does not "
          "apply. off: no tone stage. The stage: tone heads on DINOv3 features of the final render's preview "
          "predict the frame's shadow level, highlight level and tonal range as a lab scan would have them, and one "
          "monotone L* curve, applied in the conversion itself, moves the image all the way to the predicted shadow "
          "level and half way to the highlight level and range, holding the mid-tones and each colour's a* and b*. "
          "Where the moves would clip either end beyond 0.5 % of the inner crop more than before the stage, they are "
          "scaled down; where no scaling keeps both ends, auto and on keep the shadow lift (the highlights clip), "
          "highlights_first keeps the highlights (the shadow lift is scaled down until they hold), and least_total "
          "keeps the curve that clips least at both ends together. highlights_first and least_total otherwise run as "
          "auto. "
          "A recorded curve (a sidecar's cli_equivalent) is replayed with --target, -E, --gain_g and --gain_b, "
          "without searching or predicting; one outside the stage's limits (black -15 to 15, white 80 to 110, gamma "
          "0.7 to 1.45, contrast 0.7 to 1.5) is refused.")
    .default_value(std::string("auto"));
  parser.add_argument("--dino_backend")
    .help("DINOv3 inference backend: auto (the default), cpu, or gpu (USE_CUDA=1 builds). cpu keeps everything on "
          "the CPU: the network, the search's passes over the frame, and the full-size debayer, pixel-shift merge "
          "and conversion, which a USE_CUDA=1 build otherwise runs on a CUDA device that runs its kernels (see "
          "CUDA_ARCH in the README's build section). On the device a value of the exported picture can differ from "
          "the CPU's by one 16-bit code.")
    .default_value(std::string("auto"));
  parser.add_argument("--check_renderer")
    .help("Check aid (the renderer interface, multi_target_optimizer.h PreviewRenderer): run the bundle search, the "
          "print's search, the curve fit and the tone plan through a renderer that runs each pass on the CPU and "
          "counts it, count every pass of theirs that runs outside it, and print both; exit 1 when one does, or when "
          "no pass ran. On a USE_CUDA=1 build with a CUDA device that runs its kernels, each pass also runs on the GPU "
          "from the same inputs (the search reading the CPU's) and the two are compared: exit 1 when a render's mean "
          "change in L* per pixel or a zone mean exceeds 0.05, a clip share or a histogram bin 0.01 % of the crop, or "
          "a median L* one bin, or a value is not finite. Either failure stops before anything is written. The output "
          "is the CPU's and must equal a conversion whose passes run on the CPU, bit for bit: compare the two outputs "
          "yourself (cmp).")
    .flag();
  parser.add_argument("--probe_intent")
    .help("Measurement aid (for tools that study the search): write what the intent network predicts from chosen "
          "renders of this frame to this JSON file and stop without converting. Renders every metered bracket at its "
          "metered exposure, then each --probe_params set at each --probe_stops offset. With --print on a picture "
          "output the print's search writes the file instead (its mid grey A at each --probe_stops offset; "
          "--probe_params does not apply) and the conversion goes on. Needs the bundle search to run: refused where "
          "it does not (an ICC profile; a bundle with all of --target, -E and the gains given and neither the curve "
          "fit nor the tone stage running; recorded values), and a previous run's <output>.json is not reused with "
          "it.")
    .default_value(std::string(""));
  parser.add_argument("--probe_params")
    .help("With --probe_intent: parameter sets \"target,E,g,b;...\" (target: a bracket's index, an integer; E in the "
          "search's units).")
    .default_value(std::string(""));
  parser.add_argument("--probe_stops")
    .help("With --probe_intent: exposure offsets in stops applied to each --probe_params set, e.g. -0.5 0 0.5.")
    .nargs(1, 16)
    .scan<'g', float>()
    .default_value(std::vector<float>{0.0f});
  parser.add_argument("--dino_iters")
    .help("Maximum iterations for DINOv3 parameter solving: the fixed-point steps of each bracket in the parameter "
          "search (defaults to 8), else the solver's iterations (defaults to 4).")
    .scan<'i', int>()
    .default_value(4);
  parser.add_argument("--target", "--bundle_target")
    .help("Target index (0-based) or target name in JSON profile bundle. The bundle search runs on that bracket alone "
          "and everything else is as without --target: E and the gains are solved (unless given), and the curve fit "
          "(white light) and the tone stage run. The clip fallback, which only changes the bracket, is left out.")
    .default_value(std::string(""));
  // Fitted G/B input curves (input_curves.h).
  parser.add_argument("--fit_curves")
    .help("auto (default), off, on, or recorded curves: ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB (one value, comma-"
          "separated; ten numbers, with u_loG,u_loB appended, for a curve held below u_lo). auto: as on wherever on "
          "applies (a white-light bundle search with -P srgb or srgb-g10, whether or not --target, -E, --gain_g or "
          "--gain_b pin part of it), and nothing elsewhere (TriLED, whose curves are the measured ones and are not "
          "fitted; another or no -P), so it can be passed for every scan. on: after the bundle search, fit g, b and G "
          "and B input curves jointly (Levenberg-Marquardt) so the cells the neutral head finds neutral (weighted by "
          "p^2) render grey, pulled at m / (m + 12) of full strength (m the evidence, sum of p^2), and the overall "
          "magenta-green balance agrees with the search's intent field, on the bracket it chose, each candidate at "
          "the E that keeps Auto's grid median L*, then re-meter E on that bracket's profile with its "
          "input tables composed with the curves (the combined profile); "
          "needs a white-light bundle and -P srgb or srgb-g10. A given g or b is held by the fit and a given E is "
          "not re-metered; with --target, -E, --gain_g and --gain_b all given only the curves are fitted. off: "
          "no fit, on a white-light bundle (refused where no fit can run: TriLED, an ICC profile, --print; the tone "
          "stage has its own flag, --tone_stage); a sidecar's cli_equivalent carries it, with "
          "--tone_stage off, when neither ran, so its given values replay as recorded. Recorded curves: "
          "convert through the --target bracket's combined profile with them at the given -E, --gain_g and --gain_b, "
          "without fitting or searching (a sidecar's input_curves; its cli_equivalent writes them in this form with "
          "the bracket, E, g and b); needs --target, -E, --gain_g and --gain_b. The form leads with the refs, which "
          "are positive, because the parser takes a value starting with '-' for an option. The older order "
          "aG,kG,aB,kB,refG,refB,u_medG,u_medB[,u_loG,u_loB] (no prefix) is still read when aG is not negative. "
          "White-light (d65) bundles and -P only.")
    .default_value(std::string("auto"));
  // TODO: --flat from an image file (a RAW of the empty gate, as negicc-station's Flat Field tab captures), not only a
  // flat field .npz measured from one.
  parser.add_argument("--film_base")
    .help("The film base: auto (default), scan, or a capture of unexposed film, <file>[:x,y,w,h]. scan reads it from "
          "the scan (the rebate, else the thinnest film). auto does the same, unless a previous run to this output "
          "read the base from a capture and kept it in <output>.json, with the same inputs, flat field and crosstalk "
          "correction: then that base is taken. A capture is decoded as the scan is (flat field, crosstalk "
          "correction) and scaled to the scan's exposure; x,y,w,h is the region to read in sensor pixels, and "
          "without it the brightest smooth stretch of film on the capture is found. Needs a TriLED bundle and a Sony "
          "CFA RAW.")
    .default_value(std::string("auto"));
  parser.add_argument("--flat")
    .help("Scan flat-field .npz file (measured on the empty gate), applied in sensor coordinates.");
  parser.add_argument("raw_files").nargs(1, 4);

  try {
    parser.parse_args(ac, av);
  } catch (const std::exception& err) {
    std::ostringstream help;
    help << parser;
    log_error("%s\n%s", err.what(), help.str().c_str());
    return 1;
  }
  o.argc = ac;
  o.argv = av;

  o.files = parser.get<std::vector<std::string>>("raw_files");
  const auto& files = o.files;
  if (files.size() == 2 || files.size() > 4) {
    log_error("Give one RAW, DNG, TIFF or JPEG file, three captures (one per light: red, green, blue) "
                    "or the four RAW files of a pixel shift capture\n");
    return 1;
  }
  o.input_mode = files.size() == 3 ? InputMode::kThreeShot
                 : files.size() == 1 && (negicc::is_jpeg_file(files[0]) || !libraw_opens(files[0]))
                     ? InputMode::kRgbFile
                 : files.size() == 1 && is_linear_dng(files[0]) ? InputMode::kLinearDng
                                                                : InputMode::kRaw;
  o.tiff_input = o.input_mode == InputMode::kRgbFile && !negicc::is_jpeg_file(files[0]);
  // The scanning light the input implies, for the built-in bundle when --light_source is not given: a linear DNG, a
  // JPEG or a TIFF is white light, three captures one per light are TriLED. A TIFF also takes tri (camera RGB).
  o.implied_light = o.input_mode == InputMode::kRaw ? nullptr
                    : o.input_mode == InputMode::kThreeShot ? "tri"
                                                            : "d65";
  if (o.input_mode != InputMode::kRaw && parser.is_used("--flat")) {
    log_error("--flat applies to a single CFA RAW or a pixel shift capture\n");
    return 1;
  }
  if (files.size() == 4 && !check_pixel_shift_files(files)) return 1;
  o.dino_iters = parser.get<int>("--dino_iters");
  o.dino_iters_given = parser.is_used("--dino_iters");
  o.probe_intent = parser.get<std::string>("--probe_intent");
  // The probe's sets and offsets do nothing without the probe file they go into.
  for (const char* f : {"--probe_params", "--probe_stops"}) {
    if (parser.is_used(f) && o.probe_intent.empty()) {
      log_error("%s needs --probe_intent: it sets what the probe file records\n", f);
      return 1;
    }
  }
  o.check_renderer = parser.get<bool>("--check_renderer");
  {
    std::stringstream all(parser.get<std::string>("--probe_params"));
    std::string item;
    while (std::getline(all, item, ';')) {
      std::array<float, 4> p{};
      if (item.empty()) continue;
      int target = -1, used = 0;
      if (std::sscanf(item.c_str(), "%d,%f,%f,%f%n", &target, &p[1], &p[2], &p[3], &used) != 4 ||
          used != (int)item.size()) {
        log_error("--probe_params: \"%s\" is not target,E,g,b (target an integer)\n", item.c_str());
        return 1;
      }
      p[0] = (float)target;
      if (!(std::isfinite(p[1]) && p[1] > 0.0f && std::isfinite(p[2]) && p[2] > 0.0f && std::isfinite(p[3]) &&
            p[3] > 0.0f)) {
        log_error("--probe_params: \"%s\": E, g and b must be positive numbers\n", item.c_str());
        return 1;
      }
      o.probe_params.push_back(p);
    }
    o.probe_stops = parser.get<std::vector<float>>("--probe_stops");
    if (o.probe_stops.empty()) o.probe_stops.push_back(0.0f);
    for (const float s : o.probe_stops) {
      if (!std::isfinite(s)) {
        log_error("--probe_stops must be finite numbers\n");
        return 1;
      }
    }
  }
  if (o.dino_iters <= 0) {
    log_error("--dino_iters must be positive (got %d)\n", o.dino_iters);
    return 1;
  }
  o.r_coeff = parser.get<std::vector<float>>("--r_coeff");
  o.g_coeff = parser.get<std::vector<float>>("--g_coeff");
  o.b_coeff = parser.get<std::vector<float>>("--b_coeff");
  o.row_given[0] = parser.is_used("--r_coeff");
  o.row_given[1] = parser.is_used("--g_coeff");
  o.row_given[2] = parser.is_used("--b_coeff");
  o.exposure_comp = parser.get<float>("--exposure_comp");
  o.exposure_given = parser.is_used("-E") || parser.is_used("--exposure_comp");
  o.gain_g = parser.get<float>("--gain_g");
  o.gain_b = parser.get<float>("--gain_b");
  o.gain_g_given = parser.is_used("--gain_g");
  o.gain_b_given = parser.is_used("--gain_b");
  o.brightness = parser.get<float>("--brightness");
  o.tint = parser.get<float>("--tint");
  o.warmth = parser.get<float>("--warmth");
  o.knobs_given = parser.is_used("--brightness") || parser.is_used("--tint") || parser.is_used("--warmth");
  if (!std::isfinite(o.brightness) || !std::isfinite(o.tint) || !std::isfinite(o.warmth)) {
    log_error("--brightness, --tint and --warmth must be finite numbers\n");
    return 1;
  }
  // E and the gains scale the input matrix: given, each must be a finite positive number.
  for (const auto& [name, given, value] : {std::make_tuple("-E", o.exposure_given, o.exposure_comp),
                                           std::make_tuple("--gain_g", o.gain_g_given, o.gain_g),
                                           std::make_tuple("--gain_b", o.gain_b_given, o.gain_b)}) {
    if (given && !(std::isfinite(value) && value > 0.0f)) {
      log_error("%s must be a finite number above 0 (got %g)\n", name, (double)value);
      return 1;
    }
  }
  // The rows of the input matrix: finite numbers.
  for (const auto& [name, row] : {std::make_pair("--r_coeff", &o.r_coeff), std::make_pair("--g_coeff", &o.g_coeff),
                                  std::make_pair("--b_coeff", &o.b_coeff)}) {
    for (const float c : *row) {
      if (!std::isfinite(c)) {
        log_error("%s must be finite numbers\n", name);
        return 1;
      }
    }
  }

  if (parser.is_used("--roi") && !parse_roi(parser.get<std::vector<std::string>>("--roi"), o)) return 1;
  if (parser.is_used("--rot")) {
    o.rot_cw = parser.get<int>("--rot");
  } else if (parser.is_used("--rot_cw")) {
    o.rot_cw = parser.get<int>("--rot_cw");
  }
  if (parser.get<bool>("--hflip")) o.hflip = true;
  if (parser.get<bool>("--vflip")) o.vflip = true;
  if (o.rot_cw % 90 != 0) {
    log_error("Rotation must be a multiple of 90 degrees (got %d)\n", o.rot_cw);
    return 1;
  }

  // The film format of the frame detector: --roi auto's search (35mm unless given, as negicc-station and the recorded
  // parity cases), and, when given, the film base's frame detection with a given frame. auto asks for every format.
  o.film_format_given = parser.is_used("--film_format");
  const std::string film_format_arg = parser.get<std::string>("--film_format");
  o.film_format = film_format_from_name(film_format_arg);
  if (o.film_format == FILM_FORMAT_AUTO && film_format_arg != "auto") {
    log_error("Invalid --film_format '%s' (must be 35mm, 645, 6x6, 6x7, 6x9, or auto)\n", film_format_arg.c_str());
    return 1;
  }

  if (parser.is_used("--flat")) o.flat = parser.get<std::string>("--flat");
  if (parser.is_used("--film_stock")) o.film_stock = parser.get<std::string>("--film_stock");
  if (parser.is_used("--light_source")) o.light_source = parser.get<std::string>("--light_source");
  if (parser.is_used("--film_profile")) o.film_profile = parser.get<std::string>("--film_profile");
  o.target_given = parser.is_used("--target");
  if (o.target_given) o.target = parser.get<std::string>("--target");
  if (parser.is_used("--fit_curves")) {
    // One value, so the flag never takes the RAW file that follows it (with nargs it did: "--fit_curves on in.ARW").
    // Recorded curves lead with "ref:" and the refs (refG,refB,u_medG,u_medB,aG,kG,aB,kB[,u_loG,u_loB]): the older
    // order led with aG, and a negative aG made the value start with '-', which argparse rejects as an unknown option
    // (with "--fit_curves=" too), so a sidecar's cli_equivalent did not parse. The older order is still read.
    std::string fcv = parser.get<std::string>("--fit_curves");
    const bool ref_first = fcv.compare(0, 4, "ref:") == 0;
    if (ref_first) fcv.erase(0, 4);
    const std::vector<std::string> fc = split_csv(fcv);
    if (!ref_first && fc.size() == 1 && (fc[0] == "on" || fc[0] == "off" || fc[0] == "auto")) {
      o.fit_curves = fc[0] == "on" ? Options::kFitCurvesOn : fc[0] == "off" ? Options::kFitCurvesOff
                                                                           : Options::kFitCurvesAuto;
    } else if (fc.size() == 8 || fc.size() == 10) {
      // Ten: the curve held below u_lo (InputCurves::hold), with u_loG and u_loB.
      float v[10];
      for (size_t i = 0; i < fc.size(); ++i) {
        if (!parse_number(fc[i], v[i])) {
          log_error("--fit_curves: '%s' is not a number (give off, auto, on, or "
                    "ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB with u_loG,u_loB appended for a held curve)\n",
                    fc[i].c_str());
          return 1;
        }
      }
      o.input_curves_given = true;
      // Where each number sits: ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB, or the older aG,kG,aB,kB,refG,refB,u_medG,
      // u_medB; u_loG,u_loB last in both.
      const int at_ref = ref_first ? 0 : 4, at_u_med = ref_first ? 2 : 6, at_ak = ref_first ? 4 : 0;
      for (int i = 0; i < 2; ++i) {
        o.input_curves.a[i] = v[at_ak + i * 2];
        o.input_curves.k[i] = v[at_ak + i * 2 + 1];
        o.input_curves.ref[i] = v[at_ref + i];
        o.input_curves.u_med[i] = v[at_u_med + i];
        if (fc.size() == 10) o.input_curves.u_lo[i] = v[8 + i];
        o.input_curves.hold = fc.size() == 10;
        if (!(o.input_curves.ref[i] > 0.0f)) {
          log_error("--fit_curves: refG and refB must be positive\n");
          return 1;
        }
      }
    } else {
      log_error("--fit_curves takes off, auto, on, or recorded curves: ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB (eight "
                "comma-separated numbers), ten for a held curve (and u_loG,u_loB)\n");
      return 1;
    }
  }
  if (parser.is_used("--dino")) {
    // Comma-separated entries; in each, a known head name before the first colon is the head and the rest the path,
    // else the whole entry is the intent adapter's path (so "adapter_dir" and drive letters such as "C:\..." keep
    // their meaning). A word of two or more letters before the colon that is not a head is an error.
    const std::string v = parser.get<std::string>("--dino");
    o.dino_arg = v;
    std::stringstream ss(v);
    std::string entry;
    bool seen[3] = {false, false, false};
    const char* heads[3] = {"intent", "neutral", "tone"};
    std::string* dst[3] = {&o.dino_adapter, &o.neutral_head_file, &o.tone_head_file};
    bool any = false;
    while (std::getline(ss, entry, ',')) {
      any = true;
      if (entry.empty()) {
        log_error("--dino: an empty entry in '%s'\n", v.c_str());
        return 1;
      }
      int h = 0;
      std::string path = entry;
      const size_t colon = entry.find(':');
      if (colon != std::string::npos) {
        const std::string name = entry.substr(0, colon);
        int k = -1;
        for (int i = 0; i < 3; ++i)
          if (name == heads[i]) k = i;
        bool word = name.size() >= 2;
        for (char ch : name) word = word && (std::isalpha((unsigned char)ch) || ch == '_');
        if (k >= 0) {
          h = k;
          path = entry.substr(colon + 1);
        } else if (word) {
          log_error("--dino: unknown head '%s' in '%s' (heads: intent, neutral, tone)\n", name.c_str(), entry.c_str());
          return 1;
        }
      }
      if (path.empty()) {
        log_error("--dino: the entry '%s' has no path\n", entry.c_str());
        return 1;
      }
      if (seen[h]) {
        log_error("--dino: the %s head is given twice\n", heads[h]);
        return 1;
      }
      seen[h] = true;
      *dst[h] = path;
    }
    if (!any || (!v.empty() && v.back() == ',')) {
      log_error("--dino: an empty entry in '%s'\n", v.c_str());
      return 1;
    }
  }
  o.dino_backend = parser.get<std::string>("--dino_backend");
  // Recorded numbers after a prefix (which keeps a negative first number from reading as an option).
  // --tone_stage: a mode, or recorded values to replay.
  auto replay = [&](const char* name, const char* prefix, int count, bool& given, double* out) {
    if (!parser.is_used(name)) return true;
    const std::string v = parser.get<std::string>(name);
    if (v == "auto" || v == "on" || v == "off") {
      o.tone_stage = v == "on" ? Options::kToneStageOn : v == "off" ? Options::kToneStageOff : Options::kToneStageAuto;
      return true;
    }
    if (v == "highlights_first" || v == "least_total") {
      o.tone_stage = Options::kToneStageAuto;
      o.tone_fallback = v == "highlights_first" ? kToneFallbackHighlightsFirst : kToneFallbackLeastTotal;
      return true;
    }
    const std::string pre = prefix;
    if (v.compare(0, pre.size(), pre) == 0) {
      const std::vector<std::string> items = split_csv(v.substr(pre.size()));
      bool ok = (int)items.size() == count;
      for (int k = 0; ok && k < count; ++k) ok = parse_number(items[k], out[k]);
      if (ok) {
        given = true;  // nothing is searched or predicted: the recorded values are replayed
        return true;
      }
    }
    log_error("%s takes auto, on, off, highlights_first, least_total, or recorded values: %s followed by %d "
              "comma-separated numbers, not '%s'\n",
              name, prefix, count, v.c_str());
    return false;
  };
  if (!replay("--tone_stage", "curve:", 4, o.tone_replay_given, o.tone_replay)) return 1;
  o.quality = parser.get<int>("--quality");
  o.no_crop = parser.get<bool>("--no_crop");
  o.output = parser.get<std::string>("--output");
  if (parser.is_used("--colorspace")) o.colorspace = parser.get<std::string>("--colorspace");
  o.no_metadata = parser.get<bool>("--no-metadata");

  // --film_base: auto, or <file>[:x,y,w,h]. The region follows the last colon when four whole numbers do.
  {
    const std::string fb = parser.get<std::string>("--film_base");
    if (fb.empty()) {
      log_error("--film_base takes auto, scan or a capture of unexposed film, <file>[:x,y,w,h]\n");
      return 1;
    }
    o.film_base_scan = fb == "scan";
    if (fb != "auto" && fb != "scan") {
      o.film_base_file = fb;
      const size_t colon = fb.rfind(':');
      if (colon != std::string::npos && colon + 1 < fb.size()) {
        const std::vector<std::string> items = split_csv(fb.substr(colon + 1));
        double v[4] = {0, 0, 0, 0};
        bool ok = items.size() == 4;
        for (int k = 0; ok && k < 4; ++k) ok = parse_number(items[k], v[k]) && v[k] == std::floor(v[k]);
        if (ok) {
          if (v[0] < 0 || v[1] < 0 || v[2] <= 0 || v[3] <= 0 || v[0] > 1e9 || v[1] > 1e9 || v[2] > 1e9 || v[3] > 1e9) {
            log_error("--film_base: the region x,y,w,h must lie on the sensor with a positive size (got %s)\n",
                      fb.substr(colon + 1).c_str());
            return 1;
          }
          for (int k = 0; k < 4; ++k) o.film_base_roi[k] = (int)v[k];
          o.film_base_roi_given = true;
          o.film_base_file = fb.substr(0, colon);
        } else if (fb.find_first_not_of("0123456789,.-+ ", colon + 1) == std::string::npos) {
          // Numbers after the colon that are not a region: not part of a file name.
          log_error("--film_base: the region after the colon takes four whole numbers, x,y,w,h in sensor pixels "
                    "(got '%s')\n", fb.substr(colon + 1).c_str());
          return 1;
        }
      }
      if (o.input_mode != InputMode::kRaw) {
        log_error("--film_base with a capture applies to a single CFA RAW or a pixel shift capture\n");
        return 1;
      }
    }
  }

  // The output file: a density (-P statusm, rp180, apd) goes to a .dpx or a .tif / .tiff; everything else is a TIFF.
  std::string ext;
  {
    const size_t dot = o.output.find_last_of('.');
    const size_t slash = o.output.find_last_of("/\\");
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) ext = o.output.substr(dot);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
  }
  for (int m = 0; m < kDensityMeasures; ++m) {
    if (o.colorspace == density_measure_name((DensityMeasure)m)) {
      o.density = true;
      o.density_measure = (DensityMeasure)m;
    }
  }
  if (parser.is_used("--print")) {
    o.prints = parser.get<std::vector<std::string>>("--print");
    // A picture output is one print: one material.
    if (!o.density && (o.prints.size() != 1 || o.prints.front() == "all")) {
      log_error("--print with a picture output renders one print: give one print material (a density output takes "
                "several, or 'all', one CLF each)\n");
      return 1;
    }
    const std::vector<PrintMaterial> known = builtin_print_materials();
    for (const std::string& id : o.prints) {
      // A spektrafilm print profile given as a file is loaded now, without a bundle (every check of the real load
      // runs; only the matrices for the bundle's film wait for the conversion), so a wrong file stops the run before
      // it starts.
      std::error_code ec;
      const bool print_file = is_print_file(id) && std::filesystem::is_regular_file(id, ec);
      // A path that does not exist is not a file; one that cannot be examined (permission, a link loop, a name too
      // long) is an error.
      if (ec && ec != std::errc::no_such_file_or_directory && ec != std::errc::not_a_directory) {
        log_error("--print %s: %s\n", id.c_str(), ec.message().c_str());
        return 1;
      }
      if (print_file) {
        std::string text, stem, why;
        PrintMaterial material;
        if (!read_print_file(id, text, stem, why) ||
            !load_spektrafilm_print(text, stem, ProfileBundle(), material, why)) {
          log_error("--print %s: %s\n", id.c_str(), why.c_str());
          return 1;
        }
      }
      bool found = id == "all" || print_file;
      std::string names;
      for (const PrintMaterial& m : known) {
        found = found || m.id == id;
        names += " " + m.id;
      }
      if (!found) {
        log_error("--print %s: no such print material (built in:%s; or the path of a spektrafilm print profile)\n",
                  id.c_str(), names.c_str());
        return 1;
      }
    }
  }
  // A print is made by its own search: no given bracket, exposure or gains with --print, on a picture or a density
  // output, so that no print differs silently from its Auto; re-running the same command reproduces it.
  if (!o.prints.empty() && (o.target_given || o.exposure_given || o.gain_g_given || o.gain_b_given)) {
    log_error("--target, -E, --gain_g and --gain_b do not apply with --print: a print is made by its own search, and "
              "re-running reproduces it\n");
    return 1;
  }
  if (!o.density) {
    if (ext == ".dpx") {
      log_error("A .dpx output holds density: give -P statusm, rp180 or apd\n");
      return 1;
    }
    return 0;
  }
  if (ext == ".dpx") {
    o.output_dpx = true;
  } else if (ext != ".tif" && ext != ".tiff") {
    log_error("-P %s writes density to a .dpx (10 bits) or a .tif / .tiff (16 bits) file, by -o's extension (got "
              "'%s')\n", o.colorspace.c_str(), o.output.c_str());
    return 1;
  }
  // Density is the negative as measured: no given bracket, exposure or gains, no curves, no tone stage and no look.
  // --fit_curves off, auto or on is taken here (the pipeline code below refuses it on a TriLED bundle and with
  // --print): on white light without --print the fit gives the file's suggested balance, not its pixels. Recorded
  // curves replay a bracket, E and gains, which a density output does not take.
  {
    struct Refused { const char* flag; bool used; };
    const bool fit_set = o.input_curves_given;
    const bool tone_set = parser.is_used("--tone_stage") && parser.get<std::string>("--tone_stage") != "auto";
    const Refused refused[] = {
        {"--target", o.target_given},
        {"-E", o.exposure_given},
        {"--gain_g", o.gain_g_given},
        {"--gain_b", o.gain_b_given},
        // The knobs move a print's CLF (each material's own print search): they apply with --print.
        {"--brightness", parser.is_used("--brightness") && o.prints.empty()},
        {"--tint", parser.is_used("--tint") && o.prints.empty()},
        {"--warmth", parser.is_used("--warmth") && o.prints.empty()},
        {"--fit_curves", fit_set},
        {"--tone_stage", tone_set},
    };
    for (const Refused& r : refused) {
      if (r.used) {
        log_error("%s does not apply to a density output (-P %s): it writes the negative's density above the film "
                  "base, with no given bracket, exposure, gains, curves, tone stage or look (the knobs apply to its "
                  "print CLFs, with --print)\n", r.flag, o.colorspace.c_str());
        return 1;
      }
    }
  }
  // The conversion takes the film base's merged matrix as it is (convert in the pipeline code below: exposure and gains
  // of 1). The parameter search still runs, as for a picture: its exposure and gains are not applied, they are recorded
  // in the file as a suggested balance (DensityRecord::has_balance in the print declarations above). On a white-light
  // bundle without --print the curve fit runs too (--fit_curves stays auto): its curves are not applied, and its E, g
  // and b are the suggested balance. With --print each material's own print search gives its CLF, with no fit. A
  // narrow-band bundle has no curve fit.
  o.tone_stage = Options::kToneStageOff;
  return 0;
}

// ====================================================================================================================
// Output: code
// The 16-bit TIFF with its ICC profile, and the JSON sidecar beside it.
// ====================================================================================================================

int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc, const uint16_t* rgb3) {
  unsigned* output_profile = NULL;
  unsigned profile_size = 0;
  bool dynamically_allocated = false;

  if (!attach_profile.empty()) {
    // If the profile to attach is not the psuedo "srgb" profile, the profile will be attached to the TIFF.
    // This is a hack to force LibRaw write the ICC profile in the TIFF without conversion.
    log_info("Attaching profile: %s\n", attach_profile.c_str());
    if (attach_profile == "srgb") {
      output_profile = reinterpret_cast<unsigned int*>(sRGB_elle_V2_srgbtrc_icc);
      profile_size = sRGB_elle_V2_srgbtrc_icc_len;
    } else if (attach_profile == "srgb-g10" || attach_profile == "srgb-linear") {
      output_profile = reinterpret_cast<unsigned int*>(sRGB_elle_V2_g10_icc);
      profile_size = sRGB_elle_V2_g10_icc_len;
    } else {
      if (read_profile(attach_profile, &output_profile, &profile_size)) {
        return -1;
      }
      dynamically_allocated = true;
    }
  } else if (!embedded_icc.empty()) {
    log_info("Attaching profile: [bundle embedded ICC] (%u bytes)\n", (unsigned)embedded_icc.size());
    output_profile = reinterpret_cast<unsigned int*>(const_cast<uint8_t*>(embedded_icc.data()));
    profile_size = (unsigned)embedded_icc.size();
  }

  const unsigned height = proc->imgdata.sizes.iheight;
  const unsigned width = proc->imgdata.sizes.iwidth;
  // A classic TIFF addresses its data with 32-bit offsets: the header, the profile and the strip of three 16-bit
  // samples a pixel must end within them (as write_density_tiff checks, the print code above).
  if (sizeof(tiff_hdr) + (uint64_t)profile_size + (uint64_t)width * height * 6 > 0xffffff00ull) {
    log_error("The image is too large for a TIFF file (%ux%u)\n", width, height);
    if (dynamically_allocated && output_profile) free(output_profile);
    return -1;
  }
  struct tiff_hdr header;
  tiff_head(proc, &header, profile_size);
  auto* fp = fopen(output.c_str(), "wb+");
  if (!fp) {
    log_error("Cannot open %s for writing\n", output.c_str());
    if (dynamically_allocated && output_profile) free(output_profile);
    return -1;
  }
  bool ok = fwrite(&header, sizeof(header), 1, fp) == 1;
  if (ok && profile_size) {
    ok = fwrite(output_profile, profile_size, 1, fp) == 1;
  }
  if (dynamically_allocated && output_profile) {
    free(output_profile);
  }

  if (ok && rgb3) {
    ok = fwrite(rgb3, 3 * sizeof(uint16_t), (size_t)width * height, fp) == (size_t)width * height;
  } else if (ok) {
    std::vector<ushort> row_buf(width * 3);
    for (unsigned row = 0; ok && row < height; ++row) {
      for (unsigned col = 0; col < width; ++col) {
        row_buf[col * 3]     = proc->imgdata.image[row * width + col][0];
        row_buf[col * 3 + 1] = proc->imgdata.image[row * width + col][1];
        row_buf[col * 3 + 2] = proc->imgdata.image[row * width + col][2];
      }
      ok = fwrite(row_buf.data(), 3 * sizeof(ushort), width, fp) == width;
    }
  }
  ok = (fclose(fp) == 0) && ok;
  if (!ok) {
    // A full disk or another write error: the truncated file is removed (not a device or pipe given as the output).
    log_error("Failed to write %s\n", output.c_str());
    std::error_code ec;
    if (std::filesystem::is_regular_file(output, ec)) unlink(output.c_str());
    return -1;
  }
  return 0;
}

#ifndef NEGICC_GIT_VERSION
#define NEGICC_GIT_VERSION "master"
#endif

using ordered_json = nlohmann::ordered_json;

// A number as the sidecar has always written it: printed with fmt, then read back (an integer when it prints as one),
// so the values do not change with the writer; null when it is not finite.
static ordered_json num(double val, const char* fmt = "%.6g") {
  if (!std::isfinite(val)) return nullptr;
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  if (!strpbrk(buf, ".eE")) return strtoll(buf, nullptr, 10);
  return strtod(buf, nullptr);
}

// A number as text, printed with fmt (the cli_equivalent's arguments).
static std::string num_str(double val, const char* fmt) {
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  return buf;
}

// A tone curve in full (17 digits): the curve the image went through, exactly.
static ordered_json tone_curve_json(const ToneCurve& q) {
  return {{"black", num(q.black, "%.17g")}, {"white", num(q.white, "%.17g")}, {"gamma", num(q.gamma, "%.17g")},
          {"contrast", num(q.contrast, "%.17g")}};
}

static ordered_json rect_json(const FilmRect& r) {
  return {{"cx", num(r.cx, "%.1f")}, {"cy", num(r.cy, "%.1f")}, {"w", num(r.w, "%.1f")}, {"h", num(r.h, "%.1f")},
          {"angle_deg", num(r.angle_deg, "%.3f")}};
}

static ordered_json frame_detection_json(const FrameDetection& fd) {
  // Sensor pixel coordinates throughout (the detector ran on the RAW before the decode). aspect and format describe
  // the frame as the previous detector did ("3:2", ...); film_format is the format the detection used. axis is "h" or
  // "v", or null when the detector left it undecided (a single square frame with nothing to orient it).
  const FilmRect& pr = fd.primary.rect;
  const float aspect = std::max(pr.w, pr.h) / std::max(1.0f, std::min(pr.w, pr.h));
  const char* ratio = "-";
  switch (fd.format) {
    case FILM_35MM: ratio = "3:2"; break;
    case FILM_645: ratio = "4:3"; break;
    case FILM_6X6: ratio = "1:1"; break;
    case FILM_6X7: ratio = "7:6"; break;
    case FILM_6X9: ratio = "3:2"; break;
    default: break;
  }
  ordered_json j;
  j["ok"] = fd.ok;
  j["axis"] = fd.axis == 'h' ? ordered_json("h") : fd.axis == 'v' ? ordered_json("v") : ordered_json(nullptr);
  j["confidence"] = num(fd.conf, "%.2f");
  j["low_confidence"] = fd.conf < kLowFrameConfidence;
  j["reason"] = fd.reason;
  ordered_json gaps = ordered_json::array();
  for (int g = 0; g < 2; ++g)
    if (fd.gaps[g][0] >= 0) gaps.push_back({fd.gaps[g][0], fd.gaps[g][1]});
  j["gaps"] = gaps;
  // segments: the frames on the strip, in strip order (secondaries are sorted by index, negative before the primary);
  // selected: the primary's position among them.
  int selected = 0;
  for (const FilmFrame& f : fd.secondaries) selected += f.index < 0 ? 1 : 0;
  j["segments"] = 1 + (int)fd.secondaries.size();
  j["n_gaps"] = fd.n_gaps;
  j["selected"] = selected;
  j["aspect"] = num(aspect, "%.4g");
  j["format"] = ratio;
  j["film_format"] = film_format_name(fd.format);
  j["orientation"] = fd.orientation;
  j["rotation_deg"] = num(fd.rotation_deg, "%.3f");
  j["px_per_mm"] = num(fd.px_per_mm, "%.2f");
  j["light_clipped"] = fd.light_clipped;
  j["crop"] = {fd.crop[0], fd.crop[1], fd.crop[2], fd.crop[3]};
  j["primary"] = rect_json(pr);
  ordered_json sec = ordered_json::array();
  for (const FilmFrame& f : fd.secondaries)
    sec.push_back({{"index", f.index}, {"rect", rect_json(f.rect)},
                   {"visible", {f.visible[0], f.visible[1], f.visible[2], f.visible[3]}},
                   {"visible_fraction", num(f.visible_fraction, "%.3f")}});
  j["secondaries"] = sec;
  j["film_base"] = fd.has_film_base ? rect_json(fd.film_base) : ordered_json(nullptr);
  j["base_patch"] = fd.has_base ? rect_json(fd.base_patch) : ordered_json(nullptr);
  j["sprockets"] = (int)fd.sprockets.size();
  return j;
}

// The search's own result: e, g, b and residual are its values before any curve fit, which a run with fitted curves
// converts at other values (input_curves.remeter's e, g and b, after the fit and the re-meter; the sidecar loader
// reads those as the knobs' base when they are there, and auto's otherwise).
static ordered_json auto_json(const ParamSearchResult& a, const int* frame_half, const Sidecar& sc) {
  ordered_json j;
  j["target"] = a.target_name;
  j["target_idx"] = a.target_idx;
  j["e"] = num(a.params.e, "%.9g");
  j["g"] = num(a.params.g, "%.9g");
  j["b"] = num(a.params.b, "%.9g");
  j["residual"] = num(a.residual, "%.6g");
  j["status"] = to_string(a.status);
  j["e_norm"] = num(a.e_norm, "%.9g");
  ordered_json m = ordered_json::array();
  for (int i = 0; i < 9; ++i) m.push_back(num(a.input_matrix[i], "%.9g"));
  j["input_matrix"] = m;
  j["frame_half"] = {frame_half[0], frame_half[1], frame_half[2], frame_half[3]};
  j["g_lo"] = num(a.g_lo, "%.6g");
  j["g_hi"] = num(a.g_hi, "%.6g");
  j["scene_ev"] = num(a.scene_ev, "%.4g");
  // The record keeps its historical keys and values: branch R is the clip fallback being tried or the highlight
  // pullback moving the result (the pullback used to report through the clip fallback's outcome), and changed is
  // either one moving the result.
  j["rescue"] = {{"branch", a.clip_fallback == ClipFallbackOutcome::kNotNeeded && !a.highlight_pullback ? "N" : "R"},
                 {"white_clip", num(a.clip_fallback_white_clip, "%.6g")},
                 {"changed", a.clip_fallback == ClipFallbackOutcome::kTaken || a.highlight_pullback}};
  ordered_json screening = ordered_json::array();
  for (const auto& r : a.screening)
    screening.push_back({{"target_idx", r.target_idx}, {"e", num(r.params.e)}, {"g", num(r.params.g)},
                         {"b", num(r.params.b)}, {"intent_loss", num(r.intent_loss)}});
  j["screening"] = screening;
  ordered_json tournament = ordered_json::array();
  for (const auto& r : a.tournament)
    tournament.push_back({{"target_idx", r.target_idx}, {"e", num(r.params.e)}, {"g", num(r.params.g)},
                          {"b", num(r.params.b)}, {"residual", num(r.residual)}, {"status", to_string(r.status)},
                          {"iters", r.steps}});
  j["tournament"] = tournament;
  j["model"] = sc.dino_model;
  j["backend"] = sc.dino_backend;
  j["inferences"] = a.inferences;
  j["time_ms"] = num(sc.t_auto_ms, "%.1f");
  if (sc.search_renderer != "cpu") j["renderer"] = sc.search_renderer;
  if (!sc.shadow_json_str.empty()) j["shadow_alternative"] = ordered_json::parse(sc.shadow_json_str);
  return j;
}

// One step of the single-profile search: its parameters, losses and penalties.
template <typename It>
static ordered_json dino_step_json(const It& it, bool with_residual) {
  ordered_json j = {{"e", num(it.e)}, {"g", num(it.g)}, {"b", num(it.b)}};
  if (with_residual) j["residual"] = num(it.residual);
  j["loss_L"] = num(it.loss_L);
  j["loss_a"] = num(it.loss_a);
  j["loss_b"] = num(it.loss_b);
  j["clip_pct"] = num(it.clip_pct);
  j["penalty_clip"] = num(it.penalty_clip);
  j["penalty_anchor"] = num(it.penalty_anchor);
  j["objective"] = num(it.objective);
  return j;
}

static ordered_json dino_json(const DinoSolveResult& d, const Sidecar& sc) {
  ordered_json j;
  j["model"] = sc.dino_model;
  j["backend"] = sc.dino_backend;
  j["grid"] = {d.grid_w, d.grid_h};
  j["target"] = {d.target_w, d.target_h};
  j["e_center"] = num(d.e_center);
  j["e_bounds"] = {num(d.bounds_lo), num(d.bounds_hi)};
  j["status"] = d.status;
  j["best_iteration"] = d.best_iteration;
  j["inferences"] = d.total_inferences;
  j["solved_relative"] = {{"e", num(d.solved_relative.e)}, {"g", num(d.solved_relative.g)},
                          {"b", num(d.solved_relative.b)}};
  const bool has_bootstrap = !d.iterations.empty() && d.iterations[0].tag == 'b';
  j["bootstrap"] = has_bootstrap ? dino_step_json(d.iterations[0], false) : ordered_json(nullptr);
  ordered_json its = ordered_json::array();
  for (const auto& it : d.iterations)
    if (it.tag != 'b') its.push_back(dino_step_json(it, true));
  j["iterations"] = its;
  ordered_json tone = ordered_json::array();
  for (const auto& t : d.tone_mass) tone.push_back(num(t));
  j["tone_mass"] = tone;
  return j;
}

// The model's short name (the backbone's directory, e.g. dinov3-small), not the path the build embedded it from.
static std::string model_stem(std::string path) {
  const std::string kBuiltIn = "[built-in] ";
  if (path.rfind(kBuiltIn, 0) == 0) path = path.substr(kBuiltIn.size());
  const size_t slash = path.find_last_of("/\\");
  std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
  if (slash != std::string::npos && file.rfind("backbone.", 0) == 0) {
    const std::string dir = path.substr(0, slash);
    const size_t s2 = dir.find_last_of("/\\");
    return s2 == std::string::npos ? dir : dir.substr(s2 + 1);
  }
  const size_t dot = file.find_last_of('.');
  return dot == std::string::npos ? file : file.substr(0, dot);
}

// The intent field the parameter search converted against: per DINOv3 grid cell, row-major over the oriented frame,
// the Lab colour (mu) and its tolerance (sigma) of the chosen step. The same block as the station's sidecar. Null when
// no search ran. source: "auto" (the bundle search or the print's search, on the frame's inner crop_ratio crop): a*
// and b* as the intent adapter predicted them from the chosen step's render, L* the frame's lightness target (the
// bootstrap field's; DESIGN.md 4.2); "dino" (the single-bracket search, on the whole frame): its best step's
// prediction in all three.
static ordered_json intent_field_json(const std::vector<float>& mu, const std::vector<float>& sigma, int grid_w,
                                      int grid_h, int target_idx, const char* source, float crop_ratio,
                                      const Sidecar& sc) {
  const size_t n = static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h);
  if (n == 0 || mu.size() != n * 3 || sigma.size() != n * 3) return nullptr;
  ordered_json j;
  j["version"] = 1;
  j["grid"] = {{"rows", grid_h}, {"cols", grid_w}, {"patch_px", 16}, {"order", "row-major, oriented frame"}};
  j["source"] = source;
  j["crop_ratio"] = num(crop_ratio, "%.3g");
  j["target_idx"] = target_idx >= 0 ? ordered_json(target_idx) : ordered_json(nullptr);
  j["model"] = {{"name", model_stem(sc.dino_model)},
                {"path", sc.dino_model},
                {"backend", sc.dino_backend},
                {"negicc", NEGICC_GIT_VERSION}};
  ordered_json m = ordered_json::array(), s = ordered_json::array();
  for (size_t i = 0; i < n; ++i) {
    m.push_back({num(mu[i * 3], "%.2f"), num(mu[i * 3 + 1], "%.2f"), num(mu[i * 3 + 2], "%.2f")});
    s.push_back({num(sigma[i * 3], "%.4g"), num(sigma[i * 3 + 1], "%.4g"), num(sigma[i * 3 + 2], "%.4g")});
  }
  j["mu_lab"] = m;
  j["sigma_lab"] = s;
  return j;
}

// The fitted G/B input curves of the conversion's bracket (input_curves.h): what rebuilds
// the combined profile (the bracket's profile with its G and B input tables composed with the curves), the fit, and
// the parameters before it. The same block as the station's sidecar; numbers to 9 significant digits, so a replay
// (--fit_curves with the recorded numbers, as the cli_equivalent writes them) rebuilds the same tables.
static ordered_json input_curves_json(const Sidecar& sc) {
  const negicc::InputCurves& c = *sc.input_curves;
  auto n9 = [](double v) { return num(v, "%.9g"); };
  ordered_json j;
  j["version"] = 1;
  j["space"] = "profile input, after the input matrix and the knee";
  j["source"] = sc.input_curves_source;
  j["target_idx"] = sc.bundle_target_idx >= 0 ? ordered_json(sc.bundle_target_idx) : ordered_json(nullptr);
  ordered_json ch;
  const char* names[2] = {"G", "B"};
  // u_lo is known from a fit, and from recorded curves only when held (the ten-number form): an eight-number replay
  // has none, and before 0 was written for it as if it had been measured.
  const bool u_lo_known = sc.input_curves_source != "cli" || c.hold;
  for (int i = 0; i < 2; ++i) {
    ch[names[i]] = {{"a", n9(c.a[i])}, {"k", n9(c.k[i])}, {"ref", n9(c.ref[i])}, {"u_med", n9(c.u_med[i])}};
    if (u_lo_known) ch[names[i]]["u_lo"] = n9(c.u_lo[i]);
  }
  // Whether the export renders through curves. Not when the fit fell back (fallback: clip, shape or intent_lightness,
  // as remeter.fallback): the bracket's own tables, so no channels and no hold.
  const bool applied = c.active();
  const WhiteLightFitResult* fr = sc.curve_fit;
  j["applied"] = applied;
  j["fallback"] = fr && fr->fallback ? ordered_json(fr->fallback) : ordered_json(nullptr);
  j["channels"] = applied ? ch : ordered_json(nullptr);
  // below u_lo the curve keeps its offset (InputCurves::hold); replay needs u_lo then
  j["hold"] = applied ? ordered_json(c.hold) : ordered_json(nullptr);
  if (const WhiteLightFitResult* f = sc.curve_fit) {
    // The options the fit ran with, all of them.
    const negicc::CurveFitOptions& op = f->options;
    j["bound"] = n9(op.curve_bound);
    j["options"] = {{"curve_bound", n9(op.curve_bound)}, {"max_dev_stops", n9(op.max_dev_stops)},
                    {"g_bound_stops", n9(op.g_bound_stops)}, {"b_bound_stops", n9(op.b_bound_stops)},
                    {"joint_lambda", n9(op.joint_lambda)},
                    {"hold_slope", op.hold_slope},
                    {"solve_e", op.solve_e}, {"e_tol_L", n9(op.e_tol_L)}};
    j["fit"] = {{"loss_before", n9(f->fit.loss_before)}, {"loss_after", n9(f->fit.loss_after)},
                // The fifth residual, in a*: the intent field's standard error, and its mean a* residual less
                // the lever share and the lean, at Auto and at the result (zero when there was no fit).
                {"intent_se", n9(f->fit.intent_se)}, {"intent_auto", n9(f->fit.intent_auto)},
                {"intent_after", n9(f->fit.intent_after)},
                {"evaluations", f->fit.evaluations}, {"g", n9(f->fit.g)}, {"b", n9(f->fit.b)},
                {"solver", "joint"}, {"e", n9(f->fit.e)}, {"renders", f->fit.renders},
                {"iterations", f->fit.iterations}, {"converged", f->fit.converged},
                // No longer varied (the coordinate polish and multiple seeds are gone); kept for readers.
                {"polish_evaluations", 0}, {"start", 0}, {"passes", f->fit_passes},
                {"intent_repredicted", f->intent_repredicted}};
    j["params_before"] = {{"e", n9(f->params_before.e)}, {"g", n9(f->params_before.g)}, {"b", n9(f->params_before.b)},
                          {"x_hi", n9(f->params_before.shoulder)}};
    j["max_dev_stops"] = n9(op.max_dev_stops);
    // Measured on the export's render: its encoding and the roll-off it measures (x_hi_auto at Auto's parameters,
    // which the first fit ran with; x_hi_fit, which the last fit ran with; x_hi at the result, which the export uses).
    j["render"] = {{"encoding", f->encoding == ENCODE_SRGB ? "srgb" : f->encoding == ENCODE_SRGB_LINEAR ? "srgb-linear"
                                                                                                       : "custom"},
                   {"x_hi_auto", n9(f->x_hi_auto)}, {"x_hi_fit", n9(f->x_hi_fit)}};
    j["remeter"] = {{"e", n9(f->params.e)}, {"g", n9(f->params.g)}, {"b", n9(f->params.b)},
                    {"target_median_L", n9(f->target_median_L)}, {"median_L", n9(f->median_L)},
                    {"x_hi", n9(f->x_hi)}, {"passes", f->remeter_passes}, {"converged", f->remeter_converged},
                    {"e_lo", n9(f->e_lo)}, {"e_hi", n9(f->e_hi)},
                    {"shape_scale", n9(f->shape_scale)},
                    {"shape_scale_ch", {n9(f->shape_scale_ch[0]), n9(f->shape_scale_ch[1])}},
                    {"fallback", f->fallback ? ordered_json(f->fallback) : ordered_json(nullptr)},
                    {"intent_lightness_failed", f->intent_lightness_failed},
                    // On "intent_lightness": E re-metered to this grid median L* (clipped: a bound of E stopped it).
                    {"fallback_remeter", f->intent_lightness_failed
                                             ? ordered_json{{"target_L", n9(f->fallback_target_L)},
                                                            {"clipped", f->fallback_clipped},
                                                            {"e_moved", f->fallback_e_moved}}
                                             : ordered_json(nullptr)},
                    {"cells_excluded", f->cells_excluded}, {"cells_total", f->cells_total},
                    {"loss_L", n9(f->parts_export.L)}, {"loss_ab", n9(f->parts_export.ab)},
                    {"auto_loss_L", n9(f->parts_auto.L)}, {"auto_loss_ab", n9(f->parts_auto.ab)},
                    {"fit_loss_L", n9(f->parts_fit.L)}, {"fit_loss_ab", n9(f->parts_fit.ab)},
                    {"neutral", ordered_json{{"anchors", f->neutral_anchors ? "head" : "none"},
                                             {"m", f->neutral_export.m},
                                             {"auto_a", n9(f->neutral_auto.a_bar)},
                                             {"auto_b", n9(f->neutral_auto.b_bar)},
                                             {"export_a", n9(f->neutral_export.a_bar)},
                                             {"export_b", n9(f->neutral_export.b_bar)},
                                             {"auto_ka", n9(f->neutral_auto.k_a)},
                                             {"auto_kb", n9(f->neutral_auto.k_b)},
                                             {"export_ka", n9(f->neutral_export.k_a)},
                                             {"export_kb", n9(f->neutral_export.k_b)},
                                             // The evidence mass (sum of p^2 over the cells in range, as the
                                             // last fit pass saw them at its roll-off; earlier builds wrote it at
                                             // Auto's roll-off, up to 2 % apart), the shrink weight
                                             // m / (m + 12) inside the residuals, the anchors' L* spread and
                                             // slope share, and their mean a*'s standard deviation.
                                             {"evidence", n9(f->fit.evidence)},
                                             {"shrink", n9(f->fit.shrink)},
                                             {"span_L", n9(f->fit.span_L)},
                                             {"slope_share", n9(f->fit.slope_share)},
                                             {"anchor_sd", n9(f->fit.anchor_sd)}}},
                    // The never-greener guard of the last fit pass: Auto's and the fit's mean a* over all cells
                    // and weighted by p^2, whether each counted, and whether the guard pulled the fit.
                    {"guard", [&] {
                       const negicc::GreenGuard& gd = f->fit.guard;
                       return ordered_json{{"tol", n9(negicc::kGreenGuardTolAb)},
                                           {"counted", {gd.counted[0], gd.counted[1]}},
                                           {"auto_a", {n9(gd.auto_a[0]), n9(gd.auto_a[1])}},
                                           {"fit_a", {n9(gd.fit_a[0]), n9(gd.fit_a[1])}},
                                           {"binding", gd.binding},
                                           // The fit ran first without the guard; refit: that result broke
                                           // the bound and this is the guarded re-run.
                                           {"refit", gd.refit},
                                           {"free_fit_a", {n9(gd.free_fit_a[0]), n9(gd.free_fit_a[1])}},
                                           {"p_source", f->guard_p_source}};
                     }()},
                    {"time_ms", num(f->time_ms, "%.1f")},
                    {"loss", n9(f->loss_export)}};
  } else {
    j["bound"] = nullptr;
    j["options"] = nullptr;
    j["fit"] = nullptr;
    j["params_before"] = nullptr;
    j["max_dev_stops"] = nullptr;
    j["render"] = nullptr;
    j["remeter"] = nullptr;
  }
  return j;
}

const char* neg_process_version() { return NEGICC_GIT_VERSION; }

FileFingerprint memory_fingerprint(const void* data, size_t size) {
  FileFingerprint fp;
  fp.ok = data != nullptr;
  fp.size = size;
  fp.crc = fp.ok ? (uint32_t)crc32_z(crc32(0L, Z_NULL, 0), reinterpret_cast<const Bytef*>(data), (z_size_t)size) : 0;
  return fp;
}

std::string utf8_replaced(const std::string& s) {
  using nlohmann::ordered_json;
  return ordered_json::parse(ordered_json(s).dump(-1, ' ', false, ordered_json::error_handler_t::replace))
      .get<std::string>();
}

bool file_fingerprint(const std::string& path, uintmax_t& size, uint32_t& crc) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::vector<char> buf(1 << 20);
  uLong c = crc32(0L, Z_NULL, 0);
  size = 0;
  while (f) {
    f.read(buf.data(), (std::streamsize)buf.size());
    const std::streamsize n = f.gcount();
    if (n <= 0) break;
    c = crc32_z(c, reinterpret_cast<const Bytef*>(buf.data()), (z_size_t)n);
    size += (uintmax_t)n;
  }
  crc = (uint32_t)c;
  return !f.bad();
}

bool write_json_sidecar(const Sidecar& sc) {
  ordered_json j;
  j["schema"] = "negicc.neg_process.sidecar/1";
  j["tool"] = {{"name", "neg_process"}, {"version", NEGICC_GIT_VERSION}};
  ordered_json argv = ordered_json::array();
  for (int i = 0; i < sc.argc; ++i) argv.push_back(sc.argv[i]);
  j["argv"] = argv;

  std::string camera = "Unknown";
  if (sc.proc) {
    const std::string make = sc.proc->imgdata.idata.make;
    const std::string model = sc.proc->imgdata.idata.model;
    if (!make.empty() && !model.empty()) camera = model.find(make) == 0 ? model : make + " " + model;
    else if (!model.empty()) camera = model;
    else if (!make.empty()) camera = make;
  }
  // Each input's size and CRC-32, so a later run reusing this sidecar (the knobs) can tell the same file from another
  // saved under the same name, also when the file was copied or downloaded again in between (a new modification time).
  ordered_json stats = ordered_json::array();
  for (size_t i = 0; i < sc.files.size(); ++i) {
    FileFingerprint fp;
    if (sc.file_fingerprints && i < sc.file_fingerprints->size() && (*sc.file_fingerprints)[i].ok) {
      fp = (*sc.file_fingerprints)[i];  // hashed once already, by the sidecar loader
    } else {
      fp.ok = file_fingerprint(sc.files[i], fp.size, fp.crc);
    }
    stats.push_back(fp.ok ? ordered_json({{"size", fp.size}, {"crc32", fp.crc}}) : ordered_json(nullptr));
  }
  j["inputs"] = {{"raw_files", sc.files},
                 {"raw_file_stats", stats},
                 {"camera", camera},
                 {"iso", sc.proc ? num(sc.proc->imgdata.other.iso_speed) : ordered_json(0)},
                 {"shutter", sc.proc ? num(sc.proc->imgdata.other.shutter) : ordered_json(0)}};
  // What the conversion used, as its log's block says it (conversion_used in the pipeline code below).
  j["used"] = sc.used_json.empty() ? ordered_json(nullptr) : ordered_json::parse(sc.used_json, nullptr, false);

  ordered_json geom;
  geom["sensor_size"] = {sc.orig_w, sc.orig_h};
  geom["half_size"] = sc.half_size;
  geom["roi_source"] = sc.roi_source;
  geom["roi"] = sc.has_roi ? ordered_json({sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3]}) : ordered_json(nullptr);
  geom["rot_cw"] = sc.rot_cw;
  geom["hflip"] = sc.hflip;
  geom["vflip"] = sc.vflip;
  geom["output_size"] = {sc.out_w, sc.out_h};
  j["geometry"] = geom;

  j["frame_detection"] = sc.frame_det ? frame_detection_json(*sc.frame_det) : ordered_json(nullptr);

  ordered_json fb;
  fb["source"] = sc.film_base_source;
  fb["applied"] = sc.film_base_applied;
  const negicc::FilmBaseDetection* d = sc.film_base_detection;
  // A base given by a capture is taken from this record by a later run (--film_base auto), so it is written with the
  // digits that give its doubles back; a base read from the scan keeps the ten digits it has always had.
  const bool given_base = sc.film_base_source == "capture" || sc.film_base_source == "sidecar";
  const char* base_fmt = given_base ? "%.17g" : "%.10g";
  auto num3 = [base_fmt](const double* v) {
    return ordered_json{num(v[0], base_fmt), num(v[1], base_fmt), num(v[2], base_fmt)};
  };
  if (sc.film_base_source == "none") {
    fb["film_base_rgb"] = nullptr;
  } else if (d && d->found) {
    fb["film_base_rgb"] = num3(d->rgb);
  } else {
    fb["film_base_rgb"] = sc.film_base_rgb;
  }
  // Read from the scan (film_base_detector.h): the base in the half-size decode's units, the rebate patch sampled and
  // the frame detector's record.
  const bool rebate = d && d->found && d->source == "rebate";
  fb["decode_rgb"] = d && d->found ? num3(d->decode_rgb) : ordered_json(nullptr);
  fb["box"] = rebate ? ordered_json{d->box[0], d->box[1], d->box[2], d->box[3]} : ordered_json(nullptr);
  fb["std_density"] = rebate ? num(d->std_d, "%.6g") : ordered_json(nullptr);
  if (d && d->has_detector) {
    ordered_json det;
    det["reason"] = d->reason;
    det["frame_conf"] = num(d->frame_conf, "%.3f");
    det["light_clipped"] = d->light_clipped;
    det["thinnest_over_patch_stops"] =
        d->has_outshone ? num(d->thinnest_over_patch_stops, "%.4f") : ordered_json(nullptr);
    fb["detector"] = det;
  } else {
    fb["detector"] = nullptr;
  }
  if (!sc.film_base_capture_json.empty()) {
    fb["capture"] = ordered_json::parse(sc.film_base_capture_json, nullptr, false);
  }
  // A print on a base that is not a capture: film_base_rgb is the base raised to be no darker than the picture's
  // thinnest film (raise_base_to_print_floor, with print_floor_factor's factor); the base as found and the factor.
  if (sc.print_floor_applied) {
    fb["print_floor"] = {{"units", "the conversion's film base space at the scan's exposure (a bundle's base is "
                                    "scaled to it; film_base_rgb is the bundle's at its own exposure when nothing was "
                                    "raised)"},
                         {"found_source", sc.print_floor_found_source},
                         {"found_rgb", {num(sc.print_floor_found_rgb[0], "%.10g"),
                                        num(sc.print_floor_found_rgb[1], "%.10g"),
                                        num(sc.print_floor_found_rgb[2], "%.10g")}},
                         {"factor", num(sc.print_floor_k, "%.6g")},
                         {"frame", sc.print_floor_frame},
                         {"frame_conf", sc.print_floor_frame_conf < 0.0f ? ordered_json(nullptr)
                                                                         : num(sc.print_floor_frame_conf, "%.4g")}};
  }
  fb["profile_film_base_rgb"] = sc.profile_film_base_rgb;
  // The gains the white-light decode put on the camera channels to balance them on the film base (load_raw);
  // film_base_rgb and decode_rgb are read after them. Nine significant digits, which a single-precision gain
  // round-trips through: with six decimals a replay from the sidecar (negicc-station's) balanced the mosaic with other
  // gains.
  const float* k = sc.balance_gains;
  fb["balance_gains"] =
      k ? ordered_json{num(k[0], "%.9g"), num(k[1], "%.9g"), num(k[2], "%.9g")} : ordered_json(nullptr);
  j["film_base"] = fb;

  if (sc.profile_path.empty()) {
    j["profile"] = nullptr;
  } else if (sc.is_bundle) {
    j["profile"] = {{"path", sc.profile_path}, {"type", "bundle"}, {"bundle_target", sc.bundle_target_name},
                    {"input_space", to_string(sc.input_space)}};
    // --print: the brackets were rendered through a print material's print ICC (the print declarations above), not the
    // chart's.
    if (!sc.print_id.empty()) {
      j["profile"]["print"] = {{"id", sc.print_id}, {"name", sc.print_name},
                               {"table", sc.print_from_bundle ? "the bundle's print ICC" : "computed by neg_process"},
                               {"curves", "computed at the print's mid grey A (a) from the bundle's "
                                          "grey_balance_curves, with the gains g and b after the knee"}};
      // A material given as a file: which file, and its content, so the replay can be checked against it.
      if (!sc.print_file.empty()) {
        j["profile"]["print"]["file"] = sc.print_file;
        j["profile"]["print"]["file_sha256"] = sc.print_file_sha256;
      }
      // The print's white, its mid grey A and the bracket's own, and its search (the pipeline code below:
      // search_print_continuous).
      j["profile"]["print"]["white"] = "paper";   // media-relative: the paper's base at the display's white
      j["profile"]["print"]["paper_white_xyz"] = {num(sc.print_paper_xyz[0], "%.6g"),
                                                  num(sc.print_paper_xyz[1], "%.6g"),
                                                  num(sc.print_paper_xyz[2], "%.6g")};
      j["profile"]["print"]["a"] = num(sc.print_a, "%.17g");
      if (std::isfinite(sc.print_bracket_a_mid))
        j["profile"]["print"]["bracket_a_mid"] = num(sc.print_bracket_a_mid, "%.17g");
      // The brackets' tilt (grey_relative.b) at that A, linear in A between brackets.
      if (std::isfinite(sc.print_tilt[0]))
        j["profile"]["print"]["tilt"] = {num(sc.print_tilt[0], "%.9g"), num(sc.print_tilt[1], "%.9g"),
                                         num(sc.print_tilt[2], "%.9g")};
      if (sc.print_searched) {
        auto three = [&](const float* v, double scale) {
          return ordered_json{num(scale * v[0], "%.4g"), num(scale * v[1], "%.4g"), num(scale * v[2], "%.4g")};
        };
        j["profile"]["print"]["search"] = {
            {"a0", num(sc.print_a0, "%.9g")},
            {"target_L", num(sc.print_target_l, "%.6g")},
            {"steps", sc.print_steps},
            {"converged", sc.print_converged},
            {"stopped_by", sc.print_stopped_by},
            {"above_knee_pct", num(100.0 * sc.print_above_knee, "%.4g")},
            {"base_factor", num(sc.print_floor_applied ? sc.print_floor_k : 1.0f, "%.6g")},
            {"past_knee_pct", three(sc.print_past_knee, 100.0)},
            {"past_cut_pct", three(sc.print_past_cut, 100.0)},
            {"above_white_pct", three(sc.print_above_white, 100.0)},
            {"at_white_pct", three(sc.print_at_white, 100.0)},
            {"a0_at_range_end", sc.print_a0_at_end},
            {"last_move", three(sc.print_last_move, 1.0)}};
      } else if (!sc.cached_print_search_json_str.empty()) {
        // A run that reused <output>.json (the knobs) starts from that run's A (knobs.base.a): its search's record
        // carries over.
        j["profile"]["print"]["search"] = ordered_json::parse(sc.cached_print_search_json_str);
      }
    }
  } else {
    j["profile"] = {{"path", sc.profile_path}, {"type", "icc"}, {"bundle_target", nullptr}};
  }

  ordered_json merged = ordered_json::array();
  for (int i = 0; i < 9; ++i) merged.push_back(num(sc.merged_matrix[i], "%.9g"));
  j["matrix"] = {{"merged", merged},
                 {"exposure_comp", num(sc.exposure_comp, "%.9g")},
                 {"gain_g", num(sc.gain_g, "%.9g")},
                 {"gain_b", num(sc.gain_b, "%.9g")},
                 {"pins", {{"e", (sc.pin_mask & PIN_E) != 0}, {"g", (sc.pin_mask & PIN_G) != 0},
                           {"b", (sc.pin_mask & PIN_B) != 0}}}};

  // The values -E, --gain_g and --gain_b take: the station's with a bundle, else the absolute ones.
  const float* egb = sc.station_params;
  const float abs_egb[3] = {sc.exposure_comp, sc.gain_g, sc.gain_b};
  if (!egb) egb = abs_egb;
  std::vector<std::string> cli = {"-E", num_str(egb[0], "%.9g"), "--gain_g", num_str(egb[1], "%.9g"), "--gain_b",
                                  num_str(egb[2], "%.9g")};
  if (sc.has_roi) {
    cli.push_back("--roi");
    for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
      cli.push_back(std::to_string(v));
  }
  if (sc.is_bundle && sc.bundle_target_idx >= 0)
    cli.insert(cli.end(), {"--target", std::to_string(sc.bundle_target_idx)});
  if (sc.input_curves && (sc.cli_input_curves ? *sc.cli_input_curves : *sc.input_curves).active()) {
    // Replays the combined profile without fitting (the curves need -P, as when they were fitted), at the E, g and b
    // above: after the knobs, the curves re-anchored there.
    // The value leads with "ref:" and the refs (always positive): the older order led with aG, and a negative aG
    // made the value start with '-', which the parser rejects as an unknown option, so the line did not replay.
    // No curves (the fit fell back): nothing to replay, and the given values alone render the same pixels.
    const negicc::InputCurves& c = sc.cli_input_curves ? *sc.cli_input_curves : *sc.input_curves;
    cli.push_back("--fit_curves");
    std::string v8 = "ref:";
    for (float v : {c.ref[0], c.ref[1], c.u_med[0], c.u_med[1], c.a[0], c.k[0], c.a[1], c.k[1]})
      v8 += (v8.size() > 4 ? "," : "") + num_str(v, "%.9g");
    if (c.hold) v8 += "," + num_str(c.u_lo[0], "%.9g") + "," + num_str(c.u_lo[1], "%.9g");  // held below u_lo
    cli.push_back(v8);
  }
  if (sc.tone_curve && sc.tone_applied) {
    const ToneCurve& q = *sc.tone_curve;
    std::string v = "curve:";
    for (double x : {q.black, q.white, q.gamma, q.contrast})
      v += (v.size() > 6 ? "," : "") + num_str(x, "%.17g");
    cli.insert(cli.end(), {"--tone_stage", v});
  }
  // A bundle conversion with neither recorded curves nor a recorded tone curve: the given values alone render these
  // pixels, and --tone_stage off (with --fit_curves off on white light, the fit's light) keeps the stages that run
  // after given values too from running on the replay.
  if (sc.is_bundle && cli.size() >= 2 && std::find(cli.begin(), cli.end(), "--fit_curves") == cli.end() &&
      std::find(cli.begin(), cli.end(), "--tone_stage") == cli.end()) {
    if (sc.input_space == InputSpace::kRec2020Linear) cli.insert(cli.end(), {"--fit_curves", "off"});
    cli.insert(cli.end(), {"--tone_stage", "off"});
  }
  // A print is made by its own search (no given values with --print): its replay is the frame, the print and the knobs
  // if any, and re-running it re-runs the search (deterministic, with --dino, --dino_iters and --dino_backend when
  // given: the last keeps the passes where they ran), then the same knobs on its result.
  if (!sc.print_id.empty() && sc.density_json.empty()) {
    cli.clear();
    if (sc.has_roi) {
      cli.push_back("--roi");
      for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
        cli.push_back(std::to_string(v));
    }
    if (!sc.dino_arg.empty()) cli.insert(cli.end(), {"--dino", sc.dino_arg});
    if (sc.dino_iters_given) cli.insert(cli.end(), {"--dino_iters", std::to_string(sc.dino_iters)});
    if (!sc.dino_backend_arg.empty()) cli.insert(cli.end(), {"--dino_backend", sc.dino_backend_arg});
    if (sc.has_knobs) {
      if (sc.brightness != 0.0f) cli.insert(cli.end(), {"--brightness", num_str(sc.brightness, "%.9g")});
      if (sc.tint != 0.0f) cli.insert(cli.end(), {"--tint", num_str(sc.tint, "%.9g")});
      if (sc.warmth != 0.0f) cli.insert(cli.end(), {"--warmth", num_str(sc.warmth, "%.9g")});
    }
  }
  // A print: the replay renders it too, from the same file when the material was given as one (its id may be a
  // built-in material's, or none).
  if (!sc.print_id.empty()) cli.insert(cli.end(), {"--print", sc.print_file.empty() ? sc.print_id : sc.print_file});
  // A density output takes no exposure, gains or bracket: its replay is the frame, --fit_curves off when a white-light
  // run without --print turned the fit off (for its suggested balance), and the --print materials and any knobs on
  // their CLFs as given: with --print the base not a capture is raised to the picture's thinnest film
  // (raise_base_to_print_floor) and the densities are written against it, so the replay needs --print to write the
  // same file and its CLFs.
  if (!sc.density_json.empty()) {
    cli.clear();
    if (sc.has_roi) {
      cli.push_back("--roi");
      for (int v : {sc.roi[0], sc.roi[1], sc.roi[2], sc.roi[3], sc.rot_cw, sc.hflip ? 1 : 0, sc.vflip ? 1 : 0})
        cli.push_back(std::to_string(v));
    }
    if (sc.density_fit_off) cli.insert(cli.end(), {"--fit_curves", "off"});
    // The search's model, iterations and backend, as on a print: the recorded balance and each print's search depend
    // on them.
    if (!sc.dino_arg.empty()) cli.insert(cli.end(), {"--dino", sc.dino_arg});
    if (sc.dino_iters_given) cli.insert(cli.end(), {"--dino_iters", std::to_string(sc.dino_iters)});
    if (!sc.dino_backend_arg.empty()) cli.insert(cli.end(), {"--dino_backend", sc.dino_backend_arg});
    for (const std::string& m : sc.density_prints) cli.insert(cli.end(), {"--print", m});   // once per material
    if (!sc.density_prints.empty()) {
      if (sc.brightness != 0.0f) cli.insert(cli.end(), {"--brightness", num_str(sc.brightness, "%.9g")});
      if (sc.tint != 0.0f) cli.insert(cli.end(), {"--tint", num_str(sc.tint, "%.9g")});
      if (sc.warmth != 0.0f) cli.insert(cli.end(), {"--warmth", num_str(sc.warmth, "%.9g")});
    }
  }
  // The format the film base's frame detection looked for, given or --roi auto's default (35mm): an explicit --roi
  // without it looks for every format, which may place the base's patch elsewhere. None when the base's detection
  // looked for every format (the CFA path's, which runs again for the base), as its replay does.
  if (!sc.film_format.empty()) cli.insert(cli.end(), {"--film_format", sc.film_format});
  j["cli_equivalent"] = cli;

  // auto: the conversion parameter search (multi_target_optimizer.h)
  if (!sc.cached_auto_json_str.empty()) {
    j["auto"] = ordered_json::parse(sc.cached_auto_json_str);
  } else {
    j["auto"] = sc.auto_res ? auto_json(*sc.auto_res, sc.auto_frame_half, sc) : ordered_json(nullptr);
  }
  if (!sc.cached_dino_json_str.empty()) {
    j["dino"] = ordered_json::parse(sc.cached_dino_json_str);
  } else {
    j["dino"] = sc.dino ? dino_json(*sc.dino, sc) : ordered_json(nullptr);
  }
  // The intent field Auto converted against; a run that reuses <output>.json (the knobs) keeps the previous run's.
  // The models: what --dino replaced, or the built-in ones.
  if (!sc.cached_heads_json_str.empty()) {
    j["heads"] = ordered_json::parse(sc.cached_heads_json_str);
  } else {
    ordered_json h;
    const char* names[3] = {"intent", "neutral", "tone"};
    for (int i = 0; i < 3; ++i) {
      if (sc.head_path[i].empty()) {
        h[names[i]] = nullptr;
        continue;
      }
      char crc[16];
      std::snprintf(crc, sizeof(crc), "%08x", sc.head_fp[i].crc);
      h[names[i]] = {{"path", sc.head_path[i]},
                     {"bytes", sc.head_fp[i].ok ? ordered_json(sc.head_fp[i].size) : ordered_json(nullptr)},
                     {"crc32", sc.head_fp[i].ok ? ordered_json(crc) : ordered_json(nullptr)}};
    }
    j["heads"] = h;
  }
  if (!sc.cached_intent_field_json_str.empty())
    j["intent_field"] = ordered_json::parse(sc.cached_intent_field_json_str);
  else if (sc.auto_res)
    j["intent_field"] = intent_field_json(sc.auto_res->intent_mu, sc.auto_res->intent_sigma,
                                          sc.auto_res->intent_grid_w, sc.auto_res->intent_grid_h,
                                          sc.auto_res->target_idx, "auto", ParamSearchOptions{}.crop_ratio, sc);
  else if (sc.dino)
    j["intent_field"] = intent_field_json(sc.dino->best_mu, sc.dino->best_sigma, sc.dino->grid_w, sc.dino->grid_h,
                                          sc.bundle_target_idx, "dino", 1.0f, sc);
  else
    j["intent_field"] = nullptr;

  if (!sc.cached_input_curves_json_str.empty())
    j["input_curves"] = ordered_json::parse(sc.cached_input_curves_json_str);
  else
    j["input_curves"] = sc.input_curves ? input_curves_json(sc) : ordered_json(nullptr);
  if (sc.has_knobs) {
    j["knobs"] = {
        {"brightness", num(sc.brightness, "%.6g")},
        {"tint", num(sc.tint, "%.6g")},
        {"warmth", num(sc.warmth, "%.6g")},
        {"limited", sc.knob_res.limited},
        {"base", std::isfinite(sc.knob_base_a)
                     ? ordered_json{{"e", num(sc.knob_base[0], "%.9g")}, {"g", num(sc.knob_base[1], "%.9g")},
                                    {"b", num(sc.knob_base[2], "%.9g")}, {"a", num(sc.knob_base_a, "%.17g")}}
                     : ordered_json{{"e", num(sc.knob_base[0], "%.9g")}, {"g", num(sc.knob_base[1], "%.9g")},
                                    {"b", num(sc.knob_base[2], "%.9g")}}},
        {"egb", {{"e", num(sc.knob_res.e, "%.9g")},
                 {"g", num(sc.knob_res.g, "%.9g")},
                 {"b", num(sc.knob_res.b, "%.9g")}}},
        {"grey_point", {num(sc.knob_res.grey_lab[0], "%.4f"),
                        num(sc.knob_res.grey_lab[1], "%.4f"),
                        num(sc.knob_res.grey_lab[2], "%.4f")}},
    };
  } else {
    j["knobs"] = nullptr;
  }

  const ShoulderInfo& sh = sc.shoulder;
  j["tone"] = {{"knee", num(sc.knee, "%.9g")},
               {"shoulder", num(sh.x_hi, "%.9g")},
               {"shoulder_source", sh.source},
               {"shoulder_q995", sh.has_q995 ? num(sh.q995) : ordered_json(nullptr)},
               {"shoulder_knee", sh.has_knee ? num(sh.knee) : ordered_json(nullptr)}};

  // The tone stage: the heads' predictions, the final render's descriptors before it, the targets, the
  // curve, and the clipping before (the clip check's sample) and after (the whole export).
  auto clip_json = [&]() {
    if (!sc.tone_applied || !sc.tone_clip) return ordered_json(nullptr);
    const ToneClip& k = *sc.tone_clip;
    return ordered_json{{"white_before", num(k.white_before, "%.6f")}, {"white_after", num(k.white_after, "%.6f")},
                        {"black_before", num(k.black_before, "%.6f")}, {"black_after", num(k.black_after, "%.6f")}};
  };
  if (sc.tone_stage_plan) {
    auto n9 = [](double v) { return num(v, "%.9g"); };
    const ToneStagePlan& tp = *sc.tone_stage_plan;
    if (tp.planned && sc.tone_plan) {
      const TonePlanCurve& pl = *sc.tone_plan;
      ordered_json pred, pred_sd;
      for (int t = 0; t < ToneHead::kTargets; ++t) {
        pred[ToneHead::target_name(t)] = n9((float)tp.pred_mean[t]);
        pred_sd[ToneHead::target_name(t)] = n9((float)tp.pred_sd[t]);
      }
      auto desc = [&](const ToneDescriptors& d) {
        return ordered_json{{"L2", n9((float)d.L2)},   {"L10", n9((float)d.L10)}, {"L25", n9((float)d.L25)},
                            {"L50", n9((float)d.L50)}, {"L75", n9((float)d.L75)}, {"L90", n9((float)d.L90)},
                            {"L98", n9((float)d.L98)}, {"R9010", n9((float)d.range())}};
      };
      // As applied (capped on the final image), else as planned.
      const ToneTargets& tt = sc.tone_targets ? *sc.tone_targets : pl.targets;
      const ToneCurve& tc = sc.tone_applied && sc.tone_curve ? *sc.tone_curve : pl.curve;
      j["tone_stage"] = {
          {"applied", sc.tone_applied},
          {"source", "plan"},
          {"features", tp.features},
          {"predicted", pred},
          {"predicted_sd", pred_sd},
          {"render", desc(pl.render)},
          {"targets", {{"L2", n9((float)tt.L2)}, {"L50", n9((float)tt.L50)}, {"L98", n9((float)tt.L98)},
                       {"R9010", n9((float)tt.range)}}},
          // The share of the highlight and range moves kept, and of the shadow move: 1, or less where they would clip
          // the final image.
          {"moves", {{"shadow", n9((float)kToneShadowMove)}, {"highlight", n9((float)kToneHighlightMove)},
                     {"range", n9((float)kToneRangeMove)}, {"highlight_share", n9(sc.tone_share)},
                     {"shadow_share", n9(sc.tone_shadow_share)}}},
          // Where no share held both ends, the rule that chose the curve (--tone_stage); null where one held.
          {"fallback", sc.tone_fallback ? ordered_json(sc.tone_fallback) : ordered_json(nullptr)},
          // In full (17 digits): the curve the image went through, exactly.
          {"curve", tone_curve_json(tc)},
          {"clip", clip_json()}};
    } else {
      j["tone_stage"] = {{"applied", false}, {"skipped", tp.skipped ? tp.skipped : ""}};
    }
  } else if (const ToneCurve* q = sc.tone_curve) {
    // A replayed curve (a previous run's, or given): the curve and the clipping on this image.
    j["tone_stage"] = {
        {"applied", sc.tone_applied},
        {"source", "replay"},
        {"curve", tone_curve_json(*q)},
        {"clip", clip_json()}};
  } else {
    j["tone_stage"] = nullptr;
  }

  // What a replay rendered from recorded values (null when nothing was replayed).
  if (!sc.replay_source.empty()) {
    j["replay"] = {{"source", sc.replay_source},
                   {"tone_curve", sc.tone_curve ? ordered_json(sc.tone_applied ? "applied" : "not applied")
                                                : ordered_json(nullptr)}};
  } else {
    j["replay"] = nullptr;
  }

  // A density output (the print declarations above): its record, and the file's bit depth (a DPX holds 10 bits).
  ordered_json density = sc.density_json.empty() ? ordered_json(nullptr)
                                                 : ordered_json::parse(sc.density_json, nullptr, false);
  const int out_bits = density.is_object() && density.contains("encoding") ? density["encoding"].value("bits", 16) : 16;
  j["output"] = {{"path", sc.output_path}, {"colorspace", sc.colorspace}, {"bits", out_bits}};
  if (density.is_object()) j["density"] = density;

  ordered_json t;
  t["raw_decode"] = num(sc.t_raw_decode_ms, "%.1f");
  t["frame_detection"] = sc.frame_det ? num(sc.t_frame_det_ms, "%.1f") : ordered_json(nullptr);
  // The bracket analysis outside Auto's time (the knobs, the roll-off); before, it counted as raw_decode.
  t["analysis"] = num(sc.t_analysis_ms, "%.1f");
  const bool has_cached_dino = !sc.cached_dino_json_str.empty();
  t["dino_views"] = sc.dino ? num(sc.dino->timing_views_ms, "%.1f")
                            : (has_cached_dino ? num(sc.cached_dino_views_ms, "%.1f") : ordered_json(nullptr));
  t["dino_render"] = sc.dino ? num(sc.dino->timing_render_ms, "%.1f")
                             : (has_cached_dino ? num(sc.cached_dino_render_ms, "%.1f") : ordered_json(nullptr));
  t["dino_inference"] = sc.dino ? num(sc.dino->timing_infer_ms, "%.1f")
                                : (has_cached_dino ? num(sc.cached_dino_infer_ms, "%.1f") : ordered_json(nullptr));
  t["dino_search"] = sc.dino ? num(sc.dino->timing_search_ms, "%.1f")
                             : (has_cached_dino ? num(sc.cached_dino_search_ms, "%.1f") : ordered_json(nullptr));
  t["full_conversion"] = num(sc.t_full_conversion_ms, "%.1f");
  t["tiff_write"] = num(sc.t_tiff_write_ms, "%.1f");
  j["timings_ms"] = t;

  // Written to a temporary file and renamed, so a reader never sees a partial sidecar.
  const std::string json_path = sc.output_path + ".json";
  const std::string tmp_path = json_path + ".tmp";
  FILE* fp = fopen(tmp_path.c_str(), "w");
  if (!fp) {
    log_warn("Cannot open sidecar file %s for writing\n", tmp_path.c_str());
    return false;
  }
  // Paths and arguments need not be UTF-8; invalid bytes are written as U+FFFD rather than failing.
  const std::string text = j.dump(2, ' ', false, ordered_json::error_handler_t::replace) + "\n";
  const bool write_ok = fwrite(text.data(), 1, text.size(), fp) == text.size();
  const bool close_ok = fclose(fp) == 0;
  if (!write_ok || !close_ok) {
    log_warn("Failed to flush/close sidecar %s\n", tmp_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }
  if (rename(tmp_path.c_str(), json_path.c_str()) != 0) {
    log_warn("Failed to rename sidecar from %s to %s\n", tmp_path.c_str(), json_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }
  return true;
}

// ====================================================================================================================
// Pipeline: code
// The backends, and the stages main runs in turn: prepare, decode, finish_search_passes, begin_conversion, render,
// solve, convert and write.
// ====================================================================================================================

void LibRawDeleter::operator()(LibRaw* p) const {
  if (!p) return;
  p->free_image();
  delete p;
}

void Scan::scan_exposure(const LibRaw& r, double& shutter_out, double& iso_out) {
  const bool have_shutter = (has_exif_exp && exif_shutter_d > 0.0) || r.imgdata.other.shutter > 0;
  const bool have_iso = (has_exif_exp && exif_iso_d > 0.0) || r.imgdata.other.iso_speed > 0;
  shutter_out = (has_exif_exp && exif_shutter_d > 0.0)
                    ? exif_shutter_d
                    : (r.imgdata.other.shutter > 0 ? (double)r.imgdata.other.shutter : kDefaultScanShutterS);
  iso_out = (has_exif_exp && exif_iso_d > 0.0)
                ? std::round(exif_iso_d)
                : (r.imgdata.other.iso_speed > 0 ? std::round((double)r.imgdata.other.iso_speed) : kDefaultScanIso);
  if ((!have_shutter || !have_iso) && !warned_default_exposure) {
    warned_default_exposure = true;
    log_warn("Missing EXIF exposure metadata in %s; falling back to %.4g s, ISO %.0f\n",
            exposure_file.c_str(), shutter_out, iso_out);
  }
}

RgbInput Scan::image() {
  RgbInput img;
  img.width = proc->imgdata.sizes.iwidth;
  img.height = proc->imgdata.sizes.iheight;
  if (!rgb3.empty()) {
    img.pixels = rgb3.data();
    img.channels = 3;
  } else {
    img.pixels = &proc->imgdata.image[0][0];
    img.channels = 4;
  }
  return img;
}

// The debayer's and the merge's arguments for a region of a mosaic.
static negicc::GrainAwareDebayerArgs debayer_args(const MosaicInput& in, const Region& region, int dst_w, int dst_h,
                                                  uint16_t* rgb_out) {
  negicc::GrainAwareDebayerArgs ga{};
  ga.cfa_bayer = in.frames[0];
  ga.src_w = in.width;
  ga.src_h = in.height;
  ga.rx = region.x;
  ga.ry = region.y;
  ga.rw = region.w;
  ga.rh = region.h;
  ga.rot_cw = region.rot_cw;
  ga.hflip = region.hflip;
  ga.vflip = region.vflip;
  ga.dst_w = dst_w;
  ga.dst_h = dst_h;
  ga.filters = in.filters;
  ga.white_level = in.white_level;
  ga.eps = 4096.0f;
  ga.rgb_out = rgb_out;
  ga.flat = in.flat;
  ga.flat_x0 = region.x;
  ga.flat_y0 = region.y;
  ga.cfa_pitch = in.pitch;
  return ga;
}

static negicc::PixelShiftMergeArgs merge_args(const MosaicInput& in, const Region& region, int dst_w, int dst_h,
                                              uint16_t* rgb_out) {
  negicc::PixelShiftMergeArgs pm{};
  pm.bayer_f0 = in.frames[0];
  pm.bayer_f1 = in.frames[1];
  pm.bayer_f2 = in.frames[2];
  pm.bayer_f3 = in.frames[3];
  pm.src_w = in.width;
  pm.src_h = in.height;
  pm.rx = region.x;
  pm.ry = region.y;
  pm.rw = region.w;
  pm.rh = region.h;
  pm.rot_cw = region.rot_cw;
  pm.hflip = region.hflip;
  pm.vflip = region.vflip;
  pm.dst_w = dst_w;
  pm.dst_h = dst_h;
  pm.filters = in.filters;
  pm.white_level = in.white_level;
  pm.rgb_out = rgb_out;
  pm.flat = in.flat;
  pm.flat_x0 = region.x;
  pm.flat_y0 = region.y;
  pm.cfa_pitch = in.pitch;
  return pm;
}

void CpuBackend::render(const MosaicInput& in, const Region& region, int dst_w, int dst_h,
                        std::vector<uint16_t>& rgb_out, const std::function<void(int)>&) {
  rgb_out.resize((size_t)std::max(dst_w, 0) * std::max(dst_h, 0) * 3);
  if (in.n_frames == 1) negicc::cpu_grain_aware_debayer(debayer_args(in, region, dst_w, dst_h, rgb_out.data()));
  else negicc::cpu_pixel_shift_merge(merge_args(in, region, dst_w, dst_h, rgb_out.data()));
  render_ran = "cpu";
}

namespace {
// Device memory, freed at the end of its scope.
struct DeviceMem {
  void* p = nullptr;
  DeviceMem() = default;
  DeviceMem(const DeviceMem&) = delete;
  DeviceMem& operator=(const DeviceMem&) = delete;
  ~DeviceMem() { reset(); }
  void reset() {
    negicc::gpu_free(p);
    p = nullptr;
  }
  void* alloc(size_t bytes) {
    reset();
    p = negicc::gpu_malloc(bytes);
    return p;
  }
  void* upload(const void* host, size_t bytes) {
    negicc::gpu_upload(alloc(bytes), host, bytes);
    return p;
  }
};
}  // namespace

void GpuBackend::render(const MosaicInput& in, const Region& region, int dst_w, int dst_h,
                        std::vector<uint16_t>& rgb_out, const std::function<void(int)>& frame_done) {
  // The CPU's when there is nothing to render or no device runs this build's kernels. The device is asked for here,
  // where a step first needs it, so a run without such a step never starts it.
  if (dst_w <= 0 || dst_h <= 0 || in.width <= 0 || in.height <= 0 || !negicc::gpu_device_available()) {
    CpuBackend::render(in, region, dst_w, dst_h, rgb_out, frame_done);
    return;
  }
  // A mosaic's rows are pitch elements apart, and its last row holds width of them.
  const size_t mosaic_bytes = ((size_t)in.pitch * (size_t)(in.height - 1) + (size_t)in.width) * sizeof(uint16_t);
  DeviceMem frames[4], flat_block, out;
  MosaicInput dev = in;
  // Each of the host's mosaics goes as soon as its copy is on the device: the two are held for one frame at a time.
  for (int k = 0; k < in.n_frames; ++k) {
    dev.frames[k] = static_cast<const uint16_t*>(frames[k].upload(in.frames[k], mosaic_bytes));
    frame_done(k);
  }
  if (in.flat.gain) {
    // The flat field's tables are one block; the copy's tables sit at the same offsets in it.
    if (!in.flat_tables) throw negicc::GpuError("a flat field without its tables cannot be copied to the device");
    const char* host = static_cast<const char*>(in.flat_tables->host_block());
    const char* copy = static_cast<const char*>(flat_block.upload(host, in.flat_tables->host_bytes()));
    auto moved = [&](const void* ptr) { return copy + (static_cast<const char*>(ptr) - host); };
    dev.flat.gain = reinterpret_cast<const float*>(moved(in.flat.gain));
    dev.flat.ax_i = reinterpret_cast<const int*>(moved(in.flat.ax_i));
    dev.flat.ax_t = reinterpret_cast<const float*>(moved(in.flat.ax_t));
    dev.flat.ay_i = reinterpret_cast<const int*>(moved(in.flat.ay_i));
    dev.flat.ay_t = reinterpret_cast<const float*>(moved(in.flat.ay_t));
  }
  const size_t out_bytes = (size_t)dst_w * dst_h * 3 * sizeof(uint16_t);
  uint16_t* dev_out = static_cast<uint16_t*>(out.alloc(out_bytes));
  if (dev.n_frames == 1) negicc::gpu_grain_aware_debayer(debayer_args(dev, region, dst_w, dst_h, dev_out));
  else negicc::gpu_pixel_shift_merge(merge_args(dev, region, dst_w, dst_h, dev_out));
  // The kernel has finished (the two calls wait for it). The device's mosaics and flat field go before the host's
  // picture is made, which is first written by the download: the picture is never held beside the mosaics.
  for (DeviceMem& f : frames) f.reset();
  flat_block.reset();
  rgb_out.resize((size_t)dst_w * dst_h * 3);
  negicc::gpu_download(rgb_out.data(), dev_out, out_bytes);
  render_ran = "gpu";
}

void GpuBackend::convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                         const RgbInput& img) {
  // The CPU's for an encoding or a layout the device's render does not make, and when no device runs this build's
  // kernels (asked for last, so that a conversion the device would not run anyway does not start it).
  if ((out.encoding != ENCODE_SRGB && out.encoding != ENCODE_SRGB_LINEAR) || img.channels != 3 ||
      !negicc::gpu_device_available()) {
    CpuBackend::convert(out, pc, merged_matrix, img);
    return;
  }
  // The picture is copied to the device and read there, so the result may be copied back over it.
  const negicc::RgbView frame = negicc::RgbView::from_interleaved_rgb(img.pixels, img.width, img.height);
  GpuRenderer renderer;
  renderer.render_over_frame = true;   // the device holds the picture once: the render replaces its copy
  renderer.bind_frame(frame);
  negicc::RenderArgs a{};
  a.in_view = &frame;
  a.width = img.width;
  a.height = img.height;
  a.matrix = merged_matrix;
  a.cfg = pc;
  a.out_type = 0;   // 16-bit codes
  a.out = img.pixels;
  renderer.render(a);
  convert_ran = "gpu";
}

// Whether the steps that can run on a CUDA device may: the search's passes and the full-size steps, by one rule. Not
// with --dino_backend cpu, which keeps everything on the CPU, and not under --check_renderer, which runs the search's
// passes on the CPU to compare them with the device's: its picture is the CPU's throughout.
static bool device_allowed(const Options& o) { return !o.check_renderer && o.dino_backend != "cpu"; }

std::unique_ptr<Backend> make_backend(const Options& o) {
  if (device_allowed(o)) return std::make_unique<GpuBackend>();
  return std::make_unique<CpuBackend>();
}

void CpuBackend::convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                         const RgbInput& img) {
  convert_ran = "cpu";
  const int width = img.width, height = img.height, ch = img.channels;
  uint16_t* pixels = img.pixels;
  // The sRGB encodings and ENCODE_NONE (the corrected RGB, already in [0, 1]) write 16-bit codes in place.
  auto convert_in_place = [&](auto stage) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        uint16_t* px = &pixels[((size_t)j * width + i) * ch];
        const Rgb v = stage(px);
        px[0] = negicc::quantize_u16(v.r);
        px[1] = negicc::quantize_u16(v.g);
        px[2] = negicc::quantize_u16(v.b);
      }
    }
  };
  if (out.encoding == ENCODE_SRGB || out.encoding == ENCODE_SRGB_LINEAR) {
    convert_in_place([&](const uint16_t* px) { return negicc::convert_pixel(pc, merged_matrix, px); });
  } else if (out.encoding == ENCODE_CUSTOM) {
    // Through Little-CMS from XYZ, a row at a time, into the image's own layout (3 samples, or LibRaw's 4).
    #pragma omp parallel
    {
      cmsHTRANSFORM thread_xform = cmsCreateTransform(out.xyz, TYPE_XYZ_FLT, out.custom,
                                                      ch == 3 ? TYPE_RGB_16 : TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
      std::vector<float> xyz_row(width * 3);

      #pragma omp for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          const uint16_t* px = &pixels[((size_t)j * width + i) * ch];
          const Rgb v = negicc::convert_pixel_pcs(pc, merged_matrix, px);
          xyz_row[i * 3 + 0] = v.r;
          xyz_row[i * 3 + 1] = v.g;
          xyz_row[i * 3 + 2] = v.b;
        }
        cmsDoTransform(thread_xform, xyz_row.data(), &pixels[(size_t)j * width * ch], width);
      }
      cmsDeleteTransform(thread_xform);
    }
  } else {
    convert_in_place([&](const uint16_t* px) { return negicc::convert_pixel_corrected(pc, merged_matrix, px); });
  }
}

// A custom -P profile is written through: the conversion writes three samples a pixel through a transform from PCS
// XYZ into it (CpuBackend::convert), which LittleCMS cannot build for a Gray or CMYK profile, or for one without a
// transform from the PCS. False, with the error logged, for one it cannot.
static bool output_profile_ok(const std::string& arg) {
  cmsHPROFILE prof = cmsOpenProfileFromFile(resolve_profile_path(arg).c_str(), "r");
  if (!prof) {
    log_error("Cannot read output ICC profile: %s\n", arg.c_str());
    return false;
  }
  cmsHPROFILE xyz = cmsCreateXYZProfile();
  cmsHTRANSFORM probe = nullptr;
  if (cmsGetColorSpace(prof) == cmsSigRgbData) {
    probe = cmsCreateTransform(xyz, TYPE_XYZ_FLT, prof, TYPE_RGB_16, INTENT_PERCEPTUAL, 0);
  }
  if (probe) cmsDeleteTransform(probe);
  cmsCloseProfile(xyz);
  cmsCloseProfile(prof);
  if (!probe) {
    log_error("Output ICC profile %s is not an RGB profile that can be converted to from PCS XYZ\n", arg.c_str());
  }
  return probe != nullptr;
}

// Whether a path a previous run's sidecar records names this run's path: the same name as the sidecar records it
// (its bytes that are not UTF-8 as U+FFFD, utf8_replaced), or the same existing file.
static bool paths_match(const std::string& recorded, const std::string& path) {
  if (recorded == path || recorded == utf8_replaced(path)) return true;
  std::error_code ec;
  if (std::filesystem::exists(recorded, ec) && std::filesystem::exists(path, ec) &&
      std::filesystem::equivalent(recorded, path, ec)) {
    return true;
  }
  return false;
}

static bool extract_cli_egb(const nlohmann::ordered_json& cli, float& e_out, float& g_out, float& b_out) {
  if (!cli.is_array()) return false;
  bool got_e = false, got_g = false, got_b = false;
  for (size_t i = 0; i + 1 < cli.size(); ++i) {
    if (!cli[i].is_string() || !cli[i + 1].is_string()) continue;
    const std::string flag = cli[i].get<std::string>();
    const std::string val_str = cli[i + 1].get<std::string>();
    char* end = nullptr;
    const float v = std::strtof(val_str.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(v)) continue;
    if (flag == "-E" || flag == "--exposure_comp") {
      e_out = v;
      got_e = true;
      ++i;
    } else if (flag == "--gain_g") {
      g_out = v;
      got_g = true;
      ++i;
    } else if (flag == "--gain_b") {
      b_out = v;
      got_b = true;
      ++i;
    }
  }
  return got_e && got_g && got_b;
}

// The arguments of a run that decide its result, normalised so that spelling and order do not matter: argv without
// the program name, the output (-o, --output, -oFILE) and the knobs (--brightness, --tint, --warmth); each option under
// its long name (-E and --exposure_comp alike) with its values ("--flag value", "--flag=value" or a short flag's
// attached "-Evalue" alike), the options sorted by name (an option given twice keeps its order), then "--" and the
// positional arguments (the inputs) in their order. Options neg_process does not know keep their own group.
static std::vector<std::string> result_args(const std::vector<std::string>& argv) {
  struct Opt {
    const char* shrt;
    const char* lng;
    int arity;  // values; -1: --roi (up to 7 numbers, or auto [h|v]); -16: up to 16 numbers (--probe_stops)
    bool skip;  // the output, the knobs and the development aids (which leave the result as it is)
  };
  static const Opt kOpts[] = {
      {"-C", "--no_crop", 0, false},      {"-q", "--quality", 1, false},        {"-r", "--r_coeff", 3, false},
      {"-g", "--g_coeff", 3, false},      {"-b", "--b_coeff", 3, false},        {"-E", "--exposure_comp", 1, false},
      {"", "--no-metadata", 0, false},    {"", "--gain_g", 1, false},           {"", "--gain_b", 1, false},
      {"", "--brightness", 1, true},      {"", "--tint", 1, true},              {"", "--warmth", 1, true},
      {"", "--film_stock", 1, false},     {"", "--light_source", 1, false},     {"-p", "--film_profile", 1, false},
      {"-P", "--colorspace", 1, false},   {"", "--roi", -1, false},             {"", "--film_format", 1, false},
      {"", "--rot", 1, false},            {"", "--rot_cw", 1, false},           {"", "--hflip", 0, false},
      {"", "--vflip", 0, false},          {"-o", "--output", 1, true},          {"", "--dino", 1, false},
      {"", "--list_profiles", 0, false},  {"", "--dino_backend", 1, false},     {"", "--dino_iters", 1, false},
      {"", "--target", 1, false},         {"", "--bundle_target", 1, false},    {"", "--fit_curves", 1, false},
      {"", "--flat", 1, false},           {"", "--tone_stage", 1, false},       {"", "--film_base", 1, false},
      {"", "--print", 1, false},           {"", "--check_renderer", 0, true},
      {"", "--probe_intent", 1, true},    {"", "--probe_params", 1, true},      {"", "--probe_stops", -16, true},
  };
  auto is_number = [](const std::string& t) {
    char* e = nullptr;
    std::strtod(t.c_str(), &e);
    return !t.empty() && e && *e == '\0';
  };
  auto canonical = [](const char* lng) {
    const std::string l = lng;
    return l == "--rot" ? std::string("--rot_cw") : l == "--bundle_target" ? std::string("--target") : l;
  };
  std::vector<std::vector<std::string>> groups;  // [name, values...]
  std::vector<std::string> positional;
  for (size_t i = 1; i < argv.size(); ++i) {
    const std::string& a = argv[i];
    if (a.size() < 2 || a[0] != '-' || is_number(a)) {
      positional.push_back(a);
      continue;
    }
    std::string flag = a, attached;
    bool has_attached = false;
    if (a.compare(0, 2, "--") == 0) {
      const size_t eq = a.find('=');
      if (eq != std::string::npos) {
        flag = a.substr(0, eq);
        attached = a.substr(eq + 1);
        has_attached = true;
      }
    } else if (a.size() > 2) {  // a short flag with its value attached
      flag = a.substr(0, 2);
      attached = a.substr(2);
      has_attached = true;
    }
    const Opt* opt = nullptr;
    for (const Opt& o : kOpts)
      if (flag == o.lng || (o.shrt[0] && flag == o.shrt)) opt = &o;
    std::vector<std::string> g = {opt ? canonical(opt->lng) : flag};
    if (has_attached) g.push_back(attached);
    if (opt && opt->arity < -1) {
      // Up to -arity numbers.
      while ((int)g.size() <= -opt->arity && i + 1 < argv.size() && is_number(argv[i + 1])) g.push_back(argv[++i]);
    } else if (opt && opt->arity < 0) {
      // --roi: auto [h|v], or x y w h [rot_cw [hflip [vflip]]].
      while (g.size() < 8 && i + 1 < argv.size()) {
        const std::string& v = argv[i + 1];
        const bool ok = g.size() == 1   ? (v == "auto" || is_number(v))
                        : g[1] == "auto" ? (g.size() == 2 && (v == "h" || v == "v" || v == "H" || v == "V"))
                                         : is_number(v);
        if (!ok) break;
        g.push_back(argv[++i]);
      }
    } else if (opt) {
      for (int k = (int)g.size() - 1; k < opt->arity && i + 1 < argv.size(); ++k) g.push_back(argv[++i]);
    }
    if (opt && opt->skip) continue;
    groups.push_back(g);
  }
  std::stable_sort(groups.begin(), groups.end(),
                   [](const std::vector<std::string>& x, const std::vector<std::string>& y) { return x[0] < y[0]; });
  std::vector<std::string> out;
  for (const auto& g : groups) out.insert(out.end(), g.begin(), g.end());
  out.push_back("--");
  out.insert(out.end(), positional.begin(), positional.end());
  return out;
}

// Whether two argument tokens are the same number (1 and 1.0, 0.5 and .5): a finite number in full on both sides.
static bool same_number(const std::string& a, const std::string& b) {
  char* ea = nullptr;
  char* eb = nullptr;
  const double va = std::strtod(a.c_str(), &ea);
  const double vb = std::strtod(b.c_str(), &eb);
  return !a.empty() && !b.empty() && ea && *ea == '\0' && eb && *eb == '\0' && std::isfinite(va) &&
         std::isfinite(vb) && va == vb;
}

// Whether the previous run (its sidecar's argv) was given the same arguments as this one, apart from the output and the
// knobs: the same normalised arguments (result_args), a number also matching the same number written otherwise (1 and
// 1.0) and a path the same existing file. Any option that changes the result (the profile, the flat field, -r/-g/-b,
// -E, --target, the gains, --roi, --film_format, -P, --fit_curves, -q, --no_crop, --dino, --dino_iters ...) must
// therefore be the same. A sidecar without an argv array matches nothing.
static bool argv_matches_except_knobs(const nlohmann::ordered_json& argv_json, const Options& o, std::string& why) {
  if (!argv_json.is_array()) {
    why = "it records no argv";
    return false;
  }
  std::vector<std::string> prev, cur;
  for (const auto& a : argv_json) {
    if (!a.is_string()) {
      why = "its argv is not a list of strings";
      return false;
    }
    prev.push_back(a.get<std::string>());
  }
  // This run's arguments as the sidecar would record them (utf8_replaced), as the previous run's were.
  for (int i = 0; i < o.argc; ++i) cur.push_back(utf8_replaced(o.argv[i]));
  const std::vector<std::string> pa = result_args(prev), ca = result_args(cur);
  if (pa.size() != ca.size()) {
    why = "it was given other arguments";
    return false;
  }
  for (size_t i = 0; i < pa.size(); ++i) {
    if (pa[i] != ca[i] && !same_number(pa[i], ca[i]) &&
        !(is_regular_file(pa[i]) && is_regular_file(ca[i]) && paths_match(pa[i], ca[i]))) {
      why = "it was given " + pa[i] + " where this run has " + ca[i];
      return false;
    }
  }
  return true;
}

// A previous run's print, for a run that reuses its sidecar: profile.print's A and its search's record; a knob run's
// sidecar records the A before the knobs (knobs.base.a), which the reusing run starts from.
static void read_cached_print(const nlohmann::ordered_json& j, CachedSidecar& c) {
  const nlohmann::ordered_json none;
  const auto& prof = j.contains("profile") && j["profile"].is_object() ? j["profile"] : none;
  const auto& print = prof.is_object() && prof.contains("print") && prof["print"].is_object() ? prof["print"] : none;
  if (print.is_object() && print.contains("a") && print["a"].is_number()) {
    c.print_a = print["a"].get<double>();
    if (print.contains("search") && print["search"].is_object()) c.print_search_json_str = print["search"].dump();
  }
  const auto& knobs = j.contains("knobs") && j["knobs"].is_object() ? j["knobs"] : none;
  const auto& base = knobs.is_object() && knobs.contains("base") && knobs["base"].is_object() ? knobs["base"] : none;
  if (base.is_object() && base.contains("a") && base["a"].is_number()) c.print_a = base["a"].get<double>();
}

// Recorded G and B input curves (--fit_curves ref:..., or a previous run's sidecar) on the bracket profile pd they are
// composed into: the shape the fit holds every curve to (held at or below ref, monotonic and within max_dev_stops of
// the base TRC over the whole input table, curve_shape_ok). Numbers typed by hand, or edited, could step the table at
// ref, fold it over or move it arbitrarily far from the bracket's TRC; a fit's own numbers (written to 9 significant
// digits, which round-trip a float) pass. -1 when both pass, else the channel that does not (0 G, 1 B).
static int recorded_curve_off_shape(const negicc::InputCurves& c, const ProfileData& pd) {
  const float dev = negicc::CurveFitOptions().max_dev_stops;
  for (int i = 0; i < 2; ++i) {
    if (!negicc::curve_shape_ok(c.a[i], c.k[i], c.u_med[i], c.ref[i], pd.in_trc_size[i + 1], dev, c.u_hold(i))) {
      return i;
    }
  }
  return -1;
}

// Whether the search's passes run on the GPU: a USE_CUDA=1 build with a device that runs its kernels, unless
// --dino_backend cpu or --check_renderer (the CPU's, counted, and compared with the GPU's where it can run them).
static bool search_runs_on_gpu(const Options& o) { return device_allowed(o) && negicc::gpu_device_available(); }

// The checks and the reading of a previous run's sidecar j (try_load_cached_sidecar); false with why when it does not
// serve this conversion.
static bool load_cached_sidecar(const nlohmann::ordered_json& j, const Options& o, Setup& s, std::string& why) {
  if (j.value("schema", "") != "negicc.neg_process.sidecar/1") return why = "it has another schema", false;
  // The same build: another neg_process may convert the same arguments differently.
  if (!j.contains("tool") || !j["tool"].is_object() ||
      j["tool"].value("version", "") != std::string(neg_process_version())) {
    return why = std::string("it was written by another neg_process build (this one is ") + neg_process_version() + ")",
           false;
  }
  if (!argv_matches_except_knobs(j.value("argv", nlohmann::ordered_json()), o, why)) return false;
  // --check_renderer checks the search's passes, so it runs them. A search's result depends on where its passes ran
  // (the GPU's and the CPU's differ in their last digits): the sidecar's must have run where this run's would.
  if (o.check_renderer) return why = "--check_renderer runs the search", false;
  if (j.contains("auto") && j["auto"].is_object()) {
    const std::string ran = j["auto"].value("renderer", std::string("cpu"));
    const bool ran_on_gpu = ran == "gpu";   // "cpu, compared with the gpu": the search read the CPU's passes
    if (ran_on_gpu != search_runs_on_gpu(o)) {
      return why = std::string("its search's passes ran on the ") + (ran_on_gpu ? "GPU" : "CPU") +
                   " and this run's run on the " + (ran_on_gpu ? "CPU" : "GPU"), false;
    }
  }

  if (!j.contains("inputs") || !j["inputs"].is_object() || !j["inputs"].contains("raw_files") ||
      !j["inputs"]["raw_files"].is_array()) {
    return why = "it records no inputs", false;
  }
  const auto& raw_files = j["inputs"]["raw_files"];
  if (raw_files.size() != o.files.size()) return why = "it converted another number of files", false;
  for (size_t i = 0; i < o.files.size(); ++i) {
    if (!raw_files[i].is_string() || !paths_match(raw_files[i].get<std::string>(), o.files[i])) {
      return why = "it converted other files", false;
    }
  }
  // The same files, not only the same names: a capture saved again under the same name has another size or CRC-32.
  // (Not the modification time: a file copied or downloaded again, as negicc-web stages its inputs, keeps its content.)
  const auto& stats = j["inputs"].value("raw_file_stats", nlohmann::ordered_json());
  if (!stats.is_array() || stats.size() != o.files.size()) {
    return why = "it records no input file sizes and CRCs", false;
  }
  s.input_fingerprints.assign(o.files.size(), FileFingerprint{});
  for (size_t i = 0; i < o.files.size(); ++i) {
    FileFingerprint& fp = s.input_fingerprints[i];
    fp.ok = file_fingerprint(o.files[i], fp.size, fp.crc);
    const auto& st = stats[i];
    // Both recorded, as unsigned integers: a missing key must not read as 0 and match an empty file's CRC.
    if (!st.is_object() || !st.contains("size") || !st["size"].is_number_unsigned() || !st.contains("crc32") ||
        !st["crc32"].is_number_unsigned()) {
      return why = "it records no size and CRC-32 for " + o.files[i], false;
    }
    if (!fp.ok || st["size"].get<uintmax_t>() != fp.size || st["crc32"].get<uint32_t>() != fp.crc) {
      return why = o.files[i] + " is not the file it converted", false;
    }
  }

  // A film base capture is part of what was converted: the same file, not only the same name.
  if (!o.film_base_file.empty()) {
    const auto fbj = j.value("film_base", nlohmann::ordered_json());
    const auto cap = fbj.is_object() ? fbj.value("capture", nlohmann::ordered_json()) : nlohmann::ordered_json();
    const auto cf = cap.is_object() ? cap.value("file", nlohmann::ordered_json()) : nlohmann::ordered_json();
    uintmax_t size = 0;
    uint32_t crc = 0;
    if (!cf.is_object() || !cf.contains("size") || !cf["size"].is_number_unsigned() || !cf.contains("crc32") ||
        !cf["crc32"].is_number_unsigned() || !file_fingerprint(o.film_base_file, size, crc) ||
        cf["size"].get<uintmax_t>() != size || cf["crc32"].get<uint32_t>() != crc) {
      return why = o.film_base_file + " is not the film base capture it converted with", false;
    }
  }

  if (!j.contains("profile") || !j["profile"].is_object()) return false;
  const auto& prof = j["profile"];
  const std::string prof_type = prof.value("type", "");
  const std::string prof_path = prof.value("path", "");
  if (s.is_bundle) {
    if (prof_type != "bundle") return false;
    if (!paths_match(prof_path, s.bundle.path)) return false;
    if (prof.value("input_space", "") != std::string(to_string(s.bundle.input_space))) return false;
  } else {
    if (prof_type != "icc") return false;
    if (!paths_match(prof_path, s.film_prof_arg) && !paths_match(prof_path, resolve_profile_path(s.film_prof_arg))) {
      return false;
    }
  }

  if (!j.contains("geometry") || !j["geometry"].is_object()) return false;
  const auto& geom = j["geometry"];
  if (!geom.contains("rot_cw") || !geom["rot_cw"].is_number_integer() || geom["rot_cw"].get<int>() != o.rot_cw) {
    return false;
  }
  if (!geom.contains("hflip") || !geom["hflip"].is_boolean() || geom["hflip"].get<bool>() != o.hflip) return false;
  if (!geom.contains("vflip") || !geom["vflip"].is_boolean() || geom["vflip"].get<bool>() != o.vflip) return false;
  const std::string roi_source = geom.value("roi_source", "");
  if (o.roi_auto) {
    if (roi_source != "auto" && roi_source != "auto_fallback") return false;
    if (j.contains("frame_detection") && j["frame_detection"].is_object()) {
      // The format the detector found (with --film_format auto it records what it found, not "auto").
      if (o.film_format != FILM_FORMAT_AUTO &&
          j["frame_detection"].value("film_format", "") != std::string(film_format_name(o.film_format))) {
        return false;
      }
    }
  } else if (o.has_roi) {
    if (roi_source != "cli" || !geom.contains("roi") || !geom["roi"].is_array() || geom["roi"].size() != 4) {
      return false;
    }
    for (int k = 0; k < 4; ++k) {
      if (!geom["roi"][k].is_number_integer() || geom["roi"][k].get<int>() != o.roi[k]) return false;
    }
  } else {
    if (roi_source != "none") return false;
  }

  if (!j.contains("matrix") || !j["matrix"].is_object() || !j["matrix"].contains("pins") ||
      !j["matrix"]["pins"].is_object()) {
    return false;
  }
  const bool saved_pin_e = j["matrix"]["pins"].value("e", false);
  if (saved_pin_e != s.exposure_given) return false;

  // Match either a previous bundle Auto run ("auto") or a previous single-bracket run on an ICC profile ("dino").
  if (s.is_bundle && (!s.exposure_given || s.pinned_search) && j.contains("auto") && j["auto"].is_object()) {
    const auto& a = j["auto"];
    if (!a.contains("target_idx") || !a["target_idx"].is_number_integer()) return false;
    const int tgt_idx = a["target_idx"].get<int>();
    if (tgt_idx < 0 || tgt_idx >= (int)s.bundle.targets.size()) return false;
    if (s.manual_target_idx >= 0 && s.manual_target_idx != tgt_idx) return false;
    if (a.value("target", "") != s.bundle.targets[tgt_idx].name) return false;
    if (!a.contains("e") || !a["e"].is_number() || !a.contains("g") || !a["g"].is_number() || !a.contains("b") ||
        !a["b"].is_number()) {
      return false;
    }
    const float base_e = a["e"].get<float>();
    const float base_g = a["g"].get<float>();
    const float base_b = a["b"].get<float>();
    auto positive = [](float v) { return std::isfinite(v) && v > 0.0f; };
    if (!positive(base_e) || !positive(base_g) || !positive(base_b))
      return why = "its Auto result is not positive", false;

    s.has_cached_sidecar = true;
    s.bundle_search = true;
    s.cached_sidecar.is_auto = true;
    read_cached_print(j, s.cached_sidecar);
    s.cached_sidecar.target_idx = tgt_idx;
    s.cached_sidecar.target_name = s.bundle.targets[tgt_idx].name;
    s.cached_sidecar.base_e = base_e;
    s.cached_sidecar.base_g = base_g;
    s.cached_sidecar.base_b = base_b;
    s.cached_sidecar.auto_json_str = a.dump();
    s.dino_path = a.value("model", "");
    s.cached_sidecar.dino_backend = a.value("backend", o.dino_backend);
    // Fitted G/B input curves of the previous run (the curve fit's): the knobs act on its result (the
    // re-metered E and the fit's g, b) through the same combined profile, so the curves are replayed, not refitted.
    // A fit that fell back (applied false, channels null) has no curves to replay, but its re-metered values are the
    // base all the same.
    const bool no_curves = j.contains("input_curves") && j["input_curves"].is_object() &&
                           j["input_curves"].value("applied", true) == false &&
                           j["input_curves"].value("channels", nlohmann::ordered_json()).is_null();
    // The re-metered values replace the checked ones: checked again (a remeter.e of -1 aborted solve_grey_knobs).
    auto read_remeter = [&](const nlohmann::ordered_json& ic) {
      if (!ic.contains("remeter") || !ic["remeter"].is_object()) return true;
      const auto& rm = ic["remeter"];
      s.cached_sidecar.base_e = rm.value("e", base_e);
      s.cached_sidecar.base_g = rm.value("g", base_g);
      s.cached_sidecar.base_b = rm.value("b", base_b);
      return positive(s.cached_sidecar.base_e) && positive(s.cached_sidecar.base_g) &&
             positive(s.cached_sidecar.base_b);
    };
    if (no_curves) {
      s.cached_sidecar.input_curves_json_str = j["input_curves"].dump();
      if (!read_remeter(j["input_curves"])) return why = "its re-metered values are not positive", false;
    }
    if (j.contains("input_curves") && j["input_curves"].is_object() && !no_curves) {
      const auto& ic = j["input_curves"];
      negicc::InputCurves c;
      bool ok = ic.contains("channels") && ic["channels"].is_object() && ic.value("target_idx", -1) == tgt_idx;
      const char* names[2] = {"G", "B"};
      for (int i = 0; ok && i < 2; ++i) {
        const auto& ch = ic["channels"].value(names[i], nlohmann::ordered_json());
        ok = ch.is_object() && ch.contains("a") && ch.contains("k") && ch.contains("ref") && ch.contains("u_med");
        if (!ok) break;
        c.a[i] = ch["a"].get<float>();
        c.k[i] = ch["k"].get<float>();
        c.ref[i] = ch["ref"].get<float>();
        c.u_med[i] = ch["u_med"].get<float>();
        if (ch.contains("u_lo")) c.u_lo[i] = ch["u_lo"].get<float>();
      }
      c.hold = ic.value("hold", false);
      if (c.hold && !(ic["channels"]["G"].contains("u_lo") && ic["channels"]["B"].contains("u_lo"))) ok = false;
      for (int i = 0; ok && i < 2; ++i) {
        ok = std::isfinite(c.a[i]) && std::isfinite(c.k[i]) && std::isfinite(c.u_med[i]) && std::isfinite(c.u_lo[i]);
      }
      if (!ok || !positive(c.ref[0]) || !positive(c.ref[1])) return why = "its input curves are not valid", false;
      if (recorded_curve_off_shape(c, s.bundle.targets[tgt_idx].film_prof.data) >= 0) {
        return why = "its input curves are held above ref, are not monotonic, or depart too far from the bracket's "
                     "input TRC", false;
      }
      s.cached_sidecar.has_curves = true;
      s.cached_sidecar.curves = c;
      s.cached_sidecar.curves_source = ic.value("source", "fit");
      s.cached_sidecar.input_curves_json_str = ic.dump();
      if (!read_remeter(ic)) return why = "its re-metered values are not positive", false;
    }
    if (j.contains("intent_field") && j["intent_field"].is_object()) {
      s.cached_sidecar.intent_field_json_str = j["intent_field"].dump();
    }
    if (j.contains("heads") && j["heads"].is_object()) s.cached_sidecar.heads_json_str = j["heads"].dump();
    // The tone curve it applied: replayed as recorded (the knobs act on top).
    if (j.contains("tone_stage") && j["tone_stage"].is_object() && j["tone_stage"].value("applied", false)) {
      const auto& cv = j["tone_stage"].value("curve", nlohmann::ordered_json());
      double q[4];
      const char* keys[4] = {"black", "white", "gamma", "contrast"};
      bool ok = cv.is_object();
      for (int i = 0; ok && i < 4; ++i) {
        ok = cv.contains(keys[i]) && cv[keys[i]].is_number();
        if (ok) q[i] = cv[keys[i]].get<double>();
      }
      if (!ok) return why = "its tone curve is not valid", false;
      ToneCurve& tc = s.cached_sidecar.tone_curve;
      tc.black = q[0];
      tc.white = q[1];
      tc.gamma = q[2];
      tc.contrast = q[3];
      if (!tone_curve_ok(tc)) return why = "its tone curve is not one the tone stage makes", false;
      s.cached_sidecar.has_tone = true;
    }
    return true;
  }

  if (!s.is_bundle && j.contains("dino") && j["dino"].is_object()) {
    const auto& d = j["dino"];
    float base_e = 0.0f, base_g = 0.0f, base_b = 0.0f;
    if (j.contains("knobs") && j["knobs"].is_object() && j["knobs"].contains("base") &&
        j["knobs"]["base"].is_object()) {
      const auto& kb = j["knobs"]["base"];
      if (!kb.contains("e") || !kb["e"].is_number() || !kb.contains("g") || !kb["g"].is_number() ||
          !kb.contains("b") || !kb["b"].is_number()) {
        return false;
      }
      base_e = kb["e"].get<float>();
      base_g = kb["g"].get<float>();
      base_b = kb["b"].get<float>();
    } else if (!j.contains("cli_equivalent") || !extract_cli_egb(j["cli_equivalent"], base_e, base_g, base_b)) {
      return false;
    }
    if (!std::isfinite(base_e) || !std::isfinite(base_g) || !std::isfinite(base_b) || !(base_e > 0.0f) ||
        !(base_g > 0.0f) || !(base_b > 0.0f)) {
      return why = "its result is not positive", false;
    }
    if (s.exposure_given && std::fabs(base_e - o.exposure_comp) > 1e-4f * std::max(1.0f, std::fabs(o.exposure_comp))) {
      return false;
    }

    s.has_cached_sidecar = true;
    s.cached_sidecar.is_auto = false;
    read_cached_print(j, s.cached_sidecar);
    s.cached_sidecar.target_idx = -1;
    s.cached_sidecar.base_e = base_e;
    s.cached_sidecar.base_g = base_g;
    s.cached_sidecar.base_b = base_b;
    s.cached_sidecar.dino_json_str = d.dump();
    if (j.contains("intent_field") && j["intent_field"].is_object()) {
      s.cached_sidecar.intent_field_json_str = j["intent_field"].dump();
    }
    s.dino_path = d.value("model", "");
    s.cached_sidecar.dino_backend = d.value("backend", o.dino_backend);
    if (j.contains("timings_ms") && j["timings_ms"].is_object()) {
      const auto& tm = j["timings_ms"];
      if (tm.contains("dino_views") && tm["dino_views"].is_number()) {
        s.cached_sidecar.t_dino_views_ms = tm["dino_views"].get<double>();
      }
      if (tm.contains("dino_render") && tm["dino_render"].is_number()) {
        s.cached_sidecar.t_dino_render_ms = tm["dino_render"].get<double>();
      }
      if (tm.contains("dino_inference") && tm["dino_inference"].is_number()) {
        s.cached_sidecar.t_dino_infer_ms = tm["dino_inference"].get<double>();
      }
      if (tm.contains("dino_search") && tm["dino_search"].is_number()) {
        s.cached_sidecar.t_dino_search_ms = tm["dino_search"].get<double>();
      }
    }
    return true;
  }

  return false;
}

// Checks whether <output>.json exists and was written by this build for the same inputs and arguments (apart from the
// output and the knobs), so --brightness, --tint and --warmth move its Auto or DINOv3 result instead of searching
// again.
// Anything unreadable or different refuses the reuse, says why, and leaves s as it was.
static bool try_load_cached_sidecar(const Options& o, Setup& s) {
  const std::string sidecar_path = o.output + ".json";
  if (!is_regular_file(sidecar_path)) return false;
  std::ifstream ifs(sidecar_path);
  if (!ifs.is_open()) return false;
  const nlohmann::ordered_json j = nlohmann::ordered_json::parse(ifs, nullptr, false);
  const bool had = s.has_cached_sidecar, had_search = s.bundle_search;
  const CachedSidecar had_cache = s.cached_sidecar;
  const std::string had_dino_path = s.dino_path;
  std::string why = "it does not match this conversion";
  bool ok = false;
  if (s.density && !o.prints.empty()) {
    // The knobs move each material's print search (density_print_searches), which runs again for them.
    why = "a density output's knobs move its print searches, which run again";
  } else if (!j.is_object()) {
    why = "it is not JSON";
  } else {
    try {
      ok = load_cached_sidecar(j, o, s, why);
    } catch (const std::exception& e) {  // a field of the wrong type
      why = std::string("it cannot be read (") + e.what() + ")";
      ok = false;
    }
    // A print's knobs move the print's A: a sidecar without it, or with one outside this print's range (another
    // bundle's curve), has no print result to move.
    if (ok && !s.print_profs.empty() &&
        !(s.cached_sidecar.print_a >= s.print_cont.a_lo() && s.cached_sidecar.print_a <= s.print_cont.a_hi())) {
      why = "it records no print mid grey A (profile.print.a) within this print's range";
      ok = false;
    }
  }
  if (!ok) {
    s.has_cached_sidecar = had;
    s.bundle_search = had_search;
    s.cached_sidecar = had_cache;
    s.dino_path = had_dino_path;
    log_info("Not reusing %s: %s; converting from scratch\n", sidecar_path.c_str(), why.c_str());
  }
  return ok;
}

// The paper's white is the display's white: a print profile made media-relative, as the ICC defines it (ICC.1:2022
// 6.3.2.2: each of X, Y and Z scaled by the PCS white's over the medium's), as an eye adapted to the paper sees it.
// The node of largest Y in the print's table (PCS XYZ D50 as a lut16 holds it, its code scaled by kPcsScale) is the
// paper's base; each of X, Y and Z of every node is scaled by D50's over the base's, so the base is exactly the
// display's white (its tint is not shown) and no node's Y passes D50's. Another node's X or Z can pass D50's, by a few
// parts in a thousand on the shipped prints, and nothing clamps them; a saturated light colour can also have an sRGB
// channel above 1, which the print, keeping no shoulder, cuts. The bundle's print ICC itself stays absolute. paper_xyz gets the
// base.
static void to_paper_white(ParsedFilmProfile& p, double* paper_xyz = nullptr) {
  const double* d50 = negicc::kPcsWhite;   // the white the outputs write at full scale
  size_t best = 0;
  for (size_t i = 1; i < p.clut.size(); i += 3)
    if (p.clut[i] > p.clut[best * 3 + 1]) best = i / 3;
  double base[3], k[3];
  for (int c = 0; c < 3; ++c) {
    base[c] = (double)p.clut[best * 3 + c] * negicc::kPcsScale;   // in double
    k[c] = d50[c] / base[c];
    if (paper_xyz) paper_xyz[c] = base[c];
  }
  for (size_t i = 0; i < p.clut.size(); ++i) p.clut[i] *= (float)k[i % 3];
  p.rebind_pointers();
}

// --check_renderer's renderer (multi_target_optimizer.h PassThroughRenderer).
static PassThroughRenderer& pass_through_renderer() {
  static PassThroughRenderer r;
  return r;
}

// Rows given with -r, -g and -b that leave the crosstalk correction, as it applies, not invertible.
static const char kRowsNotInvertible[] =
    "-r, -g and -b: the crosstalk correction matrix is not an invertible 3x3 of numbers within float range (its "
    "float determinant a normal float)\n";

// The renderer of the search's passes: --check_renderer's (counting, and comparing with the GPU's where a device runs
// its kernels), the GPU's (search_runs_on_gpu), or null (the CPU's).
static PreviewRenderer* search_renderer(const Options& o, const Setup& s) {
  if (o.check_renderer && !s.renderer) return &pass_through_renderer();
  return s.renderer.get();
}

int prepare(const Options& o, Setup& s) {
  if (o.check_renderer) negicc::cpu_pass_hook = audit_cpu_pass;
  if (!o.flat.empty()) {
    try {
      s.flat_field = negicc::load_flat_field_file(o.flat);
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
  }

  // The profile: -p, else the built-in bundle of --film_stock and --light_source (or the light the input implies).
  ProfileBundle& bundle = s.bundle;
  if (!o.light_source.empty() && o.implied_light && o.light_source != o.implied_light && !o.tiff_input) {
    log_error("--light_source %s does not fit %s, which takes %s\n", o.light_source.c_str(),
            o.input_mode == InputMode::kLinearDng ? "a linear DNG"
            : o.input_mode == InputMode::kThreeShot ? "three captures, one per light"
                                                    : "a JPEG, converted from sRGB to linear Rec.2020",
            o.implied_light);
    return 1;
  }
  if (!o.light_source.empty() && o.film_stock.empty()) {
    log_error("--light_source requires --film_stock\n");
    return 1;
  }
  if (!o.film_profile.empty()) {
    s.film_prof_arg = o.film_profile;
    try {
      s.is_bundle = load_bundle_arg(s.film_prof_arg, bundle);
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    if (!s.is_bundle && !is_regular_file(resolve_profile_path(s.film_prof_arg))) {
      log_error("No profile bundle JSON or ICC profile '%s'\n", s.film_prof_arg.c_str());
      return 1;
    }
  } else if (!o.film_stock.empty()) {
    if (o.light_source.empty() && !o.implied_light) {
      log_error("--film_stock requires --light_source (tri or d65)\n");
      return 1;
    }
    const std::string light = !o.light_source.empty() ? o.light_source : std::string(o.implied_light);
    if (o.light_source.empty()) log_info("Light source: %s, from the input\n", light.c_str());
    std::string error;
    const std::string key = embedded_bundle_key(o.film_stock, light, error);
    if (key.empty()) {
      log_error("%s (--list_profiles lists them)\n", error.c_str());
      return 1;
    }
    try {
      if (!load_embedded_bundle(key, bundle)) {
        log_error("Cannot load built-in profile bundle %s\n", key.c_str());
        return 1;
      }
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    s.is_bundle = true;
  }
  if (s.is_bundle) {
    log_info("Profile bundle: %s (%s, %zu brackets)\n", bundle.key.c_str(), bundle.path.c_str(), bundle.targets.size());
  }
  s.has_profile = s.is_bundle || !s.film_prof_arg.empty();
  if (o.knobs_given && !s.has_profile) {
    log_error("--brightness, --tint and --warmth require a film profile (--film_stock and --light_source, or "
              "--film_profile)\n");
    return 1;
  }

  if (o.target_given) {
    if (!s.is_bundle) {
      log_error("--target requires a JSON profile bundle\n");
      return 1;
    }
    const std::string& tgt_str = o.target;
    char* end = nullptr;
    long val = strtol(tgt_str.c_str(), &end, 10);
    if (end && *end == '\0' && val >= 0 && val < (long)bundle.targets.size()) {
      s.manual_target_idx = (int)val;
    } else {
      for (size_t i = 0; i < bundle.targets.size(); ++i) {
        if (bundle.targets[i].name == tgt_str || bundle.targets[i].display_name == tgt_str) {
          s.manual_target_idx = (int)i;
          break;
        }
      }
    }
    if (s.manual_target_idx < 0) {
      log_error("Target '%s' not found in bundle\n", tgt_str.c_str());
      return 1;
    }
  }
  s.exposure_given = o.exposure_given;

  // The crosstalk correction: the bundle's, with any rows -r, -g and -b give in place of its own (for the search too).
  s.r_coeff = o.r_coeff;
  s.g_coeff = o.g_coeff;
  s.b_coeff = o.b_coeff;
  const bool* row_given = o.row_given;
  std::vector<float>* rows[3] = {&s.r_coeff, &s.g_coeff, &s.b_coeff};
  // A rec2020_linear (white-light) bundle takes the white-light decode, which has no crosstalk to correct and, as in
  // negicc-station, decodes single-shot captures only.
  s.white_light = s.is_bundle && bundle.input_space == InputSpace::kRec2020Linear;
  s.camera_rgb_input = o.tiff_input && s.is_bundle && !s.white_light;
  // A film base capture is read as a TriLED scan's mosaic is: camera RGB through the bundle's crosstalk correction.
  if (!o.film_base_file.empty() && (!s.is_bundle || s.white_light)) {
    log_error("--film_base with a capture needs a TriLED (camera_raw_rgb) profile bundle: the capture is read as the "
              "scan's camera RGB, through the bundle's crosstalk correction\n");
    return 1;
  }
  // The density output: the bundle's matrix for the measure, on either light.
  if (o.density) {
    if (!s.is_bundle) {
      log_error("-P %s needs a profile bundle (--film_stock and --light_source, or -p a bundle): the density "
                "matrices are the bundle's\n", o.colorspace.c_str());
      return 1;
    }
    if (!bundle.density_matrix[o.density_measure].present) {
      log_error("The profile bundle %s has no density_matrices.%s: rebuild the bundle with negicc-station "
                "(tools/fit_density_matrices.py), which fits the matrix from the light's spectrum\n",
                bundle.path.c_str(), density_measure_key(o.density_measure));
      return 1;
    }
    // The curve fit is white light's: on a TriLED bundle the curves are the measured ones, and an explicit
    // --fit_curves is refused rather than ignored.
    if (!s.white_light && o.fit_curves != Options::kFitCurvesAuto) {
      log_error("--fit_curves does not apply to a density output from a TriLED bundle (-P %s): its curves are the "
                "measured ones and are not fitted\n", o.colorspace.c_str());
      return 1;
    }
    s.density = true;
  }
  // --print. With a picture output the print's own search (A, g, b) runs on the print profile, with no chart profile,
  // and the conversion renders that profile at the search's A: the bundle's print ICC (the print declarations above
  // bracket_print_profiles), built here and made relative to the paper's white. A density output writes a CLF per
  // material, each material's print searched for its CLF (density_print_searches). A white-light bundle takes the same
  // print on its decode, with no curve fit: the fit composes into
  // the chart profiles, which a print does not use.
  if (!o.prints.empty()) {
    // On a density output the knobs move each print's search, which needs the film's grey balance curve.
    if (s.density && o.knobs_given && !bundle.grey_balance.present) {
      log_error("--brightness, --tint and --warmth on a density output move each print's search, which needs the "
                "profile bundle's grey_balance_curves\n");
      return 1;
    }
    if (!s.is_bundle) {
      log_error("--print needs a profile bundle (--film_stock and --light_source, or -p a bundle): the print is "
                "computed for the bundle's film\n");
      return 1;
    }
    if (o.fit_curves != Options::kFitCurvesAuto || o.input_curves_given) {
      log_error("--fit_curves does not apply with --print: the print is made by its own search, without the curve "
                "fit\n");
      return 1;
    }
    if (!s.density) {
      const std::string& print = o.prints.front();
      BracketPrints prints;
      std::string why;
      if (!bracket_print_profiles(bundle, print, prints, why)) {
        log_error("--print %s: %s\n", print.c_str(), why.c_str());
        return 1;
      }
      for (size_t i = 0; i < bundle.targets.size(); ++i) {
        std::vector<uint8_t>& icc = prints.icc[i];
        cmsHPROFILE h = cmsOpenProfileFromMem(icc.data(), (cmsUInt32Number)icc.size());
        ParsedFilmProfile prof;
        const bool ok = h && parse_film_profile_pipeline(h, prof);
        if (h) cmsCloseProfile(h);
        if (!ok) {
          log_error("--print %s: the print ICC of %s cannot be read\n", print.c_str(), bundle.targets[i].name.c_str());
          return 1;
        }
        s.print_profs.push_back(std::move(prof));
        s.print_iccs.push_back(std::move(icc));
      }
      for (ParsedFilmProfile& p : s.print_profs) p.rebind_pointers();
      // The paper's white is the display's white: the print is media-relative (to_paper_white); the profile an output
      // embeds carries the same scaled table.
      for (ParsedFilmProfile& p : s.print_profs) to_paper_white(p, s.print_paper_xyz);
      {
        double k[3];
        const double* d50 = negicc::kPcsWhite;
        for (int c = 0; c < 3; ++c) k[c] = d50[c] / s.print_paper_xyz[c];
        // The table is now media-relative: the paper's XYZ goes into wtpt, as the ICC means a media-relative table (a
        // viewer on the relative intent shows the picture; on the absolute intent it gets the paper back). negicc
        // reads only the A2B0.
        for (std::vector<uint8_t>& icc : s.print_iccs) {
          scale_print_icc_table(icc, k);
          set_icc_wtpt(icc, s.print_paper_xyz);
        }
      }
      s.print_cont = prints.continuum;
      s.print_a = NAN;   // the print's search sets it (or a reused run's)
      {
        const PrintContinuum* pc = &s.print_cont;
        s.print_in.trc_at = [pc](double a, const double gains[3], std::vector<float> trc[3]) {
          pc->trc_at(a, gains, trc);
        };
        s.print_in.a_lo = pc->a_lo();
        s.print_in.a_hi = pc->a_hi();
        s.print_in.a_mid = pc->a_mid;
      }
      s.print_id = prints.id;
      s.print_name = prints.name;
      s.print_file = prints.file;
      s.print_file_sha256 = prints.file_sha256;
      s.print_from_bundle = prints.from_bundle;
      log_info("Print: %s (%s), the print ICC %s: with a picture output the print's own search runs on the print "
               "profile (no chart profile) and the conversion renders the print\n",
               prints.name.c_str(), prints.id.c_str(),
               prints.from_bundle ? "of the bundle with each bracket's curves" : "computed for the bundle");
    }
  }
  if (s.white_light) {
    if (row_given[0] || row_given[1] || row_given[2]) {
      log_error("-r, -g and -b cannot be used with the rec2020_linear profile bundle %s: its white-light "
                      "decode has no crosstalk to correct\n", bundle.key.c_str());
      return 1;
    }
    if (o.files.size() == 4) {
      log_error("The rec2020_linear profile bundle %s converts single-shot captures only, not pixel "
                      "shift\n", bundle.key.c_str());
      return 1;
    }
    if (o.input_mode == InputMode::kThreeShot) {
      log_error("The rec2020_linear profile bundle %s is for white light; three captures, one per "
                      "light, take a camera_raw_rgb (tri) bundle\n", bundle.key.c_str());
      return 1;
    }
  }
  // Three captures, one per light, have no crosstalk between the lights to correct: each channel comes from the
  // capture under its own light, so the correction is the identity unless -r, -g or -b give one.
  if (o.input_mode == InputMode::kThreeShot && s.is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) continue;
      for (int c = 0; c < 3; ++c) {
        bundle.crosstalk_matrix[r * 3 + c] = r == c ? 1.0f : 0.0f;
        bundle.crosstalk_matrix_d[r * 3 + c] = r == c ? 1.0 : 0.0;
      }
    }
    bundle.has_crosstalk = true;
  }
  if (s.is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) {
        for (int c = 0; c < 3; ++c) {
          bundle.crosstalk_matrix[r * 3 + c] = (*rows[r])[c];
          bundle.crosstalk_matrix_d[r * 3 + c] = (double)(*rows[r])[c];
        }
      } else if (bundle.has_crosstalk) {
        rows[r]->assign(bundle.crosstalk_matrix + r * 3, bundle.crosstalk_matrix + r * 3 + 3);
      }
    }
  }
  s.has_crosstalk = row_given[0] || row_given[1] || row_given[2] || (s.is_bundle && bundle.has_crosstalk);
  for (int r = 0; s.has_crosstalk && r < 3; ++r) {
    s.crosstalk_row_source[r] = row_given[r] ? "given"
                                : (s.is_bundle && o.input_mode != InputMode::kThreeShot && bundle.has_crosstalk)
                                    ? "bundle"
                                    : "identity";
  }
  // The correction is the inverse of the light's mixing (DESIGN.md 2.4), so it is invertible: rows that leave it
  // singular (all zero, or linearly dependent) would collapse the picture. Checked on the matrix as it applies. With
  // an ICC profile that is the rows here; with a bundle it is known only once the scan's camera has decided which
  // rows the bundle gives (decode).
  if (!s.is_bundle && s.has_crosstalk && (row_given[0] || row_given[1] || row_given[2])) {
    float x[9];
    for (int i = 0; i < 9; ++i) x[i] = (*rows[i / 3])[i % 3];
    if (!crosstalk_matrix_invertible(x)) {
      log_error("%s", kRowsNotInvertible);
      return 1;
    }
  }
  // An RGB TIFF or JPEG is colour-managed RGB, converted to linear Rec.2020: not the camera RGB a crosstalk
  // correction (a camera_raw_rgb bundle) needs.
  if (o.input_mode == InputMode::kRgbFile && !s.camera_rgb_input && s.has_crosstalk) {
    log_error("%s is an RGB TIFF or JPEG, converted to linear Rec.2020: give a rec2020_linear (d65) "
                    "profile bundle, not one with a crosstalk correction (a TIFF of linear camera RGB takes "
                    "--light_source tri)\n", o.files[0].c_str());
    return 1;
  }
  if (o.input_mode == InputMode::kLinearDng && s.is_bundle && !s.white_light) {
    log_error("A linear DNG is converted to linear Rec.2020 with its own colour data: give a "
                    "rec2020_linear (d65) profile bundle\n");
    return 1;
  }
  s.raw_decode = s.white_light ? RawDecode::kWhiteLight
                 : s.has_crosstalk ? RawDecode::kCameraRgb
                                   : RawDecode::kRec2020;
  // Pixel shift merges the four captures' CFA as camera RGB (no LibRaw decode): a profile or an output encoding it
  // goes through must read camera RGB; without either the merge is written as camera-linear. It takes no -q.
  if (o.files.size() == 4 && (s.has_profile || !o.colorspace.empty()) && s.raw_decode != RawDecode::kCameraRgb) {
    log_error("4-shot pixel shift merges the captures' camera RGB: it needs a TriLED (camera_raw_rgb) profile bundle, "
              "or -r, -g or -b with -p (a profile of the corrected camera RGB)\n");
    return 1;
  }
  if (o.files.size() == 4 && o.quality >= 0) {
    log_error("-q does not apply to 4-shot pixel shift: the four captures fill each pixel's colours, with no "
              "demosaicing\n");
    return 1;
  }

  // Fitted G/B input curves (input_curves.h): white light only. A TriLED
  // bundle's curves are measured under the light the scan was made with, so fitting them per frame has no basis; the
  // white-light fit corrects for a lamp and camera the profiles were not made with. The output must apply the film
  // profile (-P): without it the TIFF is the corrected camera RGB with the bracket's own ICC profile, which does not
  // carry the curves.
  const bool fit_on = o.fit_curves == Options::kFitCurvesOn;
  // Where no fit can run, an explicit --fit_curves, off included, is refused rather than ignored (as on a density
  // output from a TriLED bundle, and with --print).
  if (o.fit_curves == Options::kFitCurvesOff && (!s.is_bundle || !s.white_light || s.has_crosstalk)) {
    log_error("--fit_curves off does not apply: the curve fit is a white-light bundle's (d65, rec2020_linear), and a "
              "TriLED bundle or an ICC profile converts through its measured curves, which are not fitted\n");
    return 1;
  }
  if (fit_on || o.input_curves_given) {
    const char* opt = fit_on ? "--fit_curves on" : "--fit_curves with recorded curves";
    if (!s.is_bundle || !s.white_light || s.has_crosstalk) {
      log_error("%s needs a white-light (d65, rec2020_linear) profile bundle: a TriLED bundle converts through its "
                "measured curves, which are not fitted\n", opt);
      return 1;
    }
    if (o.colorspace.empty()) {
      log_error("%s needs -P (e.g. -P srgb): without it the output is the corrected camera RGB with the bracket's ICC "
                "profile, which does not carry the curves\n", opt);
      return 1;
    }
  }
  // The fit scores the grid in CIELAB through linear sRGB (input_curves.h, to_display), which is what -P srgb and
  // srgb-g10 write; a custom -P goes through its ICC profile with LittleCMS, which the fit does not model, while the
  // sidecar recorded the fit as made for it. So fitting takes -P srgb or srgb-g10 (srgb-linear); replaying recorded
  // curves takes any -P, since it renders the combined profile and fits nothing.
  // A density output on white light fits as -P srgb does: its E, g and b are the file's suggested balance, not the
  // pixels' (with --print no fit runs: fit_curves_active).
  const bool fit_encoding_ok = o.colorspace == "srgb" || o.colorspace == "srgb-g10" || o.colorspace == "srgb-linear" ||
                               (s.density && s.white_light);
  // A custom -P profile is checked here, before anything is decoded or searched (output_profile_ok).
  if (!o.colorspace.empty() && !o.density && o.colorspace != "srgb" && o.colorspace != "srgb-g10" &&
      o.colorspace != "srgb-linear" && !output_profile_ok(o.colorspace)) {
    return 1;
  }
  // The tone stage is one L* curve between the display value and the sRGB encoding (color_pipeline.h apply_tone), so
  // it takes -P srgb; it runs on white light and TriLED alike, with or without the curve fit.
  // A print (--print on a profile bundle) has its tone from the print material's own curve: no tone stage.
  const bool tone_on = o.tone_stage == Options::kToneStageOn;
  const bool tone_wanted = o.tone_stage != Options::kToneStageOff && !o.tone_replay_given && o.colorspace == "srgb" &&
                           s.print_id.empty();
  if ((tone_on || o.tone_replay_given) && !s.print_id.empty()) {
    log_error("--tone_stage does not apply with --print: the print material's curve gives the print its tone\n");
    return 1;
  }

  // The parameter search. On a bundle it is negicc-station's conversion parameter search (multi_target_optimizer.h),
  // which picks the bracket, the exposure and the gains.
  //
  // A bundle, on either light: whatever --target, -E, --gain_g and --gain_b give is held and the same search solves the
  // rest (pinned_search), followed by the curve fit (white light) and the tone stage as after an unpinned search,
  // unless turned off. With all four given there is nothing to search: the search then runs only where the curve fit
  // or the tone stage runs (they need the search's intent field and frame), and --fit_curves off with --tone_stage
  // off, another -P, or recorded curves or a recorded tone curve (a sidecar's cli_equivalent) convert with the given
  // values alone.
  //
  // A single ICC profile: the single-bracket solver solves the gains, and the exposure unless -E gives it, unless
  // --gain_g or --gain_b is given.
  {
    const bool gains_given = o.gain_g_given || o.gain_b_given;
    const bool any_pin = s.manual_target_idx >= 0 || o.exposure_given || gains_given;
    const bool all_pinned = s.manual_target_idx >= 0 && o.exposure_given && o.gain_g_given && o.gain_b_given;
    const bool replay = o.input_curves_given || o.tone_replay_given;
    const bool fit_wanted = o.fit_curves != Options::kFitCurvesOff && fit_encoding_ok;
    const bool fit_runs = fit_wanted && s.white_light && !s.has_crosstalk;   // the curve fit is white light's
    s.pinned_search = s.is_bundle && any_pin && !replay && (!all_pinned || fit_runs || tone_wanted);
    s.use_dino = s.has_profile && (!gains_given || s.pinned_search);
    s.bundle_search = s.use_dino && s.is_bundle &&
                      (s.pinned_search || (s.manual_target_idx < 0 && !s.exposure_given));
  }
  if (!o.probe_intent.empty()) {
    // The probe is written by the bundle search: without it there would be no file and nothing said.
    if (!s.bundle_search) {
      log_error("--probe_intent needs the bundle search: a profile bundle, no recorded curves or tone curve, and not "
                "all of --target, -E and the gains given unless the curve fit or the tone stage runs\n");
      return 1;
    }
    for (const std::array<float, 4>& p : o.probe_params) {
      if (p[0] < 0.0f || p[0] >= (float)s.bundle.targets.size()) {
        log_error("--probe_params: target %d is not a bracket of this bundle (0 to %zu)\n", (int)p[0],
                  s.bundle.targets.size() - 1);
        return 1;
      }
    }
  }
  if ((!o.dino_adapter.empty() || !o.neutral_head_file.empty() || !o.tone_head_file.empty()) && !s.use_dino) {
    log_error("--dino requires the parameter search: a profile bundle, unless --target, -E and both gains are given "
              "with neither the curve fit nor the tone stage running, or recorded values; or an ICC profile without "
              "--gain_g or --gain_b\n");
    return 1;
  }
  if (fit_on && !fit_encoding_ok) {
    log_error("--fit_curves on needs -P srgb or -P srgb-g10: the fit scores the frame through linear sRGB, not "
              "through a custom output profile\n");
    return 1;
  }
  if (fit_on && !s.bundle_search) {
    log_error("--fit_curves on fits after the bundle search, which did not run for these arguments\n");
    return 1;
  }
  if ((tone_on || o.tone_replay_given) && o.colorspace != "srgb") {
    log_error("--tone_stage %s needs -P srgb: the stage is a lightness curve applied before the sRGB encoding\n",
              tone_on ? "on" : "curve:");
    return 1;
  }
  if (tone_on && !(s.is_bundle && s.bundle_search)) {
    log_error("--tone_stage on runs after the bundle search, which did not run for these arguments\n");
    return 1;
  }
  // --fit_curves auto or on: the curve fit wherever it applies (on has checked above that it does), nothing
  // elsewhere; off: none. --tone_stage likewise, after every bundle search with -P srgb.
  s.fit_curves_active = o.fit_curves != Options::kFitCurvesOff && s.is_bundle && s.white_light && !s.has_crosstalk &&
                        fit_encoding_ok && s.bundle_search && o.prints.empty();   // a print takes no curve fit
  s.tone_active = tone_wanted && s.is_bundle && s.bundle_search;
  if (o.input_curves_given) {
    // Recorded curves replay a recorded result (a sidecar's cli_equivalent): its bracket and its E, g and b on the
    // combined profile. Without all four values a search would run on the combined profile, which the curves were not
    // fitted with, and give other values.
    if (s.manual_target_idx < 0 || !o.exposure_given || !o.gain_g_given || !o.gain_b_given) {
      log_error("--fit_curves with recorded curves needs --target, -E, --gain_g and --gain_b: the curves belong to "
                "one bracket's profile at the recorded E, g and b (the sidecar's cli_equivalent gives all of them)\n");
      return 1;
    }
    // The shape the fit holds every curve to (recorded_curve_off_shape).
    const int off = recorded_curve_off_shape(o.input_curves, bundle.targets[s.manual_target_idx].film_prof.data);
    if (off >= 0) {
      log_error("--fit_curves: the recorded %s curve is held above ref (u_lo above 0), is not monotonic, or departs "
                "more than %g stops from the bracket's input TRC, over its input table\n", off == 0 ? "G" : "B",
                negicc::CurveFitOptions().max_dev_stops);
      return 1;
    }
    // The --target bracket's profile becomes the combined profile before anything renders through it (its un-curved
    // tables kept for the knobs).
    s.curves_base_idx = s.manual_target_idx;
    s.curves_base_prof = bundle.targets[s.manual_target_idx].film_prof;
    s.curves_base_prof.rebind_pointers();
    apply_input_curves(bundle.targets[s.manual_target_idx].film_prof, o.input_curves);
    s.input_curves = o.input_curves;
    s.input_curves_source = "cli";
  }

  // A recorded tone curve (a sidecar's cli_equivalent) replays a recorded result like the recorded curves: on the
  // --target bracket at the given E, g and b, with nothing searched or predicted.
  if (o.input_curves_given || o.tone_replay_given) s.replay_source = "cli";
  if (o.tone_replay_given) {
    if (!s.is_bundle || s.manual_target_idx < 0 || !o.exposure_given || !o.gain_g_given || !o.gain_b_given) {
      log_error("--tone_stage curve:... replays a recorded result: it needs a profile bundle, --target, -E, --gain_g "
                "and --gain_b (the sidecar's cli_equivalent gives all of them)\n");
      return 1;
    }
    const double* q = o.tone_replay;
    s.tone_curve = {q[0], q[1], q[2], q[3]};
    if (!tone_curve_ok(s.tone_curve)) {
      log_error("--tone_stage curve:%g,%g,%g,%g is not a curve the tone stage makes (black %g to %g, white %g to %g, "
                "gamma %g to %g, contrast %g to %g)\n", q[0], q[1], q[2], q[3], kToneLo[0], kToneHi[0], kToneLo[1],
                kToneHi[1], kToneLo[2], kToneHi[2], kToneLo[3], kToneHi[3]);
      return 1;
    }
    s.has_tone = true;
  }
  // On a bundle the search is always the bundle search: the single-bracket solver is an ICC profile's alone. With the
  // recorded curves' and tone curve's refusals above, no command line reaches the opposite.
  if (s.is_bundle && s.use_dino && !s.bundle_search) {
    log_error("Internal error: a bundle's search is not the bundle search\n");
    return 1;
  }

  // --brightness, --tint and --warmth on a previous run's output (<output>.json): its Auto or DINOv3 result is the
  // base the knobs move, so neither the backbone nor the search runs again; its fitted curves are replayed on its
  // bracket.
  // Not with --probe_intent, which needs the search to run.
  if (o.knobs_given && s.use_dino && o.probe_intent.empty() && try_load_cached_sidecar(o, s)) {
    log_info("Reusing cached metadata from %s (base E %.4f g %.4f b %.4f%s%s%s)\n", (o.output + ".json").c_str(),
             s.cached_sidecar.base_e, s.cached_sidecar.base_g, s.cached_sidecar.base_b,
             s.is_bundle ? ", target " : "", s.is_bundle ? s.cached_sidecar.target_name.c_str() : "",
             s.cached_sidecar.has_curves ? ", fitted curves replayed" : "");
    if (s.cached_sidecar.has_curves && s.input_curves_source.empty()) {
      s.curves_base_idx = s.cached_sidecar.target_idx;
      s.curves_base_prof = bundle.targets[s.cached_sidecar.target_idx].film_prof;
      s.curves_base_prof.rebind_pointers();
      apply_input_curves(bundle.targets[s.cached_sidecar.target_idx].film_prof, s.cached_sidecar.curves);
      s.input_curves = s.cached_sidecar.curves;
      s.input_curves_source = s.cached_sidecar.curves_source;
    }
    s.fit_curves_active = false;  // the curves (or none) come from the previous run
    s.tone_active = false;        // and so does the tone curve (or none)
    s.replay_source = "knobs";
    if (s.cached_sidecar.has_tone) {
      s.has_tone = true;
      s.tone_curve = s.cached_sidecar.tone_curve;
    }
  } else if (s.use_dino) {
    const EmbeddedFile* trt = find_embedded_backbone(true);
    const EmbeddedFile* onnx = find_embedded_backbone(false);
    const EmbeddedFile* backbone = trt ? trt : onnx;
    if (!backbone) {
      // A bundle holds given gains in its search (pinned_search), so they alone do not turn it off. A print and a
      // density output are made by the search and take no given values: they need the backbone.
      log_error("This neg_process was built without the DINOv3 backbone of the parameter search (make "
                "dinov3_model, or make dinov3_engine for USE_CUDA=1, then rebuild)%s\n",
                !s.print_id.empty() || s.density
                    ? ": a print and a density output are made by the search and need it"
                : s.is_bundle && s.white_light
                    ? "; give --target, -E, --gain_g and --gain_b with --fit_curves off and --tone_stage off to "
                      "convert without it"
                : s.is_bundle
                    ? "; give --target, -E, --gain_g and --gain_b with --tone_stage off to convert without it"
                    : "; give --gain_g or --gain_b to convert without it");
      return 1;
    }
    s.dino_path = std::string("[built-in] ") + backbone->name;
    log_info("Loading DINOv3 backbone %s (backend: %s)...\n", s.dino_path.c_str(), o.dino_backend.c_str());
    try {
      s.engine.load_memory(trt ? trt->data : nullptr, trt ? trt->size : 0, onnx ? onnx->data : nullptr,
                           onnx ? onnx->size : 0, o.dino_backend);
      // The file that runs: the .onnx when the engine was not loaded.
      if (trt && std::strcmp(s.engine.backend_used(), "gpu") != 0) {
        s.dino_path = std::string("[built-in] ") + onnx->name;
      }
      // A model's file named by --dino, read whole (an error names the head).
      // A model's file named by --dino, read whole (an error names the head): a regular file read to its end.
      auto read_file = [](const std::string& path, const char* head) {
        if (!is_regular_file(path)) throw std::runtime_error(std::string("--dino ") + head + ": " + path + " is not a file");
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error(std::string("--dino ") + head + ": cannot read " + path);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (in.bad()) throw std::runtime_error(std::string("--dino ") + head + ": cannot read all of " + path);
        return bytes;
      };
      if (!o.dino_adapter.empty()) {
        // The record names the file that ran: adapter.engine behind a TensorRT backbone, adapter.onnx otherwise.
        s.engine.load_adapter_file(o.dino_adapter, 0);  // 0: the program's thread count, as the backbone
        s.head_path[0] = s.engine.adapter_path();
        s.head_fp[0].ok = file_fingerprint(s.head_path[0], s.head_fp[0].size, s.head_fp[0].crc);
      } else {
        std::string adapter_dir;
        const std::string profile_name = !o.film_stock.empty() ? o.film_stock
                                         : s.is_bundle ? bundle.key : s.film_prof_arg;
        bool fell_back = false;
        const EmbeddedFile* a = find_embedded_adapter(profile_name, adapter_dir, &fell_back);
        if (!a) {
          log_error("No intent adapter for profile '%s'\n", profile_name.c_str());
          return 1;
        }
        if (fell_back) {
          log_warn("no intent adapter matches profile '%s'; using %s (--dino gives another)\n",
                   profile_name.c_str(), adapter_dir.c_str());
        }
        // USE_CUDA=1 builds carry the adapter's TensorRT engine beside it (make dinov3_engine).
        const EmbeddedFile* ae = find_embedded_file(adapter_dir + "/adapter.engine");
        // 0: the program's thread count, as the backbone.
        s.engine.set_adapter(a->data, a->size, adapter_dir, 0, ae ? ae->data : nullptr, ae ? ae->size : 0);
        // The record names what ran: the engine on the GPU, the .onnx on the CPU.
        const EmbeddedFile* ran = s.engine.adapter_on_device() ? ae : a;
        s.head_path[0] = std::string("[built-in] ") + ran->name;
        s.head_fp[0] = memory_fingerprint(ran->data, ran->size);
      }
      // The neutral-patch head on the backbone's tokens, for the white-light curve fit's scene-neutral anchors
      // (input_curves.h, JointCurveModel::set_scene_neutral): only where the fit runs.
      if (s.fit_curves_active) {
        std::vector<char> file;
        const void* data = nullptr;
        size_t size = 0;
        std::string name = "[built-in] models/neutral_head/neutral_head.bin";
        if (o.neutral_head_file.empty()) {
          const EmbeddedFile* f = find_embedded_file("models/neutral_head/neutral_head.bin");
          if (f) {
            data = f->data;
            size = f->size;
          }
        } else {
          file = read_file(o.neutral_head_file, "neutral");
          data = file.data();
          size = file.size();
          name = o.neutral_head_file;
        }
        if (data && s.engine.needs_adapter()) {
          try {
            s.engine.set_neutral_head(data, size);
          } catch (const std::exception& e) {  // not a neutral head file: say which file
            if (o.neutral_head_file.empty()) throw;
            throw std::runtime_error("--dino neutral: " + o.neutral_head_file + ": " + e.what());
          }
          s.neutral_head = name;
          s.head_path[1] = name;
          s.head_fp[1] = memory_fingerprint(data, size);
          log_info("Neutral head %s\n", name.c_str());
        }
      }
      // The tone heads read the pooled features and the tokens' spread of one inference (only where the tone stage
      // runs).
      if (s.tone_active && s.engine.needs_adapter()) {
        std::string err;
        if (o.tone_head_file.empty()) {
          const EmbeddedFile* f = find_embedded_file("models/tone_head/tone_head.bin");
          if (!f || !s.tone_head.load(f->data, f->size, err)) {
            throw std::runtime_error("the built-in tone heads (models/tone_head/tone_head.bin) " +
                                     (f ? err : std::string("are missing")));
          }
          s.head_path[2] = "[built-in] models/tone_head/tone_head.bin";
          s.head_fp[2] = memory_fingerprint(f->data, f->size);
        } else {
          const std::vector<char> file = read_file(o.tone_head_file, "tone");
          if (!s.tone_head.load(file.data(), file.size(), err))
            throw std::runtime_error("--dino tone: " + o.tone_head_file + ": " + err);
          s.head_path[2] = o.tone_head_file;
          s.head_fp[2] = memory_fingerprint(file.data(), file.size());
        }
        s.engine.set_token_spread(true);
        log_info("Tone heads %s\n", s.head_path[2].c_str());
      }
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    log_info("DINOv3 engine loaded [%s] adapter %s (memory: %.1f MB)\n", s.engine.backend_name(),
           s.engine.adapter_name().c_str(), (double)s.engine.memory_bytes() / (1024.0 * 1024.0));
  }
  // --check_renderer checks the bundle search's passes (the scene's or the print's, with the curve fit and the tone
  // plan after it): refused before anything is written where none runs.
  if (o.check_renderer && (!s.bundle_search || !s.replay_source.empty())) {
    log_error("--check_renderer needs the bundle search to run: not with given values, recorded curves or a recorded "
              "tone curve, or a single ICC profile\n");
    return 1;
  }
  // The search's renderer, made only where a search runs (search_renderer): with --check_renderer the comparison where
  // a device runs this build's kernels (else the pass-through renderer alone), else the GPU's.
  if (s.bundle_search && s.replay_source.empty()) {
    if (o.check_renderer) {
      if (negicc::gpu_device_available()) s.renderer = std::make_unique<CompareRenderer>();
    } else if (search_runs_on_gpu(o)) {
      s.renderer = std::make_unique<GpuRenderer>();
    }
  }
  return 0;
}

int finish_search_passes(const Options& o, Setup& s) {
  if (auto* g = dynamic_cast<GpuRenderer*>(s.renderer.get())) {
    if (g->binds + g->renders + g->clips > 0) {
      log_info("GPU renderer: %ld binds, %ld renders and %ld clip passes on the device\n", g->binds, g->renders,
               g->clips);
    }
    g->release_frame();
  }
  if (auto* cmp = dynamic_cast<CompareRenderer*>(s.renderer.get())) {
    log_info("Renderer comparison, GPU against CPU: %s\n", cmp->summary().c_str());
    cmp->gpu.release_frame();
    if (!cmp->within_tolerance()) {
      log_error("Renderer comparison: a GPU pass is outside the tolerance (a render's mean change in L* or a zone "
                "mean %g, a share or a histogram bin %g %%, a median %g L*), a value is not finite, or no pass ran; "
                "nothing is written\n", CompareRenderer::kCompareMeanDL, CompareRenderer::kCompareShare,
                CompareRenderer::kCompareMedianL);
      return 1;
    }
  }
  if (o.check_renderer) {
    const PassAudit& pa = pass_audit();
    log_info("Renderer check: %ld binds, %ld previews and %ld clip passes through the renderer; %ld passes of the "
             "search outside it\n", pa.binds, pa.renders, pa.clips, pa.outside);
    if (pa.outside > 0) {
      log_error("Renderer check: %ld passes of the search ran outside the renderer\n", pa.outside);
      return 1;
    }
    if (pa.binds + pa.renders + pa.clips == 0) {
      log_error("Renderer check: no pass of the search ran, so nothing was checked\n");
      return 1;
    }
    if (!dynamic_cast<CompareRenderer*>(s.renderer.get()))
      log_info("Renderer check: no CUDA device that runs this build's GPU passes, so none was compared with the GPU\n");
  }
  return 0;
}

// --target, -E, --gain_g and --gain_b on a pinned search (Setup::pinned_search): held by the search and the curve
// fit, which solve the rest.
static void set_search_pins(const Options& o, const Setup& s, ParamSearchOptions& opts) {
  if (!s.pinned_search) return;
  opts.pin_target = s.manual_target_idx;
  if (o.exposure_given) opts.pin_mask |= PIN_E;
  if (o.gain_g_given) opts.pin_mask |= PIN_G;
  if (o.gain_b_given) opts.pin_mask |= PIN_B;
  opts.pin_e = o.exposure_comp;
  opts.pin_g = o.gain_g;
  opts.pin_b = o.gain_b;
}


// What the warnings about an estimated or clipped film base offer on TriLED: a capture of this roll's unexposed film
// (--film_base), which parse_options takes with a single CFA RAW or a pixel shift capture alone (prepare refuses it on
// white light).
static const char* triled_base_remedy(const Options& o) {
  return o.input_mode == InputMode::kRaw ? "give --film_base <capture of this roll's unexposed film>"
                                         : "a capture of this roll's unexposed film (--film_base) needs the frame as "
                                           "a single CFA RAW or a pixel shift capture";
}

// What a film base capture was read with, which a later run must share to take its base from the sidecar: the flat
// field (its file's size and CRC-32; null for none) and the crosstalk matrix.
static nlohmann::ordered_json film_base_read_with(const Options& o, const Setup& s) {
  nlohmann::ordered_json j;
  if (o.flat.empty()) {
    j["flat"] = nullptr;
  } else {
    uintmax_t size = 0;
    uint32_t crc = 0;
    const bool ok = file_fingerprint(o.flat, size, crc);
    j["flat"] = ok ? nlohmann::ordered_json({{"size", size}, {"crc32", crc}}) : nlohmann::ordered_json("unreadable");
  }
  nlohmann::ordered_json xt = nlohmann::ordered_json::array();
  for (int i = 0; i < 9; ++i)
    xt.push_back(s.bundle.has_crosstalk ? s.bundle.crosstalk_matrix_d[i] : (i % 4 == 0 ? 1.0 : 0.0));
  j["crosstalk"] = xt;
  j["bundle"] = s.bundle.key;
  return j;
}

// --film_base auto: the film base a previous run of this output read from a capture (its <output>.json has
// film_base.source "capture", or "sidecar" when it took it the same way), when that run converted these input files
// (their names, sizes and CRC-32s) with this flat field and this crosstalk correction, which the base was read
// through. The scan's own base is then not read. A base read from the scan is not taken from the sidecar: reading it
// again gives it bit for bit, while the sidecar's ten digits would not. A sidecar that cannot be read is left alone.
static void load_sidecar_film_base_checked(const Options& o, Setup& s, Scan& scan, const std::string& sidecar_path,
                                           const nlohmann::ordered_json& j) {
  if (!j.is_object() || !j.contains("schema") || !j["schema"].is_string() ||
      j["schema"].get<std::string>() != "negicc.neg_process.sidecar/1") {
    return;
  }
  if (!j.contains("film_base") || !j["film_base"].is_object()) return;
  const auto& fb = j["film_base"];
  if (!fb.contains("source") || !fb["source"].is_string()) return;
  const std::string source = fb["source"].get<std::string>();
  if (source != "capture" && source != "sidecar") return;
  std::string why;
  const auto rgb = fb.contains("film_base_rgb") ? fb["film_base_rgb"] : nlohmann::ordered_json();
  if (!rgb.is_array() || rgb.size() != 3 || !rgb[0].is_number() || !rgb[1].is_number() || !rgb[2].is_number() ||
      !(rgb[0].get<double>() > 0.0) || !(rgb[1].get<double>() > 0.0) || !(rgb[2].get<double>() > 0.0)) {
    why = "it records no positive film base";
  }
  const auto inputs = j.contains("inputs") ? j["inputs"] : nlohmann::ordered_json();
  const auto files = inputs.is_object() && inputs.contains("raw_files") ? inputs["raw_files"]
                                                                         : nlohmann::ordered_json();
  const auto stats = inputs.is_object() && inputs.contains("raw_file_stats") ? inputs["raw_file_stats"]
                                                                              : nlohmann::ordered_json();
  if (why.empty() && (!files.is_array() || files.size() != o.files.size() || !stats.is_array() ||
                      stats.size() != o.files.size())) {
    why = "it converted other files";
  }
  if (why.empty()) {
    if (s.input_fingerprints.size() != o.files.size()) s.input_fingerprints.assign(o.files.size(), FileFingerprint{});
    for (size_t i = 0; why.empty() && i < o.files.size(); ++i) {
      FileFingerprint& fp = s.input_fingerprints[i];
      if (!fp.ok) fp.ok = file_fingerprint(o.files[i], fp.size, fp.crc);
      const auto& st = stats[i];
      if (!files[i].is_string() || !paths_match(files[i].get<std::string>(), o.files[i])) {
        why = "it converted other files";
      } else if (!st.is_object() || !st.contains("size") || !st["size"].is_number_unsigned() ||
                 !st.contains("crc32") || !st["crc32"].is_number_unsigned() || !fp.ok ||
                 st["size"].get<uintmax_t>() != fp.size || st["crc32"].get<uint32_t>() != fp.crc) {
        why = o.files[i] + " is not the file it converted";
      }
    }
  }
  // The base is in the film base space of the run that read it: through its flat field and crosstalk correction.
  const auto cap = fb.contains("capture") && fb["capture"].is_object() ? fb["capture"] : nlohmann::ordered_json();
  if (why.empty()) {
    const nlohmann::ordered_json now = film_base_read_with(o, s);
    const auto was = cap.is_object() && cap.contains("read_with") ? cap["read_with"] : nlohmann::ordered_json();
    if (!was.is_object()) {
      why = "it does not record the flat field and crosstalk correction the base was read with";
    } else if (!was.contains("flat") || was["flat"] != now["flat"]) {
      why = "the base was read with another flat field";
    } else if (!was.contains("crosstalk") || was["crosstalk"] != now["crosstalk"]) {
      why = "the base was read with another crosstalk correction";
    }
  }
  if (!why.empty()) {
    log_info("Not taking the film base of %s: %s; reading it from the scan\n", sidecar_path.c_str(), why.c_str());
    return;
  }
  scan.has_given_base = true;
  scan.given_base = negicc::FilmBaseDetection{};
  scan.given_base.found = true;
  scan.given_base.source = "sidecar";
  for (int c = 0; c < 3; ++c) scan.given_base.rgb[c] = rgb[c].get<double>();
  const auto dec = fb.contains("decode_rgb") ? fb["decode_rgb"] : nlohmann::ordered_json();
  if (dec.is_array() && dec.size() == 3 && dec[0].is_number() && dec[1].is_number() && dec[2].is_number()) {
    for (int c = 0; c < 3; ++c) scan.given_base.decode_rgb[c] = dec[c].get<double>();
  }
  scan.base_capture_json = cap.dump();
  log_info("Film base (a previous run's capture, from %s; --film_base scan reads the scan's own): R=%.1f G=%.1f "
           "B=%.1f\n", sidecar_path.c_str(), scan.given_base.rgb[0], scan.given_base.rgb[1], scan.given_base.rgb[2]);
}

static void load_sidecar_film_base(const Options& o, Setup& s, Scan& scan) {
  const std::string sidecar_path = o.output + ".json";
  if (!is_regular_file(sidecar_path)) return;
  std::ifstream ifs(sidecar_path);
  if (!ifs.is_open()) return;
  const nlohmann::ordered_json j = nlohmann::ordered_json::parse(ifs, nullptr, false);
  try {
    load_sidecar_film_base_checked(o, s, scan, sidecar_path, j);
  } catch (const std::exception& e) {  // a field of the wrong type: the sidecar gives no base
    scan.has_given_base = false;
    scan.base_capture_json.clear();
    log_info("Not taking the film base of %s: it cannot be read (%s); reading it from the scan\n",
             sidecar_path.c_str(), e.what());
  }
}

// ---- The search's shadow alternative: its own stages after the search, and the choice in solve -------------------

// The state the stages after the search leave for one candidate (ShadowCandidate), and putting it back.
static void capture_candidate(const Setup& s, const Scan& scan, ShadowCandidate& k) {
  k.auto_res = scan.auto_res;
  k.auto_res.shadow = ShadowAlternative{};
  k.has_curve_params = scan.has_curve_params;
  k.curve_params = scan.curve_params;
  k.has_rolloff = scan.has_rolloff;
  k.rolloff_x_hi = scan.rolloff_x_hi;
  k.rolloff_q995 = scan.rolloff_q995;
  k.curve_fit = s.curve_fit;
  k.tone_stage = s.tone_stage;
  k.tone_stage_ran = s.tone_stage_ran;
  k.tone_plan = s.tone_plan;
  k.has_tone = s.has_tone;
  k.tone_curve = s.tone_curve;
  k.input_curves = s.input_curves;
  k.input_curves_source = s.input_curves_source;
  k.curves_base_idx = s.curves_base_idx;
  k.curves_base_prof = s.curves_base_prof;
  k.curves_base_prof.rebind_pointers();
  k.has_knob_res = scan.has_knob_res;
  std::copy(scan.knob_base, scan.knob_base + 3, k.knob_base);
  k.knob_res = scan.knob_res;
  k.has_knob_curves = scan.has_knob_curves;
  k.knob_curves = scan.knob_curves;
  k.knob_base_a = scan.knob_base_a;
  k.has_plain = scan.has_plain;
  k.plain_x_hi = scan.plain_x_hi;
  k.plain_q995 = scan.plain_q995;
  k.has_plain_prof = scan.has_plain_prof;
  k.plain_prof = scan.plain_prof;
  k.plain_prof.rebind_pointers();
  k.prof_idx = scan.auto_res.target_idx;
  k.prof = s.bundle.targets[k.prof_idx].film_prof;
  k.prof.rebind_pointers();
}
static void restore_candidate(Setup& s, Scan& scan, const ShadowCandidate& k) {
  scan.auto_res = k.auto_res;
  scan.has_curve_params = k.has_curve_params;
  scan.curve_params = k.curve_params;
  scan.has_rolloff = k.has_rolloff;
  scan.rolloff_x_hi = k.rolloff_x_hi;
  scan.rolloff_q995 = k.rolloff_q995;
  s.curve_fit = k.curve_fit;
  s.tone_stage = k.tone_stage;
  s.tone_stage_ran = k.tone_stage_ran;
  s.tone_plan = k.tone_plan;
  s.has_tone = k.has_tone;
  s.tone_curve = k.tone_curve;
  s.input_curves = k.input_curves;
  s.input_curves_source = k.input_curves_source;
  s.curves_base_idx = k.curves_base_idx;
  s.curves_base_prof = k.curves_base_prof;
  s.curves_base_prof.rebind_pointers();
  scan.has_knob_res = k.has_knob_res;
  std::copy(k.knob_base, k.knob_base + 3, scan.knob_base);
  scan.knob_res = k.knob_res;
  scan.has_knob_curves = k.has_knob_curves;
  scan.knob_curves = k.knob_curves;
  scan.knob_base_a = k.knob_base_a;
  scan.has_plain = k.has_plain;
  scan.plain_x_hi = k.plain_x_hi;
  scan.plain_q995 = k.plain_q995;
  scan.has_plain_prof = k.has_plain_prof;
  scan.plain_prof = k.plain_prof;
  scan.plain_prof.rebind_pointers();
  s.bundle.targets[k.prof_idx].film_prof = k.prof;
  s.bundle.targets[k.prof_idx].film_prof.rebind_pointers();
}
// Switches the pipeline's state from candidate from to candidate to: from's bracket back to its profile before the
// stages (the curve fit composes its curves into the bracket's tables), then to's state.
static void switch_candidate(Setup& s, Scan& scan, const ShadowCandidate& from, const ShadowCandidate& to) {
  s.bundle.targets[from.prof_idx].film_prof = from.prof0;
  s.bundle.targets[from.prof_idx].film_prof.rebind_pointers();
  restore_candidate(s, scan, to);
}

int decode(const Options& o, Setup& s, Scan& scan) {
  ProfileBundle& bundle = s.bundle;
  std::copy(o.roi, o.roi + 4, scan.roi);
  scan.has_roi = o.has_roi;
  scan.rot_cw = o.rot_cw;
  scan.hflip = o.hflip;
  scan.vflip = o.vflip;

  // The camera, from the first input's metadata (an RGB TIFF or JPEG states none that LibRaw reads), and its inset
  // crop. A camera_raw_rgb bundle's crosstalk correction holds for the camera models the bundle lists: a scan that
  // states a model the list does not hold converts with the correction off (the identity), here before anything reads
  // through it (the film base capture, the film base, the search, the conversion), except the rows -r, -g and -b give.
  // A scan that states no model, or a bundle without the list (from before it), is not compared: the bundle's
  // correction applies. Three captures, one per light, take the identity already.
  if (o.input_mode != InputMode::kRgbFile) {
    LibRaw probe;
    if (open_raw_file(&probe, o.files[0]) == LIBRAW_SUCCESS) {
      scan.camera_make = probe.imgdata.idata.make;
      scan.camera_model = probe.imgdata.idata.model;
      scan.has_inset = inset_crop(probe, scan.inset);
    }
  }
  if (s.is_bundle && !s.white_light && !scan.camera_model.empty() && bundle.has_camera_models) {
    scan.camera_compared = true;
    scan.camera_listed = std::find(bundle.camera_models.begin(), bundle.camera_models.end(), scan.camera_model) !=
                         bundle.camera_models.end();
    if (!scan.camera_listed && bundle.has_crosstalk && o.input_mode != InputMode::kThreeShot) {
      s.crosstalk_off = "the RAW's camera " + scan.camera_model + " is not one the bundle lists";
      std::vector<float>* rows[3] = {&s.r_coeff, &s.g_coeff, &s.b_coeff};
      for (int r = 0; r < 3; ++r) {
        if (o.row_given[r]) continue;
        for (int c = 0; c < 3; ++c) {
          bundle.crosstalk_matrix[r * 3 + c] = r == c ? 1.0f : 0.0f;
          bundle.crosstalk_matrix_d[r * 3 + c] = r == c ? 1.0 : 0.0;
        }
        rows[r]->assign(bundle.crosstalk_matrix + r * 3, bundle.crosstalk_matrix + r * 3 + 3);
        s.crosstalk_row_source[r] = "identity";
      }
      log_warn("the crosstalk correction of the profile bundle %s is off: %s%s; -r, -g and -b give rows measured on "
               "this camera\n", bundle.key.c_str(), s.crosstalk_off.c_str(),
               o.row_given[0] || o.row_given[1] || o.row_given[2] ? " (the rows given are applied)" : "");
    }
  }
  // The matrix as it applies, now that the camera check has decided the rows the bundle gives: with rows given it is
  // tested as a bundle's own matrix is when it loads, before anything reads through it.
  if (s.is_bundle && s.has_crosstalk && (o.row_given[0] || o.row_given[1] || o.row_given[2]) &&
      !crosstalk_matrix_invertible(bundle.crosstalk_matrix)) {
    log_error("%s", kRowsNotInvertible);
    return 1;
  }
  // Rows given on a camera_raw_rgb bundle without a matrix of its own are its correction from here on, as a bundle's
  // matrix is: the film base capture, the film base found in the frame, what the base was read with and the search
  // read through them, as the conversion does. (Set after the camera check, which turns off only a bundle's own.)
  if (s.is_bundle && s.has_crosstalk) bundle.has_crosstalk = true;

  scan.exposure_file = o.files[0];
  scan.has_exif_exp = read_exif_exposure_d(o.files[0], scan.exif_shutter_d, scan.exif_iso_d);

  // The film base given in place of the scan's own (Scan::has_given_base): --film_base's capture, read here, before
  // the scan, so a capture that cannot be used fails first; or, with --film_base auto, a previous run's.
  if (!o.film_base_file.empty()) {
    double scan_shutter_s = scan.exif_shutter_d, scan_iso = scan.exif_iso_d;
    if (!scan.has_exif_exp || !(scan_shutter_s > 0.0) || !(scan_iso > 0.0)) {
      LibRaw probe;
      if (open_raw_file(&probe, o.files[0]) == LIBRAW_SUCCESS) scan.scan_exposure(probe, scan_shutter_s, scan_iso);
    }
    if (!(scan_shutter_s > 0.0) || !(scan_iso > 0.0)) {
      log_error("%s records no shutter speed or ISO, which --film_base needs to scale the capture to the scan's "
                "exposure\n", o.files[0].c_str());
      return 1;
    }
    const double* xt = s.bundle.has_crosstalk ? s.bundle.crosstalk_matrix_d : nullptr;
    std::string err;
    log_info("Loading film base capture %s\n", o.film_base_file.c_str());
    const double* expected = s.bundle.has_film_base ? s.bundle.film_base_avg_d : nullptr;
    if (!read_film_base_capture(o.film_base_file, o.film_base_roi_given ? o.film_base_roi : nullptr, s.flat_field, xt,
                                expected, scan_shutter_s, scan_iso, scan.base_capture, err)) {
      log_error("--film_base: %s\n", err.c_str());
      return 1;
    }
    const FilmBaseCapture& cap = scan.base_capture;
    scan.has_base_capture = true;
    scan.has_given_base = true;
    scan.given_base = negicc::FilmBaseDetection{};
    scan.given_base.found = true;
    scan.given_base.source = "capture";
    for (int c = 0; c < 3; ++c) {
      scan.given_base.rgb[c] = cap.rgb[c];
      scan.given_base.decode_rgb[c] = cap.decode_rgb[c];
    }
    nlohmann::ordered_json cj;
    cj["path"] = o.film_base_file;
    cj["roi"] = {cap.roi[0], cap.roi[1], cap.roi[2], cap.roi[3]};
    cj["roi_source"] = cap.roi_source;
    cj["tiles_kept"] = cap.roi_source == "given" || cap.roi_source == "statistics"
                           ? nlohmann::ordered_json(nullptr)
                           : nlohmann::ordered_json(cap.tiles_kept);
    cj["tiles_total"] = cap.roi_source == "given" || cap.roi_source == "statistics"
                            ? nlohmann::ordered_json(nullptr)
                            : nlohmann::ordered_json(cap.tiles_total);
    cj["spread_density"] = cap.roi_source == "statistics" ? nlohmann::ordered_json(nullptr)
                                                           : nlohmann::ordered_json(cap.spread_density);
    cj["decode_rgb"] = {cap.decode_rgb[0], cap.decode_rgb[1], cap.decode_rgb[2]};
    cj["shutter_s"] = cap.shutter_s;
    cj["iso"] = cap.iso;
    cj["exposure_scale"] = cap.exposure_scale;
    {
      uintmax_t size = 0;
      uint32_t crc = 0;
      cj["file"] = file_fingerprint(o.film_base_file, size, crc)
                       ? nlohmann::ordered_json({{"size", size}, {"crc32", crc}})
                       : nlohmann::ordered_json(nullptr);
    }
    cj["read_with"] = film_base_read_with(o, s);
    // The path need not be UTF-8: written as the sidecar writes it (utf8_replaced).
    scan.base_capture_json = cj.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
    // A base far from the bundle's own, at the scan's exposure, is probably not a capture of this film's base.
    if (s.bundle.has_film_base) {
      const double base_t =
          s.bundle.film_base_shutter.empty() ? 0.125 : shutter_to_seconds_d(s.bundle.film_base_shutter);
      const double ratio = (scan_shutter_s * scan_iso) / (base_t * (double)s.bundle.film_base_iso);
      double worst = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double want = s.bundle.film_base_avg_d[c] * ratio;
        if (want > 0.0) worst = std::max(worst, std::fabs(std::log2(cap.rgb[c] / want)));
      }
      if (worst > 2.0) {
        log_warn("the film base read from %s is %.1f stops from the profile bundle's own at this exposure: check that "
                 "it is a capture of this film's unexposed base under the same light\n",
                 o.film_base_file.c_str(), worst);
      }
    }
    log_info("Film base (capture, %s, region %d,%d,%d,%d%s): R=%.1f G=%.1f B=%.1f (exposure scale %.4f)\n",
             cap.roi_source.c_str(), cap.roi[0], cap.roi[1], cap.roi[2], cap.roi[3],
             cap.tiles_total ? (", " + std::to_string(cap.tiles_kept) + " of " + std::to_string(cap.tiles_total) +
                                " tiles").c_str()
                             : "",
             cap.rgb[0], cap.rgb[1], cap.rgb[2], cap.exposure_scale);
  } else if (s.is_bundle && !s.white_light && !o.film_base_scan) {
    load_sidecar_film_base(o, s, scan);
  }

  // The bracket of given values (the search's is its own): on a bundle without the search, --target, with -E and
  // both gains (prepare).
  auto given_target_idx = [&]() { return s.is_bundle ? s.manual_target_idx : -1; };
  // A frame that is the whole image (--roi auto that fell back, or a given --roi that is the whole image, as --roi
  // auto's fallback records it, so that its replay fits the same): the grid may hold rebate, bare light or holder,
  // which the curve fit and its re-anchoring leave out (ParamSearchOptions::fallback_crop).
  auto set_fit_frame = [&](ParamSearchOptions& p) {
    p.fallback_crop = o.roi_auto ? scan.frame_det.conf < kLowFrameConfidence : scan.has_roi && scan.frame_is_whole;
    p.film_base_is_rebate = scan.film_base.is_rebate();
  };
  // The bracket's final E, g, b: Auto's (or the curve fit's) or the given values, moved by --brightness, --tint and
  // --warmth when given (grey-point knobs, solved on the bracket's profile, combined when curves apply); then the
  // highlight roll-off measured at them.
  auto solve_knobs_and_rolloff = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso) {
    const int target_idx = scan.has_auto_res ? scan.auto_res.target_idx : given_target_idx();
    const float base_e = scan.has_auto_res ? scan.params().e : o.exposure_comp;
    const float base_g = scan.has_auto_res ? scan.params().g : o.gain_g;
    const float base_b = scan.has_auto_res ? scan.params().b : o.gain_b;
    float e = base_e, g = base_g, b = base_b;
    // Why solve_grey_knobs failed on bracket target_idx's profile (combined: with the fitted input curves).
    auto knob_failure = [&](const KnobSolveResult& r, bool combined) {
      const char* name = bundle.targets[target_idx].name.c_str();
      if (!(base_e > 0.0f) || !(base_g > 0.0f) || !(base_b > 0.0f)) {
        log_error("--brightness, --tint and --warmth move E, g and b from the bracket's, and %s's are not all positive "
                  "(E %g g %g b %g)\n", name, base_e, base_g, base_b);
        return;
      }
      log_error("--brightness, --tint and --warmth move the grey that %s renders as neutral L* 50, and none was found "
                "on its profile%s: the grey-point solve ended %.2f dE76 from L* 50 a* 0 b* 0. Convert without the "
                "knobs%s\n", name, combined ? " combined with the fitted input curves" : "", r.grey_miss,
                combined ? ", or with --fit_curves off" : "");
    };
    // On a print the knobs act on the print's own A, g and b, solved on its profile; E held.
    // A density output's knobs move its print CLFs (density_print_searches), not its suggested balance.
    const bool knobs_here = o.knobs_given && !s.density;
    if (knobs_here && !s.print_profs.empty() && std::isfinite(s.print_a) && target_idx >= 0 &&
        target_idx < (int)s.print_profs.size()) {
      scan.knob_base[0] = base_e;
      scan.knob_base[1] = base_g;
      scan.knob_base[2] = base_b;
      scan.knob_base_a = s.print_a;
      auto print_at = [&](double a, double g, double b, ParsedFilmProfile& prof) {
        prof = s.print_profs[target_idx];
        prof.rebind_pointers();
        const double gains[3] = {1.0, g, b};
        s.print_cont.set_curves_at(a, prof, gains);
      };
      PrintKnobResult pk;
      if (!solve_print_knobs(print_at, s.print_a, base_g, base_b, s.print_cont.a_lo(), s.print_cont.a_hi(),
                             o.brightness, o.tint, o.warmth, pk)) {
        log_error("Failed to solve the knobs on the print\n");
        return false;
      }
      s.print_a = pk.a;
      scan.knob_res.e = base_e;
      scan.knob_res.g = pk.g;
      scan.knob_res.b = pk.b;
      scan.knob_res.limited = pk.limited;
      for (int i = 0; i < 3; ++i) scan.knob_res.grey_lab[i] = pk.grey_lab[i];
      scan.has_knob_res = true;
      scan.has_plain = s.tone_active && !s.has_cached_sidecar;  // the print without them: knob_base_a, base g, b
      log_info("Knobs on the print: A %.4f -> %.4f, g %.4f -> %.4f, b %.4f -> %.4f (E %.4f held); the grey L* %.3f "
               "a* %.3f b* %.3f%s\n", scan.knob_base_a, pk.a, base_g, pk.g, base_b, pk.b, base_e, pk.grey_lab[0],
               pk.grey_lab[1], pk.grey_lab[2], pk.limited ? " (limited)" : "");
      return true;
    }
    if (knobs_here && target_idx >= 0 && target_idx < (int)bundle.targets.size()) {
      // Where the tone stage runs and no record is replayed, the conversion without the knobs (Scan::has_plain),
      // which its clip check is made on: the roll-off as a run without them measures it, and the bracket's profile
      // before the knobs re-anchor its curves in place.
      if (s.tone_active && !s.has_cached_sidecar && s.print_id.empty()) {
        if (scan.has_curve_params && s.curve_fit.rolloff_measured) {
          scan.plain_x_hi = s.curve_fit.x_hi;
          scan.plain_q995 = s.curve_fit.x_hi_q995;
        } else {
          scan.plain_x_hi = measure_rolloff_x_hi(bundle, target_idx, frame, scan_shutter_s, scan_iso, base_e, base_g,
                                                 base_b, ParamSearchOptions{}.crop_ratio, &scan.plain_q995,
                                                 scan.film_base_rgb(), search_renderer(o, s));
        }
        scan.has_plain_prof = target_idx == s.curves_base_idx && s.input_curves.active();
        if (scan.has_plain_prof) {
          scan.plain_prof = bundle.targets[target_idx].film_prof;
          scan.plain_prof.rebind_pointers();
        }
        scan.has_plain = true;
      }
      scan.knob_base[0] = base_e;
      scan.knob_base[1] = base_g;
      scan.knob_base[2] = base_b;
      if (!solve_grey_knobs(bundle.targets[target_idx].film_prof.data, base_e, base_g, base_b, o.brightness, o.tint,
                            o.warmth, scan.knob_res)) {
        // The bracket the curves apply to renders through its combined profile here.
        knob_failure(scan.knob_res, target_idx == s.curves_base_idx && s.input_curves.active());
        return false;
      }
      scan.has_knob_res = true;
      e = scan.knob_res.e;
      g = scan.knob_res.g;
      b = scan.knob_res.b;
      // The knobs moved E, g and b off the anchor of the curves the bracket renders through (fitted, recorded or a
      // previous run's): re-anchor them there and compose the bracket's input tables again from its un-curved ones, so
      // the curves stay on the film densities they were fitted to.
      //
      // The knobs were solved on the profile at the base's anchor, and the re-anchored curves change it: the grey then
      // missed its target (on two white-LED frames up to 0.08 stop and 5 a* for a 0.5 stop brightness). So, until the
      // gains hold still (each within kKnobTol of the previous pass, at most kMaxKnobPasses): the knobs solved on the
      // re-anchored profile for the same grey (the one the base's profile renders as neutral L* 50), and the curves
      // re-anchored at that result. Each pass cuts the miss about fourfold.
      const bool curves_here = target_idx == s.curves_base_idx && s.input_curves.active();
      if (curves_here) {
        constexpr int kMaxKnobPasses = 8;
        constexpr float kKnobTol = 1e-4f;
        ParsedFilmProfile& prof = bundle.targets[target_idx].film_prof;
        ParsedFilmProfile grey_prof = prof;  // at the base's anchor: its neutral L* 50 is the grey
        grey_prof.rebind_pointers();
        ParamSearchOptions ro;
        ro.film_base_rgb_d = scan.film_base_rgb();
        ro.film_base_from_scan = ro.film_base_rgb_d != nullptr;
        set_fit_frame(ro);
        const float from[3] = {base_e, base_g, base_b};
        float scale[2] = {1.0f, 1.0f};
        auto reanchor_at = [&](const float to[3]) {
          if (!reanchor_input_curves(bundle, s.engine, target_idx, frame, scan_shutter_s, scan_iso, ro,
                                     s.input_curves, from, to, negicc::CurveFitOptions().max_dev_stops,
                                     scan.knob_curves, scale)) {
            return false;
          }
          apply_input_curves(prof, s.curves_base_prof, scan.knob_curves);
          return true;
        };
        const float to1[3] = {e, g, b};
        if (!reanchor_at(to1)) return false;
        float lab1[3];
        knob_grey_lab(grey_prof.data, prof.data, from, to1, lab1);
        KnobSolveResult k2 = scan.knob_res;
        int passes = 0;
        for (bool moved = true; moved && passes < kMaxKnobPasses; ++passes) {
          const float prev[3] = {k2.e, k2.g, k2.b};
          if (!solve_grey_knobs(prof.data, base_e, base_g, base_b, o.brightness, o.tint, o.warmth, k2,
                                &grey_prof.data)) {
            knob_failure(k2, true);
            return false;
          }
          const float to2[3] = {k2.e, k2.g, k2.b};
          if (!reanchor_at(to2)) return false;
          moved = false;
          for (int i = 0; i < 3; ++i) moved = moved || std::fabs(to2[i] - prev[i]) > kKnobTol * prev[i];
        }
        const float to2[3] = {k2.e, k2.g, k2.b};
        knob_grey_lab(grey_prof.data, prof.data, from, to2, k2.grey_lab);
        scan.knob_res = k2;
        e = k2.e;
        g = k2.g;
        b = k2.b;
        scan.has_knob_curves = true;
        log_info("Input tables re-anchored at the knobs' E %.4f g %.4f b %.4f (refG %.6g refB %.6g%s); the grey "
                 "L* %.3f a* %.3f b* %.3f, %+.4f stop from the target after %d more solves (L* %.3f a* %.3f b* %.3f, "
                 "%+.4f stop, before them)\n",
                 e, g, b, scan.knob_curves.ref[0], scan.knob_curves.ref[1],
                 scale[0] < 1.0f || scale[1] < 1.0f ? ", scaled to keep the shape" : "", k2.grey_lab[0],
                 k2.grey_lab[1], k2.grey_lab[2], knob_grey_miss_stops(o.brightness, k2.grey_lab[0]), passes, lab1[0],
                 lab1[1], lab1[2], knob_grey_miss_stops(o.brightness, lab1[0]));
      }
    }
    // A print keeps no highlight shoulder (solve: the paper rolls the highlights off), so no roll-off is measured.
    if (!s.print_id.empty()) return true;
    if (!scan.has_knob_res && scan.has_curve_params && s.curve_fit.rolloff_measured) {
      // The curve fit measured the roll-off on this frame, at these E, g and b, on this combined profile, as below
      // (its last re-meter pass): the same numbers, not measured a third time.
      scan.rolloff_x_hi = s.curve_fit.x_hi;
      scan.rolloff_q995 = s.curve_fit.x_hi_q995;
    } else {
      scan.rolloff_x_hi = measure_rolloff_x_hi(bundle, target_idx, frame, scan_shutter_s, scan_iso, e, g, b,
                                               ParamSearchOptions{}.crop_ratio, &scan.rolloff_q995,
                                               scan.film_base_rgb(), search_renderer(o, s));
    }
    scan.has_rolloff = true;
    return true;
  };
  // A density run with --print, on either light: each material's own print search, so that its CLF prints the frame
  // as the picture conversion would (DensityRecord::print_search). The scene's search before it gives the file's
  // suggested balance: on TriLED as without --print; on white light the search's own, where without --print it is the
  // curve fit's (with --print no fit runs).
  auto density_print_searches = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                                    const ParamSearchOptions& opts) -> bool {
    // In density_prints' order, which writes the CLFs: the built-in materials asked for (or all), then the files in
    // command-line order; a material id met twice keeps its first search, as its first CLF.
    const bool all = std::find(o.prints.begin(), o.prints.end(), "all") != o.prints.end();
    std::vector<std::string> ids;
    for (const PrintMaterial& m : builtin_print_materials(&bundle))
      if (all || std::find(o.prints.begin(), o.prints.end(), m.id) != o.prints.end()) ids.push_back(m.id);
    for (const std::string& id : o.prints)
      if (is_print_file(id)) ids.push_back(id);
    // A bundle without the film's grey balance curve has no print search: its CLFs take the scene search's bracket
    // and slopes (frame_print). Any other failure is the material's, and fails the run as on a picture output.
    if (!bundle.grey_balance.present) {
      log_warn("No print search for the CLFs: the profile bundle has no grey_balance_curves; they take the scene "
               "search's bracket\n");
      return true;
    }
    // Each material's print profile and continuum, then the searches of all of them (search_print_continuous, which
    // reads the frame once for them), then each one's record. At fixed addresses: the searches point at them.
    struct Material {
      BracketPrints prints;
      ParsedFilmProfile prof;
      ContinuousPrint in;
      ParamSearchResult res;
      ContinuousPrintResult cr;
    };
    std::vector<std::unique_ptr<Material>> materials;
    for (const std::string& id : ids) {
      auto m = std::make_unique<Material>();
      BracketPrints& prints = m->prints;
      std::string why;
      if (!bracket_print_profiles(bundle, id, prints, why)) {
        log_error("--print %s: %s\n", id.c_str(), why.c_str());
        return false;
      }
      if (s.density_print_search.count(prints.id) ||
          std::any_of(materials.begin(), materials.end(),
                      [&](const std::unique_ptr<Material>& o) { return o->prints.id == prints.id; })) {
        continue;
      }
      std::vector<uint8_t>& icc = prints.icc[prints.icc.size() / 2];
      cmsHPROFILE h = cmsOpenProfileFromMem(icc.data(), (cmsUInt32Number)icc.size());
      ParsedFilmProfile& prof = m->prof;
      const bool ok = h && parse_film_profile_pipeline(h, prof);
      if (h) cmsCloseProfile(h);
      if (!ok) {
        log_error("--print %s: its print ICC cannot be read\n", id.c_str());
        return false;
      }
      prof.rebind_pointers();
      to_paper_white(prof);
      const PrintContinuum& pc = prints.continuum;
      m->in.trc_at = [&pc](double a, const double gains[3], std::vector<float> trc[3]) { pc.trc_at(a, gains, trc); };
      m->in.a_lo = pc.a_lo();
      m->in.a_hi = pc.a_hi();
      m->in.a_mid = pc.a_mid;
      materials.push_back(std::move(m));
    }
    std::vector<PrintMaterialSearch> searches;
    for (const std::unique_ptr<Material>& m : materials) searches.push_back({&m->prof, &m->in, &m->res, &m->cr});
    if (!search_print_continuous(bundle, searches, s.engine, frame, scan_shutter_s, scan_iso, opts)) return false;
    for (const std::unique_ptr<Material>& m : materials) {
      const BracketPrints& prints = m->prints;
      const ParsedFilmProfile& prof = m->prof;
      const PrintContinuum& pc = prints.continuum;
      const ParamSearchResult& res = m->res;
      const ContinuousPrintResult& cr = m->cr;
      DensityRecord::PrintSearch ps;
      ps.a = cr.a;
      ps.e = res.params.e;
      ps.g = res.params.g;
      ps.b = res.params.b;
      ps.bracket = res.target_name;
      ps.converged = cr.converged;
      ps.steps = cr.steps;
      for (int c = 0; c < 3; ++c) ps.last_move[c] = cr.last_move[c];
      ps.above_knee = cr.above_knee;
      for (int c = 0; c < 3; ++c) ps.past_knee[c] = cr.past_knee[c], ps.past_cut[c] = cr.past_cut[c];
      ps.stopped_by = cr.stopped_by;
      ps.a0 = cr.a0;
      ps.target_l = cr.target_mean_l;
      for (int c = 0; c < 3; ++c) {
        ps.above_white[c] = cr.above_white[c];
        ps.at_white[c] = cr.at_white[c];
      }
      if (o.knobs_given) {
        // The knobs on this print as on the picture (solve_knobs_and_rolloff): A, g and b moved on its profile, E held.
        auto print_at = [&](double a, double g, double b, ParsedFilmProfile& p) {
          p = prof;
          p.rebind_pointers();
          const double gains[3] = {1.0, g, b};
          pc.set_curves_at(a, p, gains);
        };
        PrintKnobResult pk;
        if (!solve_print_knobs(print_at, cr.a, res.params.g, res.params.b, pc.a_lo(), pc.a_hi(), o.brightness, o.tint,
                               o.warmth, pk)) {
          log_error("Failed to solve the knobs on the print %s\n", prints.id.c_str());
          return false;
        }
        ps.knobs = true;
        ps.knobs_limited = pk.limited;
        ps.knob_base_a = ps.a;
        ps.knob_base_g = ps.g;
        ps.knob_base_b = ps.b;
        ps.knob_given[0] = o.brightness;
        ps.knob_given[1] = o.tint;
        ps.knob_given[2] = o.warmth;
        for (int c = 0; c < 3; ++c) ps.knob_grey_lab[c] = pk.grey_lab[c];
        ps.a = pk.a;
        ps.g = pk.g;
        ps.b = pk.b;
        // The bracket is the one nearest the moved A, as the search names the one nearest its own (the CLF is computed
        // at A from the curve; the bracket only names it).
        ps.bracket = bundle.targets[nearest_bracket(pc.a_mid, ps.a)].name;
        log_info("Knobs on %s's CLF: A %.4f -> %.4f, g %.4f -> %.4f, b %.4f -> %.4f; the grey L* %.3f a* %.3f "
                 "b* %.3f%s\n",
                 prints.id.c_str(), ps.knob_base_a, ps.a, ps.knob_base_g, ps.g, ps.knob_base_b, ps.b, pk.grey_lab[0],
                 pk.grey_lab[1], pk.grey_lab[2], pk.limited ? " (limited)" : "");
      }
      s.density_print_search[prints.id] = ps;
      log_info("Print for %s's CLF: A %.4f (nearest %s), g %.4f, b %.4f%s\n", prints.id.c_str(), ps.a,
               ps.bracket.c_str(), ps.g, ps.b, ps.knobs ? ", the search's moved by the knobs" : "");
    }
    return true;
  };
  // The bundle search: on a print (--print with a picture output) the print's own search, the scene's otherwise.
  auto bundle_search = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                           const ParamSearchOptions& opts) -> bool {
    if (s.print_profs.empty()) {
      if (!search_conversion_params(bundle, s.engine, frame, scan_shutter_s, scan_iso, opts, scan.auto_res))
        return false;
      if (s.density && !o.prints.empty() && !scan.auto_res.probed &&
          !density_print_searches(frame, scan_shutter_s, scan_iso, opts)) {
        return false;
      }
      return true;
    }
    const PrintMaterialSearch print_search{&s.print_profs[s.print_profs.size() / 2], &s.print_in, &scan.auto_res,
                                           &s.print_res};
    if (!search_print_continuous(bundle, {print_search}, s.engine, frame, scan_shutter_s, scan_iso, opts)) {
      return false;
    }
    s.print_a = s.print_res.a;
    s.print_searched = true;
    {
      // The print's level is relative to the film base (E where the base reads n0): an estimated base moves it and puts
      // the frame's thin film past the knee or the cut. Said as the density output says it, with what it does here.
      // Also whatever the base's source: a rebate read on clipped film (a long scan) is a measured base in name only,
      // and film thinner than the base passes the knee at E. kPrintCutWarnShare: the share of the crop past the cut in
      // a channel from which the print says so, 1 %. It lies between the two cases: a print with a captured base, or on
      // a base held no darker than the thinnest film, stays below it (the gains act after the knee), while a print
      // crushed by an estimated base taken as it is has tens of percent of the crop past the knee.
      constexpr double kPrintCutWarnShare = 0.01;
      int c = 0;
      for (int k = 1; k < 3; ++k) {
        if (s.print_res.past_cut[k] > s.print_res.past_cut[c] ||
            (s.print_res.past_cut[k] == s.print_res.past_cut[c] &&
             s.print_res.past_knee[k] > s.print_res.past_knee[c])) {
          c = k;
        }
      }
      const char* name = c == 0 ? "red" : c == 1 ? "green" : "blue";
      const char* remedy = s.white_light ? "crop so that the frame shows some rebate (--roi)" : triled_base_remedy(o);
      const bool bundle_base = !scan.film_base.found || scan.film_base.source == "bundle";
      const bool estimate = bundle_base || scan.film_base.source == "statistics";
      char raised[160] = "";
      if (scan.print_floor_applied && scan.print_floor_k > 1.0f)
        std::snprintf(raised, sizeof raised, ", raised %.3fx to be no darker than the picture's thinnest film",
                      scan.print_floor_k);
      if (estimate) {
        // With no --roi the shares are of the search's crop, the RAW's inset crop or the whole scan (said above).
        log_warn("the print's level is relative to the film base, and this one is %s%s: %.2f %% of the %s is past the "
                 "knee and %.2f %% past the cut (no detail kept) in %s at the print's settings; %s\n",
                 bundle_base ? "the profile bundle's own, not this roll's"
                             : "estimated from the thinnest film on this frame, which shows no rebate",
                 raised, 100.0 * s.print_res.past_knee[c],
                 scan.has_roi           ? "crop"
                 : scan.frame_is_whole ? "search's crop (the whole scan)"
                                       : "search's crop (the RAW's inset crop)",
                 100.0 * s.print_res.past_cut[c], name, remedy);
      } else {
        // The base was read (a rebate, or a capture on TriLED), so a crop does not help: the cause these two name is a
        // scan long enough to clip unexposed film. On white light the remedy is a shorter scan; on TriLED a capture
        // (triled_base_remedy).
        const char* read_remedy = s.white_light ? "scan with a shorter exposure, so that unexposed film does not clip"
                                                : triled_base_remedy(o);
        if (raised[0]) {   // a rebate read and raised: a rebate on clipped film reads darker than the film
          log_warn("the print's film base is the scan's %s%s: the film reads thinner than it in places, so it is not "
                   "the base (a long scan clips unexposed film); %s\n",
                   scan.film_base.source.c_str(), raised, read_remedy);
        } else if (scan.has_roi && s.print_res.past_cut[c] >= kPrintCutWarnShare) {
          log_warn("%.2f %% of the crop is past the cut (no detail kept) in %s at the print's settings, with the film "
                   "base from %s: a rebate read on a scan long enough to clip unexposed film is not the base; if this "
                   "one is not, %s\n",
                   100.0 * s.print_res.past_cut[c], name, scan.film_base.source.c_str(), read_remedy);
        }
      }
    }
    log_info("Print search: A0 %.4f -> A %.4f (nearest %s), g %.4f, b %.4f, %d steps%s; lightness target L* %.2f; "
             "%.2f %% of the crop above the knee at E = 1\n", s.print_res.a0, s.print_res.a,
             scan.auto_res.target_name.c_str(), scan.auto_res.params.g, scan.auto_res.params.b, s.print_res.steps,
             s.print_res.converged ? "" : ", not converged", s.print_res.target_mean_l, 100.0 * s.print_res.above_knee);
    return true;
  };
  // The previous run's Auto result (bracket and its final E, g, b; with fitted curves, the re-metered E and the fit's
  // g, b on the combined profile the curves were replayed into) in place of the search.
  auto restore_cached_auto = [&]() {
    if (!s.print_profs.empty() && std::isfinite(s.cached_sidecar.print_a))
      s.print_a = s.cached_sidecar.print_a;
    scan.auto_res.target_idx = s.cached_sidecar.target_idx;
    scan.auto_res.target_name = s.cached_sidecar.target_name;
    scan.auto_res.params = {s.cached_sidecar.base_e, s.cached_sidecar.base_g, s.cached_sidecar.base_b, 1.0f};
    scan.has_auto_res = true;
  };
  // A print on a base that is not a capture: the base held no darker than the picture's thinnest film
  // (print_floor_factor), on the frame the search crops. The raised base replaces the found one for the whole
  // conversion (the print's search, given values, the density file, its print searches and CLFs), so a replay
  // derives the same; a capture (given, or a capture's values kept from an earlier run) is taken as it is, and so is
  // any base without --print. Called once per conversion, with the frame, before the search.
  //
  // The floor reads the picture's frame: the search's frame when a --roi is given (or --roi auto's), else the frame
  // detector's crop of the half-size image |half| when the detection ran and is at or above kBaseMinFrameConf (the
  // level the film base takes the detector's patch at), else the search's frame: the RAW's inset crop, or the whole
  // scan. Without a frame the scan's bare light
  // (perforations, the holder's gaps, the strip's edges) lies in the floor's crop and reads as the thinnest film, which
  // can raise the factor by up to 2.5 times.
  // Light inside the frame is not masked; the record's factor and frame show what was read. The print's search itself
  // reads the search frame (without --roi, the RAW's inset crop or the whole scan). A caller that gains a print
  // (negicc-station) chooses the frame here, as this does.
  // |det| the detection the base read (its boxes in the pixels of a sensor_w x sensor_h mosaic), mapped onto |half| by
  // the two scales when they agree within 2 % (the base patch's own test, film_base_detector.cc: a decode turned
  // against the sensor does not map). A --roi auto that fell back to the whole image is the whole scan. frame_conf is
  // written only where the detection decided the frame.
  auto apply_print_floor = [&](const negicc::RgbView& frame, const negicc::RgbView& half, const FrameDetection* det,
                               int sensor_w, int sensor_h, double scan_shutter_s, double scan_iso) {
    const bool print = !s.print_profs.empty() || (s.density && !o.prints.empty());
    // A print without --roi: its search reads the RAW's inset crop, or the whole scan (whatever the base, a capture
    // included). Said once.
    if (print && s.is_bundle && !scan.has_roi && !scan.print_floor_applied)
      log_warn("no --roi: the print's search reads %s; give --roi or --roi auto\n",
               scan.frame_is_whole ? "the whole scan" : "the RAW's inset crop");
    // white light too (crosstalk: identity)
    if (!s.is_bundle || !print || scan.has_given_base || scan.print_floor_applied)
      return;
    negicc::RgbView floor_frame = frame;
    scan.print_floor_frame_conf = -1.0f;
    if (scan.has_roi) {
      const bool fell_back = o.roi_auto && scan.frame_det.conf < kLowFrameConfidence;
      scan.print_floor_frame = fell_back ? "whole_scan" : "roi";
      if (fell_back) scan.print_floor_frame_conf = scan.frame_det.conf;
    } else {
      scan.print_floor_frame = scan.frame_is_whole ? "whole_scan" : "inset_crop";
      const double sx = sensor_w > 0 ? half.width / (double)sensor_w : 0.0,
                   sy = sensor_h > 0 ? half.height / (double)sensor_h : 0.0;
      const bool aligned = det && sx > 0.0 && sy > 0.0 && std::fabs(sx - sy) <= 0.02 * std::max(sx, sy);
      if (aligned) scan.print_floor_frame_conf = det->conf;
      if (aligned && det->ok && det->conf >= negicc::kBaseMinFrameConf) {
        const int x0 = (int)(det->crop[0] * sx), y0 = (int)(det->crop[1] * sy);
        int b[4] = {x0, y0, std::max(1, (int)((det->crop[0] + det->crop[2]) * sx) - x0),
                    std::max(1, (int)((det->crop[1] + det->crop[3]) * sy) - y0)};
        clamp_rect(b, half.width, half.height);
        floor_frame = half.crop_and_orient(b[0], b[1], b[2], b[3], scan.rot_cw, scan.hflip, scan.vflip);
        scan.print_floor_frame = "detected";
      }
    }
    double fb[3], raised_rgb[3];
    const std::string src = scan.film_base.found ? scan.film_base.source : "bundle";
    const float k = raise_base_to_print_floor(s.bundle, floor_frame, ParamSearchOptions{}.crop_ratio,
                                              scan.film_base.found ? scan.film_base.rgb : nullptr, scan_shutter_s,
                                              scan_iso, fb, raised_rgb);
    scan.print_floor_applied = true;
    scan.print_floor_found_source = src;
    scan.print_floor_k = k;
    for (int c = 0; c < 3; ++c) scan.print_floor_found_rgb[c] = fb[c];
    if (k > 1.0f) {
      if (!scan.film_base.found) {
        scan.film_base = negicc::FilmBaseDetection{};
        scan.film_base.found = true;
        scan.film_base.source = "bundle";
      }
      for (int c = 0; c < 3; ++c) scan.film_base.rgb[c] = raised_rgb[c];
      log_info("Film base for the print: %s, raised %.4fx to be no darker than the picture's thinnest film "
               "(R=%.1f G=%.1f B=%.1f)\n", src.c_str(), k, scan.film_base.rgb[0], scan.film_base.rgb[1],
               scan.film_base.rgb[2]);
    }
  };

  // The film base read from the scan for a bundle conversion (film_base_detector.h): from the whole half-size image
  // (flat field applied) and the frame detector's base patch on the mosaic, as negicc-station's Auto reads it. On the
  // zero-copy CFA path (Sony RAWs, pixel shift) that is the station's reading; the white-light decode runs the detector
  // on the mosaic before decoding it (find_frame), as the station's does; RGB files have no mosaic for the detector, so
  // the base comes from the statistics.
  // A film base given in its place (--film_base's capture, or a previous run's: Scan::has_given_base) is taken as it
  // is.
  // |white|: the clipping level of half's samples before its flat field (film_base_detector.h).
  auto detect_base = [&](const negicc::RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                         float white) {
    if (!s.is_bundle) return;
    if (scan.has_given_base) {  // --film_base's capture, or a previous run's: logged where it was read
      scan.film_base = scan.given_base;
      return;
    }
    const double* xt = (s.bundle.has_crosstalk && !s.white_light) ? s.bundle.crosstalk_matrix_d : nullptr;
    const bool found = negicc::detect_film_base(half, det, sensor_w, sensor_h, xt, white, scan.film_base);
    const negicc::FilmBaseDetection& fb = scan.film_base;
    // A base at the sensor's clip: the rebate's patch refused for it, or the thinnest film the base is then read
    // from at the clip itself.
    const bool from_thinnest = found && fb.source == negicc::kFilmBaseSourceStatistics;
    // A capture of unexposed film is the other remedy where it is taken (triled_base_remedy).
    const char* clip_remedy = !s.white_light && o.input_mode == InputMode::kRaw
                                  ? "give --film_base <capture of unexposed film> or scan at a shorter exposure"
                                  : "scan at a shorter exposure";
    if (from_thinnest && fb.thinnest_clipped) {
      log_warn("the scan clips its film base (%s at the sensor's clip): the base read from it is the clip, below the "
               "true base; %s\n",
               fb.rebate_clipped ? "the rebate and the thinnest film are" : "the thinnest film is", clip_remedy);
    } else if (fb.rebate_clipped) {
      log_warn("the scan clips its film base (the rebate is at the sensor's clip): the base is read from %s instead, "
               "below the true base; %s\n",
               found ? "the thinnest film" : "the profile bundle", clip_remedy);
    }
    if (found) {
      log_info("Film base (%s): R=%.1f G=%.1f B=%.1f\n", fb.source.c_str(), fb.rgb[0], fb.rgb[1], fb.rgb[2]);
    } else {
      log_info("Film base: none read from the scan; the bundle's\n");
    }
  };

  // The frame detection the film base reads with a given --roi on the white-light decode (find_frame, below).
  FrameDetection base_det;
  bool has_base_det = false;
  // The CFA path's frame detection for the film base (below), kept for the print's floor (apply_print_floor).
  FrameDetection floor_det;
  bool has_floor_det = false;
  // The size of the mosaic the frame detector ran on (its boxes are in those pixels), for the base patch's scale.
  int det_sensor_w = 0, det_sensor_h = 0;
  // The analysis of the half-size image on the LibRaw, TIFF, JPEG and three-capture paths: for every bundle conversion,
  // since every one reads the film base from it (the CFA path does that inline).
  // --roi auto's frame: the detected crop, or the whole full_w x full_h image when the detection is not confident.
  auto take_detected_frame = [&](int full_w, int full_h) {
    if (scan.frame_det.conf < kLowFrameConfidence) {
      scan.roi[0] = 0;
      scan.roi[1] = 0;
      scan.roi[2] = full_w;
      scan.roi[3] = full_h;
    } else {
      for (int i = 0; i < 4; ++i) scan.roi[i] = scan.frame_det.crop[i];
    }
    scan.has_roi = true;
  };
  // The film base read from a half-size image (detect_base); gives the image as a view.
  auto read_base = [&](const LibRaw& raw) {
    const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
    negicc::RgbView half_view = negicc::RgbView::from_interleaved_rgb(
        &raw.imgdata.image[0][0], half_w, half_h, 4, (ptrdiff_t)half_w * 4);
    // The detector's boxes are in the mosaic's pixels: give its size, not the decode's (the half-size decode's
    // own size put the base patch off the image, so neg_process fell back to the statistics).
    // A single RAW's or linear DNG's detection (--roi auto's, or base_det with a given frame) ran on its raw data (a
    // RAW's mosaic, a linear DNG's RGB) at the sensor's size, on either light, and runs for a given --roi too, so the
    // replay reads the same base. An RGB file and three captures detect only for --roi auto's crop, on the assembled
    // image: their base comes from the statistics, with no detection recorded for it.
    const bool raw_det = o.files.size() == 1 &&
                         (o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng);
    // The half-size image is black-subtracted (LibRaw's raw2image_ex, or the reader's for an RGB file or three
    // captures), and its clip is imgdata.color.maximum; a flat field, where given, divided the mosaic before it.
    detect_base(half_view, (raw_det && o.roi_auto) ? &scan.frame_det : has_base_det ? &base_det : nullptr,
                raw_det ? det_sensor_w : 0, raw_det ? det_sensor_h : 0, (float)raw.imgdata.color.maximum);
    return half_view;
  };
  // The white-light decode's balance of the camera channels on the film base: the base read from its first half-size
  // decode is put on the bundle's (film_base_balance_gains). The analysis then reads the base again, balanced.
  ChannelBalance balance;
  if (s.white_light && s.bundle.has_film_base) {
    balance = [&](const LibRaw& raw, float k[3]) {
      read_base(raw);
      if (scan.film_base.found)
        negicc::film_base_balance_gains(raw, scan.film_base.rgb, s.bundle.profile_film_base_rgb, k);
      std::copy(k, k + 3, scan.balance_gains);
      log_info("Film-base balance before the camera matrix: camera channel gains %.6f %.6f %.6f\n", k[0], k[1], k[2]);
    };
  }
  // What follows a finished bundle search, on either decode path: on white light the curve fit (with the tone stage's
  // plan on its result), elsewhere the tone stage's plan on the search's own result.
  auto solve_tone_plan = [&](bool verbose) {
    // The descriptors of the final render's preview, and the curve between them and the heads' predictions.
    s.tone_plan = TonePlanCurve{};
    s.tone_stage_ran = true;
    ToneStagePlan& tp = s.tone_stage;
    if (tp.skipped || tp.preview.empty()) return;
    const size_t npx = tp.preview.size() / 3;
    s.tone_plan.render = tone_descriptors(tp.preview.data(), npx);
    s.tone_plan.curve = tone_plan(s.tone_plan.render, tp.pred_mean[ToneHead::kL2], tp.pred_mean[ToneHead::kL98],
                                  tp.pred_mean[ToneHead::kR9010], 1.0, &s.tone_plan.targets);
    tp.planned = true;
    s.has_tone = true;
    s.tone_curve = s.tone_plan.curve;
    if (verbose) {
      const TonePlanCurve& t = s.tone_plan;
      log_info("Tone: L2 %.1f -> %.1f, L98 %.1f -> %.1f, L90-L10 %.1f -> %.1f; curve black %.2f white %.2f "
               "gamma %.3f contrast %.3f\n",
               (float)t.render.L2, (float)t.targets.L2, (float)t.render.L98, (float)t.targets.L98,
               (float)t.render.range(), (float)t.targets.range, (float)t.curve.black, (float)t.curve.white,
               (float)t.curve.gamma, (float)t.curve.contrast);
    }
  };
  auto stages_after_search = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                                 const ParamSearchOptions& opts) -> bool {
    const ToneHead* tone_head = s.tone_active && s.tone_head.loaded() ? &s.tone_head : nullptr;
    if (s.fit_curves_active) {
      // The curves on the bracket the search chose, then E re-metered on its combined profile to Auto's median
      // L* (the station's Fit curves toggle); the conversion takes those (Scan::params). Auto: the same on every
      // white-light bundle search.
      negicc::CurveFitOptions fit_opts;  // the defaults (input_curves.h); the sidecar records them
      // The fit and the re-meter render the grid as the export will: -P's encoding (srgb or srgb-g10, prepare).
      const int export_encoding = (o.colorspace == "srgb" || s.density) ? ENCODE_SRGB : ENCODE_SRGB_LINEAR;
      s.curves_base_idx = scan.auto_res.target_idx;  // the fit composes the curves into it in place
      s.curves_base_prof = bundle.targets[s.curves_base_idx].film_prof;
      s.curves_base_prof.rebind_pointers();
      ParamSearchOptions fit_search_opts = opts;
      set_fit_frame(fit_search_opts);
      if (!run_white_light_fit(bundle, s.engine, frame, scan_shutter_s, scan_iso, fit_search_opts, scan.auto_res,
                               fit_opts, export_encoding, tone_head, s.curve_fit)) {
        log_error("Fitting the input curves failed\n");
        return false;
      }
      s.tone_stage = std::move(s.curve_fit.tone);
      solve_tone_plan(fit_search_opts.verbose);
      if (!s.curve_fit.neutral_anchors) {
        log_info("Curves: %s; no scene-neutral anchors, the fit keeps Auto's colour\n",
                 s.neutral_head.empty() ? "no neutral head" : "no neutral evidence from the head");
      }
      s.input_curves = s.curve_fit.fit.curves;
      s.input_curves_source = o.fit_curves == Options::kFitCurvesOn ? "fit" : "auto";
      scan.has_curve_params = true;
      scan.curve_params = s.curve_fit.params;
    } else if (tone_head) {
      // No curve fit (TriLED, whose curves are the measured ones, or --fit_curves off): the search's result stands
      // and the tone stage is planned on its render (-P srgb, prepare).
      if (!plan_auto_tone_stage(bundle, s.engine, frame, scan_shutter_s, scan_iso, opts, scan.auto_res, ENCODE_SRGB,
                                *tone_head, s.tone_stage)) {
        log_error("Planning the tone stage failed\n");
        return false;
      }
      solve_tone_plan(opts.verbose);
    }
    return true;
  };
  // With a shadow alternative (ParamSearchResult::shadow), the alternative's own stages after the search (the curve
  // fit on white light, the tone plan) and its roll-off, from the same state as the result's, kept in
  // scan.shadow_candidate for solve; the state is then the result's again, whose roll-off the caller measures.
  // Whether the search looks for a shadow alternative: only where one could be kept, an sRGB export with the tone
  // stage on (the choice is its clip check's), no density output, no print, and more than one bracket.
  const bool shadow_possible = s.tone_active && o.colorspace == "srgb" && !s.density && s.print_profs.empty() &&
                               o.prints.empty() && bundle.targets.size() > 1;
  auto stages_and_shadow = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                               const ParamSearchOptions& opts) -> bool {
    scan.shadow_search = scan.auto_res.shadow;
    scan.shadow_candidate.reset();
    scan.shadow_choice = ShadowChoice{};
    const std::shared_ptr<ParamSearchResult> alt = scan.shadow_search.result;
    scan.auto_res.shadow = ShadowAlternative{};
    if (!alt) return stages_after_search(frame, scan_shutter_s, scan_iso, opts);
    const double t0 = omp_get_wtime();
    auto prof_of = [&](int idx) {
      ParsedFilmProfile p = bundle.targets[idx].film_prof;
      p.rebind_pointers();
      return p;
    };
    const int ia = scan.auto_res.target_idx, ib = alt->target_idx;
    ShadowCandidate a, b;
    a.prof0 = prof_of(ia);
    b.prof0 = prof_of(ib);
    if (!stages_after_search(frame, scan_shutter_s, scan_iso, opts)) return false;
    capture_candidate(s, scan, a);
    bundle.targets[ia].film_prof = a.prof0;
    bundle.targets[ia].film_prof.rebind_pointers();
    s.curve_fit = WhiteLightFitResult{};
    s.tone_stage = ToneStagePlan{};
    s.tone_stage_ran = false;
    s.tone_plan = TonePlanCurve{};
    s.has_tone = false;
    s.tone_curve = ToneCurve{};
    s.input_curves = negicc::InputCurves{};
    s.input_curves_source.clear();
    s.curves_base_idx = -1;
    scan.has_curve_params = false;
    scan.has_rolloff = false;
    scan.has_knob_res = false;
    scan.has_knob_curves = false;
    scan.knob_res = KnobSolveResult{};
    scan.has_plain = false;
    scan.has_plain_prof = false;
    scan.auto_res = *alt;
    log_info("The shadow alternative (%s): its stages after the search, the knobs and the roll-off:\n",
             alt->target_name.c_str());
    const bool ok = stages_after_search(frame, scan_shutter_s, scan_iso, opts) &&
                    solve_knobs_and_rolloff(frame, scan_shutter_s, scan_iso);
    if (ok) {
      capture_candidate(s, scan, b);
      b.result_prof0 = a.prof0;
      b.result_prof0.rebind_pointers();
    } else {
      log_warn("the shadow alternative's stages after the search failed; the search's result stands\n");
    }
    bundle.targets[ib].film_prof = b.prof0;  // the alternative's bracket as it was, then the result's state
    bundle.targets[ib].film_prof.rebind_pointers();
    restore_candidate(s, scan, a);
    scan.has_rolloff = false;  // the result's, measured by the caller
    if (ok) scan.shadow_candidate = std::make_shared<ShadowCandidate>(std::move(b));
    else scan.shadow_choice.not_checked = "its stages after the search failed";
    scan.shadow_choice.stages_ms = (omp_get_wtime() - t0) * 1000.0;
    return true;
  };
  HalfSizeAnalysis analyse;
  if (s.is_bundle) {
    analyse = [&](const LibRaw& raw) -> bool {
      double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
      scan.scan_exposure(raw, scan_shutter_s, scan_iso);
      // An RGB file or three captures with --roi auto: the detection ran on the whole image before this analysis,
      // which sees the whole image too, so the frame is placed from it here (the RAW path's find_frame does so before
      // the half-size decode). raw's rawdata sizes are the full image's (load_rgb_input).
      if (o.roi_auto && !scan.has_roi &&
          (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot)) {
        take_detected_frame(raw.imgdata.rawdata.sizes.width, raw.imgdata.rawdata.sizes.height);
      }
      // The frame the output writes, halved, as the CFA path searches it: --roi (or --roi auto's); else, unless
      // --no_crop, one RAW's inset crop (inset_crop) or three captures' (load_rgb_input gives it in raw's rawdata
      // sizes); else the whole image.
      const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
      int* bbox = scan.auto_frame_half;
      bbox[0] = 0;
      bbox[1] = 0;
      bbox[2] = half_w;
      bbox[3] = half_h;
      const auto& inset = raw.imgdata.rawdata.sizes.raw_inset_crops[0];
      int full_frame[4] = {inset.cleft, inset.ctop, inset.cwidth, inset.cheight};
      bool cropped = o.input_mode == InputMode::kThreeShot && full_frame[2] > 0 && full_frame[3] > 0;
      if (scan.has_roi)
        std::copy(scan.roi, scan.roi + 4, full_frame);
      else if (!o.roi_auto && !o.no_crop && (o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng))
        cropped = inset_crop(raw, full_frame);
      if (scan.has_roi || cropped) {
        // The right and bottom edges are summed in 64 bits: a given --roi's x + w (or y + h) can pass INT_MAX. Their
        // halves fit an int, and clamp_rect then clamps the frame to the half-size image.
        bbox[0] = full_frame[0] / 2;
        bbox[1] = full_frame[1] / 2;
        bbox[2] = std::max(1, (int)(((long long)full_frame[0] + full_frame[2]) / 2) - bbox[0]);
        bbox[3] = std::max(1, (int)(((long long)full_frame[1] + full_frame[3]) / 2) - bbox[1]);
        clamp_rect(bbox, half_w, half_h);
      }
      scan.frame_is_whole = bbox[0] == 0 && bbox[1] == 0 && bbox[2] == half_w && bbox[3] == half_h;
      // Step 3 & Step 4: zero-copy RgbView over the half-size image, cropped and oriented in place.
      negicc::RgbView half_view = read_base(raw);
      negicc::RgbView frame =
          half_view.crop_and_orient(bbox[0], bbox[1], bbox[2], bbox[3], scan.rot_cw, scan.hflip, scan.vflip);
      // A single RAW's or linear DNG's base detection (find_frame, below) serves the floor as on the CFA path; TIFF,
      // JPEG and three captures have no detection without --roi auto, so their floor reads the whole scan.
      apply_print_floor(frame, half_view, has_base_det ? &base_det : nullptr, det_sensor_w, det_sensor_h,
                        scan_shutter_s, scan_iso);
      if (s.bundle_search && s.has_cached_sidecar && s.cached_sidecar.is_auto) {
        restore_cached_auto();
      } else if (s.bundle_search) {
        // Auto's time (the sidecar's auto.time_ms): the search and the curve fit, as on the CFA path.
        const double t_auto_0 = omp_get_wtime();
        ParamSearchOptions opts;
        opts.renderer = search_renderer(o, s);
        if (o.dino_iters_given) opts.max_steps = o.dino_iters;
        opts.film_base_rgb_d = scan.film_base_rgb();
        opts.film_base_from_scan = opts.film_base_rgb_d != nullptr;
        set_search_pins(o, s, opts);
        opts.shadow_alternative = shadow_possible;
        opts.probe_path = o.probe_intent;
        opts.probe_params = o.probe_params;
        opts.probe_stops = o.probe_stops;
        if (!bundle_search(frame, scan_shutter_s, scan_iso, opts)) {
          log_error("The conversion parameter search failed\n");
          return false;
        }
        if (scan.auto_res.probed) {  // --probe_intent: the probe file is written and closed; nothing is converted
          const int r = finish_search_passes(o, s);
          std::fflush(stdout);
          std::fflush(stderr);
          std::_Exit(r);  // without static teardown: the inference engine's threads are still alive here
        }
        scan.has_auto_res = true;
        if (!stages_and_shadow(frame, scan_shutter_s, scan_iso, opts)) return false;
        scan.t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
      }
      // The bracket analysis (timings_ms.analysis, not raw_decode): the knobs and the roll-off.
      const double t_analysis_0 = omp_get_wtime();
      if (!solve_knobs_and_rolloff(frame, scan_shutter_s, scan_iso)) return false;
      scan.t_analysis_ms += (omp_get_wtime() - t_analysis_0) * 1000.0;
      return true;
    };
  }


  // An analysis that fails has logged why; the decode then returns nothing, which is not the file failing to open.
  bool analysis_failed = false;
  if (analyse) {
    analyse = [&analysis_failed, run = std::move(analyse)](const LibRaw& raw) {
      if (run(raw)) return true;
      analysis_failed = true;
      return false;
    };
  }

  FrameDetectionParams det_params;
  det_params.format = o.film_format;
  det_params.axis_hint = o.roi_axis_hint;
  RawRegionFinder find_frame;
  // The film base's frame detection when the frame is given (--roi x y w h, or none): negicc-station's white-light
  // decode runs the detector on every scan's mosaic as unpacked, with the default parameters
  // (RawProcessor::load_raw_white_light), and reads the base from its rebate patch; neg_process does the same, with
  // --film_format when given (every format otherwise, as the station), so both read the same base. With --roi auto
  // the detection that found the frame serves (base_det: above).
  // A linear DNG too: --roi auto detects on it (film_image_from_raw reads its RGB), so its replay with the recorded
  // --roi must as well.
  if (!o.roi_auto && s.is_bundle && o.files.size() == 1 &&
      (o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng)) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      FilmImage mosaic;
      if (film_image_from_raw(raw, mosaic)) {
        FrameDetectionParams base_params;
        if (o.film_format_given) base_params.format = o.film_format;  // else every format, as the station
        detect_film_frame(mosaic, base_params, base_det);
        has_base_det = true;
        scan.base_det_format = base_params.format;
        det_sensor_w = mosaic.width;
        det_sensor_h = mosaic.height;
      }
      roi[0] = roi[1] = roi[2] = roi[3] = 0;   // only looked: the given roi stays
      return true;
    };
  }
  if (o.roi_auto && o.files.size() != 4) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        return false;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, scan.frame_det);
      scan.base_det_format = det_params.format;   // read_base takes this detection for the base
      scan.t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      det_sensor_w = mosaic.width;
      det_sensor_h = mosaic.height;
      take_detected_frame(mosaic.width, mosaic.height);
      for (int i = 0; i < 4; ++i) roi[i] = scan.roi[i];
      return true;
    };
  }

  // Logs --roi auto's detection, and warns when it fell back or the frame is vertical and not turned.
  auto print_frame_detection = [&]() {
    const FrameDetection& det = scan.frame_det;
    const char* axis_name = det.axis == 'h' ? "h" : (det.axis == 'v' ? "v" : "undecided");
    log_info("Film frame: axis=%s format=%s orientation=%s rotation=%.2f deg gaps=%d frames=%d conf=%.2f%s%s\n",
           axis_name, film_format_name(det.format), det.orientation.c_str(), det.rotation_deg,
           det.n_gaps, 1 + (int)det.secondaries.size(), det.conf,
           det.reason.empty() ? "" : " reason: ", det.reason.c_str());
    if (det.n_gaps > 0) {
      log_info("  Gaps (sensor px):");
      for (int g = 0; g < 2; ++g)
        if (det.gaps[g][0] >= 0) log_info(" [%d, %d]", det.gaps[g][0], det.gaps[g][1]);
      log_info("\n");
    }
    log_info("  Frame (sensor px): x=%d y=%d w=%d h=%d (frame centre %.1f, %.1f, size %.1f x %.1f, %.1f px/mm)\n",
           scan.roi[0], scan.roi[1], scan.roi[2], scan.roi[3], det.primary.rect.cx,
           det.primary.rect.cy, det.primary.rect.w, det.primary.rect.h, det.px_per_mm);
    for (const FilmFrame& f : det.secondaries)
      log_info("  Frame %+d (sensor px): visible x=%d y=%d w=%d h=%d (%.0f%% of a frame)\n", f.index, f.visible[0],
             f.visible[1], f.visible[2], f.visible[3], 100.0f * f.visible_fraction);
    if (det.conf < kLowFrameConfidence) {
      log_warn("low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               det.conf, det.reason.empty() ? "" : ", reason: ", det.reason.c_str());
    }
    if (s.use_dino && det.axis == 'v' && scan.rot_cw == 0) {
      log_warn("axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
    }
  };


  // Determine whether the 5-step zero-copy CFA pipeline applies (Step 1..5).
  bool use_cfa_pipeline = false;
  scan.t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  LibRaw *unpacked_probe = nullptr;
  float cfa_white_level = 65535.0f;
  int cfa_rx = 0, cfa_ry = 0, cfa_rw = 0, cfa_rh = 0;
  negicc::RgbView full_half_view;
  negicc::RgbView oriented_half_view;
  negicc::RgbView search_half_view;
  negicc::FlatSampleArgs flat_full{};

  if ((o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng) &&
      (o.files.size() == 4 || (s.raw_decode == RawDecode::kCameraRgb && o.quality < 0))) {
    LibRaw* probe = new LibRaw();
    log_info("Loading RAW file %s\n", o.files[0].c_str());
    int ret = open_and_unpack_raw_file(probe, o.files[0]);
    if (ret != LIBRAW_SUCCESS) {
      log_error("Cannot open/unpack %s: %s\n", o.files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    if (o.files.size() == 4) {
      if (!is_sony_a7rm4(probe)) {
        log_error("4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
                probe->imgdata.idata.make, probe->imgdata.idata.model);
        delete probe;
        return 1;
      }
      use_cfa_pipeline = true;
    } else if (is_sony_cfa_geometry(probe) && probe->imgdata.rawdata.raw_image) {
      use_cfa_pipeline = true;
    }
    if (use_cfa_pipeline) {
      proc = probe;
    } else {
      unpacked_probe = probe;
    }
  }


  if (use_cfa_pipeline) {
    if (o.roi_auto) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(*proc, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        delete proc;
        return 1;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, scan.frame_det);
      scan.t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
    }

    // Step 1: Per-channel black subtraction in-place on the unpacked CFA mosaic.
    if (!subtract_cfa_black_in_place(proc, cfa_white_level)) {
      log_error("Invalid CFA mosaic geometry in %s\n", o.files[0].c_str());
      delete proc;
      return 1;
    }
    const int src_w = proc->imgdata.sizes.width;
    const int src_h = proc->imgdata.sizes.height;
    const size_t cfa_pitch = proc->imgdata.sizes.raw_pitch / 2;
    const unsigned int filters = proc->imgdata.idata.filters;
    const uint16_t* cfa_origin = proc->imgdata.rawdata.raw_image +
                                 (size_t)proc->imgdata.sizes.top_margin * cfa_pitch +
                                 proc->imgdata.sizes.left_margin;

    // For 4x pixel shift, unpack and black-subtract frames 1..3 in Step 1 and keep all 4 CFA mosaics in memory.
    if (o.files.size() == 4) {
      for (int mi = 1; mi < 4; ++mi) {
        scan.ps_extra[mi - 1].reset(new LibRaw());
        LibRaw* fp = scan.ps_extra[mi - 1].get();
        log_info("Loading RAW file %s\n", o.files[mi].c_str());
        int fret = fp->open_file(o.files[mi].c_str());
        if (fret != LIBRAW_SUCCESS || (fret = fp->unpack()) != LIBRAW_SUCCESS) {
          log_error("Cannot open/unpack %s: %s\n", o.files[mi].c_str(), libraw_strerror(fret));
          delete proc;
          return 1;
        }
        if (!raw_levels_valid(*fp, o.files[mi])) {
          delete proc;
          return 1;
        }
        const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
        if (!is_sony_a7rm4(fp) || fp->imgdata.sizes.width != src_w || fp->imgdata.sizes.height != src_h ||
            fpitch != cfa_pitch || fp->imgdata.idata.filters != filters ||
            fp->imgdata.sizes.top_margin != proc->imgdata.sizes.top_margin ||
            fp->imgdata.sizes.left_margin != proc->imgdata.sizes.left_margin ||
            std::fabs(fp->imgdata.other.iso_speed - proc->imgdata.other.iso_speed) > 1e-3f ||
            std::fabs(fp->imgdata.other.shutter - proc->imgdata.other.shutter) > 1e-6f) {
          log_error("4-shot pixel shift merge is only available with matching Sony A7RM4 frames\n");
          delete proc;
          return 1;
        }
        float fw_white = 0.0f;
        if (!subtract_cfa_black_in_place(fp, fw_white)) {
          log_error("Invalid CFA mosaic geometry in %s\n", o.files[mi].c_str());
          delete proc;
          return 1;
        }
      }
    }

    // Step 2: Downsizing via zero-copy RgbView over the CFA mosaic.
    const int half_w = src_w / 2;
    const int half_h = src_h / 2;
    std::shared_ptr<const negicc::FlatTables> flat_half_tables;
    negicc::FlatSampleArgs flat_half{};
    if (s.flat_field) {
      try {
        flat_half_tables = s.flat_field->tables(half_w, half_h, true);
        flat_half = flat_half_tables->host();
        scan.flat_full_tables = s.flat_field->tables(src_w, src_h, false);
        flat_full = scan.flat_full_tables->host();
      } catch (const std::exception& e) {
        log_error("%s\n", e.what());
        delete proc;
        return 1;
      }
    }
    full_half_view = negicc::RgbView::from_cfa_half(
        cfa_origin, cfa_pitch, src_w, src_h, filters, nullptr, flat_half, flat_half_tables);
    if (s.is_bundle) {
      // The frame detector on the black-subtracted mosaic of the first frame, with its default parameters, as
      // negicc-station runs it for the film base (CapturedImage::film_frame_detection).
      FrameDetection base_det;
      FilmImage mosaic;
      const float no_black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      // A Bayer mosaic: LibRaw's filters is a 2x2 bitmask above 1000 (9 is X-Trans).
      const bool has_det =
          filters > 1000 &&
          film_image_from_cfa(cfa_origin, src_w, src_h, (ptrdiff_t)cfa_pitch, 2,
                              [filters](int row, int col) { return negicc::bayer_color_at(row, col, filters); },
                              no_black, cfa_white_level, mosaic);
      FrameDetectionParams base_params;
      if (o.film_format_given) base_params.format = o.film_format;  // else every format, as the station
      if (has_det) {
        detect_film_frame(mosaic, base_params, base_det);
        scan.base_det_format = base_params.format;
      }
      detect_base(full_half_view, has_det ? &base_det : nullptr, src_w, src_h, cfa_white_level);
      if (has_det) {
        floor_det = base_det;
        has_floor_det = true;
      }
    }

    // Step 3: Crop, rotate, flip & inline ROI search on the zero-copy view.
    if (o.roi_auto) {
      take_detected_frame(src_w, src_h);
      print_frame_detection();
    }

    cfa_rx = 0;
    cfa_ry = 0;
    cfa_rw = src_w;
    cfa_rh = src_h;
    // --roi, or a single RAW's inset crop unless --no_crop: refused when it lies outside the image, clamped to its
    // right and bottom edges.
    int inset[4];
    const int* frame = scan.has_roi ? scan.roi
                       : (!o.no_crop && o.files.size() == 1 && inset_crop(*proc, inset)) ? inset
                                                                                       : nullptr;
    if (frame) {
      if (frame[0] < 0 || frame[1] < 0 || frame[0] >= src_w || frame[1] >= src_h || frame[2] <= 0 || frame[3] <= 0) {
        log_error("%s [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n", scan.has_roi ? "ROI" : "The inset crop",
                  frame[0], frame[1], frame[2], frame[3], src_w, src_h);
        delete proc;
        return 1;
      }
      cfa_rx = frame[0];
      cfa_ry = frame[1];
      cfa_rw = std::min(frame[2], src_w - cfa_rx);
      cfa_rh = std::min(frame[3], src_h - cfa_ry);
    }

    int out_half[4] = {
        cfa_rx / 2,
        cfa_ry / 2,
        std::max(1, (cfa_rx + cfa_rw) / 2 - cfa_rx / 2),
        std::max(1, (cfa_ry + cfa_rh) / 2 - cfa_ry / 2),
    };
    clamp_rect(out_half, half_w, half_h);
    for (int i = 0; i < 4; ++i) scan.auto_frame_half[i] = out_half[i];
    scan.frame_is_whole = out_half[0] == 0 && out_half[1] == 0 && out_half[2] == half_w && out_half[3] == half_h;
    search_half_view = full_half_view.crop_and_orient(
        scan.auto_frame_half[0], scan.auto_frame_half[1], scan.auto_frame_half[2], scan.auto_frame_half[3],
        scan.rot_cw, scan.hflip, scan.vflip);
    oriented_half_view = search_half_view;
    {
      double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      apply_print_floor(search_half_view, full_half_view, has_floor_det ? &floor_det : nullptr, src_w, src_h,
                        scan_shutter_s, scan_iso);
    }

    // Step 4: Conversion parameter search inline on the zero-copy oriented RgbView.
    if (s.bundle_search && s.has_cached_sidecar && s.cached_sidecar.is_auto) {
      restore_cached_auto();
    } else if (s.bundle_search) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      ParamSearchOptions opts;
      opts.renderer = search_renderer(o, s);
      if (o.dino_iters_given) opts.max_steps = o.dino_iters;
      opts.film_base_rgb_d = scan.film_base_rgb();
      opts.film_base_from_scan = opts.film_base_rgb_d != nullptr;
      set_search_pins(o, s, opts);
      opts.shadow_alternative = shadow_possible;
      opts.probe_path = o.probe_intent;
      opts.probe_params = o.probe_params;
      opts.probe_stops = o.probe_stops;
      if (!bundle_search(search_half_view, scan_shutter_s, scan_iso, opts)) {
        log_error("The conversion parameter search failed\n");
        delete proc;
        return 1;
      }
      if (scan.auto_res.probed) {  // --probe_intent: the probe file is written and closed; nothing is converted
        const int r = finish_search_passes(o, s);
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(r);  // without static teardown: the inference engine's threads are still alive here
      }
      scan.has_auto_res = true;
      if (!stages_and_shadow(search_half_view, scan_shutter_s, scan_iso, opts)) {
        delete proc;
        return 1;
      }
      scan.t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    }
    if (s.is_bundle) {
      double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      const double t_analysis_0 = omp_get_wtime();  // timings_ms.analysis, as on the LibRaw path
      if (!solve_knobs_and_rolloff(search_half_view, scan_shutter_s, scan_iso)) {
        delete proc;
        return 1;
      }
      scan.t_analysis_ms += (omp_get_wtime() - t_analysis_0) * 1000.0;
    }
    scan.t_raw_decode_ms = (omp_get_wtime() - scan.t_raw_0) * 1000.0 - scan.t_auto_ms - scan.t_frame_det_ms -
                           scan.t_analysis_ms;

    // The mosaic and the frame, for render.
    scan.mosaic.frames[0] = cfa_origin;
    scan.mosaic.n_frames = o.files.size() == 4 ? 4 : 1;
    for (int i = 0; i + 1 < scan.mosaic.n_frames; ++i) {
      LibRaw* fp = scan.ps_extra[i].get();
      const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
      scan.mosaic.frames[i + 1] = fp->imgdata.rawdata.raw_image + (size_t)fp->imgdata.sizes.top_margin * fpitch +
                                  fp->imgdata.sizes.left_margin;
    }
    scan.mosaic.pitch = cfa_pitch;
    scan.mosaic.width = src_w;
    scan.mosaic.height = src_h;
    scan.mosaic.filters = filters;
    scan.mosaic.white_level = cfa_white_level;
    scan.mosaic.flat = flat_full;
    scan.mosaic.flat_tables = scan.flat_full_tables.get();
    scan.cfa_region = {cfa_rx, cfa_ry, cfa_rw, cfa_rh, scan.rot_cw, scan.hflip, scan.vflip};
  } else {
    if (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot) {
      proc = load_rgb_input(o.files, o.input_mode, s.camera_rgb_input, o.quality,
                            !o.no_crop && !scan.has_roi && !o.roi_auto,
                            (!o.roi_auto && scan.has_roi) ? scan.roi : nullptr, analyse,
                            o.roi_auto ? &scan.frame_det : nullptr, det_params, scan.input_source);
    } else {
      proc = load_raw(o.files[0], true,
                      o.quality,
                      !o.no_crop && !scan.has_roi && !o.roi_auto,
                      s.raw_decode,
                      (!o.roi_auto && scan.has_roi) ? scan.roi : nullptr,
                      analyse, balance, find_frame, s.flat_field.get(),
                      unpacked_probe);
    }
    unpacked_probe = nullptr;
    if (!proc) {
      if (!analysis_failed) log_error("Cannot open %s\n", o.files[0].c_str());
      return 1;
    }
    if (o.roi_auto) {
      if (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot) {
        const auto& rs = proc->imgdata.rawdata.sizes;
        const auto& is = proc->imgdata.sizes;
        take_detected_frame(rs.width ? rs.width : is.width, rs.height ? rs.height : is.height);
      }
      print_frame_detection();
    }
    if (scan.rot_cw != 0 || scan.hflip || scan.vflip) {
      if (!extract_roi_and_transform(proc, 0, 0, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                     scan.rot_cw, scan.hflip, scan.vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      log_info("Transformed subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, scan.rot_cw, scan.hflip ? 1 : 0,
             scan.vflip ? 1 : 0);
    }
    scan.t_raw_decode_ms = (omp_get_wtime() - scan.t_raw_0) * 1000.0 - scan.t_auto_ms - scan.t_frame_det_ms -
                           scan.t_analysis_ms;
  }

  const int orig_w = proc->imgdata.rawdata.sizes.width ? proc->imgdata.rawdata.sizes.width : proc->imgdata.sizes.width;
  const int orig_h =
      proc->imgdata.rawdata.sizes.height ? proc->imgdata.rawdata.sizes.height : proc->imgdata.sizes.height;

  scan.proc.reset(proc);
  scan.cfa = use_cfa_pipeline;
  scan.orig_w = orig_w;
  scan.orig_h = orig_h;
  log_info("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  log_info("Shutter Speed: %f\n", proc->imgdata.other.shutter);
  return 0;
}

int begin_conversion(const Options& o, const Setup& s, Conversion& c) {
  const std::vector<int> unit_fb = {1, 1, 1};
  c.exposure_comp = o.exposure_comp;
  c.gain_g = o.gain_g;
  c.gain_b = o.gain_b;
  c.knee = kKnee;

  // The sidecar records the bundle's film base. The film base scale of the input matrix is 1: a bundle conversion's
  // film base (the scan's, else the bundle's) is in the station's merged matrix (solve).
  c.sidecar_profile_film_base_rgb = {1, 1, 1};
  if (s.is_bundle && s.bundle.has_film_base) {
    c.sidecar_profile_film_base_rgb = {s.bundle.profile_film_base_rgb[0], s.bundle.profile_film_base_rgb[1],
                                       s.bundle.profile_film_base_rgb[2]};
  }
  c.profile_film_base_rgb = {1, 1, 1};

  if (s.is_bundle) {
    // The bundle's film base (when none is read from the scan) is in the station's merged matrix, so in E, g and b,
    // not applied as a film base scale.
    c.sidecar_film_base_rgb = c.sidecar_profile_film_base_rgb;
    c.film_base_source = "bundle";
  }

  if (o.exposure_given) c.pin_mask |= PIN_E;
  if (o.gain_g_given) c.pin_mask |= PIN_G;
  if (o.gain_b_given) c.pin_mask |= PIN_B;

  adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff,
                           c.exposure_comp,
                           c.gain_g, c.gain_b,
                           c.profile_film_base_rgb,
                           unit_fb,
                           c.merged_matrix);
  return 0;
}

int render(const Options& o, const Setup& s, Scan& scan, Backend& backend) {
  (void)o;
  (void)s;
  if (!scan.cfa) return 0;  // LibRaw's decode, a TIFF, a JPEG or three captures: the image is in scan.proc already
  const double t_step5_debayer_0 = omp_get_wtime();
  LibRaw* proc = scan.proc.get();
  const Region& r = scan.cfa_region;
  int norm_rot = r.rot_cw;
  bool norm_hf = r.hflip, norm_vf = r.vflip;
  normalize_rotation(norm_rot, norm_hf, norm_vf);
  const bool swap_dims = (norm_rot == 90 || norm_rot == 270);
  const int dst_w = swap_dims ? r.h : r.w;
  const int dst_h = swap_dims ? r.w : r.h;
  log_info(scan.mosaic.n_frames == 1 ? "Debayer quality: grain-aware\n" : "Merging 4 images...\n");
  // A mosaic is freed once it is not needed any more: after the render, or as soon as the backend has its own copy of
  // it. Frame 0 is LibRaw's raw buffer (proc keeps its metadata); a pixel shift's other three are their own.
  bool freed[4] = {false, false, false, false};
  const auto free_frame = [&](int k) {
    if (freed[k]) return;
    freed[k] = true;
    if (k > 0) {
      scan.ps_extra[k - 1].reset();
      return;
    }
    if (proc->imgdata.rawdata.raw_alloc && !proc->imgdata.image) {
      proc->imgdata.image = (ushort (*)[4])proc->imgdata.rawdata.raw_alloc;
      proc->imgdata.rawdata.raw_alloc = nullptr;
      proc->imgdata.rawdata.raw_image = nullptr;
      proc->free_image();
    }
    proc->recycle_datastream();
  };
  backend.render(scan.mosaic, r, dst_w, dst_h, scan.rgb3, free_frame);   // sizes scan.rgb3
  scan.full_size_decode = backend.render_ran;
  for (int k = 0; k < 4; ++k) free_frame(k);
  scan.mosaic = MosaicInput();

  proc->imgdata.sizes.iwidth = dst_w;
  proc->imgdata.sizes.iheight = dst_h;
  proc->imgdata.idata.colors = 3;
  proc->imgdata.params.output_bps = 16;
  log_info("Processed image size: %dx%d\n", dst_w, dst_h);
  scan.t_raw_decode_ms += (omp_get_wtime() - t_step5_debayer_0) * 1000.0;
  return 0;
}

static int solve_one(const Options& o, Setup& s, Scan& scan, Conversion& c, bool quiet = false) {
  ProfileBundle& bundle = s.bundle;
  LibRaw* proc = scan.proc.get();
  // With a bundle, E, g and b are the station's (the search's, and what -E, --gain_g and --gain_b take): the
  // conversion is the station's, the bundle's merged matrix (crosstalk, film base and the scan's exposure ratio)
  // scaled per row by E, E*g and E*b, so neg_process renders what negicc-station renders. row_scale turns them into
  // the absolute exposure and gains on the crosstalk (the sidecar's matrix.*) and back.
  float station_base[9];
  double row_scale[3] = {1.0, 1.0, 1.0};
  if (s.is_bundle) {
    double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, scan.film_base_rgb(),
                                  scan.film_base_rgb() != nullptr, station_base, nullptr, row_scale);
  }
  auto apply_station = [&](float e, float g, float b) {
    c.station_params = std::array<float, 3>{e, g, b};
    negicc::candidate_matrix(station_base, e, g, b, c.merged_matrix);
    c.exposure_comp = (float)(row_scale[0] * (double)e);
    c.gain_g = (float)(row_scale[1] / row_scale[0] * (double)g);
    c.gain_b = (float)(row_scale[2] / row_scale[0] * (double)b);
  };
  const std::vector<int> unit_fb = {1, 1, 1};

  if (s.is_bundle && !s.bundle_search) {
    // Without the search a bundle converts with the given values: prepare runs the search unless --target, -E and
    // both gains are all given.
    if (s.manual_target_idx < 0 || !s.exposure_given) {
      log_error("Internal error: a bundle conversion without the search lacks --target or -E\n");
      return 1;
    }
  } else if (!s.is_bundle) {
    if (!s.film_prof_arg.empty() && !load_film_profile(s.film_prof_arg, c.film_prof)) {
      log_error("Cannot load film profile %s\n", s.film_prof_arg.c_str());
      return 1;
    }
  }

  auto log_knobs = [&]() {
    if (!scan.has_knob_res) return;
    log_info("Knobs (brightness %+.2f, tint %+.2f, warmth %+.2f%s): grey L*=%.2f a*=%.2f b*=%.2f -> E %.4f g %.4f "
             "b %.4f\n",
             o.brightness, o.tint, o.warmth, scan.knob_res.limited ? ", limited" : "",
             scan.knob_res.grey_lab[0], scan.knob_res.grey_lab[1], scan.knob_res.grey_lab[2],
             scan.knob_res.e, scan.knob_res.g, scan.knob_res.b);
  };

  if (scan.has_auto_res) {
    const ParamSearchResult& res = scan.auto_res;
    c.bundle_target_idx = res.target_idx;
    c.bundle_target_name = res.target_name;
    c.film_prof = bundle.targets[c.bundle_target_idx].film_prof;
    c.film_prof.rebind_pointers();
    // The search's E, g and b (after the curve fit's re-meter where the fit runs), moved by the knobs when
    // given, converted as the station converts them (the sidecar's cli_equivalent reproduces it).
    const ConversionParams& p = scan.params();
    const float fe = scan.has_knob_res ? scan.knob_res.e : p.e;
    const float fg = scan.has_knob_res ? scan.knob_res.g : p.g;
    const float fb = scan.has_knob_res ? scan.knob_res.b : p.b;
    apply_station(fe, fg, fb);
    if (!quiet && !(s.has_cached_sidecar && s.cached_sidecar.is_auto)) {
      log_info("Auto: %s E %.4f g %.4f b %.4f (resid %.4f, %s), %d inferences\n", res.target_name.c_str(),
             res.params.e, res.params.g, res.params.b, res.residual, to_string(res.status),
             res.inferences);
      auto ms = [](std::chrono::steady_clock::duration d) {
        return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
      };
      log_info("Timings (ms): render %lld, infer %lld, residual %lld, search %lld, total %lld\n",
             ms(res.render_time), ms(res.infer_time), ms(res.residual_time), ms(res.search_time),
             ms(res.total_time));
    }
    if (quiet) {
      // The conversion without the knobs (Scan::has_plain), for the tone stage's check (solve logs it).
    } else {
      if (scan.has_curve_params && s.curve_fit.fallback) {
        log_info("Curve fit fell back (%s): no curves; E %.4f g %.4f b %.4f on the bracket's profile\n",
                 s.curve_fit.fallback, p.e, p.g, p.b);
      } else if (scan.has_curve_params) {
        log_info("Fitted curves: E %.4f g %.4f b %.4f on the combined profile\n", p.e, p.g, p.b);
      }
      log_knobs();
      if (!s.print_id.empty()) {
        // A print is made by its own search, which takes no -E, gains or --target: its A, g and b.
        // The bracket nearest the print's final A (after the knobs, their moved A), as the sidecar records it.
        log_info("Print: A %.6g g %.9g b %.9g (E held; nearest %s)%s\n\n", s.print_a, fg, fb,
                 bundle.targets[nearest_bracket(s.print_cont.a_mid, s.print_a)].name.c_str(),
                 scan.has_knob_res ? ", moved by the knobs" : "");
      } else {
        log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n\n", fe, fg, fb, c.bundle_target_idx);
      }
    }
  } else if (s.use_dino) {
    // The single-bracket solver, on an ICC profile (a bundle's search is the bundle search).
    if (!c.film_prof.data.has_profile) {
      log_error("The parameter search requires a film profile to evaluate intent losses.\n");
      return 1;
    }

    SolverConfig cfg;
    cfg.knee = c.knee;
    cfg.prof = &c.film_prof.data;

    const int max_iters = o.dino_iters;
    const IntentGains user_gains{c.exposure_comp, c.gain_g, c.gain_b};

    // The base E, g, b: a previous run's (<output>.json, with the knobs) or the solver's absolute ones.
    float base_e = 1.0f, base_g = 1.0f, base_b = 1.0f;
    if (s.has_cached_sidecar && !s.cached_sidecar.is_auto) {
      base_e = s.cached_sidecar.base_e;
      base_g = s.cached_sidecar.base_g;
      base_b = s.cached_sidecar.base_b;
    } else {
      // TODO: Run `solve_dinov3_intent` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
      // full-resolution `cfa_export_rgb` buffer once single-target solver baselines are updated.
      const bool solve_ok = scan.cfa
          ? solve_dinov3_intent(s.engine, cfg, c.merged_matrix,
                                negicc::RgbView::from_interleaved_rgb(
                                    scan.rgb3.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                                user_gains, c.pin_mask, max_iters, c.dino_res)
          : solve_dinov3_intent(s.engine, cfg, c.merged_matrix, (const uint16_t (*)[4])proc->imgdata.image,
                                proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                user_gains, c.pin_mask, max_iters, c.dino_res);
      if (!solve_ok) {
        log_error("DINOv3 intent solver failed.\n");
        return 1;
      }


      std::string pins_str = "none";
      if (c.pin_mask != 0) {
        pins_str = "";
        if (c.pin_mask & PIN_E) pins_str += "E ";
        if (c.pin_mask & PIN_G) pins_str += "g ";
        if (c.pin_mask & PIN_B) pins_str += "b";
        if (!pins_str.empty() && pins_str.back() == ' ') pins_str.pop_back();
      }
      log_info("\nDINOv3 solver: grid %dx%d (%dx%d), backend %s, E_c %.3f, bounds [%.3f, %.3f], max iters %d, "
               "pins %s\n",
             c.dino_res.grid_w, c.dino_res.grid_h, c.dino_res.target_w, c.dino_res.target_h,
             s.engine.backend_name(), c.dino_res.e_center, c.dino_res.bounds_lo, c.dino_res.bounds_hi,
             max_iters, pins_str.c_str());
      if (c.dino_res.anchor_clamped) {
        log_warn("ANCHOR_CLAMPED (median L* 50 not reached within E_c 2^[-4, 4])\n");
      }
      log_info(" it      E      g      b   resid    L_L    L_a    L_b  clip%%  P_clip  P_anch       J\n");
      for (const auto& it : c.dino_res.iterations) {
        if (it.residual < 0.0f) {
          log_info("  %c  %5.3f  %5.3f  %5.3f       -  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
                 it.tag, it.e, it.g, it.b, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
                 it.penalty_clip, it.penalty_anchor, it.objective);
        } else {
          log_info("  %c  %5.3f  %5.3f  %5.3f  %6.3f  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
                 it.tag, it.e, it.g, it.b, it.residual, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
                 it.penalty_clip, it.penalty_anchor, it.objective);
        }
      }
      log_info("Status: %s after %d inferences (best it %d, resid %.3f)\n",
             c.dino_res.status.c_str(), c.dino_res.total_inferences, c.dino_res.best_iteration,
             c.dino_res.best_residual);
      log_info("Solved (relative): E %.3f g %.3f b %.3f\n",
             c.dino_res.solved_relative.e, c.dino_res.solved_relative.g, c.dino_res.solved_relative.b);
      log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g\n", c.dino_res.solved_absolute.e,
               c.dino_res.solved_absolute.g, c.dino_res.solved_absolute.b);
      if (c.dino_res.solved_relative.g < 0.5f || c.dino_res.solved_relative.g > 2.0f ||
          c.dino_res.solved_relative.b < 0.5f || c.dino_res.solved_relative.b > 2.0f) {
        log_warn("GAIN_RANGE (relative g or b outside [0.5, 2.0])\n");
      }
      if (c.dino_res.best_iteration >= 0 && (size_t)c.dino_res.best_iteration + 1 < c.dino_res.iterations.size()) {
        const auto& bit = c.dino_res.iterations[c.dino_res.best_iteration + 1];
        float log2_ratio = (bit.e > 0.0f) ? std::log2(bit.e) : 0.0f;
        log_info("Penalties: clip %.3f (%.2f%% %s 2.50%%), anchor %.3f (log2(E/E_c) %+.3f)\n",
               bit.penalty_clip, bit.clip_pct, bit.clip_pct <= 2.5f ? "<=" : ">", bit.penalty_anchor, log2_ratio);
      }
      log_info("Timings (ms): views %.1f, render %.1f, infer %.1f, residual %.1f, search %.1f\n\n",
             c.dino_res.timing_views_ms, c.dino_res.timing_render_ms, c.dino_res.timing_infer_ms,
             c.dino_res.timing_residual_ms, c.dino_res.timing_search_ms);
      base_e = c.dino_res.solved_absolute.e;
      base_g = c.dino_res.solved_absolute.g;
      base_b = c.dino_res.solved_absolute.b;
    }
    float final_e = base_e, final_g = base_g, final_b = base_b;
    if (o.knobs_given) {
      scan.knob_base[0] = base_e;
      scan.knob_base[1] = base_g;
      scan.knob_base[2] = base_b;
      if (!solve_grey_knobs(c.film_prof.data, base_e, base_g, base_b, o.brightness, o.tint, o.warmth, scan.knob_res)) {
        log_error("Failed to solve grey-point knobs\n");
        return 1;
      }
      scan.has_knob_res = true;
      final_e = scan.knob_res.e;
      final_g = scan.knob_res.g;
      final_b = scan.knob_res.b;
      log_knobs();
    }
    c.exposure_comp = final_e;
    c.gain_g = final_g;
    c.gain_b = final_b;
    adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff, c.exposure_comp, c.gain_g, c.gain_b,
                             c.profile_film_base_rgb, unit_fb, c.merged_matrix);
  } else if (s.is_bundle) {
    c.bundle_target_idx = s.manual_target_idx;
    log_info("Selected target: %s\n", bundle.targets[c.bundle_target_idx].display_name.c_str());
    c.bundle_target_name = bundle.targets[c.bundle_target_idx].name;
    c.film_prof = bundle.targets[c.bundle_target_idx].film_prof;
    c.film_prof.rebind_pointers();
    // The given values are the station's.
    double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    if (scan.has_knob_res) {
      apply_station(scan.knob_res.e, scan.knob_res.g, scan.knob_res.b);
      log_knobs();
    } else {
      apply_station(o.exposure_comp, c.gain_g, c.gain_b);
    }
  } else if (o.knobs_given) {
    // A single ICC profile with given gains: the knobs move the given (absolute) E, g and b through that profile, as
    // they move the solver's result above.
    if (!c.film_prof.data.has_profile) {
      log_error("--brightness, --tint and --warmth need the film profile to solve the grey point\n");
      return 1;
    }
    scan.knob_base[0] = c.exposure_comp;
    scan.knob_base[1] = c.gain_g;
    scan.knob_base[2] = c.gain_b;
    if (!solve_grey_knobs(c.film_prof.data, c.exposure_comp, c.gain_g, c.gain_b, o.brightness, o.tint, o.warmth,
                          scan.knob_res)) {
      log_error("Failed to solve grey-point knobs\n");
      return 1;
    }
    scan.has_knob_res = true;
    c.exposure_comp = scan.knob_res.e;
    c.gain_g = scan.knob_res.g;
    c.gain_b = scan.knob_res.b;
    adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff, c.exposure_comp, c.gain_g, c.gain_b,
                             c.profile_film_base_rgb, unit_fb, c.merged_matrix);
    log_knobs();
  }

  if (s.is_bundle && c.bundle_target_idx >= 0) {
    c.embedded_icc = bundle.targets[c.bundle_target_idx].icc_bytes;
  }
  // A print: the conversion renders the print profile at the print's mid grey A (its search's or a reused run's).
  // Given values do not apply with --print (parse_options), so every print has its search's A, g and b.
  if (s.is_bundle && c.bundle_target_idx >= 0 && !s.print_profs.empty()) {
    c.film_prof = s.print_profs[c.bundle_target_idx];
    c.film_prof.rebind_pointers();
    c.embedded_icc = s.print_iccs[c.bundle_target_idx];
    if (!std::isfinite(s.print_a) || !c.station_params) {
      log_error("Internal error: a print without its search's A, g and b\n");
      return 1;
    }
    // The print's g and b in its input curves, after the knee (dL_c = -log10(unknee(x) g_c / n0)), and the matrix at E
    // alone, so that film at the base's density never passes the knee whatever the gains: with them in the matrix a
    // gain of 1.2 put the base past the knee (DESIGN.md, "The print's exposure and its curves"). The CLF applies them
    // the same way.
    const double gains[3] = {1.0, (double)(*c.station_params)[1], (double)(*c.station_params)[2]};
    negicc::candidate_matrix(station_base, (*c.station_params)[0], 1.0f, 1.0f, c.merged_matrix);
    s.print_cont.set_curves_at(s.print_a, c.film_prof, gains);
    set_print_icc_curves(c.embedded_icc, s.print_cont.curves_at(s.print_a, gains));
  }
  // A print keeps no highlight shoulder: the paper's own shoulder rolls the highlights off to its white.
  if (s.is_bundle && c.bundle_target_idx >= 0 && c.station_params && s.print_id.empty()) {
    // The highlight roll-off of the conversion, as the bracket search measures it at its final parameters (on the
    // half-size frame), so given values (cli_equivalent) reproduce it: Auto's result and given values were measured
    // in decode.
    if (!scan.has_rolloff) {
      log_error("Internal error: the roll-off was not measured\n");
      return 1;
    }
    c.shoulder.x_hi = scan.rolloff_x_hi;
    const float q995 = scan.rolloff_q995;
    c.shoulder.has_q995 = true;
    c.shoulder.q995 = q995;
    if (c.shoulder.active()) {
      c.shoulder.source = "rolloff";
      negicc::PipelineConfig pc{};
      negicc::set_shoulder(pc, c.shoulder.x_hi);
      c.shoulder.has_knee = true;
      c.shoulder.knee = pc.shoulder_x_k;
    }
    if (!quiet) log_info("Roll-off: q995 %.3f, x_hi %.3f (sRGB output only)\n", q995, c.shoulder.x_hi);
  }
  return 0;
}

// One candidate's tone stage clip check, as convert makes it (the sRGB encoding): its sample of the image the export
// converts (tone_clip_sample, once), its shares at white and at black before the stage, the plan's curve
// (tone_plan_for_image) and the shares it counted with that curve. shares: white before, black before, white after,
// black after.
static void candidate_tone_check(const Options& o, const Setup& s, Scan& scan, const Conversion& cc, ToneCheck& t,
                                 double shares[4]) {
  LibRaw* tp = scan.proc.get();
  const uint16_t* rgb = !scan.rgb3.empty() ? scan.rgb3.data() : &tp->imgdata.image[0][0];
  const int stride = !scan.rgb3.empty() ? 3 : 4;
  const int iw = tp->imgdata.sizes.iwidth, ih = tp->imgdata.sizes.iheight;
  const negicc::PipelineConfig pc = make_pipeline_config(cc.film_prof.data, cc.knee, cc.shoulder.x_hi, ENCODE_SRGB);
  const std::vector<negicc::Rgb> sample = tone_clip_sample(pc, cc.merged_matrix, rgb, iw, ih, stride);
  t = ToneCheck{};
  t.valid = true;
  tone_clip_shares(pc, sample, nullptr, &t.white_before, &t.black_before);
  t.curve = s.tone_curve;
  double wa = 0.0, ka = 0.0;
  const ToneStagePlan& plan = s.tone_stage;
  if (plan.planned && !plan.preview.empty()) {
    t.curve = tone_plan_for_image(s.tone_plan.render, plan.pred_mean[ToneHead::kL2], plan.pred_mean[ToneHead::kL98],
                                  plan.pred_mean[ToneHead::kR9010], pc, sample, t.white_before, t.black_before,
                                  s.tone_plan.curve, s.tone_plan.targets, &t.targets, &t.share, &t.shadow_share,
                                  o.tone_fallback, &t.fell_back, &wa, &ka);
  } else {
    tone_clip_shares(pc, sample, &t.curve, &wa, &ka);
  }
  shares[0] = t.white_before;
  shares[1] = t.black_before;
  shares[2] = wa;
  shares[3] = ka;
}

// One candidate's tone stage clip check where its conversion c has the knobs and this run planned the stage
// (Scan::has_plain): made on the same candidate without the knobs (from c0, the conversion before solve_one), so that
// the curve, and the choice between the result and its shadow alternative, are those of the run without them, and the
// knobs act under the curve as on a replay of that run's record. Otherwise the check is c's own.
static bool knob_free_tone_check(const Options& o, Setup& s, Scan& scan, const Conversion& c0, const Conversion& c,
                                 ToneCheck& t, double shares[4]) {
  if (!scan.has_knob_res || !scan.has_plain) {
    candidate_tone_check(o, s, scan, c, t, shares);
    return true;
  }
  ParsedFilmProfile& prof = s.bundle.targets[c.bundle_target_idx].film_prof;
  ParsedFilmProfile knob_prof;
  if (scan.has_plain_prof) {
    knob_prof = prof;
    prof = scan.plain_prof;
    prof.rebind_pointers();
  }
  const float x_hi = scan.rolloff_x_hi, q995 = scan.rolloff_q995;
  const double print_a = s.print_a;
  scan.rolloff_x_hi = scan.plain_x_hi;
  scan.rolloff_q995 = scan.plain_q995;
  if (!s.print_id.empty()) s.print_a = scan.knob_base_a;
  scan.has_knob_res = false;
  Conversion cp = c0;
  cp.film_prof.rebind_pointers();
  const bool ok = solve_one(o, s, scan, cp, true) == 0;
  if (ok) {
    candidate_tone_check(o, s, scan, cp, t, shares);
    t.without_knobs = true;
    log_info("Tone stage check without the knobs (-E %.9g --gain_g %.9g --gain_b %.9g --target %d): the knobs act "
             "under its curve\n", cp.station_params ? (*cp.station_params)[0] : 0.0f,
             cp.station_params ? (*cp.station_params)[1] : 0.0f, cp.station_params ? (*cp.station_params)[2] : 0.0f,
             cp.bundle_target_idx);
  }
  scan.has_knob_res = true;
  s.print_a = print_a;
  scan.rolloff_x_hi = x_hi;
  scan.rolloff_q995 = q995;
  if (scan.has_plain_prof) {
    prof = knob_prof;
    prof.rebind_pointers();
  }
  return ok;
}

// solve, and with a shadow alternative (Scan::shadow_candidate) the choice between it and the search's result: the
// alternative is taken only when its shares at white and at black on the tone stage's clip check sample, before and
// after the stage, are each no higher than the result's and at least one is lower. Each candidate's sample is
// converted once; the kept one's check goes to convert (Scan::tone_check), which does not make it again.
int solve(const Options& o, Setup& s, Scan& scan, Conversion& c) {
  scan.tone_check = ToneCheck{};
  const std::shared_ptr<ShadowCandidate> bp = scan.shadow_candidate;
  scan.shadow_candidate.reset();
  // With the knobs on a run that planned the tone stage, its check is made without them (knob_free_tone_check).
  const bool tone_here = s.has_tone && o.colorspace == "srgb" && !s.density;
  if (!bp && !(tone_here && scan.has_knob_res && scan.has_plain)) return solve_one(o, s, scan, c);
  ShadowChoice& ch = scan.shadow_choice;
  const Conversion c0 = c;
  int r = solve_one(o, s, scan, c);
  if (r) return r;
  if (!bp) {
    double shares[4];
    if (!knob_free_tone_check(o, s, scan, c0, c, scan.tone_check, shares)) return 1;
    return 0;
  }
  if (!(tone_here && s.print_id.empty())) {
    ch.not_checked = "no tone stage on this conversion";
    return 0;
  }
  const double t0 = omp_get_wtime();
  ToneCheck ta, tb;
  if (!knob_free_tone_check(o, s, scan, c0, c, ta, ch.result)) return 1;
  ShadowCandidate a;
  capture_candidate(s, scan, a);
  a.prof0 = bp->result_prof0;
  a.prof0.rebind_pointers();
  switch_candidate(s, scan, a, *bp);
  Conversion cb = c0;
  cb.film_prof.rebind_pointers();
  log_info("The shadow alternative (%s), converted to compare:\n", bp->auto_res.target_name.c_str());
  const bool b_ok = solve_one(o, s, scan, cb) == 0 && knob_free_tone_check(o, s, scan, c0, cb, tb, ch.alternative);
  bool no_higher = b_ok, lower = false;
  for (int i = 0; b_ok && i < 4; ++i) {
    no_higher = no_higher && ch.alternative[i] <= ch.result[i];
    lower = lower || ch.alternative[i] < ch.result[i];
  }
  ch.checked = b_ok;
  ch.kept_alternative = no_higher && lower;
  if (ch.kept_alternative) {
    c = cb;
    c.film_prof.rebind_pointers();
    scan.tone_check = tb;
  } else {
    switch_candidate(s, scan, *bp, a);
    scan.tone_check = ta;
    if (c.station_params) {   // the conversion is the result's again, not the alternative's solved above
      log_info("Solved (the result, kept): -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n",
               (*c.station_params)[0], (*c.station_params)[1], (*c.station_params)[2], c.bundle_target_idx);
    }
  }
  ch.check_ms = (omp_get_wtime() - t0) * 1000.0;
  log_info("Shadow alternative %s: %s; at white %.4f%% -> %.4f%% and at black %.4f%% -> %.4f%% (the result), "
           "%.4f%% -> %.4f%% and %.4f%% -> %.4f%% (the alternative) on the clip check's sample; %.0f ms\n",
           bp->auto_res.target_name.c_str(), ch.kept_alternative ? "taken" : "not taken", 100.0 * ch.result[0],
           100.0 * ch.result[2], 100.0 * ch.result[1], 100.0 * ch.result[3], 100.0 * ch.alternative[0],
           100.0 * ch.alternative[2], 100.0 * ch.alternative[1], 100.0 * ch.alternative[3], ch.check_ms);
  return 0;
}

int convert(const Options& o, const Setup& s, Scan& scan, Conversion& c, Backend& backend) {
  OutputSpec& out = c.out;
  if (s.density) {
    // The density output (the print declarations above): the film base's merged matrix as the station builds it
    // (crosstalk correction and film base scale, the scan's exposure ratio with the bundle's own base), E = g = b = 1,
    // then the log and the measure's matrix. No knee, no film profile.
    out.encoding = ENCODE_DENSITY;
    const double t_start = omp_get_wtime();
    double scan_shutter_s = kDefaultScanShutterS, scan_iso = kDefaultScanIso;
    scan.scan_exposure(*scan.proc, scan_shutter_s, scan_iso);
    DensityOutput& d = c.density;
    compute_station_merged_matrix(s.bundle, scan_shutter_s, scan_iso, scan.film_base_rgb(),
                                  scan.film_base_rgb() != nullptr, d.merged, nullptr, nullptr);
    d.base_level = s.bundle.normalization_target / 65535.0;
    d.encoding = density_encoding(o.output_dpx, o.density_measure);
    const DensityMatrix& dm = s.bundle.density_matrix[o.density_measure];
    const RgbInput img = scan.image();
    density_encode_in_place(img.pixels, img.width, img.height, img.channels, d.merged, d.base_level, dm.m, d.encoding,
                            d.clip);
    const double px = (double)std::max<uint64_t>(1, d.clip.pixels);
    // The film base is the zero of a density file. Read from a capture or the scan's rebate it is measured; taken from
    // the thinnest film of a frame with no unexposed film in view, or from the bundle's roll, it is an estimate, and
    // every density is off by its error, differently per channel and per frame.
    // The remedy: a capture of unexposed film (--film_base) on TriLED, where the input takes one (triled_base_remedy);
    // on white light, which takes no capture, a crop that shows the rebate.
    const char* remedy = s.white_light ? "crop so that the frame shows some rebate (--roi)" : triled_base_remedy(o);
    if (!scan.film_base.found || scan.film_base.source == "bundle") {
      log_warn("the film base is the profile bundle's own, not this roll's: the densities are relative to another "
               "roll's base; %s\n", remedy);
    } else if (scan.film_base.source == "statistics") {
      log_warn("the film base was estimated from the thinnest film on this frame, which shows no rebate: the densities "
               "are too low wherever the frame has no unexposed film; %s\n", remedy);
    }
    if (scan.print_floor_applied && scan.print_floor_k > 1.0f) {
      log_warn("with --print, the density file is written against the film base raised %.3fx (the base as found: %s) "
               "to be no darker than the picture's thinnest film, as the print takes it, so that the CLF prints the "
               "picture's print; the same frame without --print is written against the base as found (the sidecar's "
               "film_base.print_floor)\n",
               scan.print_floor_k, scan.print_floor_found_source.c_str());
    }
    {
      // Much of the frame thinner than the base, or at the density floor: the film base is probably wrong.
      double low = 0.0, floor = 0.0;
      for (int k = 0; k < 3; ++k) {
        low = std::max(low, (double)d.clip.low[k] / px);
        floor = std::max(floor, (double)d.clip.floor[k] / px);
      }
      if (low > 0.05) {
        log_warn("%.1f%% of the output's pixels are thinner than the film base by more than the encoding holds. The "
                 "count is over the whole output: bare light beside the film counts too, so crop to the frame "
                 "(--roi) before reading it as the film. If the output is the frame, the film base is probably too "
                 "dark (see the sidecar's film_base)\n", 100.0 * low);
      }
      if (floor > 0.05) {
        log_warn("%.1f%% of the pixels are at the density floor (%.1f): the film base is probably too bright, or the "
                 "scan is underexposed\n", 100.0 * floor, -std::log10(kDensityMinT));
      }
    }
    log_info("Density (%s, %d bits, code = %d + %d k D, k = %.2f %.2f %.2f): clamped below 0 %.3f%% %.3f%% %.3f%%, "
             "above %d %.3f%% %.3f%% %.3f%% of the pixels (R G B)\n",
             density_measure_name(o.density_measure), d.encoding.bits, d.encoding.offset, d.encoding.scale,
             d.encoding.k[0], d.encoding.k[1], d.encoding.k[2],
             100.0 * d.clip.low[0] / px, 100.0 * d.clip.low[1] / px, 100.0 * d.clip.low[2] / px, d.encoding.max_code(),
             100.0 * d.clip.high[0] / px, 100.0 * d.clip.high[1] / px, 100.0 * d.clip.high[2] / px);
    const double t_end = omp_get_wtime();
    scan.t_full_conversion_ms = (t_end - t_start) * 1000.0;
    log_info("Processing time: %f\n", t_end - t_start);
    (void)backend;
    return 0;
  }
  if (!o.colorspace.empty()) {
    const std::string& colorspace_str = o.colorspace;
    if (colorspace_str == "srgb") {
      out.encoding = ENCODE_SRGB;
      out.attach_profile = "srgb";
      log_info("Using elle sRGB TRC as output profile\n");
    } else if (colorspace_str == "srgb-g10" || colorspace_str == "srgb-linear") {
      out.encoding = ENCODE_SRGB_LINEAR;
      out.attach_profile = "srgb-g10";
      log_info("Using elle sRGB Linear (Gamma = 1.0) as output profile\n");
    } else {
      std::string resolved_custom = resolve_profile_path(colorspace_str);
      out.custom = cmsOpenProfileFromFile(resolved_custom.c_str(), "r");
      if (!out.custom) {
        log_error("Cannot read output ICC profile: %s\n", colorspace_str.c_str());
        return 1;
      }
      out.xyz = cmsCreateXYZProfile();   // an RGB profile with a transform from PCS XYZ (prepare checked it)
      out.encoding = ENCODE_CUSTOM;
      out.attach_profile = resolved_custom;
      log_info("Reading output ICC profile: %s\n", resolved_custom.c_str());
    }
  } else if (s.has_profile) {
    out.encoding = ENCODE_NONE;
    if (!s.is_bundle) out.attach_profile = s.film_prof_arg;
  }

  const double t_start = omp_get_wtime();
  // The shared pipeline (color_pipeline.h) with the frozen shoulder: the stage every consumer runs. Only the sRGB
  // encoding applies it.
  const negicc::PipelineConfig export_pc =
      make_pipeline_config(c.film_prof.data, c.knee, c.shoulder.x_hi, out.encoding);
  // The tone stage (tone_stage.h): this run's plan, or the recorded curve on a replay, applied in the
  // conversion kernel (color_pipeline.h apply_tone) between the display value and the sRGB encoding. Under the knobs
  // the curve is the one without them, applied after them: on a replay the recorded curve, as recorded; on a fresh
  // run the plan checked on the conversion without the knobs (solve, knob_free_tone_check), so it equals the knobs on
  // the record of a run without them. A fixed monotone map on top of the knobbed render (zero knobs reproduce the
  // export; a brighter knob gives a brighter result), not re-solved against the knobbed render, which would pull the
  // shadow level back to the prediction and undo the knob.
  scan.tone_applied = false;
  const bool tone_here = s.has_tone && out.encoding == ENCODE_SRGB;
  LibRaw* tp = scan.proc.get();
  uint16_t* rgb = !scan.rgb3.empty() ? scan.rgb3.data() : &tp->imgdata.image[0][0];
  const int stride = !scan.rgb3.empty() ? 3 : 4;
  const ToneStagePlan& plan = s.tone_stage;
  const bool tone_plan_here = tone_here && plan.planned && !plan.preview.empty();
  const int iw = tp->imgdata.sizes.iwidth, ih = tp->imgdata.sizes.iheight;
  if (tone_here) {
    // The clip check on a sparse sample of the input through the kernel, before the one full conversion: the sample
    // is converted once up to the tone curve, and each curve tried is applied to it.
    scan.tone_curve = s.tone_curve;
    scan.tone_share = 1.0;
    scan.tone_shadow_share = 1.0;
    scan.tone_fell_back = false;
    const double t_tone_0 = omp_get_wtime();
    if (scan.tone_check.valid && !scan.tone_check.without_knobs) {
      // solve made this check already, on the same sample, choosing between the result and its shadow alternative.
      scan.tone_clip.white_before = scan.tone_check.white_before;
      scan.tone_clip.black_before = scan.tone_check.black_before;
      scan.tone_curve = scan.tone_check.curve;
      if (tone_plan_here) {
        scan.tone_targets = scan.tone_check.targets;
        scan.tone_share = scan.tone_check.share;
        scan.tone_shadow_share = scan.tone_check.shadow_share;
        scan.tone_fell_back = scan.tone_check.fell_back;
      }
    } else {
      const std::vector<negicc::Rgb> sample = tone_clip_sample(export_pc, c.merged_matrix, rgb, iw, ih, stride);
      tone_clip_shares(export_pc, sample, nullptr, &scan.tone_clip.white_before, &scan.tone_clip.black_before);
      if (scan.tone_check.valid) {
        // solve made the check without the knobs (knob_free_tone_check): its curve, under which the knobs act.
        scan.tone_curve = scan.tone_check.curve;
        if (tone_plan_here) {
          scan.tone_targets = scan.tone_check.targets;
          scan.tone_share = scan.tone_check.share;
          scan.tone_shadow_share = scan.tone_check.shadow_share;
          scan.tone_fell_back = scan.tone_check.fell_back;
        }
      } else if (tone_plan_here) {
        // This run's plan: its moves scaled where they would clip this image at either end (tone_plan_for_image).
        scan.tone_curve = tone_plan_for_image(s.tone_plan.render, plan.pred_mean[ToneHead::kL2],
                                              plan.pred_mean[ToneHead::kL98], plan.pred_mean[ToneHead::kR9010],
                                              export_pc, sample, scan.tone_clip.white_before,
                                              scan.tone_clip.black_before, s.tone_plan.curve, s.tone_plan.targets,
                                              &scan.tone_targets, &scan.tone_share, &scan.tone_shadow_share,
                                              o.tone_fallback, &scan.tone_fell_back);
      }
    }
    const double t_tone_1 = omp_get_wtime();
    const ToneCurve& q = scan.tone_curve;
    const std::vector<float> lut = tone_table(q);
    negicc::PipelineConfig tone_pc = export_pc;
    tone_pc.tone.lut = lut.data();
    tone_pc.tone.n = kToneTableSize;
    backend.convert(out, tone_pc, c.merged_matrix, scan.image());
    const double t_tone_2 = omp_get_wtime();
    // The shares after the stage on every pixel of the inner crop, as the clip check counts them.
    const negicc::InnerCrop tic = negicc::compute_inner_crop(iw, ih, kToneClipCrop);
    long long wa = 0, ka = 0;
#pragma omp parallel for schedule(static) reduction(+ : wa, ka)
    for (int y = tic.y0; y < tic.y0 + tic.h; ++y) {
      for (int x = tic.x0; x < tic.x0 + tic.w; ++x) {
        const uint16_t* px = rgb + ((size_t)y * iw + x) * stride;
        wa += (px[0] == 65535 || px[1] == 65535 || px[2] == 65535);
        ka += (px[0] == 0 || px[1] == 0 || px[2] == 0);
      }
    }
    const long long tn = (long long)tic.w * tic.h;
    if (tn > 0) {
      scan.tone_clip.white_after = (double)wa / tn;
      scan.tone_clip.black_after = (double)ka / tn;
    }
    scan.tone_applied = true;
    log_info("Tone stage: black %.3f white %.3f gamma %.4f contrast %.4f (highlight share %.3f); inner crop at white "
             "%.4f%% -> %.4f%%, at black %.4f%% -> %.4f%%; %.0f ms checking the clipping, %.0f ms converting with the "
             "curve\n",
             q.black, q.white, q.gamma, q.contrast, scan.tone_share, 100.0 * scan.tone_clip.white_before,
             100.0 * scan.tone_clip.white_after, 100.0 * scan.tone_clip.black_before, 100.0 * scan.tone_clip.black_after,
             1000.0 * (t_tone_1 - t_tone_0), 1000.0 * (t_tone_2 - t_tone_1));
  } else {
    backend.convert(out, export_pc, c.merged_matrix, scan.image());
  }
  scan.full_size_conversion = backend.convert_ran;
  if (out.encoding == ENCODE_CUSTOM) {
    cmsCloseProfile(out.xyz);
    cmsCloseProfile(out.custom);
    out.xyz = out.custom = nullptr;
  }
  const double t_end = omp_get_wtime();
  scan.t_full_conversion_ms = (t_end - t_start) * 1000.0;
  log_info("Processing time: %f\n", t_end - t_start);
  return 0;
}

// What the conversion used, logged once per conversion as a block and written to the sidecar as "used", in the
// pipeline's order: the profile (a bundle: its key, where it was read and the camera models it lists), the camera, the
// decode, the flat field, the crop, the crosstalk correction, the film base, and the ICC profiles the picture renders
// through (the bracket's, or a print's). The strings from the input and the command line (the camera, paths, names
// taken from file names) need not be UTF-8: the block and "used" both carry them as the sidecar records them, each
// byte that is not UTF-8 as U+FFFD (utf8_replaced). The bundle's own strings are read as JSON and are UTF-8.
static nlohmann::ordered_json conversion_used(const Options& o, const Setup& s, const Scan& scan, const Conversion& c) {
  using nlohmann::ordered_json;
  const ProfileBundle& bundle = s.bundle;
  ordered_json u;
  log_info("Used:\n");
  const std::string camera_make = utf8_replaced(scan.camera_make), camera_model = utf8_replaced(scan.camera_model);
  const std::string bundle_key = utf8_replaced(bundle.key), bundle_path = utf8_replaced(bundle.path);

  // The camera: the one the bundle records (camera_name, as stored) and this scan's (make and model); a narrow-band
  // bundle also lists the models its crosstalk matrix holds for, compared with the scan's model (decode). A bundle
  // without a matrix (white light) is not compared.
  std::string scan_camera = camera_model;
  if (!camera_make.empty() && camera_model.rfind(camera_make, 0) != 0)
    scan_camera = camera_make + (camera_model.empty() ? "" : " " + camera_model);
  u["camera"] = {{"scan", scan_camera.empty() ? ordered_json(nullptr) : ordered_json(scan_camera)},
                 {"scan_model", camera_model.empty() ? ordered_json(nullptr) : ordered_json(camera_model)},
                 {"bundle", s.is_bundle ? ordered_json(bundle.camera_name) : ordered_json(nullptr)}};
  if (s.is_bundle) {
    const bool built_in = bundle.path.rfind("[built-in] ", 0) == 0;
    u["profile_bundle"] = {{"key", bundle_key},
                           {"path", bundle_path},
                           {"built_in", built_in},
                           {"camera_models", bundle.has_camera_models ? ordered_json(bundle.camera_models)
                                                                      : ordered_json(nullptr)}};
    log_info("  Profile bundle: %s, %s\n", bundle_key.c_str(), built_in ? "built in" : ("from " + bundle_path).c_str());
  } else {
    u["profile_bundle"] = nullptr;
    log_info("  Profile bundle: none%s\n", s.film_prof_arg.empty() ? "" : " (an ICC profile)");
  }
  log_info("  Camera: %s%s\n", scan_camera.empty() ? "the scan states none" : ("this scan's " + scan_camera).c_str(),
           s.is_bundle ? ("; the bundle records " + bundle.camera_name).c_str() : "");
  if (s.is_bundle && !s.white_light) {
    std::string models;
    for (const std::string& m : bundle.camera_models) models += (models.empty() ? "" : ", ") + m;
    u["camera"]["compared"] = scan.camera_compared;
    u["camera"]["listed"] = scan.camera_compared ? ordered_json(scan.camera_listed) : ordered_json(nullptr);
    log_info("  Camera models the crosstalk matrix holds for: %s; %s\n",
             !bundle.has_camera_models ? "no list (a bundle from before camera_models)"
             : models.empty()          ? "none"
                                       : models.c_str(),
             !bundle.has_camera_models    ? "not compared, the bundle's correction applies"
             : camera_model.empty()       ? "the camera is not stated by the input: not compared, the bundle's "
                                            "correction applies"
             : scan.camera_listed         ? ("this scan's model " + camera_model + " is on the list").c_str()
                                          : ("this scan's model " + camera_model + " is not on the list").c_str());
  } else {
    u["camera"]["compared"] = false;
    u["camera"]["listed"] = nullptr;
  }

  std::string decode, decode_text;
  if (o.input_mode == InputMode::kRgbFile) {
    decode = s.camera_rgb_input ? "camera_rgb" : "rec2020";
    decode_text = s.camera_rgb_input ? "a TIFF of linear camera RGB" : "an RGB file converted to linear Rec.2020";
  } else if (o.input_mode == InputMode::kThreeShot) {
    decode = "camera_rgb";
    decode_text = "camera RGB of three captures, one channel from each";
  } else if (o.files.size() == 4) {
    decode = "camera_rgb";
    decode_text = "camera RGB of the pixel shift's four captures, merged";
  } else if (s.raw_decode == RawDecode::kCameraRgb) {
    decode = "camera_rgb";
    decode_text = "camera RGB (black subtracted, no white balance)";
  } else if (s.raw_decode == RawDecode::kWhiteLight) {
    decode = "white_light_rec2020";
    decode_text = "LibRaw's white-light decode to linear Rec.2020";
  } else {
    decode = "rec2020";
    decode_text = "LibRaw's decode to linear Rec.2020";
  }
  u["decode"] = {{"space", decode}, {"input", scan.input_source}};
  log_info("  Decode: %s; input %s\n", decode_text.c_str(), scan.input_source.c_str());
  // Where the full-size steps ran: on the device where make_backend allows it and one runs this build's kernels.
  u["full_size"] = {{"decode", scan.full_size_decode}, {"conversion", scan.full_size_conversion}};
  log_info("  Full-size steps: decode on the %s, conversion on the %s\n", scan.full_size_decode,
           scan.full_size_conversion);

  if (o.flat.empty()) {
    u["flat_field"] = nullptr;
    log_info("  Flat field: none\n");
  } else {
    // The id is the flat field's record's, else its folder's name.
    const std::string flat = utf8_replaced(o.flat), id = utf8_replaced(s.flat_field ? s.flat_field->id() : "");
    u["flat_field"] = {{"path", flat}, {"id", id}};
    log_info("  Flat field: %s (id %s)\n", flat.c_str(), id.empty() ? "none recorded" : id.c_str());
  }

  // The crop in sensor pixels: --roi (given, or --roi auto's), else a single RAW's or three captures' inset crop
  // unless --no_crop, else the whole image; clamped to the image as the decode clamps it.
  int crop[4] = {0, 0, scan.orig_w, scan.orig_h};
  std::string crop_source = "whole image";
  const bool inset_applies = !o.no_crop && scan.has_inset &&
                             (o.input_mode == InputMode::kThreeShot ||
                              (o.files.size() == 1 &&
                               (o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng)));
  if (scan.has_roi) {
    std::copy(scan.roi, scan.roi + 4, crop);
    crop_source = !o.roi_auto ? "given" : scan.frame_det.conf < kLowFrameConfidence ? "whole image (--roi auto found "
                                                                                      "no frame)"
                                                                                    : "detected";
  } else if (inset_applies) {
    std::copy(scan.inset, scan.inset + 4, crop);
    crop_source = "the RAW's inset crop";
  }
  crop[2] = std::min(crop[2], scan.orig_w - crop[0]);
  crop[3] = std::min(crop[3], scan.orig_h - crop[1]);
  u["crop"] = {{"region", {crop[0], crop[1], crop[2], crop[3]}}, {"source", crop_source}};
  log_info("  Crop: x %d y %d, %dx%d sensor pixels, %s\n", crop[0], crop[1], crop[2], crop[3], crop_source.c_str());

  if (!s.has_crosstalk) {
    const char* why = s.white_light ? "not applicable (white-light decode to linear Rec.2020)"
                                    : "none (no TriLED bundle and no -r, -g or -b)";
    u["crosstalk"] = {{"matrix", nullptr}, {"rows", nullptr}, {"off", why}};
    log_info("  Crosstalk correction: %s\n", why);
  } else {
    double m[9];
    const std::vector<float>* rows[3] = {&s.r_coeff, &s.g_coeff, &s.b_coeff};
    for (int i = 0; i < 9; ++i) m[i] = s.is_bundle ? bundle.crosstalk_matrix_d[i] : (double)(*rows[i / 3])[i % 3];
    ordered_json mj = ordered_json::array(), rj = ordered_json::array();
    for (int i = 0; i < 9; ++i) mj.push_back(m[i]);
    for (int r = 0; r < 3; ++r) rj.push_back(s.crosstalk_row_source[r]);
    std::string off = utf8_replaced(s.crosstalk_off);   // it can name the scan's camera
    if (off.empty() && o.input_mode == InputMode::kThreeShot && s.is_bundle)
      off = "three captures, one per light, have no crosstalk between the lights";
    u["crosstalk"] = {{"matrix", mj}, {"rows", rj}, {"off", off.empty() ? ordered_json(nullptr) : ordered_json(off)}};
    const std::string* src = s.crosstalk_row_source;
    std::string from;
    if (src[0] == src[1] && src[1] == src[2]) {
      from = src[0] == "bundle" ? "the bundle's" : src[0] == "given" ? "given (-r, -g, -b)" : "the identity";
    } else {
      from = "R " + src[0] + ", G " + src[1] + ", B " + src[2];
    }
    if (!off.empty()) from += (s.crosstalk_off.empty() ? ": " : "; the bundle's is off: ") + off;
    log_info("  Crosstalk correction: %s\n    [%.9g %.9g %.9g; %.9g %.9g %.9g; %.9g %.9g %.9g]\n", from.c_str(), m[0],
             m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
  }

  if (!s.is_bundle) {
    u["film_base"] = nullptr;
    log_info("  Film base: none (no profile bundle)\n");
  } else {
    const bool found = scan.film_base.found;
    const double* fb = found ? scan.film_base.rgb : bundle.film_base_avg_d;
    const std::string src = found ? scan.film_base.source : "bundle";
    u["film_base"] = {{"rgb", {fb[0], fb[1], fb[2]}}, {"source", src}};
    log_info("  Film base: R=%.1f G=%.1f B=%.1f (%s%s)\n", fb[0], fb[1], fb[2], src.c_str(),
             found ? "" : ", at the bundle's own exposure");
  }

  ordered_json bracket = nullptr, print = nullptr;
  const bool has_target = s.is_bundle && c.bundle_target_idx >= 0 && c.bundle_target_idx < (int)bundle.targets.size();
  if (!s.has_profile) {
    log_info("  Profile: none\n");
  } else if (!s.is_bundle) {
    const std::string icc = utf8_replaced(s.film_prof_arg);
    bracket = {{"icc", icc}};
    log_info("  ICC profile: %s\n", icc.c_str());
  } else if (s.density) {
    log_info("  Bracket ICC: none (a density output, -P %s, through the bundle's density matrix)\n",
             utf8_replaced(o.colorspace).c_str());
    if (!o.prints.empty()) {
      std::string list;
      ordered_json clfs = ordered_json::array();
      for (const std::string& p : o.prints) {
        list += (list.empty() ? "" : ", ") + utf8_replaced(p);
        clfs.push_back(utf8_replaced(p));
      }
      print = {{"clfs", clfs}};
      log_info("  Prints: a CLF for each of %s\n", list.c_str());
    }
  } else if (has_target) {
    const BundleTarget& t = bundle.targets[c.bundle_target_idx];
    const std::string file = t.icc_file.empty() ? "inline in the bundle" : t.icc_file;
    if (s.print_id.empty()) {
      bracket = {{"target", t.name}, {"icc", file}};
      log_info("  Bracket: %s, ICC %s\n", t.name.c_str(), file.c_str());
    } else {
      bracket = {{"target", t.name}, {"icc", nullptr}};
      log_info("  Bracket: %s (the print's; its chart ICC %s is not used)\n", t.name.c_str(), file.c_str());
    }
  }
  if (!s.print_id.empty() && !s.density) {
    std::string icc;
    if (s.print_from_bundle) {
      for (const BundlePrint& p : bundle.prints)
        if (p.id == s.print_id) icc = p.icc_file.empty() ? "inline in the bundle" : p.icc_file;
    }
    const std::string id = utf8_replaced(s.print_id), file = utf8_replaced(s.print_file);
    print = {{"id", id},
             {"icc", s.print_from_bundle ? ordered_json(icc) : ordered_json(nullptr)},
             {"computed_from", s.print_from_bundle ? ordered_json(nullptr) : ordered_json(file)}};
    log_info("  Print ICC: %s (%s), %s\n", id.c_str(), utf8_replaced(s.print_name).c_str(),
             s.print_from_bundle ? ("the bundle's " + icc).c_str() : ("computed by neg_process from " + file).c_str());
  }
  u["bracket"] = bracket;
  u["print"] = print;
  return u;
}

int write(const Options& o, const Setup& s, Scan& scan, Conversion& c) {
  LibRaw* proc = scan.proc.get();
  const std::string& attach_profile = c.out.attach_profile;
  const OutputEncoding encoding = c.out.encoding;
  const int out_w = proc->imgdata.sizes.iwidth, out_h = proc->imgdata.sizes.iheight;

  const nlohmann::ordered_json used = conversion_used(o, s, scan, c);

  const double t_tiff_0 = omp_get_wtime();
  int ret = 0;
  std::string density_json;
  if (s.density) {
    // The density file and its metadata (the print declarations above): the same record in the file and in the sidecar.
    DensityRecord rec;
    rec.measure = o.density_measure;
    rec.encoding = c.density.encoding;
    rec.container = o.output_dpx ? "dpx" : "tiff";
    rec.bundle_key = s.bundle.key;
    const double* fb = scan.film_base_rgb() ? scan.film_base_rgb() : s.bundle.film_base_avg_d;
    for (int k = 0; k < 3; ++k) rec.film_base_rgb[k] = fb[k];
    rec.film_base_source = scan.film_base.found ? scan.film_base.source : "bundle";
    rec.matrix = &s.bundle.density_matrix[o.density_measure];
    rec.bundle = &s.bundle;
    rec.clip = &c.density.clip;
    if (c.station_params) {
      rec.has_balance = true;
      rec.balance_e = (*c.station_params)[0];
      rec.balance_g = (*c.station_params)[1];
      rec.balance_b = (*c.station_params)[2];
      rec.balance_target = c.bundle_target_name;
    }
    rec.print_search = s.density_print_search;
    rec.tool_version = std::string("neg_process ") + neg_process_version();
    // The prints of the file (the print declarations above), for the print materials --print names: how the frame
    // prints on each, recorded in the metadata, and the CLFs that go beside it. Without --print there are none.
    const DensityPrints prints = density_prints(rec, o.output, o.prints);
    // A print asked for and not made fails the run, as on a picture output (and writes nothing, below).
    if (!prints.error.empty()) log_error("--print: %s\n", prints.error.c_str());
    rec.print_json = prints.metadata_json;
    density_json = density_metadata_json(rec);
    DensityFileInfo info;
    info.make = proc->imgdata.idata.make;
    info.model = proc->imgdata.idata.model;
    info.software = rec.tool_version;
    info.timestamp = (long long)proc->imgdata.other.timestamp;
    const RgbInput img = scan.image();
    // The CLFs first: one that cannot be written stops the run before the density file exists, and a density file
    // that cannot be written takes its CLFs with it. Nothing is left half done.
    ret = prints.error.empty() ? write_print_clfs(rec, prints) : 1;
    if (ret == 0) {
      log_info("Writing %s '%s'\n", o.output_dpx ? "DPX" : "TIFF", o.output.c_str());
      ret = o.output_dpx ? write_density_dpx(o.output, img.pixels, img.width, img.height, img.channels, density_json,
                                             c.density.encoding, o.density_measure, info)
                         : write_density_tiff(o.output, img.pixels, img.width, img.height, img.channels, density_json,
                                              info);
      if (ret != 0) remove_print_clfs(prints);
    }
  } else {
    log_info("Writing TIFF '%s'\n", o.output.c_str());
    ret = write_tiff(proc, attach_profile, o.output, s.is_bundle ? c.embedded_icc : std::vector<uint8_t>{},
                     scan.rgb3.empty() ? nullptr : scan.rgb3.data());
  }
  scan.t_tiff_write_ms = (omp_get_wtime() - t_tiff_0) * 1000.0;

  if (ret != 0 || o.no_metadata) {
    unlink((o.output + ".json").c_str());
    return ret;
  }

  std::string cs_name = "camera-linear";
  if (encoding == ENCODE_SRGB) {
    cs_name = "srgb";
  } else if (encoding == ENCODE_SRGB_LINEAR) {
    cs_name = "srgb-linear";
  } else if (encoding == ENCODE_CUSTOM) {
    cs_name = attach_profile;
  } else if (encoding == ENCODE_NONE) {
    cs_name = attach_profile.empty() ? "camera-linear" : attach_profile;
  } else if (encoding == ENCODE_DENSITY) {
    cs_name = o.colorspace;
  }

  std::string roi_source = "none";
  if (o.roi_auto) {
    roi_source = (scan.frame_det.conf < kLowFrameConfidence) ? "auto_fallback" : "auto";
  } else if (scan.has_roi) {
    roi_source = "cli";
  }

  // A bundle conversion takes the film base read from the scan (Scan::film_base), else the bundle's.
  const bool film_base_applied = s.is_bundle && scan.film_base.found;

  Sidecar sc;
  sc.output_path = o.output;
  sc.argc = o.argc;
  sc.argv = o.argv;
  sc.proc = proc;
  sc.files = o.files;
  sc.file_fingerprints = &s.input_fingerprints;
  sc.orig_w = scan.orig_w;
  sc.orig_h = scan.orig_h;
  sc.has_roi = scan.has_roi;
  sc.roi = scan.roi;
  sc.roi_source = roi_source;
  sc.frame_det = o.roi_auto ? &scan.frame_det : nullptr;
  // The format the base detection used, given or --roi auto's default, as the frame (--roi) is recorded given or found:
  // the replay reads the base around the same format, and its own replay command is the same.
  if (scan.base_det_format != FILM_FORMAT_AUTO) sc.film_format = film_format_name(scan.base_det_format);
  sc.rot_cw = scan.rot_cw;
  sc.hflip = scan.hflip;
  sc.vflip = scan.vflip;
  sc.out_w = out_w;
  sc.out_h = out_h;
  sc.film_base_applied = film_base_applied;
  sc.film_base_rgb.assign(c.sidecar_film_base_rgb.begin(), c.sidecar_film_base_rgb.end());
  sc.profile_film_base_rgb = c.sidecar_profile_film_base_rgb;
  sc.film_base_source = film_base_applied ? scan.film_base.source : c.film_base_source;
  sc.print_floor_applied = scan.print_floor_applied;
  sc.print_floor_found_source = scan.print_floor_found_source;
  sc.print_floor_k = scan.print_floor_k;
  sc.knob_base_a = scan.knob_base_a;
  sc.print_floor_frame = scan.print_floor_frame;
  sc.print_floor_frame_conf = scan.print_floor_frame_conf;
  for (int k = 0; k < 3; ++k) sc.print_floor_found_rgb[k] = scan.print_floor_found_rgb[k];
  sc.film_base_detection = s.is_bundle ? &scan.film_base : nullptr;
  sc.film_base_capture_json = scan.base_capture_json;
  sc.density_json = density_json;
  // As the sidecar's writer writes it: a byte that is not UTF-8 as U+FFFD.
  sc.used_json = used.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
  sc.density_fit_off = s.density && s.white_light && o.prints.empty() && o.fit_curves == Options::kFitCurvesOff;
  if (s.density) sc.density_prints = o.prints;
  sc.print_id = s.print_id;
  sc.print_name = s.print_name;
  sc.print_file = s.print_file;
  sc.print_file_sha256 = s.print_file_sha256;
  sc.print_from_bundle = s.print_from_bundle;
  std::string print_bracket_name;   // a picture print's bracket as recorded: nearest its final A
  if (!s.print_profs.empty()) {
    sc.print_a = s.print_a;
    for (int c = 0; c < 3; ++c) sc.print_paper_xyz[c] = s.print_paper_xyz[c];
    // The recorded bracket and its A_mid: the one nearest the final A (after the knobs, their moved A), as the search
    // names the one nearest its own. The picture renders through the search's bracket's print ICC either way.
    if (c.bundle_target_idx >= 0 && !s.print_cont.a_mid.empty()) {
      const int nearest = nearest_bracket(s.print_cont.a_mid, s.print_a);
      sc.print_bracket_a_mid = s.print_cont.a_mid[nearest];
      print_bracket_name = s.bundle.targets[nearest].name;
    }
    print_tilt_at(s.bundle, s.print_a, sc.print_tilt);   // a picture print always has its search's A (solve)
    if (s.print_searched) {
      sc.print_searched = true;
      sc.print_a0 = s.print_res.a0;
      sc.print_target_l = s.print_res.target_mean_l;
      sc.print_steps = s.print_res.steps;
      sc.print_converged = s.print_res.converged;
      sc.print_above_knee = s.print_res.above_knee;
      for (int c = 0; c < 3; ++c)
        sc.print_past_knee[c] = s.print_res.past_knee[c], sc.print_past_cut[c] = s.print_res.past_cut[c];
      for (int c = 0; c < 3; ++c) sc.print_above_white[c] = s.print_res.above_white[c];
      for (int c = 0; c < 3; ++c) sc.print_at_white[c] = s.print_res.at_white[c];
      sc.print_a0_at_end = s.print_res.a0_at_range_end;
      for (int c = 0; c < 3; ++c) sc.print_last_move[c] = s.print_res.last_move[c];
      sc.print_stopped_by = s.print_res.stopped_by;
    }
  }
  sc.balance_gains = s.white_light ? scan.balance_gains : nullptr;
  sc.profile_path = s.is_bundle ? s.bundle.path : s.film_prof_arg;
  sc.is_bundle = s.is_bundle;
  sc.bundle_target_name = print_bracket_name.empty() ? c.bundle_target_name : print_bracket_name;
  sc.bundle_target_idx = c.bundle_target_idx;
  sc.input_space = s.bundle.input_space;
  sc.merged_matrix = c.merged_matrix;
  sc.exposure_comp = c.exposure_comp;
  sc.gain_g = c.gain_g;
  sc.gain_b = c.gain_b;
  sc.station_params = c.station_params ? c.station_params->data() : nullptr;
  sc.pin_mask = c.pin_mask;
  sc.knee = c.knee;
  sc.shoulder = c.shoulder;
  if (encoding != ENCODE_SRGB && sc.shoulder.active()) {
    // Only the sRGB output applies the roll-off: record it as off (keeping the measured 99.5% point).
    sc.shoulder.x_hi = 1.0f;
    sc.shoulder.source = "off";
    sc.shoulder.has_knee = false;
  }
  sc.dino = s.use_dino && !scan.has_auto_res && !s.has_cached_sidecar ? &c.dino_res : nullptr;
  sc.search_renderer = dynamic_cast<const GpuRenderer*>(s.renderer.get())       ? "gpu"
                       : dynamic_cast<const CompareRenderer*>(s.renderer.get()) ? "cpu, compared with the gpu"
                                                                                : "cpu";
  sc.dino_model = s.dino_path;
  sc.dino_backend = s.has_cached_sidecar ? s.cached_sidecar.dino_backend
                                         : (s.use_dino ? std::string(s.engine.backend_used()) : o.dino_backend);
  sc.dino_arg = o.dino_arg;
  sc.dino_iters_given = o.dino_iters_given;
  if (o.dino_backend != "auto") sc.dino_backend_arg = o.dino_backend;
  sc.dino_iters = o.dino_iters;
  sc.auto_res = scan.has_auto_res && !s.has_cached_sidecar ? &scan.auto_res : nullptr;
  if (sc.auto_res && scan.shadow_search.ran) {
    // The search's shadow alternative (ParamSearchResult::shadow) and the choice (solve): each screened bracket's
    // shares in the clip pass, the bracket it chose and why there was none, and the two candidates' shares on the tone
    // stage's clip check sample.
    const ShadowAlternative& sa = scan.shadow_search;
    const ShadowChoice& ch = scan.shadow_choice;
    nlohmann::ordered_json j, rows = nlohmann::ordered_json::array();
    for (const ShadowBracketRow& r : sa.rows)
      rows.push_back({{"target_idx", r.target_idx}, {"white_pct", (double)(r.white)},
                      {"black_pct", (double)(r.black)}});
    j["clip_pass"] = rows;
    j["result_white_pct"] = (double)(sa.result_white);
    j["result_black_pct"] = (double)(sa.result_black);
    j["bracket"] = sa.picked >= 0 ? nlohmann::ordered_json(sa.picked) : nlohmann::ordered_json(nullptr);
    j["none"] = sa.none ? nlohmann::ordered_json(sa.none) : nlohmann::ordered_json(nullptr);
    j["chosen_within_allowance"] = sa.chosen_within_allowance;
    if (sa.result) {
      const ParamSearchResult& b = *sa.result;
      j["alternative"] = {{"target_idx", b.target_idx}, {"e", (double)(b.params.e)}, {"g", (double)(b.params.g)},
                          {"b", (double)(b.params.b)}, {"residual", (double)(b.residual)},
                          {"reused_tournament_run", sa.reused_run}};
      auto shares = [](const double* v) {
        return nlohmann::ordered_json{{"white_before", (double)(v[0])}, {"black_before", (double)(v[1])},
                                      {"white_after", (double)(v[2])}, {"black_after", (double)(v[3])}};
      };
      j["checked"] = ch.checked;
      if (ch.checked) {
        j["result_sample"] = shares(ch.result);
        j["alternative_sample"] = shares(ch.alternative);
      } else {
        j["not_checked"] = ch.not_checked ? nlohmann::ordered_json(ch.not_checked) : nlohmann::ordered_json(nullptr);
      }
      j["kept"] = ch.kept_alternative ? "alternative" : "result";
    }
    j["inferences"] = sa.inferences;
    j["time_ms"] = {{"search_step", (double)(sa.ms)}, {"stages", (double)(ch.stages_ms)},
                    {"check", (double)(ch.check_ms)}};
    sc.shadow_json_str = j.dump();
  }
  if (s.has_cached_sidecar) {
    // The previous run's search records, its intent field and its curves' record carry over unchanged.
    if (s.cached_sidecar.is_auto) {
      sc.cached_auto_json_str = s.cached_sidecar.auto_json_str;
    } else {
      sc.cached_dino_json_str = s.cached_sidecar.dino_json_str;
      sc.cached_dino_views_ms = s.cached_sidecar.t_dino_views_ms;
      sc.cached_dino_render_ms = s.cached_sidecar.t_dino_render_ms;
      sc.cached_dino_infer_ms = s.cached_sidecar.t_dino_infer_ms;
      sc.cached_dino_search_ms = s.cached_sidecar.t_dino_search_ms;
    }
    sc.cached_intent_field_json_str = s.cached_sidecar.intent_field_json_str;
    sc.cached_heads_json_str = s.cached_sidecar.heads_json_str;
    sc.cached_input_curves_json_str = s.cached_sidecar.input_curves_json_str;
    if (!s.print_profs.empty() && std::isfinite(s.print_a))
      sc.cached_print_search_json_str = s.cached_sidecar.print_search_json_str;
  }
  if (scan.has_knob_res) {
    sc.has_knobs = true;
    sc.brightness = o.brightness;
    sc.tint = o.tint;
    sc.warmth = o.warmth;
    for (int i = 0; i < 3; ++i) sc.knob_base[i] = scan.knob_base[i];
    sc.knob_res = scan.knob_res;
  } else if (s.density && o.knobs_given && !o.prints.empty()) {
    sc.brightness = o.brightness;
    sc.tint = o.tint;
    sc.warmth = o.warmth;
  }
  sc.auto_frame_half = scan.auto_frame_half;
  sc.colorspace = cs_name;
  sc.t_raw_decode_ms = scan.t_raw_decode_ms;
  sc.t_frame_det_ms = scan.t_frame_det_ms;
  sc.t_full_conversion_ms = scan.t_full_conversion_ms;
  sc.t_tiff_write_ms = scan.t_tiff_write_ms;
  sc.t_auto_ms = scan.t_auto_ms;
  sc.t_analysis_ms = scan.t_analysis_ms;
  for (int i = 0; i < 3; ++i) {
    sc.head_path[i] = s.head_path[i];
    sc.head_fp[i] = s.head_fp[i];
  }
  sc.replay_source = s.replay_source;
  sc.tone_curve = scan.tone_applied ? &scan.tone_curve : s.has_tone ? &s.tone_curve : nullptr;
  sc.tone_targets = scan.tone_applied ? &scan.tone_targets : nullptr;
  sc.tone_share = scan.tone_share;
  sc.tone_shadow_share = scan.tone_shadow_share;
  // The rule that chose the curve where no share of the moves held both ends (Options::tone_fallback).
  if (scan.tone_fell_back)
    sc.tone_fallback = o.tone_fallback == kToneFallbackShadowsFirst      ? "shadows_first"
                       : o.tone_fallback == kToneFallbackHighlightsFirst ? "highlights_first"
                                                                         : "least_total";
  sc.tone_applied = scan.tone_applied;
  sc.tone_clip = &scan.tone_clip;
  if (!s.input_curves_source.empty()) {
    sc.input_curves_source = s.input_curves_source;
    sc.input_curves = &s.input_curves;
    sc.cli_input_curves = scan.has_knob_curves ? &scan.knob_curves : nullptr;
    sc.curve_fit = (s.input_curves_source == "fit" || s.input_curves_source == "auto") && !s.has_cached_sidecar
                       ? &s.curve_fit : nullptr;
  }
  if (s.tone_stage_ran && !s.has_cached_sidecar) {
    sc.tone_stage_plan = &s.tone_stage;
    sc.tone_plan = &s.tone_plan;
  }
  write_json_sidecar(sc);
  return 0;
}

// ====================================================================================================================
// The program
// How neg_process is meant to be used, and main, which runs the pipeline's stages in turn.
// ====================================================================================================================

#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21))
#error This code requires LibRaw 0.21+ (vendored in 3rd_party/libraw)
#endif

// How neg_process is meant to be used, by capture (to move to the docs later):
//
// - A single RAW captured under the narrowband RGB LED (tri) or white (d65) light, e.g.
//     neg_process --film_stock portra400 --light_source tri --roi 1700 0 6686 6376 -o out.tif scan.ARW
//   A tri bundle (input_space camera_raw_rgb) converts the sensor RGB with its crosstalk correction; -r, -g and -b
//   override it with one measured for this capture. A d65 bundle (rec2020_linear) converts LibRaw's white-light decode
//   (the camera's daylight white balance and colour matrix, to linear Rec.2020) with no crosstalk correction, as
//   negicc-station does, so it takes no -r, -g or -b, and no pixel shift. -p overrides the bundle with a negicc-station
//   bundle JSON (e.g. to test parity with negicc-station) or a single ICC profile (for research).
// - Four RAW files of a Sony A7R IV 4-shot pixel shift capture, merged into one image, the same way.
// - A linear (demosaiced) DNG, e.g. a film scanner's, decoded by LibRaw as a RAW is: camera RGB and the bundle's
//   crosstalk correction for a tri bundle, the DNG's colour data to linear Rec.2020 for a d65 bundle. A fourth
//   sample per pixel (a scanner's infrared) is left out.
// - An RGB TIFF (8 or 16 bits) or JPEG, with a d65 bundle: converted to linear Rec.2020 through its embedded ICC
//   profile (np_input.h); without one, 8-bit data is taken as sRGB and 16-bit data as linear Rec.2020.
// - Three images, each captured under one of the red, green and blue lights (RAWs, DNGs or TIFFs, the same size):
//     neg_process --film_stock portra400 --light_source tri -o out.tif red.ARW green.ARW blue.ARW
//   R comes from the first, G from the second and B from the third (a grey TIFF gives its one channel), each scaled
//   to the first capture's exposure when all three record theirs. There is no crosstalk between the lights to
//   correct, so the correction is the identity (-r, -g and -b give another). --flat applies to RAWs only.
//
// With a bundle, the film base is read from the scan itself, as negicc-station's Auto reads it (film_base_detector.h):
// the unexposed film beside the frame (the rebate or a gap between frames) when the frame detector finds it, else the
// thinnest film on the scan; the bundle's when neither gives one. The sidecar's film_base records which.
//
// The parameter search (the calibration bracket, the exposure and the gains, solved with DINOv3 intent) runs
// whenever a profile is given. With a bundle, on either light, --target, -E, --gain_g and --gain_b pin what they give
// and the same search solves the rest, with the tone stage (and on white light the curve fit) after it (--tone_stage
// off and --fit_curves off leave them out; with all four given and both off the conversion takes the values as
// given). With an ICC profile, --gain_g or --gain_b turns the search off, and -E keeps the exposure as given and
// solves only the gains (the single-bracket solver).
// The search sees the frame --roi gives or --roi auto finds, on the half-size image.
//
// --roi auto [h|v] finds the frame to crop (film_frame_detector.h): 35mm, 645, 6x6, 6x7 or 6x9 film on bare light, in
// a strip holder or in a frame holder, tilted up to 5 degrees, partial neighbour frames beside it. --film_format gives
// the format (more accurate than trying each); h or v gives the strip direction. The detection reads the RAW's CFA
// mosaic in place before anything is decoded, and only the frame is decoded, e.g.
//     neg_process --film_stock portra400 --light_source tri --target 0 --roi auto --film_format 6x7 -o out.tif scan.ARW
int main(int ac, char* av[]) {
  Options o;
  if (const int r = parse_options(ac, av, o)) return r < 0 ? 0 : r;
  negicc::apply_default_thread_count();  // at most 16 threads, unless OMP_NUM_THREADS is set
  // A failed step on the GPU, a pass of the search or a full-size step, throws negicc::GpuError; nothing else is
  // meant to throw. It ends the conversion before the picture is written.
  try {
    Setup s;
    if (prepare(o, s)) return 1;
    Scan scan;
    if (decode(o, s, scan)) return 1;
    if (finish_search_passes(o, s)) return 1;
    Conversion c;
    if (begin_conversion(o, s, c)) return 1;
    const std::unique_ptr<Backend> backend_owner = make_backend(o);
    Backend& backend = *backend_owner;
    if (render(o, s, scan, backend)) return 1;
    if (solve(o, s, scan, c)) return 1;
    if (convert(o, s, scan, c, backend)) return 1;
    return write(o, s, scan, c);
  } catch (const negicc::GpuError& e) {
    log_error("%s (a step on the GPU failed: --dino_backend cpu keeps everything on the CPU)\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    log_error("%s\n", e.what());
    return 1;
  }
}
