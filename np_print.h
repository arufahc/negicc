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
// py-ref: negicc_station/src/print_slopes.py (PrintModel, fit_slopes, print_clf)

#pragma once

#include <array>
#include <string>
#include <vector>

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
};
PrintSlopeFit fit_print_slopes(const PrintMaterial& m, const std::vector<std::array<double, 3>>& print_density,
                               int grey_at);

// The material as CLF nodes, to follow a Matrix that gives its printing density: Matrix (density -> log exposure) /
// Range / LUT1D (the curve of each layer) / Matrix / LUT3D (dye amounts -> linear sRGB, cube points a side) /
// Exponent (the sRGB curve). The text ends before </ProcessList>.
std::string print_clf_nodes(const PrintMaterial& m, int curve_points = 4096, int cube = 45);
