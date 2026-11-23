/-
The print emulation.
-/
module

public import Negicc.Print.Defs
public import Negicc.Print.Props
public import Negicc.Print.Constants

/-!
# The print emulation

* `Negicc.Print.Cie`: the CIE 1931 observer and the illuminants D50 and D55.
* `Negicc.Print.Spectral`: a print material's spectral model, tables, the observer, the lamp.
* `Negicc.Print.Matrix`: 3x3 inverse, transpose and sums.
* `Negicc.Print.Material`: `Q`, the print matrix, the colour matrices, `aim`, a material's load.
* `Negicc.Print.Grey`: the grey balance curve, a bracket's mid grey, the tilt, the slopes.
* `Negicc.Print.Curves`: `Paper_c`, the print ICC's input curves and table.
* `Negicc.Print.White`: the print relative to its paper.
* `Negicc.Print.Density`: the density file's codes and the balance.
* `Negicc.Print.Clf`: the density file's CLF.
* `Negicc.Print.Floor`: the film base held no darker than the thinnest film.
* `Negicc.Print.Search`: the print's search and the knobs.
* `Negicc.Print.Defs`: all of the above (no Mathlib).
* `Negicc.Print.Props`: proved properties over the real numbers.
* `Negicc.Print.Constants`: the print's constants against their formulas, and its spectral sums.
-/
