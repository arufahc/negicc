/-
The print emulation's definitions, without Mathlib.
-/
module

public import Negicc.Print.Cie
public import Negicc.Print.Spectral
public import Negicc.Print.Matrix
public import Negicc.Print.Material
public import Negicc.Print.Grey
public import Negicc.Print.Curves
public import Negicc.Print.White
public import Negicc.Print.Density
public import Negicc.Print.Clf
public import Negicc.Print.Floor
public import Negicc.Print.Search

/-!
# The print emulation's definitions

The print of a film on a print material (DESIGN.md section 5): the material's spectral model, the
print matrix, the print ICC's curves and table, the print's white, the density file and its CLF,
the film base's floor and the print's search. No Mathlib: an executable imports this module.
-/
