/-
The base of the library.
-/
module

public import Negicc.Base.Defs
public import Negicc.Base.Real
public import Negicc.Base.Props

/-!
# The base of the library

* `Negicc.Base.Scalar`: the class of numbers the formulas are written over, and its `Float`
  instance.
* `Negicc.Base.Vec`: 3-vectors and 3x3 matrices.
* `Negicc.Base.Table`: sampled tables with linear interpolation, sums and means.
* `Negicc.Base.Colour`: the sRGB curve, CIELAB and density.
* `Negicc.Base.Defs`: the four above together (no Mathlib).
* `Negicc.Base.Real`: the real numbers as scalars.
* `Negicc.Base.Props`: proved properties of the base definitions over the real numbers.
-/
