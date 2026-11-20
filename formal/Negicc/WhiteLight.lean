/-
What a white-light scan adds to the conversion.
-/
module

public import Negicc.WhiteLight.Defs
public import Negicc.WhiteLight.Props

/-!
# White light

* `Negicc.WhiteLight.Balance`: gains on the camera channels that put the scan's film base on the
  profile's before the camera matrix.
* `Negicc.WhiteLight.Curve`: given green and blue input curves and their composition into the
  profile's tables.
* `Negicc.WhiteLight.Defs`: the two above together (no Mathlib).
* `Negicc.WhiteLight.Props`: proved properties over the real numbers.
-/
