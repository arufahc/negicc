/-
The tone stage.
-/
module

public import Negicc.Tone.Defs
public import Negicc.Tone.Props

/-!
# The tone stage

* `Negicc.Tone.Apply`: the curve on one display linear sRGB pixel.
* `Negicc.Tone.Curve`: the lightness curve, its table, the render's descriptors, the targets, the
  Levenberg-Marquardt solve, the choice of the highlight share.
* `Negicc.Tone.Clip`: the clip check on a sample of the export.
* `Negicc.Tone.Defs`: the three above together (no Mathlib).
* `Negicc.Tone.Props`: proved properties over the real numbers.
-/
