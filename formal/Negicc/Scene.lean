/-
The per-pixel conversion of a narrow-band scan.
-/
module

public import Negicc.Scene.Defs
public import Negicc.Scene.Props

/-!
# The scene conversion

* `Negicc.Scene.Input`: flat field, load, the input matrix with exposure and gains, the soft knee.
* `Negicc.Scene.Profile`: the bracket's film profile (input curves, colour table, output curves).
* `Negicc.Scene.Linear`: the profile connection space to linear sRGB; the clipping rules.
* `Negicc.Scene.Shoulder`: the highlight roll-off and how a frame's `x_hi` is found.
* `Negicc.Scene.Convert`: the whole pixel, to encoded sRGB and to CIELAB.
* `Negicc.Scene.Defs`: the five above together (no Mathlib).
* `Negicc.Scene.Props`: proved properties over the real numbers.
-/
