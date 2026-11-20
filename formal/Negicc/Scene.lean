/-
Reading the scan and the per-pixel conversion, for white-light and narrow-band scans.
-/
module

public import Negicc.Scene.Defs
public import Negicc.Scene.Props

/-!
# The scene conversion

* `Negicc.Scene.Decode`: black level, the half-size image, the debayers, pixel shift, three captures.
* `Negicc.Scene.Input`: flat field, load, the input matrix with exposure and gains, the soft knee.
* `Negicc.Scene.FilmBase`: where the film base is read from, and the statistics that read it.
* `Negicc.Scene.Profile`: the bracket's film profile (input curves, colour table, output curves).
* `Negicc.Scene.Linear`: the profile connection space to linear sRGB; the clipping rules.
* `Negicc.Scene.Shoulder`: the highlight roll-off and how a frame's `x_hi` is found.
* `Negicc.Scene.Convert`: the whole pixel, to encoded sRGB or linear and to CIELAB; the outputs.
* `Negicc.Scene.Crop`: the inner crop the statistics read.
* `Negicc.Scene.Defs`: the modules above together (no Mathlib).
* `Negicc.Scene.Props`: proved properties over the real numbers.
-/
