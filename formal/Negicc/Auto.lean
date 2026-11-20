/-
The Auto search.
-/
module

public import Negicc.Auto.Defs
public import Negicc.Auto.Props

/-!
# The Auto search (DESIGN.md 3.5 and 4.2)

* `Negicc.Auto.Geometry`: the inner crop and the network's grid.
* `Negicc.Auto.Resample`: area average, bilinear sample, Pillow's antialiased bilinear filter.
* `Negicc.Auto.Network`: the network's input, the intent field, the networks as functions of
  stated shape.
* `Negicc.Auto.Cells`: the trained cells, the residual, the frame's statistics, the optimiser's
  loss and correction, the lightness wish, the highlight statistics.
* `Negicc.Auto.Enclosure`: the brackets' ranges, which are searched, the rankings, the frame's
  scale.
* `Negicc.Auto.Optimizer`: the `(E, g, b)` coordinate descent.
* `Negicc.Auto.Loop`: the fixed-point loop and the tournament's winner.
* `Negicc.Auto.Metering`: the grey point, the metered exposure, the single-bracket anchor.
* `Negicc.Auto.Search`: the whole search and its decisions.
* `Negicc.Auto.Knobs`: brightness, tint and warmth on a result.
* `Negicc.Auto.Given`: the single-profile solver, the brackets' exposures for `-E`, the roll-off at
  given parameters.
* `Negicc.Auto.Defs`: the modules above together (no Mathlib).
* `Negicc.Auto.Props`: proved properties over the real numbers.
-/
