/-
The tone curve applied to a pixel.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The tone curve on a pixel

The tone stage's curve is one lightness curve. It reaches the pixels as a table `lut` of
`Y' = f(Y)` on the display's linear luminance, sampled at `Y = (i / n)^2` for `i = 0 .. n`
(`n = 8192`; the table has `n + 1` entries and is built by `toneTable` in `Negicc.Tone.Curve`).

`applyTone` moves a pixel's luminance along the curve and carries its colour along: the
luminance `Y` goes to `Y'` on the neutral axis, and the pixel's difference from the neutral of its
own luminance, `v - Y`, is added back scaled by `f'(Y) / f'(Y')`, where `f` is CIELAB's cube-root
compression. To first order that keeps `a*` and `b*` where they were. No pixel is converted to
CIELAB. `tone_table_Y` and `apply_tone` in `color_pipeline.h`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The derivative of CIELAB's `f`: `1 / (3 t^(2/3))` above `216/24389`, the slope of the linear
segment `(24389/27) / 116` below (`tone_lab_fprime`). -/
def toneLabFprime (t : α) : α :=
  if 216.0 / 24389.0 < t then (1.0 / 3.0) / cbrt (t * t) else 24389.0 / 27.0 / 116.0

/-- The luminance of a display linear sRGB pixel as the tone curve reads it (the weights of
`apply_tone`; they differ from those of `linearSrgbLuminance` in the sixth decimal). -/
def toneLuminance (v : Vec3 α) : α :=
  0.2126729 * v.x + 0.7151522 * v.y + 0.0721750 * v.z

/-- The curve's luminance `Y'` for a luminance `Y`, read from the table by linear interpolation
in `sqrt Y`. `Y` is held to `[0, 1]` for the lookup; above 1 the ratio `Y' / Y` keeps its value at
1 (`tone_table_Y`). -/
def toneTableY (lut : Array α) (Y : α) : α :=
  let n := lut.size - 1
  let y := min (max Y 0.0) 1.0
  let t := sqrt y * (n : α)
  let i := min (natFloor t) (n - 1)
  let d := t - (i : α)
  let yp := lut.getD i 0.0 + d * (lut.getD (i + 1) 0.0 - lut.getD i 0.0)
  if 1.0 < Y then yp * Y else yp

/-- The tone curve on one display linear sRGB pixel: `Y' + (v - Y) f'(Y) / f'(Y')`, with the
ratio held at 8 (`apply_tone`). -/
def applyTone (lut : Array α) (v : Vec3 α) : Vec3 α :=
  let y0 := toneLuminance v
  let y1 := toneTableY lut y0
  let sc := min (toneLabFprime (max y0 0.0) / toneLabFprime (max y1 1e-9)) 8.0
  ⟨y1 + sc * (v.x - y0), y1 + sc * (v.y - y0), y1 + sc * (v.z - y0)⟩

end Negicc
