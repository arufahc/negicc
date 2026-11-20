/-
The sRGB curve, CIELAB, and density.
-/
module

public import Negicc.Base.Vec

@[expose] public section

/-!
# Colour primitives

* The sRGB transfer curve and its inverse (`srgb_encode`, `srgb_decode` in `color_pipeline.h`).
* Linear sRGB (D65) to XYZ relative to the white point, and to CIELAB, with the constants the
  pipeline's scoring uses (`lab_f`, `linear_srgb_to_lab`, `linear_srgb_to_L`, `srgb_to_lab`).
* Density, `-log10` of a transmission.

The constants are the ones written in `color_pipeline.h`. Two remarks on them:

* The C++ writes the encoding exponent as the single precision literal `0.41666667f`, the
  nearest single precision number to `1 / 2.4`. Here it is `1 / 2.4`, the value the sRGB
  standard defines.
* CIELAB's break and slope are CIE 15's exact fractions, `labEpsilon = 216/24389 = (6/29)^3` and
  `labKappa = 24389/27 = (29/3)^3` (`kLabEpsilon`, `kLabKappa`), so the two branches of `labF` and of
  `lstarOfY` meet at the break. The sRGB to XYZ matrix and the white are OpenCV's, rounded to six
  decimals; the white `(0.950456, 1, 1.088754)` is the matrix's row sums.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The sRGB transfer curve -/

/-- The sRGB encoding of a linear value: a line near black, a power curve above it
(`srgb_encode`). -/
def srgbEncode (l : α) : α :=
  if l ≤ 0.0031308 then l * 12.92 else pow l (1.0 / 2.4) * 1.055 - 0.055

/-- The linear value of an sRGB encoded one (`srgb_decode`). -/
def srgbDecode (c : α) : α :=
  if c ≤ 0.04045 then c * (1.0 / 12.92) else pow ((c + 0.055) * (1.0 / 1.055)) 2.4

/-! ## CIELAB -/

/-- A CIELAB colour: lightness `L` (0 to 100) and the two colour axes `a` (green to magenta) and
`b` (blue to yellow). -/
@[ext] structure Lab (α : Type) where
  L : α
  a : α
  b : α

namespace Lab

variable {β γ : Type}

/-- The colour with `f` applied to each of `L`, `a`, `b`. -/
def map (f : α → β) (c : Lab α) : Lab β := ⟨f c.L, f c.a, f c.b⟩

/-- Two colours combined component by component. -/
def zipWith (f : α → β → γ) (c : Lab α) (d : Lab β) : Lab γ := ⟨f c.L d.L, f c.a d.a, f c.b d.b⟩

/-- The sum, component by component. -/
def add (c d : Lab α) : Lab α := ⟨c.L + d.L, c.a + d.a, c.b + d.b⟩

/-- The difference, component by component. -/
def sub (c d : Lab α) : Lab α := ⟨c.L - d.L, c.a - d.a, c.b - d.b⟩

instance : Add (Lab α) := ⟨add⟩
instance : Sub (Lab α) := ⟨sub⟩

end Lab

/-- CIELAB's break `(6/29)^3 = 216/24389` (`kLabEpsilon`). -/
def labEpsilon : α := 216.0 / 24389.0

/-- CIELAB's slope `(29/3)^3 = 24389/27` (`kLabKappa`): `L* = labKappa Y` below the break. -/
def labKappa : α := 24389.0 / 27.0

/-- CIELAB's compression of a relative tristimulus value: the cube root, with a line near zero,
`(labKappa t + 16) / 116` (`lab_f`). -/
def labF (t : α) : α :=
  if labEpsilon < t then cbrt t else (labKappa * t + 16.0) / 116.0

/-- The inverse of `labF`: the cube above the break's `f`, the line's inverse below it
(`tone_lab_finv` in `tone_stage.cc`, and the Lab inverse of `pcs_lab_to_srgb_u8` in
`multi_target_optimizer.cc`). -/
def labFInv (f : α) : α :=
  if labEpsilon < f * f * f then f * f * f else (116.0 * f - 16.0) / labKappa

/-- The luminance `Y` of a linear sRGB colour: the middle row of the sRGB to XYZ matrix. White
`(1, 1, 1)` has `Y = 1`. -/
def linearSrgbLuminance (c : Vec3 α) : α :=
  0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z

/-- Linear sRGB (D65) to XYZ, each divided by the D65 white's value: `(X / 0.950456, Y,
Z / 1.088754)`. White `(1, 1, 1)` maps to `(1, 1, 1)`. -/
def linearSrgbToXyzRel (c : Vec3 α) : Vec3 α :=
  ⟨(0.412453 * c.x + 0.357580 * c.y + 0.180423 * c.z) / 0.950456,
   linearSrgbLuminance c,
   (0.019334 * c.x + 0.119193 * c.y + 0.950227 * c.z) / 1.088754⟩

/-- CIE lightness `L*` of a relative luminance `Y`. -/
def lstarOfY (Y : α) : α :=
  if labEpsilon < Y then 116.0 * cbrt Y - 16.0 else labKappa * Y

/-- CIELAB of XYZ relative to the white point. -/
def xyzRelToLab (w : Vec3 α) : Lab α :=
  ⟨if labEpsilon < w.y then 116.0 * labF w.y - 16.0 else labKappa * w.y,
   500.0 * (labF w.x - labF w.y),
   200.0 * (labF w.y - labF w.z)⟩

/-- CIELAB (D65) of a linear sRGB colour (`linear_srgb_to_lab`). -/
def linearSrgbToLab (c : Vec3 α) : Lab α := xyzRelToLab (linearSrgbToXyzRel c)

/-- CIE lightness `L*` of a linear sRGB colour (`linear_srgb_to_L`). -/
def linearSrgbToL (c : Vec3 α) : α := lstarOfY (linearSrgbLuminance c)

/-- CIELAB (D65) of an sRGB encoded colour (`srgb_to_lab`). -/
def srgbToLab (c : Vec3 α) : Lab α := linearSrgbToLab (c.map srgbDecode)

/-! ## Density -/

/-- The density of a transmission `t` (a ratio of light let through, `0 < t`): `-log10 t`. -/
def density (t : α) : α := -(log10 t)

/-- The transmission of a density `d`: `10 ^ (-d)`. -/
def transmission (d : α) : α := pow 10.0 (-d)

end Negicc
