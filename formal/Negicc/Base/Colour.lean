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
  `lstarOfY` meet at the break.
* The sRGB to XYZ matrix is OpenCV's, which comes from a formula (`rgbToXyz`, SMPTE RP 177): the
  BT.709 primaries and OpenCV's D65 white `(0.950456, 1, 1.088754)` give `srgbToXyzOpenCV`. The
  C++ writes its entries to six decimals (`srgbToXyzOpenCVLiteral`); each is the formula's value
  correctly rounded (`srgbToXyzOpenCV_literal` in `Negicc.Base.Props`), and the rows still sum to
  the white exactly. The scoring here uses those six-decimal entries, as the C++ does: the rounding
  moves `L*` by up to `1e-4`, and the tone stage, which the numeric check compares to `1e-9`, reads
  `L*`.
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

/-! ## Colour spaces from their primaries -/

/-- The chromaticities `(x, y)` of the red, green and blue primaries of an RGB space. -/
structure Primaries (α : Type) where
  r : α × α
  g : α × α
  b : α × α

/-- The primaries of ITU-R BT.709, which sRGB (IEC 61966-2-1) shares: red `(0.64, 0.33)`, green
`(0.30, 0.60)`, blue `(0.15, 0.06)`. -/
def bt709Primaries : Primaries α := ⟨(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)⟩

/-- The primaries of ITU-R BT.2020: red `(0.708, 0.292)`, green `(0.170, 0.797)`, blue
`(0.131, 0.046)`. -/
def bt2020Primaries : Primaries α := ⟨(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)⟩

/-- The white of BT.709 and sRGB, D65 as a chromaticity: `(0.3127, 0.3290)`. -/
def bt709White : α × α := (0.3127, 0.3290)

/-- D65 as OpenCV (and dcraw) give it with their sRGB matrix: `(0.950456, 1, 1.088754)`. -/
def opencvD65 : Vec3 α := ⟨0.950456, 1.0, 1.088754⟩

/-- D65 as Bruce Lindbloom tabulates it: `(0.95047, 1, 1.08883)`. -/
def lindbloomD65 : Vec3 α := ⟨0.95047, 1.0, 1.08883⟩

/-- The XYZ of the chromaticity `(x, y)` at `Y = 1`: `(x / y, 1, (1 − x − y) / y)`. -/
def xyToXyz (c : α × α) : Vec3 α := ⟨c.1 / c.2, 1.0, (1.0 - c.1 - c.2) / c.2⟩

/-- The matrix whose columns are the primaries' XYZ at `Y = 1`. -/
def primaryMatrix (p : Primaries α) : Mat3 α :=
  let r := xyToXyz p.r
  let g := xyToXyz p.g
  let b := xyToXyz p.b
  ⟨⟨r.x, g.x, b.x⟩, ⟨r.y, g.y, b.y⟩, ⟨r.z, g.z, b.z⟩⟩

/-- Linear RGB to XYZ for the primaries `p` and the white `w` (SMPTE RP 177):
`C diag(C⁻¹ w)` with `C = primaryMatrix p`, the primaries' columns scaled so that RGB `(1, 1, 1)`
is `w` (`rgbToXyz_white` in `Negicc.Base.Props`). -/
def rgbToXyz (p : Primaries α) (w : Vec3 α) : Mat3 α :=
  (primaryMatrix p).mul (Mat3.diag ((Mat3.adjInv (primaryMatrix p)).mulVec w))

/-- Linear sRGB to XYZ (D65) as the scoring computes it (`linear_srgb_to_lab`, OpenCV's matrix):
the BT.709 primaries with OpenCV's D65. -/
def srgbToXyzOpenCV : Mat3 α := rgbToXyz bt709Primaries opencvD65

/-- OpenCV's sRGB to XYZ matrix as `color_pipeline.h` writes it, to six decimals, the matrix the
scoring computes with: the value of `srgbToXyzOpenCV` correctly rounded, its rows summing to
`opencvD65` exactly (`srgbToXyzOpenCV_literal`, `srgbToXyzOpenCVLiteral_white` in
`Negicc.Base.Props`). -/
def srgbToXyzOpenCVLiteral : Mat3 α :=
  ⟨⟨0.412453, 0.357580, 0.180423⟩, ⟨0.212671, 0.715160, 0.072169⟩, ⟨0.019334, 0.119193, 0.950227⟩⟩

/-! ## Chromatic adaptation -/

/-- The linearised Bradford matrix of ICC.1:2022 Annex E.3 (equation E.1), from XYZ to the
Bradford transform's sharpened cone responses (`B` in `print_table`). Lam fitted it to corresponding
colour data in 1985; no formula gives its entries. -/
def bradford : Mat3 α :=
  ⟨⟨0.8951, 0.2664, -0.1614⟩, ⟨-0.7502, 1.7135, 0.0367⟩, ⟨0.0389, -0.0685, 1.0296⟩⟩

/-- The CIECAM02 chromatic adaptation matrix of CIE 159:2004 (`cat02` in
`load_spektrafilm_print`). It too was fitted to corresponding colour data; no formula gives its
entries. -/
def cat02 : Mat3 α :=
  ⟨⟨0.7328, 0.4296, -0.1624⟩, ⟨-0.7036, 1.6975, 0.0061⟩, ⟨0.0030, 0.0136, 0.9834⟩⟩

/-- The von Kries adaptation from the white `src` to the white `dst` in the cone space of `cone`:
`cone⁻¹ diag(cone dst / cone src) cone`, each cone response scaled by the ratio of the two whites'
responses. -/
def vonKries (cone : Mat3 α) (src dst : Vec3 α) : Mat3 α :=
  let to := cone.mulVec dst
  let fr := cone.mulVec src
  (Mat3.adjInv cone).mul ((Mat3.diag ⟨to.x / fr.x, to.y / fr.y, to.z / fr.z⟩).mul cone)

/-! ## Linear sRGB to CIELAB -/

/-- The luminance `Y` of a linear sRGB colour: the middle row of the sRGB to XYZ matrix. White
`(1, 1, 1)` has `Y = 1`. -/
def linearSrgbLuminance (c : Vec3 α) : α := srgbToXyzOpenCVLiteral.row1.dot c

/-- Linear sRGB (D65) to XYZ, each divided by the D65 white's value: `(X / 0.950456, Y,
Z / 1.088754)`. White `(1, 1, 1)` maps to `(1, 1, 1)`. -/
def linearSrgbToXyzRel (c : Vec3 α) : Vec3 α :=
  ⟨srgbToXyzOpenCVLiteral.row0.dot c / opencvD65.x,
   linearSrgbLuminance c,
   srgbToXyzOpenCVLiteral.row2.dot c / opencvD65.z⟩

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
