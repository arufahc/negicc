/-
From the profile connection space to linear sRGB, and the conversion up to the pre-shoulder value.
-/
module

public import Negicc.Scene.Input
public import Negicc.Scene.Profile

@[expose] public section

/-!
# To linear sRGB

The film profile gives XYZ under D50. One fixed matrix takes it to linear sRGB: the inverse of the
D50-adapted colorants of the sRGB profile the outputs embed (Elle Stone's sRGB V2), so that the codes
written mean, through the embedded profile, the PCS they were computed from, and the PCS white
`pcsWhite` maps to `(1, 1, 1)` (`pcs_to_linear_srgb_unclamped` in `color_pipeline.h`). The result is
held at zero from below only: values above 1, brighter than the chart's white, are kept for the
highlight shoulder (`pcs_to_linear_srgb_pre_shoulder`).

`convertPixelLinear` is the conversion of one pixel up to this point, the *pre-shoulder* value.
The clipping statistics and the measurement of the shoulder read it.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The PCS illuminant: D50 to four decimals, as ICC.1:2022 (7.2.16) defines it (`kD50` in
`np_print.cc`). The ICC states these numbers; they are not the CIE sum of D50 times the observer,
whose `Z` is `0.82502` on the print's tables (`observerWhite_d50` in `Negicc.Print.Constants`). -/
def iccD50 : Vec3 α := ⟨0.9642, 1.0, 0.8249⟩

/-- A non-negative number as an ICC `s15Fixed16Number`: the nearest multiple of `1/65536`, a half
upward (`put_s15` in `np_print.cc`, `lround(v · 65536)`). -/
def s15Fixed16 (v : α) : α := (roundNat (v * 65536.0) : α) / 65536.0

/-- The PCS white: the PCS illuminant as an ICC profile stores it, `s15Fixed16` of `iccD50`
(`kPcsWhite`, which `color_pipeline.h` writes as `(63190, 65536, 54061) / 65536`; `pcsWhite_eq` in
`Negicc.Scene.Props`). -/
def pcsWhite : Vec3 α := iccD50.map s15Fixed16

/-- The colorants `rXYZ`, `gXYZ`, `bXYZ` of the sRGB profile `neg_process` embeds (Elle Stone's
sRGB V2; its sRGB-TRC and gamma 1.0 variants carry the same), as the columns of a matrix, each an
`s15Fixed16Number` read from the profile. They are the profile's data: its maker adapted the BT.709
primaries to D50 and quantised them (within 1.03 units of `1/65536` of Bradford from D65 to
`iccD50`, `elleSrgbColorants_bradford` in `Negicc.Scene.Props`), and each row sums to the PCS
white exactly. -/
def elleSrgbColorants : Mat3 α :=
  ⟨⟨28576.0 / 65536.0, 25239.0 / 65536.0, 9375.0 / 65536.0⟩,
   ⟨14581.0 / 65536.0, 46983.0 / 65536.0, 3972.0 / 65536.0⟩,
   ⟨912.0 / 65536.0, 6361.0 / 65536.0, 46788.0 / 65536.0⟩⟩

/-- PCS XYZ (D50) to linear sRGB: the inverse of the embedded profile's colorants. -/
def pcsToLinearSrgbMatrix : Mat3 α := Mat3.adjInv elleSrgbColorants

/-- The inverse of the colorants as `pcs_to_linear_srgb_unclamped` writes it, to nine significant
digits, the matrix the conversion computes with: each entry the value of `pcsToLinearSrgbMatrix`
correctly rounded (`pcsToLinearSrgbMatrix_literal` in `Negicc.Scene.Props`). The digits are kept, as
the C++ has them: the white-light fit's numeric check moves outside its tolerance when the matrix
changes by that rounding. -/
def pcsToLinearSrgbMatrixLiteral : Mat3 α :=
  ⟨⟨3.13427485, -1.61727573, -0.490724277⟩,
   ⟨-0.978795611, 1.91616171, 0.0334533327⟩,
   ⟨0.0719770074, -0.228984979, 1.40571822⟩⟩

/-- PCS XYZ (D50) to linear sRGB, not clamped (`pcs_to_linear_srgb_unclamped`). -/
def pcsToLinearSrgbUnclamped (v : Vec3 α) : Vec3 α :=
  pcsToLinearSrgbMatrixLiteral.mulVec v

/-- PCS XYZ (D50) to linear sRGB, held at zero from below
(`pcs_to_linear_srgb_pre_shoulder`). -/
def pcsToLinearSrgbPreShoulder (v : Vec3 α) : Vec3 α :=
  (pcsToLinearSrgbUnclamped v).map (max 0.0)

/-- The film profile's PCS XYZ of a loaded pixel `v` under the input matrix `m`
(`convert_pixel_pcs`). -/
def convertPixelPcs (p : Profile α) (knee : α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  applyProfile p (correctedRgb knee m v)

/-- The pre-shoulder linear sRGB of a loaded pixel `v` under the input matrix `m`: zero or more,
with the chart's white at 1 (`convert_pixel_linear`). -/
def convertPixelLinear (p : Profile α) (knee : α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  pcsToLinearSrgbPreShoulder (convertPixelPcs p knee m v)

/-! ## Clipping rules -/

/-- A linear channel counts as white-clipped from the linear value of the 16-bit sRGB code
65399.5 on (`linear_channel_white_clipped`). -/
def linearChannelWhiteClipped (l : α) : Bool := decide (0.9953029 ≤ l)

/-- A linear channel counts as black-clipped below the linear value of the 16-bit sRGB code 200.5
(`linear_channel_black_clipped`). -/
def linearChannelBlackClipped (l : α) : Bool := decide (l < 0.00023679829)

/-- A pixel is white-clipped when any channel is, measured on the pre-shoulder value. -/
def pixelWhiteClipped (lin : Vec3 α) : Bool :=
  linearChannelWhiteClipped lin.x || linearChannelWhiteClipped lin.y || linearChannelWhiteClipped lin.z

/-- A pixel is black-clipped when every channel is, measured on the pre-shoulder value. -/
def pixelBlackClipped (lin : Vec3 α) : Bool :=
  linearChannelBlackClipped lin.x && linearChannelBlackClipped lin.y && linearChannelBlackClipped lin.z

end Negicc
