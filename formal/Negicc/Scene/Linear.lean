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
`pcsWhite` maps to `(1, 1, 1)` (`pcs_to_linear_srgb_unclamped` in `negicc_core.h`). The result is
held at zero from below only: values above 1, brighter than the chart's white, are kept for the
highlight shoulder (`pcs_to_linear_srgb_pre_shoulder`).

`convertPixelLinear` is the conversion of one pixel up to this point, the *pre-shoulder* value.
The clipping statistics and the measurement of the shoulder read it.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The PCS white: D50 as an ICC profile stores it, `(63190, 65536, 54061) / 65536`, the PCS
illuminant as an `s15Fixed16Number` (`kPcsWhite`). -/
def pcsWhite : Vec3 α := ⟨63190.0 / 65536.0, 1.0, 54061.0 / 65536.0⟩

/-- PCS XYZ (D50) to linear sRGB: the inverse of the D50-adapted colorants `rXYZ`, `gXYZ`, `bXYZ`
of the embedded sRGB profile, as written in `pcs_to_linear_srgb_unclamped`. -/
def pcsToLinearSrgbMatrix : Mat3 α :=
  ⟨⟨3.13427485, -1.61727573, -0.490724277⟩,
   ⟨-0.978795611, 1.91616171, 0.0334533327⟩,
   ⟨0.0719770074, -0.228984979, 1.40571822⟩⟩

/-- PCS XYZ (D50) to linear sRGB, not clamped (`pcs_to_linear_srgb_unclamped`). -/
def pcsToLinearSrgbUnclamped (v : Vec3 α) : Vec3 α :=
  pcsToLinearSrgbMatrix.mulVec v

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
