/-
The film-base balance of the white-light decode.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The film-base balance of a white-light scan

Before LibRaw's camera matrix mixes the camera's channels, the white-light decode multiplies them
by gains that put this scan's film base on the profile's (`film_base_balance_gains` in
`white_light_balance.h`):

  `K_c = (C⁻¹ f_profile)_c / (C⁻¹ f)_c`, divided by the largest of the three,

with `C = rec2020_rgb · rgb_cam` the matrix the decode applies (LibRaw's camera to linear
Rec.2020), `f` this scan's base and `f_profile` the profile's, both as linear Rec.2020. `C⁻¹` is
taken as the adjugate over the determinant; the determinant cancels in the ratio and only gives the
sign. When either base has a camera channel that is not positive the gains are left as they were
(`none` here).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- LibRaw's linear sRGB to Rec.2020 matrix (`LibRaw_constants::rec2020_rgb`) as a formula: XYZ to
BT.2020 after BT.709 to XYZ, both with Lindbloom's D65, so that each row sums to one. LibRaw does not
say how it computed its numbers; this formula reproduces all nine within `5.3e-7`, where D65 as the
chromaticity `(0.3127, 0.3290)` misses by `4.8e-5`. -/
def rec2020Rgb : Mat3 α :=
  (Mat3.adjInv (rgbToXyz bt2020Primaries lindbloomD65)).mul (rgbToXyz bt709Primaries lindbloomD65)

/-- LibRaw's matrix as it writes it, to six decimals. Eight entries are the value of `rec2020Rgb`
correctly rounded; the ninth, `0.329249`, is `5.2e-7` from it (its rounding is `0.329248`), and with
it each written row sums to one exactly (`rec2020Rgb_literal` in `Negicc.WhiteLight.Props`). -/
def rec2020RgbLiteral : Mat3 α :=
  ⟨⟨0.627452, 0.329249, 0.043299⟩, ⟨0.069109, 0.919531, 0.011360⟩, ⟨0.016398, 0.088030, 0.895572⟩⟩

/-- The gains on the camera channels that put the base `base` on the profile's base `profileBase`
(both linear Rec.2020), for the decode's camera matrix `rgbCam` (`film_base_balance_gains`); `none`
when a camera channel of either base is not positive. -/
def filmBaseBalanceGains (rgbCam : Mat3 α) (base profileBase : Vec3 α) : Option (Vec3 α) :=
  let m := rec2020Rgb.mul rgbCam
  let adj := m.adjugate
  let det := m.det
  let cam := adj.mulVec base
  let camP := adj.mulVec profileBase
  if (decide (0.0 < cam.x / det) && decide (0.0 < camP.x / det)) &&
     (decide (0.0 < cam.y / det) && decide (0.0 < camP.y / det)) &&
     (decide (0.0 < cam.z / det) && decide (0.0 < camP.z / det)) then
    let gain : Vec3 α := ⟨camP.x / cam.x, camP.y / cam.y, camP.z / cam.z⟩
    let largest := gain.maxComp
    some (gain.map (· / largest))
  else none

end Negicc
