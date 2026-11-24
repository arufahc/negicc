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
the white-light balance part of `negicc.h`):

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

/-- LibRaw's linear sRGB to Rec.2020 matrix (`LibRaw_constants::rec2020_rgb`). -/
def rec2020Rgb : Mat3 α :=
  ⟨⟨0.627452, 0.329249, 0.043299⟩, ⟨0.069109, 0.919531, 0.011360⟩, ⟨0.016398, 0.088030, 0.895572⟩⟩

/-- The entry `(i, j)` of a matrix, indices taken modulo 3. -/
def Mat3.entry (m : Mat3 α) (i j : Nat) : α :=
  let row := match i % 3 with | 0 => m.row0 | 1 => m.row1 | _ => m.row2
  match j % 3 with | 0 => row.x | 1 => row.y | _ => row.z

/-- The adjugate `det(m) m⁻¹`, entry `(j, i)` the cofactor of `(i, j)`, as `film_base_balance_gains`
forms it. -/
def Mat3.adjugate (m : Mat3 α) : Mat3 α :=
  let cof := fun (i j : Nat) =>
    m.entry (i + 1) (j + 1) * m.entry (i + 2) (j + 2) - m.entry (i + 1) (j + 2) * m.entry (i + 2) (j + 1)
  ⟨⟨cof 0 0, cof 1 0, cof 2 0⟩, ⟨cof 0 1, cof 1 1, cof 2 1⟩, ⟨cof 0 2, cof 1 2, cof 2 2⟩⟩

/-- The determinant, along the first row with the adjugate's first column. -/
def Mat3.det (m : Mat3 α) : α :=
  m.row0.x * m.adjugate.row0.x + m.row0.y * m.adjugate.row1.x + m.row0.z * m.adjugate.row2.x

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
