/-
The density file: a pixel's density above the film base in a measure, and its code values.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# Density

A pixel's density above the film base in the light's channels, in a measure, and as a code value
(DESIGN.md 5.2 and 5.9; `density_encode_in_place` in `np_print.cc`):

```
T       = merged v / 65535 / n0                 transmittance against bare film
D_light = −log10(max(T, 0.0001))
D       = M D_light                             the measure's density
code    = offset + scale κ_c D_c                10 bits: 95 + 500 κ D     16 bits: 1520 + 8000 κ D
```

`κ` is `(1.00, 0.92, 0.95)` for APD (ADX) and 1 otherwise. A code below 0 or above the largest is
held there and counted, as is a transmittance at or below the floor.

The search's balance in density (`density_balance_offset`): the gains `g`, `b` multiply the light's
green and blue channels, which is the light density offset `(0, −log10 g, −log10 b)`, through `M`;
its mean is taken out, so the offsets sum to zero and the mean is the level.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The transmittance floor before the logarithm (`kDensityMinT`): density 4.0. -/
def densityMinT : α := 1.0e-4

/-- A density file's code values (`DensityEncoding`): bits, offset and codes per unit density, and
the channel factors `κ`. -/
structure DensityEncoding (α : Type) where
  bits : Nat
  offset : Nat
  scale : Nat
  k : Vec3 α

/-- The largest code, `2^bits − 1`. -/
def DensityEncoding.maxCode (e : DensityEncoding α) : Nat := 2 ^ e.bits - 1

/-- The encoding of a density file (`density_encoding`): 10 bits with offset 95 and 500 codes per
unit density in a DPX, else 16 bits with 1520 and 8000; ADX's channel factors `(1.00, 0.92, 0.95)`
for APD. -/
def densityEncoding (dpx apd : Bool) : DensityEncoding α :=
  { bits := if dpx then 10 else 16, offset := if dpx then 95 else 1520,
    scale := if dpx then 500 else 8000,
    k := if apd then ⟨1.00, 0.92, 0.95⟩ else ⟨1.0, 1.0, 1.0⟩ }

/-- The transmittance against bare film of a 16-bit pixel `px` in each light channel: the merged
matrix (crosstalk correction and film base scale) on `px / 65535`, times `1 / n0`. -/
def densityTransmittance (merged : Mat3 α) (n0 : α) (px : Vec3 Nat) : Vec3 α :=
  let v : Vec3 α := ⟨(px.x : α) / 65535.0, (px.y : α) / 65535.0, (px.z : α) / 65535.0⟩
  let inv := 1.0 / n0
  (merged.mulVec v).map (· * inv)

/-- The light density above the film base, the transmittance held at the floor:
`−log10(max(T, 1e-4))`. -/
def lightDensity (t : α) : α := -(log10 (max t densityMinT))

/-- Whether a transmittance is at or below the floor (`at_density_floor`). -/
def atDensityFloor (t : α) : Bool := !(decide (densityMinT < t))

/-- One code value: `offset + scale κ D` rounded to the nearest whole number (halves away from
zero, C `lround`), held to `[0, maxCode]`. The second and third components say whether it was
held at 0 (`low`) or at the largest code (`high`). -/
def densityCode (e : DensityEncoding α) (kc d : α) : Nat × Bool × Bool :=
  let x := (e.offset : α) + (e.scale : α) * kc * d
  if x ≤ -0.5 then (0, true, false)
  else
    let c := roundNat x
    if e.maxCode < c then (e.maxCode, false, true) else (c, false, false)

/-- The density file's code values of a pixel, with each channel's floor, low and high flags
(`density_encode_in_place`, one pixel). `m` is the measure's matrix. -/
def densityEncodePixel (merged : Mat3 α) (n0 : α) (m : Mat3 α) (e : DensityEncoding α)
    (px : Vec3 Nat) : Vec3 (Nat × Bool × Bool) × Vec3 Bool :=
  let t := densityTransmittance merged n0 px
  let d := m.mulVec (t.map lightDensity)
  (⟨densityCode e e.k.x d.x, densityCode e e.k.y d.y, densityCode e e.k.z d.z⟩, t.map atDensityFloor)

/-- The search's colour balance as density offsets with their mean taken out, and that mean
(`density_balance_offset`): `M (0, −log10 g, −log10 b)`, minus its mean. `none` (no balance)
unless `E`, `g` and `b` are positive. -/
def densityBalanceOffset (m : Mat3 α) (e g b : α) : Option (Vec3 α × α) :=
  if 0.0 < e ∧ 0.0 < g ∧ 0.0 < b then
    let off := m.mulVec ⟨0.0, -(log10 g), -(log10 b)⟩
    let mean := (off.x + off.y + off.z) / 3.0
    some (⟨off.x - mean, off.y - mean, off.z - mean⟩, mean)
  else none

/-- The density offsets the print CLF applies for the balance, level included: the zero-sum
offsets plus their mean (`print_balance`); zero without a balance. -/
def printBalance (m : Mat3 α) (e g b : α) : Vec3 α :=
  match densityBalanceOffset m e g b with
  | some (off, mean) => ⟨off.x + mean, off.y + mean, off.z + mean⟩
  | none => ⟨0.0, 0.0, 0.0⟩

end Negicc
