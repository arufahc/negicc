/-
The film's grey balance curve, a bracket's mid grey and the brackets' tilt.
-/
module

public import Negicc.Print.Spectral

@[expose] public section

/-!
# The grey balance curve and the brackets

The film's grey balance curve `G_c` (DESIGN.md 5.2) is a table per light channel `c`: the APD of
the film's grey that has light density `start + step i` in channel `c` (`GreyBalanceCurves` in
`profile_bundle.h`). Read forwards (`onGrid`) it takes a light density to APD; read backwards
(`curveGreyAt`, an `interp` with the axes swapped) it gives the light densities of the grey with
a given green APD `A`. Each bracket has its mid grey as green APD (`bracketMidApd`) and its tilt
`t_c`, interpolated linearly in `A` between the brackets (`printTiltAt`). The curve's tangent at
`A` gives the slopes that give the film's grey scale equal contrast in the three channels
(`curveSlopesAt`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The film's grey balance curve (`GreyBalanceCurves`): for channel `c`, `apd.c[i]` is the APD of
the grey with light density `start + step i` in channel `c`, rising. -/
structure GreyBalance (α : Type) where
  start : α
  step : α
  apd : Vec3 (Array α)

namespace GreyBalance

/-- The light densities of the table, `start + step i` (`grey_grid`). -/
def grid (g : GreyBalance α) : Array α :=
  (Array.range g.apd.x.size).map fun (i : Nat) => g.start + g.step * (i : α)

/-- `G_c(d)`: the APD of the grey with light density `d` in channel `c`. -/
def eval (g : GreyBalance α) (apd : Array α) (d : α) : α := onGrid apd g.start g.step d

end GreyBalance

/-- The light densities of the curve's grey with green APD `a`: each channel's table read
backwards (`curve_grey_at`). Held at the table's ends outside its APD range. -/
def curveGreyAt (g : GreyBalance α) (a : α) : Vec3 α :=
  ⟨interp g.apd.x g.grid a, interp g.apd.y g.grid a, interp g.apd.z g.grid a⟩

/-- A bracket's mid grey as green APD: the light density of its chart's 18 % grey step through the
APD matrix's green row (`bracket_mid_apd`). -/
def bracketMidApd (mApd : Mat3 α) (greyLight : Vec3 α) : α := mApd.row1.dot greyLight

/-- `γ = M (dn/dA)` at green APD `a`: with `n` the curve's grey at `a` and
`dn_j/dA = 1 / G_j'(n_j)` (a central difference of one table step), the contrast of the curve's
grey scale in each channel of the measure `M` (`curve_slopes_at`). -/
def curveGamma (g : GreyBalance α) (mApd : Mat3 α) (a : α) : Vec3 α :=
  let n := curveGreyAt g a
  let inv (apd : Array α) (nc : α) : α :=
    1.0 / ((onGrid apd g.start g.step (nc + g.step) - onGrid apd g.start g.step (nc - g.step)) /
      (2.0 * g.step))
  mApd.mulVec ⟨inv g.apd.x n.x, inv g.apd.y n.y, inv g.apd.z n.z⟩

/-- The slopes at green APD `a` (DESIGN.md 5.9, `curve_slopes_at`):
`(γ_green / γ_red, 1, γ_green / γ_blue)`. -/
def curveSlopesAt (g : GreyBalance α) (mApd : Mat3 α) (a : α) : Vec3 α :=
  let k := curveGamma g mApd a
  ⟨k.y / k.x, 1.0, k.y / k.z⟩

/-! ## The brackets' tilt at a mid grey -/

/-- The tilt between two brackets: `t_i + f (t_j − t_i)` with `f` the share of the way from
`A_mid,i` to `A_mid,j`, held to `[0, 1]`; `f = 0` when the two `A_mid` are not increasing. -/
def tiltSegment (ai aj : α) (ti tj : Vec3 α) (a : α) : Vec3 α :=
  let f := if ai < aj then clamp ((a - ai) / (aj - ai)) 0.0 1.0 else 0.0
  ⟨ti.x + f * (tj.x - ti.x), ti.y + f * (tj.y - ti.y), ti.z + f * (tj.z - ti.z)⟩

/-- The first segment `k` (from `k`) of the brackets in the order `o` whose upper end is not below
`a`, or the last segment: the C++ `while (k + 2 < n && a > a_mid[o[k + 1]]) ++k`. -/
def tiltSegmentIndex (aMid : Array α) (o : Array Nat) (a : α) : Nat → Nat → Nat
  | 0, k => k
  | fuel + 1, k =>
    if k + 2 < o.size ∧ aMid.getD (o.getD (k + 1) 0) 0.0 < a then tiltSegmentIndex aMid o a fuel (k + 1)
    else k

/-- The brackets' tilt at a mid grey `a` (`print_tilt_at`): the brackets sorted by `A_mid`, the
tilt linear in `a` between the two whose `A_mid` enclose it and held beyond the end ones; one
bracket gives its own tilt and none gives zero. `aMid` and `tilt` are per bracket, in any order.

The C++ sorts with `std::sort`, which does not keep the order of brackets with equal `A_mid`; this
specification sorts with `Array.qsort`. The two can differ only on brackets with equal `A_mid`. -/
def printTiltAt (aMid : Array α) (tilt : Array (Vec3 α)) (a : α) : Vec3 α :=
  if aMid.size = 0 ∨ aMid.size ≠ tilt.size then ⟨0.0, 0.0, 0.0⟩
  else
    let o := (Array.range aMid.size).qsort fun i j => decide (aMid.getD i 0.0 < aMid.getD j 0.0)
    if o.size = 1 then tilt.getD (o.getD 0 0) ⟨0.0, 0.0, 0.0⟩
    else
      let k := tiltSegmentIndex aMid o a o.size 0
      let i := o.getD k 0
      let j := o.getD (k + 1) 0
      tiltSegment (aMid.getD i 0.0) (aMid.getD j 0.0) (tilt.getD i ⟨0.0, 0.0, 0.0⟩)
        (tilt.getD j ⟨0.0, 0.0, 0.0⟩) a

end Negicc
