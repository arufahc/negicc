/-
The white-light fit's input curves.
-/
module

public import Negicc.Scene.Convert

@[expose] public section

/-!
# The fitted input curves

On a white-light scan the fit bends the green and blue input of the bracket's profile
(`input_curves.h`). Per channel, on the value `v` entering the profile's input curve (after the
input matrix and the knee), with `ref` the film base at that point:

  `u = log2(v / ref)` for `v < ref`, `f(u) = u (1 + a + k tanh(u - u_med))`, `v' = ref 2^f(u)`;
  `v ≥ ref`: `v` (`input_curve`, `curve_stops`).

Below `u_lo`, the densest grid cell's `u`, the held curve stops bending: `f(u) = f(u_lo) + (u - u_lo)`
(slope 1 in stops). The fit solves `a` (the bend) and holds `k` at 0. The fitted curve is composed
into the bracket's green and blue input tables, `in(curve(i / (n - 1)))` (`compose_input_table`,
`CombinedProfile::build`), so the one conversion applies it.

`ref`, `u_med` (the median over the grid's cells of `min(u, 0)`) and `u_lo` are taken at the `E`,
`g`, `b` the conversion renders with (`input_curve_reference`, `JointCurveModel::anchored`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- One channel's curve (the `[0]` or `[1]` entries of `InputCurves`): the bend `a`, the term `k`,
the pivot `ref`, `u_med`, `u_lo`, and whether the curve is held below `u_lo`. -/
structure ChannelCurve (α : Type) where
  a : α
  k : α
  ref : α
  uMed : α
  uLo : α
  hold : Bool

/-- The fitted curves of green and blue (`InputCurves`). -/
structure InputCurves (α : Type) where
  g : ChannelCurve α
  b : ChannelCurve α

/-- The identity curve of a channel (`InputCurves{}`): no bend, pivot 1, not held. -/
def ChannelCurve.zero : ChannelCurve α := ⟨0.0, 0.0, 1.0, 0.0, 0.0, false⟩

/-- No curves (`InputCurves{}`). -/
def InputCurves.zero : InputCurves α := ⟨ChannelCurve.zero, ChannelCurve.zero⟩

/-- A value is zero, decided through `≤` both ways (the C++'s `== 0.0f`). -/
def isZero (x : α) : Bool := decide (x ≤ 0.0) && decide (0.0 ≤ x)

/-- A curve does something (`InputCurves::active`): a bend or a `k` on either channel. -/
def InputCurves.active (c : InputCurves α) : Bool :=
  !(isZero c.g.a && isZero c.g.k && isZero c.b.a && isZero c.b.k)

/-- `f(u)`, the curve in stops (`curve_stops`): `u (1 + a + k tanh(u - u_med))`, and below the hold
point `uHold` (when there is one) `f(uHold) + (u - uHold)`. -/
def curveStops (u uMed a k : α) (uHold : Option α) : α :=
  match uHold with
  | some h => if u < h then h * (1.0 + a + k * tanh (h - uMed)) + (u - h) else u * (1.0 + a + k * tanh (u - uMed))
  | none => u * (1.0 + a + k * tanh (u - uMed))

/-- The hold point of a channel: `u_lo` when held (`InputCurves::u_hold`). -/
def ChannelCurve.uHold (c : ChannelCurve α) : Option α := if c.hold then some c.uLo else none

/-- The curve on one value (`input_curve`): the identity at and above `ref` and for the identity
curve; below it `ref 2^f(u)`, `u = min(log2(max(v, 1e-12) / ref), 0)`. -/
def inputCurve (c : ChannelCurve α) (v : α) : α :=
  if !(decide (v < c.ref)) || (isZero c.a && isZero c.k) then v
  else
    let vv := max v 1e-12
    let u := min (log2 (vv / c.ref)) 0.0
    c.ref * exp2 (curveStops u c.uMed c.a c.k c.uHold)

/-- A channel's input table with the curve composed into it (`compose_input_table`): entry `i` of
`n` is the base table read at `clamp01(curve(i / (n - 1)))`. -/
def composeInputTable (base : Array α) (c : ChannelCurve α) : Array α :=
  let n := base.size
  (Array.range n).map fun (i : Nat) =>
    tableEval base (clamp01 (inputCurve c ((i : α) / ((n - 1 : Nat) : α))))

/-- The bracket's profile with the green and blue input tables replaced by the curves composed into
them (`CombinedProfile::build`); the profile itself when the curves do nothing. A table of fewer than
two entries is left as it is. -/
def combinedProfile (p : Profile α) (c : InputCurves α) : Profile α :=
  if !c.active then p
  else
    let ig := if 1 < p.inCurves.y.size then composeInputTable p.inCurves.y c.g else p.inCurves.y
    let ib := if 1 < p.inCurves.z.size then composeInputTable p.inCurves.z c.b else p.inCurves.z
    { p with inCurves := ⟨p.inCurves.x, ig, ib⟩ }

/-- The median of a list (`median` in `input_curves.h`): the middle value, the mean of the two middle
ones for an even count, `0` for none. -/
def median (xs : Array α) : α :=
  if xs.size = 0 then 0.0
  else
    let s := xs.qsort (fun a b => decide (a < b))
    let h := xs.size / 2
    if xs.size % 2 = 1 then s.getD h 0.0 else 0.5 * (s.getD (h - 1) 0.0 + s.getD h 0.0)

/-- The smallest entry, `0` for none. -/
def minimum (xs : Array α) : α :=
  if xs.size = 0 then 0.0 else xs.foldl min (xs.getD 0 0.0)

/-- One channel's `u` of a corrected value against its pivot: `min(log2(max(v, 1e-12) /
max(ref, 1e-12)), 0)`. -/
def stopsBelowRef (ref v : α) : α := min (log2 (max v 1e-12 / max ref 1e-12)) 0.0

/-- `ref`, `u_med` and `u_lo` of green and blue (`input_curve_reference`): the film base (camera
units) and each grid cell (16-bit camera RGB) through the candidate matrix at `(e, g, b)` and the
knee. `shape` gives the bends and the hold. -/
def inputCurveReference (shape : InputCurves α) (merged : Mat3 α) (e g b knee : α) (filmBase : Vec3 α)
    (grid : Array (Vec3 Nat)) : InputCurves α :=
  let m := candidateMatrix merged e g b
  let fb := m.mulVec (filmBase.map (· / 65535.0))
  let refG := kneeCompress knee fb.y
  let refB := kneeCompress knee fb.z
  let cells := grid.map fun px => m.mulVec (loadU16 px)
  let ug := cells.map fun v => stopsBelowRef refG (kneeCompress knee v.y)
  let ub := cells.map fun v => stopsBelowRef refB (kneeCompress knee v.z)
  ⟨{ shape.g with ref := refG, uMed := median ug, uLo := minimum ug },
   { shape.b with ref := refB, uMed := median ub, uLo := minimum ub }⟩

/-- One channel's curve stays monotone and within `maxDev` stops of the base table over the whole
table it is composed into (`curve_shape_ok`): walked from `ref` down over the entries `i / (n - 1)`
below `ref`, each `f(u)` below the one above it (from `f = 0` at `ref`) and `|f(u) - u| ≤ maxDev`;
and entry 0, read at `1e-12`, below the last. The identity curve, a pivot that is not positive and a
table of fewer than two entries pass. -/
def curveShapeOk (c : ChannelCurve α) (tableN : Nat) (maxDev : α) : Bool := Id.run do
  if isZero c.a && isZero c.k then return true
  if !(decide (0.0 < c.ref)) || tableN < 2 then return true
  let mut prev : α := 0.0
  for j in [0:tableN - 1] do
    let i := tableN - 1 - j
    let v : α := (i : α) / ((tableN - 1 : Nat) : α)
    if v < c.ref then
      let u := log2 (v / c.ref)
      let f := curveStops u c.uMed c.a c.k c.uHold
      if !(decide (f < prev)) || maxDev < fabs (f - u) then return false
      prev := f
  pure (!(decide (1e-12 < c.ref)) || decide (curveStops (log2 (1e-12 / c.ref)) c.uMed c.a c.k c.uHold < prev))

end Negicc
