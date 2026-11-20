/-
The white-light fit's input curves.
-/
module

public import Negicc.Scene.Convert

@[expose] public section

/-!
# Given input curves composed into the profile

A white-light conversion may carry curves on the green and blue input of the bracket's profile
(`input_curves.h`), found by the curve fit or recorded by an earlier run. Per channel, on the value `v`
entering the profile's input curve (after the input matrix and the knee), with its pivot `ref`:

  `u = log2(v / ref)` for `v < ref`, `f(u) = u (1 + a + k tanh(u - u_med))`, `v' = ref 2^f(u)`;
  `v ≥ ref`: `v` (`input_curve`, `curve_stops`).

Below its hold point `u_lo`, a held curve stops bending: `f(u) = f(u_lo) + (u - u_lo)`. The curves are
composed into the bracket's green and blue input tables, `in(curve(i / (n - 1)))`
(`compose_input_table`, `CombinedProfile::build`), so the one conversion applies them.

How the curves are fitted, anchored on a frame and re-anchored after the knobs is not in this
specification: the curves here are given.
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

/-- A value is zero, decided through `≤` both ways (the C++'s `== 0.0f`). -/
def decZero (x : α) : Bool := decide (x ≤ 0.0) && decide (0.0 ≤ x)

/-- A curve does something (`InputCurves::active`): a bend or a `k` on either channel. -/
def InputCurves.active (c : InputCurves α) : Bool :=
  !(decZero c.g.a && decZero c.g.k && decZero c.b.a && decZero c.b.k)

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
  if !(decide (v < c.ref)) || (decZero c.a && decZero c.k) then v
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

end Negicc
