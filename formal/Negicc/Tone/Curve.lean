/-
The tone stage: the lightness curve, its table, the render's descriptors, the targets and the solve.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The tone stage's curve and plan

The tone stage sets contrast and the black and white levels with one lightness curve per frame
(`tone_stage.cc`). The curve acts on CIE `L*`:

  `f(L) = black + (white - black) h(g(L / 100))`, `g(x) = x^gamma`,
  `h(x) = x^c / (x^c + (1 - x)^c)` (`c` the contrast),

with `L / 100` held to `[1e-5, 1 - 1e-5]` (`toneCurveL`). Its four numbers are solved so that `f`
of the render's percentiles meets targets: the render's shadow level `L2` moved all the way to the
tone heads' prediction, the highlight level `L98` and the range `L90 - L10` half way, the mid-tone
`L50` held (`toneTargets`). The solve is a bounded Levenberg-Marquardt on nine residuals: the four
targets, a weak prior toward the identity, and a term holding the rise above `L98` to the input's
(`toneResiduals`, `toneSolve`), from three starting white points, the lowest cost kept.

The curve reaches the pixels as a table on display linear luminance, `Y' = Y(f(L*(Y)))` at
`Y = (i / 8192)^2` (`toneTable`), applied by `applyTone` (`Negicc.Tone.Apply`). Where the curve
would clip more than 0.5 % more of the export's pixels white than the conversion without it, the
highlight and range moves are scaled down to the largest of the shares `1, 0.75, 0.5, 0.25, 0` that
keeps within that allowance (`tonePickClipShare`, `tonePlanForImage`).

The descriptors are read on the 8-bit preview of the final render (`toneDescriptors`): `L*` per
pixel from the sRGB decode and the luminance row of `linear_srgb_to_lab`, with CIE 15's exact
break, and percentiles interpolated linearly between ranks (Hyndman and Fan's definition 7).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The curve -/

/-- The curve's four numbers (`ToneCurve`). -/
structure ToneCurve (α : Type) where
  black : α
  white : α
  gamma : α
  contrast : α

namespace ToneCurve

/-- The identity curve, the default and the solve's starting shape (`ToneCurve{}`). -/
def identity : ToneCurve α := ⟨0.0, 100.0, 1.0, 1.0⟩

/-- The four numbers in the order of the solve's parameter vector: black, white, gamma, contrast. -/
def toArray (q : ToneCurve α) : Array α := #[q.black, q.white, q.gamma, q.contrast]

/-- The curve of a parameter vector (`curve_of` in `tone_solve`). -/
def ofArray (a : Array α) : ToneCurve α := ⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0, a.getD 3 0.0⟩

end ToneCurve

/-- The curve's lower bounds: black, white, gamma, contrast (`kToneLo`). -/
def toneLo : Array α := #[-15.0, 80.0, 0.7, 0.7]

/-- The curve's upper bounds (`kToneHi`). -/
def toneHi : Array α := #[15.0, 110.0, 1.45, 1.5]

/-- `f(L)` without the final clamp to `[0, 100]` (`tone_curve_L`). -/
def toneCurveL (q : ToneCurve α) (L : α) : α :=
  let x0 := L / 100.0
  let x1 := if x0 < 1e-5 then 1e-5 else if 1.0 - 1e-5 < x0 then 1.0 - 1e-5 else x0
  let x := pow x1 q.gamma
  let xc := pow x q.contrast
  let yc := pow (1.0 - x) q.contrast
  q.black + (q.white - q.black) * xc / (xc + yc)

/-- A curve the tone stage can produce: its four numbers within `toneLo .. toneHi`
(`tone_curve_ok`; a recorded curve outside them is refused). Finiteness, which the C++ also tests,
is not modelled. -/
def toneCurveOk (q : ToneCurve α) : Bool :=
  (List.range 4).all fun i =>
    decide (toneLo.getD i 0.0 ≤ q.toArray.getD i 0.0) && decide (q.toArray.getD i 0.0 ≤ toneHi.getD i 0.0)

/-- The number of steps of the kernel's table (`kToneTableSize`); the table has one entry more. -/
def toneTableSize : Nat := 8192

/-- The kernel's table of a curve: entry `i` is `Y' = Y(f(L*(Y)))` at `Y = (i / 8192)^2`, with `f`
clamped to `[0, 100]` (`tone_table`). -/
def toneTable (q : ToneCurve α) : Array α :=
  (Array.range (toneTableSize + 1)).map fun (i : Nat) =>
    let t : α := (i : α) / (toneTableSize : α)
    let L2 := toneCurveL q (116.0 * labF (t * t) - 16.0)
    let L2c := if L2 < 0.0 then 0.0 else if 100.0 < L2 then 100.0 else L2
    labFInv ((L2c + 16.0) / 116.0)

/-! ## The render's descriptors -/

/-- The render's descriptors: pixel `L*` percentiles (`ToneDescriptors`). -/
structure ToneDescriptors (α : Type) where
  L2 : α
  L10 : α
  L25 : α
  L50 : α
  L75 : α
  L90 : α
  L98 : α

/-- The tonal range `L90 - L10`. -/
def ToneDescriptors.range (d : ToneDescriptors α) : α := d.L90 - d.L10

/-- The `p` percentile (`0 ≤ p ≤ 100`) of values sorted ascending, linear between ranks: rank
`r = p / 100 (n - 1)`, Hyndman and Fan's definition 7, numpy's default (`pct` in
`tone_descriptors`). -/
def percentileSorted (xs : Array α) (p : α) : α :=
  let n := xs.size
  let r := p / 100.0 * ((n - 1 : Nat) : α)
  let i := natFloor r
  if i + 1 < n then xs.getD i 0.0 + (r - (i : α)) * (xs.getD (i + 1) 0.0 - xs.getD i 0.0)
  else xs.getD (n - 1) 0.0

/-- `L*` of an 8-bit sRGB pixel as the descriptors read it: each channel decoded through the sRGB
curve, the luminance row of `linear_srgb_to_lab`, and CIE `L* = 116 f(Y) - 16`. -/
def toneLstarOfU8 (px : Vec3 Nat) : α :=
  let lin : Vec3 α := px.map fun (c : Nat) => srgbDecode ((c : α) / 255.0)
  116.0 * labF (linearSrgbLuminance lin) - 16.0

/-- The descriptors of an 8-bit RGB image (`tone_descriptors`); all zero for an empty image. -/
def toneDescriptors (rgb : Array (Vec3 Nat)) : ToneDescriptors α :=
  if rgb.size = 0 then ⟨0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0⟩
  else
    let L := (rgb.map (toneLstarOfU8 (α := α))).qsort (fun a b => decide (a < b))
    let pct := percentileSorted L
    ⟨pct 2.0, pct 10.0, pct 25.0, pct 50.0, pct 75.0, pct 90.0, pct 98.0⟩

/-! ## Targets and the solve -/

/-- The descriptors the curve aims at (`ToneTargets`). -/
structure ToneTargets (α : Type) where
  L2 : α
  L50 : α
  L98 : α
  range : α

/-- The shares of the way from the render's descriptor to the heads' prediction: shadow
(`kToneShadowMove`), highlight (`kToneHighlightMove`), range (`kToneRangeMove`). -/
def toneShadowMove : α := 1.0
@[inherit_doc toneShadowMove] def toneHighlightMove : α := 0.5
@[inherit_doc toneShadowMove] def toneRangeMove : α := 0.5

/-- The targets at a highlight share (`tone_targets`): `L2` moved all the way to the prediction,
`L98` and the range by `share / 2` of the way, `L50` held. -/
def toneTargets (cur : ToneDescriptors α) (predL2 predL98 predRange share : α) : ToneTargets α :=
  ⟨cur.L2 + toneShadowMove * (predL2 - cur.L2),
   cur.L50,
   cur.L98 + share * toneHighlightMove * (predL98 - cur.L98),
   cur.range + share * toneRangeMove * (predRange - cur.range)⟩

/-- The prior's weight (`kTonePrior`) and the top-slope weight (`kToneTopSlope`). -/
def tonePrior : α := 0.02
@[inherit_doc tonePrior] def toneTopSlope : α := 0.07

/-- The nine residuals of a curve (`resid` in `tone_solve`): the three percentile targets and the
range; the prior toward the identity on each number (scales 5 `L*` for black and white, 0.1 in log
for gamma and contrast); and the rise above `L98`, `f(100) - min(100, f(L98) + 100 - L98)`. -/
def toneResiduals (cur : ToneDescriptors α) (t : ToneTargets α) (q : ToneCurve α) : Array α :=
  let f98 := toneCurveL q cur.L98
  #[toneCurveL q cur.L2 - t.L2,
    f98 - t.L98,
    toneCurveL q cur.L50 - t.L50,
    toneCurveL q cur.L90 - toneCurveL q cur.L10 - t.range,
    tonePrior * (q.black / 5.0),
    tonePrior * ((q.white - 100.0) / 5.0),
    tonePrior * (log q.gamma / 0.1),
    tonePrior * (log q.contrast / 0.1),
    toneTopSlope * (toneCurveL q 100.0 - min 100.0 (f98 + (100.0 - cur.L98)))]

/-- The sum of the squared residuals. -/
def toneCost (cur : ToneDescriptors α) (t : ToneTargets α) (q : Array α) : α :=
  (toneResiduals cur t (ToneCurve.ofArray q)).foldl (fun s v => s + v * v) 0.0

/-- The solution `d` of `M d = rhs` for a 4x4 system given as rows of five entries (the matrix and
the right-hand side): Gaussian elimination with partial pivoting (the first of equal pivots kept),
a pivot below `1e-300` in magnitude replaced by `1e-300`, and back substitution, as `tone_solve`
and the white-light fit write it. -/
def solve4 (m0 : Array (Array α)) : Array α := Id.run do
  let mut m := m0
  for c in [0:4] do
    let mut piv := c
    for rr in [c + 1:4] do
      if fabs (m[piv]!.getD c 0.0) < fabs (m[rr]!.getD c 0.0) then piv := rr
    let rowC := m[piv]!
    m := (m.set! piv m[c]!).set! c rowC
    if fabs (m[c]!.getD c 0.0) < 1e-300 then m := m.modify c (·.set! c 1e-300)
    for rr in [c + 1:4] do
      let f := m[rr]!.getD c 0.0 / m[c]!.getD c 0.0
      let rowc := m[c]!
      m := m.modify rr fun row => Id.run do
        let mut row := row
        for cc in [c:5] do
          row := row.set! cc (row.getD cc 0.0 - f * rowc.getD cc 0.0)
        pure row
  let mut d : Array α := Array.replicate 4 0.0
  for k in [0:4] do
    let c := 3 - k
    let mut sum := m[c]!.getD 4 0.0
    for cc in [c + 1:4] do
      sum := sum - m[c]!.getD cc 0.0 * d.getD cc 0.0
    d := d.set! c (sum / m[c]!.getD c 0.0)
  pure d

/-- One Levenberg-Marquardt run of `tone_solve` from the start `q0`: central differences kept
inside the bounds, the normal equations damped by `lambda max(A_aa, 1e-12)`, a number at a bound
that the gradient pushes outward held there, the step clamped into the bounds and accepted when it
lowers the cost (`lambda / 3`, at least `1e-12`) or retried with `4 lambda` (at most 20 tries); at
most 200 iterations, ended when no try improves or the step is below `1e-12`. Returns the end
point and its cost. -/
def toneLM (cur : ToneDescriptors α) (t : ToneTargets α) (q0 : Array α) : Array α × α := Id.run do
  let lo : Array α := toneLo
  let hi : Array α := toneHi
  let resid := fun (q : Array α) => toneResiduals cur t (ToneCurve.ofArray q)
  let cost := toneCost cur t
  let mut q := q0
  let mut c := cost q
  let mut lambda : α := 1e-3
  for _ in [0:200] do
    let r := resid q
    -- The Jacobian by central differences, column j.
    let mut J : Array (Array α) := Array.replicate 9 (Array.replicate 4 0.0)
    for j in [0:4] do
      let qj := q.getD j 0.0
      let h := 1e-6 * max 1.0 (fabs qj)
      let qp := q.set! j (min (hi.getD j 0.0) (qj + h))
      let qm := q.set! j (max (lo.getD j 0.0) (qj - h))
      let rp := resid qp
      let rm := resid qm
      for k in [0:9] do
        J := J.modify k (·.set! j ((rp.getD k 0.0 - rm.getD k 0.0) / (qp.getD j 0.0 - qm.getD j 0.0)))
    let mut A : Array (Array α) := Array.replicate 4 (Array.replicate 4 0.0)
    let mut g : Array α := Array.replicate 4 0.0
    for k in [0:9] do
      for a in [0:4] do
        g := g.set! a (g.getD a 0.0 + J[k]!.getD a 0.0 * r.getD k 0.0)
        for b in [0:4] do
          A := A.modify a (·.set! b (A[a]!.getD b 0.0 + J[k]!.getD a 0.0 * J[k]!.getD b 0.0))
    let held : Array Bool := (Array.range 4).map fun a =>
      (decide (q.getD a 0.0 ≤ lo.getD a 0.0) && decide (0.0 < g.getD a 0.0)) ||
      (decide (hi.getD a 0.0 ≤ q.getD a 0.0) && decide (g.getD a 0.0 < 0.0))
    let mut improved := false
    let mut step : α := 0.0
    for _ in [0:20] do
      if improved then break
      let M : Array (Array α) := (Array.range 4).map fun a =>
        ((Array.range 4).map fun b =>
          if held.getD a false || held.getD b false then (if a = b then 1.0 else 0.0)
          else A[a]!.getD b 0.0 + (if a = b then lambda * max (A[a]!.getD a 0.0) 1e-12 else 0.0)).push
          (if held.getD a false then 0.0 else -(g.getD a 0.0))
      let d := solve4 M
      let qn : Array α := (Array.range 4).map fun j =>
        clamp (q.getD j 0.0 + d.getD j 0.0) (lo.getD j 0.0) (hi.getD j 0.0)
      step := (Array.range 4).foldl (fun s j => max s (fabs (qn.getD j 0.0 - q.getD j 0.0))) 0.0
      let cn := cost qn
      if cn < c then
        q := qn
        c := cn
        lambda := max (lambda / 3.0) 1e-12
        improved := true
      else
        lambda := lambda * 4.0
      if step < 1e-13 then break
    if !improved || step < 1e-12 then break
  pure (q, c)

/-- The white points the solve starts from (`kToneWhiteStarts`). -/
def toneWhiteStarts : List α := [100.0, 110.0, 85.0]

/-- The curve that brings `cur` to `t` (`tone_solve`): one run of `toneLM` from black 0, gamma and
contrast 1 and each starting white point, the run of lowest cost kept (the first of equal ones). -/
def toneSolve (cur : ToneDescriptors α) (t : ToneTargets α) : ToneCurve α :=
  let best := toneWhiteStarts.foldl (fun (acc : Option (Array α × α)) w0 =>
    let run := toneLM cur t #[0.0, w0, 1.0, 1.0]
    match acc with
    | none => some run
    | some b => if run.2 < b.2 then some run else some b) none
  match best with
  | some b => ToneCurve.ofArray b.1
  | none => ToneCurve.identity

/-- The plan at a highlight share (`tone_plan`): the targets and the curve solved for them. -/
def tonePlan (render : ToneDescriptors α) (predL2 predL98 predRange share : α) :
    ToneTargets α × ToneCurve α :=
  let t := toneTargets render predL2 predL98 predRange share
  (t, toneSolve render t)

/-- The targets with the shadow move scaled by its own share (`tone_targets` with a shadow share;
`toneTargets` moves it all the way). -/
def toneTargetsShares (cur : ToneDescriptors α) (predL2 predL98 predRange share shadowShare : α) :
    ToneTargets α :=
  ⟨cur.L2 + shadowShare * toneShadowMove * (predL2 - cur.L2),
   cur.L50,
   cur.L98 + share * toneHighlightMove * (predL98 - cur.L98),
   cur.range + share * toneRangeMove * (predRange - cur.range)⟩

/-- The plan with the shadow move scaled by its own share (`tone_plan_shares`). -/
def tonePlanShares (render : ToneDescriptors α) (predL2 predL98 predRange share shadowShare : α) :
    ToneTargets α × ToneCurve α :=
  let t := toneTargetsShares render predL2 predL98 predRange share shadowShare
  (t, toneSolve render t)

/-! ## The clip check -/

/-- The share of the export's pixels the curve may clip white beyond the conversion's own
(`kToneClipAllowance`). -/
def toneClipAllowance : α := 0.005

/-- The highlight shares tried in turn (`kToneClipShares`). -/
def toneClipShares : Array α := #[1.0, 0.75, 0.5, 0.25, 0.0]

/-- The share kept, as an index into `toneClipShares` (`tone_pick_clip_share`): `0` when the
designed curve clips within the allowance; else the first later share that does; else the share
that clips least (the first of equal ones). `whiteAfter k` is the white clipping with the curve
planned at share `k`. -/
def tonePickClipShare (whiteBefore : α) (whiteAfter : Nat → α) : Nat := Id.run do
  let a0 := whiteAfter 0
  if a0 ≤ whiteBefore + toneClipAllowance then return 0
  let mut best := 0
  let mut bestAfter := a0
  for k in [1:5] do
    let a := whiteAfter k
    if a ≤ whiteBefore + toneClipAllowance then return k
    if a < bestAfter then
      best := k
      bestAfter := a
  pure best

end Negicc
