/-
The print's search: the exposure held, A0 metered, the line search in A, the fixed point, the knobs.
-/
module

public import Negicc.Print.Matrix

@[expose] public section

/-!
# The print's search

A print is searched on its own profile (DESIGN.md 5.6; `search_print_material` in
`multi_target_optimizer.cc`): its lightness once and continuously in the mid grey `A`, its colour
in `g` and `b`. The search's control is specified here; what it calls on the intent network and
the optimiser (a render's predicted field, its residual, the optimiser's `g` and `b` at an `A`) is
an oracle, a function given as a parameter.

1. **E is held** where the frame's film base reads `n0` (`searchExposure`).
2. **A0 is metered**: the `A` at which the frame's log-average cell prints at `L*` 50, by 60
   bisection steps over the green curve's APD range (`meterA0`).
3. **The lightness target** is the mean of the fields predicted at `A0 − h`, `A0`, `A0 + h`
   (`bootstrapAs`), `h = log10(2) / 2`.
4. **The fixed point over (A, g, b)** (`printSearchLoop`): each step scores the print at its
   `(A, g, b)`, then solves `A` by a march of 0.1 APD downhill and a golden-section search to 0.001
   (`lineSearchA`), with `g` and `b` optimised at each `A`. It stops when the residual changes by
   1.5 % or less, or one step after the parameters move by 0.02 or less, `A` counted in stops; the
   lowest-residual step is kept.

`--brightness`, `--tint` and `--warmth` move the print's `A`, `g` and `b` by Newton's method on
`(A, ln g, ln b)` (`solvePrintKnobs`, `solve_print_knobs`); the render of the grey is again an
oracle.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## Constants -/

/-- The march's step in `A` (`kContinuousStep`), APD. -/
def continuousStep : α := 0.1

/-- The golden-section search's tolerance in `A` (`kContinuousTol`), APD. -/
def continuousTol : α := 0.001

/-- A stop of printing exposure in density, `log10 2` to five decimals (`kStop`). -/
def stopApd : α := 0.30103

/-- The bootstrap's spread, half a stop of printing exposure (`ContinuousPrint::boot_h`). -/
def bootH : α := 0.150515

/-- The fixed point's parameter tolerance, `|ΔA| / kStop + |Δg| + |Δb|` (`TOL_PARAM`). -/
def tolParam : α := 0.02

/-- The fixed point's relative residual tolerance (`TOL_REL_ERR`). -/
def tolRelErr : α := 0.015

/-- `(√5 − 1) / 2` to sixteen digits, the golden section's ratio. -/
def goldenPhi : α := 0.6180339887498949

/-- Equality decided by the two comparisons, as the C++ `==` on numbers that are not NaN. -/
def eqA (x y : α) : Bool := decide (x ≤ y ∧ y ≤ x)

/-! ## The exposure and A0 -/

/-- The exposure the search holds, in its own units: `e_norm` on a narrow-band scan (whose search
runs on `M_frame / e_norm`), 1 on white light (which runs on `M_frame` itself), so that the film
base reads `n0`. -/
def searchExposure (whiteLight : Bool) (eNorm : α) : α := if whiteLight then 1.0 else eNorm

/-- The result's exposure in the units of `-E`: `E / e_norm` on a narrow-band scan, `E` on white
light. At the held exposure it is 1 (`resultExposure_held`). -/
def resultExposure (whiteLight : Bool) (e eNorm : α) : α := if whiteLight then e else e / eNorm

/-- Bisection on a function that falls through `level`: `fuel` halvings of `[lo, hi]`, keeping the
end above `level` as `lo`. -/
def bisectLevel (f : α → α) (level : α) : Nat → α → α → α × α
  | 0, lo, hi => (lo, hi)
  | fuel + 1, lo, hi =>
    let mid := 0.5 * (lo + hi)
    if level < f mid then bisectLevel f level fuel mid hi else bisectLevel f level fuel lo mid

/-- A0: the `A` at which the log-average cell's print has `L*` 50, `lstar a` its `L*` through the
print at `a`, by 60 bisection steps over `[aLo, aHi]` (`L*` falls as `A` rises). -/
def meterA0 (lstar : α → α) (aLo aHi : α) : α :=
  let e := bisectLevel lstar 50.0 60 aLo aHi
  0.5 * (e.1 + e.2)

/-- Whether A0 lies within `1e-6` of an end of its range (`a0_at_range_end`). -/
def a0AtRangeEnd (a0 aLo aHi : α) : Bool := decide (a0 - aLo < 1.0e-6 ∨ aHi - a0 < 1.0e-6)

/-- The bootstrap's mid greys: `A0 − h`, `A0`, `A0 + h`, each held to the range; `A0` alone for
`h = 0`. -/
def bootstrapAs (a0 h aLo aHi : α) : List α :=
  (if 0.0 < h then [a0 - h, a0, a0 + h] else [a0]).map fun a => clamp a aLo aHi

/-! ## The line search in A -/

/-- A bracket of the minimum: `[lo, hi]` and the loss at its inner point. -/
structure ABracket (α : Type) where
  lo : α
  hi : α
  f : α

/-- The march's loop from the state `(xo, xm, fm, xn, fn)`: while the next point `xn` is lower and
moved, step on by `dir` times 0.1 APD within `[aLo, aHi]`. The result is `[min(xo, xn),
max(xo, xn)]` with the loss at `xm`. `fuel` bounds the steps; the range is bounded, so a march of
more than `(aHi − aLo) / 0.1 + 2` steps does not happen. -/
def marchLoop (loss : α → α) (aLo aHi dir : α) : Nat → α → α → α → α → α → ABracket α
  | 0, xo, _, fm, xn, _ => ⟨min xo xn, max xo xn, fm⟩
  | fuel + 1, xo, xm, fm, xn, fn =>
    if fn < fm ∧ ¬ eqA xn xm then
      let xn' := clamp (xn + dir * continuousStep) aLo aHi
      if eqA xn' xn then ⟨min xm xn', max xm xn', fn⟩
      else marchLoop loss aLo aHi dir fuel xm xn fn xn' (loss xn')
    else ⟨min xo xn, max xo xn, fm⟩

/-- The march from `x0` (loss `f0`) in direction `dir` (`march` in `search_print_material`). -/
def march (loss : α → α) (aLo aHi dir x0 f0 : α) (fuel : Nat) : ABracket α :=
  let xo := clamp (x0 - dir * continuousStep) aLo aHi
  let xn := clamp (x0 + dir * continuousStep) aLo aHi
  marchLoop loss aLo aHi dir fuel xo x0 f0 xn (if eqA xn x0 then f0 else loss xn)

/-- The bracket of the minimum from the current `A` (`search_print_material`): the two neighbours
at 0.1 APD; a lower side is marched (both lower: both, and the lower bracket kept); else the two
neighbours. -/
def bracketA (loss : α → α) (aLo aHi a : α) (fuel : Nat) : ABracket α :=
  let x0 := clamp a aLo aHi
  let f0 := loss x0
  let xl := clamp (x0 - continuousStep) aLo aHi
  let xr := clamp (x0 + continuousStep) aLo aHi
  let fl := if eqA xl x0 then f0 else loss xl
  let fr := if eqA xr x0 then f0 else loss xr
  if fl < f0 ∧ fr < f0 then
    let left := march loss aLo aHi (-1.0) xl fl fuel
    let right := march loss aLo aHi 1.0 xr fr fuel
    if left.f ≤ right.f then left else right
  else if fl < f0 then march loss aLo aHi (-1.0) xl fl fuel
  else if fr < f0 then march loss aLo aHi 1.0 xr fr fuel
  else ⟨xl, xr, f0⟩

/-- The golden-section loop on `[lo, hi]` with inner points `x1 ≤ x2` and their losses, until the
interval is at most `continuousTol` wide (`fuel` bounds the steps). The result is the final
interval. -/
def goldenLoop (loss : α → α) : Nat → α → α → α → α → α → α → α × α
  | 0, lo, hi, _, _, _, _ => (lo, hi)
  | fuel + 1, lo, hi, x1, x2, f1, f2 =>
    if continuousTol < hi - lo then
      if f1 ≤ f2 then
        let x1' := x2 - goldenPhi * (x2 - lo)
        goldenLoop loss fuel lo x2 x1' x1 (loss x1') f1
      else
        let x2' := x1 + goldenPhi * (hi - x1)
        goldenLoop loss fuel x1 hi x2 x2' f2 (loss x2')
    else (lo, hi)

/-- The golden-section search on `[lo, hi]`: the midpoint of the final interval. -/
def goldenSection (loss : α → α) (lo hi : α) (fuel : Nat) : α :=
  let x1 := hi - goldenPhi * (hi - lo)
  let x2 := lo + goldenPhi * (hi - lo)
  let e := goldenLoop loss fuel lo hi x1 x2 (loss x1) (loss x2)
  0.5 * (e.1 + e.2)

/-- The step budget of the march and the golden section: the march moves by 0.1 APD inside the
range and the golden section shrinks its interval by the ratio each step, so 1000 steps are never
reached on a range of a few APD. -/
def lineSearchFuel : Nat := 1000

/-- The line search in `A` from the current `a` (`search_print_material`): the bracket, then the
golden section to 0.001 APD. -/
def lineSearchA (loss : α → α) (aLo aHi a : α) : α :=
  let br := bracketA loss aLo aHi a lineSearchFuel
  goldenSection loss br.lo br.hi lineSearchFuel

/-! ## The fixed point over (A, g, b) -/

/-- What one step of the fixed point gets from the network and the optimiser, with their state `σ`
(the optimiser's correction): from the print rendered at `(A, g, b)`, the step's residual, and the
loss with the optimiser's `g` and `b` at a trial `A`, aimed at this step's field. -/
structure PrintSearchOracle (σ α : Type) where
  step : σ → α → α → α → σ × α × (α → α × α × α)

/-- The fixed point's result: the kept `(A, g, b)` and residual, the steps taken, whether it
converged and by which rule, and the last step's move `(|ΔA| / kStop, |Δg|, |Δb|)`. -/
structure PrintSearchResult (α : Type) where
  a : α
  g : α
  b : α
  residual : α
  steps : Nat
  converged : Bool
  stoppedBy : String
  lastMove : Vec3 α

/-- The state of the fixed point between steps. -/
structure PrintSearchState (σ α : Type) where
  oracle : σ
  a : α
  g : α
  b : α
  best : Option (PrintSearchResult α)
  prevErr : Option α
  stopNext : Bool
  lastMove : Vec3 α

/-- The step's result kept against the best so far: the step's own when it has the lower residual
(or is the first), else the best so far with the step count updated. -/
def keepBest (best : Option (PrintSearchResult α)) (here : PrintSearchResult α) : PrintSearchResult α :=
  match best with
  | none => here
  | some b => if here.residual < b.residual then here else { b with steps := here.steps }

/-- The fixed point's loop: step `it` of `maxSteps` (`search_print_material`). Each step scores the
print at its `(A, g, b)` and keeps the lowest residual; it stops after a step when the parameters
moved by `tolParam` or less in the step before, or when the residual changed by `tolRelErr` of the
previous or less; otherwise it solves the next `A` by `lineSearchA` with the optimiser's `g` and
`b` there. -/
def printSearchLoop {σ : Type} (o : PrintSearchOracle σ α) (aLo aHi : α) (maxSteps : Nat) :
    Nat → Nat → PrintSearchState σ α → Option (PrintSearchResult α)
  | 0, _, s => s.best
  | fuel + 1, it, s =>
    let (os, err, optimise) := o.step s.oracle s.a s.g s.b
    let best := keepBest s.best ⟨s.a, s.g, s.b, err, it + 1, false, "steps", s.lastMove⟩
    let byResidual : Bool := match s.prevErr with
      | some prev => decide (fabs (err - prev) ≤ tolRelErr * prev)
      | none => false
    if s.stopNext ∨ byResidual then
      some { best with converged := true, stoppedBy := if s.stopNext then "parameters" else "residual",
                       lastMove := s.lastMove }
    else if maxSteps ≤ it + 1 then some { best with lastMove := s.lastMove }
    else
      let nextA := lineSearchA (fun x => (optimise x).1) aLo aHi s.a
      let (_, g', b') := optimise nextA
      let mv : Vec3 α := ⟨fabs (nextA - s.a) / stopApd, fabs (g' - s.g), fabs (b' - s.b)⟩
      printSearchLoop o aLo aHi maxSteps fuel (it + 1)
        { oracle := os, a := nextA, g := g', b := b', best := some best, prevErr := some err,
          stopNext := decide (mv.x + mv.y + mv.z ≤ tolParam), lastMove := mv }

/-- The fixed point from A0 with the screened `g0`, `b0`: at most `max(1, maxSteps)` steps; the
first step's residual is always kept. -/
def printSearch {σ : Type} (o : PrintSearchOracle σ α) (s0 : σ) (aLo aHi a0 g0 b0 : α)
    (maxSteps : Nat) : Option (PrintSearchResult α) :=
  let n := max 1 maxSteps
  printSearchLoop o aLo aHi n n 0
    { oracle := s0, a := a0, g := g0, b := b0, best := none, prevErr := none, stopNext := false,
      lastMove := ⟨0.0, 0.0, 0.0⟩ }

/-- The bracket whose `A_mid` is nearest the search's `A` (the first of equals): the result's
`target_idx`, named in the sidecar only. -/
def nearestBracket (aMid : Array α) (a : α) : Nat :=
  (List.range aMid.size).foldl
    (fun n i => if fabs (aMid.getD i 0.0 - a) < fabs (aMid.getD n 0.0 - a) then i else n) 0

/-! ## The knobs on a print -/

/-- `solve_print_knobs`' constants: the step of `d Lab / d A` (APD), the largest Newton step in `A`,
the steps in `ln g`, `ln b` and their largest Newton step, the Lab tolerance that ends a solve, and
the largest miss of a reachable target. -/
def knobDA : α := 1.0e-3
@[inherit_doc knobDA] def knobMaxDA : α := 0.1
@[inherit_doc knobDA] def knobFdStep : α := 0.05
@[inherit_doc knobDA] def knobMaxLogStep : α := 0.25
@[inherit_doc knobDA] def knobTolerance : α := 0.02
@[inherit_doc knobDA] def knobReachable : α := 0.25

/-- The knobs' Newton loop on `x = (A, ln g, ln b)` toward the Lab `target`, `render x` the grey's
Lab through the print at `x` (`solve_to` in `solve_print_knobs`): at most `fuel` steps, ending when
every component is within `knobTolerance` or the Jacobian (central differences, one-sided at an end
of `A`'s range) is singular; each step is held to `knobMaxDA` in `A` and `knobMaxLogStep` in
`ln g`, `ln b`, and `A` to `[aLo, aHi]`. -/
def knobNewton (render : Vec3 α → Vec3 α) (aLo aHi : α) (target : Vec3 α) : Nat → Vec3 α → Vec3 α → Vec3 α
  | 0, x, _ => x
  | fuel + 1, x, lab =>
    let r : Vec3 α := ⟨target.x - lab.x, target.y - lab.y, target.z - lab.z⟩
    if fabs r.x ≤ knobTolerance ∧ fabs r.y ≤ knobTolerance ∧ fabs r.z ≤ knobTolerance then x
    else
      let col (xp xm : Vec3 α) (h : α) : Vec3 α :=
        let lp := render xp
        let lm := render xm
        ⟨(lp.x - lm.x) / h, (lp.y - lm.y) / h, (lp.z - lm.z) / h⟩
      let ap := min (x.x + knobDA) aHi
      let am := max (x.x - knobDA) aLo
      let c0 := col ⟨ap, x.y, x.z⟩ ⟨am, x.y, x.z⟩ (ap - am)
      let c1 := col ⟨x.x, x.y + knobFdStep, x.z⟩ ⟨x.x, x.y - knobFdStep, x.z⟩
        ((x.y + knobFdStep) - (x.y - knobFdStep))
      let c2 := col ⟨x.x, x.y, x.z + knobFdStep⟩ ⟨x.x, x.y, x.z - knobFdStep⟩
        ((x.z + knobFdStep) - (x.z - knobFdStep))
      let j : Mat3 α := ⟨⟨c0.x, c1.x, c2.x⟩, ⟨c0.y, c1.y, c2.y⟩, ⟨c0.z, c1.z, c2.z⟩⟩
      if ¬ (1.0e-12 < fabs (Mat3.det j)) then x
      else
        let st := (Mat3.adjInv j).mulVec r
        let x' : Vec3 α :=
          ⟨clamp (x.x + clamp st.x (-knobMaxDA) knobMaxDA) aLo aHi,
           x.y + clamp st.y (-knobMaxLogStep) knobMaxLogStep,
           x.z + clamp st.z (-knobMaxLogStep) knobMaxLogStep⟩
        knobNewton render aLo aHi target fuel x' (render x')

/-- One solve toward `target` from `x0`: the point reached and its miss, the Euclidean distance in
Lab (`solve_to`; at most 30 steps). -/
def knobSolve (render : Vec3 α → Vec3 α) (aLo aHi : α) (x0 target : Vec3 α) : Vec3 α × α :=
  let x := knobNewton render aLo aHi target 30 x0 (render x0)
  let lab := render x
  (x, sqrt (sq (target.x - lab.x) + sq (target.y - lab.y) + sq (target.z - lab.z)))

/-- The largest share `f` of a target path `make f` that is reachable, by 7 halvings of `[0, 1]`
(`largest` in `solve_print_knobs`): the point of the last reachable share, else `best`. -/
def knobLargest (render : Vec3 α → Vec3 α) (aLo aHi : α) (x0 : Vec3 α) (make : α → Vec3 α) :
    Nat → α → α → Vec3 α → Vec3 α
  | 0, _, _, best => best
  | fuel + 1, lo, hi, best =>
    let f := 0.5 * (lo + hi)
    let r := knobSolve render aLo aHi x0 (make f)
    if r.2 ≤ knobReachable then knobLargest render aLo aHi x0 make fuel f hi r.1
    else knobLargest render aLo aHi x0 make fuel lo f best

/-- The knobs on a print (`solve_print_knobs`): `(A, g, b)` so that the grey the print at
`(a0, g0, b0)` renders as neutral `L*` 50 renders at `(fullL, tint, warmth)`, `knobs` being
`(brightness in stops, tint, warmth)` and `fullL` the lightness the brightness asks for
(`lstar_after_stops(50, brightness)`), and whether the target was scaled back; `none` for inputs
the C++ refuses. Knobs all zero leave `(a0, g0, b0)`. A target that
cannot be reached is scaled back: with the colour alone reachable, its lightness by halvings;
else lightness and colour together. `render x` is the grey's Lab at `x = (A, ln g, ln b)`. -/
def solvePrintKnobs (render : Vec3 α → Vec3 α) (aLo aHi a0 g0 b0 : α) (knobs : Vec3 α) (fullL : α) :
    Option (Vec3 α × Bool) :=
  let x0 : Vec3 α := ⟨a0, log g0, log b0⟩
  let out (x : Vec3 α) : Vec3 α := ⟨x.x, exp x.y, exp x.z⟩
  let full : Vec3 α := ⟨fullL, knobs.y, knobs.z⟩
  if ¬ (0.0 < g0 ∧ 0.0 < b0 ∧ aLo < aHi ∧ aLo ≤ a0 ∧ a0 ≤ aHi) then none
  else if eqA knobs.x 0.0 ∧ eqA knobs.y 0.0 ∧ eqA knobs.z 0.0 then some (out x0, false)
  else
    let rf := knobSolve render aLo aHi x0 full
    if rf.2 ≤ knobReachable then some (out rf.1, false)
    else
      let rc := knobSolve render aLo aHi x0 ⟨50.0, full.y, full.z⟩
      if rc.2 ≤ knobReachable then
        some (out (knobLargest render aLo aHi x0 (fun f => ⟨50.0 + f * (full.x - 50.0), full.y, full.z⟩)
          7 0.0 1.0 rc.1), true)
      else
        some (out (knobLargest render aLo aHi x0
          (fun f => ⟨50.0 + f * (full.x - 50.0), f * full.y, f * full.z⟩) 7 0.0 1.0 x0), true)

end Negicc
