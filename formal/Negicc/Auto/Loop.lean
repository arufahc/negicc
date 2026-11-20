/-
The fixed-point loop and the tournament's winner.
-/
module

public import Negicc.Auto.Optimizer

@[expose] public section

/-!
# The fixed-point loop

A bracket's loop (`SearchFrame::fixed_point` in `multi_target_optimizer.cc`; the single-bracket
solver's loop in `solve_dinov3_intent`, `dinov3_auto_solver.cc`) alternates two steps from its
start:

* **measure** the current parameters: render the preview, update the bracket's correction,
  predict a field, put the frame's lightness wish in its `L*`, and take the residual of the
  preview against it;
* **search** the next parameters against that field with the optimiser.

It is written here over the two steps as functions: `measure : P → Option (P × α × F)` gives the
parameters as measured (with the roll-off it measured), the residual and the field, or nothing when
the inference fails; `search : F → P → P` gives the next parameters. `dist` is the move
`|ΔE| + |Δg| + |Δb|`.

The rules, at step `it` (from 0) with residual `err` and the previous step's `prev`:

* the lowest residual so far is kept (the first of equal ones);
* **diverged** when `it > 0`, `err > 1.25 prev` and `err - prev > 1.5`, checked before the search;
* **converged** when `it > 0` and the stopping rule holds (`ConvRule`); the bundle search's rule is
  `bundleConv`, the single-bracket solver's `solverConv`;
* otherwise the next step starts from the searched parameters, at most `maxSteps` steps
  (`--dino_iters`, 8).

The loop ends after at most `maxSteps` steps (`fixedPoint_steps_le` in `Negicc.Auto.Props`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- How a loop stopped (`FixedPointStatus`). -/
inductive FPStatus
  | converged
  | diverged
  | maxIters
deriving DecidableEq, Repr

/-- The parameter tolerance: the loop stops when `|ΔE| + |Δg| + |Δb| ≤ 0.02` (`TOL_PARAM`). -/
def tolParam : α := 0.02

/-- The residual tolerance: a relative change of at most 1.5 % (`TOL_REL_ERR`). -/
def tolRelErr : α := 0.015

/-- A stopping rule: from the step index, the residual, the previous step's, the lowest so far
(this step's included), the move and the relative change, whether the loop has converged. -/
abbrev ConvRule (α : Type) := Nat → α → α → α → α → α → Bool

/-- The bundle search's rule (`fixed_point`): the parameters held still (`dp ≤ 0.02`); or, on a
narrow-band (TriLED) bundle, the residual held still within 1.5 % (in the pullback's search also not
rising); or, on a white-light bundle, from the third step, a residual at or above the previous one
while the previous one was above the lowest. -/
def bundleConv (triled pullback : Bool) : ConvRule α := fun it err prev best dp rel =>
  decide (dp ≤ tolParam) ||
  (triled && (!pullback || decide (err ≤ prev)) && decide (rel ≤ tolRelErr)) ||
  (!triled && decide (2 ≤ it) && decide (prev ≤ err) && decide (best < prev))

/-- The single-bracket solver's rule (`solve_dinov3_intent`): the parameters or the residual held
still. -/
def solverConv : ConvRule α := fun _ _ _ _ dp rel => decide (dp ≤ tolParam) || decide (rel ≤ tolRelErr)

/-- The relative change of the residual: `|err - prev| / max(1e-4, prev)` from the second step,
1 at the first. -/
def relChange (it : Nat) (err prev : α) : α :=
  if it = 0 then 1.0 else fabs (err - prev) / max 0.0001 prev

/-- The divergence test: the residual grew by more than 25 % and by more than 1.5. -/
def diverges (it : Nat) (err prev : α) : Bool :=
  it != 0 && decide (prev * 1.25 < err) && decide (1.5 < err - prev)

/-- What a loop gives back: the kept step (parameters, residual, field), how it stopped, how many
steps ran, and the last measured step's field (the bracket's correction stays at that step's). -/
structure FPResult (α P F : Type) where
  best : P × α × F
  status : FPStatus
  steps : Nat
  lastF : F

/-- The loop from step `it`, with `fuel` steps left, at parameters `curr`, the previous residual
`prev`, the step kept so far and the last measured field; `none` when an inference failed. -/
def fpGo {P F : Type} (measure : P → Option (P × α × F)) (search : F → P → P) (dist : P → P → α)
    (conv : ConvRule α) : Nat → Nat → P → α → Option ((P × α × F) × F) → Nat → Option (FPResult α P F)
  | _, 0, _, _, acc, steps => acc.map fun (b, l) => ⟨b, .maxIters, steps, l⟩
  | it, fuel + 1, curr, prev, acc, steps =>
    match measure curr with
    | none => none
    | some (cur, err, f) =>
      let best' : P × α × F := match acc with
        | some (b, _) => if err < b.2.1 then (cur, err, f) else b
        | none => (cur, err, f)
      if diverges it err prev then some ⟨best', .diverged, steps + 1, f⟩
      else
        let next := search f cur
        let dp := dist next cur
        let rel := relChange it err prev
        if it != 0 && conv it err prev best'.2.1 dp rel then some ⟨best', .converged, steps + 1, f⟩
        else fpGo measure search dist conv (it + 1) fuel next err (some (best', f)) (steps + 1)

/-- A bracket's fixed-point loop from `start`, at most `maxSteps` steps (at least one). -/
def fixedPoint {P F : Type} (measure : P → Option (P × α × F)) (search : F → P → P) (dist : P → P → α)
    (conv : ConvRule α) (maxSteps : Nat) (start : P) : Option (FPResult α P F) :=
  fpGo measure search dist conv 0 (max 1 maxSteps) start 0.0 none 0

/-- The move between two parameter sets: `|ΔE| + |Δg| + |Δb|`. -/
def gainsDist (a b : Gains α) : α := fabs (a.e - b.e) + fabs (a.g - b.g) + fabs (a.b - b.b)

/-! ## The winner -/

/-- The winner of a tournament (`winner`): the first run with the lowest score, among the converged
runs when `preferConverged` and any run converged. -/
def winner {R : Type} (runs : List R) (preferConverged : Bool) (conv : R → Bool) (score : R → α) : Option R :=
  let anyConv := preferConverged && runs.any conv
  minFirst (fun a b => decide (score a < score b)) (runs.filter fun a => !anyConv || conv a)

end Negicc
