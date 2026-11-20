/-
The (E, g, b) optimiser: a coordinate descent against a fixed intent field.
-/
module

public import Negicc.Auto.Enclosure

@[expose] public section

/-!
# The (E, g, b) optimiser

`optimize_impl` in `dinov3_auto_solver.cc` (called through `optimize_target_for_intent`) finds
the exposure `E` and the gains `g`, `b` of one bracket against a fixed intent field. It sees the
field only through the loss of a candidate (`intent_loss`, `intentLoss`), so here it is written
over any loss function `loss : Gains α → IntentLoss α`; the search passes the bracket's
`intentLoss` against the field it aims at.

1. **Exposure.** Every candidate of a list (`eCandidates`) is scored on the `L*` loss plus the
   penalties (`penalty`); the first lowest wins. Then `E ± 1` and `± 2` steps (`eStep`, 1/40 of
   the window, at least 0.01), each inside the window, from that winner: a step replaces it when
   it scores strictly lower.
2. **Gains.** Every `g` of a grid at the chosen `E` and the start `b`, scored on the `a*` loss;
   then every `b` of a grid at the chosen `E` and `g`, on the `b*` loss; the first lowest wins.
3. **Refinement.** `g - 0.02`, `g + 0.02`, `b - 0.02`, `b + 0.02` in turn, each taken when the sum
   of the `a*` and `b*` losses falls below the best so far (the first sum is the `a*` loss of
   step 2's `g` plus the `b*` loss of its `b`).
4. In the highlight pullback's screening (enclosure penalties and no start), steps 1 to 3 run a
   second time from the result, and step 1 once more when the gains moved.
5. The result's score: the mean of its three losses plus its penalties.

A pinned parameter keeps its start value. The penalties are `15 ((clip - 2.5) / 2)²` once more
than 2.5 % of the cells' channels clip, and either `5 log2(E / centre)²` (no pull when the centre
is 0) or, in the pullback's search, the enclosure penalties (`enclosurePenalty`).

The search state is in double precision and a candidate is rounded to single precision only where
its loss is computed; that rounding is not modelled.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- Which parameters are held (`pin_mask`: `PIN_E`, `PIN_G`, `PIN_B`). -/
structure Pins where
  e : Bool := false
  g : Bool := false
  b : Bool := false

/-- What the optimiser takes besides the loss (`SolverConfig`): the calibrated range for the
pullback's enclosure penalties (`0` when not in the pullback) and the values pinned parameters
take when there is no start. -/
structure OptConfig (α : Type) where
  encMin : α
  encMax : α
  pinE : α
  pinG : α
  pinB : α

/-- The optimiser runs as the highlight pullback's search: its enclosure is set. -/
def OptConfig.pullback (c : OptConfig α) : Bool := decide (0.0 < c.encMin) && decide (0.0 < c.encMax)

/-- The window `E` is searched in and the centre the penalty pulls toward: `(lo, hi, centre)`. -/
structure EBounds (α : Type) where
  lo : α
  hi : α
  centre : α

/-- The clip penalty: `15 ((clip - 2.5) / 2)²` above 2.5 % of the channels, else 0. -/
def clipPenalty (clip : α) : α := 15.0 * sq (max 0.0 (clip - 2.5) / 2.0)

/-- The pull toward the centre: `5 log2(E / centre)²`, none for a centre of 0 or less. -/
def centrePenalty (e centre : α) : α := if 0.0 < centre then 5.0 * sq (log2 (e / centre)) else 0.0

/-- The penalties of a candidate (`compute_penalties`): the clip penalty plus, in the pullback's
search (with `E > 0`), the enclosure penalties, otherwise the pull toward the centre. -/
def penalty (cfg : OptConfig α) (e clip centre : α) : α :=
  if cfg.pullback && decide (0.0 < e) then clipPenalty clip + enclosurePenalty cfg.encMin cfg.encMax e
  else clipPenalty clip + centrePenalty e centre

/-- The first element of a list with the lowest score, and that score: a later element replaces
the current one only when it scores strictly lower. -/
def firstMin {β : Type} (score : β → α) : List β → Option (β × α)
  | [] => none
  | x :: xs => some (xs.foldl (fun best y => let s := score y; if s < best.2 then (y, s) else best) (x, score x))

/-- One step of leaving out repeats from the right: `x` is put in front unless it equals the
first element already kept. -/
def dedupStep (x : α) (acc : List α) : List α :=
  match acc with
  | y :: _ => if scalarEq x y then acc else x :: acc
  | [] => [x]

/-- Sorted with repeated values left out (`std::sort` then `std::unique`). -/
def sortUnique (xs : List α) : List α :=
  (xs.mergeSort fun a b => decide (a ≤ b)).foldr dedupStep []

/-- The exposure candidates and the window and centre they belong to (`e_cands_d`, `e_min`,
`e_max`, `e_center`):
* pinned `E`: the start's `E` (or the configured one), alone;
* a window: 13 equally spaced points from `lo` to `hi` (`np.linspace`), with the start's
  `E + {-0.06, -0.03, 0, 0.03, 0.06}` that fall inside it, sorted, repeats left out;
* no window, a start: the start's `E` held to `[0.2, 3]`, plus `{-0.15, …, 0.15}`, those inside
  `[0.2, 3]`, around the held start;
* neither: eleven fixed points from 0.4 to 1.7, in `[0.4, 1.8]` around 1. -/
def eCandidates (cfg : OptConfig α) (pins : Pins) (initial : Option (Gains α)) (bounds : Option (EBounds α)) :
    List α × EBounds α :=
  if pins.e then
    let e := match initial with | some i => i.e | none => cfg.pinE
    ([e], ⟨e, e, e⟩)
  else match bounds with
  | some bd =>
    let step := (bd.hi - bd.lo) / 12.0
    let lin := (List.range 12).map (fun (i : Nat) => bd.lo + (i : α) * step) ++ [bd.hi]
    let extra := match initial with
      | some i => ([-0.06, -0.03, 0.0, 0.03, 0.06] : List α).filterMap fun d =>
          let c := i.e + d
          if bd.lo ≤ c ∧ c ≤ bd.hi then some c else none
      | none => []
    (sortUnique (lin ++ extra), bd)
  | none => match initial with
    | some i =>
      let c0 := clamp i.e 0.20 3.0
      ((([-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15] : List α).map (c0 + ·)).filter
        (fun c => decide (0.20 ≤ c) && decide (c ≤ 3.0)), ⟨0.20, 3.0, c0⟩)
    | none => ([0.40, 0.50, 0.60, 0.70, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70], ⟨0.40, 1.80, 1.0⟩)

/-- The refinement step of `E`: 1/40 of the window, at least 0.01; 0.02 for an empty window. -/
def eStep (bd : EBounds α) : α := if bd.lo < bd.hi then max 0.01 ((bd.hi - bd.lo) / 40.0) else 0.02

/-- One refinement step of `E` from `base`: `base + de`, outside the window skipped, taken when it
scores strictly lower than the best so far. -/
def eRefineStep (score : α → α) (bd : EBounds α) (base : α) (best : α × α) (de : α) : α × α :=
  let t := base + de
  if t < bd.lo ∨ bd.hi < t then best
  else let s := score t; if s < best.2 then (t, s) else best

/-- The refinement moves of `E`: `-2`, `-1`, `+1`, `+2` steps. -/
def eMoves (bd : EBounds α) : List α := [-2.0 * eStep bd, -eStep bd, eStep bd, 2.0 * eStep bd]

/-- Step 1 at gains `g`, `b`: the sweep over the candidates on `L*` plus the penalties, then, unless
`E` is pinned, the steps of `± 1` and `± 2` `eStep` from the sweep's winner, each inside the
window, taken when strictly lower. The chosen `E` and its score. -/
def ePass (loss : Gains α → IntentLoss α) (cfg : OptConfig α) (refine : Bool) (cands : List α) (bd : EBounds α)
    (g b : α) : α × α :=
  let score := fun (e : α) => let r := loss ⟨e, g, b⟩; r.L + penalty cfg e r.clip bd.centre
  let p0 := (firstMin score cands).getD (bd.lo, score bd.lo)
  if !refine then p0 else (eMoves bd).foldl (eRefineStep score bd p0.1) p0

/-- The `g` grid: the start's `g` when pinned; with a start, `g ± 0.02, ± 0.04`; in the pullback's
screening, 0.74 to 1.28 by 0.06; otherwise 0.98 to 1.28 by 0.06. -/
def gGrid (pins : Pins) (pullback : Bool) (initial : Option (Gains α)) (currG : α) : List α :=
  if pins.g then [currG]
  else match initial with
  | some i => ([-0.04, -0.02, 0.0, 0.02, 0.04] : List α).map (i.g + ·)
  | none => if pullback then [0.74, 0.80, 0.86, 0.92, 0.98, 1.04, 1.10, 1.16, 1.22, 1.28]
            else [0.98, 1.04, 1.10, 1.16, 1.22, 1.28]

/-- The `b` grid: the start's `b` when pinned; with a start, `b ± 0.03, ± 0.06`; in the pullback's
screening, 0.65 to 1.35 by 0.10; otherwise 0.85 to 1.35 by 0.10. -/
def bGrid (pins : Pins) (pullback : Bool) (initial : Option (Gains α)) (currB : α) : List α :=
  if pins.b then [currB]
  else match initial with
  | some i => ([-0.06, -0.03, 0.0, 0.03, 0.06] : List α).map (i.b + ·)
  | none => if pullback then [0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]
            else [0.85, 0.95, 1.05, 1.15, 1.25, 1.35]

/-- Step 2's sweep of `g` at `E`, `b` on the `a*` loss: the first lowest, and its loss (the grids
are never empty). -/
def gSweep (loss : Gains α → IntentLoss α) (grid : List α) (e b : α) : α × α :=
  (firstMin (fun g => (loss ⟨e, g, b⟩).a) grid).getD (0.0, 0.0)

/-- Step 2's sweep of `b` at `E`, `g` on the `b*` loss: the first lowest, and its loss. -/
def bSweep (loss : Gains α → IntentLoss α) (grid : List α) (e g : α) : α × α :=
  (firstMin (fun b => (loss ⟨e, g, b⟩).b) grid).getD (0.0, 0.0)

/-- The four refinement moves of step 3, `(dg, db)`. -/
def chainMoves : List (α × α) := [(-0.02, 0.0), (0.02, 0.0), (0.0, -0.02), (0.0, 0.02)]

/-- One refinement move of step 3 from `(g, b)` with the sum `best` to beat: skipped when it moves
a pinned gain, taken when the sum of the `a*` and `b*` losses is strictly lower. -/
def chainStep (loss : Gains α → IntentLoss α) (pins : Pins) (e : α) (st : α × α × α) (d : α × α) : α × α × α :=
  if (pins.g && !scalarEq d.1 0.0) || (pins.b && !scalarEq d.2 0.0) then st
  else
    let r := loss ⟨e, st.1 + d.1, st.2.1 + d.2⟩
    if r.a + r.b < st.2.2 then (st.1 + d.1, st.2.1 + d.2, r.a + r.b) else st

/-- Step 3 from `(g, b)` with the sum `best` to beat: the four moves in turn (`chainStep`). The
gains and the last accepted sum. -/
def chain (loss : Gains α → IntentLoss α) (pins : Pins) (e g b best : α) : α × α × α :=
  (chainMoves (α := α)).foldl (chainStep loss pins e) (g, b, best)

/-- Step 4, the pullback screening's second pass from step 3's result: step 1 again (unless `E` is
pinned), the `g` and `b` sweeps again (unless pinned), and step 3 from the loss at that point. -/
def secondPass (loss : Gains α → IntentLoss α) (cfg : OptConfig α) (pins : Pins) (ec : List α × EBounds α)
    (gs bs : List α) (e1 g3 b3 : α) : Gains α :=
  let e' := if pins.e then e1 else (ePass loss cfg true ec.1 ec.2 g3 b3).1
  let g' := if pins.g then g3 else (gSweep loss gs e' b3).1
  let b' := if pins.b then b3 else (bSweep loss bs e' g').1
  let r0 := loss ⟨e', g', b'⟩
  let c := chain loss pins e' g' b' (r0.a + r0.b)
  ⟨e', c.1, c.2.1⟩

/-- The optimiser (`optimize_impl`, `optimize_target_for_intent`): the chosen `(E, g, b)` and its
score, the mean of its three losses plus its penalties. -/
def optimize (loss : Gains α → IntentLoss α) (cfg : OptConfig α) (initial : Option (Gains α))
    (bounds : Option (EBounds α)) (pins : Pins) : Gains α × α :=
  let pb := cfg.pullback
  let currG := match initial with | some i => i.g | none => if pins.g then cfg.pinG else 1.22
  let currB := match initial with | some i => i.b | none => if pins.b then cfg.pinB else 1.15
  let ec := eCandidates cfg pins initial bounds
  -- 1. exposure
  let e1 := (ePass loss cfg (!pins.e) ec.1 ec.2 currG currB).1
  -- 2. gains
  let gs := gGrid pins pb initial currG
  let bs := bGrid pins pb initial currB
  let s2g := gSweep loss gs e1 currB
  let s2b := bSweep loss bs e1 s2g.1
  -- 3. refinement
  let c3 := chain loss pins e1 s2g.1 s2b.1 (s2g.2 + s2b.2)
  -- 4. the pullback's screening: a second pass
  let r4 : Gains α := if pb && initial.isNone then secondPass loss cfg pins ec gs bs e1 c3.1 c3.2.1
    else ⟨e1, c3.1, c3.2.1⟩
  -- re-align E when the pullback's gains moved
  let e5 :=
    if pb && !pins.e && (decide (0.000001 < fabs (r4.g - currG)) || decide (0.000001 < fabs (r4.b - currB))) then
      (ePass loss cfg true ec.1 ec.2 r4.g r4.b).1
    else r4.e
  -- 5. the result's score
  let r := loss ⟨e5, r4.g, r4.b⟩
  (⟨e5, r4.g, r4.b⟩, r.mean + penalty cfg e5 r.clip ec.2.centre)

end Negicc
