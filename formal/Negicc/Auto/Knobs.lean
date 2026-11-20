/-
The knobs: brightness, tint and warmth on a search's result.
-/
module

public import Negicc.Auto.Metering

@[expose] public section

/-!
# The knobs

`--brightness`, `--tint` and `--warmth` move a result afterwards (`solve_grey_knobs` in
`multi_target_optimizer.cc`, DESIGN.md 4.2 "Given values"). The bracket's reference grey (the
input it renders as neutral `L*` 50, `referenceGrey`) is moved to the target `L*` of 50 brightened
by the given stops of luminance (`lstarAfterStops`), with the given `a*` and `b*`: a Newton solve
from the reference grey finds the input that renders at the target, and the gains are scaled by
the input's change (`greyGains`). A target the solve does not reach within 0.25 is scaled back:
the colour change kept and the brightness bisected, or, when the colour alone is out of reach, both
together, each by 7 halvings.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The relative luminance of a lightness `L*`. -/
def yOfLstar (L : α) : α := if 8.0 < L then pow ((L + 16.0) / 116.0) 3.0 else L / labKappa

/-- The `L*` of a grey of lightness `L` whose luminance is multiplied by `2^stops`
(`lstar_after_stops`). -/
def lstarAfterStops (L stops : α) : α :=
  let Y := yOfLstar L * pow 2.0 stops
  if labEpsilon < Y then 116.0 * cbrt Y - 16.0 else labKappa * Y

/-- How far, in stops of luminance, a grey at `L*` lies from the `L*` the brightness asks for
(`knob_grey_miss_stops`). -/
def knobGreyMissStops (brightness greyL : α) : α :=
  log2 (yOfLstar greyL / yOfLstar (lstarAfterStops 50.0 brightness))

/-- The gains `(E, g, b)` that move the grey the parameters `base` render as neutral `L*` 50
(reference input `exp(log t₀)`) to `target`, and the Newton solve's miss (`GreyPoint::gains`): a
16-step Newton solve from `log t₀`, then each channel's gain scaled by its input's change. -/
def greyGains (p : Profile α) (logT0 : Vec3 α) (base : Vec3 α) (target : Lab α) : Vec3 α × α :=
  let (lt, err) := newton p target 16 logT0
  let lb : Vec3 α := ⟨log base.x, log (base.x * base.y), log (base.x * base.z)⟩
  let G : Vec3 α := ⟨exp (lb.x + lt.x - logT0.x), exp (lb.y + lt.y - logT0.y), exp (lb.z + lt.z - logT0.z)⟩
  (⟨G.x, G.y / G.x, G.z / G.x⟩, err)

/-- The CIELAB of the grey at gains `egb` (`GreyPoint::grey`). -/
def greyAt (p : Profile α) (logT0 base egb : Vec3 α) : Lab α :=
  let lb : Vec3 α := ⟨log base.x, log (base.x * base.y), log (base.x * base.z)⟩
  let le : Vec3 α := ⟨log egb.x, log (egb.x * egb.y), log (egb.x * egb.z)⟩
  greyLstarLab p ⟨exp (logT0.x + le.x - lb.x), exp (logT0.y + le.y - lb.y), exp (logT0.z + le.z - lb.z)⟩
where
  /-- The CIELAB of one input. -/
  greyLstarLab (p : Profile α) (t : Vec3 α) : Lab α := (greyLab p [t]).headD ⟨0.0, 0.0, 0.0⟩

/-- A target counts as reached when the Newton solve lands within 0.25 of it. -/
def knobReachable : α := 0.25

/-- The largest share `f` (7 halvings of `[0, 1]`) of a change that is reachable, and the gains
of the last reachable share; none when no share is (`largest`). -/
def largestReachable (solve : Lab α → Vec3 α × α) (target : α → Lab α) : Nat → α → α → Option (Vec3 α) → Option (Vec3 α)
  | 0, _, _, best => best
  | n + 1, lo, hi, best =>
    let f := 0.5 * (lo + hi)
    let (r, err) := solve (target f)
    if err ≤ knobReachable then largestReachable solve target n f hi (some r)
    else largestReachable solve target n lo f best

/-- The result of the knobs (`KnobSolveResult`): the gains, whether the target was out of reach,
and the grey's CIELAB at the gains. -/
structure KnobResult (α : Type) where
  egb : Vec3 α
  limited : Bool
  greyLab : Lab α

/-- The knobs on parameters `base` through a bracket (`solve_grey_knobs`); `none` when a parameter
is not positive or the bracket has no neutral grey. `greyProf` is the profile whose neutral grey
is moved (the bracket's own by default). -/
def solveGreyKnobs (p : Profile α) (base : Vec3 α) (brightness tint warmth : α) (greyProf : Option (Profile α) := none) :
    Option (KnobResult α) :=
  if !(decide (0.0 < base.x) && decide (0.0 < base.y) && decide (0.0 < base.z)) then none
  else
    let (ok, logT0) := referenceGrey (greyProf.getD p)
    if !ok then none
    else if scalarEq brightness 0.0 && scalarEq tint 0.0 && scalarEq warmth 0.0 then
      some ⟨base, false, greyAt p logT0 base base⟩
    else
      let solve := greyGains p logT0 base
      let full : Lab α := ⟨lstarAfterStops 50.0 brightness, tint, warmth⟩
      let (egb, err) := solve full
      if err ≤ knobReachable then some ⟨egb, false, greyAt p logT0 base egb⟩
      else
        let (cEgb, cErr) := solve ⟨50.0, tint, warmth⟩
        let best :=
          if cErr ≤ knobReachable then
            (largestReachable solve (fun f => ⟨50.0 + f * (full.L - 50.0), tint, warmth⟩) 7 0.0 1.0 none).getD cEgb
          else
            (largestReachable solve (fun f => ⟨50.0 + f * (full.L - 50.0), f * tint, f * warmth⟩) 7 0.0 1.0 none).getD
              base
        some ⟨best, true, greyAt p logT0 base best⟩

end Negicc
