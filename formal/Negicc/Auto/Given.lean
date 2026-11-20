/-
Given values: the single-profile solver, the brackets' exposures for -E, the roll-off at given
parameters.
-/
module

public import Negicc.Auto.Search

@[expose] public section

/-!
# Given values and a single profile

* **The single-profile solver** (`solve_dinov3_intent` in `dinov3_auto_solver.cc`), which a
  single ICC profile (`-p file.icc`) runs: the whole image resampled to the network's grid; the
  anchor `E_c` at which the median `L*` of the grid's cells is 50 (`anchorExposure`); `E` searched
  in `[2^-1.5, 2^1.5]` around it; a bootstrap render, then the fixed-point loop of
  `Negicc.Auto.Loop` with the solver's stopping rule (`solverConv`). No lightness wish and no
  correction: the field predicted from each render is used as predicted.
* **The brackets' exposures** (`meter_bracket_exposures`): each bracket's calibrated and usable
  range and its metered exposure, in the units of `-E`, so that given values default to what the
  search meters.
* **The roll-off at given parameters** (`measure_rolloff_x_hi`): the frame's 99.5 % point of the
  largest channel of the pre-shoulder render at `(E, g, b)`, mapped to code 242.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The solver's result (`DinoSolveResult`): the status (`none` for pinned, with all three
parameters given), the kept step's residual, the parameters relative to the caller's and in the
caller's units, the anchor in the caller's `E`. -/
structure SolveResult (α : Type) where
  status : Option FPStatus
  residual : α
  relative : Gains α
  absolute : Gains α
  eCenter : α
  steps : Nat

/-- The single-profile solver (`solve_dinov3_intent`) on a `w × h` image of 16-bit camera RGB
through the profile `prof` (knee `knee`, roll-off `xHi`), the caller's matrix `merged` (it holds
the caller's `E`) and gains `user`, with the pins and their values (relative, 1 by default) and at
most `maxIters` steps; `none` when an inference fails. -/
def solveIntent (predict : Array (Vec3 Nat) → IntentField α) (prof : Profile α) (knee xHi : α) (merged : Mat3 α)
    (user : Gains α) (pins : Pins) (pinVals : Gains α) (img : Array (Vec3 Nat)) (w h maxSize patch maxIters : Nat) :
    Option (SolveResult α) := do
  let s := aspectShape (α := α) w h maxSize patch
  let grid := resampleU16 (α := α) img w h s.gridW s.gridH
  let bootG : α := if pins.g then 1.0 else 1.10
  let bootB : α := if pins.b then 1.0 else 1.15
  -- 1. The anchor: the median L* of the grid rendered without the roll-off at 50.
  let anchorCfg : PipelineConfig α := ⟨prof, knee, Shoulder.off, none⟩
  let medianL := fun (e : α) =>
    upperMedian (grid.toList.map fun px => (srgbToLab (convertPixel anchorCfg (candidateMatrix merged e bootG bootB)
      (loadU16 px))).L)
  let eC : α := if pins.e then 1.0 else (anchorExposure medianL).1
  let norm := Mat3.scaleRows (Vec3.fill eC) merged
  let setup : RenderSetup α := ⟨prof, knee, setShoulder xHi, norm⟩
  let preview := fun (g : Gains α) => img.map (previewPixel setup.config (setup.matrix g))
  let predictOk := fun (pv : Array (Vec3 Nat)) =>
    let f := predict pv
    if fieldValid f s.cells then some f else none
  let cfg : OptConfig α := ⟨0.0, 0.0, pinVals.e, pinVals.g, pinVals.b⟩
  let bounds : EBounds α := ⟨pow 2.0 (-1.5), pow 2.0 1.5, 1.0⟩
  let resid := fun (pv : Array (Vec3 Nat)) (f : IntentField α) => residual pv w h s.gridW s.gridH patch f
  if pins.e && pins.g && pins.b then
    let pv := preview ⟨1.0, 1.0, 1.0⟩
    let f ← predictOk pv
    some ⟨none, resid pv f, ⟨1.0, 1.0, 1.0⟩, user, user.e * eC, 0⟩
  else
    -- 2. The bootstrap: a render at (1, 1.10, 1.15), searched from no start.
    let f0 ← predictOk (preview ⟨1.0, bootG, bootB⟩)
    let start := (optimize (intentLoss setup grid f0.mu f0.sigma) cfg none (some bounds) pins).1
    -- 3. The loop.
    let r ← fixedPoint (fun (g : Gains α) => do
        let pv := preview g
        let f ← predictOk pv
        some (g, resid pv f, f))
      (fun f g => (optimize (intentLoss setup grid f.mu f.sigma) cfg (some g) (some bounds) pins).1)
      gainsDist solverConv maxIters start
    let b := r.best.1
    some ⟨some r.status, r.best.2.1, ⟨eC * b.e, b.g, b.b⟩, ⟨user.e * eC * b.e, user.g * b.g, user.b * b.b⟩,
      user.e * eC, r.steps⟩

/-- One bracket's exposures on a frame (`BracketExposure`), in the units of `-E`. -/
structure BracketExposure (α : Type) where
  valid : Bool
  feasible : Bool
  eMin : α
  eMax : α
  eCenter : α
  searchLo : α
  searchHi : α
  hasMetered : Bool
  eMetered : α

/-- Each bracket's exposures on a frame's inner crop (`meter_bracket_exposures`): the search's
enclosure and metering (statistics at the network's default patch of 16 on a 224 grid, which
leaves `g_lo`, `g_hi` and the log-average unchanged), converted to the units of `-E` (divided by
`e_norm` on a narrow-band bundle). A bracket without a positive exposure or without patches gets
nothing. -/
def meterBracketExposures (brackets : Array (SearchBracket α)) (hasCrosstalk : Bool) (scale : FrameScale α)
    (crop : Array (Vec3 Nat)) : Array (Option (BracketExposure α)) :=
  let white := !hasCrosstalk
  let r := sceneGreenRange (α := α) crop
  let gr := sceneRange r.1 r.2
  let gScale := if white then scale.eNorm else 1.0
  let matDiv := if white then 1.0 else scale.eNorm
  let pScene := sceneVector (scale.frameMatrix.divScalar matDiv) (logAverage crop)
  let toE := if white then 1.0 else 1.0 / scale.eNorm
  (brackets.mapIdx fun i b =>
    if !(decide ((0.0 : α) < b.data.exposure)) then none
    else match bracketEnclosure b.data i (gr.1 * gScale) (gr.2 * gScale) with
      | none => none
      | some e =>
        let eMin := e.eMin * toE
        let eMax := e.eMax * toE
        let eCenter := e.eCenter * toE
        let valid := decide (0.0 < eMin) && decide (0.0 < eMax) && decide (0.0 < eCenter)
        let m := if (referenceGrey b.profile).1 then
            let x := exposureForLightness b.profile pScene 1.10 1.15
            if x.2 && decide (0.0 < x.1) then some x.1 else none
          else none
        some ⟨valid, e.feasible, eMin, eMax, eCenter, e.searchLo * toE, e.searchHi * toE, m.isSome,
          match m with | some x => clamp x e.searchLo e.searchHi * toE | none => eCenter⟩)

/-- The roll-off of a frame's inner crop at given parameters (`measure_rolloff_x_hi`): `E` in the
units of `-E`, rendered through `(M_frame / e_norm)` at `E e_norm`. -/
def measureRolloffXHi (prof : Profile α) (scale : FrameScale α) (crop : Array (Vec3 Nat)) (g : Gains α) : α :=
  let m := candidateMatrix (scale.frameMatrix.divScalar scale.eNorm) (g.e * scale.eNorm) g.g g.b
  shoulderXHiFromHist (maxHistOf (crop.map fun px => convertPixelLinear prof autoKnee m (loadU16 px)))

end Negicc
