/-
The Auto search: the bracket and (E, g, b) of a frame.
-/
module

public import Negicc.Auto.Metering

@[expose] public section

/-!
# The Auto search

`search_conversion_params` in `multi_target_optimizer.cc` (DESIGN.md 4.2) chooses the bracket and
`(E, g, b)` of a frame from its inner crop. It is written here over the intent network as a
function (`predict`, a preview to a field), so that everything but the network's weights is
stated. The steps:

1. **Brackets and their ranges** (`filterAdmissible`, `bracketRange`); with a pinned bracket,
   that bracket alone.
2. **Metering and the lightness wish**: each admissible bracket (every one when none is) metered
   (`referenceGrey`, `exposureForLightness`); the bootstrap field is the mean of the fields
   predicted from every metered bracket's render at its metered `E` held to its window; its `L*`
   is the frame's lightness wish. With no bracket metered, or `E` pinned, one render makes it.
3. **Screening**: each metered bracket (each of the pool when none metered) optimised against the
   bootstrap field from no start, ranked by its loss (a stable sort).
4. **Tournament**: the best two run the fixed-point loop in their trust region; the winner is the
   lowest residual, on a narrow-band bundle among the converged runs when any converged.
5. **Clip fallback** (`clipFallbackVerdict`).
6. **Highlight pullback** (`pullbackTier`, `pullbackNudge`, `pullbackAccept`).
7. **Final roll-off**: `x_hi` measured once more at the final parameters.

The search keeps one piece of state between its steps: each bracket's last correction
(`SearchFrame::update_correction`). It is threaded through as `Corr`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The decisions -/

/-- What the clip fallback did (`ClipFallbackOutcome`). -/
inductive ClipFallback
  | notNeeded
  | notScreened
  | notAdmissible
  | clipsTooMuch
  | shiftsLightness
  | worseResidual
  | taken
deriving DecidableEq, Repr

/-- The clip fallback is tried when the chosen render white-clips more than 0.03 % of the crop and
the bracket is not pinned (`CLIP_FALLBACK_CLIP_TAU`). -/
def clipFallbackNeeded (clip0 : α) (pinned : Bool) : Bool := decide (0.0003 < clip0) && !pinned

/-- The verdict on the next bracket's run (`ClipFallbackOutcome`): it must at least halve the
clipping, move the median `L*` by no more than −4.5 to +0.5, and have a residual no more than 10 %
higher, in that order. -/
def clipFallbackVerdict (clip0 clipAdj dl residAdj residChosen : α) : ClipFallback :=
  if 0.5 * clip0 < clipAdj then .clipsTooMuch
  else if !(decide (-4.5 ≤ dl) && decide (dl ≤ 0.5)) then .shiftsLightness
  else if (1.0 + 0.10) * residChosen < residAdj then .worseResidual
  else .taken

/-- The highlight pullback's gate: median `L*` at least 58 with at least 2.2 % of the crop above
sRGB 242 in a channel, or median `L*` at least 48 with at least 7.5 % above 242 or, on bracket
index 3 and later, at least 11 % above 235. -/
def pullbackTier (medianL exc242 exc235 : α) (idx : Nat) : Bool :=
  (decide (58.0 ≤ medianL) && decide (0.022 ≤ exc242)) ||
  (decide (48.0 ≤ medianL) && (decide (0.075 ≤ exc242) || (decide (3 ≤ idx) && decide (0.110 ≤ exc235))))

/-- The nudge on a feasible bracket: `E` raised toward `e_min` by at most 0.15 stop, never past
`e_max`; none when that does not raise it. -/
def pullbackNudge (eMin eMax e : α) : Option α :=
  let ep := min eMax (min eMin (e * pow 2.0 0.15))
  if e < ep then some ep else none

/-- How far the re-search may darken the midtones: 7.5 `L*` on a frame with median `L*` at least
58, 4.5 otherwise. -/
def pullbackDlMin (medianL0 : α) : α := if 58.0 ≤ medianL0 then -7.5 else -4.5

/-- The re-search's exposure after the darkening check: lowered (brightening) by 0.038 stop per
`L*` of darkening beyond `dlMin`, once. -/
def pullbackBrighten (e dl dlMin : α) : α := e * pow 2.0 ((dl - dlMin) * 0.038)

/-- The re-search's result is taken when the share above 242 grows by at most 0.5 percentage
point, the median `L*` stays at least `min(48, first median − 6)`, and it moves to a later bracket
or raises `E` by more than 2 %. -/
def pullbackAccept (excRes exc0 medRes med0 : α) (idxRes idx0 : Nat) (eRes e0 : α) : Bool :=
  decide (excRes ≤ exc0 + 0.005) && decide (min 48.0 (med0 - 6.0) ≤ medRes) &&
    (decide (idx0 < idxRes) || decide (e0 * 1.02 < eRes))

/-! ## The search's data -/

/-- One bracket as the search reads it: its patch statistics and its film profile. -/
structure SearchBracket (α : Type) where
  data : Bracket α
  profile : Profile α

/-- Parameters with their roll-off (`ConversionParams`): `E`, `g`, `b` and the shoulder's `x_hi`
(1: off). -/
structure Params (α : Type) where
  gains : Gains α
  shoulder : α

/-- The search's input: the bundle's brackets, whether it is narrow-band (it has a crosstalk
matrix), the frame's scale, the frame's inner crop and the network's geometry, the loop's step
limit and the pins (`ParamSearchOptions`). `pinE` is in the search's units. -/
structure SearchInput (α : Type) where
  brackets : Array (SearchBracket α)
  hasCrosstalk : Bool
  scale : FrameScale α
  crop : Array (Vec3 Nat)
  cw : Nat
  ch : Nat
  maxSize : Nat
  patch : Nat
  maxSteps : Nat
  pinTarget : Option Nat
  pins : Pins
  pinE : α
  pinG : α
  pinB : α

/-- A screened bracket (`ScreeningResult` with its enclosure). -/
structure Screened (α : Type) where
  enc : BracketRange α
  params : Params α
  loss : α
  lo : α
  hi : α
  eMet : α

/-- A bracket's fixed-point run (`FixedPointResult`): the kept step's parameters, residual and
field, how it stopped and how many steps ran. -/
structure Run (α : Type) where
  idx : Nat
  params : Params α
  residual : α
  status : FPStatus
  steps : Nat
  field : IntentField α
  /-- The shoulder guard, recorded only: the kept step's `L*` loss is more than 2 % worse with its
  roll-off than without (`shoulder_guard`). -/
  guard : Bool

/-- The search's result (`ParamSearchResult`): the bracket, `(E, g, b)` in the frame's units with
the final roll-off, the kept step's residual and status, what the clip fallback and the pullback
did, the calibrated range in the frame's units (0 for an infeasible bracket), the screening rows
and the tournament's runs. -/
structure SearchResult (α : Type) where
  idx : Nat
  params : Params α
  residual : α
  status : FPStatus
  clipFallback : ClipFallback
  clipFallbackWhiteClip : α
  highlightPullback : Bool
  shoulderGuard : Bool
  eRangeLo : α
  eRangeHi : α
  screening : List (Screened α)
  tournament : List (Run α)

/-- Each bracket's last correction. -/
abbrev Corr (α : Type) := Array (Option (Array (Lab α)))

/-- The search's computations: they read and update the corrections, and fail when an inference
fails. -/
abbrev SM (α : Type) := StateT (Corr α) Option

namespace SearchInput

variable (inp : SearchInput α)

/-- The bootstrap's gains: 1.10 and 1.15, or the pinned ones (`BOOTSTRAP_G`, `BOOTSTRAP_B`). -/
def bootG : α := if inp.pins.g then inp.pinG else 1.10

@[inherit_doc bootG] def bootB : α := if inp.pins.b then inp.pinB else 1.15

/-- On a narrow-band bundle the search runs on `M_frame / e_norm` and the scene's range is used as
it is; on a white-light bundle on `M_frame`, with the scene's range scaled by `e_norm`
(`SearchFrame::set_frame_matrix`). -/
def inputMatrix : Mat3 α :=
  if inp.hasCrosstalk then inp.scale.frameMatrix.divScalar inp.scale.eNorm else inp.scale.frameMatrix

/-- The factor on the scene's range (`g_scale`). -/
def gScale : α := if inp.hasCrosstalk then 1.0 else inp.scale.eNorm

/-- The frame's statistics over the crop. -/
def stats : FrameStats α := frameStats inp.crop inp.cw inp.ch inp.maxSize inp.patch

/-- The scene's range as the enclosures read it: `(g_lo, g_hi)` held as in `sceneRange`, times
`g_scale`. -/
def gRef : α × α :=
  let r := sceneRange inp.stats.gLo inp.stats.gHi
  (r.1 * inp.gScale, r.2 * inp.gScale)

/-- The bracket at an index (a bracket without patches and with an empty profile outside the
bundle). -/
def bracket (i : Nat) : SearchBracket α :=
  inp.brackets.getD i ⟨⟨false, 0.0, 0.0, 0.0, 0.0⟩,
    ⟨⟨#[], #[], #[]⟩, Mat3.identity, Vec3.fill 0.0, ⟨0, 0, 0⟩, #[], ⟨#[], #[], #[]⟩⟩⟩

/-- The enclosure of a bracket on this frame. -/
def enclosureOf (i : Nat) : Option (BracketRange α) :=
  bracketRange (inp.bracket i).data i inp.gRef.1 inp.gRef.2

/-- What bracket `i` renders a candidate with at roll-off `xHi`. -/
def setup (i : Nat) (xHi : α) : RenderSetup α :=
  ⟨(inp.bracket i).profile, autoKnee, setShoulder xHi, inp.inputMatrix⟩

/-- The pipeline configuration of a render of bracket `i` at roll-off `xHi`. -/
def config (i : Nat) (xHi : α) : PipelineConfig α := (inp.setup i xHi).config

/-- The 8-bit preview of the crop through bracket `i` at `p` (`render_preview`). -/
def preview (i : Nat) (p : Params α) : Array (Vec3 Nat) :=
  inp.crop.map (previewPixel (inp.config i p.shoulder) (candidateMatrix inp.inputMatrix p.gains.e p.gains.g p.gains.b))

/-- The roll-off at `p` (`measure_shoulder`): the 99.5 % point of the largest channel of the crop's
pre-shoulder render, mapped to code 242. -/
def measureShoulder (i : Nat) (p : Gains α) : α :=
  let m := candidateMatrix inp.inputMatrix p.e p.g p.b
  shoulderXHiFromHist (maxHistOf (inp.crop.map fun px =>
    convertPixelLinear (inp.bracket i).profile autoKnee m (loadU16 px)))

/-- The highlight statistics of the crop through bracket `i` at `p` with one 8-bit ceiling on all
channels (`highlight_clip`). -/
def clip (i : Nat) (p : Params α) (ceil : α) : ClipStats α :=
  clipStats (inp.config i p.shoulder) (candidateMatrix inp.inputMatrix p.gains.e p.gains.g p.gains.b) inp.crop
    (Vec3.fill (ceilLin ceil))

/-- The bracket's correction from a preview at `p`: its trained cells minus the optimiser's cells. -/
def correctionAt (i : Nat) (p : Params α) (pv : Array (Vec3 Nat)) : Array (Lab α) :=
  let s := inp.stats.shape
  correction (trainedCellsLab pv inp.cw inp.ch s.gridW s.gridH inp.patch)
    (intentCells (inp.setup i p.shoulder) inp.stats.grid p.gains)

/-- Renders bracket `i` at `p` and keeps its correction (`render_preview`, `update_correction`). -/
def renderAndCorrect (i : Nat) (p : Params α) : SM α (Array (Vec3 Nat)) := do
  let pv := inp.preview i p
  modify fun c => c.set! i (some (inp.correctionAt i p pv))
  pure pv

/-- The field predicted from a preview, when it fits the grid (`predict`). -/
def predictChecked (predict : Array (Vec3 Nat) → IntentField α) (pv : Array (Vec3 Nat)) : Option (IntentField α) :=
  let f := predict pv
  if fieldValid f inp.stats.shape.cells then some f else none

/-- The optimiser's configuration for bracket `i` (pullback: the enclosure's calibrated range). -/
def optConfig (i : Nat) (pullback : Bool) : OptConfig α :=
  let enc : α × α := match (if pullback then inp.enclosureOf i else none) with
    | some e => (e.eMin, e.eMax)
    | none => (0.0, 0.0)
  ⟨enc.1, enc.2, inp.pinE, inp.pinG, inp.pinB⟩

/-- The optimiser on bracket `i` against a field (`SearchFrame::search`), aimed at the field minus
the bracket's correction `corr`, from `initial` (pinned values substituted) within `eb`: the
parameters (with the start's roll-off, 1 without a start) and the loss. -/
def search (corr : Option (Array (Lab α))) (i : Nat) (f : IntentField α) (initial : Option (Params α))
    (eb : EBounds α) (pullback : Bool) : Params α × α :=
  let xHi := match initial with | some p => p.shoulder | none => 1.0
  let init := initial.map fun p =>
    (⟨if inp.pins.e then inp.pinE else p.gains.e, if inp.pins.g then inp.pinG else p.gains.g,
      if inp.pins.b then inp.pinB else p.gains.b⟩ : Gains α)
  let loss := intentLoss (inp.setup i xHi) inp.stats.grid (aimField f.mu corr) f.sigma
  let (g, l) := optimize loss (inp.optConfig i pullback) init (some eb) inp.pins
  (⟨g, xHi⟩, l)

/-- One measured step of bracket `i`'s loop at `p`: the roll-off measured, the preview, its
correction, the predicted field with the lightness wish in its `L*`, and the residual
(`fixed_point`). -/
def measureStep (predict : Array (Vec3 Nat) → IntentField α) (wish : Array α × Array α) (i : Nat) (p : Params α) :
    Option (Params α × α × (IntentField α × Array (Lab α))) := do
  let cur : Params α := ⟨p.gains, inp.measureShoulder i p.gains⟩
  let pv := inp.preview i cur
  let corr := inp.correctionAt i cur pv
  let f ← inp.predictChecked predict pv
  let fw := withWish wish.1 wish.2 f
  let s := inp.stats.shape
  some (cur, residual pv inp.cw inp.ch s.gridW s.gridH inp.patch fw, (fw, corr))

/-- Bracket `i`'s fixed-point loop from `start` within the trust region `eb` (`fixed_point`); the
bracket's correction is left at its last step's. -/
def runLoop (predict : Array (Vec3 Nat) → IntentField α) (wish : Array α × Array α) (i : Nat) (start : Params α)
    (eb : EBounds α) (pullback : Bool) : SM α (Run α) := do
  let lo := min eb.lo eb.hi
  let hi := max eb.lo eb.hi
  let eb' : EBounds α := ⟨lo, hi, eb.centre⟩
  let res ← liftM (fixedPoint (inp.measureStep predict wish i)
    (fun fc p => (inp.search (some fc.2) i fc.1 (some p) eb' pullback).1)
    (fun a b => gainsDist a.gains b.gains) (bundleConv inp.hasCrosstalk pullback) inp.maxSteps start)
  modify fun c => c.set! i (some res.lastF.2)
  let best := res.best.1
  let f := res.best.2.2.1
  let lossL (xHi : α) : α := (intentLoss (inp.setup i xHi) inp.stats.grid f.mu f.sigma best.gains).L
  let guard := decide (shoulderOffLimit < best.shoulder) && decide (1.02 * lossL 1.0 < lossL best.shoulder)
  pure ⟨i, best, res.best.2.1, res.status, res.steps, res.best.2.2.1, guard⟩
where
  /-- Lifts an optional value into the search. -/
  liftM {β : Type} (o : Option β) : SM α β := StateT.lift o

/-- The trust region of a screened bracket's loop: `trustBounds` around its screened `E`. -/
def trust (s : Screened α) : EBounds α :=
  let t := trustBounds s.lo s.hi s.params.gains.e s.enc.centre
  ⟨t.1, t.2.1, t.2.2⟩

/-- The screening (step 3): each metered bracket (each of the pool from its `e_centre` when none
metered) optimised against the field from no start within its window, aimed at the field minus
its correction; ranked by loss, a stable sort. -/
def screenAll (pool : List (BracketRange α)) (ds : List (Metered α)) (f : IntentField α) (pullback : Bool) :
    SM α (List (Screened α)) := do
  let items : List (BracketRange α × α × α × α) :=
    if ds.isEmpty then pool.map fun c => (c, c.searchLo, c.searchHi, c.eCenter)
    else ds.map fun d => (d.enc, d.lo, d.hi, d.eMet)
  let corrs ← get
  let rows := items.map fun (c, lo, hi, em) =>
    let (p, l) := inp.search (corrs.getD c.idx none) c.idx f none ⟨lo, hi, c.centre⟩ pullback
    (⟨c, p, l, lo, hi, em⟩ : Screened α)
  pure (rows.mergeSort fun a b => decide (a.loss ≤ b.loss))

/-- The tournament (step 4): the loops of the best two screened brackets. -/
def tournament (predict : Array (Vec3 Nat) → IntentField α) (wish : Array α × Array α)
    (rows : List (Screened α)) (pullback : Bool) : SM α (List (Run α)) :=
  (rows.take 2).mapM fun s => inp.runLoop predict wish s.enc.idx s.params (trust s) pullback

/-- The search (`search_conversion_params`), from no corrections; `none` when the bundle has no
bracket with patch readings, a pinned bracket is not in it, or an inference fails. -/
def searchParams (predict : Array (Vec3 Nat) → IntentField α) : Option (SearchResult α) :=
  (go predict).run' (Array.replicate inp.brackets.size none)
where
  /-- The steps. -/
  go (predict : Array (Vec3 Nat) → IntentField α) : SM α (SearchResult α) := do
    let triled := inp.hasCrosstalk
    let pinE := inp.pins.e
    let boot (e : α) : Params α := ⟨⟨e, inp.bootG, inp.bootB⟩, 1.0⟩
    -- Step 1: the brackets and their ranges.
    if inp.brackets.size = 0 then failure
    let idxs ← match inp.pinTarget with
      | some t => if t < inp.brackets.size then pure [t] else failure
      | none => pure (filterAdmissible (inp.brackets.toList.map (·.data.tonalCapacity)))
    let enclosure := idxs.filterMap inp.enclosureOf
    let some bestFit := Negicc.bestFit enclosure | failure
    -- Step 2: metering and the bootstrap field.
    let admissible := enclosure.filter (·.admissible)
    let pool := if admissible.isEmpty then enclosure else admissible
    let pScene := sceneVector inp.inputMatrix inp.stats.logAvg
    let depths := deepestFirst (pool.filterMap fun c =>
      let prof := (inp.bracket c.idx).profile
      if !(referenceGrey prof).1 then none
      else
        let m := exposureForLightness prof pScene inp.bootG inp.bootB
        if m.2 then some (metered c m.1 c.searchLo c.searchHi) else none)
    let (cand1, e1) := match depths with
      | d :: _ => (d.enc, clamp d.eMet d.lo d.hi)
      | [] => (bestFit, bestFit.eCenter)
    let e1 := if pinE then inp.pinE else e1
    let single := depths.isEmpty || pinE
    let renders : List (Nat × α) :=
      if single then [(cand1.idx, e1)] else depths.map fun d => (d.enc.idx, clamp d.eMet d.lo d.hi)
    let fields ← renders.mapM fun (i, e) => do
      let pv ← inp.renderAndCorrect i (boot e)
      StateT.lift (inp.predictChecked predict pv)
    let f1 := meanField fields
    let wish := wishOf f1
    -- Step 3: screening, after measuring the corrections of brackets the bootstrap did not render.
    if single then
      let items : List (Nat × α) :=
        if depths.isEmpty then pool.map fun c => (c.idx, if pinE then inp.pinE else c.eCenter)
        else depths.map fun d => (d.enc.idx, inp.pinE)
      for (i, e) in items do
        if i != cand1.idx then discard (inp.renderAndCorrect i (boot e))
    let screening ← inp.screenAll pool depths f1 false
    -- Step 4: the tournament.
    let adaptive ← inp.tournament predict wish screening false
    let conv := fun (r : Run α) => r.status == FPStatus.converged
    let some chosen0 := winner adaptive triled conv (·.residual) | failure
    let mut chosen := chosen0
    let mut params := chosen0.params
    let boundsOf := fun (i : Nat) => match enclosure.find? (·.idx == i) with
      | some c => some c
      | none => inp.enclosureOf i
    -- Step 5: the clip fallback (the white clip and the median do not depend on the ceiling).
    let hc0 := inp.clip chosen.idx params 255.0
    let clip0 := hc0.whiteClip / 100.0
    let mut outcome := ClipFallback.notNeeded
    if clipFallbackNeeded clip0 inp.pinTarget.isSome then
      let tAdj := chosen.idx + 1
      match screening.find? (·.enc.idx == tAdj) with
      | none => outcome := .notScreened
      | some s =>
        if !s.enc.admissible then outcome := .notAdmissible
        else
          let fpAdj ← match adaptive.find? (·.idx == tAdj) with
            | some r => pure r
            | none => inp.runLoop predict wish tAdj s.params (trust s) false
          let hcAdj := inp.clip tAdj fpAdj.params 255.0
          outcome := clipFallbackVerdict clip0 (hcAdj.whiteClip / 100.0) (hcAdj.medianL - hc0.medianL)
            fpAdj.residual chosen.residual
          if outcome == .taken then
            chosen := fpAdj
            params := fpAdj.params
    -- Step 6: the highlight pullback.
    let mut pulled := false
    if outcome != .taken && 1 < inp.brackets.size && !pinE then
      let hc242 := inp.clip chosen.idx params 242.0
      let hc235 := inp.clip chosen.idx params 235.0
      let exc242 := hc242.maxExcess
      let tier := pullbackTier hc242.medianL exc242 hc235.maxExcess chosen.idx
      let encC := boundsOf chosen.idx
      let feasibleBelow := match encC with
        | some c => c.feasible && decide (params.gains.e < c.eMin)
        | none => false
      let infeasible := match encC with
        | some c => !c.feasible
        | none => true
      if feasibleBelow && tier then
        match encC with
        | some c =>
          match pullbackNudge c.eMin c.eMax params.gains.e with
          | some ep =>
            params := { params with gains := { params.gains with e := ep } }
            pulled := true
          | none => pure ()
        | none => pure ()
      else if infeasible && tier then
        if !depths.isEmpty then
          for (i, e) in renders do
            discard (inp.renderAndCorrect i (boot e))
        let rows ← inp.screenAll pool depths f1 true
        let resRuns ← inp.tournament predict wish rows true
        let score := fun (r : Run α) => tourneyScore (enclosure.find? (·.idx == r.idx)) r.residual r.params.gains.e
        match winner resRuns triled conv score with
        | some fpRes =>
          let mut pRes := fpRes.params
          let mut hcRes := inp.clip fpRes.idx pRes 242.0
          let dlMin := pullbackDlMin hc242.medianL
          let dl := hcRes.medianL - hc242.medianL
          if dl < dlMin then
            pRes := { pRes with gains := { pRes.gains with e := pullbackBrighten pRes.gains.e dl dlMin } }
            hcRes := inp.clip fpRes.idx pRes 242.0
          if pullbackAccept hcRes.maxExcess exc242 hcRes.medianL hc242.medianL fpRes.idx chosen.idx
              pRes.gains.e params.gains.e then
            chosen := fpRes
            params := pRes
            pulled := true
        | none => pure ()
    -- Step 7: the final roll-off.
    params := { params with shoulder := inp.measureShoulder chosen.idx params.gains }
    let toFrame := fun (e : α) => if triled then e / inp.scale.eNorm else e
    let range : α × α := match boundsOf chosen.idx with
      | some c => if c.feasible then (toFrame c.eMin, toFrame c.eMax) else (0.0, 0.0)
      | none => (0.0, 0.0)
    pure { idx := chosen.idx
           params := { params with gains := { params.gains with e := toFrame params.gains.e } }
           residual := chosen.residual
           status := chosen.status
           clipFallback := outcome
           clipFallbackWhiteClip := clip0
           highlightPullback := pulled
           shoulderGuard := chosen.guard
           eRangeLo := range.1
           eRangeHi := range.2
           screening, tournament := adaptive }

end SearchInput

end Negicc
