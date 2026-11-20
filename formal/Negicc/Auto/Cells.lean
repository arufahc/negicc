/-
The cells a render is judged on, the frame's statistics, and the losses.
-/
module

public import Negicc.Scene.Defs
public import Negicc.Auto.Network

@[expose] public section

/-!
# Cells, statistics and losses

**The trained cells.** A preview is compared with an intent field cell by cell, and its colour per
cell is read the way the network's targets were made (`trained_cells_lab` in
`dinov3_auto_solver.cc`): the 8-bit preview decoded to linear light, resampled by area average to
the network's size (`gw · patch × gh · patch`), a horizontal pass then a vertical pass, CIELAB of
each resampled pixel, and the mean of the `patch²` CIELAB values of each cell.

**The residual** of a preview against a field is the mean squared z-score
`z2 = 1 / (3 N) Σ_i Σ_k ((lab_ik - μ_ik) / σ_ik)²` (`self_consistency_residual`).

**The frame's statistics** (`compute_frame_stats` in `multi_target_optimizer.cc`) over its inner
crop: the green 0.5 % and 99.5 % points (`g_lo`, `g_hi`), the geometric mean of each channel, and
the crop resampled to the network's grid, one 16-bit camera RGB value per cell.

**The optimiser's loss** (`intent_loss`): each cell of that grid rendered at a candidate `(E, g, b)`
through the bracket, its CIELAB after the shoulder against the field, as the mean squared z-score
of `L*`, of `a*` and of `b*` separately, and the share of the cells' channels that clip white
before the shoulder, in percent. Rendering a cell's mean is not the mean of the cell's render, so
the search keeps per bracket the difference between the preview's trained cells and the
optimiser's cells at the same parameters (`SearchFrame::update_correction`) and aims the optimiser
at the field minus that difference (`aimField`).

**The highlight statistics** of a render of the crop (`cpu_highlight_clip` in `rgb_view.cc`): per
channel the share of pixels at or above a ceiling, the share with any channel clipping white, and
the median `L*` from a histogram of 1000 bins.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The trained cells and the residual -/

/-- The preview's CIELAB per cell, made as the intent head's training targets are made
(`trained_cells_lab`): `gw · gh` values, row-major. -/
def trainedCellsLab (preview : Array (Vec3 Nat)) (w h gw gh patch : Nat) : Array (Lab α) :=
  let W := gw * patch
  let H := gh * patch
  let lin : Array (Vec3 α) := preview.map (·.map fun n : Nat => srgbDecode ((n : α) / 255.0))
  let tmp : Array (Vec3 α) := ((List.range h).flatMap fun y => (List.range W).map fun dx =>
    weightedMean ((areaTaps (α := α) w W dx).map fun (xx, wt) => (pixelAt lin w xx y, wt))).toArray
  let pix (dx dy : Nat) : Lab α :=
    linearSrgbToLab (weightedMean ((areaTaps (α := α) h H dy).map fun (yy, wt) =>
      (pixelAt tmp W dx yy, wt)))
  let inv : α := 1.0 / ((patch * patch : Nat) : α)
  ((List.range gh).flatMap fun gy => (List.range gw).map fun gx =>
    let s := ((List.range patch).flatMap fun py => (List.range patch).map fun qx =>
      pix (gx * patch + qx) (gy * patch + py)).foldl Lab.add ⟨0.0, 0.0, 0.0⟩
    s.map (· * inv)).toArray

/-- Entry `i` of an array of colours; a zero colour outside it. -/
def labAt (xs : Array (Lab α)) (i : Nat) : Lab α := xs.getD i ⟨0.0, 0.0, 0.0⟩

/-- The squared z-scores of one cell's colour against the field's `μ` and `σ`, per component. -/
def zsq (lab mu sigma : Lab α) : Lab α :=
  ⟨sq ((lab.L - mu.L) / sigma.L), sq ((lab.a - mu.a) / sigma.a), sq ((lab.b - mu.b) / sigma.b)⟩

/-- The mean squared z-score of cells against a field, `z2`: per cell the mean of its three
squared z-scores, then the mean over the cells (`self_consistency_residual` after
`trained_cells_lab`). -/
def z2 (cells mu sigma : Array (Lab α)) : α :=
  let n := cells.size
  let s := (List.range n).foldl (fun acc i =>
    let z := zsq (labAt cells i) (labAt mu i) (labAt sigma i)
    acc + (z.L + z.a + z.b) / 3.0) 0.0
  s / (n : α)

/-- The residual of a preview against a field: `z2` of its trained cells
(`self_consistency_residual`). -/
def residual (preview : Array (Vec3 Nat)) (w h gw gh patch : Nat) (f : IntentField α) : α :=
  z2 (trainedCellsLab preview w h gw gh patch) f.mu f.sigma

/-! ## The frame's statistics -/

/-- The first code whose cumulative count exceeds `k`, scanning the histogram from code `v` with
`below` pixels counted already; 65535 when none does. -/
def firstAbove (k : Nat) : List Nat → Nat → Nat → Nat
  | [], _, _ => 65535
  | c :: rest, v, below => if k < below + c then v else firstAbove k rest (v + 1) (below + c)

/-- The `q` point of a 16-bit histogram of `total` pixels, interpolated between the two order
statistics around rank `q (total - 1)` (`u16_percentile_linear`); zero for no pixels. -/
def u16Percentile (hist : Array Nat) (total : Nat) (q : α) : α :=
  if total = 0 then 0.0
  else
    let pos : α := q * ((total - 1 : Nat) : α)
    let iLo := natFloor pos
    let iHi := natCeil pos
    let frac := pos - (iLo : α)
    let vLo := firstAbove iLo hist.toList 0 0
    let vHi := firstAbove iHi hist.toList 0 0
    (1.0 - frac) * (vLo : α) + frac * (vHi : α)

/-- The histogram of the green codes of a set of 16-bit pixels. -/
def greenHistogram (px : Array (Vec3 Nat)) : Array Nat :=
  px.foldl (fun h p => h.modify p.y (· + 1)) (Array.replicate 65536 0)

/-- The scene's range: its green 0.5 % and 99.5 % points, `g_lo` at least 1 and `g_hi` at least
one above `g_lo` (`compute_frame_stats`, and again where the search reads them). -/
def sceneRange (pLo pHi : α) : α × α :=
  let gLo := max 1.0 pLo
  (gLo, max (gLo + 1.0) pHi)

/-- The scene's tail: its range is taken between the 0.5 % and the 99.5 % points
(`SCENE_TAIL`). -/
def sceneTail : α := 0.005

/-- The scene's green range over the crop's pixels: `(g_lo, g_hi)`. -/
def sceneGreenRange (crop : Array (Vec3 Nat)) : α × α :=
  let hist := greenHistogram crop
  sceneRange (u16Percentile hist crop.size sceneTail) (u16Percentile hist crop.size (1.0 - sceneTail))

/-- The geometric mean of each channel over the crop, in `[0, 1]`: codes below 1 count as 1
(`log_avg_rgb`). -/
def logAverage (crop : Array (Vec3 Nat)) : Vec3 α :=
  let lg (n : Nat) : α := log ((max n 1 : Nat) : α)
  let tot := crop.foldl (fun s p => s.add ⟨lg p.x, lg p.y, lg p.z⟩) (Vec3.fill 0.0)
  tot.map fun t => exp (t * (1.0 / (crop.size : α))) / 65535.0

/-- The crop's 16-bit camera RGB per cell of the network's grid: the crop resampled to the
network's size (`resampleImage`), then each cell's `patch²` pixels averaged and rounded
(`FrameStats::grid`). -/
def statsGrid (crop : Array (Vec3 Nat)) (cw ch : Nat) (s : NetShape) (patch : Nat) :
    Array (Vec3 Nat) :=
  let p := max patch 1
  let dino := resampleImage (crop.map (·.map fun n : Nat => (n : α))) cw ch s.targetW s.targetH
  let inv : α := 1.0 / ((p * p : Nat) : α)
  ((List.range s.gridH).flatMap fun gy => (List.range s.gridW).map fun gx =>
    let acc := ((List.range p).flatMap fun py => (List.range p).map fun qx =>
      pixelAt dino s.targetW (gx * p + qx) (gy * p + py)).foldl Vec3.add (Vec3.fill 0.0)
    acc.map fun v => toU16 (v * inv)).toArray

/-- A frame's statistics over its inner crop (`FrameStats`). -/
structure FrameStats (α : Type) where
  gLo : α
  gHi : α
  logAvg : Vec3 α
  shape : NetShape
  grid : Array (Vec3 Nat)

/-- The statistics of a crop `cw × ch` pixels (`compute_frame_stats`). -/
def frameStats (crop : Array (Vec3 Nat)) (cw ch maxSize patch : Nat) : FrameStats α :=
  let s := aspectShape (α := α) cw ch maxSize patch
  let r := sceneGreenRange (α := α) crop
  ⟨r.1, r.2, logAverage crop, s, statsGrid (α := α) crop cw ch s patch⟩

/-! ## The optimiser's cells and loss -/

/-- The optimiser's four numbers for one candidate (`IntentLoss`): the mean squared z-scores of
`L*`, `a*` and `b*` over the cells, and the share of the cells' channels clipping white before the
shoulder, in percent. -/
structure IntentLoss (α : Type) where
  L : α
  a : α
  b : α
  clip : α

/-- What a candidate renders with: the bracket's profile, the knee, the shoulder and the input
matrix the gains scale (`SolverConfig`, with the gains in the matrix). -/
structure RenderSetup (α : Type) where
  profile : Profile α
  knee : α
  shoulder : Shoulder α
  merged : Mat3 α

/-- The search's three parameters: the exposure `E` and the green and blue gains relative to red
(`IntentGains`, `ConversionParams` without the shoulder). -/
structure Gains (α : Type) where
  e : α
  g : α
  b : α

/-- The pipeline configuration of a render setup (no tone curve). -/
def RenderSetup.config (r : RenderSetup α) : PipelineConfig α := ⟨r.profile, r.knee, r.shoulder, none⟩

/-- The input matrix of a candidate: the merged matrix with its rows scaled by `E`, `E g`,
`E b` (`candidate_matrix`). -/
def RenderSetup.matrix (r : RenderSetup α) (c : Gains α) : Mat3 α := candidateMatrix r.merged c.e c.g c.b

/-- The optimiser's CIELAB of each cell of the grid rendered at a candidate (`intent_cells`): the
display value after the shoulder. -/
def intentCells (r : RenderSetup α) (grid : Array (Vec3 Nat)) (c : Gains α) : Array (Lab α) :=
  grid.map fun px => pixelLab r.config (r.matrix c) (loadU16 px)

/-- The number of a pre-shoulder pixel's channels that clip white. -/
def whiteClippedChannels (lin : Vec3 α) : Nat :=
  (if linearChannelWhiteClipped lin.x then 1 else 0) + (if linearChannelWhiteClipped lin.y then 1 else 0) +
    (if linearChannelWhiteClipped lin.z then 1 else 0)

/-- The optimiser's loss of a candidate against a field (`intent_loss`). -/
def intentLoss (r : RenderSetup α) (grid : Array (Vec3 Nat)) (mu sigma : Array (Lab α)) (c : Gains α) :
    IntentLoss α :=
  let n := grid.size
  let m := r.matrix c
  let step := fun (acc : Lab α × Nat) (i : Nat) =>
    let lin := convertPixelLinear r.profile r.knee m (loadU16 (grid.getD i ⟨0, 0, 0⟩))
    let z := zsq (linearSrgbToLab (toDisplay r.shoulder lin)) (labAt mu i) (labAt sigma i)
    (acc.1.add z, acc.2 + whiteClippedChannels lin)
  let (s, clip) := (List.range n).foldl step (⟨0.0, 0.0, 0.0⟩, 0)
  ⟨s.L / (n : α), s.a / (n : α), s.b / (n : α), (clip : α) / (3.0 * (n : α)) * 100.0⟩

/-- The mean of the three squared z-scores of a loss: the intent part of the optimiser's score. -/
def IntentLoss.mean (l : IntentLoss α) : α := (l.L + l.a + l.b) / 3.0

/-- A bracket's correction: the preview's trained cells minus the optimiser's cells at the same
parameters (`SearchFrame::update_correction`). -/
def correction (picture optimiser : Array (Lab α)) : Array (Lab α) :=
  Array.zipWith Lab.sub picture optimiser

/-- The field the optimiser aims at: `μ` minus the bracket's correction, when one of the field's
size was measured; `μ` itself otherwise (`SearchFrame::search`). -/
def aimField (mu : Array (Lab α)) (corr : Option (Array (Lab α))) : Array (Lab α) :=
  match corr with
  | some c => if c.size = mu.size then Array.zipWith Lab.sub mu c else mu
  | none => mu

/-! ## The lightness wish and the bootstrap field -/

/-- A field with its `L*` (mean and spread) replaced by the frame's lightness wish, when the wish
has one value per cell (`SearchFrame::fixed_point`); `a*` and `b*` stay as predicted. -/
def withWish (wishMu wishSigma : Array α) (f : IntentField α) : IntentField α :=
  if wishMu.size = f.mu.size then
    ⟨Array.zipWith (fun l m => { m with L := l }) wishMu f.mu,
     Array.zipWith (fun l s => { s with L := l }) wishSigma f.sigma⟩
  else f

/-- The lightness wish of a field: its `L*` channel, mean and spread
(`SearchFrame::set_lightness_wish`). -/
def wishOf (f : IntentField α) : Array α × Array α := (f.mu.map (·.L), f.sigma.map (·.L))

/-- The mean of the fields predicted from several renders, each `μ` and `σ` added divided by the
number of renders, in the renders' order (the bootstrap field, `mean_field` in
`search_conversion_params`). -/
def meanField (fs : List (IntentField α)) : IntentField α :=
  let k : α := (fs.length : α)
  let n := (fs.headD ⟨#[], #[]⟩).mu.size
  let zero : Array (Lab α) := Array.replicate n ⟨0.0, 0.0, 0.0⟩
  let addDiv (acc xs : Array (Lab α)) := Array.zipWith (fun a x => a.add (x.map (· / k))) acc xs
  ⟨fs.foldl (fun acc f => addDiv acc f.mu) zero, fs.foldl (fun acc f => addDiv acc f.sigma) zero⟩

/-! ## The highlight statistics of a render -/

/-- The linear threshold of an 8-bit sRGB ceiling: a pixel counts as above it when it encodes to a
higher code (`floor(ceil) + 1` and up, at most 255), the threshold taken at the middle of that
code's step. A ceiling of at most 1.0001 is a fraction of full scale (`measure_highlight_clip`). -/
def ceilLin (ceil : α) : α :=
  let c8 := if 1.0001 < ceil then ceil else ceil * 255.0
  let thr : α := min ((natFloor c8 : α) + 1.0) 255.0
  srgbDecode (max 0.0 ((thr - 0.5) / 255.0))

/-- The bin of an `L*` value: 1000 bins over `[0, 100]` (`lab_bin`). -/
def labBin (L : α) : Nat := min (natFloor (L * (1000.0 / 100.0))) 999

/-- The first bin at which the cumulative count reaches `target`, scanning from bin `k` with `cum`
counted; 1000 when none does. -/
def firstReaching (target : α) : List Nat → Nat → α → Nat
  | [], k, _ => k
  | c :: rest, k, cum =>
    let cum' := cum + (c : α)
    if target ≤ cum' then k else firstReaching target rest (k + 1) cum'

/-- The `q` point of an `L*` histogram of `tot` pixels: the middle of the first bin at which the
cumulative count reaches `q tot` (`percentile_from_hist_L`); zero for no pixels. -/
def lHistPercentile (hist : Array Nat) (tot q : α) : α :=
  if tot ≤ 0.0 then 0.0
  else ((min (firstReaching (tot * q) hist.toList 0 0.0) 999 : Nat) + 0.5) * (100.0 / 1000.0)

/-- The highlight statistics of one render (`ClipStats`), in percent of the pixels. -/
structure ClipStats (α : Type) where
  exc : Vec3 α
  whiteClip : α
  medianL : α

/-- The highlight statistics of the crop's pixels rendered with one input matrix
(`cpu_highlight_clip`): per channel the share of pre-shoulder values at or above the channel's
threshold (`ceilLin`), the share of pixels with a channel clipping white, and the median `L*` of
the display values. -/
def clipStats (cfg : PipelineConfig α) (m : Mat3 α) (crop : Array (Vec3 Nat)) (thr : Vec3 α) :
    ClipStats α :=
  let lins := crop.map fun px => convertPixelLinear cfg.profile cfg.knee m (loadU16 px)
  let cnt (f : Vec3 α → Bool) : α := ((lins.filter f).size : α)
  let n : α := (crop.size : α)
  let pct (k : α) : α := k / n * 100.0
  let hist := lins.foldl (fun h l => h.modify (labBin (linearSrgbToL (toDisplay cfg.shoulder l))) (· + 1))
    (Array.replicate 1000 0)
  if crop.size = 0 then ⟨Vec3.fill 0.0, 0.0, 0.0⟩
  else
    ⟨⟨pct (cnt fun l => decide (thr.x ≤ l.x)), pct (cnt fun l => decide (thr.y ≤ l.y)),
      pct (cnt fun l => decide (thr.z ≤ l.z))⟩,
     pct (cnt fun l => pixelWhiteClipped l), lHistPercentile hist n 0.5⟩

/-- The largest per-channel share above the ceiling, as a fraction (`max_excess`). -/
def ClipStats.maxExcess (c : ClipStats α) : α := c.exc.maxComp / 100.0

end Negicc
