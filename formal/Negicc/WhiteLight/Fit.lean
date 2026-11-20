/-
The white-light curve fit: the per-cell model, the anchors, the residuals, the guard and the solve.
-/
module

public import Negicc.WhiteLight.Curve
public import Negicc.Tone.Curve

@[expose] public section

/-!
# The white-light curve fit

After the search, on a white-light scan, one Levenberg-Marquardt solve over four numbers, the
green and blue gains `g`, `b` and the green and blue bends `a_G`, `a_B`, makes what the frame shows
as neutral come out grey (`fit_input_curves_joint` in `input_curves.h`). Each candidate is scored at
the `E` that renders the grid's median `L*` at Auto's (`JointCurveModel::solve_e`).

**The cells.** The grid's cells (the search's patch-averaged camera RGB) are rendered through a
per-cell model of the conversion (`JointCurveModel::cell_lab`): each cell's three rows of the
merged matrix are taken once and a candidate's `E`, `g`, `b` only scale them; the knee; the curves
applied to green and blue analytically; the bracket's profile; the display value; CIELAB.

**The anchors.** Every cell Auto renders with `2 < L* < 92` (the set `B`), weighted by the neutral
head's `p^2`: the evidence `m = Σ p^2`, the weights `ω_i = (p_i^2 + 10^-3 / |B|) / (m + 10^-3)`, the
weighted means and the slopes `k = 50 cov_ω(L*, ·) / (var_ω(L*) + 25)` (`joint_moments`). The
targets shrink Auto's values by `w = m / (m + 12)` and the slope targets by `w s`,
`s = var / (var + 36)`.

**The residuals** (`R` in `fit_input_curves_joint`): the anchors' mean `a*` and `b*` against their
targets (the `a*` term divided by `σ_anc = sqrt((144 + m) / (m + 12))`), the two slopes against
theirs (weighted `0.5 sqrt(var / (var + 100))`), the intent field's precision-weighted mean `a*`
residual less the slope's lever and one standard error, divided by `sqrt(SE^2 + 1)`, the guard's
two hinges on the guarded pass, and `sqrt(1.5)` times each bend. The `a*`, `b*` and intent terms
carry the factor `1 / 1.5`.

**The guard.** The fit runs first without the guard; when its result's mean `a*` over the guard's
cells, plain or `p^2`-weighted, ends more than 0.5 below Auto's, it runs again from the start with
those excesses as residuals of weight 4 (`JointFitGuard`). A cell Auto renders in deep shadow that
ends greener counts at Auto's `a*`, fading out between `L*` 10 and 20 (`green_guard_cell_a`).

Non-finite values, which the C++ skips or replaces, are not modelled: the definitions treat every
cell as finite.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The frame and the per-cell model -/

/-- What anchors the curves on a frame (`CurveFrame`): the merged matrix before the candidate's
rows, the knee, the film base in camera units, the grid's cells (16-bit camera RGB) and the neutral
head's `p` per cell when it ran. -/
structure CurveFrame (α : Type) where
  merged : Mat3 α
  knee : α
  filmBase : Vec3 α
  grid : Array (Vec3 Nat)
  neutralP : Option (Array α)

/-- Each cell's rows of the merged matrix, `merged · load(cell)` (`JointCurveModel` constructor). -/
def CurveFrame.rows (f : CurveFrame α) : Array (Vec3 α) := f.grid.map fun px => f.merged.mulVec (loadU16 px)

/-- The film base through the merged matrix, over 65535 (`fb_`). -/
def CurveFrame.fb (f : CurveFrame α) : Vec3 α := (f.merged.mulVec f.filmBase).map (· / 65535.0)

/-- The curves of `shape` (its bends and hold) anchored at `(e, g, b)` on the per-cell model
(`JointCurveModel::anchored`): `ref` the knee of the scaled film base, `u_lo` and `u_med` the
minimum and the median of the cells' `u`. -/
def anchored (f : CurveFrame α) (shape : InputCurves α) (e g b : α) : InputCurves α :=
  let refG := kneeCompress f.knee (e * g * f.fb.y)
  let refB := kneeCompress f.knee (e * b * f.fb.z)
  let ug := f.rows.map fun r => stopsBelowRef refG (kneeCompress f.knee (e * g * r.y))
  let ub := f.rows.map fun r => stopsBelowRef refB (kneeCompress f.knee (e * b * r.z))
  ⟨{ shape.g with ref := refG, uMed := cellMedian ug, uLo := cellMinimum ug },
   { shape.b with ref := refB, uMed := cellMedian ub, uLo := cellMinimum ub }⟩

/-- One cell through the conversion with the curves `c` (anchored) applied to its green and blue
(`JointCurveModel::cell_lab`). `row` is the cell's rows of the merged matrix. -/
def cellLab (cfg : PipelineConfig α) (knee : α) (row : Vec3 α) (e g b : α) (c : InputCurves α) : Lab α :=
  let v : Vec3 α := ⟨kneeCompress knee (e * row.x),
    clamp01 (inputCurve c.g (kneeCompress knee (e * g * row.y))),
    clamp01 (inputCurve c.b (kneeCompress knee (e * b * row.z)))⟩
  linearSrgbToLab (toDisplay cfg.activeShoulder (pcsToLinearSrgbPreShoulder (applyProfile cfg.profile v)))

/-- Every cell's CIELAB at `(e, g, b)` with the curves of `shape` anchored there (`labs_at`). -/
def cellLabs (cfg : PipelineConfig α) (f : CurveFrame α) (shape : InputCurves α) (e g b : α) : Array (Lab α) :=
  let c := anchored f shape e g b
  f.rows.map fun r => cellLab cfg f.knee r e g b c

/-- The upper median (nearest rank `n / 2` of the sorted values) of a non-empty array, `0` for an
empty one (`nth_element` at `n / 2`). -/
def upperMedian (xs : Array α) : α :=
  if xs.size = 0 then 0.0 else (xs.qsort (fun a b => decide (a < b))).getD (xs.size / 2) 0.0

/-- The grid's median `L*` at `(e, g, b)` (`JointCurveModel::median_L`). -/
def medianL (cfg : PipelineConfig α) (f : CurveFrame α) (shape : InputCurves α) (e g b : α) : α :=
  upperMedian ((cellLabs cfg f shape e g b).map (·.L))

/-! ## Solving `E` for the median lightness -/

/-- The bracket of `solve_e`: the largest log `E` seen with the median above the target and the
smallest with it below, each with whether it was seen. -/
structure EBracket (α : Type) where
  ta : α
  hasA : Bool
  tc : α
  hasC : Bool

/-- Records a point in the bracket (`note` in `solve_e`). -/
def EBracket.note (br : EBracket α) (t ft : α) : EBracket α :=
  if 0.0 < ft then { br with ta := if !br.hasA || br.ta < t then t else br.ta, hasA := true }
  else { br with tc := if !br.hasC || t < br.tc then t else br.tc, hasC := true }

/-- `E` with `fn(log E) = median L* - target` within `tol` of zero inside `[eLo, eHi]`
(`JointCurveModel::solve_e`): a start at `eWarm` clamped into the bounds; a second point 0.05 in
log `E` toward the target; up to 8 secant steps kept inside the bracket; then the bound the
bracket lacks tested and up to 40 bisections. Returns whether the target was reached and `E`. -/
def solveE (fn : α → α) (tol eWarm eLo eHi : α) : Bool × α := Id.run do
  let loT := log eLo
  let hiT := log eHi
  let mut br : EBracket α := ⟨loT, false, hiT, false⟩
  let mut t0 := min hiT (max loT (log eWarm))
  let mut f0 := fn t0
  if fabs f0 ≤ tol then return (true, exp t0)
  br := br.note t0 f0
  let d : α := if 0.0 < f0 then 0.05 else -0.05
  let tBound := if 0.0 < d then hiT else loT
  if decide (t0 ≤ tBound) && decide (tBound ≤ t0) then return (false, exp t0)
  let mut t1 := if 0.0 < d then min hiT (t0 + d) else max loT (t0 + d)
  let mut f1 := fn t1
  if fabs f1 ≤ tol then return (true, exp t1)
  br := br.note t1 f1
  if decide (t1 ≤ tBound) && decide (tBound ≤ t1) && (decide (0.0 < f1) == decide (0.0 < f0)) then
    return (false, exp t1)
  for _ in [0:8] do
    if decide (f1 ≤ f0) && decide (f0 ≤ f1) then break
    let t2 := t1 - f1 * (t1 - t0) / (f1 - f0)
    let bLo := if br.hasA then br.ta else loT
    let bHi := if br.hasC then br.tc else hiT
    if !(decide (bLo < t2) && decide (t2 < bHi)) then break
    t0 := t1
    f0 := f1
    t1 := t2
    f1 := fn t1
    if fabs f1 ≤ tol then return (true, exp t1)
    br := br.note t1 f1
  if !br.hasA then
    let fl := fn loT
    if fabs fl ≤ tol then return (true, exp loT)
    if fl < 0.0 then return (false, exp loT)
    br := br.note loT fl
  if !br.hasC then
    let fh := fn hiT
    if fabs fh ≤ tol then return (true, exp hiT)
    if 0.0 < fh then return (false, exp hiT)
    br := br.note hiT fh
  for _ in [0:40] do
    if !(decide (1e-6 < br.tc - br.ta)) then break
    let mid := 0.5 * (br.ta + br.tc)
    let fm := fn mid
    if fabs fm ≤ tol then return (true, exp mid)
    br := br.note mid fm
  pure (true, exp (0.5 * (br.ta + br.tc)))

/-! ## The anchors and their moments -/

/-- Weighted moments over a cell set (`JointMoments`): the means of `L*`, `a*`, `b*`, the variance of
`L*` and the slopes `50 cov(L*, a*) / (var + 25)`, `50 cov(L*, b*) / (var + 25)`. -/
structure JointMoments (α : Type) where
  L : α
  a : α
  b : α
  vL : α
  ka : α
  kb : α

/-- The moments of the cells `idx` with weights `w` (`joint_moments`). When the weights do not add
up to a positive number, the unnormalised sums and zero spread. -/
def jointMoments (labs : Array (Lab α)) (idx : Array Nat) (w : Array α) : JointMoments α :=
  let n := idx.size
  let at_ (k : Nat) : Lab α := labs.getD (idx.getD k 0) ⟨0.0, 0.0, 0.0⟩
  let ws := (List.range n).foldl (fun s k => s + w.getD k 0.0) 0.0
  let sL := (List.range n).foldl (fun s k => s + w.getD k 0.0 * (at_ k).L) 0.0
  let sa := (List.range n).foldl (fun s k => s + w.getD k 0.0 * (at_ k).a) 0.0
  let sb := (List.range n).foldl (fun s k => s + w.getD k 0.0 * (at_ k).b) 0.0
  if !(decide (0.0 < ws)) then ⟨sL, sa, sb, 0.0, 0.0, 0.0⟩
  else
    let mL := sL / ws
    let ma := sa / ws
    let mb := sb / ws
    let sums := (List.range n).foldl (fun (vL, sLa, sLb) k =>
      let l := at_ k
      let dL := l.L - mL
      let wk := w.getD k 0.0 / ws
      (vL + wk * dL * dL, sLa + wk * dL * (l.a - ma), sLb + wk * dL * (l.b - mb))) ((0.0 : α), (0.0 : α), (0.0 : α))
    let (vL, sLa, sLb) := sums
    ⟨mL, ma, mb, vL, 50.0 * sLa / (vL + 25.0), 50.0 * sLb / (vL + 25.0)⟩

/-- The evidence mass at which the anchors pull at half strength (`kEvidenceHalfMass`). -/
def evidenceHalfMass : α := 12.0

/-- The slope shrink's span, in `L*` (`kSlopeSpanL`). -/
def slopeSpanL : α := 6.0

/-- The shrink weight `w = m / (m + 12)` (`shrink_weight`). -/
def shrinkWeight (m : α) : α := m / (m + evidenceHalfMass)

/-- Each head's systematic error on the overall `a*` (`kHeadSystematicSd`) and the magenta lean in
standard errors (`kGreenLeanSe`). -/
def headSystematicSd : α := 1.0
@[inherit_doc headSystematicSd] def greenLeanSe : α := 1.0

/-! ## The never-greener guard -/

/-- A cell's `a*` as the guard counts it (`green_guard_cell_a`): a cell greener than at Auto
(`d = a - a_Auto < 0`) counts as `a_Auto + (1 - w_s) d`, with the shadow weight
`w_s = 1 - smoothstep((L_Auto - 15) / 10 + 0.5)`, `smoothstep(x) = 3x^2 - 2x^3` on `[0, 1]`. -/
def greenGuardCellA (a aAuto LAuto : α) : α :=
  let d := a - aAuto
  if 0.0 ≤ d then a
  else
    let x := clamp ((LAuto - 15.0) / 10.0 + 0.5) 0.0 1.0
    let ws := 1.0 - x * x * (3.0 - 2.0 * x)
    aAuto + (1.0 - ws) * d

/-- The guard's tolerance, weight and the least cells and mass for a mean to count
(`kGreenGuardTolAb`, `kGreenGuardWeight`, `kGreenGuardMinCells`, `kGreenGuardMinMass`). -/
def greenGuardTolAb : α := 0.5
@[inherit_doc greenGuardTolAb] def greenGuardWeight : α := 4.0
@[inherit_doc greenGuardTolAb] def greenGuardMinCells : α := 8.0
@[inherit_doc greenGuardTolAb] def greenGuardMinMass : α := 2.0

/-- The guard of one pass (`JointFitGuard`): its cells (Auto's render at `2 < L* < 92`), their `p^2`,
Auto's `a*` and `L*` per cell, Auto's two means and which of them count. -/
structure JointFitGuard (α : Type) where
  inCell : Array Bool
  pw : Array α
  autoA : Array α
  autoL : Array α
  ref : α × α
  counted : Bool × Bool

/-- The two means of a candidate whose cell `i` has `a*` `aOf i` (`JointFitGuard::means`): over the
guard's cells, plain and `p^2`-weighted, each cell's `a*` as `greenGuardCellA` counts it. -/
def guardMeans (gd : JointFitGuard α) (aOf : Nat → α) : α × α :=
  let r := (List.range gd.inCell.size).foldl (fun (sa, na, sp, wp) i =>
    if gd.inCell.getD i false then
      let a := greenGuardCellA (aOf i) (gd.autoA.getD i 0.0) (gd.autoL.getD i 0.0)
      (sa + a, na + 1.0, sp + gd.pw.getD i 0.0 * a, wp + gd.pw.getD i 0.0)
    else (sa, na, sp, wp)) ((0.0 : α), (0.0 : α), (0.0 : α), (0.0 : α))
  let (sa, na, sp, wp) := r
  (if 0.0 < na then sa / na else 0.0, if 0.0 < wp then sp / wp else 0.0)

/-- The guard on Auto's render (`JointFitGuard::set_cells` and its `ref`): `labAuto` the cells'
CIELAB at Auto, `p` the neutral head's per cell. -/
def guardOf (labAuto : Array (Lab α)) (p : Option (Array α)) : JointFitGuard α :=
  let n := labAuto.size
  let inCell := labAuto.map fun l => decide (2.0 < l.L) && decide (l.L < 92.0)
  let pw := (Array.range n).map fun i =>
    if inCell.getD i false then
      match p with
      | some ps => let q := clamp (ps.getD i 0.0) 0.0 1.0; q * q
      | none => 0.0
    else 0.0
  let na := (List.range n).foldl (fun s i => if inCell.getD i false then s + 1.0 else s) (0.0 : α)
  let wp := (List.range n).foldl (fun s i => s + pw.getD i 0.0) (0.0 : α)
  let g0 : JointFitGuard α := ⟨inCell, pw, labAuto.map (·.a), labAuto.map (·.L), (0.0, 0.0),
    (decide (greenGuardMinCells ≤ na), decide (greenGuardMinMass ≤ wp))⟩
  { g0 with ref := guardMeans g0 fun i => (labAuto.getD i ⟨0.0, 0.0, 0.0⟩).a }

/-- A mean's excess as the guarded pass's residual (`hinge`): `4 max(0, ref - 0.5 - v)` when the
mean counts. -/
def guardHinge (counted : Bool) (ref v : α) : α :=
  if counted then greenGuardWeight * max 0.0 (ref - greenGuardTolAb - v) else 0.0

/-- A result breaks the bound (`binds`): a counted mean more than 0.5 below Auto's. -/
def guardBinds (counted : Bool) (ref v : α) : Bool := counted && decide (0.0 < ref - greenGuardTolAb - v)

/-! ## The fit -/

/-- The fit's options (`CurveFitOptions`): the bounds on the bends, the curves' departure from the
base table, the gains' range in stops around Auto's, the damping on the bends, the hold, whether
`E` is solved, and the tolerance of each `E` solve. -/
structure CurveFitOptions (α : Type) where
  curveBound : α
  maxDevStops : α
  gBoundStops : α
  bBoundStops : α
  jointLambda : α
  holdSlope : Bool
  solveE : Bool
  eTolL : α

/-- The options the conversion uses (`CurveFitOptions()`). -/
def CurveFitOptions.default : CurveFitOptions α := ⟨0.35, 2.2, 0.75, 1.25, 1.5, true, true, 2e-5⟩

/-- The intent field the fifth residual reads (`JointFitField`): `μ` and `σ` per cell (CIELAB) and
each cell's 4x4 block of the network's grid. -/
structure JointFitField (α : Type) where
  mu : Array (Lab α)
  sigma : Array (Lab α)
  bx : Array Nat
  by_ : Array Nat
  nbx : Nat

/-- The fixed inputs of one fit. -/
structure FitProblem (α : Type) where
  cfg : PipelineConfig α
  frame : CurveFrame α
  field : Option (JointFitField α)
  e0 : α
  g0 : α
  b0 : α
  targetL : α
  tableN : Nat × Nat
  opts : CurveFitOptions α
  eLo : α
  eHi : α

namespace FitProblem

variable (P : FitProblem α)

/-- The curves of a parameter vector `(g, b, a_G, a_B)` (`JointLM::shape_of`): bends `a_G`, `a_B`,
`k = 0`, held as the options say. -/
def shapeOf (p : Array α) : InputCurves α :=
  ⟨{ ChannelCurve.zero with a := p.getD 2 0.0, hold := P.opts.holdSlope },
   { ChannelCurve.zero with a := p.getD 3 0.0, hold := P.opts.holdSlope }⟩

/-- The box bounds (`JointLM::bounds`): `g` within `g0 2^±0.75` and `b` within `b0 2^±1.25`, floored
at `10^-3`, the bends within `±0.35`. -/
def lo : Array α :=
  #[max 1e-3 (P.g0 * exp2 (-P.opts.gBoundStops)), max 1e-3 (P.b0 * exp2 (-P.opts.bBoundStops)),
    -P.opts.curveBound, -P.opts.curveBound]
@[inherit_doc lo] def hi : Array α :=
  #[P.g0 * exp2 P.opts.gBoundStops, P.b0 * exp2 P.opts.bBoundStops, P.opts.curveBound, P.opts.curveBound]

/-- The cells' CIELAB at a parameter vector and `E`. -/
def labsAt (p : Array α) (e : α) : Array (Lab α) :=
  cellLabs P.cfg P.frame (P.shapeOf p) e (p.getD 0 0.0) (p.getD 1 0.0)

/-- `E` for a parameter vector and whether the point is one (`JointLM::solve_point`): `E` solved
from `eWarm` (or Auto's `e0` when `E` is not solved), and, when asked, both curves' shape
(`curveShapeOk`) at the point's anchor. -/
def solvePoint (p : Array α) (eWarm : α) (checkShape : Bool) : Bool × α :=
  let shape := P.shapeOf p
  let (ok, e) := if P.opts.solveE then
      solveE (fun t => medianL P.cfg P.frame shape (exp t) (p.getD 0 0.0) (p.getD 1 0.0) - P.targetL)
        P.opts.eTolL eWarm P.eLo P.eHi
    else (true, P.e0)
  if !ok then (false, e)
  else if !checkShape then (true, e)
  else
    let c := anchored P.frame shape e (p.getD 0 0.0) (p.getD 1 0.0)
    (curveShapeOk c.g P.tableN.1 P.opts.maxDevStops && curveShapeOk c.b P.tableN.2 P.opts.maxDevStops, e)

end FitProblem

/-- What the residuals are built from, fixed at Auto's point (`fit_input_curves_joint`). -/
structure FitSetup (α : Type) where
  eA : α
  lab0 : Array (Lab α)
  B : Array Nat
  om : Array α
  m : α
  w : α
  M0 : JointMoments α
  slopeShare : α
  wSlope : α
  sigAnc : α
  ktA : α
  ktB : α
  al : Array α
  intent : Bool
  se : α
  dL50 : α
  r4Scale : α
  guard : JointFitGuard α

/-- The fixed part of the residuals at Auto's `E` `eA` (`fit_input_curves_joint`, before the solve):
the anchors `B`, their weights and moments, the shrinkage, the slope targets, the intent field's
weights, its standard error from 4x4 block sums detrended in `L*`, the lever `dL50` and the guard. -/
def FitProblem.setup (P : FitProblem α) (eA : α) : FitSetup α :=
  let p0 : Array α := #[P.g0, P.b0, 0.0, 0.0]
  let lab0 := P.labsAt p0 eA
  let n := lab0.size
  let B := (Array.range n).filter fun i =>
    let l := lab0.getD i ⟨0.0, 0.0, 0.0⟩
    decide (2.0 < l.L) && decide (l.L < 92.0)
  let p2 := B.map fun i => match P.frame.neutralP with
    | some ps => let q := clamp (ps.getD i 0.0) 0.0 1.0; q * q
    | none => 0.0
  let m := p2.foldl (· + ·) 0.0
  let w := shrinkWeight m
  let om := p2.map fun q => (q + 1e-3 / (B.size : α)) / (m + 1e-3)
  let M0 := jointMoments lab0 B om
  let slopeShare := M0.vL / (M0.vL + slopeSpanL * slopeSpanL)
  let wSlope := 0.5 * sqrt (M0.vL / (M0.vL + 100.0))
  let h : α := evidenceHalfMass
  let sigAnc := sqrt ((h * h + m) / (m + h)) * headSystematicSd
  let ktA := (1.0 - w * slopeShare) * M0.ka
  let ktB := (1.0 - w * slopeShare) * M0.kb
  let guard := guardOf (cellLabs P.cfg P.frame InputCurves.zero eA P.g0 P.b0) P.frame.neutralP
  match P.field with
  | none => ⟨eA, lab0, B, om, m, w, M0, slopeShare, wSlope, sigAnc, ktA, ktB, Array.replicate n 0.0,
      false, 0.0, 0.0, (1.0 / 1.5) / sqrt (headSystematicSd * headSystematicSd), guard⟩
  | some fi =>
    let al0 := (Array.range n).map fun i => let s := max (fi.sigma.getD i ⟨0.0, 0.0, 0.0⟩).a 1e-3; 1.0 / (s * s)
    let asum := al0.foldl (· + ·) 0.0
    if !(decide (0.0 < asum)) then
      ⟨eA, lab0, B, om, m, w, M0, slopeShare, wSlope, sigAnc, ktA, ktB, Array.replicate n 0.0,
        false, 0.0, 0.0, (1.0 / 1.5) / sqrt (headSystematicSd * headSystematicSd), guard⟩
    else
      let al := al0.map (· / asum)
      let L0 := fun i => (lab0.getD i ⟨0.0, 0.0, 0.0⟩).L
      let La := (List.range n).foldl (fun s i => s + al.getD i 0.0 * L0 i) 0.0
      let d0 := (Array.range n).map fun i =>
        if 0.0 < al.getD i 0.0 then (fi.mu.getD i ⟨0.0, 0.0, 0.0⟩).a - (lab0.getD i ⟨0.0, 0.0, 0.0⟩).a else 0.0
      let dbar := (List.range n).foldl (fun s i => s + al.getD i 0.0 * d0.getD i 0.0) 0.0
      let szz := (List.range n).foldl (fun s i => let z := (L0 i - La) / 50.0; s + al.getD i 0.0 * z * z) 0.0
      let szd := (List.range n).foldl (fun s i =>
        let z := (L0 i - La) / 50.0; s + al.getD i 0.0 * z * (d0.getD i 0.0 - dbar)) 0.0
      let kd := if 0.0 < szz then szd / szz else 0.0
      let nby := 1 + fi.by_.foldl max 0
      let bs := (List.range n).foldl (fun (bs : Array α) i =>
        let dt := (d0.getD i 0.0 - dbar) - kd * (L0 i - La) / 50.0
        bs.modify (fi.by_.getD i 0 * fi.nbx + fi.bx.getD i 0) (· + al.getD i 0.0 * dt))
        (Array.replicate (fi.nbx * nby) 0.0)
      let se := sqrt (bs.foldl (fun s v => s + v * v) 0.0)
      let dL50 := (La - M0.L) / 50.0 * (M0.vL + 25.0) / max M0.vL 25.0
      ⟨eA, lab0, B, om, m, w, M0, slopeShare, wSlope, sigAnc, ktA, ktB, al, true, se, dL50,
        (1.0 / 1.5) / sqrt (se * se + headSystematicSd * headSystematicSd), guard⟩

/-- The fifth residual (`r4_of`): the intent field's precision-weighted mean `a*` residual, less the
lever `(k_a - target k_a) dL50` and one standard error, scaled. Zero without a field. -/
def FitProblem.r4 (P : FitProblem α) (S : FitSetup α) (labs : Array (Lab α)) (Mp : JointMoments α) : α :=
  match P.field with
  | none => 0.0
  | some fi =>
    if !S.intent then 0.0
    else
      let s := (List.range labs.size).foldl (fun s i =>
        if 0.0 < S.al.getD i 0.0 then
          s + S.al.getD i 0.0 * ((labs.getD i ⟨0.0, 0.0, 0.0⟩).a - (fi.mu.getD i ⟨0.0, 0.0, 0.0⟩).a)
        else s) 0.0
      S.r4Scale * (s - (Mp.ka - S.ktA) * S.dL50 - greenLeanSe * S.se)

/-- The nine residuals at a parameter vector `q = (g, b, a_G, a_B)` and `E` (`R` in
`fit_input_curves_joint`); the guard's two entries only on the guarded pass. -/
def FitProblem.residuals (P : FitProblem α) (S : FitSetup α) (guarded : Bool) (q : Array α) (e : α) :
    Array α :=
  let labs := P.labsAt q e
  let Mp := jointMoments labs S.B S.om
  let cRes : α := 1.0 / 1.5
  let gv := guardMeans S.guard fun i => (labs.getD i ⟨0.0, 0.0, 0.0⟩).a
  let sl := sqrt (max 0.0 P.opts.jointLambda)
  #[cRes / S.sigAnc * (Mp.a - (1.0 - S.w) * S.M0.a),
    cRes * (Mp.b - (1.0 - S.w) * S.M0.b),
    S.wSlope * (Mp.ka - S.ktA),
    S.wSlope * (Mp.kb - S.ktB),
    P.r4 S labs Mp,
    if guarded then guardHinge S.guard.counted.1 S.guard.ref.1 gv.1 else 0.0,
    if guarded then guardHinge S.guard.counted.2 S.guard.ref.2 gv.2 else 0.0,
    sl * q.getD 2 0.0,
    sl * q.getD 3 0.0]

/-- The data loss (`loss_at`): the sum of the first five squared residuals. -/
def FitProblem.loss (P : FitProblem α) (S : FitSetup α) (q : Array α) (e : α) : α :=
  ((P.residuals S false q e).extract 0 5).foldl (fun s v => s + v * v) 0.0

/-- The sum of the squares. -/
def sumSq (r : Array α) : α := r.foldl (fun s v => s + v * v) 0.0

/-- A point evaluated (`JointLM::evaluate`): solved and, when asked, shape-checked; its `E`,
residuals and cost. -/
def FitProblem.evaluate (P : FitProblem α) (S : FitSetup α) (guarded : Bool) (p : Array α) (eWarm : α)
    (checkShape : Bool) : Option (α × Array α × α) :=
  let (ok, e) := P.solvePoint p eWarm checkShape
  if !ok then none
  else
    let r := P.residuals S guarded p e
    some (e, r, sumSq r)

/-- The furthest share `t` of the step from `p` to `q` whose point is feasible (`JointLM::cut_back`):
ten bisections of `[0, 1]`. Returns `t` and that point's `E`. -/
def FitProblem.cutBack (P : FitProblem α) (p q : Array α) (e : α) : α × α := Id.run do
  let mut tOk : α := 0.0
  let mut tBad : α := 1.0
  let mut eT := e
  for _ in [0:10] do
    let t := 0.5 * (tOk + tBad)
    let qt := (Array.range 4).map fun j => p.getD j 0.0 + t * (q.getD j 0.0 - p.getD j 0.0)
    let (ok, et) := P.solvePoint qt e true
    if ok then
      tOk := t
      eT := et
    else tBad := t
  pure (tOk, eT)

/-- One Levenberg-Marquardt run from `p` at `E` `e` (`JointLM::run`): forward differences of step
`±10^-3`, a parameter at a bound that the gradient pushes outward held, the normal equations damped
by `lambda max(A_aa, 10^-6)`, the step tried at the shares `1, 0.5, 0.25, 0.1, 0.05` (the full step
cut back to the shape's limit first), accepted when it lowers the cost; at most 25 iterations, ended
when the step is below `10^-5` or the gain below `10^-6` of the cost. Returns the end point, its
`E`, and whether it settled; `none` when the start is not feasible. -/
def FitProblem.run (P : FitProblem α) (S : FitSetup α) (guarded : Bool) (p0 : Array α) (e0 : α) :
    Option (Array α × α × Bool) := Id.run do
  let some (e1, r1, c1) := P.evaluate S guarded p0 e0 true | return none
  let lo := P.lo
  let hi := P.hi
  let mut p := p0
  let mut e := e1
  let mut r := r1
  let mut cost := c1
  let mut lambda : α := 1e-3
  let mut settled := decide (cost ≤ 1e-12)
  for _ in [0:25] do
    if settled then break
    let nr := r.size
    let mut J : Array (Array α) := Array.replicate nr (Array.replicate 4 0.0)
    for j in [0:4] do
      let h : α := if p.getD j 0.0 + 1e-3 ≤ hi.getD j 0.0 then 1e-3 else -1e-3
      let q := p.set! j (p.getD j 0.0 + h)
      match P.evaluate S guarded q e false with
      | some (_, rj, _) =>
        if rj.size = nr then
          for k in [0:nr] do
            J := J.modify k (·.set! j ((rj.getD k 0.0 - r.getD k 0.0) / h))
      | none => pure ()
    let mut A : Array (Array α) := Array.replicate 4 (Array.replicate 4 0.0)
    let mut g : Array α := Array.replicate 4 0.0
    for k in [0:nr] do
      for a in [0:4] do
        g := g.set! a (g.getD a 0.0 + J[k]!.getD a 0.0 * r.getD k 0.0)
        for b in [0:4] do
          A := A.modify a (·.set! b (A[a]!.getD b 0.0 + J[k]!.getD a 0.0 * J[k]!.getD b 0.0))
    for a in [0:4] do
      if (decide (p.getD a 0.0 ≤ lo.getD a 0.0 + 1e-5) && decide (0.0 < g.getD a 0.0)) ||
          (decide (hi.getD a 0.0 - 1e-5 ≤ p.getD a 0.0) && decide (g.getD a 0.0 < 0.0)) then
        A := (A.map (·.set! a 0.0)).set! a ((Array.replicate 4 0.0).set! a 1.0)
        g := g.set! a 0.0
    let mut accepted := false
    for _ in [0:10] do
      if accepted then break
      let M : Array (Array α) := (Array.range 4).map fun a =>
        ((Array.range 4).map fun b =>
          A[a]!.getD b 0.0 + (if a = b then lambda * max (A[a]!.getD a 0.0) 1e-6 else 0.0)).push (-(g.getD a 0.0))
      let delta := solve4 M
      let clampAt := fun (s : α) (j : Nat) => min (hi.getD j 0.0) (max (lo.getD j 0.0) (p.getD j 0.0 + s * delta.getD j 0.0))
      let step := (Array.range 4).foldl (fun s j => max s (fabs (clampAt 1.0 j - p.getD j 0.0))) 0.0
      if step < 1e-5 then
        settled := true
        break
      let mut found : Option (Array α × α × Array α × α) := none
      for s in [(1.0 : α), 0.5, 0.25, 0.1, 0.05] do
        if found.isSome then break
        let mut qt : Array α := (Array.range 4).map (clampAt s)
        let mut ev := P.evaluate S guarded qt e true
        if ev.isNone && decide (s ≤ 1.0) && decide (1.0 ≤ s) then
          let (tOk, eT) := P.cutBack p qt e
          if 0.0 < tOk then
            qt := (Array.range 4).map fun j => p.getD j 0.0 + tOk * (qt.getD j 0.0 - p.getD j 0.0)
            ev := P.evaluate S guarded qt eT true
        match ev with
        | some (eq, rt, cq) => if cq < cost then found := some (qt, eq, rt, cq)
        | none => pure ()
      match found with
      | some (q, eq, rt, cq) =>
        let gain := cost - cq
        p := q
        e := eq
        r := rt
        cost := cq
        lambda := max (lambda / 3.0) 1e-9
        accepted := true
        if gain < 1e-6 * max 1.0 cost || cost < 1e-8 then settled := true
      | none => lambda := lambda * 4.0
    if !accepted && !settled then break
  pure (some (p, e, settled))

/-- The result of the fit (`JointCurveFit`, the parts the conversion uses): whether it ran, the
curves anchored at the result, `E`, `g`, `b`, the data loss before and after, and whether the
guarded pass ran. -/
structure JointCurveFit (α : Type) where
  ok : Bool
  curves : InputCurves α
  e : α
  g : α
  b : α
  lossBefore : α
  lossAfter : α
  refit : Bool

/-- **The joint fit** (`fit_input_curves_joint`). Auto's `E` is solved first (no fit when that is
not feasible, or with fewer than 8 anchors); the free run; and, when its result breaks the guard's
bound, the guarded run from the same start. -/
def FitProblem.fit (P : FitProblem α) : JointCurveFit α :=
  let p0 : Array α := #[P.g0, P.b0, 0.0, 0.0]
  let none_ : JointCurveFit α := ⟨false, InputCurves.zero, P.e0, P.g0, P.b0, 0.0, 0.0, false⟩
  let (okA, eA) := P.solvePoint p0 P.e0 true
  if !okA then none_
  else
    let S := P.setup eA
    if S.B.size < 8 then none_
    else
      let free := (P.run S false p0 eA).getD (p0, eA, false)
      let gv := fun (p : Array α) (e : α) => guardMeans S.guard fun i => ((P.labsAt p e).getD i ⟨0.0, 0.0, 0.0⟩).a
      let v := gv free.1 free.2.1
      let binds := guardBinds S.guard.counted.1 S.guard.ref.1 v.1 || guardBinds S.guard.counted.2 S.guard.ref.2 v.2
      let res := if binds then (P.run S true p0 eA).getD (p0, eA, false) else free
      let (p, e, _) := res
      ⟨true, anchored P.frame (P.shapeOf p) e (p.getD 0 0.0) (p.getD 1 0.0), e, p.getD 0 0.0, p.getD 1 0.0,
        P.loss S p0 eA, P.loss S p e, binds⟩

/-! ## The re-meter on the combined tables -/

/-- The grid's median `L*` through a configuration at `(e, g, b)` (`grid_median_L`). -/
def gridMedianL (cfg : PipelineConfig α) (merged : Mat3 α) (e g b : α) (grid : Array (Vec3 Nat)) : α :=
  let m := candidateMatrix merged e g b
  upperMedian (grid.map fun px => (pixelLab cfg m (loadU16 px)).L)

/-- `E` re-metered so the grid's median `L*` through the combined profile (the curves anchored at
each `E` tried) is `targetL` (`solve_e_for_median_L`): 40 bisections in log `E` over
`[eCentre / 2, 2 eCentre]` within `[eLo, eHi]` when the target lies there, else over `[eLo, eHi]`;
a target outside those too gives the bound reached (`0.05 L*` of tolerance). -/
def solveEForMedianL (cfg : PipelineConfig α) (f : CurveFrame α) (shape : InputCurves α)
    (eCentre eLo eHi g b targetL : α) (iters : Nat := 40) : α := Id.run do
  let medianAt := fun (e : α) =>
    let c : PipelineConfig α := { cfg with profile := combinedProfile cfg.profile (inputCurveReference shape f.merged e g b f.knee f.filmBase f.grid) }
    gridMedianL c f.merged e g b f.grid
  let mut lo := max eLo (eCentre * 0.5)
  let mut hi := min eHi (eCentre * 2.0)
  let enclosed := decide (lo < hi) && decide (targetL ≤ medianAt lo) && decide (medianAt hi ≤ targetL)
  if !enclosed then
    lo := eLo
    hi := eHi
    let Llo := medianAt lo
    let Lhi := medianAt hi
    if Llo + 0.05 < targetL || targetL < Lhi - 0.05 then
      return (if Llo < targetL then lo else hi)
    if Llo ≤ targetL then return lo
    if targetL ≤ Lhi then return hi
  for _ in [0:iters] do
    let mid := sqrt (lo * hi)
    if medianAt mid < targetL then hi := mid else lo := mid
  pure (sqrt (lo * hi))

/-! ## The fit's target and the re-anchoring after the knobs -/

/-- The bounds of the fit's target median `L*` (`kIntentMinL`, `kIntentMaxL`). -/
def intentMinL : α := 18.0
@[inherit_doc intentMinL] def intentMaxL : α := 85.0

/-- The median `L*` the fit keeps (`fit_target_L_` in `WhiteLightFitCore`): Auto's grid median held to
`[18, 85]`, or Auto's own when `E` was given. -/
def fitTargetL (pinE : Bool) (targetL : α) : α :=
  if pinE then targetL else clamp targetL intentMinL intentMaxL

/-- The largest share `t` in `[0, 1]` of a curve's bend that keeps the shape `ok` (`shape_scale_of`): 1
when the whole bend passes, else 24 bisections of `[0, 1]`, the lower end. -/
def shapeScaleOf (ok : α → Bool) : α :=
  if ok 1.0 then 1.0 else (bisect ok true 24 0.0 1.0).1

/-- One channel re-anchored after the knobs moved `E`, `g`, `b` (`reanchor_input_curves`): `ref` scaled
and `u_med`, `u_lo` shifted by what the two anchors `a0` (at the curves' own parameters) and `a1` (at
the knobbed ones) differ by, then the bend scaled down to keep the shape over a table of `tableN`
entries. Returns the channel and the scale. -/
def reanchorChannel (c a0 a1 : ChannelCurve α) (tableN : Nat) (maxDev : α) : ChannelCurve α × α :=
  let ref := if decide (0.0 < a0.ref) && decide (0.0 < a1.ref) then c.ref * (a1.ref / a0.ref) else c.ref
  let c1 := { c with ref := ref, uMed := c.uMed + (a1.uMed - a0.uMed), uLo := c.uLo + (a1.uLo - a0.uLo) }
  let s := shapeScaleOf fun t => curveShapeOk { c1 with a := t * c1.a, k := t * c1.k } tableN maxDev
  ({ c1 with a := c1.a * s, k := c1.k * s }, s)

/-- The curves re-anchored after the knobs (`reanchor_input_curves`), with the anchors `a0` and `a1`
from `inputCurveReference` at the two parameter sets on the frame's grid (its picture cells on a
fallback crop); unchanged when the curves do nothing. -/
def reanchorCurves (c a0 a1 : InputCurves α) (tableN : Nat × Nat) (maxDev : α) : InputCurves α :=
  if !c.active then c
  else ⟨(reanchorChannel c.g a0.g a1.g tableN.1 maxDev).1, (reanchorChannel c.b a0.b a1.b tableN.2 maxDev).1⟩

/-! ## The frame's cells and film base -/

/-- On a fallback crop (the whole image), the cells that are picture (`fallback_picture_cells`):
not at the film base's level (green and blue within 0.25 stop of a rebate base) and not holder-dark
(green more than 9 stops below the base); all cells (an empty list) unless that keeps at least
`max(16, n / 4)` cells and drops some. -/
def fallbackPictureCells (grid : Array (Vec3 Nat)) (filmBase : Vec3 α) (baseIsRebate : Bool) : Array Nat :=
  if !(decide (0.0 < filmBase.y) && decide (0.0 < filmBase.z)) then #[]
  else
    let limG := filmBase.y * exp2 (-0.25)
    let limB := filmBase.z * exp2 (-0.25)
    let holder := filmBase.y * exp2 (-9.0)
    let kept := (Array.range grid.size).filter fun i =>
      let px := grid.getD i ⟨0, 0, 0⟩
      let g : α := (px.y : α)
      let b : α := (px.z : α)
      !((baseIsRebate && decide (limG ≤ g) && decide (limB ≤ b)) || decide (g ≤ holder))
    if kept.size < max 16 (grid.size / 4) || kept.size = grid.size then #[] else kept

/-- The film base that anchors the curves (`curve_frame_of`): the base read from the scan when
there is one, else the brightest grid cell per channel. -/
def curveFrameBase (fromScan : Option (Vec3 α)) (grid : Array (Vec3 Nat)) : Vec3 α :=
  match fromScan with
  | some fb => fb
  | none => grid.foldl (fun acc px => ⟨max acc.x (px.x : α), max acc.y (px.y : α), max acc.z (px.z : α)⟩)
      ⟨0.0, 0.0, 0.0⟩

end Negicc
