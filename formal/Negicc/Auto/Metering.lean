/-
Metering: the exposure at which a bracket renders the scene at L* 50.
-/
module

public import Negicc.Auto.Loop
public import Negicc.WhiteLight.Defs

@[expose] public section

/-!
# Metering

**The grey point** (`GreyPoint` in `multi_target_optimizer.cc`) renders synthetic camera-linear
inputs through a bracket (identity matrix, knee 0.95, no shoulder): a batch of inputs is divided by
its largest value over 0.9, quantised to 16 bits, and converted with that scale as the exposure
(`greyLab`). A bracket *has a neutral grey* when some input renders as `L*` 50, `a* = b* = 0`
within 0.25 ΔE (`referenceGrey`): from the best of 27 inputs `k (1.4, 1, 0.88)`, `k = 2^-10 … 2^3`
in half stops, a Newton solve in `log t` (`newton`: Jacobian by central differences of ±0.05,
least-squares steps with small singular values dropped (`lstsq3`), each channel's step held to
±0.25, until every component is within 0.02, at most 20 steps).

**The metered exposure** (`exposure_for_lightness`): the `E` at which the scene's log-average,
with the bootstrap gains, renders at `L*` 50 through the bracket, by bisection of `log2 E` over
`[-6, 6]`, stopping as soon as `|L* - 50| ≤ 0.1`, at most 25 halvings; metered only when the final
`E` is within that tolerance.

**The single-bracket solver's anchor** (`solve_dinov3_intent`): the `E` at which the median `L*` of
the grid's cells is 50, by 16 bisections of `log2 E` over `[-4, 4]`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## Least squares with small singular values dropped -/

/-- A 3 × 3 matrix as nine entries, row-major. -/
abbrev M9 (α : Type) := Array α

/-- Entry `(i, j)` of a 3 × 3 matrix stored row-major. -/
def m9 (a : M9 α) (i j : Nat) : α := a.getD (i * 3 + j) 0.0

/-- The matrix with entry `(i, j)` set to `v`. -/
def m9set (a : M9 α) (i j : Nat) (v : α) : M9 α := a.set! (i * 3 + j) v

/-- One Jacobi rotation of the pair `(p, q)` of the symmetric matrix `A`, accumulated in `V`
(`lstsq3`): skipped when `A_pq` is 0. -/
def jacobiRotate (st : M9 α × M9 α) (pq : Nat × Nat) : M9 α × M9 α :=
  let (A, V) := st
  let (p, q) := pq
  if scalarEq (m9 A p q) 0.0 then st
  else
    let theta := (m9 A q q - m9 A p p) / (2.0 * m9 A p q)
    let t := (if 0.0 ≤ theta then 1.0 else -1.0) / (fabs theta + sqrt (theta * theta + 1.0))
    let c := 1.0 / sqrt (t * t + 1.0)
    let s := t * c
    let A1 := [0, 1, 2].foldl (fun A k =>
      let akp := m9 A k p
      let akq := m9 A k q
      m9set (m9set A k p (c * akp - s * akq)) k q (s * akp + c * akq)) A
    let A2 := [0, 1, 2].foldl (fun A k =>
      let apk := m9 A p k
      let aqk := m9 A q k
      m9set (m9set A p k (c * apk - s * aqk)) q k (s * apk + c * aqk)) A1
    let V1 := [0, 1, 2].foldl (fun V k =>
      let vkp := m9 V k p
      let vkq := m9 V k q
      m9set (m9set V k p (c * vkp - s * vkq)) k q (s * vkp + c * vkq)) V
    (A2, V1)

/-- At most `n` Jacobi sweeps, stopping when the off-diagonal entries sum below `1e-30`. -/
def jacobiSweeps : Nat → M9 α × M9 α → M9 α × M9 α
  | 0, st => st
  | n + 1, st =>
    let A := st.1
    if fabs (m9 A 0 1) + fabs (m9 A 0 2) + fabs (m9 A 1 2) < 1e-30 then st
    else jacobiSweeps n ([(0, 1), (0, 2), (1, 2)].foldl jacobiRotate st)

/-- The least-squares solution of `J x = r` with the singular values of `J` below `rcond` times the
largest dropped (`lstsq3`, as numpy's `lstsq`): the eigen decomposition of `JᵀJ` by Jacobi
rotations (at most 50 sweeps), then `x = Σ_e v_e (v_e · Jᵀr) / λ_e` over the kept eigenvalues. -/
def lstsq3 (J : M9 α) (r : Vec3 α) (rcond : α) : Vec3 α :=
  let rv : Array α := #[r.x, r.y, r.z]
  let A0 : M9 α := ((List.range 3).flatMap fun i => (List.range 3).map fun j =>
    [0, 1, 2].foldl (fun s k => s + m9 J k i * m9 J k j) 0.0).toArray
  let I : M9 α := #[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  let (A, V) := jacobiSweeps 50 (A0, I)
  let jtr : Array α := ((List.range 3).map fun i => [0, 1, 2].foldl (fun s k => s + m9 J k i * rv.getD k 0.0) 0.0).toArray
  let smax := sqrt (max (max (m9 A 0 0) (max (m9 A 1 1) (m9 A 2 2))) 0.0)
  let x := [0, 1, 2].foldl (fun (x : Array α) e =>
    let lam := m9 A e e
    if !(decide (0.0 < lam)) || decide (sqrt lam ≤ rcond * smax) then x
    else
      let proj := [0, 1, 2].foldl (fun s k => s + m9 V k e * jtr.getD k 0.0) 0.0
      ((List.range 3).map fun k => x.getD k 0.0 + m9 V k e * proj / lam).toArray) #[0.0, 0.0, 0.0]
  ⟨x.getD 0 0.0, x.getD 1 0.0, x.getD 2 0.0⟩

/-! ## The grey point -/

/-- The configuration the grey point renders with: the bracket's profile, knee 0.95, no shoulder,
no tone curve. -/
def greyConfig (p : Profile α) : PipelineConfig α :=
  { profile := p, knee := autoKnee, shoulder := Shoulder.off, tone := none }

/-- A camera-linear input as the grey point feeds it: divided by the batch's scale, held to
`[0, 1]`, rounded to 16 bits (`std::lround`). -/
def greyInput (s : α) (t : Vec3 α) : Vec3 Nat := t.map fun v => roundNat (clamp01 (v / s) * 65535.0)

/-- The CIELAB of a batch of camera-linear inputs through a bracket (`GreyPoint::lab`): the batch's
largest value over 0.9 is the scale `s`; each input is divided by it, quantised, and converted with
`s` as the exposure (identity matrix), so the 16-bit values sit near 0.9 of full scale. -/
def greyLab (p : Profile α) (ts : List (Vec3 α)) : List (Lab α) :=
  let tmax := ts.foldl (fun m t => max m t.maxComp) ((ts.headD (Vec3.fill 0.0)).maxComp)
  let s := tmax / 0.9
  let m := candidateMatrix Mat3.identity s 1.0 1.0
  ts.map fun t => srgbToLab (convertPixel (greyConfig p) m (loadU16 (greyInput s t)))

/-- The `L*` of one input through a bracket (`GreyPoint::lstar`). -/
def greyLstar (p : Profile α) (t : Vec3 α) : α := ((greyLab p [t]).headD ⟨0.0, 0.0, 0.0⟩).L

/-- The distance of a colour from the target, ΔE 1976. -/
def labDist (target l : Lab α) : α := sqrt (sq (target.L - l.L) + sq (target.a - l.a) + sq (target.b - l.b))

/-- The Newton solve's finite-difference step, step bound and tolerance (`FD_STEP`,
`MAX_LOG_STEP`, `TOLERANCE`). -/
def newtonFd : α := 0.05

/-- The seven inputs of one Newton step: `t`, and `t` with each channel scaled by `exp(± 0.05)`. -/
def newtonPoints (logT : Vec3 α) : List (Vec3 α) :=
  let base := logT.map exp
  let shifted (c : Nat) (sgn : α) : Vec3 α :=
    ⟨exp (logT.x + (if c = 0 then sgn * newtonFd else 0.0)),
     exp (logT.y + (if c = 1 then sgn * newtonFd else 0.0)),
     exp (logT.z + (if c = 2 then sgn * newtonFd else 0.0))⟩
  base :: ([0, 1, 2].flatMap fun c => [shifted c 1.0, shifted c (-1.0)])

/-- The Newton solve in `log t` for the input the bracket renders as `target`
(`GreyPoint::newton`), at most `iters` steps: the final `log t` and its distance from the target.
Each step renders the seven points as one batch; it stops when every component of the miss is
below 0.02; otherwise it takes the least-squares step, each channel held to ±0.25. After the last
step the final `t` is rendered alone. -/
def newton (p : Profile α) (target : Lab α) : Nat → Vec3 α → Vec3 α × α
  | 0, logT =>
    let l := (greyLab p [logT.map exp]).headD ⟨0.0, 0.0, 0.0⟩
    (logT, labDist target l)
  | n + 1, logT =>
    let ls := (greyLab p (newtonPoints logT)).toArray
    let l0 := ls.getD 0 ⟨0.0, 0.0, 0.0⟩
    let r : Vec3 α := ⟨target.L - l0.L, target.a - l0.a, target.b - l0.b⟩
    if max (fabs r.x) (max (fabs r.y) (fabs r.z)) < 0.02 then (logT, labDist target l0)
    else
      let comp (l : Lab α) (row : Nat) : α := if row = 0 then l.L else if row = 1 then l.a else l.b
      let J : M9 α := ((List.range 3).flatMap fun row => (List.range 3).map fun c =>
        (comp (ls.getD (1 + 2 * c) ⟨0.0, 0.0, 0.0⟩) row - comp (ls.getD (2 + 2 * c) ⟨0.0, 0.0, 0.0⟩) row) /
          (2.0 * newtonFd)).toArray
      let st := lstsq3 J r 0.000001
      newton p target n (Vec3.zipWith (fun l s => l + clamp s (-0.25) 0.25) logT st)

/-- The direction of the grey point's start: `(1.4, 1, 0.88)`, the neutral input of the Portra
brackets normalised to green. -/
def greyDirection : Vec3 α := ⟨1.4, 1.0, 0.88⟩

/-- Whether a bracket renders some input as neutral `L*` 50 within 0.25 ΔE, and the `log t` found
(`GreyPoint::reference_grey`): the start is the first of `k (1.4, 1, 0.88)`, `k = 2^(-10 + i/2)`,
`i < 27`, nearest `L*` 50. -/
def referenceGrey (p : Profile α) : Bool × Vec3 α :=
  let ks := (List.range 27).map fun (i : Nat) => pow 2.0 (-10.0 + 0.5 * (i : α))
  let kBest := ((firstMin (fun k => fabs (greyLstar p (greyDirection.map (k * ·)) - 50.0)) ks).getD (1.0, 0.0)).1
  let logT := greyDirection.map fun d => log (kBest * d)
  let (lt, err) := newton p ⟨50.0, 0.0, 0.0⟩ 20 logT
  (decide (err ≤ 0.25), lt)

/-! ## The metered exposure -/

/-- The bisection of `exposure_for_lightness` from `[lo, hi]` with `n` halvings left: the `E` and
whether it renders within 0.1 of `L*` 50. -/
def meterGo (lstar : α → α) : Nat → α → α → α × Bool
  | 0, lo, hi =>
    let e := pow 2.0 (0.5 * (lo + hi))
    (e, decide (fabs (lstar e - 50.0) ≤ 0.1))
  | n + 1, lo, hi =>
    let mid := 0.5 * (lo + hi)
    let e := pow 2.0 mid
    let L := lstar e
    if fabs (L - 50.0) ≤ 0.1 then (e, true)
    else if 50.0 < L then meterGo lstar n mid hi else meterGo lstar n lo mid

/-- The metered exposure of a bracket (`exposure_for_lightness`): the `E` at which the scene vector
`p` with gains `g`, `b` renders at `L*` 50, by bisection of `log2 E` over `[-6, 6]` (a brighter
render for a smaller `E`), and whether it is within 0.1. -/
def exposureForLightness (prof : Profile α) (p : Vec3 α) (g b : α) : α × Bool :=
  let base : Vec3 α := ⟨p.x, p.y * g, p.z * b⟩
  meterGo (fun e => greyLstar prof (base.map (e * ·))) 25 (-6.0) 6.0

/-- The scene vector that is metered: the search's input matrix applied to the scene's log-average
(`p_scene`). -/
def sceneVector (inputMatrix : Mat3 α) (logAvg : Vec3 α) : Vec3 α := inputMatrix.mulVec logAvg

/-! ## The single-bracket solver's anchor -/

/-- The single-bracket solver's anchor (`solve_dinov3_intent`, step 1): 16 bisections of `log2 E`
over `[-4, 4]` on the median `L*` of the grid's cells rendered (no shoulder) at `E` and the
bootstrap gains; `E` grows while the median is at least 50. The anchor `E` and whether the
bisection ended at an end of the range. -/
def anchorExposure (medianL : α → α) : α × Bool :=
  let (lo, hi) := bisect (fun m => !decide (medianL (pow 2.0 m) < 50.0)) true 16 (-4.0) 4.0
  (pow 2.0 (0.5 * (lo + hi)), scalarEq lo (-4.0) || scalarEq hi 4.0)

end Negicc
