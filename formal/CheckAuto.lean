/-
The numeric check of the Auto search: the specification, evaluated over `Float`, against values
computed by the C++.

Run it from this directory with `lake exe checkAuto`. It reads `testvectors/auto/auto.json` and
`testvectors/auto/auto_search.json` and the bracket profiles they name (ICC files of the
repository, under `../profiles/`), evaluates the Lean definitions on the vectors' inputs, and
compares the results with the values the C++ gave. It prints one line per group of comparisons and
ends with a non-zero exit code if any comparison is outside its tolerance.
-/
import Lean.Data.Json
import Negicc.Base.Defs
import Negicc.Scene.Defs
import Negicc.Auto.Defs

open Lean Negicc Negicc.Scalar

namespace CheckAuto

/-! ## Tolerances

The C++ computes most of the search in single precision and the specification is evaluated in
double precision. A tolerance below is stated with its reason; a relative tolerance is relative to
the value, or absolute for values below 1. -/

/-- Values the C++ computes in double precision, or exactly. -/
def tolDouble : Float := 1e-9

/-- Single precision values of order one after a few dozen operations (as `Check.lean`'s
`tolSingle`). -/
def tolSingle : Float := 5e-6

/-- Single precision values after many operations or of any size, relative: a sum over a few
thousand pixels, an exponential of such a sum, a ratio of tables. -/
def tolRel : Float := 2e-5

/-- CIELAB at the end of the whole conversion, as `Check.lean`'s `tolLabPixel`. -/
def tolLabPixel : Float := 1e-3

/-- A z-score loss or residual, relative: a mean of squared z-scores of CIELAB values that agree
to `tolLabPixel`, with spreads of 2 to 9, moves by about `2 · 1e-3 / 2` per unit of z-score. -/
def tolLoss : Float := 1e-3

/-- A Newton solve's `log t`, which stops as soon as the render is within 0.02 of its target in
every CIELAB component: single and double precision can stop at different points of that ball,
whose radius is about `0.02 / (d L* / d log t) ≈ 0.02 / 40` in `log t`. -/
def tolNewton : Float := 2e-3

/-- The roll-off `x_hi` found by bisection on a curve evaluated in single precision (as
`Check.lean`'s `tolRoot`). -/
def tolRoot : Float := 2e-5

/-! ## Bookkeeping -/

structure Group where
  name : String
  tol : Float
  count : Nat := 0
  failures : Nat := 0
  maxDiff : Float := 0.0

abbrev CheckM := StateT (Array Group) IO

def record (name : String) (tol diff : Float) : CheckM Unit :=
  modify fun gs =>
    let ok := diff ≤ tol
    match gs.findIdx? (·.name == name) with
    | some i =>
      gs.modify i fun g =>
        { g with count := g.count + 1, failures := g.failures + (if ok then 0 else 1),
                 maxDiff := if diff > g.maxDiff then diff else g.maxDiff }
    | none => gs.push { name, tol, count := 1, failures := if ok then 0 else 1, maxDiff := diff }

/-- Compares two numbers within an absolute tolerance. A NaN on either side fails. -/
def near (name : String) (tol : Float) (got want : Float) : CheckM Unit := do
  let diff := Float.abs (got - want)
  let diff := if diff.isNaN then 1e30 else diff
  unless diff ≤ tol do IO.println s!"  outside tolerance in [{name}]: Lean {got}, C++ {want}"
  record name tol diff

/-- Compares two numbers within a tolerance relative to the C++ value (absolute below 1). -/
def nearRel (name : String) (tol : Float) (got want : Float) : CheckM Unit := do
  let scale := if Float.abs want > 1.0 then Float.abs want else 1.0
  let diff := Float.abs (got - want) / scale
  let diff := if diff.isNaN then 1e30 else diff
  unless diff ≤ tol do IO.println s!"  outside tolerance in [{name}]: Lean {got}, C++ {want}"
  record name tol diff

/-- Compares two natural numbers within a number of units. -/
def nearNat (name : String) (tol : Nat) (got want : Nat) : CheckM Unit := do
  let diff := if got ≥ want then got - want else want - got
  unless diff ≤ tol do IO.println s!"  outside tolerance in [{name}]: Lean {got}, C++ {want}"
  record name tol.toFloat diff.toFloat

/-- Compares two values exactly. -/
def same {β : Type} [BEq β] [ToString β] (name : String) (got want : β) : CheckM Unit := do
  unless got == want do IO.println s!"  different in [{name}]: Lean {got}, C++ {want}"
  record name 0.0 (if got == want then 0.0 else 1.0)

/-! ## Reading JSON -/

def fail {α : Type} (msg : String) : IO α := throw (IO.userError msg)

def get (j : Json) (key : String) : IO Json :=
  match j.getObjVal? key with
  | .ok v => pure v
  | .error e => fail s!"missing {key}: {e}"

def num (j : Json) : IO Float :=
  match j.getNum? with
  | .ok n => pure n.toFloat
  | .error e => fail e

def nat (j : Json) : IO Nat :=
  match j.getNat? with
  | .ok n => pure n
  | .error e => fail e

def int (j : Json) : IO Int :=
  match j.getInt? with
  | .ok n => pure n
  | .error e => fail e

def arr (j : Json) : IO (Array Json) :=
  match j.getArr? with
  | .ok a => pure a
  | .error e => fail e

def numAt (j : Json) (key : String) : IO Float := do num (← get j key)
def natAt (j : Json) (key : String) : IO Nat := do nat (← get j key)
def intAt (j : Json) (key : String) : IO Int := do int (← get j key)
def arrAt (j : Json) (key : String) : IO (Array Json) := do arr (← get j key)
def numsAt (j : Json) (key : String) : IO (Array Float) := do (← arrAt j key).mapM num
def natsAt (j : Json) (key : String) : IO (Array Nat) := do (← arrAt j key).mapM nat

def strAt (j : Json) (key : String) : IO String := do
  match (← get j key).getStr? with
  | .ok s => pure s
  | .error e => fail e

def bool (j : Json) : IO Bool :=
  match j.getBool? with
  | .ok b => pure b
  | .error e => fail e

def boolAt (j : Json) (key : String) : IO Bool := do bool (← get j key)

def numsOf (a : Array Json) : IO (Array Float) := a.mapM num

instance : Inhabited (Vec3 Nat) := ⟨⟨0, 0, 0⟩⟩
instance : Inhabited (Vec3 Float) := ⟨⟨0.0, 0.0, 0.0⟩⟩
instance : Inhabited (Lab Float) := ⟨⟨0.0, 0.0, 0.0⟩⟩
instance : Inhabited (SearchBracket Float) :=
  ⟨⟨⟨false, 0.0, 0.0, 0.0, 0.0⟩, ⟨⟨#[], #[], #[]⟩, Mat3.identity, ⟨0.0, 0.0, 0.0⟩, ⟨0, 0, 0⟩, #[], ⟨#[], #[], #[]⟩⟩⟩⟩

def vec3 (a : Array Float) : Vec3 Float := ⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0⟩

def mat3 (a : Array Float) : Mat3 Float :=
  ⟨⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0⟩, ⟨a.getD 3 0.0, a.getD 4 0.0, a.getD 5 0.0⟩,
   ⟨a.getD 6 0.0, a.getD 7 0.0, a.getD 8 0.0⟩⟩

/-- Interleaved RGB values as pixels. -/
def pixels (a : Array Nat) : Array (Vec3 Nat) :=
  (Array.range (a.size / 3)).map fun i => ⟨a[3 * i]!, a[3 * i + 1]!, a[3 * i + 2]!⟩

/-- Interleaved CIELAB values as colours. -/
def labs (a : Array Float) : Array (Lab Float) :=
  (Array.range (a.size / 3)).map fun i => ⟨a[3 * i]!, a[3 * i + 1]!, a[3 * i + 2]!⟩

/-- The inner crop as a record. -/
structure Crop where
  x0 : Nat
  y0 : Nat
  w : Nat
  h : Nat

def cropOf (W H : Nat) (r : Float) : Crop :=
  let (x0, y0, w, h) := innerCrop W H r
  ⟨x0, y0, w, h⟩

def readJson (path : System.FilePath) : IO Json := do
  let text ← IO.FS.readFile path
  match Json.parse text with
  | .ok j => pure j
  | .error e => fail s!"{path}: {e}"

/-! ## Reading a bracket's ICC profile

As in `Check.lean`: the `A2B0` tag of an ICC version 2 profile, a `lut16Type`, read as value over
65535. -/

def be16 (b : ByteArray) (o : Nat) : Nat := b[o]!.toNat * 256 + b[o + 1]!.toNat
def be32 (b : ByteArray) (o : Nat) : Nat := be16 b o * 65536 + be16 b (o + 2)

def table16 (b : ByteArray) (o n : Nat) : Array Float :=
  (Array.range n).map fun i => (be16 b (o + 2 * i)).toFloat / 65535.0

def readProfile (path : System.FilePath) : IO (Profile Float) := do
  let b ← IO.FS.readBinFile path
  let tags := be32 b 128
  let mut found : Option Nat := none
  for i in [0:tags] do
    if be32 b (132 + 12 * i) == 0x41324230 then found := some (be32 b (132 + 12 * i + 4))
  let some o := found | fail s!"{path}: no A2B0 tag"
  unless be32 b o == 0x6D667432 do fail s!"{path}: the A2B0 tag is not a lut16Type"
  let inCh := b[o + 8]!.toNat
  let outCh := b[o + 9]!.toNat
  let grid := b[o + 10]!.toNat
  unless inCh == 3 && outCh == 3 do fail s!"{path}: not a 3 to 3 channel table"
  let identity := #[65536, 0, 0, 0, 65536, 0, 0, 0, 65536]
  for i in [0:9] do
    unless be32 b (o + 12 + 4 * i) == identity[i]! do fail s!"{path}: the lut16Type's matrix is not the identity"
  let nIn := be16 b (o + 48)
  let nOut := be16 b (o + 50)
  let inBase := o + 52
  let clutBase := inBase + 2 * 3 * nIn
  let clutLen := grid * grid * grid * 3
  let outBase := clutBase + 2 * clutLen
  let curve (base n c : Nat) : Array Float := table16 b (base + 2 * n * c) n
  pure {
    inCurves := ⟨curve inBase nIn 0, curve inBase nIn 1, curve inBase nIn 2⟩
    matrix := Mat3.identity
    offset := ⟨0.0, 0.0, 0.0⟩
    dim := ⟨grid, grid, grid⟩
    clut := table16 b clutBase clutLen
    outCurves := ⟨curve outBase nOut 0, curve outBase nOut 1, curve outBase nOut 2⟩ }

/-! ## The bundle -/

/-- A bundle as the vectors describe it, with its brackets' profiles. -/
structure BundleIn where
  hasCrosstalk : Bool
  scale : BundleScale Float
  brackets : Array (SearchBracket Float)

instance : Inhabited (BundleScale Float) := ⟨⟨Mat3.identity, 0.0, ⟨0.0, 0.0, 0.0⟩, 0.0⟩⟩
instance : Inhabited BundleIn := ⟨⟨false, default, #[]⟩⟩

def readBundle (root : System.FilePath) (j : Json) : IO BundleIn := do
  let dir ← strAt j "dir"
  let mut bs : Array (SearchBracket Float) := #[]
  for t in ← arrAt j "targets" do
    let name ← strAt t "name"
    let file := name.replace " " "_" ++ ".icc"
    let prof ← readProfile (root / dir / file)
    bs := bs.push ⟨⟨← boolAt t "has_patches", ← numAt t "tgt_min_g", ← numAt t "tgt_max_g", ← numAt t "exposure",
      ← numAt t "tonal_capacity"⟩, prof⟩
  pure ⟨← boolAt j "has_crosstalk",
    ⟨mat3 (← numsAt j "crosstalk"), ← numAt j "normalization_target", vec3 (← numsAt j "film_base"),
     ← numAt j "film_base_exposure"⟩, bs⟩

/-! ## The checks -/

def checkGeometry (j : Json) : CheckM Unit := do
  for c in ← arrAt j "inner_crop" do
    let (x0, y0, w, h) := innerCrop (← natAt c "W") (← natAt c "H") (← numAt c "ratio")
    let o ← natsAt c "out"
    same "geometry: innerCrop" #[x0, y0, w, h] o
  for c in ← arrAt j "aspect" do
    let s := aspectShape (α := Float) (← natAt c "w") (← natAt c "h") (← natAt c "max") (← natAt c "patch")
    same "geometry: aspectShape" #[s.targetW, s.targetH, s.gridW, s.gridH] (← natsAt c "out")

def checkResample (j : Json) : CheckM Unit := do
  for c in ← arrAt j "resample_u16" do
    let (sw, sh, dw, dh) := (← natAt c "sw", ← natAt c "sh", ← natAt c "dw", ← natAt c "dh")
    let out := resampleU16 (α := Float) (pixels (← natsAt c "src")) sw sh dw dh
    let want := pixels (← natsAt c "out")
    -- The area average is computed in double precision; the enlarging branch in single precision, whose
    -- rounding can move a value that is within 1e-3 of a half to the neighbouring code.
    let enlarging := sw < dw || sh < dh
    for (g, w) in out.zip want do
      for (a, b) in [(g.x, w.x), (g.y, w.y), (g.z, w.z)] do
        if enlarging then nearNat "resample: resampleU16 (bilinear, codes)" 1 a b
        else nearNat "resample: resampleU16 (area, codes)" 0 a b
  for c in ← arrAt j "aa_resize" do
    let (sw, sh, dw, dh) := (← natAt c "sw", ← natAt c "sh", ← natAt c "dw", ← natAt c "dh")
    let out := aaResize (α := Float) (pixels (← natsAt c "src")) sw sh dw dh
    let want := pixels (← natsAt c "out")
    same "resample: aaResize (sizes)" out.size want.size
    for (g, w) in out.zip want do
      same "resample: aaResize (codes)" #[g.x, g.y, g.z] #[w.x, w.y, w.z]
  let t ← get j "tone_weights"
  let w := toneWeights (α := Float) (pixels (← natsAt t "img")) (← natAt t "gw") (← natAt t "gh") (← natAt t "patch")
  let want ← numsAt t "out"
  for i in [0:w.size] do
    near "network: toneWeights" tolSingle w[i]!.x want[3 * i]!
    near "network: toneWeights" tolSingle w[i]!.y want[3 * i + 1]!
    near "network: toneWeights" tolSingle w[i]!.z want[3 * i + 2]!

def checkStats (j : Json) : CheckM Unit := do
  for c in ← arrAt j "frame_stats" do
    let (w, h) := (← natAt c "w", ← natAt c "h")
    let px := pixels (← natsAt c "px")
    let cr := cropOf w h (← numAt c "crop_ratio")
    let crop := (List.range cr.h).flatMap (fun y => (List.range cr.w).map fun x =>
      px[(cr.y0 + y) * w + cr.x0 + x]!) |>.toArray
    let st := frameStats (α := Float) crop cr.w cr.h (← natAt c "max") (← natAt c "patch")
    nearRel "stats: g_lo, g_hi" tolRel st.gLo (← numAt c "g_lo")
    nearRel "stats: g_lo, g_hi" tolRel st.gHi (← numAt c "g_hi")
    let la ← numsAt c "log_avg"
    nearRel "stats: logAverage" tolRel st.logAvg.x la[0]!
    nearRel "stats: logAverage" tolRel st.logAvg.y la[1]!
    nearRel "stats: logAverage" tolRel st.logAvg.z la[2]!
    same "stats: grid size" #[st.shape.gridW, st.shape.gridH] #[← natAt c "grid_w", ← natAt c "grid_h"]
    let want := pixels (← natsAt c "grid")
    -- The grid is averaged in single precision and rounded: a mean within 1e-3 of a half may round to the
    -- neighbouring code.
    for (g, ww) in st.grid.zip want do
      for (a, b) in [(g.x, ww.x), (g.y, ww.y), (g.z, ww.z)] do nearNat "stats: statsGrid (codes)" 1 a b
  for c in ← arrAt j "percentile" do
    let vals ← natsAt c "values"
    let hist := vals.foldl (fun h v => h.modify v (· + 1)) (Array.replicate 65536 0)
    nearRel "stats: u16Percentile" tolRel (u16PercentileLinear hist vals.size (← numAt c "q")) (← numAt c "out")
  for c in ← arrAt j "trained_cells" do
    let pv := pixels (← natsAt c "preview")
    let (w, h, gw, gh, p) := (← natAt c "w", ← natAt c "h", ← natAt c "gw", ← natAt c "gh", ← natAt c "patch")
    let lab := trainedCellsLab (α := Float) pv w h gw gh p
    let want := labs (← numsAt c "lab")
    for (g, ww) in lab.zip want do
      near "cells: trainedCellsLab" tolLabPixel g.L ww.L
      near "cells: trainedCellsLab" tolLabPixel g.a ww.a
      near "cells: trainedCellsLab" tolLabPixel g.b ww.b
    let r := residual (α := Float) pv w h gw gh p ⟨labs (← numsAt c "mu"), labs (← numsAt c "sigma")⟩
    nearRel "cells: residual (z2)" tolLoss r (← numAt c "residual")

def checkOptimiser (b : BundleIn) (j : Json) : CheckM Unit := do
  let br := b.brackets[← natAt j "target_idx"]!
  let merged := mat3 (← numsAt j "merged")
  let grid := pixels (← natsAt j "grid")
  let mu := labs (← numsAt j "mu")
  let sigma := labs (← numsAt j "sigma")
  for c in ← arrAt j "loss" do
    let r : RenderSetup Float := ⟨br.profile, autoKnee, setShoulder (← numAt c "shoulder"), merged⟩
    let cand := vec3 (← numsAt c "cand")
    let g : Gains Float := ⟨cand.x, cand.y, cand.z⟩
    for (l, w) in (intentCells r grid g).zip (labs (← numsAt c "lab")) do
      near "cells: intentCells" tolLabPixel l.L w.L
      near "cells: intentCells" tolLabPixel l.a w.a
      near "cells: intentCells" tolLabPixel l.b w.b
    let l := intentLoss r grid mu sigma g
    let o ← numsAt c "out"
    nearRel "cells: intentLoss (L, a, b)" tolLoss l.L o[0]!
    nearRel "cells: intentLoss (L, a, b)" tolLoss l.a o[1]!
    nearRel "cells: intentLoss (L, a, b)" tolLoss l.b o[2]!
    near "cells: intentLoss (clip %)" tolSingle l.clip o[3]!
  for c in ← arrAt j "optimize" do
    let name ← strAt c "name"
    let r : RenderSetup Float := ⟨br.profile, autoKnee, setShoulder (← numAt c "shoulder"), merged⟩
    let enc := vec3 ((← numsAt c "enc").push 0.0)
    let pv := vec3 (← numsAt c "pin_values")
    let cfg : OptConfig Float := ⟨enc.x, enc.y, pv.x, pv.y, pv.z⟩
    let pinsJ ← arrAt c "pins"
    let pins : Pins := ⟨← bool pinsJ[0]!, ← bool pinsJ[1]!, ← bool pinsJ[2]!⟩
    let initJ ← get c "init"
    let init : Option (Gains Float) ← if initJ.isNull then pure none else do
      let a ← numsOf (initJ.getArr?.toOption.getD #[])
      pure (some ⟨a[0]!, a[1]!, a[2]!⟩)
    let bJ ← get c "bounds"
    let bounds : Option (EBounds Float) ← if bJ.isNull then pure none else do
      let a ← numsOf (bJ.getArr?.toOption.getD #[])
      pure (some ⟨a[0]!, a[1]!, a[2]!⟩)
    let (g, score) := optimize (intentLoss r grid mu sigma) cfg init bounds pins
    let o ← numsAt c "out"
    near s!"optimiser: optimize ({name})" tolSingle g.e o[0]!
    near s!"optimiser: optimize ({name})" tolSingle g.g o[1]!
    near s!"optimiser: optimize ({name})" tolSingle g.b o[2]!
    nearRel s!"optimiser: optimize ({name})" tolLoss score (← numAt c "score")

def checkBrackets (b : BundleIn) (j : Json) : CheckM Unit := do
  for c in ← arrAt j "enclosure" do
    let i ← natAt c "idx"
    match bracketRange b.brackets[i]!.data i (← numAt c "g_lo") (← numAt c "g_hi") with
    | none => same "brackets: bracketEnclosure (exists)" false true
    | some e =>
      same "brackets: bracketEnclosure (flags)" #[e.feasible, e.admissible, e.usableInverted]
        #[← boolAt c "feasible", ← boolAt c "admissible", ← boolAt c "inverted"]
      for (got, key) in [(e.eMin, "e_min"), (e.eMax, "e_max"), (e.searchLo, "search_lo"), (e.searchHi, "search_hi"),
                         (e.eCenter, "e_center"), (e.slackEv, "slack_ev"), (e.fillRatio, "fill_ratio")] do
        nearRel "brackets: bracketEnclosure (values)" tolRel got (← numAt c key)
  for c in ← arrAt j "filter" do
    same "brackets: filterAdmissible" (filterAdmissible (← numsAt c "caps").toList).toArray (← natsAt c "out")

def checkScale (b : BundleIn) (j : Json) : CheckM Unit := do
  for c in ← arrAt j "frame_scale" do
    let cj ← get c "custom"
    let custom : Option (Vec3 Float) ← if cj.isNull then pure none else do
      pure (some (vec3 (← numsOf (cj.getArr?.toOption.getD #[]))))
    let s := frameScale b.scale (← numAt c "shutter") (← numAt c "iso") custom (← boolAt c "from_scan")
    nearRel "scale: frameScale" tolRel s.ratioFrame (← numAt c "ratio")
    nearRel "scale: frameScale" tolRel s.eNorm (← numAt c "e_norm")
    let m ← numsAt c "matrix"
    let got := #[s.frameMatrix.row0.x, s.frameMatrix.row0.y, s.frameMatrix.row0.z, s.frameMatrix.row1.x,
      s.frameMatrix.row1.y, s.frameMatrix.row1.z, s.frameMatrix.row2.x, s.frameMatrix.row2.y, s.frameMatrix.row2.z]
    for i in [0:9] do nearRel "scale: frameScale" tolRel got[i]! m[i]!

def checkMetering (b : BundleIn) (j : Json) : CheckM Unit := do
  for c in ← arrAt j "lstsq3" do
    let x := lstsq3 (← numsAt c "J") (vec3 (← numsAt c "r")) 0.000001
    let w ← numsAt c "x"
    nearRel "metering: lstsq3" tolRel x.x w[0]!
    nearRel "metering: lstsq3" tolRel x.y w[1]!
    nearRel "metering: lstsq3" tolRel x.z w[2]!
  for c in ← arrAt j "grey" do
    let prof := b.brackets[← natAt c "idx"]!.profile
    let (ok, lt) := referenceGrey prof
    same "metering: referenceGrey (ok)" ok (← boolAt c "ok")
    let w ← numsAt c "log_t"
    near "metering: referenceGrey (log t)" tolNewton lt.x w[0]!
    near "metering: referenceGrey (log t)" tolNewton lt.y w[1]!
    near "metering: referenceGrey (log t)" tolNewton lt.z w[2]!
    let batch ← (← arrAt c "batch").mapM fun t => do pure (vec3 (← numsOf (t.getArr?.toOption.getD #[])))
    let want ← (← arrAt c "batch_lab").mapM fun t => numsOf (t.getArr?.toOption.getD #[])
    for (l, wl) in (greyLab prof batch.toList).zip want.toList do
      near "metering: greyLab" tolLabPixel l.L wl[0]!
      near "metering: greyLab" tolLabPixel l.a wl[1]!
      near "metering: greyLab" tolLabPixel l.b wl[2]!
    for m in ← arrAt c "meter" do
      let (e, okm) := exposureForLightness prof (vec3 (← numsAt m "p")) 1.10 1.15
      same "metering: exposureForLightness (ok)" okm (← boolAt m "ok")
      nearRel "metering: exposureForLightness (E)" tolRel e (← numAt m "e")

def checkClip (b : BundleIn) (j : Json) : CheckM Unit := do
  let (w, h) := (← natAt j "w", ← natAt j "h")
  let px := pixels (← natsAt j "px")
  let cr := cropOf w h (← numAt j "crop_ratio")
  let crop := (List.range cr.h).flatMap (fun y => (List.range cr.w).map fun x => px[(cr.y0 + y) * w + cr.x0 + x]!) |>.toArray
  let merged := mat3 (← numsAt j "merged")
  let prof := b.brackets[← natAt j "target_idx"]!.profile
  -- One pixel's share of the crop, in percent. A pixel whose pre-shoulder value lies within 1e-4 (relative) of a
  -- threshold can count on the other side in single precision: the thinnest film reaches the input curves' steepest
  -- segments, where the conversion's single precision error grows to about 2e-5. The tolerance is the share of such
  -- pixels, plus the single precision rounding of a percentage (1e-5).
  let onePixel := 100.0 / crop.size.toFloat
  let near1e4 (v t : Float) : Bool := Float.abs (v - t) ≤ 1e-4 * t
  for c in ← arrAt j "cases" do
    let cand := vec3 (← numsAt c "cand")
    let m := candidateMatrix merged cand.x cand.y cand.z
    let cfg : PipelineConfig Float :=
      { profile := prof, knee := autoKnee, shoulder := setShoulder (← numAt c "shoulder"), tone := none }
    let ceil := vec3 (← numsAt c "ceil")
    let thr := ceil.map ceilLin
    let s := clipStats cfg m crop thr
    let lins := crop.map fun p => convertPixelLinear prof autoKnee m (loadU16 p)
    let amb (f : Vec3 Float → Bool) : Float := ((lins.filter f).size.toFloat) * onePixel + 1e-5
    let exc ← numsAt c "exc"
    near "clip: clipStats (exceedance %; pixels within 1e-4 of the ceiling)" (amb fun l => near1e4 l.x thr.x) s.exc.x exc[0]!
    near "clip: clipStats (exceedance %; pixels within 1e-4 of the ceiling)" (amb fun l => near1e4 l.y thr.y) s.exc.y exc[1]!
    near "clip: clipStats (exceedance %; pixels within 1e-4 of the ceiling)" (amb fun l => near1e4 l.z thr.z) s.exc.z exc[2]!
    let wc : Float := 0.9953029
    near "clip: clipStats (white clip %; pixels within 1e-4 of the clip)"
      (amb fun l => near1e4 l.x wc || near1e4 l.y wc || near1e4 l.z wc) s.whiteClip (← numAt c "white_clip")
    -- The median is the middle of a 0.1-wide bin: a pixel on a bin's edge may fall in the neighbour.
    near "clip: clipStats (median L*)" 0.1 s.medianL (← numAt c "median_L")
    near "clip: shoulderXHiFromHist" tolRoot (shoulderXHiFromHist (maxHistOf lins)) (← numAt c "x_hi")

def checkKnobs (b : BundleIn) (j : Json) : CheckM Unit := do
  for c in ← arrAt j "lstar_after_stops" do
    near "knobs: lstarAfterStops" tolSingle (lstarAfterStops (← numAt c "L") (← numAt c "stops")) (← numAt c "out")
  for c in ← arrAt j "miss_stops" do
    near "knobs: knobGreyMissStops" tolSingle (knobGreyMissStops (← numAt c "brightness") (← numAt c "L")) (← numAt c "out")
  let prof := b.brackets[← natAt j "target_idx"]!.profile
  for c in ← arrAt j "solve" do
    let k := vec3 (← numsAt c "knobs")
    match solveGreyKnobs prof (vec3 (← numsAt c "base")) k.x k.y k.z with
    | none => same "knobs: solveGreyKnobs (ok)" false (← boolAt c "ok")
    | some r =>
      same "knobs: solveGreyKnobs (ok, limited)" s!"true {r.limited}" s!"{← boolAt c "ok"} {← boolAt c "limited"}"
      let w ← numsAt c "egb"
      -- The gains come out of a Newton solve that stops within 0.02 of its target (see tolNewton).
      nearRel "knobs: solveGreyKnobs (gains)" tolNewton r.egb.x w[0]!
      nearRel "knobs: solveGreyKnobs (gains)" tolNewton r.egb.y w[1]!
      nearRel "knobs: solveGreyKnobs (gains)" tolNewton r.egb.z w[2]!
      let l ← numsAt c "grey_lab"
      -- The grey's CIELAB at those gains: within the solve's stopping ball of 0.02.
      near "knobs: solveGreyKnobs (grey L*a*b*)" 0.05 r.greyLab.L l[0]!
      near "knobs: solveGreyKnobs (grey L*a*b*)" 0.05 r.greyLab.a l[1]!
      near "knobs: solveGreyKnobs (grey L*a*b*)" 0.05 r.greyLab.b l[2]!

/-! ## The whole search -/

/-- The stand-in for the intent network (`StandInRunner` in `gen_auto.cc`): per cell, the mean `m`
of each plane over the cell's patch and the cell's tone weights `w` give
`μ = (46 + 9 mean(m) + 4 (w_hi - w_sh), 1.5 + 3 (m_r - m_g), 4 + 3 (m_g - m_b))` and
`σ = (7 + 2 |m_g|, 4 + w_mid, 5)`. -/
def standIn (patch : Nat) : IntentNet Float where
  run inp :=
    let gw := inp.width / patch
    let gh := inp.height / patch
    let plane := inp.width * inp.height
    let cells := (List.range gh).flatMap fun gy => (List.range gw).map fun gx =>
      let m (c : Nat) : Float :=
        ((List.range patch).flatMap fun py => (List.range patch).map fun px =>
          inp.pixels.getD (c * plane + (gy * patch + py) * inp.width + gx * patch + px) 0.0).foldl (· + ·) 0.0 /
          (patch * patch).toFloat
      let w := inp.poolWeights.getD (gy * gw + gx) ⟨0.0, 0.0, 0.0⟩
      let (mr, mg, mb) := (m 0, m 1, m 2)
      ((⟨46.0 + 9.0 * (mr + mg + mb) / 3.0 + 4.0 * (w.z - w.x), 1.5 + 3.0 * (mr - mg), 4.0 + 3.0 * (mg - mb)⟩ : Lab Float),
       (⟨7.0 + 2.0 * Float.abs mg, 4.0 + w.y, 5.0⟩ : Lab Float))
    ⟨(cells.map (·.1)).toArray, (cells.map (·.2)).toArray⟩

def statusName : FPStatus → String
  | .converged => "converged"
  | .diverged => "diverged"
  | .maxIters => "max_iters"

def outcomeName : ClipFallback → String
  | .notNeeded => "not needed"
  | .notScreened => "next bracket not screened"
  | .notAdmissible => "next bracket not admissible"
  | .clipsTooMuch => "next bracket clips too much"
  | .shiftsLightness => "next bracket shifts lightness"
  | .worseResidual => "next bracket worse residual"
  | .taken => "taken"

def checkSearch (bundles : Array BundleIn) (c : Json) : CheckM Unit := do
  let name ← strAt c "name"
  let some b := bundles[← natAt c "bundle"]? | fail "no such bundle"
  let (w, h) := (← natAt c "w", ← natAt c "h")
  let px := pixels (← natsAt c "px")
  let cr := cropOf w h (← numAt c "crop_ratio")
  let crop := (List.range cr.h).flatMap (fun y => (List.range cr.w).map fun x => px[(cr.y0 + y) * w + cr.x0 + x]!) |>.toArray
  let scale := frameScale b.scale (← numAt c "shutter") (← numAt c "iso") none false
  let pinsJ ← arrAt c "pins"
  let pins : Pins := ⟨← bool pinsJ[0]!, ← bool pinsJ[1]!, ← bool pinsJ[2]!⟩
  let pv := vec3 (← numsAt c "pin_values")
  let pt ← intAt c "pin_target"
  let (maxSize, patch) := (← natAt c "max_size", ← natAt c "patch")
  let inp : SearchInput Float := {
    brackets := b.brackets, hasCrosstalk := b.hasCrosstalk, scale, crop, cw := cr.w, ch := cr.h, maxSize, patch
    maxSteps := 8, pinTarget := if pt < 0 then none else some pt.toNat, pins
    pinE := if b.hasCrosstalk then pv.x * scale.eNorm else pv.x, pinG := pv.y, pinB := pv.z }
  let g := s!"search ({name})"
  match inp.searchParams fun pvw => infer (standIn patch) pvw cr.w cr.h maxSize patch with
  | none => same s!"{g}: ran" false true
  | some r =>
    let scr ← arrAt c "screening"
    same s!"{g}: screening order" (r.screening.map (·.enc.idx)).toArray (← scr.mapM fun s => natAt s "idx")
    for (s, ws) in r.screening.zip scr.toList do
      let p ← numsAt ws "params"
      near s!"{g}: screening (E, g, b)" tolSingle s.params.gains.e p[0]!
      near s!"{g}: screening (E, g, b)" tolSingle s.params.gains.g p[1]!
      near s!"{g}: screening (E, g, b)" tolSingle s.params.gains.b p[2]!
      nearRel s!"{g}: screening (loss)" tolLoss s.loss (← numAt ws "loss")
    let tour ← arrAt c "tournament"
    same s!"{g}: tournament (brackets)" (r.tournament.map (·.idx)).toArray (← tour.mapM fun s => natAt s "idx")
    for (t, wt) in r.tournament.zip tour.toList do
      same s!"{g}: tournament (status, steps, guard)" s!"{statusName t.status} {t.steps} {t.guard}"
        s!"{← strAt wt "status"} {← natAt wt "steps"} {← boolAt wt "shoulder_guard"}"
      let p ← numsAt wt "params"
      near s!"{g}: tournament (E, g, b)" tolSingle t.params.gains.e p[0]!
      near s!"{g}: tournament (E, g, b)" tolSingle t.params.gains.g p[1]!
      near s!"{g}: tournament (E, g, b)" tolSingle t.params.gains.b p[2]!
      near s!"{g}: tournament (x_hi)" tolRoot t.params.shoulder p[3]!
      nearRel s!"{g}: tournament (residual)" tolLoss t.residual (← numAt wt "residual")
    let res ← get c "result"
    same s!"{g}: result (bracket, status, fallback, pullback, guard)"
      s!"{r.idx} {statusName r.status} {outcomeName r.clipFallback} {r.highlightPullback} {r.shoulderGuard}"
      s!"{← natAt res "idx"} {← strAt res "status"} {← strAt res "clip_fallback"} {← boolAt res "highlight_pullback"} {← boolAt res "shoulder_guard"}"
    let p ← numsAt res "params"
    nearRel s!"{g}: result (E, g, b)" tolRel r.params.gains.e p[0]!
    near s!"{g}: result (E, g, b)" tolSingle r.params.gains.g p[1]!
    near s!"{g}: result (E, g, b)" tolSingle r.params.gains.b p[2]!
    near s!"{g}: result (x_hi)" tolRoot r.params.shoulder p[3]!
    nearRel s!"{g}: result (residual)" tolLoss r.residual (← numAt res "residual")
    near s!"{g}: result (white clip)" (1.0 / crop.size.toFloat) r.clipFallbackWhiteClip (← numAt res "clip_fallback_white_clip")
    let er ← numsAt res "e_range"
    nearRel s!"{g}: result (calibrated range)" tolRel r.eRangeLo er[0]!
    nearRel s!"{g}: result (calibrated range)" tolRel r.eRangeHi er[1]!

def solverStatus : Option FPStatus → String
  | none => "PINNED"
  | some .converged => "CONVERGED"
  | some .diverged => "DIVERGED"
  | some .maxIters => "MAX_ITERS"

def checkGiven (bundles : Array BundleIn) (j : Json) : CheckM Unit := do
  let im ← get j "solver_image"
  let (w, h, maxSize, patch) := (← natAt im "w", ← natAt im "h", ← natAt im "max_size", ← natAt im "patch")
  let img := pixels (← natsAt im "px")
  let prof := bundles[0]!.brackets[← natAt im "target_idx"]!.profile
  for c in ← arrAt j "solver" do
    let name ← strAt c "name"
    let g := s!"solver ({name})"
    let pinsJ ← arrAt c "pins"
    let pins : Pins := ⟨← bool pinsJ[0]!, ← bool pinsJ[1]!, ← bool pinsJ[2]!⟩
    let u := vec3 (← numsAt c "user")
    let r? := solveIntent (fun pv => infer (standIn patch) pv w h maxSize patch) prof autoKnee (← numAt c "shoulder")
      (mat3 (← numsAt c "merged")) ⟨u.x, u.y, u.z⟩ pins ⟨1.0, 1.0, 1.0⟩ img w h maxSize patch 8
    match r? with
    | none => same s!"{g}: ran" false (← boolAt c "ok")
    | some r =>
      let steps := if r.status.isNone then 0 else r.steps + 1
      same s!"{g}: status, steps" s!"{solverStatus r.status} {steps}" s!"{← strAt c "status"} {← natAt c "steps"}"
      nearRel s!"{g}: residual" tolLoss r.residual (← numAt c "residual")
      let rel ← numsAt c "relative"
      let abs ← numsAt c "absolute"
      nearRel s!"{g}: parameters" tolRel r.relative.e rel[0]!
      near s!"{g}: parameters" tolSingle r.relative.g rel[1]!
      near s!"{g}: parameters" tolSingle r.relative.b rel[2]!
      nearRel s!"{g}: parameters" tolRel r.absolute.e abs[0]!
      nearRel s!"{g}: parameters" tolRel r.absolute.g abs[1]!
      nearRel s!"{g}: parameters" tolRel r.absolute.b abs[2]!
      nearRel s!"{g}: anchor" tolRel r.eCenter (← numAt c "e_center")
  for f in ← arrAt j "frames" do
    let b := bundles[← natAt f "bundle"]!
    let (w, h) := (← natAt f "w", ← natAt f "h")
    let px := pixels (← natsAt f "px")
    let cr := cropOf w h (← numAt f "crop_ratio")
    let crop := (List.range cr.h).flatMap (fun y => (List.range cr.w).map fun x => px[(cr.y0 + y) * w + cr.x0 + x]!) |>.toArray
    let scale := frameScale b.scale (← numAt f "shutter") (← numAt f "iso") none false
    let ex := meterBracketExposures b.brackets b.hasCrosstalk scale crop cr.w cr.h
    for (e, we) in ex.toList.zip (← arrAt f "exposures").toList do
      match e with
      | none => same "given: meterBracketExposures (valid)" false (← boolAt we "valid")
      | some e =>
        same "given: meterBracketExposures (flags)" s!"{e.valid} {e.feasible} {e.hasMetered}"
          s!"{← boolAt we "valid"} {← boolAt we "feasible"} {← boolAt we "has_metered"}"
        for (got, key) in [(e.eMin, "e_min"), (e.eMax, "e_max"), (e.eCenter, "e_center"), (e.searchLo, "search_lo"),
                           (e.searchHi, "search_hi"), (e.eMetered, "e_metered")] do
          nearRel "given: meterBracketExposures (values)" tolRel got (← numAt we key)
    for r in ← arrAt f "rolloff" do
      let p := vec3 (← numsAt r "params")
      near "given: measureRolloffXHi" tolRoot
        (measureRolloffXHi b.brackets[← natAt r "target_idx"]!.profile scale crop ⟨p.x, p.y, p.z⟩) (← numAt r "x_hi")

def run (root : System.FilePath) : CheckM Unit := do
  let a ← readJson "testvectors/auto/auto.json"
  let b ← readBundle root (← get a "bundle")
  checkGeometry (← get a "geometry")
  checkResample (← get a "resample")
  checkStats (← get a "stats")
  checkOptimiser b (← get a "optimiser")
  checkBrackets b (← get a "brackets")
  checkScale b (← get a "bundle")
  checkMetering b (← get a "metering")
  checkClip b (← get a "clip")
  checkKnobs b (← get a "knobs")
  let bw ← readBundle root (← get a "bundle_white")
  checkGiven #[b, bw] (← get a "given")
  let s ← readJson "testvectors/auto/auto_search.json"
  let bundles ← (← arrAt s "bundles").mapM fun j => (readBundle root j : IO BundleIn)
  for c in ← arrAt s "searches" do checkSearch bundles c

end CheckAuto

/-- A small non-negative number in scientific notation with two digits, such as `1.5e-4`. -/
def sci (x : Float) : String :=
  if x == 0.0 then "0"
  else
    let e := (Float.log10 x).floor
    let m := (x / Float.pow 10.0 e * 10.0).round / 10.0
    let (m, e) := if m ≥ 10.0 then (m / 10.0, e + 1.0) else (m, e)
    let digits := (m * 10.0).round.toUInt64.toNat
    s!"{digits / 10}.{digits % 10}e{e.toInt64}"

def main (args : List String) : IO UInt32 := do
  let root : System.FilePath := args.headD ".."
  let (_, groups) ← (CheckAuto.run root).run #[]
  let mut failures := 0
  let mut total := 0
  let pad (s : String) (n : Nat) := s ++ String.ofList (List.replicate (n - s.length) ' ')
  IO.println s!"{pad "group" 64}{pad "checks" 8}{pad "failed" 8}{pad "largest difference" 20}tolerance"
  for g in groups do
    total := total + g.count
    failures := failures + g.failures
    IO.println s!"{pad g.name 64}{pad (toString g.count) 8}{pad (toString g.failures) 8}{pad (sci g.maxDiff) 20}{sci g.tol}"
  IO.println s!"{total} comparisons, {failures} outside tolerance"
  pure (if failures == 0 then 0 else 1)
