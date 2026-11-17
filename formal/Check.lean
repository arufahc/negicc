/-
The numeric check: the specification, evaluated over `Float`, against values computed by the C++.

Run it from this directory with `lake exe check`. It reads the test vectors in `testvectors/` and
the bracket profile they name (an ICC file of the repository, under `../profiles/`), evaluates
the Lean definitions on the vectors' inputs, and compares the results with the values the C++
functions gave. It prints one line per group of comparisons and ends with a non-zero exit code if
any comparison is outside its tolerance.
-/
import Lean.Data.Json
import Negicc.Base.Defs
import Negicc.Scene.Defs

open Lean Negicc Negicc.Scalar

namespace Check

/-! ## Tolerances

The C++ computes a pixel in single precision; the specification is evaluated in double precision.
One single precision operation is exact to 6e-8 of its result, and a pixel goes through a few
dozen of them, so the two agree to a few parts in a million. Each tolerance below is about three
times the largest difference observed and is stated with its reason; none was widened to make a
comparison pass. -/

/-- Values the C++ computes in single precision from inputs of order one: linear and encoded
colours, curve constants, matrix entries. The largest difference observed is 1.7e-6, on a
pre-shoulder value near 1 after the whole conversion. -/
def tolSingle : Float := 5e-6

/-- CIELAB computed in single precision directly from a given linear colour: `L*` up to 100, and
`a*`, `b*` as differences of cube roots multiplied by 500 and 200. Largest observed: 2e-5. -/
def tolLab : Float := 1e-4

/-- CIELAB at the end of the whole conversion. A relative difference of 2e-6 in the linear value
becomes up to `500 / 3 * 2e-6 = 3e-4` in `a*`. Largest observed: 1.5e-4. -/
def tolLabPixel : Float := 1e-3

/-- Values the C++ computes in double precision. -/
def tolDouble : Float := 1e-9

/-- The roll-off `x_hi` that `shoulder_x_hi_anchored` finds by bisection: the C++ evaluates the
curve it bisects on in single precision, and the curve's code value changes slowly with `x_hi`
(a few hundredths per unit), so its rounding noise of 6e-8 moves the crossing by a few millionths. -/
def tolRoot : Float := 2e-5

/-! ## Bookkeeping -/

/-- One group of comparisons: how many, how many failed, and the largest difference. -/
structure Group where
  name : String
  tol : Float
  count : Nat := 0
  failures : Nat := 0
  maxDiff : Float := 0.0

abbrev CheckM := StateT (Array Group) IO

/-- Records one comparison in its group. -/
def record (name : String) (tol diff : Float) : CheckM Unit :=
  modify fun gs =>
    let ok := diff ≤ tol
    match gs.findIdx? (·.name == name) with
    | some i =>
      gs.modify i fun g =>
        { g with count := g.count + 1, failures := g.failures + (if ok then 0 else 1),
                 maxDiff := if diff > g.maxDiff then diff else g.maxDiff }
    | none => gs.push { name, tol, count := 1, failures := if ok then 0 else 1, maxDiff := diff }

/-- Compares two numbers within the group's tolerance. A NaN on either side fails. A comparison
outside its tolerance is printed with both values. -/
def near (name : String) (tol : Float) (got want : Float) : CheckM Unit := do
  let diff := Float.abs (got - want)
  let diff := if diff.isNaN then 1e30 else diff
  unless diff ≤ tol do IO.println s!"  outside tolerance in [{name}]: Lean {got}, C++ {want}"
  record name tol diff

/-- Compares two triples. -/
def near3 (name : String) (tol : Float) (got : Vec3 Float) (want : Array Float) : CheckM Unit := do
  near name tol got.x (want.getD 0 0.0)
  near name tol got.y (want.getD 1 0.0)
  near name tol got.z (want.getD 2 0.0)

/-- Compares two natural numbers exactly. -/
def same (name : String) (got want : Nat) : CheckM Unit := do
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

def arr (j : Json) : IO (Array Json) :=
  match j.getArr? with
  | .ok a => pure a
  | .error e => fail e

def numAt (j : Json) (key : String) : IO Float := do num (← get j key)
def natAt (j : Json) (key : String) : IO Nat := do nat (← get j key)
def arrAt (j : Json) (key : String) : IO (Array Json) := do arr (← get j key)
def numsAt (j : Json) (key : String) : IO (Array Float) := do (← arrAt j key).mapM num
def natsAt (j : Json) (key : String) : IO (Array Nat) := do (← arrAt j key).mapM nat

def strAt (j : Json) (key : String) : IO String := do
  match (← get j key).getStr? with
  | .ok s => pure s
  | .error e => fail e

def boolAt (j : Json) (key : String) : IO Bool := do
  match (← get j key).getBool? with
  | .ok b => pure b
  | .error e => fail e

def vec3 (a : Array Float) : Vec3 Float := ⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0⟩
def vec3Nat (a : Array Nat) : Vec3 Nat := ⟨a.getD 0 0, a.getD 1 0, a.getD 2 0⟩

def mat3 (a : Array Float) : Mat3 Float :=
  ⟨⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0⟩, ⟨a.getD 3 0.0, a.getD 4 0.0, a.getD 5 0.0⟩,
   ⟨a.getD 6 0.0, a.getD 7 0.0, a.getD 8 0.0⟩⟩

def readJson (path : System.FilePath) : IO Json := do
  let text ← IO.FS.readFile path
  match Json.parse text with
  | .ok j => pure j
  | .error e => fail s!"{path}: {e}"

/-! ## Reading a bracket's ICC profile

The bundles' profiles are ICC version 2 files whose `A2B0` tag is a `lut16Type` (`mft2`): three
input tables, a colour table and three output tables of 16-bit values, read here as value over
65535, the way `parse_film_profile_pipeline` in `profile_bundle.cc` stores what Little-CMS
parsed. -/

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

/-! ## The checks -/

def checkBase (j : Json) : CheckM Unit := do
  for c in ← arrAt j "srgb_encode" do
    near "base: srgbEncode" tolSingle (srgbEncode (← numAt c "x")) (← numAt c "y")
  for c in ← arrAt j "srgb_decode" do
    near "base: srgbDecode" tolSingle (srgbDecode (← numAt c "x")) (← numAt c "y")
  for c in ← arrAt j "lab_f" do
    near "base: labF" tolSingle (labF (← numAt c "x")) (← numAt c "y")
  for c in ← arrAt j "lab" do
    let rgb := vec3 (← numsAt c "rgb")
    let lab := linearSrgbToLab rgb
    let want ← numsAt c "lab"
    near "base: linearSrgbToLab" tolLab lab.L (want.getD 0 0.0)
    near "base: linearSrgbToLab" tolLab lab.a (want.getD 1 0.0)
    near "base: linearSrgbToLab" tolLab lab.b (want.getD 2 0.0)
    near "base: linearSrgbToL" tolLab (linearSrgbToL rgb) (← numAt c "L")
    let lab2 := srgbToLab (vec3 (← numsAt c "srgb"))
    let want2 ← numsAt c "lab_of_srgb"
    near "base: srgbToLab" tolLab lab2.L (want2.getD 0 0.0)
    near "base: srgbToLab" tolLab lab2.a (want2.getD 1 0.0)
    near "base: srgbToLab" tolLab lab2.b (want2.getD 2 0.0)
  let trc ← get j "trc"
  let table ← numsAt trc "table"
  for c in ← arrAt trc "cases" do
    near "base: tableEval" tolSingle (tableEval table (← numAt c "x")) (← numAt c "y")
  for c in ← arrAt j "knee" do
    near "scene: kneeCompress" tolSingle (kneeCompress (← numAt c "knee") (← numAt c "x")) (← numAt c "y")

def checkFlat (j : Json) : CheckM Unit := do
  let gw ← natAt j "grid_w"
  let gh ← natAt j "grid_h"
  let block ← natAt j "block"
  let gain ← numsAt j "gain"
  for c in ← arrAt j "cases" do
    let half ← boolAt c "half"
    let (i, tx) := flatAxis (α := Float) half block gw (← natAt c "x")
    let (r, ty) := flatAxis (α := Float) half block gh (← natAt c "y")
    let input ← natsAt c "in"
    let want ← natsAt c "out"
    for ch in [0:3] do
      let g (row col : Nat) : Float := gain.getD ((row * gw + col) * 3 + ch) 0.0
      let gn := flatGain (g r i) (g r (i + 1)) (g (r + 1) i) (g (r + 1) (i + 1)) tx ty
      let got := flatDivide gn (input.getD ch 0)
      -- Where the exact quotient is within 0.01 of a half, the C++'s single precision quotient may round to the
      -- neighbouring code (see `flatDivide`); such a value may differ by one code, and is counted apart.
      let q := (input.getD ch 0).toFloat / gn
      if Float.abs (q - q.floor - 0.5) < 0.01 then
        record "scene: flatDivide, quotient near a half" 1.0 (Float.abs (got.toFloat - (want.getD ch 0).toFloat))
      else
        same "scene: flatDivide" got (want.getD ch 0)

def shoulderOf (j : Json) : IO (Shoulder Float) := do
  pure ⟨← numAt j "x_hi", ← numAt j "x_k", ← numAt j "t_k", ← numAt j "t_hi", ← numAt j "s_k", ← numAt j "c",
        ← numAt j "y_c"⟩

def nearShoulder (name : String) (got want : Shoulder Float) : CheckM Unit := do
  near name tolSingle got.xHi want.xHi
  near name tolSingle got.xK want.xK
  near name tolSingle got.tK want.tK
  near name tolSingle got.tHi want.tHi
  near name tolSingle got.sK want.sK
  near name tolSingle got.c want.c
  near name tolSingle got.yC want.yC

def checkPixel (root : System.FilePath) (j : Json) : CheckM Unit := do
  let prof ← readProfile (root / (← strAt j "bundle_dir") / (← strAt j "icc"))
  -- The profile as read here against the tables the C++ parsed.
  let pc ← get j "profile_check"
  same "profile: table sizes" prof.inCurves.x.size (← natAt pc "in_size")
  same "profile: table sizes" prof.outCurves.x.size (← natAt pc "out_size")
  same "profile: table sizes" prof.dim.x (← natAt pc "clut_dim")
  near "profile: table entries" tolSingle (prof.inCurves.x.getD 1000 0.0) (← numAt pc "in_r_1000")
  near "profile: table entries" tolSingle (prof.inCurves.y.getD 2000 0.0) (← numAt pc "in_g_2000")
  near "profile: table entries" tolSingle (prof.inCurves.z.getD 4095 0.0) (← numAt pc "in_b_4095")
  near "profile: table entries" tolSingle (prof.clut.getD 12345 0.0) (← numAt pc "clut_12345")
  near "profile: table entries" tolSingle (prof.clut.getD 200000 0.0) (← numAt pc "clut_200000")
  near "profile: table entries" tolSingle (prof.outCurves.y.getD 1 0.0) (← numAt pc "out_g_1")
  -- The merged matrix from the bundle's crosstalk correction and film base.
  let ratio : Float :=
    captureExposure (← numAt j "film_base_shutter") (← numAt j "film_base_iso") /
      captureExposure (← numAt j "scan_shutter") (← numAt j "scan_iso")
  let merged := mergedMatrix (mat3 (← numsAt j "crosstalk")) (vec3 (← numsAt j "film_base"))
    (← numAt j "normalization_target") ratio
  let wantMerged ← numsAt j "merged"
  near3 "scene: mergedMatrix" tolDouble merged.row0 (wantMerged.extract 0 3)
  near3 "scene: mergedMatrix" tolDouble merged.row1 (wantMerged.extract 3 6)
  near3 "scene: mergedMatrix" tolDouble merged.row2 (wantMerged.extract 6 9)
  let pixels ← (← arrAt j "pixels").mapM fun p => (do pure (vec3Nat (← (← arr p).mapM nat)) : IO (Vec3 Nat))
  for c in ← arrAt j "cases" do
    let m := candidateMatrix merged (← numAt c "e") (← numAt c "g") (← numAt c "b")
    let wantM ← numsAt c "matrix"
    near3 "scene: candidateMatrix" tolSingle m.row0 (wantM.extract 0 3)
    near3 "scene: candidateMatrix" tolSingle m.row1 (wantM.extract 3 6)
    near3 "scene: candidateMatrix" tolSingle m.row2 (wantM.extract 6 9)
    let knee ← numAt c "knee"
    let sh := setShoulder (← numAt c "x_hi")
    nearShoulder "scene: setShoulder" sh (← shoulderOf (← get c "shoulder"))
    let cfg : PipelineConfig Float := { profile := prof, knee := knee, shoulder := sh, tone := none }
    let outs ← arrAt c "out"
    for i in [0:pixels.size] do
      let v : Vec3 Float := loadU16 (pixels.getD i ⟨0, 0, 0⟩)
      let o := outs.getD i Json.null
      let lin := convertPixelLinear prof knee m v
      near3 "pixel: correctedRgb" tolSingle (correctedRgb knee m v) (← numsAt o "corrected")
      near3 "pixel: convertPixelPcs" tolSingle (convertPixelPcs prof knee m v) (← numsAt o "pcs")
      near3 "pixel: convertPixelLinear" tolSingle lin (← numsAt o "linear")
      near3 "pixel: convertPixelDisplay" tolSingle (convertPixelDisplay cfg m v) (← numsAt o "display")
      near3 "pixel: convertPixel" tolSingle (convertPixel cfg m v) (← numsAt o "encoded")
      let lab := pixelLab cfg m v
      let wantLab ← numsAt o "lab"
      near "pixel: pixelLab" tolLabPixel lab.L (wantLab.getD 0 0.0)
      near "pixel: pixelLab" tolLabPixel lab.a (wantLab.getD 1 0.0)
      near "pixel: pixelLab" tolLabPixel lab.b (wantLab.getD 2 0.0)
      same "pixel: pixelWhiteClipped" (if pixelWhiteClipped lin then 1 else 0)
        (if (← boolAt o "white_clipped") then 1 else 0)

def checkShoulder (j : Json) : CheckM Unit := do
  for s in ← arrAt j "set" do
    let sh := setShoulder (← numAt s "x_hi_in")
    nearShoulder "shoulder: setShoulder" sh (← shoulderOf s)
    for c in ← arrAt s "channel" do
      near "shoulder: applyShoulderChannel" tolSingle (applyShoulderChannel sh (← numAt c "x")) (← numAt c "y")
  for c in ← arrAt j "anchored" do
    near "shoulder: shoulderXHiAnchored" tolRoot (shoulderXHiAnchored (← numAt c "q")) (← numAt c "x_hi")
  let h ← get j "hist"
  let hist ← natsAt h "hist"
  for c in ← arrAt h "quantiles" do
    near "shoulder: maxHistQuantile" tolDouble (maxHistQuantile hist (← numAt c "q")) (← numAt c "value")
  near "shoulder: shoulderXHiFromHist" tolRoot (shoulderXHiFromHist hist) (← numAt h "x_hi")
  for c in ← arrAt h "bins" do
    same "shoulder: maxHistBin" (maxHistBin (← numAt c "v")) (← natAt c "bin")

def run (root : System.FilePath) : CheckM Unit := do
  let scene ← readJson "testvectors/scene.json"
  checkBase (← get scene "base")
  checkFlat (← get scene "flat")
  checkPixel root (← get scene "pixel")
  checkShoulder (← get scene "shoulder")

end Check

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
  let (_, groups) ← (Check.run root).run #[]
  let mut failures := 0
  let mut total := 0
  let pad (s : String) (n : Nat) := s ++ String.ofList (List.replicate (n - s.length) ' ')
  IO.println s!"{pad "group" 44}{pad "checks" 8}{pad "failed" 8}{pad "largest difference" 20}tolerance"
  for g in groups do
    total := total + g.count
    failures := failures + g.failures
    IO.println s!"{pad g.name 44}{pad (toString g.count) 8}{pad (toString g.failures) 8}{pad (sci g.maxDiff) 20}{sci g.tol}"
  IO.println s!"{total} comparisons, {failures} outside tolerance"
  pure (if failures == 0 then 0 else 1)
