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
import Negicc.Tone.Defs
import Negicc.WhiteLight.Defs

open Lean Negicc Negicc.Scalar

namespace Check

/-! ## Tolerances

The C++ computes a pixel in single precision; the specification is evaluated in double precision.
One single precision operation is exact to 6e-8 of its result, and a pixel goes through a few
dozen of them, so the two agree to a few parts in a million. Each tolerance below is stated with its
reason and the largest difference observed, and is a small multiple of it; none was widened to make a
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

/-- The tone curve solved by Levenberg-Marquardt: both sides run the same iteration in double
precision; the descriptors it starts from differ in the last bits (the sRGB decode's `c * (1 / 12.92)`
against `c / 12.92`), which the solve carries to the curve. -/
def tolSolve : Float := 1e-7

/-- `E` solved for a median lightness: the C++ solves in single precision to within `2e-5 L*`, which
moves `E` by up to about `1e-5` of itself. -/
def tolSolveE : Float := 1e-6

/-- The fit's fixed part at Auto's `E`: the weights and moments of cells whose `L*` agree to
`tolLabPixel`, at an `E` that agrees to `tolSolveE`. -/
def tolFitSetup : Float := 1e-5

/-- The fit's data loss at a given point, a sum of squared residuals of order 10 to 100 built from
those values. -/
def tolFitLoss : Float := 3e-4

/-- The fit's result: the C++ runs the solve in single precision for its parameters and `E` (double
for the residuals); the two runs take the same steps and stop within the solve's own tolerance
(a gain below `1e-6` of the cost, a step below `1e-5`). The loss at the end point is compared with
`tolFitResultLoss`. -/
def tolFit : Float := 4e-3

@[inherit_doc tolFit] def tolFitResultLoss : Float := 3e-3

/-- numpy's percentile in single precision (`percentile_f32` interpolates in float32) of values near
3000. -/
def tolPercentile32 : Float := 5e-4

/-- `u16_percentile_linear` interpolates in single precision between 16-bit values (observed: no
difference on the vectors). -/
def tolPercentileU16 : Float := 1e-3

/-- The 99.5 % point of a histogram of 4096 bins over `[0, 2]`, filled from single precision values:
a pixel near a bin's edge may fall in the neighbouring bin, which moves the point by at most a
bin's width, `2 / 4096`. Observed: 3e-8 (the two histograms were equal). -/
def tolHistBin : Float := 2.0 / 4096.0

/-- An 8-bit code computed in single precision (`pcs_lab_to_srgb_u8`, codes up to 255). -/
def tolCode8 : Float := 3e-4

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
  near "scene: autoKnee" tolSingle autoKnee (← numAt j "auto_knee")
  for c in ← arrAt j "quantize" do
    same "scene: quantizeU8, quantizeU16" (quantizeU8 (← numAt c "x")) (← natAt c "u8")
    same "scene: quantizeU8, quantizeU16" (quantizeU16 (← numAt c "x")) (← natAt c "u16")
  for c in ← arrAt j "clip" do
    let x ← numAt c "x"
    same "scene: clipping rules" (if linearChannelWhiteClipped x then 1 else 0) (if (← boolAt c "white") then 1 else 0)
    same "scene: clipping rules" (if linearChannelBlackClipped x then 1 else 0) (if (← boolAt c "black") then 1 else 0)
  for c in ← arrAt j "tone_fprime" do
    near "tone: toneLabFprime" tolSingle (toneLabFprime (← numAt c "x")) (← numAt c "y")

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
    let toneCurve ← match (← get c "tone_curve").getArr? with
      | .ok a => do pure (some (ToneCurve.ofArray (← (a.mapM num : IO (Array Float)))))
      | .error _ => pure none
    let encoding : Encoding := if (← natAt c "encoding") = 1 then .linear else .srgb
    let cfg : PipelineConfig Float :=
      { profile := prof, knee := knee, shoulder := sh, tone := toneCurve.map toneTable, encoding := encoding }
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
      same "pixel: pixelBlackClipped" (if pixelBlackClipped lin then 1 else 0)
        (if (← boolAt o "black_clipped") then 1 else 0)
      let u16 := (convertPixel cfg m v).map quantizeU16
      let wantU16 ← natsAt o "u16"
      for (got, want) in [(u16.x, wantU16.getD 0 0), (u16.y, wantU16.getD 1 0), (u16.z, wantU16.getD 2 0)] do
        record "pixel: outputPixel (16-bit code)" 1.0 (Float.abs (got.toFloat - want.toFloat))
  -- The exposure normalisation: the frame's merged matrix over the reference's (the bundle's base at 1/8 s, ISO 100).
  let fbExposure := captureExposure (← numAt j "film_base_shutter") (← numAt j "film_base_iso")
  let reference := mergedMatrix (mat3 (← numsAt j "crosstalk")) (vec3 (← numsAt j "film_base"))
    (← numAt j "normalization_target") (fbExposure / referenceExposure)
  for c in ← arrAt j "e_norm" do
    let frame := mergedMatrix (mat3 (← numsAt j "crosstalk")) (vec3 (← numsAt j "film_base"))
      (← numAt j "normalization_target") (fbExposure / captureExposure (← numAt c "shutter") (← numAt c "iso"))
    near "scene: exposureNorm" tolSingle (exposureNorm frame reference) (← numAt c "e_norm")

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

def toneCurveOf (j : Json) : IO (ToneCurve Float) := do pure (ToneCurve.ofArray (← (← arr j).mapM num))

def nearCurve (name : String) (tol : Float) (got : ToneCurve Float) (want : Array Float) : CheckM Unit := do
  near name tol got.black (want.getD 0 0.0)
  near name tol got.white (want.getD 1 0.0)
  near name tol got.gamma (want.getD 2 0.0)
  near name tol got.contrast (want.getD 3 0.0)

def nearTargets (name : String) (tol : Float) (got : ToneTargets Float) (want : Array Float) : CheckM Unit := do
  near name tol got.L2 (want.getD 0 0.0)
  near name tol got.L50 (want.getD 1 0.0)
  near name tol got.L98 (want.getD 2 0.0)
  near name tol got.range (want.getD 3 0.0)

def pixelsOf (a : Array Nat) : Array (Vec3 Nat) :=
  (Array.range (a.size / 3)).map fun i => ⟨a.getD (3 * i) 0, a.getD (3 * i + 1) 0, a.getD (3 * i + 2) 0⟩

def checkTone (root : System.FilePath) (pixel : Json) (j : Json) : CheckM Unit := do
  for t in ← arrAt j "tables" do
    let lut := toneTable (← toneCurveOf (← get t "curve"))
    for e in ← arrAt t "entries" do
      near "tone: toneTable" tolSingle (lut.getD (← natAt e "i") 0.0) (← numAt e "y")
  for c in ← arrAt j "ok" do
    same "tone: toneCurveOk" (if toneCurveOk (← toneCurveOf (← get c "curve")) then 1 else 0)
      (if (← boolAt c "ok") then 1 else 0)
  let lut := toneTable (← toneCurveOf (← get j "pixel_curve"))
  for c in ← arrAt j "table_Y" do
    near "tone: toneTableY" tolSingle (toneTableY lut (← numAt c "x")) (← numAt c "y")
  for c in ← arrAt j "apply" do
    near3 "tone: applyTone" tolSingle (applyTone lut (vec3 (← numsAt c "x"))) (← numsAt c "y")
  let img ← get j "image8"
  let rgb := pixelsOf (← natsAt img "rgb")
  let d : ToneDescriptors Float := toneDescriptors rgb
  let wantD ← numsAt j "descriptors"
  let gotD := #[d.L2, d.L10, d.L25, d.L50, d.L75, d.L90, d.L98]
  for k in [0:7] do near "tone: toneDescriptors" tolDouble (gotD.getD k 0.0) (wantD.getD k 0.0)
  for c in ← arrAt j "plans" do
    let pr ← numsAt c "pred"
    let (t, q) := tonePlan d (pr.getD 0 0.0) (pr.getD 1 0.0) (pr.getD 2 0.0) (← numAt c "share")
    nearTargets "tone: toneTargets" tolDouble t (← numsAt c "targets")
    nearCurve "tone: tonePlan (the solve)" tolSolve q (← numsAt c "curve")
  for c in ← arrAt j "pick" do
    let after ← numsAt c "after"
    same "tone: tonePickClipShare" (tonePickClipShare (← numAt c "before") fun k => after.getD k 0.0) (← natAt c "k")
  -- The clip check through the bracket's conversion.
  let cl ← get j "clip"
  let prof ← readProfile (root / (← strAt pixel "bundle_dir") / (← strAt pixel "icc"))
  let m := mat3 (← numsAt cl "matrix")
  let cfg : PipelineConfig Float := { profile := prof, knee := autoKnee, shoulder := setShoulder (← numAt cl "x_hi"),
                                      tone := none }
  let w ← natAt cl "w"
  let h ← natAt cl "h"
  let img16 := pixelsOf (← natsAt cl "img")
  let (wb, kb) := toneClipSample cfg m img16 w h none
  let wantNone ← numsAt cl "none"
  near "tone: toneClipSample" tolDouble wb (wantNone.getD 0 0.0)
  near "tone: toneClipSample" tolDouble kb (wantNone.getD 1 0.0)
  for c in ← arrAt cl "curves" do
    let (wa, ka) := toneClipSample cfg m img16 w h (some (← toneCurveOf (← get c "curve")))
    near "tone: toneClipSample" tolDouble wa (← numAt c "white")
    near "tone: toneClipSample" tolDouble ka (← numAt c "black")
  let pl ← get cl "plan"
  let pr ← numsAt pl "pred"
  let designed := tonePlan d (pr.getD 0 0.0) (pr.getD 1 0.0) (pr.getD 2 0.0) 1.0
  let (share, t, q) := tonePlanForImage d (pr.getD 0 0.0) (pr.getD 1 0.0) (pr.getD 2 0.0) cfg m img16 w h wb designed
  near "tone: tonePlanForImage" tolDouble share (← numAt pl "share")
  nearTargets "tone: tonePlanForImage" tolDouble t (← numsAt pl "targets")
  nearCurve "tone: tonePlanForImage" tolSolve q (← numsAt pl "curve")

def holdOf (j : Json) : IO (Option Float) := do
  match (← get j "hold").getNum? with
  | .ok n => pure (some n.toFloat)
  | .error _ => pure none

def channelOf (j : Json) (hold : Bool) : IO (ChannelCurve Float) := do
  pure ⟨← numAt j "a", ← numAt j "k", ← numAt j "ref", ← numAt j "u_med", ← numAt j "u_lo", hold⟩

def curvesOf (j : Json) : IO (InputCurves Float) := do
  let hold ← boolAt j "hold"
  pure ⟨← channelOf (← get j "g") hold, ← channelOf (← get j "b") hold⟩

def nearChannel (name : String) (tol : Float) (got : ChannelCurve Float) (want : ChannelCurve Float) :
    CheckM Unit := do
  near name tol got.a want.a
  near name tol got.k want.k
  near name tol got.ref want.ref
  near name tol got.uMed want.uMed
  near name tol got.uLo want.uLo

def labOf (a : Array Float) : Lab Float := ⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0⟩

def checkWhite (root : System.FilePath) (j : Json) : CheckM Unit := do
  for c in ← arrAt j "stops" do
    near "white: curveStops" tolSingle
      (curveStops (← numAt c "u") (← numAt c "u_med") (← numAt c "a") (← numAt c "k") (← holdOf c)) (← numAt c "y")
  for c in ← arrAt j "input" do
    let hold ← holdOf c
    let ch : ChannelCurve Float := ⟨← numAt c "a", ← numAt c "k", ← numAt c "ref", ← numAt c "u_med",
      hold.getD 0.0, hold.isSome⟩
    near "white: inputCurve" tolSingle (inputCurve ch (← numAt c "v")) (← numAt c "y")
  let prof ← readProfile (root / (← strAt j "bundle_dir") / (← strAt j "icc"))
  let cp ← get j "compose"
  let chc : ChannelCurve Float := ⟨← numAt cp "a", ← numAt cp "k", ← numAt cp "ref", ← numAt cp "u_med",
    ← numAt cp "hold", true⟩
  let composed := composeInputTable prof.inCurves.y chc
  same "white: composeInputTable (size)" composed.size (← natAt cp "n")
  for e in ← arrAt cp "entries" do
    near "white: composeInputTable" tolSingle (composed.getD (← natAt e "i") 0.0) (← numAt e "y")
  for c in ← arrAt j "shape" do
    let hold ← holdOf c
    let ch : ChannelCurve Float := ⟨← numAt c "a", ← numAt c "k", ← numAt c "ref", ← numAt c "u_med",
      hold.getD 0.0, hold.isSome⟩
    same "white: curveShapeOk" (if curveShapeOk ch (← natAt c "n") 2.2 then 1 else 0)
      (if (← boolAt c "ok") then 1 else 0)
  -- The frame.
  let merged := mat3 (← numsAt j "merged_f32")
  let fb := vec3 (← numsAt j "film_base")
  let fr ← get j "frame"
  let grid := pixelsOf (← natsAt fr "grid")
  let neutralP ← numsAt fr "neutral_p"
  let mu ← numsAt fr "mu"
  let sigma ← numsAt fr "sigma"
  let labsOf3 := fun (a : Array Float) => (Array.range (a.size / 3)).map fun i =>
    (⟨a.getD (3 * i) 0.0, a.getD (3 * i + 1) 0.0, a.getD (3 * i + 2) 0.0⟩ : Lab Float)
  let field : JointFitField Float := ⟨labsOf3 mu, labsOf3 sigma, ← natsAt fr "bx", ← natsAt fr "by", ← natAt fr "nbx"⟩
  let frame : CurveFrame Float := ⟨merged, autoKnee, fb, grid, some neutralP⟩
  let cfg : PipelineConfig Float := { profile := prof, knee := autoKnee, shoulder := setShoulder (← numAt j "x_hi"),
                                      tone := none }
  let au ← numsAt j "auto"
  let (e0, g0, b0) := (au.getD 0 0.0, au.getD 1 0.0, au.getD 2 0.0)
  let rf ← get j "reference"
  let shape ← curvesOf (← get rf "shape")
  let ref := inputCurveReference shape merged e0 g0 b0 autoKnee fb grid
  let wantRef ← curvesOf (← get rf "curves")
  nearChannel "white: inputCurveReference" tolSingle ref.g wantRef.g
  nearChannel "white: inputCurveReference" tolSingle ref.b wantRef.b
  let md ← get j "model"
  let anc := anchored frame shape e0 g0 b0
  let wantAnc ← curvesOf (← get md "curves")
  nearChannel "white: anchored" tolSingle anc.g wantAnc.g
  nearChannel "white: anchored" tolSingle anc.b wantAnc.b
  let labs := cellLabs cfg frame shape e0 g0 b0
  let wantLabs ← arrAt md "labs"
  for i in [0:labs.size] do
    let wl ← ((← arr (wantLabs.getD i Json.null)).mapM num : IO (Array Float))
    let l := labs.getD i ⟨0.0, 0.0, 0.0⟩
    near "white: cellLab" tolLabPixel l.L (wl.getD 0 0.0)
    near "white: cellLab" tolLabPixel l.a (wl.getD 1 0.0)
    near "white: cellLab" tolLabPixel l.b (wl.getD 2 0.0)
  near "white: medianL" tolLabPixel (medianL cfg frame shape e0 g0 b0) (← numAt md "median_L")
  let targetL ← numAt j "target_L"
  let se ← get j "solve_e"
  let (okE, eE) := solveE (fun t => medianL cfg frame InputCurves.zero (exp t) g0 b0 - targetL) 2e-5 e0 (e0 * 0.25)
    (e0 * 4.0)
  same "white: solveE (reached)" (if okE then 1 else 0) (if (← boolAt se "ok") then 1 else 0)
  near "white: solveE" tolSolveE eE (← numAt se "e")
  let mo ← get j "moments"
  let mm := jointMoments labs (← natsAt mo "idx") (← numsAt mo "w")
  let wantM ← numsAt mo "m"
  let gotM := #[mm.L, mm.a, mm.b, mm.vL, mm.ka, mm.kb]
  for k in [0:6] do near "white: jointMoments" tolLabPixel (gotM.getD k 0.0) (wantM.getD k 0.0)
  for c in ← arrAt j "guard_cell" do
    near "white: greenGuardCellA" tolDouble (greenGuardCellA (← numAt c "a") (← numAt c "a_auto") (← numAt c "L_auto"))
      (← numAt c "y")
  -- The joint fit.
  let ft ← get j "fit"
  let P : FitProblem Float := ⟨cfg, frame, some field, e0, g0, b0, targetL, (prof.inCurves.y.size, prof.inCurves.z.size),
    CurveFitOptions.default, e0 * 0.25, e0 * 4.0⟩
  let (okA, eA) := P.solvePoint #[g0, b0, 0.0, 0.0] e0 true
  same "white: fit runs" (if okA then 1 else 0) (if (← boolAt ft "ok") then 1 else 0)
  let S := P.setup eA
  near "white: setup (anchors)" tolSingle S.m (← numAt ft "evidence")
  near "white: setup (anchors)" tolSingle S.w (← numAt ft "shrink")
  near "white: setup (anchors)" tolFitSetup S.slopeShare
    (← numAt ft "slope_share")
  near "white: setup (anchors)" tolFitSetup (Float.sqrt S.M0.vL)
    (← numAt ft "span_L")
  near "white: setup (anchors)" tolSingle S.sigAnc (← numAt ft "anchor_sd")
  near "white: setup (intent field)" tolFitSetup S.se (← numAt ft "intent_se")
  near "white: setup (intent field)" tolFitSetup
    (P.r4 S S.lab0 S.M0 / S.r4Scale) (← numAt ft "intent_auto")
  near "white: loss at Auto" tolFitLoss (P.loss S #[g0, b0, 0.0, 0.0] eA) (← numAt ft "loss_before")
  let wantCurves ← curvesOf (← get ft "curves")
  let pC : Array Float := #[← numAt ft "g", ← numAt ft "b", wantCurves.g.a, wantCurves.b.a]
  let eC ← numAt ft "e"
  near "white: loss at the C++ result" tolFitLoss (P.loss S pC eC) (← numAt ft "loss_after")
  let res := P.fit
  same "white: fit (guarded pass)" (if res.refit then 1 else 0) (if (← boolAt ft "refit") then 1 else 0)
  near "white: fit (E, g, b, bends)" tolFit res.e eC
  near "white: fit (E, g, b, bends)" tolFit res.g (pC.getD 0 0.0)
  near "white: fit (E, g, b, bends)" tolFit res.b (pC.getD 1 0.0)
  near "white: fit (E, g, b, bends)" tolFit res.curves.g.a (pC.getD 2 0.0)
  near "white: fit (E, g, b, bends)" tolFit res.curves.b.a (pC.getD 3 0.0)
  near "white: fit (loss after)" tolFitResultLoss res.lossAfter (← numAt ft "loss_after")
  -- The combined profile and the re-meter, from the C++ result's curves.
  let cb ← get j "combined"
  let anchoredC := anchored frame (P.shapeOf pC) eC (pC.getD 0 0.0) (pC.getD 1 0.0)
  let cc : PipelineConfig Float := { cfg with profile := combinedProfile prof wantCurves }
  near "white: gridMedianL (combined profile)" tolLabPixel (gridMedianL cc merged eC (pC.getD 0 0.0) (pC.getD 1 0.0) grid)
    (← numAt cb "median_L")
  near "white: anchored at the result" tolSingle anchoredC.g.ref wantCurves.g.ref
  near "white: solveEForMedianL" tolSolveE
    (solveEForMedianL cfg frame wantCurves eC (e0 * 0.25) (e0 * 4.0) (pC.getD 0 0.0) (pC.getD 1 0.0) targetL)
    (← numAt cb "remeter")
  for c in ← arrAt j "picture_cells" do
    let got := fallbackPictureCells (pixelsOf (← natsAt c "grid")) fb (← boolAt c "rebate")
    let want ← natsAt c "kept"
    same "white: fallbackPictureCells" got.size want.size
    for k in [0:want.size] do same "white: fallbackPictureCells" (got.getD k 0) (want.getD k 0)
  for c in ← arrAt j "balance" do
    let got := filmBaseBalanceGains (mat3 (← numsAt c "rgb_cam")) (vec3 (← numsAt c "base"))
      (vec3 (← numsAt c "profile_base"))
    let want ← numsAt c "k"
    match got with
    | some k => near3 "white: filmBaseBalanceGains" tolSingle k want
    | none => same "white: filmBaseBalanceGains (refused)" 1 (if want.getD 0 0.0 < 0.0 then 1 else 0)

def checkRead (root : System.FilePath) (pixel : Json) (j : Json) : CheckM Unit := do
  let mo ← get j "mosaic"
  let W ← natAt mo "w"
  let H ← natAt mo "h"
  let white ← natAt mo "white"
  let mk := fun (key : String) => (do pure (⟨W, H, ← natsAt mo key⟩ : Mosaic) : IO Mosaic)
  let f0 ← mk "f0"
  let f1 ← mk "f1"
  let f2 ← mk "f2"
  let f3 ← mk "f3"
  let half ← natsAt j "half"
  for y in [0:H / 2] do
    for x in [0:W / 2] do
      let p := halfSizePixel f0 (2 * y : Nat) (2 * x : Nat)
      let o := (y * (W / 2) + x) * 3
      for (got, want) in [(p.x, half.getD o 0), (p.y, half.getD (o + 1) 0), (p.z, half.getD (o + 2) 0)] do
        same "read: halfSizePixel" got want
  let ga ← natsAt j "grain_aware"
  let ps ← natsAt j "pixel_shift"
  for y in [0:H] do
    for x in [0:W] do
      let p : Vec3 Nat := grainAwarePixelU16 f0 (white.toFloat) debayerEps (y : Nat) (x : Nat)
      let o := (y * W + x) * 3
      for (got, want) in [(p.x, ga.getD o 0), (p.y, ga.getD (o + 1) 0), (p.z, ga.getD (o + 2) 0)] do
        same "read: grainAwarePixel" got want
      let q := pixelShiftPixel f0 f1 f2 f3 white (y : Nat) (x : Nat)
      for (got, want) in [(q.x, ps.getD o 0), (q.y, ps.getD (o + 1) 0), (q.z, ps.getD (o + 2) 0)] do
        same "read: pixelShiftPixel" got want
  for c in ← arrAt j "stats" do
    let v ← numsAt c "f32"
    near "film base: medianU16" tolDouble (medianU16 (← natsAt c "u16")) (← numAt c "median_u16")
    for p in ← arrAt c "pct" do
      near "film base: percentileF32" tolPercentile32 (percentileF32 v (← numAt p "q")) (← numAt p "y")
  for c in ← arrAt j "enclosure" do
    let i ← numsAt c "in"
    let o ← numsAt c "out"
    let r : Enclosure Float := bracketEnclosure (i.getD 0 0.0) (i.getD 1 0.0) (i.getD 2 0.0) (i.getD 3 0.0) (i.getD 4 0.0)
    same "bracket: bracketEnclosure (flags)" (if r.feasible then 1 else 0) (if o.getD 0 0.0 > 0.5 then 1 else 0)
    same "bracket: bracketEnclosure (flags)" (if r.admissible then 1 else 0) (if o.getD 1 0.0 > 0.5 then 1 else 0)
    same "bracket: bracketEnclosure (flags)" (if r.usableInverted then 1 else 0) (if o.getD 7 0.0 > 0.5 then 1 else 0)
    let got := #[r.eMin, r.eMax, r.searchLo, r.searchHi, r.eCentre]
    let want := #[o.getD 2 0.0, o.getD 3 0.0, o.getD 4 0.0, o.getD 5 0.0, o.getD 6 0.0]
    for k in [0:5] do near "bracket: bracketEnclosure" tolSingle ((got.getD k 0.0) / (want.getD k 1.0)) 1.0
    near "bracket: bracketEnclosure" tolSingle r.slackEv (o.getD 8 0.0)
    near "bracket: bracketEnclosure" tolSingle r.fillRatio (o.getD 9 0.0)
  for c in ← arrAt j "inner_crop" do
    let i ← arrAt c "in"
    let (x0, y0, w, h) := innerCrop (← nat (i.getD 0 Json.null)) (← nat (i.getD 1 Json.null)) (← num (i.getD 2 Json.null))
    let o ← natsAt c "out"
    for (got, want) in [(x0, o.getD 0 0), (y0, o.getD 1 0), (w, o.getD 2 0), (h, o.getD 3 0)] do
      same "bracket: innerCrop" got want
  let up ← get j "u16_percentile"
  let total ← natAt up "total"
  let mut hist : Array Nat := Array.replicate 65536 0
  for e in ← arrAt up "hist" do
    let a ← ((← arr e).mapM nat : IO (Array Nat))
    hist := hist.set! (a.getD 0 0) (a.getD 1 0)
  for p in ← arrAt up "pct" do
    near "bracket: u16PercentileLinear" tolPercentileU16 (u16PercentileLinear hist total (← numAt p "q")) (← numAt p "y")
  let mt ← get j "meter"
  let FW ← natAt mt "w"
  let FH ← natAt mt "h"
  let frame := pixelsOf (← natsAt mt "frame")
  let (gh, gt) := greenHistogram frame FW FH (0.9 : Float)
  let (gLo, gHi) := sceneGreenRange gh gt
  for b in ← arrAt mt "bundles" do
    let wl ← boolAt b "white_light"
    let eNorm ← numAt b "e_norm"
    for t in ← arrAt b "targets" do
      let r := meteredEnclosure wl eNorm (← numAt t "g_min") (← numAt t "g_max") (← numAt t "exposure") gLo gHi
      same "bracket: meteredEnclosure (feasible)" (if r.feasible then 1 else 0) (if (← boolAt t "feasible") then 1 else 0)
      let want ← numsAt t "range"
      let got := #[r.eMin, r.eMax, r.eCentre, r.searchLo, r.searchHi]
      for k in [0:5] do near "bracket: meteredEnclosure" tolSingle ((got.getD k 0.0) / (want.getD k 1.0)) 1.0
  -- The frame's highlight roll-off through bracket 3 of the TriLED bundle, scanned at 1/4 s, ISO 100.
  let fbExposure := captureExposure (← numAt pixel "film_base_shutter") (← numAt pixel "film_base_iso")
  let X := mat3 (← numsAt pixel "crosstalk")
  let fbv := vec3 (← numsAt pixel "film_base")
  let nT ← numAt pixel "normalization_target"
  let frameM := mergedMatrix X fbv nT (fbExposure / captureExposure 0.25 100.0)
  let eNormT := exposureNorm frameM (mergedMatrix X fbv nT (fbExposure / referenceExposure))
  let prof ← readProfile (root / (← strAt pixel "bundle_dir") / (← strAt pixel "icc"))
  let (cx, cy, cw, ch) := innerCrop FW FH (0.9 : Float)
  for c in ← arrAt mt "rolloff" do
    let m := candidateMatrix (frameM.divScalar eNormT) ((← numAt c "e") * eNormT) (← numAt c "g") (← numAt c "b")
    let lins := (Array.range (cw * ch)).map fun k =>
      convertPixelLinear prof (autoKnee : Float) m (loadU16 (frame.getD ((cy + k / cw) * FW + cx + k % cw) ⟨0, 0, 0⟩))
    let mh := maxHistOf lins
    near "shoulder: maxHistOf, the 99.5 % point" tolHistBin (maxHistQuantile mh 0.995) (← numAt c "q995")
    near "shoulder: maxHistOf, x_hi" tolRoot (shoulderXHiFromHist mh) (← numAt c "x_hi")
  for c in ← arrAt j "lab_to_srgb" do
    near3 "bracket: pcsLabToSrgbU8" tolCode8 (pcsLabToSrgbU8 (labOf (← numsAt c "lab"))) (← numsAt c "rgb")
  let labs ← (← arrAt j "lab_to_srgb").mapM fun c => (do pure (labOf (← numsAt c "lab")) : IO (Lab Float))
  for c in ← arrAt j "ceiling" do
    let rows ← natAt c "rows"
    near3 "bracket: profileHighlightCeiling" tolCode8 (profileHighlightCeiling (labs.toList.take rows)) (← numsAt c "ceil")

def run (root : System.FilePath) : CheckM Unit := do
  let scene ← readJson "testvectors/scene.json"
  checkBase (← get scene "base")
  checkFlat (← get scene "flat")
  checkPixel root (← get scene "pixel")
  checkShoulder (← get scene "shoulder")
  checkTone root (← get scene "pixel") (← readJson "testvectors/scene_tone.json")
  checkWhite root (← readJson "testvectors/scene_white.json")
  checkRead root (← get scene "pixel") (← readJson "testvectors/scene_read.json")

/-- The number of comparisons each group makes on the committed vectors. A group with another count,
or one missing or unknown, is a failure, so that a short or truncated vectors file cannot pass. When the
vectors are generated again with other cases, these numbers change with them. -/
def expectedCounts : List (String × Nat) := [
  ("base: srgbEncode", 12),
  ("base: srgbDecode", 10),
  ("base: labF", 8),
  ("base: linearSrgbToLab", 30),
  ("base: linearSrgbToL", 10),
  ("base: srgbToLab", 30),
  ("base: tableEval", 10),
  ("scene: kneeCompress", 16),
  ("scene: autoKnee", 1),
  ("scene: quantizeU8, quantizeU16", 12),
  ("scene: clipping rules", 14),
  ("tone: toneLabFprime", 7),
  ("scene: flatDivide", 52),
  ("scene: flatDivide, quotient near a half", 2),
  ("profile: table sizes", 3),
  ("profile: table entries", 6),
  ("scene: mergedMatrix", 9),
  ("scene: candidateMatrix", 90),
  ("scene: setShoulder", 70),
  ("pixel: correctedRgb", 600),
  ("pixel: convertPixelPcs", 600),
  ("pixel: convertPixelLinear", 600),
  ("pixel: convertPixelDisplay", 600),
  ("pixel: convertPixel", 600),
  ("pixel: pixelLab", 600),
  ("pixel: pixelWhiteClipped", 200),
  ("pixel: pixelBlackClipped", 200),
  ("pixel: outputPixel (16-bit code)", 600),
  ("scene: exposureNorm", 3),
  ("shoulder: setShoulder", 77),
  ("shoulder: applyShoulderChannel", 110),
  ("shoulder: shoulderXHiAnchored", 11),
  ("shoulder: maxHistQuantile", 5),
  ("shoulder: shoulderXHiFromHist", 1),
  ("shoulder: maxHistBin", 8),
  ("tone: toneTable", 45),
  ("tone: toneCurveOk", 5),
  ("tone: toneTableY", 9),
  ("tone: applyTone", 21),
  ("tone: toneDescriptors", 7),
  ("tone: toneTargets", 36),
  ("tone: tonePlan (the solve)", 36),
  ("tone: tonePickClipShare", 4),
  ("tone: toneClipSample", 12),
  ("tone: tonePlanForImage", 9),
  ("white: curveStops", 15),
  ("white: inputCurve", 24),
  ("white: composeInputTable (size)", 1),
  ("white: composeInputTable", 9),
  ("white: curveShapeOk", 6),
  ("white: inputCurveReference", 10),
  ("white: anchored", 10),
  ("white: cellLab", 144),
  ("white: medianL", 1),
  ("white: solveE (reached)", 1),
  ("white: solveE", 1),
  ("white: jointMoments", 6),
  ("white: greenGuardCellA", 6),
  ("white: fit runs", 1),
  ("white: setup (anchors)", 5),
  ("white: setup (intent field)", 2),
  ("white: loss at Auto", 1),
  ("white: loss at the C++ result", 1),
  ("white: fit (guarded pass)", 1),
  ("white: fit (E, g, b, bends)", 5),
  ("white: fit (loss after)", 1),
  ("white: gridMedianL (combined profile)", 1),
  ("white: anchored at the result", 1),
  ("white: solveEForMedianL", 1),
  ("white: fallbackPictureCells", 86),
  ("white: filmBaseBalanceGains", 12),
  ("read: halfSizePixel", 144),
  ("read: grainAwarePixel", 576),
  ("read: pixelShiftPixel", 576),
  ("film base: medianU16", 5),
  ("film base: percentileF32", 20),
  ("bracket: bracketEnclosure (flags)", 15),
  ("bracket: bracketEnclosure", 35),
  ("bracket: innerCrop", 20),
  ("bracket: u16PercentileLinear", 5),
  ("bracket: meteredEnclosure (feasible)", 14),
  ("bracket: meteredEnclosure", 70),
  ("bracket: pcsLabToSrgbU8", 18),
  ("bracket: profileHighlightCeiling", 9),
  ("shoulder: maxHistOf, the 99.5 % point", 2),
  ("shoulder: maxHistOf, x_hi", 2)]

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
  let mut countErrors := 0
  for (name, n) in Check.expectedCounts do
    match groups.find? (·.name == name) with
    | some g => if g.count != n then
        IO.println s!"group [{name}]: {g.count} comparisons, {n} expected"
        countErrors := countErrors + 1
    | none =>
      IO.println s!"group [{name}]: missing"
      countErrors := countErrors + 1
  for g in groups do
    unless Check.expectedCounts.any (·.1 == g.name) do
      IO.println s!"group [{g.name}]: not in the expected counts"
      countErrors := countErrors + 1
  if countErrors > 0 then IO.println s!"{countErrors} groups with an unexpected number of comparisons"
  pure (if failures == 0 && countErrors == 0 then 0 else 1)
