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
precision from the same descriptors, and on the committed vectors they agree exactly. A compiler or
C library whose `pow` or `cbrt` rounds otherwise changes the descriptors in the last bits, which the
solve carries to the curve. -/
def tolSolve : Float := 1e-7

/-- numpy's percentile in single precision (`percentile_f32` interpolates in float32) of values near
3000. -/
def tolPercentile32 : Float := 5e-4

/-- The 99.5 % point of a histogram of 4096 bins over `[0, 2]`, filled from single precision values:
a pixel near a bin's edge may fall in the neighbouring bin, which moves the point by at most a
bin's width, `2 / 4096`. Observed: 3e-8 (the two histograms were equal). -/
def tolHistBin : Float := 2.0 / 4096.0

/-- The film base through the crosstalk correction, in double precision on both sides, over 65535. -/
def tolDoubleRel : Float := 1e-9

/-- A quantised code (`quantize_u8`, `quantize_u16`: `roundf` of a single precision value) is compared
exactly. The one exception is a value that the specification and the C++ put on the two sides of a
half (each value is compared within `tolSingle` on its own): there the codes differ by one, and such
codes are counted apart with a tolerance of one code. The quantisation itself is compared exactly on
the C++'s own values, except where the C++'s single precision product `v · 65535` (or `v · 255`) and
the exact one lie on the two sides of a half. -/
def straddlesHalf (lean cpp scale : Float) : Bool :=
  (lean * scale + 0.5).floor != (cpp * scale + 0.5).floor

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

/-- `v · scale` as the C++ forms it, in single precision. -/
def f32Product (v scale : Float) : Float := (v.toFloat32 * scale.toFloat32).toFloat

/-- Compares the quantisation of the C++'s own value with the C++'s code: exactly, or within one code
where the single precision product and the exact one lie on the two sides of a half. -/
def sameQuantize (name : String) (cpp scale : Float) (got want : Nat) : CheckM Unit := do
  if (cpp * scale + 0.5).floor != (f32Product cpp scale + 0.5).floor then
    record (name ++ ", across a half") 1.0 (Float.abs (got.toFloat - want.toFloat))
  else same name got want

/-- Compares a quantised code with the C++'s: exactly, or within one code where the two values lie on
the two sides of a half (`straddlesHalf`). -/
def sameCode (name : String) (lean cpp scale : Float) (got want : Nat) : CheckM Unit := do
  if straddlesHalf lean cpp scale then record (name ++ ", across a half") 1.0 (Float.abs (got.toFloat - want.toFloat))
  else same name got want

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
  near3 "scene: pcsWhite" tolDouble pcsWhite (← numsAt j "pcs_white")
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
  same "profile: matrixAccepted" (if prof.matrixAccepted then 1 else 0) 1
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
      let px := pixels.getD i ⟨0, 0, 0⟩
      let enc := convertPixel cfg m v
      let cor := correctedRgb knee m v
      let outP := outputPixel cfg m px
      let corP := correctedPixel cfg m px
      let prvP := previewPixel cfg m px
      let wantU16 ← natsAt o "u16"
      let wantC ← natsAt o "corrected_u16"
      let wantP ← natsAt o "preview_u8"
      let encC ← numsAt o "encoded"
      let corC ← numsAt o "corrected"
      for (val, cv, got, want) in [(enc.x, encC.getD 0 0.0, outP.x, wantU16.getD 0 0),
          (enc.y, encC.getD 1 0.0, outP.y, wantU16.getD 1 0), (enc.z, encC.getD 2 0.0, outP.z, wantU16.getD 2 0)] do
        sameCode "pixel: outputPixel" val cv 65535.0 got want
        sameQuantize "pixel: quantizeU16 of the C++ value" cv 65535.0 (quantizeU16 cv) want
      for (val, cv, got, want) in [(cor.x, corC.getD 0 0.0, corP.x, wantC.getD 0 0),
          (cor.y, corC.getD 1 0.0, corP.y, wantC.getD 1 0), (cor.z, corC.getD 2 0.0, corP.z, wantC.getD 2 0)] do
        sameCode "pixel: correctedPixel" val cv 65535.0 got want
        sameQuantize "pixel: quantizeU16 of the C++ value" cv 65535.0 (quantizeU16 cv) want
      for (val, cv, got, want) in [(enc.x, encC.getD 0 0.0, prvP.x, wantP.getD 0 0),
          (enc.y, encC.getD 1 0.0, prvP.y, wantP.getD 1 0), (enc.z, encC.getD 2 0.0, prvP.z, wantP.getD 2 0)] do
        sameCode "pixel: previewPixel" val cv 255.0 got want
        sameQuantize "pixel: quantizeU8 of the C++ value" cv 255.0 (quantizeU8 cv) want

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
  let dark := toneTable (← toneCurveOf (← get j "dark_curve"))
  for c in ← arrAt j "apply_dark" do
    near3 "tone: applyTone" tolSingle (applyTone dark (vec3 (← numsAt c "x"))) (← numsAt c "y")
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
  -- The plan on that image, and on a darker one where the black end steps the shares down.
  let dk ← get cl "dark"
  let imgDark := pixelsOf (← natsAt dk "img")
  let (dwb, dkb) := toneClipSample cfg m imgDark w h none
  let wantDark ← numsAt dk "none"
  near "tone: toneClipSample" tolDouble dwb (wantDark.getD 0 0.0)
  near "tone: toneClipSample" tolDouble dkb (wantDark.getD 1 0.0)
  for (pl, img, wb, kb) in [(← get cl "plan", img16, wb, kb), (← get dk "plan", imgDark, dwb, dkb)] do
    let pr ← numsAt pl "pred"
    let designed := tonePlan d (pr.getD 0 0.0) (pr.getD 1 0.0) (pr.getD 2 0.0) 1.0
    let (share, shadow, t, q) :=
      tonePlanForImage d (pr.getD 0 0.0) (pr.getD 1 0.0) (pr.getD 2 0.0) cfg m img w h wb kb designed
    near "tone: tonePlanForImage" tolDouble share (← numAt pl "share")
    near "tone: tonePlanForImage" tolDouble shadow (← numAt pl "shadow_share")
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
  for c in ← arrAt j "combined" do
    let av ← numsAt c "a"
    let rv ← numsAt c "ref"
    let mv ← numsAt c "u_med"
    let lv ← numsAt c "u_lo"
    let hold ← boolAt c "hold"
    let ic : InputCurves Float := ⟨⟨av.getD 0 0.0, 0.0, rv.getD 0 0.0, mv.getD 0 0.0, lv.getD 0 0.0, hold⟩,
      ⟨av.getD 1 0.0, 0.0, rv.getD 1 0.0, mv.getD 1 0.0, lv.getD 1 0.0, hold⟩⟩
    let cp := combinedProfile prof ic
    for e in ← arrAt c "entries" do
      let i ← natAt e "i"
      near "white: combinedProfile" tolSingle (cp.inCurves.x.getD i 0.0) (← numAt e "r")
      near "white: combinedProfile" tolSingle (cp.inCurves.y.getD i 0.0) (← numAt e "g")
      near "white: combinedProfile" tolSingle (cp.inCurves.z.getD i 0.0) (← numAt e "b")
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
  let cropC ← numAt j "crop_ratio"
  same "crop: searchCropRatio" (if Float.abs ((searchCropRatio : Float) - cropC) ≤ tolSingle then 1 else 0) 1
  for c in ← arrAt j "inner_crop" do
    let i ← arrAt c "in"
    let (x0, y0, w, h) := innerCrop (← nat (i.getD 0 Json.null)) (← nat (i.getD 1 Json.null)) (← num (i.getD 2 Json.null))
    let o ← natsAt c "out"
    for (got, want) in [(x0, o.getD 0 0), (y0, o.getD 1 0), (w, o.getD 2 0), (h, o.getD 3 0)] do
      same "crop: innerCrop" got want
  let fr ← get j "frame"
  let FW ← natAt fr "w"
  let FH ← natAt fr "h"
  let frame := pixelsOf (← natsAt fr "frame")
  -- The frame's highlight roll-off through bracket 3 of the TriLED bundle at the reference exposure's merged
  -- matrix: the histogram over the inner crop, its 99.5 % point and x_hi.
  let fbExposure := captureExposure (← numAt pixel "film_base_shutter") (← numAt pixel "film_base_iso")
  let merged := mergedMatrix (mat3 (← numsAt pixel "crosstalk")) (vec3 (← numsAt pixel "film_base"))
    (← numAt pixel "normalization_target") (fbExposure / captureExposure 0.125 100.0)
  let prof ← readProfile (root / (← strAt pixel "bundle_dir") / (← strAt pixel "icc"))
  let (cx, cy, cw, ch) := innerCrop FW FH (searchCropRatio : Float)
  for c in ← arrAt fr "rolloff" do
    let m := candidateMatrix merged (← numAt c "e") (← numAt c "g") (← numAt c "b")
    let lins := (Array.range (cw * ch)).map fun k =>
      convertPixelLinear prof (autoKnee : Float) m (loadU16 (frame.getD ((cy + k / cw) * FW + cx + k % cw) ⟨0, 0, 0⟩))
    let mh := maxHistOf lins
    near "shoulder: maxHistOf, the 99.5 % point" tolHistBin (maxHistQuantile mh 0.995) (← numAt c "q995")
    near "shoulder: maxHistOf, x_hi" tolRoot (shoulderXHiFromHist mh) (← numAt c "x_hi")
/-- The film base's decision on synthetic half-size decodes (`detect_film_base`): where the rebate
patch is placed, its box, the clip test, its median, how far the thinnest film outshines it, and the
source taken. The thinnest film itself comes from the segmentation, which is not specified; the
C++'s value is the input. -/
def checkFilmBase (j : Json) : CheckM Unit := do
  let X := mat3 (← numsAt j "crosstalk")
  let W ← natAt j "w"
  let H ← natAt j "h"
  let sensor ← numsAt j "sensor"
  let white ← numAt j "white"
  let sx := W.toFloat / sensor.getD 0 1.0
  let sy := H.toFloat / sensor.getD 1 1.0
  for c in ← arrAt j "cases" do
    let half := pixelsOf (← natsAt c "half")
    let thinnest := vec3 (← numsAt c "thinnest")
    let bs ← numsAt c "box_sensor"
    let placed := rebatePlaced (← numAt c "conf") sx sy
    let box := if placed then rebateBox (bs.getD 0 0.0) (bs.getD 1 0.0) (bs.getD 2 0.0) (bs.getD 3 0.0) sx sy W H
      else none
    let samples : Array (Vec3 Nat) := match box with
      | some (x0, y0, w, h) => (Array.range (w * h)).map fun k => half.getD ((y0 + k / w) * W + x0 + k % w) ⟨0, 0, 0⟩
      | none => #[]
    let clipped := box.isSome && patchClipped samples white
    same "film base: patchClipped" (if clipped then 1 else 0) (if (← boolAt c "rebate_clipped") then 1 else 0)
    let tested := box.isSome && !clipped
    same "film base: rebatePlaced, rebateBox" (if tested then 1 else 0) (if (← boolAt c "has_outshone") then 1 else 0)
    let dec : Vec3 Float :=
      ⟨medianU16 (samples.map (·.x)), medianU16 (samples.map (·.y)), medianU16 (samples.map (·.z))⟩
    let rebateOk : Option Bool := if tested then some (rebateAccepted X thinnest dec) else none
    if tested then near "film base: outshoneStops" tolSingle (outshoneStops thinnest dec) (← numAt c "outshone")
    let src := filmBaseSource false rebateOk (thinnestAccepted X thinnest)
    let wantSrc ← strAt c "source"
    same "film base: source (rebate, thinnest)"
      (if src = .rebate then 0 else if src = .thinnest then 1 else 2)
      (if wantSrc == "rebate" then 0 else if wantSrc == "statistics" then 1 else 2)
    let rgb := X.mulVec (if src = .rebate then dec else thinnest)
    near3 "film base: base through X" tolDoubleRel
      (rgb.map (· / 65535.0)) ((← numsAt c "rgb").map (· / 65535.0))
    if src = .rebate then
      match box with
      | some (x0, y0, w, h) =>
        let wb ← natsAt c "box"
        for (got, want) in [(x0, wb.getD 0 0), (y0, wb.getD 1 0), (w, wb.getD 2 0), (h, wb.getD 3 0)] do
          same "film base: rebatePlaced, rebateBox" got want
      | none => pure ()

def run (root : System.FilePath) : CheckM Unit := do
  let scene ← readJson "testvectors/scene.json"
  checkBase (← get scene "base")
  checkFlat (← get scene "flat")
  checkPixel root (← get scene "pixel")
  checkShoulder (← get scene "shoulder")
  checkTone root (← get scene "pixel") (← readJson "testvectors/scene_tone.json")
  checkWhite root (← readJson "testvectors/scene_white.json")
  let rd ← readJson "testvectors/scene_read.json"
  checkRead root (← get scene "pixel") rd
  checkFilmBase (← get rd "film_base")

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
  ("scene: pcsWhite", 3),
  ("scene: quantizeU8, quantizeU16", 12),
  ("scene: clipping rules", 14),
  ("tone: toneLabFprime", 7),
  ("scene: flatDivide", 52),
  ("scene: flatDivide, quotient near a half", 2),
  ("profile: matrixAccepted", 1),
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
  ("pixel: outputPixel", 598),
  ("pixel: quantizeU16 of the C++ value", 1199),
  ("pixel: correctedPixel", 600),
  ("pixel: previewPixel", 600),
  ("pixel: quantizeU8 of the C++ value", 600),
  ("pixel: outputPixel, across a half", 2),
  ("pixel: quantizeU16 of the C++ value, across a half", 1),
  ("shoulder: setShoulder", 77),
  ("shoulder: applyShoulderChannel", 110),
  ("shoulder: shoulderXHiAnchored", 11),
  ("shoulder: maxHistQuantile", 5),
  ("shoulder: shoulderXHiFromHist", 1),
  ("shoulder: maxHistBin", 8),
  ("tone: toneTable", 45),
  ("tone: toneCurveOk", 5),
  ("tone: toneTableY", 9),
  ("tone: applyTone", 30),
  ("tone: toneDescriptors", 7),
  ("tone: toneTargets", 36),
  ("tone: tonePlan (the solve)", 36),
  ("tone: tonePickClipShare", 4),
  ("tone: toneClipSample", 14),
  ("tone: tonePlanForImage", 20),
  ("white: curveStops", 15),
  ("white: inputCurve", 24),
  ("white: composeInputTable (size)", 1),
  ("white: composeInputTable", 9),
  ("white: combinedProfile", 42),
  ("white: filmBaseBalanceGains", 12),
  ("read: halfSizePixel", 144),
  ("read: grainAwarePixel", 576),
  ("read: pixelShiftPixel", 576),
  ("film base: medianU16", 5),
  ("film base: percentileF32", 20),
  ("crop: searchCropRatio", 1),
  ("crop: innerCrop", 24),
  ("shoulder: maxHistOf, the 99.5 % point", 2),
  ("shoulder: maxHistOf, x_hi", 2),
  ("film base: patchClipped", 4),
  ("film base: rebatePlaced, rebateBox", 8),
  ("film base: outshoneStops", 2),
  ("film base: source (rebate, thinnest)", 4),
  ("film base: base through X", 12)]

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
  IO.println s!"{pad "group" 60}{pad "checks" 8}{pad "failed" 8}{pad "largest difference" 20}tolerance"
  for g in groups do
    total := total + g.count
    failures := failures + g.failures
    IO.println s!"{pad g.name 60}{pad (toString g.count) 8}{pad (toString g.failures) 8}{pad (sci g.maxDiff) 20}{sci g.tol}"
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
