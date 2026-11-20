/-
The numeric check of the print emulation: the specification, evaluated over `Float`, against values
computed by the C++.

Run it from this directory with `lake exe checkPrint`. It reads `testvectors/print.json`, the
spektrafilm print profiles and the bundles' print ICCs it names (files of the repository), evaluates
the Lean definitions on the vectors' inputs and compares the results with the values the C++
functions gave. It prints one line per group of comparisons and ends with a non-zero exit code if
any comparison is outside its tolerance.
-/
import Lean.Data.Json
import Negicc.Base.Defs
import Negicc.Scene.Defs
import Negicc.Print.Defs

open Lean Negicc Negicc.Scalar

namespace CheckPrint

/-! ## Tolerances

The print emulation's C++ computes in double precision, except the conversion's pixel stages and
the tables it hands them (single precision), and the CLF's text, which prints numbers to 6 or 10
significant digits. Each tolerance is stated with its reason; none was widened to make a comparison
pass. -/

/-- Values the C++ computes in double precision from the same inputs by the same operations in the
same order. The largest difference observed is of the order of 1e-15. -/
def tolDouble : Float := 1e-9

/-- Values that pass through single precision: the conversion's pixel stages, the input tables as
the conversion holds them (`float`), a table scaled in `float`, the floor factor (a `float`). -/
def tolSingle : Float := 5e-6

/-- Values computed in double precision through the soft knee's inverse: the C++'s knee is the single
precision `0.95f` widened to double, 0.94999998807907, the specification's the decimal 0.95. Above
the knee the inverse's `1 − knee` differs by 2.4e-7 of itself, which moves a curve's value by up to
1.3e-8 (observed). -/
def tolKnee : Float := 5e-8

/-- CIELAB at the end of a pixel's conversion (as `Check.lean`'s `tolLabPixel`): a relative
difference of 2e-6 in the linear value becomes up to 3e-4 in `a*`. -/
def tolLabPixel : Float := 1e-3

/-- A number printed with `%.6g` in the CLF: half a unit in the sixth significant digit, relative. -/
def relG6 : Float := 5e-6

/-- A number printed with `%.10g` in the CLF, relative. -/
def relG10 : Float := 5e-10

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

/-- Compares two numbers within a tolerance relative to the C++ value (for printed numbers); the
recorded difference is the relative one. -/
def nearRel (name : String) (rel : Float) (got want : Float) : CheckM Unit := do
  let scale := if Float.abs want > 1e-300 then Float.abs want else 1.0
  let diff := Float.abs (got - want) / scale
  let diff := if diff.isNaN then 1e30 else diff
  unless diff ≤ rel do IO.println s!"  outside tolerance in [{name}]: Lean {got}, C++ {want}"
  record name rel diff

def near3 (name : String) (tol : Float) (got : Vec3 Float) (want : Array Float) : CheckM Unit := do
  near name tol got.x (want.getD 0 0.0)
  near name tol got.y (want.getD 1 0.0)
  near name tol got.z (want.getD 2 0.0)

def nearMat (name : String) (tol : Float) (got : Mat3 Float) (want : Array Float) : CheckM Unit := do
  near3 name tol got.row0 (want.extract 0 3)
  near3 name tol got.row1 (want.extract 3 6)
  near3 name tol got.row2 (want.extract 6 9)

def nearArr (name : String) (tol : Float) (got want : Array Float) : CheckM Unit := do
  if got.size != want.size then
    IO.println s!"  different sizes in [{name}]: Lean {got.size}, C++ {want.size}"
    record name tol 1e30
  for i in [0:want.size] do near name tol (got.getD i 0.0) (want.getD i 0.0)

def same (name : String) (got want : Nat) : CheckM Unit := do
  unless got == want do IO.println s!"  different in [{name}]: Lean {got}, C++ {want}"
  record name 0.0 (if got == want then 0.0 else 1.0)

def sameBool (name : String) (got want : Bool) : CheckM Unit := same name (if got then 1 else 0) (if want then 1 else 0)

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
def nums (j : Json) : IO (Array Float) := do (← arr j).mapM num
def numsAt (j : Json) (key : String) : IO (Array Float) := do nums (← get j key)
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
def vec3At (j : Json) (key : String) : IO (Vec3 Float) := do pure (vec3 (← numsAt j key))
def matAt (j : Json) (key : String) : IO (Mat3 Float) := do pure (Mat3.ofArray (← numsAt j key))
/-- Three arrays of numbers. -/
def arrs3At (j : Json) (key : String) : IO (Vec3 (Array Float)) := do
  let a ← arrAt j key
  pure ⟨← nums (a.getD 0 Json.null), ← nums (a.getD 1 Json.null), ← nums (a.getD 2 Json.null)⟩

def readJson (path : System.FilePath) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => fail s!"{path}: {e}"

/-! ## Reading a spektrafilm print profile -/

/-- A number or a null. -/
def optNum (j : Json) : IO (Option Float) :=
  match j with
  | .null => pure none
  | _ => do pure (some (← num j))

/-- Column `c` of an array of rows, or the array itself for `c = none`. -/
def column (rows : Array Json) (c : Option Nat) : IO (Array (Option Float)) :=
  rows.mapM fun r => match c with
    | none => optNum r
    | some k => do
      match r.getArr? with
      | .ok row => optNum (row.getD k Json.null)
      | .error _ => pure none

def readSpektrafilm (path : System.FilePath) : IO (SpektrafilmProfile Float) := do
  let j ← readJson path
  let d ← get j "data"
  let info ← get j "info"
  let support ← strAt info "support"
  let ls ← arrAt d "log_sensitivity"
  let cd ← arrAt d "channel_density"
  let dc ← arrAt d "density_curves"
  let curve (c : Nat) : IO (Array Float) := dc.mapM fun r => do num ((← arr r).getD c Json.null)
  pure { wavelengths := ← numsAt d "wavelengths",
         logSensitivity := ⟨← column ls (some 0), ← column ls (some 1), ← column ls (some 2)⟩,
         channelDensity := ⟨← column cd (some 0), ← column cd (some 1), ← column cd (some 2)⟩,
         baseDensity := ← column (← arrAt d "base_density") none,
         logExposure := ← numsAt d "log_exposure",
         densityCurves := ⟨← curve 0, ← curve 1, ← curve 2⟩,
         paper := support == "paper" }

/-! ## Reading a print ICC's lut16 codes -/

def be16 (b : ByteArray) (o : Nat) : Nat := b[o]!.toNat * 256 + b[o + 1]!.toNat
def be32 (b : ByteArray) (o : Nat) : Nat := be16 b o * 65536 + be16 b (o + 2)

/-- The A2B0 lut16's input curve codes (three tables), table codes and output curve codes. -/
structure Lut16 where
  inCodes : Array (Array Nat)
  table : Array Nat
  outCodes : Array (Array Nat)
  grid : Nat

def readLut16 (path : System.FilePath) : IO Lut16 := do
  let b ← IO.FS.readBinFile path
  let tags := be32 b 128
  let mut found : Option Nat := none
  for i in [0:tags] do
    if be32 b (132 + 12 * i) == 0x41324230 then found := some (be32 b (132 + 12 * i + 4))
  let some o := found | fail s!"{path}: no A2B0 tag"
  unless be32 b o == 0x6D667432 do fail s!"{path}: the A2B0 tag is not a lut16Type"
  let grid := b[o + 10]!.toNat
  let nIn := be16 b (o + 48)
  let nOut := be16 b (o + 50)
  let inBase := o + 52
  let clutBase := inBase + 2 * 3 * nIn
  let clutLen := grid * grid * grid * 3
  let outBase := clutBase + 2 * clutLen
  let codes (base n : Nat) : Array Nat := (Array.range n).map fun i => be16 b (base + 2 * i)
  pure { inCodes := #[codes inBase nIn, codes (inBase + 2 * nIn) nIn, codes (inBase + 4 * nIn) nIn],
         table := codes clutBase clutLen,
         outCodes := #[codes outBase nOut, codes (outBase + 2 * nOut) nOut, codes (outBase + 4 * nOut) nOut],
         grid := grid }

/-! ## The bundle's inputs -/

structure Bundle where
  dir : String
  n0 : Float
  crosstalk : Mat3 Float
  filmBase : Vec3 Float
  film : FilmModel Float
  dm : String → IO (DensityMatrices Float)
  grey : GreyBalance Float
  aMid : Array Float
  tilt : Array (Vec3 Float)
  bracketNames : Array String
  greyLight : Array (Vec3 Float)

def readBundle (j : Json) : IO Bundle := do
  let fm ← get j "film_model"
  let dms ← get j "density_matrices"
  let g ← get j "grey_balance"
  let brs ← arrAt j "brackets"
  let targets ← arrAt j "targets"
  -- The brackets in the order of the bundle's targets (the continuum's order).
  let mut aMid := #[]
  let mut tilt := #[]
  let mut names := #[]
  let mut greyLight := #[]
  for t in targets do
    let name ← match t.getStr? with | .ok s => pure s | .error e => fail e
    for b in brs do
      if (← strAt b "name") == name then
        names := names.push name
        tilt := tilt.push (← vec3At b "tilt")
        greyLight := greyLight.push (← vec3At b "grey_light")
        aMid := aMid.push (← numAt b "mid_apd")
  let dm (key : String) : IO (DensityMatrices Float) := do
    let d ← get dms key
    pure { m := ← matAt d "m", filmToMeasure := ← matAt d "film_to_measure",
           lightToFilm := ← matAt d "light_to_film", factors := ← vec3At d "light_density_factors" }
  pure { dir := ← strAt j "bundle_dir", n0 := (← numAt j "normalization_target") / 65535.0,
         crosstalk := ← matAt j "crosstalk", filmBase := ← vec3At j "film_base",
         film := { dye := ← arrs3At fm "dye", base := ← numsAt fm "base",
                   logExposure := ← numsAt fm "log_exposure", neutral := ← arrs3At fm "neutral" },
         dm := dm,
         grey := { start := ← numAt g "start", step := ← numAt g "step", apd := ← arrs3At g "apd" },
         aMid := aMid, tilt := tilt, bracketNames := names, greyLight := greyLight }

/-! ## The checks -/

def checkGrey (b : Bundle) (j : Json) (tag : String) : CheckM Unit := do
  let mApd := (← b.dm "apd").m
  for i in [0:b.aMid.size] do
    near s!"{tag}grey: bracketMidApd" tolDouble (bracketMidApd mApd (b.greyLight.getD i ⟨0, 0, 0⟩))
      (b.aMid.getD i 0.0)
  for c in ← arr j do
    let a ← numAt c "a"
    near3 s!"{tag}grey: curveGreyAt" tolDouble (curveGreyAt b.grey a) (← numsAt c "grey_light")
    near3 s!"{tag}grey: curveSlopesAt" tolDouble (curveSlopesAt b.grey mApd a) (← numsAt c "slopes")
    near3 s!"{tag}grey: printTiltAt" tolDouble (printTiltAt b.aMid b.tilt a) (← numsAt c "tilt")

/-- The bundle's print ICC's colour table and input curves against the specification: every table
code within one code of the table computed here, and the reference bracket's curves within one code
(the criterion `bracket_print_profiles` takes the bundle's table by). -/
def checkBundleIcc (root : System.FilePath) (b : Bundle) (j : Json) (tag : String)
    (m : PrintMaterial Float) (pApd : Mat3 Float) (paper : Vec3 (Array Float)) : CheckM Lut16 := do
  let lut ← readLut16 (root / (← strAt j "icc"))
  let mApd := (← b.dm "apd").m
  let table := printTableCodes (printTable m pApd mApd paper)
  same s!"{tag}print ICC: table size" table.size lut.table.size
  for i in [0:table.size] do
    record s!"{tag}print ICC: table codes (bundle file)" 1.0
      (Float.abs ((table.getD i 0).toFloat - (lut.table.getD i 0).toFloat))
  let ref ← strAt j "reference_bracket"
  let some k := b.bracketNames.findIdx? (· == ref) | fail s!"no bracket {ref}"
  let curves := bracketPrintCurves b.grey paper (b.aMid.getD k 0.0) b.n0 (b.tilt.getD k ⟨0, 0, 0⟩)
    ⟨1.0, 1.0, 1.0⟩
  for c in [0:3] do
    let mine := (if c = 0 then curves.x else if c = 1 then curves.y else curves.z).map iccCode16
    let theirs := lut.inCodes.getD c #[]
    for i in [0:mine.size] do
      record s!"{tag}print ICC: input curve codes (bundle file)" 1.0
        (Float.abs ((mine.getD i 0).toFloat - (theirs.getD i 0).toFloat))
  pure lut

def checkMaterial (root : System.FilePath) (b : Bundle) (j : Json) (tag : String) : CheckM Unit := do
  let prof ← readSpektrafilm (root / (← strAt j "profile"))
  let some m := loadMaterial prof | fail s!"{tag}: the material does not load"
  -- The material as loaded.
  nearArr s!"{tag}material: log exposure, curves" 0.0 m.logExposure (← numsAt j "log_exposure")
  let curves ← arrs3At j "curves"
  nearArr s!"{tag}material: log exposure, curves" 0.0 m.curves.x curves.x
  nearArr s!"{tag}material: log exposure, curves" 0.0 m.curves.y curves.y
  nearArr s!"{tag}material: log exposure, curves" 0.0 m.curves.z curves.z
  let dye ← arrs3At j "dye"
  nearArr s!"{tag}material: resampleColumn (dyes, base)" tolDouble m.dye.x dye.x
  nearArr s!"{tag}material: resampleColumn (dyes, base)" tolDouble m.dye.y dye.y
  nearArr s!"{tag}material: resampleColumn (dyes, base)" tolDouble m.dye.z dye.z
  nearArr s!"{tag}material: resampleColumn (dyes, base)" tolDouble m.base (← numsAt j "base")
  let obs ← arrs3At j "observer"
  nearArr s!"{tag}material: observerWeights" tolDouble m.observer.x obs.x
  nearArr s!"{tag}material: observerWeights" tolDouble m.observer.y obs.y
  nearArr s!"{tag}material: observerWeights" tolDouble m.observer.z obs.z
  near3 s!"{tag}material: observerWhite" tolDouble m.white (← numsAt j "white")
  near3 s!"{tag}material: aimSolve" tolDouble m.aim (← numsAt j "aim")
  near3 s!"{tag}material: paperXyz" tolDouble (paperXyz m) (← numsAt j "paper_xyz")
  nearMat s!"{tag}material: toSrgbMatrix" tolDouble (toSrgbMatrix m.white) (← numsAt j "to_srgb")
  let tp ← get j "to_print"
  for key in ["statusm", "rp180", "apd"] do
    let some p := materialPrintMatrix prof b.film (← b.dm key) | fail s!"{tag}: no print matrix {key}"
    nearMat s!"{tag}material: materialPrintMatrix (Q, P)" tolDouble p (← nums (← get tp key))
  let some pApd := materialPrintMatrix prof b.film (← b.dm "apd") | fail "no APD print matrix"
  -- Paper_c and the input curves.
  let paper := paperCurve m pApd
  let pc ← arrs3At j "paper_curve"
  nearArr s!"{tag}curves: paperCurve" tolDouble paper.x pc.x
  nearArr s!"{tag}curves: paperCurve" tolDouble paper.y pc.y
  nearArr s!"{tag}curves: paperCurve" tolDouble paper.z pc.z
  let ix ← natsAt j "curve_indices"
  let compareCurves (name : String) (tol : Float) (got : Vec3 (Array Float)) (want : Json) : CheckM Unit := do
    let w ← arrs3At want "_"
    for i in [0:ix.size] do
      near name tol (got.x.getD (ix.getD i 0) 0.0) (w.x.getD i 0.0)
      near name tol (got.y.getD (ix.getD i 0) 0.0) (w.y.getD i 0.0)
      near name tol (got.z.getD (ix.getD i 0) 0.0) (w.z.getD i 0.0)
  for c in ← arrAt j "bracket_curves" do
    let got := bracketPrintCurves b.grey paper (← numAt c "a_mid") b.n0 (← vec3At c "tilt") ⟨1.0, 1.0, 1.0⟩
    compareCurves s!"{tag}curves: bracketPrintCurves" tolKnee got (Json.mkObj [("_", ← get c "curves")])
  let cont : PrintContinuum Float := { grey := b.grey, paper := paper, n0 := b.n0, aMid := b.aMid, tilt := b.tilt }
  near s!"{tag}curves: aLo, aHi" tolDouble cont.aLo (← numAt j "a_lo")
  near s!"{tag}curves: aLo, aHi" tolDouble cont.aHi (← numAt j "a_hi")
  let gainsOf (c : Json) : IO (Vec3 Float) := do
    match ← get c "gains" with
    | .null => pure ⟨1.0, 1.0, 1.0⟩
    | g => pure (vec3 (← nums g))
  for c in ← arrAt j "continuum" do
    let a ← numAt c "a"
    let gains ← gainsOf c
    near3 s!"{tag}curves: printTiltAt (continuum)" tolDouble (printTiltAt b.aMid b.tilt a) (← numsAt c "tilt")
    compareCurves s!"{tag}curves: curvesAt" tolKnee (cont.curvesAt a gains) (Json.mkObj [("_", ← get c "curves")])
    compareCurves s!"{tag}curves: trcAt" tolSingle (cont.trcAt a gains) (Json.mkObj [("_", ← get c "trc")])
  -- The table, against the C++'s at every fourth node and against the bundle's print ICC.
  let mApd := (← b.dm "apd").m
  let lut ← checkBundleIcc root b j tag m pApd paper
  let lutTable := lut.table
  if let .ok nodes := (j.getObjVal? "table_nodes") then
    let table := printTable m pApd mApd paper
    for nd in ← arr nodes do
      let v ← nums nd
      let o := ((v.getD 0 0.0).toUInt64.toNat * 45 * 45 + (v.getD 1 0.0).toUInt64.toNat * 45 +
        (v.getD 2 0.0).toUInt64.toNat) * 3
      near s!"{tag}table: printTable" tolDouble (table.getD o 0.0) (v.getD 3 0.0)
      near s!"{tag}table: printTable" tolDouble (table.getD (o + 1) 0.0) (v.getD 4 0.0)
      near s!"{tag}table: printTable" tolDouble (table.getD (o + 2) 0.0) (v.getD 5 0.0)
  -- The print's white and the pixels through the print profile.
  if let .ok pw := (j.getObjVal? "paper_white") then
    let clut : Array Float := lutTable.map fun c => c.toFloat / 65535.0
    let base := paperBase clut
    near3 s!"{tag}white: paperBase" tolSingle base (← numsAt pw "paper_xyz")
    let rel := toPaperWhite clut
    for e in ← arrAt pw "clut" do
      let v ← nums e
      near s!"{tag}white: toPaperWhite" tolSingle (rel.getD (v.getD 0 0.0).toUInt64.toNat 0.0) (v.getD 1 0.0)
    let scaled := scaleIccCodes (paperWhiteGains base) lutTable
    for e in ← arrAt pw "scaled_codes" do
      let v ← nums e
      same s!"{tag}white: scaleIccCodes" (scaled.getD (v.getD 0 0.0).toUInt64.toNat 0) (v.getD 1 0.0).toUInt64.toNat
    -- How far the scaled table's X and Z pass the PCS white's (no comparison: a measurement).
    let mut over := 0
    let mut worst := 0.0
    for n in [0:rel.size / 3] do
      let rx := rel.getD (3 * n) 0.0 / pcsWhite.x * pcsMax
      let rz := rel.getD (3 * n + 2) 0.0 / pcsWhite.z * pcsMax
      if rx > 1.0 + 1e-9 || rz > 1.0 + 1e-9 then over := over + 1
      worst := max worst (max rx rz)
    IO.println s!"  note [{tag}white]: {over} nodes with X or Z above the PCS white's after the scaling, at most {worst}"
    let px ← get j "pixels"
    let merged ← matAt px "merged"
    let pins ← (← arrAt px "in").mapM fun p => do
      let a ← liftM ((← arr p).mapM nat : IO (Array Nat))
      pure (⟨a.getD 0 0, a.getD 1 0, a.getD 2 0⟩ : Vec3 Nat)
    let outC := lut.outCodes
    let outCurve (c : Nat) : Array Float := (outC.getD c #[]).map fun v => v.toFloat / 65535.0
    for c in ← arrAt px "cases" do
      let trc := cont.trcAt (← numAt c "a") (← gainsOf c)
      let p : Profile Float := { inCurves := trc, matrix := Mat3.identity, offset := ⟨0.0, 0.0, 0.0⟩,
                                 dim := ⟨45, 45, 45⟩, clut := rel, outCurves := ⟨outCurve 0, outCurve 1, outCurve 2⟩ }
      let mat := candidateMatrix merged (← numAt c "e") 1.0 1.0
      let outs ← arrAt c "pcs"
      let disp ← arrAt c "display"
      let labs ← arrAt c "lab"
      -- The print as the conversion shows and scores it: no shoulder, no tone stage.
      let cfg : PipelineConfig Float := { profile := p, knee := autoKnee, shoulder := Shoulder.off, tone := none }
      for i in [0:pins.size] do
        let v := loadU16 (pins.getD i ⟨0, 0, 0⟩)
        near3 s!"{tag}pixel: convertPixelPcs through the print profile" tolSingle
          (convertPixelPcs p autoKnee mat v) (← nums (outs.getD i Json.null))
        near3 s!"{tag}pixel: convertPixelDisplay through the print profile" tolSingle
          (convertPixelDisplay cfg mat v) (← nums (disp.getD i Json.null))
        let lab := pixelLab cfg mat v
        let want ← nums (labs.getD i Json.null)
        near s!"{tag}pixel: pixelLab through the print profile" tolLabPixel lab.L (want.getD 0 0.0)
        near s!"{tag}pixel: pixelLab through the print profile" tolLabPixel lab.a (want.getD 1 0.0)
        near s!"{tag}pixel: pixelLab through the print profile" tolLabPixel lab.b (want.getD 2 0.0)

def checkDensity (b : Bundle) (j : Json) : CheckM Unit := do
  let merged ← matAt j "merged"
  let n0 ← numAt j "base_level"
  let pins ← (← arrAt j "in").mapM fun p => do
    let a ← liftM ((← arr p).mapM nat : IO (Array Nat))
    pure (⟨a.getD 0 0, a.getD 1 0, a.getD 2 0⟩ : Vec3 Nat)
  for c in ← arrAt j "encode" do
    let key ← strAt c "measure"
    let e : DensityEncoding Float := densityEncoding (← boolAt c "dpx") (key == "apd")
    same "density: densityEncoding" e.bits (← natAt c "bits")
    same "density: densityEncoding" e.offset (← natAt c "offset")
    same "density: densityEncoding" e.scale (← natAt c "scale")
    near3 "density: densityEncoding" 0.0 e.k (← numsAt c "k")
    let m := (← b.dm key).m
    let codes ← natsAt c "codes"
    let mut floor : Vec3 Nat := ⟨0, 0, 0⟩
    let mut low : Vec3 Nat := ⟨0, 0, 0⟩
    let mut high : Vec3 Nat := ⟨0, 0, 0⟩
    for i in [0:pins.size] do
      let (cs, fl) := densityEncodePixel merged n0 m e (pins.getD i ⟨0, 0, 0⟩)
      same "density: densityEncodePixel (codes)" cs.x.1 (codes.getD (3 * i) 0)
      same "density: densityEncodePixel (codes)" cs.y.1 (codes.getD (3 * i + 1) 0)
      same "density: densityEncodePixel (codes)" cs.z.1 (codes.getD (3 * i + 2) 0)
      let n (x : Bool) : Nat := if x then 1 else 0
      floor := ⟨floor.x + n fl.x, floor.y + n fl.y, floor.z + n fl.z⟩
      low := ⟨low.x + n cs.x.2.1, low.y + n cs.y.2.1, low.z + n cs.z.2.1⟩
      high := ⟨high.x + n cs.x.2.2, high.y + n cs.y.2.2, high.z + n cs.z.2.2⟩
    for (name, got) in [("floor", floor), ("low", low), ("high", high)] do
      let want ← natsAt c name
      same s!"density: counts ({name})" got.x (want.getD 0 0)
      same s!"density: counts ({name})" got.y (want.getD 1 0)
      same s!"density: counts ({name})" got.z (want.getD 2 0)
  for c in ← arrAt j "balance" do
    let m := (← b.dm (← strAt c "measure")).m
    let some (off, mean) := densityBalanceOffset m 1.3 (← numAt c "g") (← numAt c "b") | fail "no balance"
    near3 "density: densityBalanceOffset" tolDouble off (← numsAt c "offset")
    near "density: densityBalanceOffset" tolDouble mean (← numAt c "mean")

def checkClf (root : System.FilePath) (b : Bundle) (j : Json) (mats : Array Json) : CheckM Unit := do
  let id ← strAt j "id"
  let some mj ← mats.findM? (fun m => do pure ((← strAt m "id") == id)) | fail s!"no material {id}"
  let prof ← readSpektrafilm (root / (← strAt mj "profile"))
  let some m := loadMaterial prof | fail "the material does not load"
  let nodes ← get j "nodes"
  let em ← numsAt nodes "exposure_matrix"
  let off := clfExposureOffsets m
  for (i, o) in [(0, off.x), (1, off.y), (2, off.z)] do
    for k in [0:3] do
      near "clf nodes: exposure matrix (−I)" 0.0 (if k = i then -1.0 else 0.0) (em.getD (4 * i + k) 0.0)
    nearRel "clf nodes: clfExposureOffsets" relG10 o (em.getD (4 * i + 3) 0.0)
  let range ← numsAt nodes "range"
  nearRel "clf nodes: Range" relG10 (m.logExposure.getD 0 0.0) (range.getD 0 0.0)
  nearRel "clf nodes: Range" relG10 (m.logExposure.getD (m.logExposure.size - 1) 0.0) (range.getD 1 0.0)
  nearRel "clf nodes: clfTop (scale 1 / top)" relG10 (1.0 / clfTop m) (← numAt nodes "scale")
  let n1 ← natAt nodes "lut1d_points"
  for r in ← arrAt nodes "lut1d" do
    let v ← nums r
    let got := clfCurveRow m n1 (v.getD 0 0.0).toUInt64.toNat
    nearRel "clf nodes: clfCurveRow (LUT1D)" relG6 got.x (v.getD 1 0.0)
    nearRel "clf nodes: clfCurveRow (LUT1D)" relG6 got.y (v.getD 2 0.0)
    nearRel "clf nodes: clfCurveRow (LUT1D)" relG6 got.z (v.getD 3 0.0)
  let n3 ← natAt nodes "cube_points"
  let toSrgb := toSrgbMatrix m.white
  for r in ← arrAt nodes "lut3d" do
    let v ← nums r
    let got := clfCubeNode m toSrgb n3 (v.getD 0 0.0).toUInt64.toNat
    nearRel "clf nodes: clfCubeNode (LUT3D)" relG6 got.x (v.getD 1 0.0)
    nearRel "clf nodes: clfCubeNode (LUT3D)" relG6 got.y (v.getD 2 0.0)
    nearRel "clf nodes: clfCubeNode (LUT3D)" relG6 got.z (v.getD 3 0.0)
  -- The frame's first matrix.
  let mApd := (← b.dm "apd").m
  let some pApd := materialPrintMatrix prof b.film (← b.dm "apd") | fail "no APD print matrix"
  let cont : PrintContinuum Float :=
    { grey := b.grey, paper := paperCurve m pApd, n0 := b.n0, aMid := b.aMid, tilt := b.tilt }
  for f in ← arrAt j "frames" do
    let key ← strAt f "measure"
    let dm ← b.dm key
    let some p := materialPrintMatrix prof b.film dm | fail s!"no print matrix {key}"
    nearMat "clf frame: print matrix" tolDouble p (← numsAt f "to_print")
    let own ← boolAt f "own_search"
    let src : FrameGrey Float ←
      if own then pure (.ownSearch (← numAt f "a"))
      else do
        let name ← strAt f "bracket"
        let some k := b.bracketNames.findIdx? (· == name) | fail s!"no bracket {name}"
        pure (.bracketCurve (b.aMid.getD k 0.0) (b.tilt.getD k ⟨0, 0, 0⟩))
    let e ← numAt f "search_e"
    let fp := framePrint dm.m mApd p cont src e
    near3 "clf frame: framePrint (grey)" tolDouble fp.grey (← numsAt f "grey")
    near3 "clf frame: framePrint (slopes)" tolDouble fp.slopes (← numsAt f "slopes")
    -- The balance: the material's own search's gains, else the picture search's (`print_balance`).
    let (be, bg, bb) ← if own then pure (e, ← numAt f "search_g", ← numAt f "search_b")
      else pure (← numAt f "balance_e", ← numAt f "balance_g", ← numAt f "balance_b")
    let bal := printBalance dm.m be bg bb
    let enc : DensityEncoding Float := densityEncoding (← boolAt f "dpx") (key == "apd")
    let (a, off) := frameClfMatrix enc fp bal
    let want ← numsAt f "matrix"
    for i in [0:3] do
      let row := if i = 0 then a.row0 else if i = 1 then a.row1 else a.row2
      let o := if i = 0 then off.x else if i = 1 then off.y else off.z
      nearRel "clf frame: frameClfMatrix" relG10 row.x (want.getD (4 * i) 0.0)
      nearRel "clf frame: frameClfMatrix" relG10 row.y (want.getD (4 * i + 1) 0.0)
      nearRel "clf frame: frameClfMatrix" relG10 row.z (want.getD (4 * i + 2) 0.0)
      nearRel "clf frame: frameClfMatrix" relG10 o (want.getD (4 * i + 3) 0.0)

def checkFloor (b : Bundle) (j : Json) : CheckM Unit := do
  let w ← natAt j "width"
  let h ← natAt j "height"
  let px ← natsAt j "pixels"
  let pixel (x y : Nat) : Vec3 Nat :=
    let o := (y * w + x) * 3
    ⟨px.getD o 0, px.getD (o + 1) 0, px.getD (o + 2) 0⟩
  for c in ← arrAt j "cases" do
    let crop := innerCrop w h (← numAt c "crop_ratio")
    let want ← natsAt c "crop"
    same "floor: innerCrop" crop.1 (want.getD 0 0)
    same "floor: innerCrop" crop.2.1 (want.getD 1 0)
    same "floor: innerCrop" crop.2.2.1 (want.getD 2 0)
    same "floor: innerCrop" crop.2.2.2 (want.getD 3 0)
    near "floor: printFloorFactor" tolSingle (printFloorFactor b.crosstalk (← vec3At c "base") pixel crop)
      (← numAt c "factor")
  for c in ← arrAt j "raise" do
    let found ← match ← get c "found" with
      | .null => pure none
      | f => do pure (some (vec3 (← nums f)))
    let base := baseBeforeFloor found b.filmBase (← numAt c "scan_shutter") (← numAt c "scan_iso")
      (← numAt c "film_base_shutter") (← numAt c "film_base_iso")
    near3 "floor: baseBeforeFloor" tolDouble base (← numsAt c "base_in")
    let beta := printFloorFactor b.crosstalk base pixel (innerCrop w h 1.0)
    near "floor: printFloorFactor (raise)" tolSingle beta (← numAt c "factor")
    -- The C++ multiplies by the factor as a float: the comparison is relative to the base.
    let want ← numsAt c "base_out"
    let got := raisedBase base (← numAt c "factor")
    nearRel "floor: raisedBase" tolDouble got.x (want.getD 0 0.0)
    nearRel "floor: raisedBase" tolDouble got.y (want.getD 1 0.0)
    nearRel "floor: raisedBase" tolDouble got.z (want.getD 2 0.0)

def run (root : System.FilePath) : CheckM Unit := do
  let j ← readJson "testvectors/print.json"
  let b ← readBundle (← get j "bundle")
  checkGrey b (← get j "grey") ""
  let mats ← arrAt j "materials"
  for m in mats do checkMaterial root b m ""
  checkDensity b (← get j "density")
  checkClf root b (← get j "clf") mats
  checkFloor b (← get j "floor")
  if let .ok wl := j.getObjVal? "white_light" then
    let wb ← readBundle (← get wl "bundle")
    checkGrey wb (← get wl "grey") "white light: "
    for m in ← arrAt wl "materials" do checkMaterial root wb m "white light: "

end CheckPrint

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
  let (_, groups) ← (CheckPrint.run root).run #[]
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
