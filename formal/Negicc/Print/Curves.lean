/-
The print ICC: its input curves at a mid grey and its colour table.
-/
module

public import Negicc.Print.Grey
public import Negicc.Print.Material

@[expose] public section

/-!
# The print ICC

The print is written as an ICC profile shaped as the chart profiles are (DESIGN.md 5.4): three
4096-entry input curves, a 45³ table, identity output curves. For the profile's input `x` (after
the soft knee) in channel `c`:

```
dL_c             = −log10(unknee(x_c) g_c / n0)                       light density
alpha_c          = (G_c(dL_c) − A) (1 + t_c(A))                       APD above the print's mid grey
in_c(x)          = Paper_c(alpha_c)                                   the input curve
Paper_c(alpha_i) = 0.75 curve_c(aim_c − η_c alpha_i) / top_c + 0.25 (1 − i / (N − 1))
table(a)         = the print of 0.70 + Q L (ψ ∘ alpha),  alpha_c = Paper_c⁻¹(a_c),  ψ = M⁻¹ (1, 1, 1)
```

`paperCurve` is `Paper_c` (`paper_curve`), `bracketPrintCurves` the input curves
(`bracket_print_curves`), `curvesAt` and `trcAt` the curves at a continuous `A` with the brackets'
tilt at `A` (`PrintContinuum::curves_at`, `trc_at`), and `printTable` the table (`print_table`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## Paper_c -/

/-- The first point of `Paper_c`'s grid of grey APD above the mid grey (`kPaperAlphaStart`). -/
def paperAlphaStart : α := -1.5

/-- The step of `Paper_c`'s grid (`kPaperAlphaStep`). -/
def paperAlphaStep : α := 0.005

/-- The number of points of `Paper_c`'s grid, −1.5 to 1.8 (`kPaperAlphaPoints`). -/
def paperAlphaPoints : Nat := 661

/-- The grey APD above the mid grey at point `i` of `Paper_c`'s grid: `−1.5 + 0.005 i`. -/
def paperAlpha (i : Nat) : α := paperAlphaStart + paperAlphaStep * (i : α)

/-- The running minimum of a list from the current minimum `r` (`none` before the first entry):
entry `i` is the least of the first `i + 1` entries and `r`. -/
def runMinList : Option α → List α → List α
  | _, [] => []
  | r, v :: vs =>
    let m := match r with | none => v | some r => min r v
    m :: runMinList (some m) vs

/-- The running maximum of a list from the current maximum `r`, as `runMinList`. -/
def runMaxList : Option α → List α → List α
  | _, [] => []
  | r, v :: vs =>
    let m := match r with | none => v | some r => max r v
    m :: runMaxList (some m) vs

/-- The running minimum: entry `i` is the least of the first `i + 1` entries (the C++ `run`
starting from `INFINITY`). -/
def runningMin (xs : Array α) : Array α := (runMinList none xs.toList).toArray

/-- The running maximum: entry `i` is the greatest of the first `i + 1` entries (the C++ `run`
starting from `-INFINITY`). -/
def runningMax (xs : Array α) : Array α := (runMaxList none xs.toList).toArray

/-- `η = P (1, 1, 1)`: the print density the print material reads per unit of grey APD, `P` the
print matrix for APD. -/
def paperEta (pApd : Mat3 α) : Vec3 α := pApd.rowSums

/-- One channel of `Paper_c` from the dye amounts `δ_i` the layer takes on the greys `alpha_i`:
their running minimum (held at zero from below) scaled by its first value `top`, times 0.75, plus
the ramp `0.25 (1 − i / (N − 1))`. -/
def paperChannel (dye : Array α) : Array α :=
  let run := runningMin (dye.map (max 0.0))
  let top := run.getD 0 0.0
  (Array.range run.size).map fun (i : Nat) =>
    0.75 * (run.getD i 0.0 / top) + 0.25 * (1.0 - (i : α) / ((paperAlphaPoints - 1 : Nat) : α))

/-- `Paper_c` on the grid `alpha_i` (`paper_curve`): the dye amount layer `c` takes on a grey
`alpha_i` above the mid grey, `curve_c(aim_c − η_c alpha_i)`, made non-increasing and scaled to
`[0, 1]` with a 0.25 linear ramp, so that it is strictly decreasing. -/
def paperCurve (m : PrintMaterial α) (pApd : Mat3 α) : Vec3 (Array α) :=
  let eta := paperEta pApd
  let dye (curve : Array α) (aim e : α) : Array α :=
    (Array.range paperAlphaPoints).map fun (i : Nat) => interp m.logExposure curve (aim - e * paperAlpha i)
  ⟨paperChannel (dye m.curves.x m.aim.x eta.x), paperChannel (dye m.curves.y m.aim.y eta.y),
   paperChannel (dye m.curves.z m.aim.z eta.z)⟩

/-! ## The input curves -/

/-- The number of entries of each input curve (`kPrintIccPoints`). -/
def printIccPoints : Nat := 4096

/-- The soft knee of the conversion, which the print's curves undo (`kPrintKnee`, the conversion's
`kKnee`; the C++ constant is the single precision `0.95f`). -/
def printKnee : α := 0.95

/-- The inverse of the soft knee: the identity up to `knee`, then
`knee + (1 − knee) atanh((x − knee) / (1 − knee))`, the argument of `atanh` held to `[0, 0.998]`
(`bracket_print_curves`). -/
def unknee (knee x : α) : α :=
  if x ≤ knee then x else knee + (1.0 - knee) * atanh (min (max ((x - knee) / (1.0 - knee)) 0.0) 0.998)

/-- The light density of the profile's input `x` in a channel with gain `gain`:
`−log10(max(unknee(x) gain, 1e-7) / n0)`. -/
def inputLightDensity (n0 gain x : α) : α := -(log10 (max (unknee printKnee x * gain) 1.0e-7 / n0))

/-- The grey APD above the print's mid grey `a` of a light density `dl` in a channel:
`(G_c(dl) − a) (1 + t_c)`. -/
def greyAbove (g : GreyBalance α) (apd : Array α) (a tilt dl : α) : α :=
  (g.eval apd dl - a) * (1.0 + tilt)

/-- One input curve before it is made non-decreasing: `Paper_c(alpha_c)` at
`x = i / 4095`. -/
def inputCurveRaw (g : GreyBalance α) (apd paper : Array α) (a n0 tilt gain : α) : Array α :=
  (Array.range printIccPoints).map fun (i : Nat) =>
    let x := (i : α) / ((printIccPoints - 1 : Nat) : α)
    onGrid paper paperAlphaStart paperAlphaStep (greyAbove g apd a tilt (inputLightDensity n0 gain x))

/-- A curve held non-decreasing (its running maximum) and made strictly increasing by `1e-9 i`. -/
def rampedRunMax (xs : Array α) : Array α :=
  let run := runningMax xs
  (Array.range run.size).map fun (i : Nat) => run.getD i 0.0 + 1.0e-9 * (i : α)

/-- A bracket's (or a mid grey's) three input curves on `x = i / 4095`, held non-decreasing (a
running maximum) and made strictly increasing by `1e-9 i` (`bracket_print_curves`). `a` is the
mid grey `A`, `tilt` the tilt `t` and `gains` the gains `(1, g, b)` after the knee (`(1, 1, 1)`
for none). -/
def bracketPrintCurves (g : GreyBalance α) (paper : Vec3 (Array α)) (a n0 : α) (tilt gains : Vec3 α) :
    Vec3 (Array α) :=
  let curve (apd p : Array α) (t gain : α) : Array α := rampedRunMax (inputCurveRaw g apd p a n0 t gain)
  ⟨curve g.apd.x paper.x tilt.x gains.x, curve g.apd.y paper.y tilt.y gains.y,
   curve g.apd.z paper.z tilt.z gains.z⟩

/-- The print at a continuous mid grey `A` (`PrintContinuum`): the grey balance curve, `Paper_c`,
`n0` and each bracket's `A_mid` and tilt. -/
structure PrintContinuum (α : Type) where
  grey : GreyBalance α
  paper : Vec3 (Array α)
  n0 : α
  aMid : Array α
  tilt : Array (Vec3 α)

/-- The input curves at mid grey `a` with gains `gains` (`PrintContinuum::curves_at`): the brackets'
tilt at `a`, or none without brackets. -/
def PrintContinuum.curvesAt (pc : PrintContinuum α) (a : α) (gains : Vec3 α) : Vec3 (Array α) :=
  let t := if pc.aMid.size = 0 then ⟨0.0, 0.0, 0.0⟩ else printTiltAt pc.aMid pc.tilt a
  bracketPrintCurves pc.grey pc.paper a pc.n0 t gains

/-- A curve value as the 16-bit code the print ICC writes (`put_u16`): `round(clamp01(v) 65535)`. -/
def iccCode16 (v : α) : Nat := roundNat (clamp01 v * 65535.0)

/-- A curve value as the 16-bit code an ICC holds, read back as `code / 65535`. -/
def icc16 (v : α) : α := (iccCode16 v : α) / 65535.0

/-- The input tables the conversion renders the print with at mid grey `a` and gains `gains`
(`PrintContinuum::trc_at`, `set_curves_at`): the curves as the 16-bit codes an ICC holds. -/
def PrintContinuum.trcAt (pc : PrintContinuum α) (a : α) (gains : Vec3 α) : Vec3 (Array α) :=
  (pc.curvesAt a gains).map (·.map icc16)

/-- The range `A` is searched in: the green curve's APD range (`a_lo`, `a_hi`). -/
def PrintContinuum.aLo (pc : PrintContinuum α) : α :=
  pc.grey.apd.y.foldl min (pc.grey.apd.y.getD 0 0.0)

@[inherit_doc PrintContinuum.aLo]
def PrintContinuum.aHi (pc : PrintContinuum α) : α :=
  pc.grey.apd.y.foldl max (pc.grey.apd.y.getD 0 0.0)

/-! ## The table -/

/-- The number of nodes along each axis of the print ICC's table (`kPrintIccCube`). -/
def printIccCube : Nat := 45

/-- The grey APD of each node index of one axis: `Paper_c` inverted at `i / 44`, read on the
reversed table because `Paper_c` decreases. -/
def nodeAlpha (paper : Array α) : Array α :=
  let n := paperAlphaPoints
  let grid : Array α := (Array.range n).map fun (j : Nat) => paperAlpha (n - 1 - j)
  (Array.range printIccCube).map fun (i : Nat) =>
    interp paper.reverse grid ((i : α) / ((printIccCube - 1 : Nat) : α))

/-- The table's value at the grey APD `alpha` above the mid grey in each channel: the printing
density `0.70 + Q L (ψ ∘ alpha)` with `Q L = P M` and `ψ = M⁻¹ (1, 1, 1)`, its print, adapted to
the PCS (`print_table`, one node). -/
def tableNode (m : PrintMaterial α) (ql pcs : Mat3 α) (psi alpha : Vec3 α) : Vec3 α :=
  let wa : Vec3 α := ⟨alpha.x * psi.x, alpha.y * psi.y, alpha.z * psi.z⟩
  let pd : Vec3 α := ⟨printGreyDensity + ql.row0.dot wa, printGreyDensity + ql.row1.dot wa,
    printGreyDensity + ql.row2.dot wa⟩
  pcs.mulVec (printOfDensity m pd)

/-- The print ICC's table: `45³` PCS XYZ (D50), the red index slowest, then green, then blue,
three values a node (`print_table`). `pApd` is the print matrix for APD and `mApd` the APD matrix
`M`. -/
def printTable (m : PrintMaterial α) (pApd mApd : Mat3 α) (paper : Vec3 (Array α)) : Array α :=
  let ql := pApd.mul mApd
  let psi := (Mat3.adjInv mApd).rowSums
  let pcs := pcsAdaptation m.paper m.white
  let na : Vec3 (Array α) := paper.map nodeAlpha
  let n := printIccCube
  (Array.range (n * n * n)).flatMap fun idx =>
    let v := tableNode m ql pcs psi
      ⟨na.x.getD (idx / (n * n)) 0.0, na.y.getD (idx / n % n) 0.0, na.z.getD (idx % n) 0.0⟩
    #[v.x, v.y, v.z]

/-- The print ICC's table as the 16-bit codes it writes (`write_print_icc`): each XYZ over
`kPcsMax`, the largest a `lut16` holds, as `iccCode16`. -/
def printTableCodes (table : Array α) : Array Nat :=
  table.map fun v => iccCode16 (v / (1.0 + 32767.0 / 32768.0))

end Negicc
