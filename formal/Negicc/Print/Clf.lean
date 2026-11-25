/-
The CLF that prints a density file: the frame's matrix and the print material's nodes.
-/
module

public import Negicc.Print.Curves
public import Negicc.Print.Density

@[expose] public section

/-!
# The density file's print (CLF)

Beside a density file goes, for each print material, a CLF that renders the file as the print
(DESIGN.md 5.9). Its first node is a matrix for the frame (`frameClfMatrix`, the first `Matrix` of
`frame_print_clf`); the others are the material's own (`print_clf_nodes`): density to log exposure
(`clfExposureOffsets`), each layer's curve as a 4096-point LUT1D (`clfCurveRow`), a scale
(`clfTop`), dye amounts to linear sRGB as a 45-point LUT3D (`clfCubeNode`), the sRGB curve.

On the file's code `q` (code over the largest code) the first matrix computes

```
D             = (q × largest code − offset) / (scale κ)
print density = 0.70 + S P (D + balance − grey)
grey          = M (n + log10 E)              n: the light density of the curve's grey at green APD A
S_c           = (γ_green / γ_c) (1 + t_c(A)),   γ = M_APD (dn/dA)
```

`M` is the file's measure's matrix; `γ` is taken through the APD matrix whatever the measure
(`curveGamma`).

(`clfPrintingDensity_slope_form` in `Negicc.Print.Props`). `framePrint` computes `grey` and `S`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- A frame's print on one material (`FramePrint`): the print matrix `P` of the file's measure,
the frame's mid grey as the measure's density, and the slopes. -/
structure FramePrint (α : Type) where
  toPrint : Mat3 α
  grey : Vec3 α
  slopes : Vec3 α

/-- Where the frame's mid grey and slopes come from (`frame_print`):

* `ownSearch a`: the material's own print search found the mid grey `A = a`: the curve's grey and
  tangent at `a`, times one plus the brackets' tilt at `a`;
* `bracketCurve aMid tilt`: the bracket the picture's search chose, through the curve: its mid grey
  APD and its own tilt;
* `bracketOwn greyLight slopes`: a bundle without the curve: the bracket's own mid grey and slopes. -/
inductive FrameGrey (α : Type) where
  | ownSearch (a : α)
  | bracketCurve (aMid : α) (tilt : Vec3 α)
  | bracketOwn (greyLight slopes : Vec3 α)

/-- The frame's mid grey (as the measure's density, `M (n + log10 E)`) and slopes (`frame_print`).
`m` is the measure's matrix, `mApd` the APD matrix, `pc` the print continuum (curve and brackets)
and `exposure` the search's `E`. -/
def framePrint (m mApd toPrint : Mat3 α) (pc : PrintContinuum α) (src : FrameGrey α) (exposure : α) :
    FramePrint α :=
  let (light, slopes) : Vec3 α × Vec3 α := match src with
    | .ownSearch a =>
      let t := printTiltAt pc.aMid pc.tilt a
      let s := curveSlopesAt pc.grey mApd a
      (curveGreyAt pc.grey a, ⟨s.x * (1.0 + t.x), s.y * (1.0 + t.y), s.z * (1.0 + t.z)⟩)
    | .bracketCurve aMid t =>
      let s := curveSlopesAt pc.grey mApd aMid
      (curveGreyAt pc.grey aMid, ⟨s.x * (1.0 + t.x), s.y * (1.0 + t.y), s.z * (1.0 + t.z)⟩)
    | .bracketOwn g s => (g, s)
  let shift := log10 exposure
  { toPrint := toPrint, slopes := slopes,
    grey := m.mulVec ⟨light.x + shift, light.y + shift, light.z + shift⟩ }

/-- The frame's first CLF matrix, as a 3x3 `a` and an offset `b` (`frame_print_clf`): row `i` is
`p_ij = S_i P_ij` scaled by `largest code / (scale κ_j)`, and the offset
`0.70 + Σ_j p_ij (bal_j − grey_j − offset / (scale κ_j))`. -/
def frameClfMatrix (e : DensityEncoding α) (fp : FramePrint α) (bal : Vec3 α) : Mat3 α × Vec3 α :=
  let maxCode : α := (e.maxCode : α)
  let sk (j : Nat) : α := (e.scale : α) * (if j = 0 then e.k.x else if j = 1 then e.k.y else e.k.z)
  let comp (v : Vec3 α) (j : Nat) : α := if j = 0 then v.x else if j = 1 then v.y else v.z
  let p (i j : Nat) : α := comp fp.slopes i * fp.toPrint.entry i j
  let a (i j : Nat) : α := p i j * maxCode / sk j
  let term (i j : Nat) : α := p i j * (comp bal j - comp fp.grey j - (e.offset : α) / sk j)
  let b (i : Nat) : α := printGreyDensity + (0.0 + term i 0 + term i 1 + term i 2)
  (⟨⟨a 0 0, a 0 1, a 0 2⟩, ⟨a 1 0, a 1 1, a 1 2⟩, ⟨a 2 0, a 2 1, a 2 2⟩⟩, ⟨b 0, b 1, b 2⟩)

/-- The file's density of the code fraction `q` (code over the largest code):
`(q × largest code − offset) / (scale κ)`. -/
def codeDensity (e : DensityEncoding α) (q : Vec3 α) : Vec3 α :=
  let maxCode : α := (e.maxCode : α)
  ⟨(q.x * maxCode - (e.offset : α)) / ((e.scale : α) * e.k.x),
   (q.y * maxCode - (e.offset : α)) / ((e.scale : α) * e.k.y),
   (q.z * maxCode - (e.offset : α)) / ((e.scale : α) * e.k.z)⟩

/-- The printing density the CLF's first matrix gives the code fraction `q`. -/
def clfPrintingDensity (e : DensityEncoding α) (fp : FramePrint α) (bal q : Vec3 α) : Vec3 α :=
  let (a, b) := frameClfMatrix e fp bal
  a.mulVec q + b

/-! ## The print material's nodes -/

/-- The offsets of the material's first matrix, `log E = aim − (D − 0.70)`: `aim + 0.70` (the matrix
part is `−I`). -/
def clfExposureOffsets (m : PrintMaterial α) : Vec3 α :=
  ⟨m.aim.x + printGreyDensity, m.aim.y + printGreyDensity, m.aim.z + printGreyDensity⟩

/-- The smallest natural number not below `x ≥ 0` (C `ceil`). -/
def natCeil (x : α) : Nat := let n := natFloor x; if (n : α) < x then n + 1 else n

/-- The largest dye amount of the three curves rounded up to a tenth: the scale of the colour cube's
axes (`top`). -/
def clfTop (m : PrintMaterial α) : α :=
  let mx (xs : Array α) : α := xs.foldl max (xs.getD 0 0.0)
  let t := max (max (max 0.0 (mx m.curves.x)) (mx m.curves.y)) (mx m.curves.z)
  (natCeil (t * 10.0) : α) / 10.0

/-- Row `i` of the LUT1D of `n` points: each layer's dye amount at the log exposure
`lo + (hi − lo) i / (n − 1)`, `lo` and `hi` the ends of the curves' log exposures (the LUT1D's
`Range`). -/
def clfCurveRow (m : PrintMaterial α) (n i : Nat) : Vec3 α :=
  let lo := m.logExposure.getD 0 0.0
  let hi := m.logExposure.getD (m.logExposure.size - 1) 0.0
  let x := lo + (hi - lo) * (i : α) / ((n - 1 : Nat) : α)
  m.amounts ⟨x, x, x⟩

/-- Node `idx` of the LUT3D of `cube` points a side (blue changing fastest): the dye amounts
`top (r, g, b) / (cube − 1)`, the print's XYZ relative to the paper's white (each of X, Y, Z times
the light's over the paper's), to linear sRGB. -/
def clfCubeNode (m : PrintMaterial α) (toSrgb : Mat3 α) (cube idx : Nat) : Vec3 α :=
  let top := clfTop m
  let paper := paperXyz m
  let rel : Vec3 α := ⟨m.white.x / paper.x, m.white.y / paper.y, m.white.z / paper.z⟩
  let s (k : Nat) : α := top * (k : α) / ((cube - 1 : Nat) : α)
  let xyz := amountsToXyz m ⟨s (idx / (cube * cube)), s (idx / cube % cube), s (idx % cube)⟩
  toSrgb.mulVec ⟨xyz.x * rel.x, xyz.y * rel.y, xyz.z * rel.z⟩

end Negicc
