/-
A print material loaded for a film: the observer, aim, the print matrix and the colour matrices.
-/
module

public import Negicc.Print.Spectral
public import Negicc.Print.Matrix

@[expose] public section

/-!
# A print material for a film

`load_spektrafilm_print` (`np_print.cc`) reads a spektrafilm print profile and computes, for the
film of a profile bundle (DESIGN.md 5.3):

* the observer under the viewing light (`observerWeights`) and its white;
* `aim`, each layer's log exposure that prints an 18 % grey (`aimSolve`, Newton's method);
* `Q`, the 3x3 from the film's dye amounts to the print's own density, a least-squares fit with
  no offset on the film model's neutral scale with 27 colour variations per point
  (`filmToPrintDensity`);
* the print matrix of each measure, `P = Q F⁻¹ + Q L diag(φ⁻¹ − 1) L⁻¹ F⁻¹` (`printMatrix`), which
  is `Q L M⁻¹` for `M = F L diag(φ)` (`printMatrix_mul_m` in `Negicc.Print.Props`);
* the matrix from XYZ under the viewing light to linear sRGB, adapted to D65 with CAT02, used by
  the print CLF (`toSrgbMatrix`).

It also holds the Bradford adaptation the print ICC applies to a print film's D55 (`bradfordToD50`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The profile bundle's film -/

/-- The film stock's spectra (`FilmModel` in `profile_bundle.h`): the spectral density of a unit
amount of each dye and of the base on the spectral grid, and the dye amounts along the film's
neutral scale. -/
structure FilmModel (α : Type) where
  dye : Vec3 (Array α)
  base : Array α
  logExposure : Array α
  neutral : Vec3 (Array α)

/-- A density measure's matrices (`DensityMatrix` in `profile_bundle.h`): `M = F L diag(φ)` from
the light's densities to the measure, `F` from the film's dye amounts to the measure, `L` from the
light's densities to the dye amounts, and the light density factors `φ` (1 unless Status M was
calibrated on the chart). -/
structure DensityMatrices (α : Type) where
  m : Mat3 α
  filmToMeasure : Mat3 α
  lightToFilm : Mat3 α
  factors : Vec3 α

/-! ## The print's own density: `Q` -/

/-- The film's transmittance at sample `w` for dye amounts `a`. -/
def FilmModel.transmittance (f : FilmModel α) (a : Vec3 α) (w : Nat) : α :=
  pow 10.0 (-(a.x * f.dye.x.getD w 0.0 + a.y * f.dye.y.getD w 0.0 + a.z * f.dye.z.getD w 0.0 +
    f.base.getD w 0.0))

/-- The 27 colour variations of a neutral point: `z` in `{−1, 0, 1}³`, the first component varying
slowest (`v / 9 − 1`, `v / 3 % 3 − 1`, `v % 3 − 1` for `v < 27`). -/
def colourVariation (v : Nat) : Vec3 α :=
  let z (k : Nat) : α := if k = 0 then -1.0 else if k = 1 then 0.0 else 1.0
  ⟨z (v / 9), z (v / 3 % 3), z (v % 3)⟩

/-- The film model's sample colours the fit runs on: each point of its neutral scale with log
exposure in `[−1.5, 1.0]`, each dye amount times `exp(0.15 √1.5 z)` for the 27 variations `z`,
held at zero from below. -/
def FilmModel.samples (f : FilmModel α) : List (Vec3 α) :=
  let spread : α := 0.15 * sqrt 1.5
  (List.range f.logExposure.size).flatMap fun i =>
    let e := f.logExposure.getD i 0.0
    if e < -1.5 ∨ 1.0 < e then []
    else (List.range 27).map fun v =>
      let z : Vec3 α := colourVariation v
      ⟨max 0.0 (f.neutral.x.getD i 0.0 * exp (spread * z.x)),
       max 0.0 (f.neutral.y.getD i 0.0 * exp (spread * z.y)),
       max 0.0 (f.neutral.z.getD i 0.0 * exp (spread * z.z))⟩

/-- `Q`: the 3x3 from a film's dye amounts to the density above the film base that `response`
reads, least squares with no offset on the film model's sample colours
(`film_to_print_density`). The normal equations `(Σ a aᵀ) X = Σ a dᵀ` are solved with the
adjugate inverse and `Q = Xᵀ`. `none` when a response has no weight or the normal matrix is
singular. -/
def filmToPrintDensity (f : FilmModel α) (resp : Vec3 (Array α)) : Option (Mat3 α) :=
  let total (r : Array α) : α := sumRange cieSamples fun w => r.getD w 0.0
  let s : Vec3 α := resp.map total
  if ¬ (0.0 < s.x ∧ 0.0 < s.y ∧ 0.0 < s.z) then none
  else
    let reads (r : Array α) (sr : α) (t : Nat → α) : α :=
      -(log10 ((sumRange cieSamples fun w => t w * r.getD w 0.0) / sr))
    let baseT : Nat → α := fun w => pow 10.0 (-(f.base.getD w 0.0))
    let baseD : Vec3 α := ⟨reads resp.x s.x baseT, reads resp.y s.y baseT, reads resp.z s.z baseT⟩
    let sums := f.samples.foldl (fun (acc : Mat3 α × Mat3 α) a =>
      let t := f.transmittance a
      let d : Vec3 α := ⟨reads resp.x s.x t - baseD.x, reads resp.y s.y t - baseD.y,
        reads resp.z s.z t - baseD.z⟩
      (acc.1.add (Mat3.outer a a), acc.2.add (Mat3.outer a d))) (Mat3.zero, Mat3.zero)
    (Mat3.invert3 sums.1).map fun inv => (inv.mul sums.2).transpose

/-! ## The print matrix -/

/-- Whether each of the three factors is exactly 1 (a measure without light density factors). -/
def allOne (v : Vec3 α) : Bool :=
  decide (1.0 ≤ v.x ∧ v.x ≤ 1.0 ∧ 1.0 ≤ v.y ∧ v.y ≤ 1.0 ∧ 1.0 ≤ v.z ∧ v.z ≤ 1.0)

/-- The print matrix of a measure, from its density to the density the print material reads:
`Q F⁻¹`, plus, when the measure has light density factors `φ`, `(Q L) (diag(φ⁻¹ − 1) (L⁻¹ F⁻¹))`
(`print_matrix` in `load_spektrafilm_print`). `none` when `F` or `L` is singular. -/
def printMatrix (q : Mat3 α) (dm : DensityMatrices α) : Option (Mat3 α) :=
  match Mat3.invert3 dm.filmToMeasure with
  | none => none
  | some invF =>
    let p := q.mul invF
    if allOne dm.factors then some p
    else
      match Mat3.invert3 dm.lightToFilm with
      | none => none
      | some invL =>
        let corr := (q.mul dm.lightToFilm).mul
          (Mat3.scaleRows (dm.factors.map fun a => 1.0 / a - 1.0) (invL.mul invF))
        some (p.add corr)

/-! ## The colour matrices -/

/-- The CIECAM02 chromatic adaptation matrix (CIE 159:2004), `cat02` in `load_spektrafilm_print`. -/
def cat02 : Mat3 α :=
  ⟨⟨0.7328, 0.4296, -0.1624⟩, ⟨-0.7036, 1.6975, 0.0061⟩, ⟨0.0030, 0.0136, 0.9834⟩⟩

/-- XYZ (D65) to linear sRGB as Bruce Lindbloom tabulates it, `xyz_to_srgb` in
`load_spektrafilm_print`. -/
def xyzToSrgbLindbloom : Mat3 α :=
  ⟨⟨3.2404542, -1.5371385, -0.4985314⟩, ⟨-0.9692660, 1.8760108, 0.0415560⟩,
   ⟨0.0556434, -0.2040259, 1.0572252⟩⟩

/-- Each entry `m_rc` multiplied by `num_r` and divided by `den_r`, in that order: the von Kries
gains of a chromatic adaptation applied to the rows of `m`. -/
def Mat3.scaleRowsBy (m : Mat3 α) (num den : Vec3 α) : Mat3 α :=
  ⟨⟨m.row0.x * num.x / den.x, m.row0.y * num.x / den.x, m.row0.z * num.x / den.x⟩,
   ⟨m.row1.x * num.y / den.y, m.row1.y * num.y / den.y, m.row1.z * num.y / den.y⟩,
   ⟨m.row2.x * num.z / den.z, m.row2.y * num.z / den.z, m.row2.z * num.z / den.z⟩⟩

/-- XYZ under the viewing light (white `white`) to linear sRGB, adapted to D65 with CAT02: D65 is
the XYZ the sRGB matrix takes to `(1, 1, 1)`, so the light's white lands on sRGB white
(`PrintMaterial::to_srgb`; `toSrgbMatrix_white` in `Negicc.Print.Props`). -/
def toSrgbMatrix (white : Vec3 α) : Mat3 α :=
  let d65 := (Mat3.adjInv xyzToSrgbLindbloom).rowSums
  let scaled := Mat3.scaleRowsBy cat02 (cat02.mulVec d65) (cat02.mulVec white)
  xyzToSrgbLindbloom.mul ((Mat3.adjInv cat02).mul scaled)

/-- The linearised Bradford matrix of ICC.1:2022 Annex E.3 (equation E.1), `B` in `print_table`. -/
def bradford : Mat3 α :=
  ⟨⟨0.8951, 0.2664, -0.1614⟩, ⟨-0.7502, 1.7135, 0.0367⟩, ⟨0.0389, -0.0685, 1.0296⟩⟩

/-- D50 to four decimals, the PCS illuminant (`kD50` in `np_print.cc`). -/
def iccD50 : Vec3 α := ⟨0.9642, 1.0, 0.8249⟩

/-- The adaptation of the print's XYZ to the PCS: the identity for a paper (D50), and for a print
film `B⁻¹ diag(gain) B` from the light's white (normalised to `Y = 1`) to D50, with
`gain_r = (B D50)_r / (B white)_r` (`pcs` in `print_table`). -/
def pcsAdaptation (paper : Bool) (white : Vec3 α) : Mat3 α :=
  if paper then Mat3.identity
  else
    let src : Vec3 α := ⟨white.x / white.y, 1.0, white.z / white.y⟩
    let to := bradford.mulVec iccD50
    let fr := bradford.mulVec src
    let gain : Vec3 α := ⟨to.x / fr.x, to.y / fr.y, to.z / fr.z⟩
    let bi := Mat3.adjInv (bradford : Mat3 α)
    let biG : Mat3 α :=
      ⟨⟨bi.row0.x * gain.x, bi.row0.y * gain.y, bi.row0.z * gain.z⟩,
       ⟨bi.row1.x * gain.x, bi.row1.y * gain.y, bi.row1.z * gain.z⟩,
       ⟨bi.row2.x * gain.x, bi.row2.y * gain.y, bi.row2.z * gain.z⟩⟩
    biG.mul bradford

/-! ## Aim: the exposure that prints an 18 % grey -/

/-- How far the print at log exposures `e` is from an 18 % grey of the viewing light, per channel
of XYZ. -/
def aimResidual (m : PrintMaterial α) (e : Vec3 α) : Vec3 α :=
  let xyz := exposureToXyz m e
  ⟨xyz.x - 0.18 * m.white.x, xyz.y - 0.18 * m.white.y, xyz.z - 0.18 * m.white.z⟩

/-- The Jacobian of the print's XYZ in the three log exposures by forward differences of `1e-4`:
column `k` is the change from moving `e_k`. -/
def aimJacobian (m : PrintMaterial α) (e : Vec3 α) : Mat3 α :=
  let xyz := exposureToXyz m e
  let h : α := 1.0e-4
  let col (moved : Vec3 α) : Vec3 α :=
    let x2 := exposureToXyz m moved
    ⟨(x2.x - xyz.x) / h, (x2.y - xyz.y) / h, (x2.z - xyz.z) / h⟩
  let c0 := col ⟨e.x + h, e.y, e.z⟩
  let c1 := col ⟨e.x, e.y + h, e.z⟩
  let c2 := col ⟨e.x, e.y, e.z + h⟩
  ⟨⟨c0.x, c1.x, c2.x⟩, ⟨c0.y, c1.y, c2.y⟩, ⟨c0.z, c1.z, c2.z⟩⟩

/-- Newton's method for `aim` with at most `fuel` steps from `e`: it stops when every residual is
below `1e-13`, and fails (`none`) on a singular Jacobian. -/
def aimNewton (m : PrintMaterial α) : Nat → Vec3 α → Option (Vec3 α)
  | 0, e => some e
  | fuel + 1, e =>
    let f := aimResidual m e
    let worst := max (max (max 0.0 (fabs f.x)) (fabs f.y)) (fabs f.z)
    if worst < 1.0e-13 then some e
    else
      match Mat3.invert3 (aimJacobian m e) with
      | none => none
      | some inv => aimNewton m fuel (e - inv.mulVec f)

/-- `aim`: Newton's method from `(0, 0, 0)` with at most 50 steps, accepted when the print is within
`1e-6` of an 18 % grey in each of X, Y and Z (`load_spektrafilm_print`). -/
def aimSolve (m : PrintMaterial α) : Option (Vec3 α) :=
  match aimNewton m 50 ⟨0.0, 0.0, 0.0⟩ with
  | none => none
  | some e =>
    let r := aimResidual m e
    if fabs r.x < 1.0e-6 ∧ fabs r.y < 1.0e-6 ∧ fabs r.z < 1.0e-6 then some e else none

/-! ## Loading a spektrafilm print profile -/

/-- The fields of a spektrafilm print profile that the print reads (its `data` and
`info.support`), as numbers and nulls. -/
structure SpektrafilmProfile (α : Type) where
  wavelengths : Array α
  logSensitivity : Vec3 (Array (Option α))
  channelDensity : Vec3 (Array (Option α))
  baseDensity : Array (Option α)
  logExposure : Array α
  densityCurves : Vec3 (Array α)
  paper : Bool

/-- The print material of a spektrafilm print profile (`load_spektrafilm_print`): its dyes and base
resampled to the spectral grid (held at the nearest value outside the profile's data), its curves
as given, the observer under D50 or D55 and `aim`. `none` when a column cannot be read or the
curves do not reach an 18 % grey. -/
def loadMaterial (p : SpektrafilmProfile α) : Option (PrintMaterial α) :=
  let col (v : Array (Option α)) := resampleColumn p.wavelengths v true 0.0 false
  match col p.baseDensity, col p.channelDensity.x, col p.channelDensity.y,
      col p.channelDensity.z with
  | some base, some d0, some d1, some d2 =>
    let obs := observerWeights (viewingLight p.paper)
    let m0 : PrintMaterial α :=
      { logExposure := p.logExposure, curves := p.densityCurves, dye := ⟨d0, d1, d2⟩, base := base,
        observer := obs, white := observerWhite obs, aim := ⟨0.0, 0.0, 0.0⟩, paper := p.paper }
    (aimSolve m0).map fun aim => { m0 with aim := aim }
  | _, _, _, _ => none

/-- The print's layers' sensitivity on the spectral grid: `10^` of the log sensitivity, zero where
the profile has none. -/
def SpektrafilmProfile.sensitivity (p : SpektrafilmProfile α) : Option (Vec3 (Array α)) :=
  let col (v : Array (Option α)) := resampleColumn p.wavelengths v false 0.0 true
  match col p.logSensitivity.x, col p.logSensitivity.y, col p.logSensitivity.z with
  | some s0, some s1, some s2 => some ⟨s0, s1, s2⟩
  | _, _, _ => none

/-- The print matrix of a measure for a film (`to_print` of `load_spektrafilm_print`): `Q` from the
layers' sensitivity under the printer lamp, then `printMatrix`. -/
def materialPrintMatrix (p : SpektrafilmProfile α) (film : FilmModel α) (dm : DensityMatrices α) :
    Option (Mat3 α) :=
  match p.sensitivity with
  | none => none
  | some s =>
    match filmToPrintDensity film (printResponse s) with
    | none => none
    | some q => printMatrix q dm

end Negicc
