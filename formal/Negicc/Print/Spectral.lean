/-
The print material's spectral model: dye amounts to the print's colour.
-/
module

public import Negicc.Base.Defs
public import Negicc.Print.Cie

@[expose] public section

/-!
# The print material's spectral model

A print material (a paper or a print film) is a spektrafilm print profile. For a log exposure of
each of its three layers the print is (`np_print.h`, DESIGN.md 5.3):

```
δ_c    = curve_c(logE_c)                                 dye amount, held at the curve's ends
τ(λ)   = 10^−(Σ_c δ_c dye_c(λ) + base(λ))                the print's transmittance or reflectance
XYZ    = Σ_λ τ(λ) light(λ) observer(λ) / Σ_λ light(λ) ȳ(λ)
logE_c = aim_c − (pd_c − 0.70)                           a printing density pd exposed
```

This module holds that model (`amountsToXyz`, `exposureToXyz`, `printOfDensity`), the tables it
reads (`interp`, `onGrid`, the resampling of a profile's columns to the 1 nm grid,
`resampleColumn`), the observer under the viewing light (`observerWeights`) and the printer lamp
(`planckLamp`). Every spectral sum runs over the 401 samples of 380 to 780 nm in order, as the
C++ loops do: each sample weighted 1, both ends included, the step of 1 nm not multiplied in, since
each sum is normalised by another on the same grid (`sumRange`; `Negicc.Print.Constants`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## Tables -/

/-- The number of samples of the spectral grid, 380 to 780 nm at 1 nm (`kCieSamples`). -/
def cieSamples : Nat := 401

/-- The first wavelength of the spectral grid in nm (`kCieFirstNm`). -/
def cieFirstNm : α := 380.0

/-- The wavelength of sample `i` of the spectral grid, in nm. -/
def cieNm (i : Nat) : α := cieFirstNm + (i : α)

/-- The sum `f 0 + f 1 + … + f (n - 1)`, added in that order from `0`, as a C++ `for` loop over
`i < n` accumulating from `0.0`. -/
def sumRange (n : Nat) (f : Nat → α) : α := (List.range n).foldl (fun s i => s + f i) 0.0

/-- The index of the first abscissa above `x` (`std::upper_bound` on ascending `xs`), or the size
when there is none. -/
def upperIndex (xs : Array α) (x : α) : Nat := (xs.findIdx? (fun v => decide (x < v))).getD xs.size

/-- `numpy.interp`: the table `(xs, ys)` (abscissas ascending) read at `x`, linear between the two
points around `x` and held at the end values outside them (`interp` in `np_print.cc`). -/
def interp (xs ys : Array α) (x : α) : α :=
  if x ≤ xs.getD 0 0.0 then ys.getD 0 0.0
  else if xs.getD (xs.size - 1) 0.0 ≤ x then ys.getD (ys.size - 1) 0.0
  else
    let hi := upperIndex xs x
    let x0 := xs.getD (hi - 1) 0.0
    let x1 := xs.getD hi 0.0
    ys.getD (hi - 1) 0.0 + (ys.getD hi 0.0 - ys.getD (hi - 1) 0.0) * (x - x0) / (x1 - x0)

/-- A table on the uniform grid `start + step i` read at `x`: linear between neighbouring samples,
the end values outside (`on_grid` in `np_print.cc`; `numpy.interp` on a uniform grid). -/
def onGrid (ys : Array α) (start step x : α) : α :=
  let t := (x - start) / step
  if t ≤ 0.0 then ys.getD 0 0.0
  else if ((ys.size - 1 : Nat) : α) ≤ t then ys.getD (ys.size - 1) 0.0
  else
    let i := natFloor t
    ys.getD i 0.0 + (t - (i : α)) * (ys.getD (i + 1) 0.0 - ys.getD i 0.0)

/-- The inverse hyperbolic tangent on `(-1, 1)`, `½ ln((1 + x) / (1 − x))` (C `atanh`). -/
def atanh (x : α) : α := 0.5 * log ((1.0 + x) / (1.0 - x))

/-! ## Resampling a profile's column to the spectral grid -/

/-- A column of spektrafilm data on its own wavelengths `w` (ascending), resampled to the 1 nm
grid (`resample_column` in `np_print.cc`). A value is `none` where the profile has a null.

* `hold`: nulls before the first and after the last number take the nearest number, and a null
  between numbers makes the column unreadable (`none`). Without `hold` every null reads `missing`.
* `powerOfTen`: each number `x` is read as `10^x` (a log sensitivity).

`none` also when the column has no number. -/
def resampleColumn (w : Array α) (v : Array (Option α)) (hold : Bool) (missing : α)
    (powerOfTen : Bool) : Option (Array α) :=
  let first := (v.findIdx? Option.isSome).getD v.size
  let last := v.size - ((v.reverse.findIdx? Option.isSome).getD v.size)
  if first = v.size then none
  else if hold && ((List.range (last - first)).any fun i => (v.getD (first + i) none).isNone) then none
  else
    let held : Array (Option α) :=
      if hold then
        (Array.range v.size).map fun (i : Nat) =>
          if i < first then v.getD first none else if last ≤ i then v.getD (last - 1) none
          else v.getD i none
      else v
    let read : Array α := held.map fun x =>
      match x with
      | none => missing
      | some y => if powerOfTen then pow 10.0 y else y
    some ((Array.range cieSamples).map fun (i : Nat) => interp w read (cieNm i))

/-! ## The material -/

/-- The printing density exposed as the print's mid grey (`PrintMaterial::grey_density`), a stated
value: 0.70. -/
def printGreyDensity : α := 0.70

/-- A print material as the print emulation computes with it (`PrintMaterial` in `np_print.h`):
each layer's curve (dye amount against log exposure), the dyes' and the base's spectral densities
and the observer under the viewing light on the spectral grid, the viewing light's white and the
exposure `aim` that prints an 18 % grey. -/
structure PrintMaterial (α : Type) where
  /-- The log exposures of the curves, ascending. -/
  logExposure : Array α
  /-- The dye amount of each layer along `logExposure`. -/
  curves : Vec3 (Array α)
  /-- The spectral density of a unit amount of each layer's dye. -/
  dye : Vec3 (Array α)
  /-- The spectral density of the base. -/
  base : Array α
  /-- X, Y and Z weights: the viewing light times the observer, `Y` of the light 1. -/
  observer : Vec3 (Array α)
  /-- The viewing light's XYZ (the sum of the observer's weights). -/
  white : Vec3 α
  /-- Each layer's log exposure that prints an 18 % grey. -/
  aim : Vec3 α
  /-- The viewing light is D50 (a paper); else D55 (a print film). -/
  paper : Bool

/-- The print's transmittance (a film) or reflectance (a paper) at sample `w` for the dye amounts
`a`: `10^−(a_r dye_r + a_g dye_g + a_b dye_b + base)`. -/
def PrintMaterial.transmittance (m : PrintMaterial α) (a : Vec3 α) (w : Nat) : α :=
  pow 10.0 (-(a.x * m.dye.x.getD w 0.0 + a.y * m.dye.y.getD w 0.0 + a.z * m.dye.z.getD w 0.0 +
    m.base.getD w 0.0))

/-- The XYZ of the print for dye amounts `a`, under the viewing light (`amounts_to_xyz`). -/
def amountsToXyz (m : PrintMaterial α) (a : Vec3 α) : Vec3 α :=
  ⟨sumRange m.base.size fun w => m.transmittance a w * m.observer.x.getD w 0.0,
   sumRange m.base.size fun w => m.transmittance a w * m.observer.y.getD w 0.0,
   sumRange m.base.size fun w => m.transmittance a w * m.observer.z.getD w 0.0⟩

/-- The dye amount of each layer at log exposures `e`: each curve read with `interp`, held at its
ends. -/
def PrintMaterial.amounts (m : PrintMaterial α) (e : Vec3 α) : Vec3 α :=
  ⟨interp m.logExposure m.curves.x e.x, interp m.logExposure m.curves.y e.y,
   interp m.logExposure m.curves.z e.z⟩

/-- The XYZ of the print for a log exposure of each layer (`exposure_to_xyz`). -/
def exposureToXyz (m : PrintMaterial α) (e : Vec3 α) : Vec3 α := amountsToXyz m (m.amounts e)

/-- The log exposure of each layer for a printing density `pd`: `aim − (pd − 0.70)`. -/
def PrintMaterial.exposureOf (m : PrintMaterial α) (pd : Vec3 α) : Vec3 α :=
  ⟨m.aim.x - (pd.x - printGreyDensity), m.aim.y - (pd.y - printGreyDensity),
   m.aim.z - (pd.z - printGreyDensity)⟩

/-- The XYZ of the print of a printing density `pd` (DESIGN.md 5.3). -/
def printOfDensity (m : PrintMaterial α) (pd : Vec3 α) : Vec3 α := exposureToXyz m (m.exposureOf pd)

/-- The paper's white: the print where every layer holds its least dye (each curve's lowest value,
the base and the layers' fog), the lightest it gets (`print_paper_xyz`). -/
def paperXyz (m : PrintMaterial α) : Vec3 α :=
  let least (xs : Array α) : α := xs.foldl min (xs.getD 0 0.0)
  amountsToXyz m ⟨least m.curves.x, least m.curves.y, least m.curves.z⟩

/-! ## The viewing light and the printer lamp -/

/-- The observer under a viewing light: X, Y and Z weights `light(λ) cmf(λ) / Σ light ȳ`, so that
the light itself has `Y = 1` (`load_spektrafilm_print`). -/
def observerWeights (light : Array α) : Vec3 (Array α) :=
  let lightY := sumRange cieSamples fun i => light.getD i 0.0 * cieYBar.getD i 0.0
  let w (cmf : Array α) : Array α :=
    (Array.range cieSamples).map fun (i : Nat) => light.getD i 0.0 * cmf.getD i 0.0 / lightY
  ⟨w cieXBar, w cieYBar, w cieZBar⟩

/-- The viewing light's XYZ: the sum of each of the observer's weights. -/
def observerWhite (obs : Vec3 (Array α)) : Vec3 α :=
  ⟨sumRange cieSamples fun i => obs.x.getD i 0.0, sumRange cieSamples fun i => obs.y.getD i 0.0,
   sumRange cieSamples fun i => obs.z.getD i 0.0⟩

/-- The viewing light of a material: D50 for a paper, D55 for a print film (by the profile's
`support`). -/
def viewingLight (paper : Bool) : Array α := if paper then cieD50 else cieD55

/-- The viewing light as the C++'s tables hold it, to six significant digits. -/
def viewingLightLiteral (paper : Bool) : Array α := if paper then cieD50Literal else cieD55Literal

/-- The printer lamp at sample `i`: Planck's law for a 3200 K blackbody in relative units,
`1 / (λ⁵ (exp(c₂ / (λ T)) − 1))` with `λ` in metres and `c₂ = 1.4388e-2 m K`. -/
def planckLamp (i : Nat) : α :=
  let metres := cieNm i * 1.0e-9
  1.0 / (pow metres 5.0 * (exp (1.4388e-2 / (metres * 3200.0)) - 1.0))

/-- What each of the print's layers reads: its spectral sensitivity times the printer lamp. -/
def printResponse (sensitivity : Vec3 (Array α)) : Vec3 (Array α) :=
  let r (s : Array α) : Array α := (Array.range cieSamples).map fun (i : Nat) => s.getD i 0.0 * planckLamp i
  ⟨r sensitivity.x, r sensitivity.y, r sensitivity.z⟩

end Negicc
