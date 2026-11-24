/-
The film base: where it is read from, and the statistics that read it.
-/
module

public import Negicc.Scene.Input

@[expose] public section

/-!
# The film base

The film base `f` is taken, in this order (`neg_process.cc`, `detect_film_base` in
`film_base_detector.cc`):

1. a capture of unexposed film (`--film_base`), scaled by the scan's exposure over the capture's
   (`baseFromCapture`); refused when the film it reads is at the sensor's clip before the flat field;
2. the rebate: the median per channel of the detector's patch beside the frame, inset by 15 % on
   each side (`rebateBox`, `medianU16`), refused when it holds too few samples, when more than 0.1 %
   of its samples reached 98 % of the white level before the flat field, or when the thinnest film
   outshines it by more than 0.5 stop (`rebateAccepted`);
3. the thinnest film: the 99.5th percentile per channel of the film pixels of the half-size image
   reduced to 512 pixels (`percentileF32`);
4. the bundle's base, scaled by the scan's exposure over the base capture's (`ratio` of
   `mergedMatrix`).

Which pixels of the reduced image are film is decided by a segmentation of bare light, the black
holder and their borders (`thinnest_film_base_impl`: connected parts, dilation, the chroma of the
orange mask). That segmentation is not part of this specification; its result, the film pixels, is
the input of `percentileF32`. The reduction itself is OpenCV's `INTER_AREA` resize, also not
specified here.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- numpy's percentile (method linear, Hyndman and Fan's definition 7) of values sorted ascending,
`0 ≤ q ≤ 100` (`percentile_f32`): rank `(n - 1) q / 100`, the two values around it blended from the
nearer side. -/
def percentileF32Sorted (xs : Array α) (q : α) : α :=
  let n := xs.size
  let vi := ((n - 1 : Nat) : α) * (q / 100.0)
  if ((n - 1 : Nat) : α) ≤ vi then xs.getD (n - 1) 0.0
  else
    let prev := natFloor vi
    let gamma := vi - (prev : α)
    let lo := xs.getD prev 0.0
    let hi := xs.getD (prev + 1) 0.0
    if 0.5 ≤ gamma then hi - (hi - lo) * (1.0 - gamma) else lo + (hi - lo) * gamma

/-- The percentile of unsorted values (`percentile_f32`); `0` for none (the C++ gives a NaN). -/
def percentileF32 (xs : Array α) (q : α) : α :=
  if xs.size = 0 then 0.0 else percentileF32Sorted (xs.qsort fun a b => decide (a < b)) q

/-- numpy's median of 16-bit values (`median_u16`): the middle value, or the mean of the two middle
ones; `0` for none (the C++ gives a NaN, which is not modelled). -/
def medianU16 (xs : Array Nat) : α :=
  let s := xs.qsort (· < ·)
  let h := xs.size / 2
  if xs.size % 2 = 1 then (s.getD h 0 : α) else ((s.getD (h - 1) 0 : α) + (s.getD h 0 : α)) / 2.0

/-- The rebate patch's inset (`kBasePatchInset`), the least samples it must hold
(`kBaseMinSamples`), the sensor clip as a share of the white level (`kSensorClip`), the share of
clipped samples that refuses a patch (`kBaseClipFraction`), and the stops by which the thinnest
film may outshine it (`kRebateMargin`). -/
def basePatchInset : α := 0.15
@[inherit_doc basePatchInset] def baseMinSamples : Nat := 32
@[inherit_doc basePatchInset] def sensorClip : α := 0.98
@[inherit_doc basePatchInset] def baseClipFraction : α := 0.001
@[inherit_doc basePatchInset] def rebateMargin : α := 0.5

/-- The rebate patch's inner box on the half-size image (`detect_film_base`): the detector's box
`(x, y, w, h)` in sensor pixels inset by 15 % on each side and scaled by `sx`, `sy`, rounded inward
and held to the image; `none` when it holds fewer than 32 samples or is empty along either axis.
Returns `(x0, y0, w, h)`. -/
def rebateBox (x y w h : α) (sx sy : α) (W H : Nat) : Option (Nat × Nat × Nat × Nat) :=
  let x0 := x + basePatchInset * w
  let x1 := x + (1.0 - basePatchInset) * w
  let y0 := y + basePatchInset * h
  let y1 := y + (1.0 - basePatchInset) * h
  let ceilN := fun (v : α) => let f := natFloor v; if v ≤ (f : α) then f else f + 1
  let b0 := ceilN (x0 * sx)
  let b1 := ceilN (y0 * sy)
  let b2 := min W (natFloor (x1 * sx))
  let b3 := min H (natFloor (y1 * sy))
  if (b2 - b0) * (b3 - b1) < baseMinSamples then none else some (b0, b1, b2 - b0, b3 - b1)

/-- The detector's patch is placed only from a confident detection (`kBaseMinFrameConf`, 0.5) on a
decode that is the sensor's area scaled alike in both directions (within 2 %). -/
def rebatePlaced (conf sx sy : α) : Bool :=
  decide (0.5 ≤ conf) && decide (fabs (sx - sy) ≤ 0.02 * max sx sy)

/-- A patch the sensor clipped reads the clip, not the base: more than 0.1 % of its samples with a
channel at 98 % of the white level or above, read before the flat field (`clipped_share`). -/
def patchClipped (samples : Array (Vec3 Nat)) (white : α) : Bool :=
  let clip := sensorClip * white
  let n := samples.foldl (fun n px =>
    if clip ≤ (px.x : α) || clip ≤ (px.y : α) || clip ≤ (px.z : α) then n + 1 else n) (0 : Nat)
  decide (baseClipFraction < (n : α) / max 1.0 (samples.size : α))

/-- The stops by which the thinnest film outshines the rebate's median `dec`:
`max_c log2(thinnest_c / max(dec_c, 1))`. -/
def outshoneStops (thinnest dec : Vec3 α) : α :=
  let s := fun (t d : α) => log2 (t / max d 1.0)
  max (s thinnest.x dec.x) (max (s thinnest.y dec.y) (s thinnest.z dec.z))

/-- The rebate's base is taken (`detect_film_base`): its median and its crosstalk-corrected value
positive in every channel, and the thinnest film outshining it by at most 0.5 stop. -/
def rebateAccepted (X : Mat3 α) (thinnest dec : Vec3 α) : Bool :=
  let rgb := X.mulVec dec
  decide (0.0 < min dec.x (min dec.y dec.z)) && decide (0.0 < min rgb.x (min rgb.y rgb.z)) &&
    decide (outshoneStops thinnest dec ≤ rebateMargin)

/-- The thinnest film's base is taken when the rebate is not: every channel above 1 and its
crosstalk-corrected value positive. -/
def thinnestAccepted (X : Mat3 α) (thinnest : Vec3 α) : Bool :=
  let rgb := X.mulVec thinnest
  decide (1.0 < min thinnest.x (min thinnest.y thinnest.z)) && decide (0.0 < min rgb.x (min rgb.y rgb.z))

/-- A base read from a capture of unexposed film, at the scan's exposure: the capture's reading
through `X`, scaled by the scan's exposure over the capture's (shutter times ISO). -/
def baseFromCapture (X : Mat3 α) (reading : Vec3 α) (scanExposure captureExposure : α) : Vec3 α :=
  (X.mulVec reading).map (· * (scanExposure / captureExposure))

/-- Where the film base comes from, in the order they are tried. -/
inductive FilmBaseSource where
  | capture
  | rebate
  | thinnest
  | bundle
  deriving DecidableEq, Repr

/-- The film base's source (`detect_film_base` and its caller): a capture when given, else the
rebate when it is accepted, else the thinnest film when it is accepted, else the bundle's. -/
def filmBaseSource (hasCapture : Bool) (rebateOk : Option Bool) (thinnestOk : Bool) : FilmBaseSource :=
  if hasCapture then .capture
  else if rebateOk = some true then .rebate
  else if thinnestOk then .thinnest
  else .bundle

end Negicc
