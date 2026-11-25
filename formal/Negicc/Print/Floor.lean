/-
The film base of a print, held no darker than the picture's thinnest film.
-/
module

public import Negicc.Scene.Bracket

@[expose] public section

/-!
# The film base of a print

A print's level is relative to the film base, so a base darker than the film's thinnest areas would
push thin film past the knee. When the base is not a capture its level is multiplied by
(DESIGN.md 5.8; `print_floor_factor` in `multi_target_optimizer.cc`)

```
β = max(1, max_c P99.5{ (X v)_c / f_c }  over the inner crop)
```

read on every fourth pixel each way of the inner crop (`innerCrop` of `Negicc.Scene.Bracket`). `X` is the crosstalk correction (the identity on white light)
and `f` the base. `β` is at least 1 and at least each channel's 99.5 % point
(`printFloorFactor_ge_one`, `printFloorFactor_ge_channel` in `Negicc.Print.Props`), so on the base
`β f` the picture's thinnest film reads at most the base's level.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The pixels the floor reads: every fourth pixel each way of the crop, rows outermost. -/
def floorSamples (crop : Nat × Nat × Nat × Nat) : List (Nat × Nat) :=
  let (x0, y0, cw, ch) := crop
  (List.range ((ch + 3) / 4)).flatMap fun j =>
    (List.range ((cw + 3) / 4)).map fun i => (x0 + 4 * i, y0 + 4 * j)

/-- The `p` percent point of `n` values in the C++ way (`std::nth_element` at
`min(n − 1, ⌊p / 100 n⌋)`): that entry of the values sorted ascending. -/
def percentilePoint (p : α) (v : Array α) : α :=
  let q := min (v.size - 1) (natFloor (p / 100.0 * (v.size : α)))
  (v.qsort fun a b => decide (a < b)).getD q 0.0

/-- The base detector's percentile (`kBasePercentile`): 99.5. -/
def basePercentile : α := 99.5

/-- One channel's 99.5 % point of `(X v)_c / f_c` over the sampled pixels of the crop, `row` the
crosstalk correction's row; `none` for a base that is not positive or a crop without pixels. -/
def floorChannel (row : Vec3 α) (fc : α) (pixel : Nat → Nat → Vec3 Nat) (crop : Nat × Nat × Nat × Nat) :
    Option α :=
  let pts := floorSamples crop
  if 0.0 < fc ∧ pts ≠ [] then
    some (percentilePoint basePercentile (pts.toArray.map fun (i, j) =>
      let px := pixel i j
      (row.x * (px.x : α) + row.y * (px.y : α) + row.z * (px.z : α)) / fc))
  else none

/-- `k` raised to a channel's value, when the channel has one. -/
def raiseTo (k : α) : Option α → α
  | some v => max k v
  | none => k

/-- `β` (`print_floor_factor`): the largest of 1 and each channel's 99.5 % point of
`(X v)_c / f_c` over the sampled pixels of the crop; a channel whose base is not positive is left
out. `pixel x y` is the frame's pixel. -/
def printFloorFactor (x : Mat3 α) (f : Vec3 α) (pixel : Nat → Nat → Vec3 Nat)
    (crop : Nat × Nat × Nat × Nat) : α :=
  raiseTo (raiseTo (raiseTo 1.0 (floorChannel x.row0 f.x pixel crop)) (floorChannel x.row1 f.y pixel crop))
    (floorChannel x.row2 f.z pixel crop)

/-- The base before the floor (`raise_base_to_print_floor`): the base found on the scan, else the
bundle's base scaled once by the exposures, the scan's (`shutter × ISO / 100`) over its base
capture's. -/
def baseBeforeFloor (found : Option (Vec3 α)) (bundleBase : Vec3 α) (scanShutter scanIso baseShutter baseIso : α) :
    Vec3 α :=
  match found with
  | some f => f
  | none =>
    let ratio := (scanShutter * (scanIso / 100.0)) / (baseShutter * (baseIso / 100.0))
    ⟨bundleBase.x * ratio, bundleBase.y * ratio, bundleBase.z * ratio⟩

/-- The base the print takes: the base times `β`, its colour kept (`raise_base_to_print_floor`). -/
def raisedBase (base : Vec3 α) (beta : α) : Vec3 α := ⟨base.x * beta, base.y * beta, base.z * beta⟩

/-! ## The frame the floor reads -/

/-- The frame the floor reads (`apply_print_floor` in `np_pipeline.cc`): the search's frame given by
`--roi`, the whole scan, or the frame detector's crop of the half-size image `(x0, y0, w, h)`. -/
inductive FloorFrame where
  | roi
  | wholeScan
  | detected (x0 y0 w h : Nat)

/-- A rectangle `(x0, y0, w, h)` held inside a `bw` × `bh` image (`clamp_rect` in `np_input.cc`). -/
def clampRect (r : Nat × Nat × Nat × Nat) (bw bh : Nat) : Nat × Nat × Nat × Nat :=
  let (x0, y0, w, h) := r
  let w' := min w bw
  let h' := min h bh
  (min x0 (bw - w'), min y0 (bh - h'), w', h')

/-- The detector's confidence below which neither `--roi auto` nor the floor takes its frame
(`kLowFrameConfidence`, `kBaseMinFrameConf`). -/
def frameMinConfidence : α := 0.5

/-- The frame the floor reads: with a `--roi`, the search's frame, unless `--roi auto` fell back to
the whole image (confidence below 0.5); without one, the frame detector's crop (in the sensor's
pixels, `sensorW` × `sensorH`) mapped onto the half-size image when the detection succeeded, its
confidence is at least 0.5 and the two scales agree within 2 %; else the whole scan. `det` is the
detection's success, confidence and crop. -/
def floorFrame (hasRoi roiAuto : Bool) (roiConf : α) (det : Option (Bool × α × (α × α × α × α)))
    (halfW halfH sensorW sensorH : Nat) : FloorFrame :=
  if hasRoi then (if roiAuto ∧ roiConf < frameMinConfidence then .wholeScan else .roi)
  else
    match det with
    | none => .wholeScan
    | some (ok, conf, (cx, cy, cw, ch)) =>
      let sx : α := if 0 < sensorW then (halfW : α) / (sensorW : α) else 0.0
      let sy : α := if 0 < sensorH then (halfH : α) / (sensorH : α) else 0.0
      let aligned := 0.0 < sx ∧ 0.0 < sy ∧ fabs (sx - sy) ≤ 0.02 * max sx sy
      if aligned ∧ ok ∧ frameMinConfidence ≤ conf then
        let x0 := natFloor (cx * sx)
        let y0 := natFloor (cy * sy)
        let r := clampRect (x0, y0, max 1 (natFloor ((cx + cw) * sx) - x0), max 1 (natFloor ((cy + ch) * sy) - y0))
          halfW halfH
        .detected r.1 r.2.1 r.2.2.1 r.2.2.2
      else .wholeScan

end Negicc
