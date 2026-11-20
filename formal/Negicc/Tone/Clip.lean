/-
The tone stage's clip check on the export.
-/
module

public import Negicc.Scene.Convert
public import Negicc.Tone.Curve

@[expose] public section

/-!
# The clip check

Before the export is converted with the curve, the curve is tried on a sparse sample of the whole
unconverted image (every 8th pixel each way, film borders included) through the conversion
(`tone_clip_sample`). Where it would clip more than `toneClipAllowance` more of those pixels white
than the conversion without it, the highlight and range moves are scaled down
(`tone_plan_for_image`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The step of the clip check's sample, in pixels each way (`kToneClipSampleStep`). -/
def toneClipSampleStep : Nat := 8

/-- The shares of the sample's pixels the conversion clips white (a channel at the 16-bit maximum)
and black (every channel 0): with the configuration as it is (`q = none`) or with curve `q`'s table
(`tone_clip_sample`). `img` is the unconverted image, `w` pixels a row, row-major. -/
def toneClipSample (c : PipelineConfig α) (m : Mat3 α) (img : Array (Vec3 Nat)) (w h : Nat)
    (q : Option (ToneCurve α)) : α × α :=
  let cfg : PipelineConfig α := match q with
    | some curve => { c with tone := some (toneTable curve) }
    | none => c
  let ys := (List.range ((h + toneClipSampleStep - 1) / toneClipSampleStep)).map (· * toneClipSampleStep)
  let xs := (List.range ((w + toneClipSampleStep - 1) / toneClipSampleStep)).map (· * toneClipSampleStep)
  let counts := ys.foldl (fun acc y => xs.foldl (fun (n, wc, kc) x =>
      let o := outputPixel cfg m (img.getD (y * w + x) ⟨0, 0, 0⟩)
      (n + 1,
       wc + (if o.x = 65535 || o.y = 65535 || o.z = 65535 then 1 else 0),
       kc + (if o.x = 0 && o.y = 0 && o.z = 0 then 1 else 0))) acc) ((0 : Nat), (0 : Nat), (0 : Nat))
  let (n, wc, kc) := counts
  if n = 0 then (0.0, 0.0) else ((wc : α) / (n : α), (kc : α) / (n : α))

/-- The plan's curve for the image the export converts (`tone_plan_for_image`): the designed curve
(share 1, `designed`), or the curve of the share `tonePickClipShare` keeps, each later share's curve
planned by `tonePlan`. Returns the share kept, the targets and the curve. -/
def tonePlanForImage (render : ToneDescriptors α) (predL2 predL98 predRange : α)
    (c : PipelineConfig α) (m : Mat3 α) (img : Array (Vec3 Nat)) (w h : Nat) (whiteBefore : α)
    (designed : ToneTargets α × ToneCurve α) : α × ToneTargets α × ToneCurve α :=
  let planAt : Nat → ToneTargets α × ToneCurve α := fun k =>
    if k = 0 then designed else tonePlan render predL2 predL98 predRange (toneClipShares.getD k 0.0)
  let k := tonePickClipShare whiteBefore fun k => (toneClipSample c m img w h (some (planAt k).2)).1
  (toneClipShares.getD k 0.0, planAt k)

end Negicc
