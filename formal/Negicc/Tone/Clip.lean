/-
The tone stage's clip check on the export.
-/
module

public import Negicc.Scene.Convert
public import Negicc.Scene.Crop
public import Negicc.Tone.Curve

@[expose] public section

/-!
# The clip check

Before the export is converted with the curve, the curve is tried on a sparse sample of the
unconverted image through the conversion (`tone_clip_sample`, `tone_clip_shares`): the 2 x 2 pixels
at the origin of every 8 x 8 block that lie inside the inner crop (`toneClipCrop` of each side), so
that every phase of a 2 x 2 colour mosaic is seen. The shares are those of the sampled pixels with a
channel at the 16-bit maximum (white) and with a channel at 0 (black). Where the curve would clip
either end more than `toneClipAllowance` beyond its own share without it, the moves are scaled down
(`tone_plan_for_image`): the highlight and range moves always, the shadow move only where the black
end fails with it whole. When no share holds both ends, the last is kept (shadows first, the
default of `--tone_stage`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The step of the clip check's sample, in pixels each way (`kToneClipSampleStep`). -/
def toneClipSampleStep : Nat := 8

/-- The inner crop the clip check reads (`kToneClipCrop`): the search's. -/
def toneClipCrop : α := 0.90

/-- The shares of the sample's pixels the conversion clips white (a channel at the 16-bit maximum)
and black (a channel at 0): with the configuration as it is (`q = none`) or with curve `q`'s table
(`tone_clip_sample`, `tone_clip_shares`). The sample is the 2 x 2 pixels at the origin of every
`toneClipSampleStep` block that lie inside the inner crop. `img` is the unconverted image, `w`
pixels a row, row-major. -/
def toneClipSample (c : PipelineConfig α) (m : Mat3 α) (img : Array (Vec3 Nat)) (w h : Nat)
    (q : Option (ToneCurve α)) : α × α :=
  let cfg : PipelineConfig α := match q with
    | some curve => { c with tone := some (toneTable curve) }
    | none => c
  let (x0, y0, cw, ch) := innerCrop (α := α) w h toneClipCrop
  let ys := (List.range h).filter fun y => y % toneClipSampleStep < 2 && y0 ≤ y && y < y0 + ch
  let xs := (List.range w).filter fun x => x % toneClipSampleStep < 2 && x0 ≤ x && x < x0 + cw
  let counts := ys.foldl (fun acc y => xs.foldl (fun (n, wc, kc) x =>
      let o := outputPixel cfg m (img.getD (y * w + x) ⟨0, 0, 0⟩)
      (n + 1,
       wc + (if o.x = 65535 || o.y = 65535 || o.z = 65535 then 1 else 0),
       kc + (if o.x = 0 || o.y = 0 || o.z = 0 then 1 else 0))) acc) ((0 : Nat), (0 : Nat), (0 : Nat))
  let (n, wc, kc) := counts
  if n = 0 then (0.0, 0.0) else ((wc : α) / (n : α), (kc : α) / (n : α))

/-- The plan's curve for the image the export converts (`tone_plan_for_image`, with its default rule):
the shares of `toneClipShares` in turn, each end against its own share before the stage
(`whiteBefore`, `blackBefore`) plus `toneClipAllowance`. At each share the curve keeps the whole
shadow move (the designed curve `designed` at the first share, `tonePlanShares` at the others); only
where the black end then fails is the shadow move scaled with the rest. The first share at which both
ends hold is kept; when none holds, the last share's curve (shadows first). Returns the highlight
share kept, the shadow share kept, the targets and the curve. -/
def tonePlanForImage (render : ToneDescriptors α) (predL2 predL98 predRange : α)
    (c : PipelineConfig α) (m : Mat3 α) (img : Array (Vec3 Nat)) (w h : Nat) (whiteBefore blackBefore : α)
    (designed : ToneTargets α × ToneCurve α) : α × α × ToneTargets α × ToneCurve α := Id.run do
  let clip := fun (q : ToneCurve α) => toneClipSample c m img w h (some q)
  let mut kept : α × α × ToneTargets α × ToneCurve α := (1.0, 1.0, designed)
  for i in [0:(toneClipShares : Array α).size] do
    let hs : α := toneClipShares.getD i 0.0
    let mut p := if i = 0 then designed else tonePlanShares render predL2 predL98 predRange hs 1.0
    let mut ss : α := 1.0
    let mut cl := clip p.2
    if i ≠ 0 ∧ toneClipAllowance < cl.2 - blackBefore then
      -- The whole shadow move clips the black end: the shadow move is scaled with the rest.
      p := tonePlanShares render predL2 predL98 predRange hs hs
      ss := hs
      cl := clip p.2
    kept := (hs, ss, p)
    if cl.1 - whiteBefore ≤ toneClipAllowance ∧ cl.2 - blackBefore ≤ toneClipAllowance then
      return kept
  pure kept

end Negicc
