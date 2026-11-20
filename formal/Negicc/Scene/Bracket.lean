/-
A bracket's calibrated range on a frame, and its highlight ceiling.
-/
module

public import Negicc.Scene.Linear
public import Negicc.Scene.Convert

@[expose] public section

/-!
# A bracket's calibrated range

A bracket's range is computed at conversion from its stored patches (`bracket_enclosure` in
`multi_target_optimizer.cc`). With `g_min`, `g_max` the smallest and largest green value of the
bracket's patches, `g_lo`, `g_hi` the scene's green 0.5 % and 99.5 % points over the inner 90 %
of the frame, and `ρ_b` the reference exposure (1/8 s, ISO 100) over the bracket's exposure:

  `e_min = g_min / g_lo ρ_b`, `e_max = g_max / g_hi ρ_b`, `e_centre = sqrt(e_min e_max)`,
  usable range `e_min / 2^1.5 .. e_max 2^1.2`.

The bracket is feasible when `e_min ≤ e_max` and admissible when the scene's span
`log2(g_hi / g_lo)` is at most the bracket's `log2(g_max / g_min)` plus 2.7 stops.

On a white-light bundle the scene's points are scaled by the frame's `e_norm`; on TriLED the range
is reported in the frame's units, divided by `e_norm` (`meter_bracket_exposures`).

# A bracket's highlight ceiling

What the conversion uses of a bracket's prediction errors is the per-channel 8-bit sRGB ceiling of
its calibration patches: the brightest code any patch's predicted colour renders to
(`profile_highlight_ceiling`), each predicted PCS `L*a*b*` (D50) taken through CIELAB's inverse, the
PCS white, the PCS to linear sRGB matrix and the sRGB curve (`pcs_lab_to_srgb_u8`). The report's
text is parsed by a regular expression, which is not part of this specification: the predicted
colours are its input. Without a report the ceiling is 255.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## The scene's green range -/

/-- The reference exposure: 1/8 s at ISO 100 (`REFERENCE_SHUTTER_S`, `REFERENCE_ISO`). -/
def referenceExposure : α := captureExposure 0.125 100.0

/-- The scene's tail on each side (`SCENE_TAIL`). -/
def sceneTail : α := 0.005

/-- The centred `ratio` share of a `W x H` frame, at least one pixel: its left, top, width and height
(`compute_inner_crop`). -/
def innerCrop (W H : Nat) (ratio : α) : Nat × Nat × Nat × Nat :=
  if W = 0 || H = 0 then (0, 0, 0, 0)
  else if 1.0 ≤ ratio then (0, 0, W, H)
  else
    let x0 := min (roundNat ((W : α) * (1.0 - ratio) * 0.5)) ((W - 1) / 2)
    let y0 := min (roundNat ((H : α) * (1.0 - ratio) * 0.5)) ((H - 1) / 2)
    (x0, y0, W - 2 * x0, H - 2 * y0)

/-- The `q` quantile of a histogram of 16-bit values, linear between ranks `floor` and `ceil` of
`q (total - 1)` (`u16_percentile_linear`); `0` for an empty histogram. -/
def u16PercentileLinear (hist : Array Nat) (total : Nat) (q : α) : α :=
  if total = 0 then 0.0
  else
    let pos := q * ((total - 1 : Nat) : α)
    let idxLo := natFloor pos
    let idxHi := if pos ≤ (idxLo : α) then idxLo else idxLo + 1
    let frac := pos - (idxLo : α)
    -- The first value whose cumulative count exceeds the rank.
    let find := fun (idx : Nat) =>
      let r := (List.range 65536).foldl (fun (acc : Nat × Option Nat) v =>
        match acc.2 with
        | some _ => acc
        | none => let cum := acc.1 + hist.getD v 0; (cum, if idx < cum then some v else none)) (0, none)
      r.2.getD 65535
    (1.0 - frac) * (find idxLo : α) + frac * (find idxHi : α)

/-- The green values of the frame's inner crop as a histogram (`g_hist` of `compute_frame_stats`). -/
def greenHistogram (frame : Array (Vec3 Nat)) (W H : Nat) (ratio : α) : Array Nat × Nat :=
  let (x0, y0, w, h) := innerCrop W H ratio
  let hist := (List.range h).foldl (fun acc ry => (List.range w).foldl (fun acc rx =>
    let px := frame.getD ((y0 + ry) * W + x0 + rx) ⟨0, 0, 0⟩
    acc.modify px.y (· + 1)) acc) (Array.replicate 65536 0)
  (hist, w * h)

/-- The scene's green `0.5 %` and `99.5 %` points, `g_lo ≥ 1` and `g_hi ≥ g_lo + 1`
(`compute_frame_stats`). -/
def sceneGreenRange (hist : Array Nat) (total : Nat) : α × α :=
  let gLo := max 1.0 (u16PercentileLinear hist total sceneTail)
  let gHi := max (gLo + 1.0) (u16PercentileLinear hist total (1.0 - sceneTail))
  (gLo, gHi)

/-! ## The enclosure -/

/-- A bracket's exposure range on a frame (`Enclosure`). -/
structure Enclosure (α : Type) where
  feasible : Bool
  admissible : Bool
  eMin : α
  eMax : α
  searchLo : α
  searchHi : α
  eCentre : α
  usableInverted : Bool
  slackEv : α
  fillRatio : α

/-- The overflow tolerances in stops: highlights 1.5, shadows 1.2. -/
def usableHiEv : α := 1.5
@[inherit_doc usableHiEv] def usableShEv : α := 1.2

/-- The range of `E` that keeps the scene's green `g_lo .. g_hi` inside a bracket whose patches span
green `gMin .. gMax` in its own capture of exposure `exposure` (`bracket_enclosure`). -/
def bracketEnclosure (gMin gMax exposure gLo gHi : α) : Enclosure α :=
  let kT := referenceExposure / exposure
  let eMin := gMin / gLo * kT
  let eMax := gMax / gHi * kT
  let uMin := eMin * pow 2.0 (-usableHiEv)
  let uMax := eMax * pow 2.0 usableShEv
  let capacity := log2 (gMax / gMin) + usableHiEv + usableShEv
  ⟨decide (eMin ≤ eMax), decide (log2 (gHi / gLo) ≤ capacity), eMin, eMax, min uMin uMax, max uMin uMax,
   sqrt (eMin * eMax), decide (uMax < uMin), log2 (eMax / eMin), gHi / gLo / (gMax / gMin)⟩

/-- The centre the search's exposure penalty pulls toward, `0` (no pull) for an inverted usable
range (`pull_centre`). -/
def pullCentre (c : Enclosure α) : α := if c.usableInverted then 0.0 else c.eCentre

/-- A bracket's range as `meter_bracket_exposures` reports it: on a white-light bundle the scene's
points scaled by `e_norm` and the range in the search's units; on TriLED the scene's points as they
are and the range divided by `e_norm` (the frame's units). -/
def meteredEnclosure (whiteLight : Bool) (eNorm gMin gMax exposure gLo gHi : α) : Enclosure α :=
  let s : α := if whiteLight then eNorm else 1.0
  let r := bracketEnclosure gMin gMax exposure (gLo * s) (gHi * s)
  let toS : α := if whiteLight then 1.0 else 1.0 / eNorm
  { r with eMin := r.eMin * toS, eMax := r.eMax * toS, eCentre := r.eCentre * toS,
           searchLo := r.searchLo * toS, searchHi := r.searchHi * toS }

/-! ## The highlight ceiling -/

/-- ICC PCS `L*a*b*` (D50) to 8-bit sRGB code values in `[0, 255]` (`pcs_lab_to_srgb_u8`): CIELAB's
inverse relative to the PCS white, the PCS to linear sRGB matrix, clamped to `[0, 1]`, the sRGB
curve, times 255. -/
def pcsLabToSrgbU8 (lab : Lab α) : Vec3 α :=
  let fy := (lab.L + 16.0) / 116.0
  let f : Vec3 α := ⟨fy + lab.a / 500.0, fy, fy - lab.b / 200.0⟩
  let xyz := Vec3.zipWith (fun fc w => labFInv fc * w) f pcsWhite
  let l := pcsToLinearSrgbUnclamped xyz
  l.map fun c => clamp01 (srgbEncode (clamp01 c)) * 255.0

/-- A bracket's per-channel 8-bit sRGB ceiling from the predicted colours of its calibration patches
(`profile_highlight_ceiling`): the largest code of each channel; 255 without any. -/
def profileHighlightCeiling (predicted : List (Lab α)) : Vec3 α :=
  match predicted with
  | [] => Vec3.fill 255.0
  | p :: ps => (ps.map pcsLabToSrgbU8).foldl (fun acc v => ⟨max acc.x v.x, max acc.y v.y, max acc.z v.z⟩)
      (pcsLabToSrgbU8 p)

end Negicc
