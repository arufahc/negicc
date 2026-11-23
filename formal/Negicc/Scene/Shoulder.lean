/-
The highlight roll-off (shoulder).
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The highlight shoulder

A linear sRGB value above the chart's white (above 1) would clip in an sRGB picture. The shoulder
compresses the highlights smoothly instead. It acts per channel on the pre-shoulder linear value
`x`, in stops `t = log2 x`:

* up to a knee `x_k` it is the identity;
* between the knee and `x_hi` the *encoded* value follows
  `s = C - (C - s_k) (1 - (t - t_k) / (t_hi - t_k))^2`, a parabola in stops that reaches the
  ceiling `C` at `x_hi` with slope zero;
* from `x_hi` on it stays at the ceiling.

`x_hi` is the one parameter, set per frame. `C` follows from it (`shoulderCeiling`), and the knee
is where the parabola leaves the identity with the same slope: `t_k` solves
`shoulderKneeGap t_hi C t_k = 0`, found by bisection (`setShoulder`, `set_shoulder` in
`color_pipeline.h`).

Auto sets `x_hi` so that the frame's own highlights, not the chart's white, decide where the
roll-off starts: the 99.5 % point `q` of the largest channel of the pre-shoulder render is mapped
to sRGB code 242 (`shoulderXHiAnchored`, `shoulder_x_hi_anchored`).

Only the sRGB output applies the shoulder. The exponent is the constant 2
(`kShoulderExponent`); the C++ carries it as a parameter that no caller changes.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## Bisection -/

/-- `steps` halvings of the interval `[lo, hi]`. Each step looks at the midpoint: where the
predicate has the value `side` the midpoint replaces `lo`, otherwise it replaces `hi`. With a
predicate that is `side` at `lo` and not at `hi`, the interval keeps a place where the predicate
changes. This is the loop of `set_shoulder` and of `shoulder_x_hi_anchored`. It finds *a* change of
the predicate, which is the root looked for when there is only one. -/
def bisect (pred : α → Bool) (side : Bool) : Nat → α → α → α × α
  | 0, lo, hi => (lo, hi)
  | steps + 1, lo, hi =>
    let mid := 0.5 * (lo + hi)
    if pred mid == side then bisect pred side steps mid hi else bisect pred side steps lo mid

/-! ## The shoulder's constants -/

/-- The shoulder of a frame: `x_hi` and the constants derived from it (the `shoulder_*` fields of
`PipelineConfig`). -/
structure Shoulder (α : Type) where
  /-- The linear value from which the curve stays at its ceiling; `1` when the shoulder is off. -/
  xHi : α
  /-- The linear knee: the curve is the identity at and below it. -/
  xK : α
  /-- `log2` of the knee. -/
  tK : α
  /-- `log2` of `xHi`. -/
  tHi : α
  /-- The encoded sRGB value at the knee. -/
  sK : α
  /-- The encoded sRGB ceiling `C`, reached at `xHi`. -/
  c : α
  /-- The linear value of the ceiling. -/
  yC : α

/-- The shoulder switched off: every constant at its neutral value. -/
def Shoulder.off : Shoulder α := ⟨1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0⟩

/-- An `x_hi` at or below this value means "off" (`kShoulderOff`). -/
def shoulderOffLimit : α := 1.0001

/-- The largest `x_hi` (`kShoulderXHiLimit`). -/
def shoulderXHiLimit : α := 2.5

/-- The encoded ceiling `C` of an `x_hi`: it goes linearly from 1 (code 255) at `x_hi = 1` to
0.988 (about code 252) at `x_hi = 1.1` and stays there, so that the picture changes continuously
where the shoulder switches on (`kShoulderCeiling`, `kShoulderRamp`). -/
def shoulderCeiling (xHi : α) : α :=
  1.0 - (1.0 - 0.988) * min 1.0 ((xHi - 1.0) / 0.10)

/-- The slope of the sRGB encoding in stops, `d encode(2^t) / dt`, on the curve's power segment:
`(1.055 / 2.4) (2^t)^(1/2.4) ln 2`. -/
def shoulderSlope (t : α) : α :=
  (1.055 / 2.4) * pow (exp2 t) (1.0 / 2.4) * log 2.0

/-- The knee condition: zero where a parabola with ceiling `c` at `tHi` leaves the identity at `t`
with the identity's slope, `tHi - t = 2 (c - encode(2^t)) / slope(t)`. -/
def shoulderKneeGap (tHi c t : α) : α :=
  tHi - t - 2.0 * (c - srgbEncode (exp2 t)) / shoulderSlope t

/-- The shoulder of an `x_hi` (`set_shoulder`). Off for `x_hi ≤ 1.0001`; an `x_hi` above 2.5 is
held at 2.5. The knee `t_k` is found by 100 bisections of `[-3, 0]` on the sign of the knee
condition; were the condition to have the same sign at both ends, the shoulder would be off. -/
def setShoulder (xHi : α) : Shoulder α :=
  if xHi ≤ shoulderOffLimit then Shoulder.off
  else
    let x := min xHi shoulderXHiLimit
    let c := shoulderCeiling x
    let tHi := log2 x
    let positive : α → Bool := fun t => decide (0.0 < shoulderKneeGap tHi c t)
    let loPos := positive (-3.0)
    if loPos == positive 0.0 then Shoulder.off
    else
      let ends := bisect positive loPos 100 (-3.0) 0.0
      let tK := 0.5 * (ends.1 + ends.2)
      ⟨x, exp2 tK, tK, tHi, srgbEncode (exp2 tK), c, srgbDecode c⟩

/-! ## Applying the shoulder -/

/-- The shoulder on one channel's pre-shoulder linear value (`apply_shoulder_channel`). -/
def applyShoulderChannel (s : Shoulder α) (x : α) : α :=
  if s.xK < x then
    let t := log2 x
    if s.tHi ≤ t then s.yC
    else
      let u := 1.0 - (t - s.tK) / (s.tHi - s.tK)
      srgbDecode (s.c - (s.c - s.sK) * (u * u))
  else x

/-- The shoulder on a pixel: per channel when it is on, the identity when it is off
(`apply_shoulder`, for the sRGB output). -/
def applyShoulder (s : Shoulder α) (v : Vec3 α) : Vec3 α :=
  if shoulderOffLimit < s.xHi then v.map (applyShoulderChannel s) else v

/-- The display value of a pre-shoulder value: the shoulder, then held to `[0, 1]`. It is linear
sRGB (D65); the picture and the search's scoring read it (`to_display`). -/
def toDisplay (s : Shoulder α) (pre : Vec3 α) : Vec3 α :=
  (applyShoulder s pre).map clamp01

/-! ## The roll-off of a frame: `x_hi` from the 99.5 % point -/

/-- The sRGB value the frame's 99.5 % point is mapped to: code 242 of 255
(`kShoulderAnchorCode`). -/
def shoulderAnchor : α := 242.0 / 255.0

/-- The encoded value at which a shoulder with this `x_hi` renders the linear value `q`. -/
def shoulderCodeAt (q xHi : α) : α :=
  srgbEncode (applyShoulderChannel (setShoulder xHi) q)

/-- The `x_hi` whose curve maps the linear value `q` to sRGB code 242 (`shoulder_x_hi_anchored`).
Off (`1`) when `q` already encodes at or below that code. A longer roll-off has a lower knee and
renders `q` lower, so the code at `q` falls as `x_hi` grows; 60 bisections of `[1.0002, 2.5]` find
where it crosses the anchor, and the upper end is returned. When even `x_hi = 2.5` leaves `q` above
the anchor, 2.5 is used. -/
def shoulderXHiAnchored (q : α) : α :=
  if q ≤ 0.0 then 1.0
  else if srgbEncode q ≤ shoulderAnchor then 1.0
  else if shoulderAnchor < shoulderCodeAt q shoulderXHiLimit then shoulderXHiLimit
  else if shoulderCodeAt q 1.0002 ≤ shoulderAnchor then 1.0002
  else (bisect (fun x => decide (shoulderAnchor < shoulderCodeAt q x)) true 60 1.0002 shoulderXHiLimit).2

/-- The number of bins of the histogram of the largest channel (`kMaxHistBins`). -/
def maxHistBins : Nat := 4096

/-- The bin of a largest-channel value: 4096 bins over `[0, 2]`, values beyond in the last bin
(`max_hist_bin`). -/
def maxHistBin (v : α) : Nat := min (natFloor (v * (4096.0 / 2.0))) (maxHistBins - 1)

/-- The scan of `maxHistQuantile`: bins from index `i` on, with `below` pixels in the bins before.
The quantile lies in the first bin whose cumulative count reaches `rank`, at the share of that
bin's count the rank still needs (the middle of an empty bin). -/
def maxHistQuantileFrom (rank : α) : List Nat → Nat → Nat → α
  | [], _, _ => 2.0
  | count :: rest, i, below =>
    if rank ≤ ((below + count : Nat) : α) then
      let share : α := if 0 < count then (rank - (below : α)) / (count : α) else 0.5
      ((i : α) + clamp share 0.0 1.0) * (2.0 / 4096.0)
    else maxHistQuantileFrom rank rest (i + 1) (below + count)

/-- The `q` point (`0 ≤ q ≤ 1`) of a largest-channel histogram, interpolated linearly inside its
bin (`max_hist_quantile` in `rgb_view.h`). Zero for an empty histogram. -/
def maxHistQuantile (hist : Array Nat) (q : α) : α :=
  let total : Nat := hist.foldl (· + ·) 0
  if total = 0 then 0.0 else maxHistQuantileFrom (q * (total : α)) hist.toList 0 0

/-- The histogram of the largest channel of a set of pre-shoulder pixels. -/
def maxHistOf (pixels : Array (Vec3 α)) : Array Nat :=
  pixels.foldl (fun h v => h.modify (maxHistBin v.maxComp) (· + 1)) (Array.replicate maxHistBins 0)

/-- The roll-off of a frame from the histogram of its pre-shoulder render: the 99.5 % point, mapped
to code 242 (`shoulder_x_hi_from_hist` in `dinov3.cc`). -/
def shoulderXHiFromHist (hist : Array Nat) : α :=
  shoulderXHiAnchored (maxHistQuantile hist 0.995)

end Negicc
