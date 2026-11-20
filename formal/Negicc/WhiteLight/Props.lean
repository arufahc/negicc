/-
Properties of the given input curves over the real numbers.
-/
module

public import Negicc.Base
public import Negicc.WhiteLight.Defs

public section

/-!
# Properties of the given input curves

Proved over `ℝ`.

* The curve is the identity at and above its pivot and for a zero bend (`inputCurve_of_ge`,
  `inputCurve_zero`), and meets the identity at the pivot (`inputCurve_at_ref`).
* Below `u_lo` the held curve has slope 1 in stops (`curveStops_hold`).
-/

namespace Negicc

open Scalar

/-- At and above the pivot the curve is the identity. -/
theorem inputCurve_of_ge (c : ChannelCurve ℝ) {v : ℝ} (h : c.ref ≤ v) : inputCurve c v = v := by
  unfold inputCurve
  have : decide (v < c.ref) = false := by simp; exact h
  simp [this]

/-- With a zero bend and a zero `k` the curve is the identity. -/
theorem inputCurve_zero (c : ChannelCurve ℝ) (ha : c.a = 0) (hk : c.k = 0) (v : ℝ) : inputCurve c v = v := by
  have hz : decZero (0 : ℝ) = true := by unfold decZero; norm_num
  unfold inputCurve
  rw [ha, hk, hz]
  simp

/-- The curve meets the identity at its pivot: `u = 0` gives `f(0) = 0` and `ref 2^0 = ref`, so the
bend leaves the film base where it is (whatever `a` and `k`, unheld). -/
theorem curveStops_zero (uMed a k : ℝ) : curveStops 0 uMed a k none = 0 := by
  simp [curveStops]

/-- Below the hold point the curve has slope 1 in stops: it keeps the offset it has at `u_lo`. -/
theorem curveStops_hold {u h uMed a k : ℝ} (hu : u < h) :
    curveStops u uMed a k (some h) = curveStops h uMed a k none + (u - h) := by
  simp [curveStops, hu]

end Negicc
