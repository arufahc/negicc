/-
Properties of the white-light fit over the real numbers.
-/
module

public import Negicc.Base
public import Negicc.WhiteLight.Defs

public section

/-!
# Properties of the white-light fit

Proved over `ℝ`.

* The curve is the identity at and above its pivot and for a zero bend (`inputCurve_of_ge`,
  `inputCurve_zero`), and meets the identity at the pivot (`inputCurve_at_ref`).
* Below `u_lo` the held curve has slope 1 in stops (`curveStops_hold`).
* The guard never counts a cell greener than it is, counts a deep-shadow cell at Auto's `a*`, and
  leaves a cell that did not turn greener alone (`greenGuardCellA_*`).
* The shrink weight lies in `[0, 1)` (`shrinkWeight_mem`).
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

/-- The guard never counts a cell greener than it renders: the counted `a*` is at least `a`. -/
theorem greenGuardCellA_ge (a aAuto LAuto : ℝ) : a ≤ greenGuardCellA a aAuto LAuto := by
  unfold greenGuardCellA
  dsimp only
  split_ifs with hd
  · exact le_refl a
  · push Not at hd
    set x := clamp ((LAuto - 15.0) / 10.0 + 0.5) 0.0 1.0 with hx
    have hx0 : 0 ≤ x := by
      rw [hx, clamp_real]; exact le_min (le_trans (by norm_num) (le_max_right _ _)) (by norm_num)
    have hx1 : x ≤ 1 := by
      rw [hx, clamp_real]; exact le_trans (min_le_right _ _) (by norm_num)
    have hw : 0 ≤ 1.0 - (1.0 - x * x * (3.0 - 2.0 * x)) := by
      norm_num; nlinarith [mul_nonneg hx0 hx0]
    have hw1 : 1.0 - (1.0 - x * x * (3.0 - 2.0 * x)) ≤ (1 : ℝ) := by
      have h3 : 0 ≤ (1 - x) ^ 2 * (1 + 2 * x) := mul_nonneg (sq_nonneg _) (by linarith)
      norm_num; nlinarith [h3]
    have hd' : a - aAuto < 0 := by norm_num at hd; linarith
    nlinarith

/-- A cell that did not turn greener than at Auto counts as it renders. -/
theorem greenGuardCellA_of_ge {a aAuto : ℝ} (LAuto : ℝ) (h : aAuto ≤ a) :
    greenGuardCellA a aAuto LAuto = a := by
  unfold greenGuardCellA
  have : (0.0 : ℝ) ≤ a - aAuto := by norm_num; linarith
  rw [ite_eq_left this]

/-- In deep shadow (Auto's `L*` at most 10) a cell that turned greener counts at Auto's `a*`: the fit
may correct a shadow cast toward green without debit. -/
theorem greenGuardCellA_shadow {a aAuto LAuto : ℝ} (hL : LAuto ≤ 10) :
    greenGuardCellA a aAuto LAuto = if aAuto ≤ a then a else aAuto := by
  unfold greenGuardCellA
  have hx : clamp ((LAuto - 15.0) / 10.0 + 0.5) 0.0 1.0 = (0 : ℝ) := by
    rw [clamp_real]
    have : (LAuto - 15.0) / 10.0 + 0.5 ≤ (0 : ℝ) := by norm_num; linarith
    rw [max_eq_right (by norm_num; linarith)]
    norm_num
  rw [hx]
  by_cases h : aAuto ≤ a
  · have : (0.0 : ℝ) ≤ a - aAuto := by norm_num; linarith
    rw [ite_eq_left this, ite_eq_left h]
  · have : ¬ (0.0 : ℝ) ≤ a - aAuto := by norm_num; linarith
    rw [ite_eq_right this, ite_eq_right h]
    norm_num

/-- The shrink weight `m / (m + 12)` lies in `[0, 1)` for evidence `m ≥ 0`. -/
theorem shrinkWeight_mem {m : ℝ} (hm : 0 ≤ m) : 0 ≤ shrinkWeight m ∧ shrinkWeight m < 1 := by
  unfold shrinkWeight evidenceHalfMass
  have h12 : (0 : ℝ) < m + 12.0 := by norm_num; linarith
  refine ⟨div_nonneg hm h12.le, ?_⟩
  rw [div_lt_one h12]
  norm_num

end Negicc
