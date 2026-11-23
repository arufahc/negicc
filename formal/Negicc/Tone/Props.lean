/-
Properties of the tone stage over the real numbers.
-/
module

public import Negicc.Base
public import Negicc.Tone.Defs

public section

/-!
# Properties of the tone stage

Proved over `ℝ`.

* The curve is monotone in `L*` for every curve within its bounds (`toneCurveL_mono`): denser
  never renders lighter, which is what lets the plan read the toned render's percentiles as `f` of
  the render's. The identity curve is the identity on `[0.001, 99.999]` (`toneCurveL_identity`).
* The table's last step inverts CIELAB's `f` exactly (`labFInv_labF`, in `Negicc.Base.Props`).
* A neutral pixel stays neutral (`applyTone_grey`).
* At the share 0 the highlight and range targets are the render's own values
  (`toneTargets_share_zero`).
-/

namespace Negicc

open Scalar

/-- The curve's clamp of `L / 100` is monotone. -/
private theorem tone_clamp_mono {a b : ℝ} (hab : a ≤ b) :
    (if a < 1e-5 then (1e-5 : ℝ) else if 1.0 - 1e-5 < a then 1.0 - 1e-5 else a) ≤
    (if b < 1e-5 then (1e-5 : ℝ) else if 1.0 - 1e-5 < b then 1.0 - 1e-5 else b) := by
  split_ifs <;> norm_num at * <;> linarith

/-- The clamp of `L / 100` lies in `[1e-5, 1 - 1e-5]`. -/
private theorem tone_clamp_mem (a : ℝ) :
    (1e-5 : ℝ) ≤ (if a < 1e-5 then (1e-5 : ℝ) else if 1.0 - 1e-5 < a then 1.0 - 1e-5 else a) ∧
    (if a < 1e-5 then (1e-5 : ℝ) else if 1.0 - 1e-5 < a then 1.0 - 1e-5 else a) ≤ 1 - 1e-5 := by
  refine ⟨?_, ?_⟩ <;> split_ifs <;> norm_num at * <;> linarith

/-- `h(x) = x^c / (x^c + (1 - x)^c)` is monotone on `(0, 1)` for `c > 0`. -/
private theorem tone_h_mono {x y c : ℝ} (hx0 : 0 < x) (hxy : x ≤ y) (hy1 : y < 1) (hc : 0 < c) :
    x ^ c / (x ^ c + (1 - x) ^ c) ≤ y ^ c / (y ^ c + (1 - y) ^ c) := by
  have hy0 : 0 < y := lt_of_lt_of_le hx0 hxy
  have hx1 : 0 < 1 - x := by linarith
  have hy1' : 0 < 1 - y := by linarith
  have a1 : x ^ c ≤ y ^ c := Real.rpow_le_rpow hx0.le hxy hc.le
  have a2 : (1 - y) ^ c ≤ (1 - x) ^ c := Real.rpow_le_rpow hy1'.le (by linarith) hc.le
  have p1 : 0 < x ^ c := Real.rpow_pos_of_pos hx0 c
  have p2 : 0 < y ^ c := Real.rpow_pos_of_pos hy0 c
  have p3 : 0 < (1 - x) ^ c := Real.rpow_pos_of_pos hx1 c
  have p4 : 0 < (1 - y) ^ c := Real.rpow_pos_of_pos hy1' c
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_le_mul a1 a2 p4.le p2.le]

/-- **The tone curve is monotone in `L*`** for black ≤ white and positive gamma and contrast (every
curve within the bounds, `toneCurveOk`): a darker pixel never comes out lighter. -/
theorem toneCurveL_mono (q : ToneCurve ℝ) (hbw : q.black ≤ q.white) (hg : 0 < q.gamma)
    (hc : 0 < q.contrast) {L L' : ℝ} (hL : L ≤ L') : toneCurveL q L ≤ toneCurveL q L' := by
  unfold toneCurveL
  simp only [pow_real]
  set a := L / 100.0 with ha
  set b := L' / 100.0 with hb
  have hab : a ≤ b := by rw [ha, hb]; exact div_le_div_of_nonneg_right hL (by norm_num)
  set xa := (if a < 1e-5 then (1e-5 : ℝ) else if 1.0 - 1e-5 < a then 1.0 - 1e-5 else a)
  set xb := (if b < 1e-5 then (1e-5 : ℝ) else if 1.0 - 1e-5 < b then 1.0 - 1e-5 else b)
  have hx : xa ≤ xb := tone_clamp_mono hab
  have xa0 : 1e-5 ≤ xa := (tone_clamp_mem a).1
  have xb0 : 1e-5 ≤ xb := (tone_clamp_mem b).1
  have xb1 : xb ≤ 1 - 1e-5 := (tone_clamp_mem b).2
  have ga : 0 < xa ^ q.gamma := Real.rpow_pos_of_pos (by linarith) _
  have gle : xa ^ q.gamma ≤ xb ^ q.gamma := Real.rpow_le_rpow (by linarith) hx hg.le
  have gb1 : xb ^ q.gamma < 1 := Real.rpow_lt_one (by linarith) (by linarith) hg
  have hh := tone_h_mono ga gle gb1 hc
  have hw : 0 ≤ q.white - q.black := by linarith
  have := mul_le_mul_of_nonneg_left hh hw
  rw [mul_div_assoc, mul_div_assoc]
  linarith

/-- **The identity curve is the identity** on `L*` in `[0.001, 99.999]` (outside it the clamp of
`L / 100` to `[1e-5, 1 - 1e-5]` holds it). -/
theorem toneCurveL_identity {L : ℝ} (h0 : 0.001 ≤ L) (h1 : L ≤ 99.999) :
    toneCurveL (ToneCurve.identity : ToneCurve ℝ) L = L := by
  unfold toneCurveL ToneCurve.identity
  simp only [pow_real]
  have c1 : ¬ (L / 100.0 : ℝ) < 1e-5 := by norm_num; linarith
  have c2 : ¬ (1.0 - 1e-5 : ℝ) < L / 100.0 := by norm_num; linarith
  rw [ite_eq_right c1, ite_eq_right c2]
  have e1 : (1.0 : ℝ) = 1 := by norm_num
  rw [e1, Real.rpow_one, Real.rpow_one, Real.rpow_one]
  rw [add_sub_cancel, div_one]
  norm_num
  ring

/-- **A neutral pixel stays neutral**: the tone curve moves a grey `(y, y, y)` to `(y', y', y')`
with `y' = toneTableY lut y`, since the luminance weights add up to one. -/
theorem applyTone_grey (lut : Array ℝ) (y : ℝ) :
    applyTone lut ⟨y, y, y⟩ = ⟨toneTableY lut y, toneTableY lut y, toneTableY lut y⟩ := by
  have hl : toneLuminance (⟨y, y, y⟩ : Vec3 ℝ) = y := by
    simp only [toneLuminance]; exact linearSrgbLuminance_grey y
  unfold applyTone
  simp only [hl, sub_self, mul_zero, add_zero]

/-- At the highlight share `0` the highlight and range targets are the render's own: the last of
the clip check's shares makes no highlight move. -/
theorem toneTargets_share_zero (cur : ToneDescriptors ℝ) (a b c : ℝ) :
    (toneTargets cur a b c 0).L98 = cur.L98 ∧ (toneTargets cur a b c 0).range = cur.range := by
  simp [toneTargets]

end Negicc
