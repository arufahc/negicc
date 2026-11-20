/-
Properties of the base definitions over the real numbers.
-/
module

public import Negicc.Base.Real

public section

/-!
# Properties of the base definitions

Proved over `ℝ`. Each statement says in its docstring what it means for the pipeline.
-/

namespace Negicc

open Scalar

/-! ## Interpolation -/

/-- A share `0` of the way from `a` to `b` is `a`. -/
theorem lerp_zero (a b : ℝ) : lerp a b 0 = a := by
  simp only [lerp]
  norm_num

/-- A share `1` of the way from `a` to `b` is `b`. -/
theorem lerp_one (a b : ℝ) : lerp a b 1 = b := by
  simp only [lerp]
  norm_num

/-- `lerp` in the form the tone table's lookup uses: `a + t (b - a)`. -/
theorem lerp_eq (a b t : ℝ) : lerp a b t = a + t * (b - a) := by
  simp only [lerp]
  norm_num
  ring

/-- A table read at a whole index gives that sample: interpolation passes through the samples. -/
theorem tableAt_natCast (tab : Array ℝ) (k : ℕ) (h2 : 2 ≤ tab.size) (hk : k < tab.size) :
    tableAt tab (k : ℝ) = tab[k] := by
  simp only [tableAt, natFloor_real, Nat.floor_natCast]
  by_cases hle : k ≤ tab.size - 2
  · have hk1 : k + 1 < tab.size := by omega
    rw [min_eq_left hle]
    simp only [Array.getD, hk, hk1, dite_true, sub_self]
    exact lerp_zero _ _
  · have hmin : min k (tab.size - 2) = tab.size - 2 := by omega
    have hk' : tab.size - 2 + 1 = k := by omega
    have hk2 : tab.size - 2 < tab.size := by omega
    rw [hmin]
    have hsub : (k : ℝ) - ((tab.size - 2 : ℕ) : ℝ) = 1 := by
      have : (k : ℝ) = ((tab.size - 2 : ℕ) : ℝ) + 1 := by
        rw [← hk']; push_cast; ring
      rw [this]; ring
    rw [hsub, lerp_one]
    simp only [Array.getD, hk', hk, dite_true]
    rfl

/-- A share `t` in `[0, 1]` of the way between two values that lie in `[lo, hi]` lies in `[lo, hi]`. -/
theorem lerp_mem {a b t lo hi : ℝ} (ha : lo ≤ a ∧ a ≤ hi) (hb : lo ≤ b ∧ b ≤ hi) (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) : lo ≤ lerp a b t ∧ lerp a b t ≤ hi := by
  rw [lerp_eq]
  constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2]

/-- **What a caller may rely on when reading a table.** For an argument in `[0, 1]` and a table of
at least two samples, `tableEval` lies between the smallest and the largest sample: inside the
table's range the interpolation never overshoots its samples. Outside `[0, 1]` the end segment's
line is continued, so there the value can leave that range; the conversion's callers feed the input
curves values in `[0, 1]` (the knee's output) and the output curves the colour table's values. -/
theorem tableEval_mem {tab : Array ℝ} (h2 : 2 ≤ tab.size) {lo hi : ℝ}
    (hs : ∀ (i : ℕ) (h : i < tab.size), lo ≤ tab[i] ∧ tab[i] ≤ hi) {v : ℝ} (hv0 : 0 ≤ v)
    (hv1 : v ≤ 1) : lo ≤ tableEval tab v ∧ tableEval tab v ≤ hi := by
  have hne : tab.size ≠ 0 := by omega
  have hn1 : tab.size ≠ 1 := by omega
  simp only [tableEval, hne, hn1, ite_false, tableAt, natFloor_real]
  set n := tab.size with hn
  have hn1 : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by rw [Nat.cast_sub (by omega)]; simp
  set s := v * ((n - 1 : ℕ) : ℝ) with hsdef
  have hs0 : 0 ≤ s := by rw [hsdef, hn1]; have : (2 : ℝ) ≤ n := by exact_mod_cast h2
                         nlinarith
  have hs1 : s ≤ (n : ℝ) - 1 := by
    rw [hsdef, hn1]; have : (2 : ℝ) ≤ n := by exact_mod_cast h2
    nlinarith
  set k := min ⌊s⌋₊ (n - 2) with hk
  have hkn : k + 1 < n := by omega
  have hkle : (k : ℝ) ≤ s := by
    have : (k : ℝ) ≤ (⌊s⌋₊ : ℝ) := by exact_mod_cast min_le_left _ _
    exact le_trans this (Nat.floor_le hs0)
  have ht1 : s - (k : ℝ) ≤ 1 := by
    by_cases hc : ⌊s⌋₊ ≤ n - 2
    · have : k = ⌊s⌋₊ := min_eq_left hc
      rw [this]; linarith [Nat.lt_floor_add_one s]
    · have : k = n - 2 := min_eq_right (by omega)
      rw [this, Nat.cast_sub (by omega)]; push_cast; linarith
  have ga : tab.getD k 0.0 = tab[k] := by
    simp only [Array.getD]; exact dite_eq_left (show k < tab.size by omega)
  have gb : tab.getD (k + 1) 0.0 = tab[k + 1] := by
    simp only [Array.getD]; exact dite_eq_left (show k + 1 < tab.size by omega)
  rw [ga, gb]
  exact lerp_mem (hs k (by omega)) (hs (k + 1) hkn) (by linarith) ht1

/-! ## Vectors and matrices -/

/-- Scaling the rows of a matrix scales the components of its product with a vector. This is how
the exposure and the gains enter the conversion: `candidate_matrix` scales the rows of the input
matrix. -/
theorem Mat3.mulVec_scaleRows (s : Vec3 ℝ) (m : Mat3 ℝ) (v : Vec3 ℝ) :
    (Mat3.scaleRows s m).mulVec v = s.mul (m.mulVec v) := by
  ext <;> simp only [Mat3.scaleRows, Mat3.mulVec, Vec3.smul, Vec3.dot, Vec3.mul] <;> ring

/-- The product with a matrix product is the two products in turn. -/
theorem Mat3.mulVec_mul (a b : Mat3 ℝ) (v : Vec3 ℝ) :
    (a.mul b).mulVec v = a.mulVec (b.mulVec v) := by
  ext <;> simp only [Mat3.mul, Mat3.mulVec, Mat3.col0, Mat3.col1, Mat3.col2, Vec3.dot] <;> ring

/-- A diagonal matrix multiplies each component by its diagonal entry. -/
theorem Mat3.mulVec_diag (d v : Vec3 ℝ) : (Mat3.diag d).mulVec v = d.mul v := by
  ext <;> simp only [Mat3.diag, Mat3.mulVec, Vec3.dot, Vec3.mul] <;> norm_num

/-- The identity matrix leaves a vector alone. -/
theorem Mat3.mulVec_identity (v : Vec3 ℝ) : (Mat3.identity : Mat3 ℝ).mulVec v = v := by
  ext <;> simp only [Mat3.identity, Mat3.diag, Mat3.mulVec, Vec3.dot, Vec3.fill] <;> norm_num

/-! ## The sRGB curve

The sRGB standard gives the curve's two segments with rounded constants, so the segments do not
meet exactly: at the linear threshold `0.0031308` the line reaches `0.040449936` and the power
segment starts at `0.0404499...`, both a little below the encoded threshold `0.04045`. A linear
value just above `0.0031308` is therefore encoded by the power segment to a code just below
`0.04045`, which the decoder takes for a code of the line. On that sliver, of width `7e-9`, the
decoder does not return the value exactly (the error is below `3e-9`). Everywhere else it does,
and the statements below say exactly where. No 8-bit and no 16-bit code lies in the matching gap
of codes, `(0.040449936, 0.04045]`, so decoding a stored code and encoding it again never meets
it. -/

/-- For non-negative reals, `a ^ q < b ^ p` gives `a < b ^ (p / q)`. -/
private theorem lt_rpow_div_of_pow_lt_pow {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (p q : ℕ) (hq : q ≠ 0)
    (h : a ^ q < b ^ p) : a < b ^ ((p : ℝ) / (q : ℝ)) := by
  have h1 : (a ^ q) ^ ((q : ℝ)⁻¹) < (b ^ p) ^ ((q : ℝ)⁻¹) :=
    Real.rpow_lt_rpow (pow_nonneg ha q) h (by positivity)
  rw [Real.pow_rpow_inv_natCast ha hq, ← Real.rpow_natCast b p, ← Real.rpow_mul hb] at h1
  rwa [div_eq_mul_inv]

/-- For non-negative reals, `b ^ p < a ^ q` gives `b ^ (p / q) < a`. -/
private theorem rpow_div_lt_of_pow_lt_pow {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (p q : ℕ) (hq : q ≠ 0)
    (h : b ^ p < a ^ q) : b ^ ((p : ℝ) / (q : ℝ)) < a := by
  have h1 : (b ^ p) ^ ((q : ℝ)⁻¹) < (a ^ q) ^ ((q : ℝ)⁻¹) :=
    Real.rpow_lt_rpow (pow_nonneg hb p) h (by positivity)
  rw [Real.pow_rpow_inv_natCast ha hq, ← Real.rpow_natCast b p, ← Real.rpow_mul hb] at h1
  rwa [div_eq_mul_inv]

/-- The encoding's line segment. -/
theorem srgbEncode_of_le {x : ℝ} (hx : x ≤ 0.0031308) : srgbEncode x = x * 12.92 := by
  unfold srgbEncode
  exact ite_eq_left hx

/-- The encoding's power segment. -/
theorem srgbEncode_of_gt {x : ℝ} (hx : 0.0031308 < x) :
    srgbEncode x = x ^ ((1.0 : ℝ) / 2.4) * 1.055 - 0.055 := by
  unfold srgbEncode
  exact ite_eq_right (not_le.mpr hx)

/-- The decoding's line segment. -/
theorem srgbDecode_of_le {c : ℝ} (hc : c ≤ 0.04045) : srgbDecode c = c * (1.0 / 12.92) := by
  unfold srgbDecode
  exact ite_eq_left hc

/-- The decoding's power segment. -/
theorem srgbDecode_of_gt {c : ℝ} (hc : 0.04045 < c) :
    srgbDecode c = ((c + 0.055) * (1.0 / 1.055)) ^ (2.4 : ℝ) := by
  unfold srgbDecode
  exact ite_eq_right (not_le.mpr hc)

/-- Decoding undoes encoding on the line segment: for every linear value up to `0.0031308`. -/
theorem srgbDecode_srgbEncode_of_le {x : ℝ} (hx : x ≤ 0.0031308) : srgbDecode (srgbEncode x) = x := by
  have h2 : x * 12.92 ≤ (0.04045 : ℝ) := by linarith
  rw [srgbEncode_of_le hx, srgbDecode_of_le h2]
  norm_num
  ring

/-- Decoding undoes encoding on the power segment: for every linear value from `0.00313081` on. -/
theorem srgbDecode_srgbEncode_of_ge {x : ℝ} (hx : 0.00313081 ≤ x) : srgbDecode (srgbEncode x) = x := by
  have hx0 : (0 : ℝ) ≤ x := by linarith
  have h1 : (0.0031308 : ℝ) < x := by linarith
  have hb : (0.09545 / 1.055 : ℝ) < x ^ ((1.0 : ℝ) / 2.4) := by
    have h5 : (0.09545 / 1.055 : ℝ) < (0.00313081 : ℝ) ^ (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) :=
      lt_rpow_div_of_pow_lt_pow (by norm_num) (by norm_num) 5 12 (by norm_num) (by norm_num)
    have he : (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) = (1.0 : ℝ) / 2.4 := by norm_num
    rw [he] at h5
    exact lt_of_lt_of_le h5 (Real.rpow_le_rpow (by norm_num) hx (by norm_num))
  have h3 : (0.04045 : ℝ) < x ^ ((1.0 : ℝ) / 2.4) * 1.055 - 0.055 := by
    have : (0.09545 / 1.055 : ℝ) * 1.055 = 0.09545 := by norm_num
    nlinarith
  rw [srgbEncode_of_gt h1, srgbDecode_of_gt h3]
  have h4 : (x ^ ((1.0 : ℝ) / 2.4) * 1.055 - 0.055 + 0.055) * (1.0 / 1.055) = x ^ ((1.0 : ℝ) / 2.4) := by
    norm_num
    ring
  rw [h4, ← Real.rpow_mul hx0]
  norm_num

/-- Between the two, the round trip is not exact: at the linear value `0.003130802` the decoder
returns a smaller number. (The sliver on which this happens ends at about `0.0031308073`.) -/
theorem srgbDecode_srgbEncode_lt_in_gap :
    srgbDecode (srgbEncode (0.003130802 : ℝ)) < 0.003130802 := by
  have h1 : (0.0031308 : ℝ) < 0.003130802 := by norm_num
  have he : (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) = (1.0 : ℝ) / 2.4 := by norm_num
  -- The power segment's code is at most 0.04045, so the decoder takes the line.
  have hle : (0.003130802 : ℝ) ^ ((1.0 : ℝ) / 2.4) < 0.09545 / 1.055 := by
    have h5 : (0.003130802 : ℝ) ^ (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) < 0.09545 / 1.055 :=
      rpow_div_lt_of_pow_lt_pow (by norm_num) (by norm_num) 5 12 (by norm_num) (by norm_num)
    rwa [he] at h5
  have h2 : (0.003130802 : ℝ) ^ ((1.0 : ℝ) / 2.4) * 1.055 - 0.055 ≤ (0.04045 : ℝ) := by
    have : (0.09545 / 1.055 : ℝ) * 1.055 = 0.09545 := by norm_num
    nlinarith
  -- And that code is below the line's code for the same value.
  have hlt : (0.003130802 : ℝ) ^ ((1.0 : ℝ) / 2.4) < (12.92 * 0.003130802 + 0.055) / 1.055 := by
    have h5 : (0.003130802 : ℝ) ^ (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) < (12.92 * 0.003130802 + 0.055) / 1.055 :=
      rpow_div_lt_of_pow_lt_pow (by norm_num) (by norm_num) 5 12 (by norm_num) (by norm_num)
    rwa [he] at h5
  rw [srgbEncode_of_gt h1, srgbDecode_of_le h2]
  have h6 : ((12.92 * 0.003130802 + 0.055) / 1.055 : ℝ) * 1.055 = 12.92 * 0.003130802 + 0.055 := by norm_num
  have h7 : (0.003130802 : ℝ) ^ ((1.0 : ℝ) / 2.4) * 1.055 - 0.055 < 12.92 * 0.003130802 := by nlinarith
  have h8 : ((12.92 : ℝ) * 0.003130802) * (1.0 / 12.92) = 0.003130802 := by norm_num
  nlinarith

/-- Encoding undoes decoding on the line segment: for every code up to `0.040449936`, the code of
the linear threshold. -/
theorem srgbEncode_srgbDecode_of_le {c : ℝ} (hc : c ≤ 0.040449936) : srgbEncode (srgbDecode c) = c := by
  have h1 : c ≤ (0.04045 : ℝ) := by linarith
  have h2 : c * (1.0 / 12.92) ≤ (0.0031308 : ℝ) := by
    norm_num
    linarith
  rw [srgbDecode_of_le h1, srgbEncode_of_le h2]
  norm_num
  ring

/-- Encoding undoes decoding on the power segment: for every code above `0.04045`. The highlight
shoulder relies on this: it computes its curve as an encoded value and decodes it, and the
picture's code is that encoded value. -/
theorem srgbEncode_srgbDecode_of_gt {c : ℝ} (hc : 0.04045 < c) : srgbEncode (srgbDecode c) = c := by
  have hb0 : (0 : ℝ) ≤ (c + 0.055) * (1.0 / 1.055) := by
    have : (0 : ℝ) ≤ c + 0.055 := by linarith
    positivity
  have hb : (0.09545 / 1.055 : ℝ) < (c + 0.055) * (1.0 / 1.055) := by
    norm_num
    linarith
  have h2 : (0.0031308 : ℝ) < ((c + 0.055) * (1.0 / 1.055)) ^ (2.4 : ℝ) := by
    have h5 : (0.0031308 : ℝ) < (0.09545 / 1.055 : ℝ) ^ (((12 : ℕ) : ℝ) / ((5 : ℕ) : ℝ)) :=
      lt_rpow_div_of_pow_lt_pow (by norm_num) (by norm_num) 12 5 (by norm_num) (by norm_num)
    have he : (((12 : ℕ) : ℝ) / ((5 : ℕ) : ℝ)) = (2.4 : ℝ) := by norm_num
    rw [he] at h5
    exact lt_trans h5 (Real.rpow_lt_rpow (by norm_num) hb (by norm_num))
  rw [srgbDecode_of_gt hc, srgbEncode_of_gt h2, ← Real.rpow_mul hb0]
  have h3 : (2.4 : ℝ) * (1.0 / 2.4) = 1 := by norm_num
  rw [h3, Real.rpow_one]
  norm_num
  ring

/-- **The decoder is monotone.** Its line reaches `0.0031308050` at `0.04045` and its power segment
starts at `0.0031308073`, a step up of `2.3e-9`, so a larger code never decodes to a smaller value. -/
theorem srgbDecode_mono {a b : ℝ} (hab : a ≤ b) : srgbDecode a ≤ srgbDecode b := by
  have hjoin : (0.04045 : ℝ) * (1.0 / 12.92) < (0.09545 / 1.055 : ℝ) ^ (2.4 : ℝ) := by
    have h5 : (0.04045 * (1.0 / 12.92) : ℝ) < (0.09545 / 1.055 : ℝ) ^ (((12 : ℕ) : ℝ) / ((5 : ℕ) : ℝ)) :=
      lt_rpow_div_of_pow_lt_pow (by norm_num) (by norm_num) 12 5 (by norm_num) (by norm_num)
    have he : (((12 : ℕ) : ℝ) / ((5 : ℕ) : ℝ)) = (2.4 : ℝ) := by norm_num
    rwa [he] at h5
  by_cases hb : b ≤ 0.04045
  · rw [srgbDecode_of_le (le_trans hab hb), srgbDecode_of_le hb]
    nlinarith
  · have hb' : 0.04045 < b := lt_of_not_ge hb
    have hb0 : (0 : ℝ) ≤ (b + 0.055) * (1.0 / 1.055) := by nlinarith
    rw [srgbDecode_of_gt hb']
    by_cases ha : a ≤ 0.04045
    · rw [srgbDecode_of_le ha]
      have h1 : a * (1.0 / 12.92) ≤ (0.04045 : ℝ) * (1.0 / 12.92) := by nlinarith
      have h2 : (0.09545 / 1.055 : ℝ) ^ (2.4 : ℝ) ≤ ((b + 0.055) * (1.0 / 1.055)) ^ (2.4 : ℝ) :=
        Real.rpow_le_rpow (by norm_num) (by norm_num; linarith) (by norm_num)
      linarith
    · have ha' : 0.04045 < a := lt_of_not_ge ha
      rw [srgbDecode_of_gt ha']
      exact Real.rpow_le_rpow (by nlinarith) (by nlinarith) (by norm_num)

/-- **The encoder is not monotone**, by the rounding of the standard's constants: just above the
linear threshold the power segment starts at `0.0404499075`, below the line's `0.040449936` at the
threshold. -/
theorem srgbEncode_not_monotone : srgbEncode (0.003130801 : ℝ) < srgbEncode 0.0031308 := by
  have h1 : (0.0031308 : ℝ) < 0.003130801 := by norm_num
  rw [srgbEncode_of_gt h1, srgbEncode_of_le (le_refl _)]
  have he : (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) = (1.0 : ℝ) / 2.4 := by norm_num
  have h5 : (0.003130801 : ℝ) ^ (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) < 0.095449936 / 1.055 :=
    rpow_div_lt_of_pow_lt_pow (by norm_num) (by norm_num) 5 12 (by norm_num) (by norm_num)
  rw [he] at h5
  have : (0.095449936 / 1.055 : ℝ) * 1.055 = 0.095449936 := by norm_num
  nlinarith

/-- **The size of the encoder's step.** Where the encoder decreases, it decreases by less than
`3e-8` (the step at the threshold is `2.85e-8`), so for `x ≤ y` the code of `x` exceeds that of `y`
by less than that: far below a 16-bit step (`1.5e-5`). -/
theorem srgbEncode_le_of_le {x y : ℝ} (hxy : x ≤ y) : srgbEncode x ≤ srgbEncode y + 3e-8 := by
  by_cases hy : y ≤ 0.0031308
  · rw [srgbEncode_of_le (le_trans hxy hy), srgbEncode_of_le hy]
    nlinarith
  · have hy' : 0.0031308 < y := lt_of_not_ge hy
    have hy0 : (0 : ℝ) ≤ y := by linarith
    rw [srgbEncode_of_gt hy']
    by_cases hx : x ≤ 0.0031308
    · rw [srgbEncode_of_le hx]
      have he : (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) = (1.0 : ℝ) / 2.4 := by norm_num
      have h5 : (0.09544990700 / 1.055 : ℝ) < (0.0031308 : ℝ) ^ (((5 : ℕ) : ℝ) / ((12 : ℕ) : ℝ)) :=
        lt_rpow_div_of_pow_lt_pow (by norm_num) (by norm_num) 5 12 (by norm_num) (by norm_num)
      rw [he] at h5
      have h6 : (0.0031308 : ℝ) ^ ((1.0 : ℝ) / 2.4) ≤ y ^ ((1.0 : ℝ) / 2.4) :=
        Real.rpow_le_rpow (by norm_num) (le_of_lt hy') (by norm_num)
      have : (0.09544990700 / 1.055 : ℝ) * 1.055 = 0.09544990700 := by norm_num
      nlinarith
    · have hx' : 0.0031308 < x := lt_of_not_ge hx
      rw [srgbEncode_of_gt hx']
      have : x ^ ((1.0 : ℝ) / 2.4) ≤ y ^ ((1.0 : ℝ) / 2.4) :=
        Real.rpow_le_rpow (by linarith) hxy (by norm_num)
      nlinarith

/-! ## CIELAB -/

/-- A neutral linear sRGB colour has the same relative X, Y and Z: the rows of the matrix add up
to the white point's values exactly. -/
theorem linearSrgbToXyzRel_grey (v : ℝ) : linearSrgbToXyzRel ⟨v, v, v⟩ = ⟨v, v, v⟩ := by
  ext <;> simp only [linearSrgbToXyzRel, linearSrgbLuminance] <;> norm_num <;> ring

/-- A neutral linear sRGB colour has `a* = 0`. -/
theorem linearSrgbToLab_grey_a (v : ℝ) : (linearSrgbToLab ⟨v, v, v⟩).a = 0 := by
  simp only [linearSrgbToLab, linearSrgbToXyzRel_grey, xyzRelToLab]
  norm_num

/-- A neutral linear sRGB colour has `b* = 0`. -/
theorem linearSrgbToLab_grey_b (v : ℝ) : (linearSrgbToLab ⟨v, v, v⟩).b = 0 := by
  simp only [linearSrgbToLab, linearSrgbToXyzRel_grey, xyzRelToLab]
  norm_num

/-- The lightness of `xyzRelToLab` is `lstarOfY` of the relative luminance. -/
theorem xyzRelToLab_L (w : Vec3 ℝ) : (xyzRelToLab w).L = lstarOfY w.y := by
  simp only [xyzRelToLab, lstarOfY, labF]
  by_cases h : (labEpsilon : ℝ) < w.y
  · rw [ite_eq_left h, ite_eq_left h, ite_eq_left h]
  · rw [ite_eq_right h, ite_eq_right h]

/-- The lightness of `linearSrgbToLab` is `linearSrgbToL`: the two C++ functions agree. -/
theorem linearSrgbToLab_L (c : Vec3 ℝ) : (linearSrgbToLab c).L = linearSrgbToL c :=
  xyzRelToLab_L (linearSrgbToXyzRel c)

/-- White, `(1, 1, 1)`, is `L* = 100`, `a* = b* = 0`. -/
theorem linearSrgbToLab_white : linearSrgbToLab (⟨1, 1, 1⟩ : Vec3 ℝ) = ⟨100, 0, 0⟩ := by
  have h1 : cbrt (1 : ℝ) = 1 := by
    rw [cbrt_of_nonneg zero_le_one, Real.one_rpow]
  ext
  · rw [linearSrgbToLab_L]
    simp only [linearSrgbToL, lstarOfY, linearSrgbLuminance, labEpsilon]
    norm_num
    rw [h1]
    norm_num
  · exact linearSrgbToLab_grey_a 1
  · exact linearSrgbToLab_grey_b 1

/-! ## CIELAB's break

With CIE 15's exact constants the two branches of `labF` and of `lstarOfY` meet at the break, so
both functions are continuous and monotone, and `labFInv` undoes `labF` everywhere. -/

/-- The cube root of the break is `6/29`. -/
theorem cbrt_labEpsilon : cbrt (labEpsilon : ℝ) = 6 / 29 := by
  have h0 : (0 : ℝ) ≤ labEpsilon := by rw [labEpsilon_real]; norm_num
  have hc := cbrt_pow_three (labEpsilon : ℝ)
  have hn := cbrt_nonneg h0
  have h3 : (6 / 29 : ℝ) ^ 3 = labEpsilon := by rw [labEpsilon_real]; norm_num
  exact (pow_left_inj₀ hn (by norm_num) (by norm_num : (3 : ℕ) ≠ 0)).mp (hc.trans h3.symm)

/-- `labF`'s line reaches `6/29` at the break: the value of the cube root there. -/
theorem labF_line_at_break : (labKappa * labEpsilon + 16.0) / 116.0 = (6 / 29 : ℝ) := by
  rw [labKappa_real, labEpsilon_real]; norm_num

/-- The cube root is monotone on the non-negative reals. -/
theorem cbrt_mono {s t : ℝ} (hs : 0 ≤ s) (hst : s ≤ t) : cbrt s ≤ cbrt t := by
  rw [cbrt_of_nonneg hs, cbrt_of_nonneg (le_trans hs hst)]
  exact Real.rpow_le_rpow hs hst (by norm_num)

/-- **`labF` is monotone** (and the two branches meet at the break). -/
theorem labF_mono {s t : ℝ} (hst : s ≤ t) : labF s ≤ labF t := by
  have hk : (0 : ℝ) < labKappa := by rw [labKappa_real]; norm_num
  have he : (0 : ℝ) < labEpsilon := by rw [labEpsilon_real]; norm_num
  unfold labF
  by_cases ht : (labEpsilon : ℝ) < t
  · rw [ite_eq_left ht]
    by_cases hs : (labEpsilon : ℝ) < s
    · rw [ite_eq_left hs]; exact cbrt_mono (by linarith) hst
    · rw [ite_eq_right hs]
      have h1 : (labKappa * s + 16.0) / 116.0 ≤ (labKappa * labEpsilon + 16.0) / (116.0 : ℝ) := by
        have : labKappa * s ≤ labKappa * labEpsilon := by
          exact mul_le_mul_of_nonneg_left (le_of_not_gt hs) (le_of_lt hk)
        norm_num; linarith
      rw [labF_line_at_break] at h1
      have h2 := cbrt_mono (le_of_lt he) (le_of_lt ht)
      rw [cbrt_labEpsilon] at h2
      linarith
  · rw [ite_eq_right ht, ite_eq_right (fun h => ht (lt_of_lt_of_le h hst))]
    have : labKappa * s ≤ labKappa * t := mul_le_mul_of_nonneg_left hst (le_of_lt hk)
    norm_num; linarith

/-- **`lstarOfY` is monotone**: its line `labKappa Y` reaches `8` at the break, as does
`116 cbrt Y - 16`. -/
theorem lstarOfY_mono {s t : ℝ} (hst : s ≤ t) : lstarOfY s ≤ lstarOfY t := by
  have hk : (0 : ℝ) < labKappa := by rw [labKappa_real]; norm_num
  have he : (0 : ℝ) < labEpsilon := by rw [labEpsilon_real]; norm_num
  unfold lstarOfY
  by_cases ht : (labEpsilon : ℝ) < t
  · rw [ite_eq_left ht]
    by_cases hs : (labEpsilon : ℝ) < s
    · rw [ite_eq_left hs]
      have := cbrt_mono (by linarith) hst
      norm_num; linarith
    · rw [ite_eq_right hs]
      have h1 : labKappa * s ≤ labKappa * labEpsilon := mul_le_mul_of_nonneg_left (le_of_not_gt hs) (le_of_lt hk)
      have h8 : labKappa * labEpsilon = (8 : ℝ) := by rw [labKappa_real, labEpsilon_real]; norm_num
      have h2 := cbrt_mono (le_of_lt he) (le_of_lt ht)
      rw [cbrt_labEpsilon] at h2
      norm_num; linarith
  · rw [ite_eq_right ht, ite_eq_right (fun h => ht (lt_of_lt_of_le h hst))]
    exact mul_le_mul_of_nonneg_left hst (le_of_lt hk)

/-- **`labFInv` undoes `labF`**, for every real number. -/
theorem labFInv_labF (t : ℝ) : labFInv (labF t) = t := by
  have hk : (0 : ℝ) < labKappa := by rw [labKappa_real]; norm_num
  have he : (0 : ℝ) < labEpsilon := by rw [labEpsilon_real]; norm_num
  unfold labF
  by_cases ht : (labEpsilon : ℝ) < t
  · rw [ite_eq_left ht]
    have h3 : cbrt t * cbrt t * cbrt t = t := by
      calc cbrt t * cbrt t * cbrt t = cbrt t ^ 3 := by ring
        _ = t := cbrt_pow_three t
    unfold labFInv
    rw [h3, ite_eq_left ht]
  · rw [ite_eq_right ht]
    set f := (labKappa * t + 16.0) / (116.0 : ℝ) with hf
    have hf1 : f ≤ 6 / 29 := by
      have h1 : labKappa * t ≤ labKappa * labEpsilon :=
        mul_le_mul_of_nonneg_left (le_of_not_gt ht) (le_of_lt hk)
      have h2 : f ≤ (labKappa * labEpsilon + 16.0) / (116.0 : ℝ) := by
        rw [hf]; exact div_le_div_of_nonneg_right (by linarith) (by norm_num)
      rwa [labF_line_at_break] at h2
    have hcube : ¬ (labEpsilon : ℝ) < f * f * f := by
      have h3 : (6 / 29 : ℝ) * (6 / 29) * (6 / 29) = labEpsilon := by rw [labEpsilon_real]; norm_num
      by_cases h0 : 0 ≤ f
      · have : f * f * f ≤ (6 / 29) * (6 / 29) * (6 / 29) := by
          have h2 : f * f ≤ (6 / 29) * (6 / 29) := mul_self_le_mul_self h0 hf1
          exact mul_le_mul h2 hf1 h0 (by norm_num)
        rw [← h3] at *
        exact not_lt.mpr this
      · have : f * f * f ≤ 0 := by
          have : 0 ≤ f * f := mul_self_nonneg f
          exact mul_nonpos_of_nonneg_of_nonpos this (le_of_lt (lt_of_not_ge h0))
        exact not_lt.mpr (le_trans this (le_of_lt he))
    unfold labFInv
    rw [ite_eq_right hcube, hf]
    have h116 : (116.0 : ℝ) * ((labKappa * t + 16.0) / 116.0) - 16.0 = labKappa * t := by
      have : (116.0 : ℝ) = 116 := by norm_num
      rw [this, mul_div_cancel₀ _ (by norm_num : (116 : ℝ) ≠ 0)]
      have : (16.0 : ℝ) = 16 := by norm_num
      rw [this]; ring
    rw [h116, mul_div_cancel_left₀ _ hk.ne']

/-! ## Density -/

/-- The density of a product of transmissions is the sum of the densities. A gain on a channel is
therefore an offset of its density. -/
theorem density_mul {s t : ℝ} (hs : 0 < s) (ht : 0 < t) :
    density (s * t) = density s + density t := by
  simp only [density, log10, log_real]
  rw [Real.log_mul hs.ne' ht.ne']
  ring

/-- `transmission` undoes `density`. -/
theorem transmission_density {t : ℝ} (ht : 0 < t) : transmission (density t) = t := by
  simp only [transmission, density, log10_real, pow_real, neg_neg]
  norm_num
  exact Real.rpow_logb (by norm_num) (by norm_num) ht

/-- `density` undoes `transmission`. -/
theorem density_transmission (d : ℝ) : density (transmission d) = d := by
  simp only [transmission, density, log10_real, pow_real]
  norm_num

end Negicc
