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
and the statements below say exactly where. -/

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
  by_cases h : (0.008856 : ℝ) < w.y
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
    simp only [linearSrgbToL, lstarOfY, linearSrgbLuminance]
    norm_num
    rw [h1]
    norm_num
  · exact linearSrgbToLab_grey_a 1
  · exact linearSrgbToLab_grey_b 1

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
