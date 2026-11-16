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
