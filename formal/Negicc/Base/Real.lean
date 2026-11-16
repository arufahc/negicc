/-
The real numbers as scalars.
-/
module

public import Mathlib
public import Negicc.Base.Defs

@[expose] public section

/-!
# The real numbers as scalars

The instance `Scalar ℝ` reads every formula of the library as mathematics: `sqrt` is
`Real.sqrt`, `pow x y` is the real power `x ^ y`, `log` is `Real.log`, `natFloor` is `⌊x⌋₊`,
and the comparisons are decided classically, so the instance is noncomputable.

The lemmas of this module restate the operations of `Scalar` over `ℝ` in Mathlib's vocabulary.
A proof about a formula unfolds the formula and rewrites with them (they are `simp` lemmas).

A practical note for proofs: in this Mathlib revision `ring` mishandles a decimal literal that
denotes a whole number, such as `1.0`. Normalise the literals with `norm_num` before `ring`.
-/

namespace Negicc

/-- The real numbers, with Mathlib's functions. This is the reading in which properties are
proved. -/
noncomputable instance instScalarReal : Scalar ℝ where
  decLt := inferInstance
  decLe := inferInstance
  sqrt := Real.sqrt
  cbrt x := if 0 ≤ x then x ^ ((1 : ℝ) / 3) else -((-x) ^ ((1 : ℝ) / 3))
  exp := Real.exp
  log := Real.log
  pow x y := x ^ y
  tanh := Real.tanh
  natFloor x := ⌊x⌋₊

namespace Scalar

@[simp] theorem sqrt_real (x : ℝ) : sqrt x = Real.sqrt x := rfl

@[simp] theorem exp_real (x : ℝ) : exp x = Real.exp x := rfl

@[simp] theorem log_real (x : ℝ) : log x = Real.log x := rfl

@[simp] theorem pow_real (x y : ℝ) : pow x y = x ^ y := rfl

@[simp] theorem tanh_real (x : ℝ) : tanh x = Real.tanh x := rfl

@[simp] theorem natFloor_real (x : ℝ) : natFloor x = ⌊x⌋₊ := rfl

theorem cbrt_real (x : ℝ) :
    cbrt x = if 0 ≤ x then x ^ ((1 : ℝ) / 3) else -((-x) ^ ((1 : ℝ) / 3)) := rfl

/-- For a non-negative number the cube root is the power `1/3`. -/
theorem cbrt_of_nonneg {x : ℝ} (hx : 0 ≤ x) : cbrt x = x ^ ((1 : ℝ) / 3) := by
  rw [cbrt_real, ite_eq_left hx]

/-- The cube root is the inverse of the cube, for every real number. -/
theorem cbrt_pow_three (x : ℝ) : cbrt x ^ 3 = x := by
  have key : ∀ y : ℝ, 0 ≤ y → (y ^ ((1 : ℝ) / 3)) ^ 3 = y := by
    intro y hy
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [cbrt_real]
  split_ifs with h
  · exact key x h
  · have h' : 0 ≤ -x := by linarith
    have := key (-x) h'
    calc (-(-x) ^ ((1 : ℝ) / 3)) ^ 3 = -(((-x) ^ ((1 : ℝ) / 3)) ^ 3) := by ring
      _ = x := by rw [this]; ring

/-- The cube root of a non-negative number is non-negative. -/
theorem cbrt_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ cbrt x := by
  rw [cbrt_of_nonneg hx]
  exact Real.rpow_nonneg hx _

theorem log2_real (x : ℝ) : log2 x = Real.logb 2 x := by
  simp only [log2, log_real, Real.logb]
  norm_num

theorem log10_real (x : ℝ) : log10 x = Real.logb 10 x := by
  simp only [log10, log_real, Real.logb]
  norm_num

theorem exp2_real (x : ℝ) : exp2 x = (2 : ℝ) ^ x := by
  simp only [exp2, pow_real]
  norm_num

theorem fabs_real (x : ℝ) : fabs x = |x| := by
  simp only [fabs]
  split_ifs with h
  · have h' : x < 0 := by norm_num at h; exact h
    rw [abs_of_neg h']
  · have h' : 0 ≤ x := by norm_num at h; exact h
    rw [abs_of_nonneg h']

theorem sq_real (x : ℝ) : sq x = x ^ 2 := by
  simp only [sq]
  ring

theorem clamp_real (x lo hi : ℝ) : clamp x lo hi = min (max x lo) hi := rfl

theorem clamp01_real (x : ℝ) : clamp01 x = min 1 (max 0 x) := by
  simp only [clamp01]
  norm_num

/-- `clamp01` puts every real number into `[0, 1]`. -/
theorem clamp01_mem (x : ℝ) : 0 ≤ clamp01 x ∧ clamp01 x ≤ 1 := by
  rw [clamp01_real]
  exact ⟨le_min zero_le_one (le_max_left 0 x), min_le_left 1 _⟩

/-- `clamp01` leaves a number of `[0, 1]` alone. -/
theorem clamp01_of_mem {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) : clamp01 x = x := by
  rw [clamp01_real, max_eq_right h0, min_eq_right h1]

end Scalar

end Negicc
