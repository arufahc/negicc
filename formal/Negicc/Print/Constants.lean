/-
The print's constants from their formulas, and its spectral sums.
-/
module

public import Negicc.Print.Props
-- The kernel decides the tables and the rational sums below. It needs the bodies of the rational
-- arithmetic and of `List.all₂`, which their modules do not expose.
import all Init.Data.Rat.Basic
import all Batteries.Data.List.Basic

public section

/-!
# The print's constants and spectral sums

Where `np_print.cc` and `multi_target_optimizer.cc` write a number that a formula gives, the
specification defines it by the formula (`xyzToSrgbLindbloom`, `cieD50`, `cieD55`, `goldenPhi`,
`stopApd`, `bootH`) and keeps the C++'s digits beside it (`…Literal`). This module proves how far
the digits are from the formula, by exact rational arithmetic where the inputs are rational and by
interval bounds where a square root or a logarithm enters:

* Lindbloom's XYZ to sRGB matrix, to its seven decimals (`xyzToSrgbLindbloom_literal`); its inverse's
  row sums are his D65 exactly (`xyzToSrgbLindbloom_d65`). The CLF's matrix is the von Kries
  adaptation with CAT02 (`toSrgbMatrixOf_eq_vonKries`), the print ICC's is with Bradford
  (`pcsAdaptation_eq_vonKries`).
* The golden ratio to sixteen digits (`goldenPhi_literal`), `log10 2` to five decimals
  (`stopApd_literal`), half of it (`bootH_literal`), the printer lamp's `c₂` against `h c / k`
  (`planck_c2`).
* D50 and D55 at 1 nm: the C++'s 401 values are the 5 nm tables read linearly, rounded to six
  significant digits (`cieD50Literal_eq`, `cieD55Literal_eq`).

**The integration rule.** Every spectral integral of the print is a sum over the 401 samples of 380
to 780 nm at 1 nm, each sample weighted 1, both ends included, added in order from the first
(`sumRange`). The step, 1 nm, is not multiplied in: each integral the print uses is normalised by
another on the same grid (the observer by the light's `Y`, a density by the response's total), so
the step cancels. The viewing light's white is therefore a ratio of two such sums
(`observerWhite_x`, `observerWhite_z`), and its value is a theorem about those sums, not a given
number: for D50, `(0.9642147, 1, 0.8250171)` (`observerWhite_d50`), whose `Z` is not the PCS
illuminant's `0.8249` (`observerWhite_d50_ne_iccD50`).

Proofs: the matrices, the bounds and the sums are decided by the kernel (`norm_num` on rationals,
`decide +kernel` on the tables); no proof here uses `native_decide`.
-/

namespace Negicc

open Scalar

/-! ## The colour matrices -/

/-- **Lindbloom's seven-decimal matrix is the formula, correctly rounded**: each entry
`load_spektrafilm_print` writes is within half a unit of its seventh decimal (the largest
difference is `4.54e-8`) of the inverse of the BT.709 matrix with his D65. -/
theorem xyzToSrgbLindbloom_literal :
    Mat3.Near xyzToSrgbLindbloomLiteral xyzToSrgbLindbloom 5e-8 := by
  norm_num [Mat3.Near, xyzToSrgbLindbloomLiteral, xyzToSrgbLindbloom, rgbToXyz, primaryMatrix,
    bt709Primaries, lindbloomD65, xyToXyz, Mat3.mul, Mat3.diag, Mat3.adjInv, Mat3.det_eq,
    Mat3.mulVec, Vec3.dot, Mat3.col0, Mat3.col1, Mat3.col2, abs_le]

/-- The inverse of an invertible matrix's adjugate inverse is the matrix. -/
theorem Mat3.adjInv_adjInv (a : Mat3 ℝ) (ha : Mat3.det a ≠ 0) (hb : Mat3.det (Mat3.adjInv a) ≠ 0) :
    Mat3.adjInv (Mat3.adjInv a) = a := by
  calc Mat3.adjInv (Mat3.adjInv a) = (Mat3.adjInv (Mat3.adjInv a)).mul Mat3.identity :=
        (Mat3.mul_identity _).symm
    _ = (Mat3.adjInv (Mat3.adjInv a)).mul ((Mat3.adjInv a).mul a) := by rw [Mat3.adjInv_mul a ha]
    _ = ((Mat3.adjInv (Mat3.adjInv a)).mul (Mat3.adjInv a)).mul a := (Mat3.mul_assoc _ _ _).symm
    _ = a := by rw [Mat3.adjInv_mul _ hb, Mat3.identity_mul]

/-- **The CLF's D65 is Lindbloom's white exactly**: the row sums of the inverse of
`xyzToSrgbLindbloom`, which `toSrgbMatrix` adapts to, are `(0.95047, 1, 1.08883)`. -/
theorem xyzToSrgbLindbloom_d65 :
    (Mat3.adjInv (xyzToSrgbLindbloom : Mat3 ℝ)).rowSums = lindbloomD65 := by
  have ha : Mat3.det (rgbToXyz bt709Primaries lindbloomD65 : Mat3 ℝ) ≠ 0 := by
    norm_num [Mat3.det_eq, rgbToXyz, primaryMatrix, bt709Primaries, lindbloomD65, xyToXyz, Mat3.mul,
      Mat3.diag, Mat3.adjInv, Mat3.mulVec, Vec3.dot, Mat3.col0, Mat3.col1, Mat3.col2]
  rw [xyzToSrgbLindbloom, Mat3.adjInv_adjInv _ ha det_xyzToSrgbLindbloom, Mat3.rowSums_eq_mulVec]
  exact rgbToXyz_white _ _ det_primaryMatrix_bt709

/-- **The CLF's matrix is a von Kries adaptation with CAT02**: `toSrgbMatrixOf` scales the rows of
`cat02` as `load_spektrafilm_print` writes it, which is `xyzToSrgb` after CAT02's von Kries
adaptation from the light's white to the D65 of `xyzToSrgb`. -/
theorem toSrgbMatrixOf_eq_vonKries (x : Mat3 ℝ) (w : Vec3 ℝ) :
    toSrgbMatrixOf x w = x.mul (vonKries cat02 w (Mat3.adjInv x).rowSums) := by
  simp only [toSrgbMatrixOf, vonKries]
  congr 1
  ext <;> simp only [Mat3.scaleRowsBy, Mat3.mul, Mat3.diag, Vec3.dot, Mat3.col0, Mat3.col1,
    Mat3.col2] <;> norm_num <;> ring

/-- **The print ICC's adaptation is Bradford's von Kries adaptation** from a print film's light
(normalised to `Y = 1`) to `iccD50`, as `print_table` arranges its products. -/
theorem pcsAdaptation_eq_vonKries (w : Vec3 ℝ) :
    pcsAdaptation false w = vonKries bradford ⟨w.x / w.y, 1.0, w.z / w.y⟩ iccD50 := by
  simp only [pcsAdaptation, Bool.false_eq_true, ↓reduceIte, vonKries]
  ext <;> simp only [Mat3.mul, Mat3.diag, Vec3.dot, Mat3.col0, Mat3.col1, Mat3.col2] <;> norm_num <;>
    ring

/-! ## The search's constants -/

/-- **The golden ratio to sixteen digits**: `0.6180339887498949` is within `1e-16` of `(√5 − 1) / 2`.
It is the shortest decimal of the nearest double, one unit of the sixteenth decimal above the
correctly rounded `0.6180339887498948`. Interval bounds on `√5` by squaring. -/
theorem goldenPhi_literal : |(goldenPhiLiteral : ℝ) - goldenPhi| ≤ 1e-16 := by
  have h0 := Real.sqrt_nonneg 5
  have h5 := Real.sq_sqrt (show (0 : ℝ) ≤ 5 by norm_num)
  have hlo : (2.2360679774997896 : ℝ) ≤ √5 := by
    by_contra h
    rw [not_le] at h
    nlinarith
  have hhi : √5 ≤ (2.2360679774997900 : ℝ) := by
    by_contra h
    rw [not_le] at h
    nlinarith
  simp only [goldenPhiLiteral, goldenPhi, sqrt_real, abs_le]
  constructor <;> norm_num <;> linarith

/-- `log10 2` lies between `8651 / 28738` and `4004 / 13301`, convergents of its continued fraction:
`10^8651 ≤ 2^28738` and `2^13301 ≤ 10^4004`, decided on natural numbers by the kernel. -/
theorem log10_two_mem :
    (8651 / 28738 : ℝ) ≤ Real.logb 10 2 ∧ Real.logb 10 2 ≤ 4004 / 13301 := by
  have hlo : ((10 : ℕ) : ℝ) ^ 8651 ≤ ((2 : ℕ) : ℝ) ^ 28738 := by
    exact_mod_cast (by decide +kernel : (10 : ℕ) ^ 8651 ≤ 2 ^ 28738)
  have hhi : ((2 : ℕ) : ℝ) ^ 13301 ≤ ((10 : ℕ) : ℝ) ^ 4004 := by
    exact_mod_cast (by decide +kernel : (2 : ℕ) ^ 13301 ≤ 10 ^ 4004)
  push_cast at hlo hhi
  have l10 : 0 < Real.log 10 := Real.log_pos (by norm_num)
  have a := Real.log_le_log (by positivity) hlo
  have b := Real.log_le_log (by positivity) hhi
  rw [Real.log_pow, Real.log_pow] at a b
  push_cast at a b
  rw [Real.logb]
  constructor
  · rw [le_div_iff₀ l10]; linarith
  · rw [div_le_iff₀ l10]; linarith

/-- **`kStop` is `log10 2` to its five decimals, and closer**: `0.30103` is within `5e-9` of it. -/
theorem stopApd_literal : |(stopApdLiteral : ℝ) - stopApd| ≤ 5e-9 := by
  obtain ⟨h1, h2⟩ := log10_two_mem
  simp only [stopApdLiteral, stopApd, log10_real, abs_le]
  constructor <;> norm_num <;> linarith

/-- **`boot_h` is half a stop**: the C++'s `0.150515` is `kStop / 2` exactly, within `2.5e-9` of
`log10(2) / 2`. -/
theorem bootH_literal :
    (0.150515 : ℝ) = stopApdLiteral / 2 ∧ |(0.150515 : ℝ) - bootH| ≤ 2.5e-9 := by
  obtain ⟨h1, h2⟩ := log10_two_mem
  simp only [stopApdLiteral, bootH, stopApd, log10_real, abs_le]
  refine ⟨by norm_num, ?_, ?_⟩ <;> norm_num <;> linarith

/-- The printer lamp's second radiation constant `c₂ = 1.4388e-2 m K` is the value ITS-90 fixes;
the exact SI value `h c / k` (`h = 6.62607015e-34 J s`, `c = 299792458 m/s`,
`k = 1.380649e-23 J/K`) is within `2.5e-7` of it. -/
theorem planck_c2 : |(1.4388e-2 : ℝ) - 6.62607015e-34 * 299792458 / 1.380649e-23| ≤ 2.5e-7 := by
  rw [abs_le]; constructor <;> norm_num

/-! ## D50 and D55 at 1 nm -/

/-- **`kCieD50` is D50's 5 nm table read linearly, to six significant digits**: each of the C++'s
401 values lies within `5e-6` of the interpolation, relative to it. The tables are read over the
rational numbers, where their literals are exact (their real reading is the cast,
`cieD50_ratCast`), and the kernel decides the statement. -/
theorem cieD50Literal_eq :
    List.Forall₂ (fun a b : ℚ => |a - b| ≤ 5 / 10 ^ 6 * b) (cieD50Literal : Array ℚ).toList
      (cieD50 : Array ℚ).toList := by
  decide +kernel

/-- **`kCieD55` is D55's 5 nm table read linearly, to six significant digits**, as
`cieD50Literal_eq`. -/
theorem cieD55Literal_eq :
    List.Forall₂ (fun a b : ℚ => |a - b| ≤ 5 / 10 ^ 6 * b) (cieD55Literal : Array ℚ).toList
      (cieD55 : Array ℚ).toList := by
  decide +kernel

/-! ## The tables over the rationals and over the reals -/

/-- An array of rationals read as reals, entry by entry. -/
theorem getD_map_ratCast (a : Array ℚ) (i : ℕ) :
    (a.map (fun q : ℚ => (q : ℝ))).getD i 0.0 = ((a.getD i 0 : ℚ) : ℝ) := by
  by_cases h : i < a.size
  · simp [Array.getD, h]
  · simp [Array.getD, h]; norm_num

/-- Reading the 5 nm knots at 1 nm commutes with reading rationals as reals. -/
theorem at1nm_ratCast (k : Array ℚ) :
    at1nm (k.map (fun q : ℚ => (q : ℝ))) = (at1nm k).map (fun q : ℚ => (q : ℝ)) := by
  have hg : ∀ j, (k.map (fun q : ℚ => (q : ℝ))).getD j 0.0 = ((k.getD j 0.0 : ℚ) : ℝ) := by
    intro j
    by_cases h : j < k.size
    · simp [Array.getD, h]
    · simp [Array.getD, h]
  simp only [at1nm, List.map_toArray, List.map_map]
  congr 1
  apply List.map_congr_left
  intro i _
  simp only [Function.comp, hg]
  split_ifs <;> push_cast <;> rfl

set_option maxRecDepth 20000 in
theorem cieXBar_ratCast : (cieXBar : Array ℝ) = (cieXBar : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
  simp only [cieXBar, List.map_toArray, List.map_cons, List.map_nil, Rat.cast_ofScientific]

set_option maxRecDepth 20000 in
theorem cieYBar_ratCast : (cieYBar : Array ℝ) = (cieYBar : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
  simp only [cieYBar, List.map_toArray, List.map_cons, List.map_nil, Rat.cast_ofScientific]

set_option maxRecDepth 20000 in
theorem cieZBar_ratCast : (cieZBar : Array ℝ) = (cieZBar : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
  simp only [cieZBar, List.map_toArray, List.map_cons, List.map_nil, Rat.cast_ofScientific,
    Rat.cast_neg]

set_option maxRecDepth 20000 in
theorem cieD50_ratCast : (cieD50 : Array ℝ) = (cieD50 : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
  have hk : (cieD50Knots : Array ℝ) = (cieD50Knots : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
    simp only [cieD50Knots, List.map_toArray, List.map_cons, List.map_nil, Rat.cast_ofScientific]
  rw [cieD50, cieD50, hk, at1nm_ratCast]

set_option maxRecDepth 20000 in
theorem cieD55_ratCast : (cieD55 : Array ℝ) = (cieD55 : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
  have hk : (cieD55Knots : Array ℝ) = (cieD55Knots : Array ℚ).map (fun q : ℚ => (q : ℝ)) := by
    simp only [cieD55Knots, List.map_toArray, List.map_cons, List.map_nil, Rat.cast_ofScientific]
  rw [cieD55, cieD55, hk, at1nm_ratCast]

/-! ## The viewing light's white -/

/-- **The white's `X` is a ratio of two sums on the grid**: `Σ light x̄ / Σ light ȳ` over the 401
samples, weight 1 each. -/
theorem observerWhite_x (light : Array ℝ) :
    (observerWhite (observerWeights light)).x =
      (∑ i ∈ Finset.range 401, light.getD i 0.0 * cieXBar.getD i 0.0) /
        ∑ i ∈ Finset.range 401, light.getD i 0.0 * cieYBar.getD i 0.0 := by
  simp only [observerWhite, observerWeights, cieSamples]
  rw [sumRange_eq, sumRange_eq, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.mem_range] at hi
  simp [Array.getD, hi]

/-- **The white's `Z` is a ratio of two sums on the grid**: `Σ light z̄ / Σ light ȳ`. -/
theorem observerWhite_z (light : Array ℝ) :
    (observerWhite (observerWeights light)).z =
      (∑ i ∈ Finset.range 401, light.getD i 0.0 * cieZBar.getD i 0.0) /
        ∑ i ∈ Finset.range 401, light.getD i 0.0 * cieYBar.getD i 0.0 := by
  simp only [observerWhite, observerWeights, cieSamples]
  rw [sumRange_eq, sumRange_eq, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.mem_range] at hi
  simp [Array.getD, hi]

/-- The sum of products of two lists, added from the first pair, is the sum over the shorter
length of the products of their entries. -/
theorem foldl_zip_mul (a b : List ℚ) (c : ℚ) :
    (a.zip b).foldl (fun s p => s + p.1 * p.2) c =
      c + ∑ i ∈ Finset.range (min a.length b.length), a.getD i 0 * b.getD i 0 := by
  induction a generalizing b c with
  | nil => simp
  | cons x a ih =>
    cases b with
    | nil => simp
    | cons y b =>
      simp only [List.zip_cons_cons, List.foldl_cons, List.length_cons, ih,
        Nat.add_min_add_right, Finset.sum_range_succ', List.getD_cons_succ, List.getD_cons_zero]
      ring

/-- The grid sum `Σ_{i<401} a_i b_i` of two tables over the rationals, as the kernel computes it:
along the two lists, from the first pair. -/
def gridDot (a b : Array ℚ) : ℚ := (a.toList.zip b.toList).foldl (fun s p => s + p.1 * p.2) 0

/-- A grid sum of the real tables is the real reading of the rational one. -/
theorem gridDot_ratCast (a b : Array ℚ) (ha : a.size = 401) (hb : b.size = 401) :
    ∑ i ∈ Finset.range 401,
        (a.map (fun q : ℚ => (q : ℝ))).getD i 0.0 * (b.map (fun q : ℚ => (q : ℝ))).getD i 0.0 =
      ((gridDot a b : ℚ) : ℝ) := by
  have hl : ∀ (x : Array ℚ) (i : ℕ), x.getD i 0 = x.toList.getD i 0 := by
    intro x i
    by_cases h : i < x.size
    · simp [Array.getD, h, List.getD_eq_getElem?_getD]
    · simp [Array.getD, h, List.getD_eq_getElem?_getD]
  rw [gridDot, foldl_zip_mul, Array.length_toList, Array.length_toList, ha, hb, min_self]
  simp only [getD_map_ratCast, zero_add, Rat.cast_sum, Rat.cast_mul, hl]

/-- The three grid sums of D50 with the observer, exact. -/
theorem gridDot_d50 :
    gridDot cieD50 cieXBar = 2531994452209484231 / 250000000000000 ∧
    gridDot cieD50 cieYBar = 5251930921362215721 / 500000000000000 ∧
    gridDot cieD50 cieZBar =
      43329326571947590184999974409916187661 / 5000000000000000000000000000000000 := by
  refine ⟨?_, ?_, ?_⟩ <;> decide +kernel

/-- **The D50 viewing light's white**, from the sums on the grid: `X` is `0.9642147` and `Z` is
`0.8250171` (to seven decimals), `Y` is 1. -/
theorem observerWhite_d50 :
    0.964214 < (observerWhite (observerWeights (cieD50 : Array ℝ))).x ∧
    (observerWhite (observerWeights (cieD50 : Array ℝ))).x < 0.964215 ∧
    0.825017 < (observerWhite (observerWeights (cieD50 : Array ℝ))).z ∧
    (observerWhite (observerWeights (cieD50 : Array ℝ))).z < 0.825018 := by
  obtain ⟨hx, hy, hz⟩ := gridDot_d50
  have s50 : (cieD50 : Array ℚ).size = 401 := by decide +kernel
  have sx : (cieXBar : Array ℚ).size = 401 := by decide +kernel
  have sy : (cieYBar : Array ℚ).size = 401 := by decide +kernel
  have sz : (cieZBar : Array ℚ).size = 401 := by decide +kernel
  rw [observerWhite_x, observerWhite_z, cieD50_ratCast, cieXBar_ratCast, cieYBar_ratCast,
    cieZBar_ratCast, gridDot_ratCast _ _ s50 sx, gridDot_ratCast _ _ s50 sy,
    gridDot_ratCast _ _ s50 sz, hx, hy, hz]
  norm_num

/-- **The D50 white the print computes is not the PCS illuminant**: its `Z` exceeds `iccD50`'s
`0.8249` by more than `1e-4`. A paper's print ICC takes the print's XYZ to the PCS unadapted
(`pcsAdaptation true`), then scales each channel to the paper's white (`toPaperWhite`). -/
theorem observerWhite_d50_ne_iccD50 :
    (iccD50 : Vec3 ℝ).z + 1e-4 < (observerWhite (observerWeights (cieD50 : Array ℝ))).z := by
  have := observerWhite_d50.2.2.1
  simp only [iccD50]
  linarith

end Negicc
