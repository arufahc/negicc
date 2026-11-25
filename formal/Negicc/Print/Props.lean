/-
Proved properties of the print emulation over the real numbers.
-/
module

public import Negicc.Base
public import Negicc.Scene
public import Negicc.Print.Defs

@[expose] public section

/-!
# Properties of the print emulation

Statements over the real numbers about the definitions of `Negicc.Print`, the ones a caller relies
on and the derivations of DESIGN.md 5.11:

* **The print matrix** (`printMatrix_mul_m`): `P M = Q L` for `M = F L diag(φ)`, so the chart's
  calibration of Status M belongs to the Status M readout alone and the APD slopes apply once; with
  no factors `P = Q F⁻¹`.
* **The colour matrices**: the CLF's matrix takes the viewing light's white to sRGB `(1, 1, 1)`
  (`toSrgbMatrix_white`), and the Bradford adaptation takes a print film's light to D50
  (`pcsAdaptation_white`); the observer gives the light `Y = 1` (`observerWhite_y`).
* **The table's grey axis** (`table_grey_axis`): with `Q L = P M` and `ψ = M⁻¹ (1, 1, 1)`, a grey
  `alpha` above the mid grey in every channel reads the printing density `0.70 + alpha η`,
  `η = P (1, 1, 1)`: the `η` of `Paper_c`.
* **Paper_c and the input curves**: `Paper_c` is strictly decreasing and spans `[0, 1]`
  (`paperChannel_strictAnti`, `paperChannel_mem`); the input curves are strictly increasing
  (`bracketPrintCurves_strictMono`).
* **The knee's inverse** (`kneeCompress_unknee`): the print's curves undo the soft knee up to its
  cap, 0.9999; the film base at `E = g = b = 1` has light density 0 (`inputLightDensity_base`).
* **The tilt** (`tiltSegment_left`, `tiltSegment_right`, …): at a bracket's `A_mid` it is that
  bracket's tilt, held beyond the ends; of two brackets at the same `A_mid` the first in the bundle
  gives it (`bracketOrder_stable`, `tiltSegment_tie`).
* **The print's white** (`toPaperWhite_y_le`, `toPaperWhite_base`): after the scaling no node's
  `Y` passes the PCS white's and the paper's base is the PCS white.
* **Density** (`lightDensity_one`, `lightDensity_le_four`, `densityBalanceOffset_sum`, …): bare
  film is density 0 and code `offset`, the floor is density 4, the balance offsets sum to zero.
* **The CLF** (`clfPrintingDensity_slope_form`, `clf_mid_grey`, `exposureOf_shift`): its first
  matrix is `0.70 + S P (D + balance − grey)`; the balanced mid grey prints at 0.70; a constant
  added to the offsets moves the print's exposure by that constant; the 0.70 cancels
  (`clf_grey_density_cancels`); the slopes give the grey scale equal contrast
  (`curveSlopesAt_equal_contrast`).
* **The film base's floor** (`printFloorFactor_ge_one`, `printFloorFactor_ge_channel`).
* **The search** (`printSearch_a_mem`, `solvePrintKnobs_a_mem`, `bisectLevel_spec`, …): every
  `A` the search and the knobs keep lies in the curve's range; the metering bisection halves its
  interval and keeps the level between its ends; the held exposure is the result's 1.
* **Aim** (`aimSolve_residual`): the mid grey 0.70 prints within `1e-6` of an 18 % grey.
-/

namespace Negicc

open Scalar

/-! ## 3x3 matrices -/

namespace Mat3

variable (a b c : Mat3 ℝ)

theorem vadd_x (u v : Vec3 ℝ) : (u + v).x = u.x + v.x := rfl
theorem vadd_y (u v : Vec3 ℝ) : (u + v).y = u.y + v.y := rfl
theorem vadd_z (u v : Vec3 ℝ) : (u + v).z = u.z + v.z := rfl
theorem vsub_x (u v : Vec3 ℝ) : (u - v).x = u.x - v.x := rfl
theorem vsub_y (u v : Vec3 ℝ) : (u - v).y = u.y - v.y := rfl
theorem vsub_z (u v : Vec3 ℝ) : (u - v).z = u.z - v.z := rfl

theorem mul_assoc : (a.mul b).mul c = a.mul (b.mul c) := by
  ext <;> simp only [mul, Vec3.dot, col0, col1, col2] <;> ring

theorem add_mul : (a.add b).mul c = (a.mul c).add (b.mul c) := by
  ext <;> simp only [mul, add, Vec3.dot, col0, col1, col2, vadd_x, vadd_y, vadd_z] <;> ring

theorem mul_add : a.mul (b.add c) = (a.mul b).add (a.mul c) := by
  ext <;> simp only [mul, add, Vec3.dot, col0, col1, col2, vadd_x, vadd_y, vadd_z] <;> ring

theorem mul_identity : a.mul identity = a := by
  ext <;> simp only [mul, identity, diag, Vec3.fill, Vec3.dot, col0, col1, col2] <;> norm_num

theorem identity_mul : identity.mul a = a := by
  ext <;> simp only [mul, identity, diag, Vec3.fill, Vec3.dot, col0, col1, col2] <;> norm_num

theorem scaleRows_eq_diag_mul (s : Vec3 ℝ) : scaleRows s a = (diag s).mul a := by
  ext <;> simp only [scaleRows, diag, mul, Vec3.smul, Vec3.dot, col0, col1, col2] <;> norm_num

theorem diag_mul_diag (s t : Vec3 ℝ) :
    (diag s).mul (diag t) = diag ⟨s.x * t.x, s.y * t.y, s.z * t.z⟩ := by
  ext <;> simp only [diag, mul, Vec3.dot, col0, col1, col2] <;> norm_num

theorem diag_add (s t : Vec3 ℝ) : (diag s).add (diag t) = diag ⟨s.x + t.x, s.y + t.y, s.z + t.z⟩ := by
  ext <;> simp only [diag, add, vadd_x, vadd_y, vadd_z] <;> norm_num

theorem identity_eq_diag_one : (identity : Mat3 ℝ) = diag ⟨1, 1, 1⟩ := by
  simp only [identity, Vec3.fill]; norm_num

/-- The determinant expanded along the first row, as `invert3` writes it. -/
theorem det_eq : det a = a.row0.x * (a.row1.y * a.row2.z - a.row1.z * a.row2.y) -
    a.row0.y * (a.row1.x * a.row2.z - a.row1.z * a.row2.x) +
    a.row0.z * (a.row1.x * a.row2.y - a.row1.y * a.row2.x) := by
  simp [det, adjugate, entry]
  ring

/-- The adjugate over a non-zero determinant is a right inverse. -/
theorem mul_adjInv (h : det a ≠ 0) : a.mul (adjInv a) = identity := by
  ext <;> simp only [mul, adjInv, identity, diag, Vec3.fill, Vec3.dot, col0, col1, col2, mul_div_assoc'] <;>
    rw [← add_div, ← add_div, div_eq_iff h] <;> rw [det_eq] <;> norm_num <;> ring

/-- The adjugate over a non-zero determinant is a left inverse. -/
theorem adjInv_mul (h : det a ≠ 0) : (adjInv a).mul a = identity := by
  ext <;> simp only [mul, adjInv, identity, diag, Vec3.fill, Vec3.dot, col0, col1, col2, div_mul_eq_mul_div] <;>
    rw [← add_div, ← add_div, div_eq_iff h] <;> rw [det_eq] <;> norm_num <;> ring

theorem adjInv_mul_cancel_left (h : det a ≠ 0) : (adjInv a).mul (a.mul b) = b := by
  rw [← mul_assoc, adjInv_mul a h, identity_mul]

theorem adjInv_mulVec_mulVec (h : det a ≠ 0) (v : Vec3 ℝ) : (adjInv a).mulVec (a.mulVec v) = v := by
  rw [← mulVec_mul, adjInv_mul a h, mulVec_identity]

theorem mulVec_adjInv_mulVec (h : det a ≠ 0) (v : Vec3 ℝ) : a.mulVec ((adjInv a).mulVec v) = v := by
  rw [← mulVec_mul, mul_adjInv a h, mulVec_identity]

theorem rowSums_eq_mulVec : a.rowSums = a.mulVec ⟨1, 1, 1⟩ := by
  ext <;> simp only [rowSums, mulVec, Vec3.dot] <;> ring

theorem mulVec_smul (k : ℝ) (v : Vec3 ℝ) :
    a.mulVec ⟨k * v.x, k * v.y, k * v.z⟩ = ⟨k * (a.mulVec v).x, k * (a.mulVec v).y, k * (a.mulVec v).z⟩ := by
  ext <;> simp only [mulVec, Vec3.dot] <;> ring

/-- `invert3` returns the adjugate inverse whenever it returns. -/
theorem invert3_eq_some {a i : Mat3 ℝ} (h : invert3 a = some i) : i = adjInv a ∧ det a ≠ 0 := by
  unfold invert3 at h
  split_ifs at h with hd
  injection h with h
  subst h
  refine ⟨rfl, fun h0 => hd ?_⟩
  rw [h0, fabs_real]; norm_num

end Mat3

/-! ## The print matrix (DESIGN.md 5.11) -/

theorem allOne_eq {v : Vec3 ℝ} (h : allOne v = true) : v = ⟨1, 1, 1⟩ := by
  simp only [allOne, decide_eq_true_eq] at h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  ext <;> norm_num at * <;> linarith

/-- **The print matrix.** With `M = F L diag(φ)`, the print matrix
`P = Q F⁻¹ + Q L diag(φ⁻¹ − 1) L⁻¹ F⁻¹` satisfies `P M = Q L`: it is `Q L M⁻¹`, so a calibrated
Status M file is printed as an uncalibrated one, and the APD slopes apply once. -/
theorem printMatrix_mul_m {q p : Mat3 ℝ} {dm : DensityMatrices ℝ}
    (hM : dm.m = (dm.filmToMeasure.mul dm.lightToFilm).mul (Mat3.diag dm.factors))
    (hφ : dm.factors.x ≠ 0 ∧ dm.factors.y ≠ 0 ∧ dm.factors.z ≠ 0)
    (hp : printMatrix q dm = some p) : p.mul dm.m = q.mul dm.lightToFilm := by
  unfold printMatrix at hp
  rcases hF : Mat3.invert3 dm.filmToMeasure with _ | invF
  · rw [hF] at hp; exact absurd hp (by simp)
  rw [hF] at hp
  obtain ⟨rfl, hdF⟩ := Mat3.invert3_eq_some hF
  dsimp only at hp
  rw [hM]
  by_cases h1 : allOne dm.factors = true
  · simp only [h1, ↓reduceIte] at hp
    injection hp with hp
    subst hp
    rw [allOne_eq h1, ← Mat3.identity_eq_diag_one, Mat3.mul_identity, Mat3.mul_assoc,
      Mat3.adjInv_mul_cancel_left _ _ hdF]
  · simp only [h1, Bool.false_eq_true, ↓reduceIte] at hp
    rcases hL : Mat3.invert3 dm.lightToFilm with _ | invL
    · rw [hL] at hp; exact absurd hp (by simp)
    rw [hL] at hp
    obtain ⟨rfl, hdL⟩ := Mat3.invert3_eq_some hL
    dsimp only at hp
    injection hp with hp
    subst hp
    obtain ⟨hx, hy, hz⟩ := hφ
    have hd : (Mat3.diag dm.factors).add
        ((Mat3.diag (dm.factors.map fun a => 1.0 / a - 1.0)).mul (Mat3.diag dm.factors)) = Mat3.identity := by
      rw [Mat3.diag_mul_diag, Mat3.diag_add, Mat3.identity_eq_diag_one]
      simp only [Vec3.map]
      congr 1
      ext <;> norm_num <;> field_simp <;> ring
    simp only [Mat3.add_mul, Mat3.scaleRows_eq_diag_mul, Mat3.mul_assoc,
      Mat3.adjInv_mul_cancel_left _ _ hdF, Mat3.adjInv_mul_cancel_left _ _ hdL]
    rw [← Mat3.mul_add, ← Mat3.mul_add, hd, Mat3.mul_identity]

/-- Without light density factors the print matrix is `Q F⁻¹`. -/
theorem printMatrix_of_allOne {q : Mat3 ℝ} {dm : DensityMatrices ℝ} (h1 : allOne dm.factors = true)
    (hF : 1.0e-12 ≤ fabs (Mat3.det dm.filmToMeasure)) :
    printMatrix q dm = some (q.mul (Mat3.adjInv dm.filmToMeasure)) := by
  have : Mat3.invert3 dm.filmToMeasure = some (Mat3.adjInv dm.filmToMeasure) := by
    unfold Mat3.invert3; simp only [not_lt.mpr hF, ↓reduceIte]
  unfold printMatrix; rw [this]; dsimp only; simp only [h1, ↓reduceIte]

/-! ## The colour matrices -/

theorem det_cat02 : Mat3.det (cat02 : Mat3 ℝ) ≠ 0 := by
  rw [Mat3.det_eq]; simp only [cat02]; norm_num

theorem det_xyzToSrgbLindbloom : Mat3.det (xyzToSrgbLindbloom : Mat3 ℝ) ≠ 0 := by
  rw [Mat3.det_eq]; simp only [xyzToSrgbLindbloom]; norm_num

theorem det_bradford : Mat3.det (bradford : Mat3 ℝ) ≠ 0 := by
  rw [Mat3.det_eq]; simp only [bradford]; norm_num

theorem scaleRowsBy_mulVec (m : Mat3 ℝ) (num den v : Vec3 ℝ)
    (h : (m.mulVec v).x ≠ 0 ∧ (m.mulVec v).y ≠ 0 ∧ (m.mulVec v).z ≠ 0) (hden : den = m.mulVec v) :
    (Mat3.scaleRowsBy m num den).mulVec v = num := by
  subst hden
  obtain ⟨hx, hy, hz⟩ := h
  simp only [Mat3.mulVec, Vec3.dot] at hx hy hz
  ext <;> simp only [Mat3.scaleRowsBy, Mat3.mulVec, Vec3.dot] <;> field_simp

/-- **The CLF's colour matrix takes the viewing light's white to sRGB white.** -/
theorem toSrgbMatrix_white (w : Vec3 ℝ)
    (hw : (cat02.mulVec w).x ≠ 0 ∧ (cat02.mulVec w).y ≠ 0 ∧ (cat02.mulVec w).z ≠ 0) :
    (toSrgbMatrix w).mulVec w = ⟨1, 1, 1⟩ := by
  simp only [toSrgbMatrix, Mat3.mulVec_mul]
  rw [scaleRowsBy_mulVec _ _ _ _ hw rfl, Mat3.adjInv_mulVec_mulVec _ det_cat02, Mat3.rowSums_eq_mulVec,
    Mat3.mulVec_adjInv_mulVec _ det_xyzToSrgbLindbloom]

/-- **The print ICC's adaptation takes a print film's light to D50**: its white, normalised to
`Y = 1`, lands on the PCS's D50. -/
theorem pcsAdaptation_white (w : Vec3 ℝ)
    (h : ∀ src : Vec3 ℝ, src = ⟨w.x / w.y, 1.0, w.z / w.y⟩ →
      (bradford.mulVec src).x ≠ 0 ∧ (bradford.mulVec src).y ≠ 0 ∧ (bradford.mulVec src).z ≠ 0) :
    (pcsAdaptation false w).mulVec ⟨w.x / w.y, 1.0, w.z / w.y⟩ = iccD50 := by
  obtain ⟨hx, hy, hz⟩ := h _ rfl
  simp only [pcsAdaptation, Bool.false_eq_true, ↓reduceIte, Mat3.mulVec_mul]
  set u := bradford.mulVec (⟨w.x / w.y, 1.0, w.z / w.y⟩ : Vec3 ℝ) with hu
  set t := (bradford : Mat3 ℝ).mulVec iccD50 with ht
  have hg : (⟨t.x / u.x * u.x, t.y / u.y * u.y, t.z / u.z * u.z⟩ : Vec3 ℝ) = t := by
    ext <;> field_simp
  conv_rhs => rw [← Mat3.adjInv_mulVec_mulVec _ det_bradford iccD50, ← ht, ← hg]
  ext <;> simp only [Mat3.mulVec, Vec3.dot] <;> ring

theorem sumRange_eq (n : Nat) (f : Nat → ℝ) : sumRange n f = ∑ i ∈ Finset.range n, f i := by
  unfold sumRange
  induction n with
  | zero => simp; norm_num
  | succ n ih => rw [List.range_succ, List.foldl_append, ih, Finset.sum_range_succ]; simp

/-- **The observer gives the viewing light `Y = 1`.** -/
theorem observerWhite_y (light : Array ℝ)
    (h : sumRange cieSamples (fun i => light.getD i 0.0 * cieYBar.getD i 0.0) ≠ 0) :
    (observerWhite (observerWeights light)).y = 1 := by
  simp only [observerWhite, observerWeights]
  rw [sumRange_eq] at h ⊢
  rw [Finset.sum_congr rfl (g := fun i => light.getD i 0.0 * cieYBar.getD i 0.0 /
      ∑ i ∈ Finset.range cieSamples, light.getD i 0.0 * cieYBar.getD i 0.0)]
  · rw [← Finset.sum_div, div_self h]
  · intro i hi
    rw [Finset.mem_range] at hi
    simp [Array.getD, hi, sumRange_eq]

/-! ## The table's grey axis -/

/-- **A grey reads the same printing density in the table as in `Paper_c`.** With `Q L = P M` and
`ψ = M⁻¹ (1, 1, 1)`, a grey `alpha` above the mid grey in every channel has the printing density
`0.70 + alpha η`, `η = P (1, 1, 1)` the factor `Paper_c` exposes with. -/
theorem table_grey_axis (m : PrintMaterial ℝ) (pApd mApd pcs : Mat3 ℝ) (h : Mat3.det mApd ≠ 0) (a : ℝ) :
    tableNode m (pApd.mul mApd) pcs (Mat3.adjInv mApd).rowSums ⟨a, a, a⟩ =
      pcs.mulVec (printOfDensity m ⟨printGreyDensity + a * (paperEta pApd).x,
        printGreyDensity + a * (paperEta pApd).y, printGreyDensity + a * (paperEta pApd).z⟩) := by
  have key : (pApd.mul mApd).mulVec ⟨a * (Mat3.adjInv mApd).rowSums.x, a * (Mat3.adjInv mApd).rowSums.y,
      a * (Mat3.adjInv mApd).rowSums.z⟩ = ⟨a * (paperEta pApd).x, a * (paperEta pApd).y, a * (paperEta pApd).z⟩ := by
    rw [Mat3.mulVec_smul, Mat3.rowSums_eq_mulVec, Mat3.mulVec_mul, Mat3.mulVec_adjInv_mulVec _ h,
      paperEta, Mat3.rowSums_eq_mulVec]
  simp only [tableNode]
  congr 2
  simp only [Mat3.mulVec] at key
  injection key with k1 k2 k3
  ext
  · show printGreyDensity + _ = _; rw [← k1]
  · show printGreyDensity + _ = _; rw [← k2]
  · show printGreyDensity + _ = _; rw [← k3]

/-! ## Paper_c and the input curves -/

theorem runMinList_le (r : ℝ) (l : List ℝ) : ∀ y ∈ runMinList (some r) l, y ≤ r := by
  induction l generalizing r with
  | nil => simp [runMinList]
  | cons v vs ih =>
    intro y hy
    simp only [runMinList, List.mem_cons] at hy
    rcases hy with rfl | hy
    · exact min_le_left _ _
    · exact (ih _ y hy).trans (min_le_left _ _)

theorem runMinList_pairwise (r : Option ℝ) (l : List ℝ) :
    (runMinList r l).Pairwise (fun a b => b ≤ a) := by
  induction l generalizing r with
  | nil => simp [runMinList]
  | cons v vs ih =>
    simp only [runMinList, List.pairwise_cons]
    exact ⟨fun y hy => runMinList_le _ vs y hy, ih _⟩

theorem runMinList_ge_some (c r : ℝ) (hcr : c ≤ r) (l : List ℝ) (hl : ∀ x ∈ l, c ≤ x) :
    ∀ y ∈ runMinList (some r) l, c ≤ y := by
  induction l generalizing r with
  | nil => simp [runMinList]
  | cons v vs ih =>
    have hm : c ≤ min r v := le_min hcr (hl v (by simp))
    intro y hy
    simp only [runMinList, List.mem_cons] at hy
    rcases hy with rfl | hy
    · exact hm
    · exact ih _ hm (fun x hx => hl x (by simp [hx])) y hy

theorem runMinList_ge (c : ℝ) (l : List ℝ) (hl : ∀ x ∈ l, c ≤ x) : ∀ y ∈ runMinList none l, c ≤ y := by
  cases l with
  | nil => simp [runMinList]
  | cons v vs =>
    have hv : c ≤ v := hl v (by simp)
    intro y hy
    simp only [runMinList, List.mem_cons] at hy
    rcases hy with rfl | hy
    · exact hv
    · exact runMinList_ge_some c v hv vs (fun x hx => hl x (by simp [hx])) y hy

theorem runMinList_length (r : Option ℝ) (l : List ℝ) : (runMinList r l).length = l.length := by
  induction l generalizing r with
  | nil => simp [runMinList]
  | cons v vs ih => simp [runMinList, ih]

theorem runMaxList_ge (r : ℝ) (l : List ℝ) : ∀ y ∈ runMaxList (some r) l, r ≤ y := by
  induction l generalizing r with
  | nil => simp [runMaxList]
  | cons v vs ih =>
    intro y hy
    simp only [runMaxList, List.mem_cons] at hy
    rcases hy with rfl | hy
    · exact le_max_left _ _
    · exact (le_max_left _ _).trans (ih _ y hy)

theorem runMaxList_pairwise (r : Option ℝ) (l : List ℝ) :
    (runMaxList r l).Pairwise (fun a b => a ≤ b) := by
  induction l generalizing r with
  | nil => simp [runMaxList]
  | cons v vs ih =>
    simp only [runMaxList, List.pairwise_cons]
    exact ⟨fun y hy => runMaxList_ge _ vs y hy, ih _⟩

theorem runMaxList_length (r : Option ℝ) (l : List ℝ) : (runMaxList r l).length = l.length := by
  induction l generalizing r with
  | nil => simp [runMaxList]
  | cons v vs ih => simp [runMaxList, ih]

theorem runningMin_size (xs : Array ℝ) : (runningMin xs).size = xs.size := by
  simp [runningMin, runMinList_length]

theorem runningMax_size (xs : Array ℝ) : (runningMax xs).size = xs.size := by
  simp [runningMax, runMaxList_length]

/-- The running minimum does not increase. -/
theorem runningMin_anti (xs : Array ℝ) {i j : Nat} (hij : i ≤ j) (hj : j < xs.size) :
    (runningMin xs).getD j 0.0 ≤ (runningMin xs).getD i 0.0 := by
  have hl := runMinList_length none xs.toList
  have hj' : j < (runMinList none xs.toList).length := by rw [hl]; simpa using hj
  have hi' : i < (runMinList none xs.toList).length := by omega
  simp only [runningMin, Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_eq_getElem hj',
    List.getElem?_eq_getElem hi', Option.getD_some]
  rcases Nat.lt_or_eq_of_le hij with h | rfl
  · exact List.pairwise_iff_getElem.mp (runMinList_pairwise none xs.toList) i j hi' hj' h
  · exact le_refl _

/-- The running maximum does not decrease. -/
theorem runningMax_mono (xs : Array ℝ) {i j : Nat} (hij : i ≤ j) (hj : j < xs.size) :
    (runningMax xs).getD i 0.0 ≤ (runningMax xs).getD j 0.0 := by
  have hl := runMaxList_length none xs.toList
  have hj' : j < (runMaxList none xs.toList).length := by rw [hl]; simpa using hj
  have hi' : i < (runMaxList none xs.toList).length := by omega
  simp only [runningMax, Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_eq_getElem hj',
    List.getElem?_eq_getElem hi', Option.getD_some]
  rcases Nat.lt_or_eq_of_le hij with h | rfl
  · exact List.pairwise_iff_getElem.mp (runMaxList_pairwise none xs.toList) i j hi' hj' h
  · exact le_refl _

/-- The running minimum of non-negative values is non-negative. -/
theorem runningMin_nonneg (xs : Array ℝ) (h : ∀ x ∈ xs.toList, 0 ≤ x) {i : Nat} (hi : i < xs.size) :
    0 ≤ (runningMin xs).getD i 0.0 := by
  have hl := runMinList_length none xs.toList
  have hi' : i < (runMinList none xs.toList).length := by rw [hl]; simpa using hi
  simp only [runningMin, Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_eq_getElem hi',
    Option.getD_some]
  exact runMinList_ge 0 xs.toList h _ (List.getElem_mem hi')

theorem getD_map_range {β : Type} (n i : Nat) (f : Nat → β) (d : β) (h : i < n) :
    ((Array.range n).map f).getD i d = f i := by
  simp [Array.getD, h]

/-- **`Paper_c` is strictly decreasing** (DESIGN.md 5.4), so the table inverts it the same way for
every `A`: the running minimum does not increase and the 0.25 ramp strictly decreases. `top`, the
first dye amount, is positive. -/
theorem paperChannel_strictAnti (dye : Array ℝ) (htop : 0 < (runningMin (dye.map (max 0.0))).getD 0 0.0)
    {i j : Nat} (hij : i < j) (hj : j < dye.size) :
    (paperChannel dye).getD j 0.0 < (paperChannel dye).getD i 0.0 := by
  have hs : (runningMin (dye.map (max 0.0))).size = dye.size := by rw [runningMin_size]; simp
  simp only [paperChannel]
  rw [getD_map_range _ j _ _ (by rw [hs]; exact hj), getD_map_range _ i _ _ (by rw [hs]; omega)]
  have hanti := runningMin_anti (dye.map (max 0.0)) hij.le (by simpa using hj)
  have hdiv : (runningMin (dye.map (max 0.0))).getD j 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 ≤
      (runningMin (dye.map (max 0.0))).getD i 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 :=
    div_le_div_of_nonneg_right hanti htop.le
  have hij' : (i : ℝ) < j := by exact_mod_cast hij
  generalize (runningMin (dye.map (max 0.0))).getD j 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 = qj at *
  generalize (runningMin (dye.map (max 0.0))).getD i 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 = qi at *
  have h660 : (((paperAlphaPoints - 1 : Nat) : ℝ)) = 660 := by norm_num [paperAlphaPoints]
  rw [h660]
  norm_num
  nlinarith [hdiv, hij']

/-- **`Paper_c` spans `[0, 1]`**: every value lies in it (the grid has 661 points). -/
theorem paperChannel_mem (dye : Array ℝ) (htop : 0 < (runningMin (dye.map (max 0.0))).getD 0 0.0)
    (hsize : dye.size ≤ 661) {i : Nat} (hi : i < dye.size) :
    0 ≤ (paperChannel dye).getD i 0.0 ∧ (paperChannel dye).getD i 0.0 ≤ 1 := by
  have hs : (runningMin (dye.map (max 0.0))).size = dye.size := by rw [runningMin_size]; simp
  simp only [paperChannel]
  rw [getD_map_range _ _ _ _ (by rw [hs]; exact hi)]
  have hle := runningMin_anti (dye.map (max 0.0)) (Nat.zero_le i) (by simpa using hi)
  have hnn := runningMin_nonneg (dye.map (max 0.0)) (by
      intro x hx; rw [Array.toList_map, List.mem_map] at hx; obtain ⟨y, _, rfl⟩ := hx
      show (0:ℝ) ≤ max (0.0:ℝ) y; norm_num) (i := i) (by simpa using hi)
  have h1 : (runningMin (dye.map (max 0.0))).getD i 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 ≤ 1 :=
    (div_le_one htop).mpr hle
  have h0 : 0 ≤ (runningMin (dye.map (max 0.0))).getD i 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 :=
    div_nonneg hnn htop.le
  have hi' : (i : ℝ) ≤ 660 := by exact_mod_cast (by omega : i ≤ 660)
  have hi0 : (0 : ℝ) ≤ i := Nat.cast_nonneg i
  generalize (runningMin (dye.map (max 0.0))).getD i 0.0 / (runningMin (dye.map (max 0.0))).getD 0 0.0 = q at *
  have h660 : (((paperAlphaPoints - 1 : Nat) : ℝ)) = 660 := by norm_num [paperAlphaPoints]
  rw [h660]
  norm_num
  constructor <;> nlinarith

/-- The first value of `Paper_c` is 1. -/
theorem paperChannel_first (dye : Array ℝ) (htop : 0 < (runningMin (dye.map (max 0.0))).getD 0 0.0) :
    (paperChannel dye).getD 0 0.0 = 1 := by
  have hpos : 0 < (runningMin (dye.map (max 0.0))).size := by
    by_contra h
    simp only [not_lt, Nat.le_zero] at h
    simp [Array.getD, h] at htop
    norm_num at htop
  simp only [paperChannel]
  rw [getD_map_range _ _ _ _ hpos, div_self htop.ne']
  norm_num

/-- **The input curves are strictly increasing** (`bracket_print_curves`): a running maximum plus
`1e-9 i`. -/
theorem rampedRunMax_strictMono (xs : Array ℝ) {i j : Nat} (hij : i < j) (hj : j < xs.size) :
    (rampedRunMax xs).getD i 0.0 < (rampedRunMax xs).getD j 0.0 := by
  simp only [rampedRunMax]
  rw [getD_map_range _ i _ _ (by rw [runningMax_size]; omega),
    getD_map_range _ j _ _ (by rw [runningMax_size]; exact hj)]
  have h := runningMax_mono xs hij.le hj
  have hij' : (i : ℝ) < j := by exact_mod_cast hij
  generalize (runningMax xs).getD i 0.0 = ri at *
  generalize (runningMax xs).getD j 0.0 = rj at *
  norm_num
  nlinarith

theorem inputCurveRaw_size (g : GreyBalance ℝ) (apd paper : Array ℝ) (a n0 tilt gain : ℝ) :
    (inputCurveRaw g apd paper a n0 tilt gain).size = printIccPoints := by
  simp [inputCurveRaw]

/-- Each of a print's three input curves is strictly increasing on its 4096 entries. -/
theorem bracketPrintCurves_strictMono (g : GreyBalance ℝ) (paper : Vec3 (Array ℝ)) (a n0 : ℝ)
    (tilt gains : Vec3 ℝ) {i j : Nat} (hij : i < j) (hj : j < printIccPoints) :
    (bracketPrintCurves g paper a n0 tilt gains).x.getD i 0.0 < (bracketPrintCurves g paper a n0 tilt gains).x.getD j 0.0 ∧
    (bracketPrintCurves g paper a n0 tilt gains).y.getD i 0.0 < (bracketPrintCurves g paper a n0 tilt gains).y.getD j 0.0 ∧
    (bracketPrintCurves g paper a n0 tilt gains).z.getD i 0.0 < (bracketPrintCurves g paper a n0 tilt gains).z.getD j 0.0 := by
  simp only [bracketPrintCurves]
  refine ⟨?_, ?_, ?_⟩ <;> exact rampedRunMax_strictMono _ hij (by rw [inputCurveRaw_size]; exact hj)

/-! ## The knee's inverse -/

/-- `tanh (atanh y) = y` on `(−1, 1)`. -/
theorem tanh_atanh {y : ℝ} (h0 : -1 < y) (h1 : y < 1) : Real.tanh (atanh y) = y := by
  have hr : 0 < (1 + y) / (1 - y) := div_pos (by linarith) (by linarith)
  simp only [atanh, log_real]
  rw [show (1.0:ℝ) = 1 by norm_num]
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq, Real.exp_neg]
  set x := (0.5:ℝ) * Real.log ((1 + y) / (1 - y)) with hxdef
  have hx2 : Real.exp x * Real.exp x = (1 + y) / (1 - y) := by
    rw [← Real.exp_add]
    have : x + x = Real.log ((1 + y) / (1 - y)) := by rw [hxdef]; norm_num; ring
    rw [this, Real.exp_log hr]
  have hpos := Real.exp_pos x
  have h1y : (1 - y) ≠ 0 := by linarith
  have key : (Real.exp x * Real.exp x - 1) = y * (Real.exp x * Real.exp x + 1) := by
    rw [hx2]; field_simp; ring
  rw [div_div_div_cancel_right₀ (two_ne_zero), div_eq_iff (by positivity)]
  field_simp
  linear_combination key

/-- `atanh` is positive on `(0, 1)`. -/
theorem atanh_pos {y : ℝ} (h0 : 0 < y) (h1 : y < 1) : 0 < atanh y := by
  simp only [atanh, log_real]
  apply mul_pos (by norm_num)
  apply Real.log_pos
  rw [one_lt_div (by norm_num; linarith)]
  norm_num; linarith

/-- **The print's curves undo the soft knee** up to its cap: for `0 < x ≤ 0.9999` the knee of
the inverse is `x`. So film thinner than the base by up to `unknee(0.9999) − 1` reaches the print
as its own density. -/
theorem kneeCompress_unknee {x : ℝ} (h0 : 0 < x) (h1 : x ≤ 0.9999) :
    kneeCompress printKnee (unknee printKnee x) = x := by
  simp only [printKnee]
  by_cases hk : x ≤ 0.95
  · simp only [unknee, hk, ↓reduceIte, kneeCompress]
    have : ¬ (x ≤ 0.0) := by norm_num; linarith
    have h2 : ¬ ((1.0 : ℝ) ≤ 0.95) := by norm_num
    simp only [this, h2, ↓reduceIte]
    exact min_eq_right (by norm_num; linarith)
  · have hk' : (0.95 : ℝ) < x := lt_of_not_ge hk
    obtain ⟨z, hz⟩ : ∃ z : ℝ, z = (x - 0.95) / (1.0 - 0.95) := ⟨_, rfl⟩
    have hz0 : 0 < z := by rw [hz]; norm_num; linarith
    have hz1 : z ≤ 0.998 := by rw [hz]; norm_num; linarith
    have hzc : min (max z 0.0) 0.998 = z := by
      rw [max_eq_left (by norm_num; linarith), min_eq_left hz1]
    have hat := atanh_pos hz0 (by norm_num at hz1 ⊢; linarith)
    simp only [unknee, hk, ↓reduceIte, kneeCompress]
    rw [← hz, hzc]
    have hu0 : ¬ (0.95 + (1.0 - 0.95) * atanh z ≤ (0.0 : ℝ)) := by norm_num; nlinarith
    have hu1 : ¬ ((1.0 : ℝ) ≤ 0.95) := by norm_num
    have hu2 : ¬ (0.95 + (1.0 - 0.95) * atanh z ≤ (0.95 : ℝ)) := by norm_num; nlinarith
    simp only [hu0, hu1, hu2, ↓reduceIte, tanh_real]
    have hd : (0.95 + (1.0 - 0.95) * atanh z - 0.95) / (1.0 - 0.95) = atanh z := by
      norm_num
    rw [hd, tanh_atanh (by linarith) (by norm_num at hz1 ⊢; linarith), hz]
    rw [show (0.95 : ℝ) + (1.0 - 0.95) * ((x - 0.95) / (1.0 - 0.95)) = x by norm_num; ring]
    exact min_eq_right h1

/-- **The film base has light density 0** in the print's curves at `E = g = b = 1`: the input `n0`
(below the knee) is the base's own level. -/
theorem inputLightDensity_base {n0 : ℝ} (h0 : 1.0e-7 ≤ n0) (h1 : n0 ≤ 0.95) :
    inputLightDensity n0 1.0 n0 = 0 := by
  have hn : 0 < n0 := lt_of_lt_of_le (by norm_num) h0
  simp only [inputLightDensity, unknee, printKnee, h1, ↓reduceIte]
  rw [show n0 * (1.0 : ℝ) = n0 by norm_num, max_eq_left h0, div_self hn.ne', log10_real, Real.logb_one]
  norm_num

/-! ## The tilt between brackets -/

theorem tiltSegment_left {ai aj : ℝ} (h : ai < aj) (ti tj : Vec3 ℝ) : tiltSegment ai aj ti tj ai = ti := by
  simp only [tiltSegment, h, ↓reduceIte, sub_self, zero_div, clamp_real]
  ext <;> norm_num

theorem tiltSegment_right {ai aj : ℝ} (h : ai < aj) (ti tj : Vec3 ℝ) : tiltSegment ai aj ti tj aj = tj := by
  simp only [tiltSegment, h, ↓reduceIte, div_self (sub_pos.mpr h).ne', clamp_real]
  ext <;> norm_num

theorem tiltSegment_below {ai aj a : ℝ} (h : ai < aj) (ha : a ≤ ai) (ti tj : Vec3 ℝ) :
    tiltSegment ai aj ti tj a = ti := by
  have : (a - ai) / (aj - ai) ≤ 0 := div_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
  simp only [tiltSegment, h, ↓reduceIte, clamp_real]
  rw [max_eq_right (by norm_num; exact this)]
  ext <;> norm_num

theorem tiltSegment_above {ai aj a : ℝ} (h : ai < aj) (ha : aj ≤ a) (ti tj : Vec3 ℝ) :
    tiltSegment ai aj ti tj a = tj := by
  have : 1 ≤ (a - ai) / (aj - ai) := by rw [le_div_iff₀ (by linarith)]; linarith
  simp only [tiltSegment, h, ↓reduceIte, clamp_real]
  rw [max_eq_left (by norm_num; linarith), min_eq_right (by norm_num; linarith)]
  ext <;> norm_num

/-- **Two brackets at the same `A_mid` take the first one's tilt**: the segment's share is 0. -/
theorem tiltSegment_tie {ai aj a : ℝ} (h : ¬ ai < aj) (ti tj : Vec3 ℝ) : tiltSegment ai aj ti tj a = ti := by
  simp only [tiltSegment, h, ↓reduceIte]
  ext <;> norm_num

/-- **The bracket order is stable**: of two brackets at the same `A_mid`, the one first in the
bundle comes first. -/
theorem bracketOrder_stable (aMid : Array ℝ) {i j : ℕ} (hij : i < j) (hj : j < aMid.size)
    (heq : aMid.getD i 0.0 = aMid.getD j 0.0) : [i, j].Sublist (bracketOrder aMid) := by
  unfold bracketOrder
  apply List.pair_sublist_mergeSort
  · intro a b c hab hbc
    simp only [decide_eq_true_eq] at hab hbc ⊢
    exact hab.trans hbc
  · intro a b
    simp only [Bool.or_eq_true, decide_eq_true_eq]
    exact le_total _ _
  · simp [heq]
  · have hr : List.range aMid.size = List.range j ++ (List.range (aMid.size - j)).map (j + ·) := by
      rw [← List.range_add]; congr 1; omega
    rw [hr, show [i, j] = [i] ++ [j] from rfl]
    apply List.Sublist.append
    · exact List.singleton_sublist.mpr (List.mem_range.mpr hij)
    · exact List.singleton_sublist.mpr (List.mem_map.mpr ⟨0, List.mem_range.mpr (by omega), by simp⟩)

/-! ## The print's white -/

/-- The fold that picks the lightest node keeps a node at least as light as its start and as every
node it has seen, and returns its start or one of them. -/
theorem lightestFold_spec (clut : Array ℝ) (l : List ℕ) (b0 : ℕ) :
    let r := l.foldl (fun best n => if clut.getD (best * 3 + 1) 0.0 < clut.getD (n * 3 + 1) 0.0 then n else best) b0
    clut.getD (b0 * 3 + 1) 0.0 ≤ clut.getD (r * 3 + 1) 0.0 ∧
      (∀ n ∈ l, clut.getD (n * 3 + 1) 0.0 ≤ clut.getD (r * 3 + 1) 0.0) ∧ (r = b0 ∨ r ∈ l) := by
  induction l generalizing b0 with
  | nil => simp
  | cons n ns ih =>
    simp only [List.foldl_cons, List.mem_cons]
    obtain ⟨h1, h2, h3⟩ := ih (if clut.getD (b0 * 3 + 1) 0.0 < clut.getD (n * 3 + 1) 0.0 then n else b0)
    by_cases hc : clut.getD (b0 * 3 + 1) 0.0 < clut.getD (n * 3 + 1) 0.0
    · simp only [hc, ↓reduceIte] at h1 h2 h3 ⊢
      refine ⟨hc.le.trans h1, fun m hm => ?_, ?_⟩
      · rcases hm with rfl | hm
        · exact h1
        · exact h2 m hm
      · rcases h3 with h | h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
    · simp only [hc, ↓reduceIte] at h1 h2 h3 ⊢
      refine ⟨h1, fun m hm => ?_, ?_⟩
      · rcases hm with rfl | hm
        · exact (not_lt.mp hc).trans h1
        · exact h2 m hm
      · rcases h3 with h | h
        · exact Or.inl h
        · exact Or.inr (Or.inr h)

/-- The lightest node is at least as light as every node. -/
theorem lightestNode_ge (clut : Array ℝ) {n : ℕ} (hn : n < clut.size / 3) :
    clut.getD (n * 3 + 1) 0.0 ≤ clut.getD (lightestNode clut * 3 + 1) 0.0 :=
  (lightestFold_spec clut (List.range (clut.size / 3)) 0).2.1 n (List.mem_range.mpr hn)

theorem lightestNode_lt (clut : Array ℝ) (h : 0 < clut.size / 3) : lightestNode clut < clut.size / 3 := by
  rcases (lightestFold_spec clut (List.range (clut.size / 3)) 0).2.2 with h0 | h0
  · simp only [lightestNode]; rw [h0]; exact h
  · exact List.mem_range.mp h0

/-- **No node of the media-relative table is lighter than the PCS white** (`Y` at most the white's
`Y`, read as XYZ through `kPcsMax`). `X` and `Z` carry no such bound: the lightest node need not
have the largest `X` or `Z`. -/
theorem toPaperWhite_y_le (clut : Array ℝ) (hpos : 0 < clut.getD (lightestNode clut * 3 + 1) 0.0)
    {n : ℕ} (hn : n < clut.size / 3) :
    (toPaperWhite clut).getD (n * 3 + 1) 0.0 * pcsMax ≤ pcsWhite.y := by
  have hle := lightestNode_ge clut hn
  have hi : n * 3 + 1 < clut.size := by omega
  have h1 : (n * 3 + 1) % 3 = 1 := by omega
  simp only [toPaperWhite, scaleTriples]
  rw [getD_map_range _ _ _ _ hi]
  simp only [h1, ↓reduceIte, one_ne_zero, paperWhiteGains, paperBase, pcsWhite, pcsMax]
  have hpm : (0:ℝ) < 1.0 + 32767.0 / 32768.0 := by norm_num
  generalize (1.0 + 32767.0 / 32768.0 : ℝ) = pm at *
  generalize clut.getD (lightestNode clut * 3 + 1) 0.0 = yb at *
  generalize clut.getD (n * 3 + 1) 0.0 = yn at *
  rw [show (1.0:ℝ) = 1 by norm_num, show yn * (1 / (yb * pm)) * pm = yn / yb by field_simp]
  exact (div_le_one hpos).mpr hle

/-- **The paper's base becomes the PCS white**: after the scaling, the lightest node's X, Y and Z
(read through `kPcsMax`) are the PCS white's. -/
theorem toPaperWhite_base (clut : Array ℝ) (hsize : 0 < clut.size / 3) (c : ℕ) (hc : c < 3)
    (hne : clut.getD (lightestNode clut * 3 + c) 0.0 ≠ 0) :
    (toPaperWhite clut).getD (lightestNode clut * 3 + c) 0.0 * pcsMax =
      (if c = 0 then pcsWhite.x else if c = 1 then pcsWhite.y else pcsWhite.z) := by
  have hb := lightestNode_lt clut hsize
  have hi : lightestNode clut * 3 + c < clut.size := by omega
  have hpm : (pcsMax : ℝ) ≠ 0 := by simp only [pcsMax]; norm_num
  have hmod : (lightestNode clut * 3 + c) % 3 = c := by omega
  simp only [toPaperWhite, scaleTriples]
  rw [getD_map_range _ _ _ _ hi, hmod]
  interval_cases c <;> simp only [paperWhiteGains, paperBase, ↓reduceIte, one_ne_zero,
    OfNat.ofNat_ne_zero, OfNat.ofNat_ne_one, Nat.add_zero] at hne ⊢ <;> field_simp

/-! ## Density -/

/-- **Bare film is density 0.** -/
theorem lightDensity_one : lightDensity (1 : ℝ) = 0 := by
  simp only [lightDensity, densityMinT]
  rw [max_eq_left (by norm_num), log10_real, Real.logb_one]
  norm_num

/-- **The density floor is 4.0**: a transmittance at or below `1e-4` reads density 4. -/
theorem lightDensity_le_four (t : ℝ) : lightDensity t ≤ 4 := by
  have h4 : Real.logb 10 (1.0e-4 : ℝ) = -4 := by
    have : (1.0e-4 : ℝ) = (10:ℝ) ^ (-4:ℝ) := by rw [Real.rpow_neg (by norm_num)]; norm_num
    rw [this, Real.logb_rpow (by norm_num) (by norm_num)]
  have hm : Real.logb 10 (1.0e-4 : ℝ) ≤ Real.logb 10 (max t 1.0e-4) :=
    Real.logb_le_logb_of_le (by norm_num) (by norm_num) (le_max_right _ _)
  simp only [lightDensity, densityMinT, log10_real]
  linarith

/-- Density falls as the transmittance rises. -/
theorem lightDensity_anti {t t' : ℝ} (h : t ≤ t') : lightDensity t' ≤ lightDensity t := by
  have hm : Real.logb 10 (max t 1.0e-4) ≤ Real.logb 10 (max t' 1.0e-4) :=
    Real.logb_le_logb_of_le (by norm_num) (lt_of_lt_of_le (by norm_num) (le_max_right _ _))
      (max_le_max h le_rfl)
  simp only [lightDensity, densityMinT, log10_real]
  linarith

/-- **Density 0 is code `offset`** (the film base), for an encoding whose offset fits. -/
theorem densityCode_zero (e : DensityEncoding ℝ) (kc : ℝ) (h : e.offset ≤ e.maxCode) :
    densityCode e kc 0 = (e.offset, false, false) := by
  have hr : roundNat ((e.offset : ℝ)) = e.offset := by
    simp only [roundNat, natFloor_real]
    rw [add_comm, Nat.floor_add_natCast (by norm_num), Nat.floor_eq_zero.mpr (by norm_num), zero_add]
  have hn : ¬ ((e.offset : ℝ) ≤ -0.5) := by
    have : (0 : ℝ) ≤ e.offset := Nat.cast_nonneg _
    norm_num; linarith
  simp only [densityCode, mul_zero, add_zero, hn, ↓reduceIte, hr, not_lt.mpr h]

/-- **The balance offsets sum to zero**: the colour without the level. -/
theorem densityBalanceOffset_sum {m : Mat3 ℝ} {e g b : ℝ} {off : Vec3 ℝ} {mean : ℝ}
    (h : densityBalanceOffset m e g b = some (off, mean)) : off.x + off.y + off.z = 0 := by
  unfold densityBalanceOffset at h
  split_ifs at h
  injection h with h
  injection h with h1 h2
  subst h1; subst h2
  simp only
  norm_num
  ring

/-- **The CLF's balance is the whole of what the gains do to the density**, level included:
`M (0, −log10 g, −log10 b)`. -/
theorem printBalance_eq (m : Mat3 ℝ) {e g b : ℝ} (he : 0 < e) (hg : 0 < g) (hb : 0 < b) :
    printBalance m e g b = m.mulVec ⟨0.0, -(log10 g), -(log10 b)⟩ := by
  have hc : (0.0 : ℝ) < e ∧ (0.0 : ℝ) < g ∧ (0.0 : ℝ) < b := by norm_num; exact ⟨he, hg, hb⟩
  simp only [printBalance, densityBalanceOffset, hc, and_self, ↓reduceIte]
  ext <;> simp only <;> norm_num

/-! ## The CLF -/

/-- **The CLF's first matrix is the slope form**: on the code fraction `q`, its printing density
is `0.70 + S_i (P (D + balance − grey))_i` with `D` the file's density of `q`. -/
theorem clfPrintingDensity_slope_form (e : DensityEncoding ℝ) (fp : FramePrint ℝ) (bal q : Vec3 ℝ)
    (hk : (e.scale : ℝ) * e.k.x ≠ 0 ∧ (e.scale : ℝ) * e.k.y ≠ 0 ∧ (e.scale : ℝ) * e.k.z ≠ 0) :
    clfPrintingDensity e fp bal q =
      ⟨printGreyDensity + fp.slopes.x * fp.toPrint.row0.dot (codeDensity e q + bal - fp.grey),
       printGreyDensity + fp.slopes.y * fp.toPrint.row1.dot (codeDensity e q + bal - fp.grey),
       printGreyDensity + fp.slopes.z * fp.toPrint.row2.dot (codeDensity e q + bal - fp.grey)⟩ := by
  obtain ⟨hx, hy, hz⟩ := hk
  ext <;> simp only [clfPrintingDensity, frameClfMatrix, codeDensity, Mat3.entry, Mat3.mulVec, Vec3.dot,
    Mat3.vadd_x, Mat3.vadd_y, Mat3.vadd_z, Mat3.vsub_x, Mat3.vsub_y, Mat3.vsub_z] <;> norm_num <;>
    field_simp <;> ring

/-- **The balanced mid grey prints at 0.70** in every channel. -/
theorem clf_mid_grey (e : DensityEncoding ℝ) (fp : FramePrint ℝ) (bal q : Vec3 ℝ)
    (hk : (e.scale : ℝ) * e.k.x ≠ 0 ∧ (e.scale : ℝ) * e.k.y ≠ 0 ∧ (e.scale : ℝ) * e.k.z ≠ 0)
    (h : codeDensity e q = fp.grey - bal) :
    clfPrintingDensity e fp bal q = ⟨printGreyDensity, printGreyDensity, printGreyDensity⟩ := by
  rw [clfPrintingDensity_slope_form e fp bal q hk, h]
  ext <;> simp only [Vec3.dot, Mat3.vadd_x, Mat3.vadd_y, Mat3.vadd_z, Mat3.vsub_x, Mat3.vsub_y,
    Mat3.vsub_z] <;> ring

/-- **A constant added to the printing density moves the print's exposure by it**: adding `c` to
the CLF's three offsets prints as `c` less log exposure on every layer (0.30 a stop). -/
theorem exposureOf_shift (m : PrintMaterial ℝ) (pd : Vec3 ℝ) (c : ℝ) :
    m.exposureOf ⟨pd.x + c, pd.y + c, pd.z + c⟩ =
      ⟨(m.exposureOf pd).x - c, (m.exposureOf pd).y - c, (m.exposureOf pd).z - c⟩ := by
  ext <;> simp only [PrintMaterial.exposureOf] <;> ring

/-- **The slopes give the grey scale equal contrast** (DESIGN.md 5.9): multiplied by its slope,
each channel's contrast `γ_c` is the green's. -/
theorem curveSlopesAt_equal_contrast (g : GreyBalance ℝ) (mApd : Mat3 ℝ) (a : ℝ)
    (hx : (curveGamma g mApd a).x ≠ 0) (hz : (curveGamma g mApd a).z ≠ 0) :
    (curveSlopesAt g mApd a).x * (curveGamma g mApd a).x = (curveGamma g mApd a).y ∧
    (curveSlopesAt g mApd a).y * (curveGamma g mApd a).y = (curveGamma g mApd a).y ∧
    (curveSlopesAt g mApd a).z * (curveGamma g mApd a).z = (curveGamma g mApd a).y := by
  simp only [curveSlopesAt]
  refine ⟨?_, ?_, ?_⟩
  · field_simp
  · norm_num
  · field_simp

/-- **The mid grey's constant cancels inside the CLF** (DESIGN.md 5.9): the log exposure the CLF's
nodes give a code, `aim + 0.70 − printing density`, is `aim − S_i (P (D + balance − grey))_i`, free
of the 0.70. -/
theorem clf_grey_density_cancels (m : PrintMaterial ℝ) (e : DensityEncoding ℝ) (fp : FramePrint ℝ)
    (bal q : Vec3 ℝ)
    (hk : (e.scale : ℝ) * e.k.x ≠ 0 ∧ (e.scale : ℝ) * e.k.y ≠ 0 ∧ (e.scale : ℝ) * e.k.z ≠ 0) :
    (clfExposureOffsets m).x - (clfPrintingDensity e fp bal q).x =
      m.aim.x - fp.slopes.x * fp.toPrint.row0.dot (codeDensity e q + bal - fp.grey) := by
  rw [clfPrintingDensity_slope_form e fp bal q hk]
  simp only [clfExposureOffsets]
  ring

/-- The mid grey's printing density 0.70 is exposed at `aim`. -/
theorem exposureOf_grey (m : PrintMaterial ℝ) :
    m.exposureOf ⟨printGreyDensity, printGreyDensity, printGreyDensity⟩ = m.aim := by
  ext <;> simp [PrintMaterial.exposureOf]

/-! ## The film base's floor -/

theorem raiseTo_ge (k : ℝ) (o : Option ℝ) : k ≤ raiseTo k o := by
  cases o with
  | none => exact le_refl k
  | some v => exact le_max_left k v

theorem raiseTo_ge_some (k v : ℝ) : v ≤ raiseTo k (some v) := le_max_right k v

/-- **β is at least 1**: the floor never darkens the base. -/
theorem printFloorFactor_ge_one (x : Mat3 ℝ) (f : Vec3 ℝ) (pixel : ℕ → ℕ → Vec3 ℕ)
    (crop : ℕ × ℕ × ℕ × ℕ) : 1 ≤ printFloorFactor x f pixel crop := by
  have h := (raiseTo_ge 1.0 (floorChannel x.row0 f.x pixel crop)).trans
    ((raiseTo_ge _ (floorChannel x.row1 f.y pixel crop)).trans (raiseTo_ge _ (floorChannel x.row2 f.z pixel crop)))
  simp only [printFloorFactor]
  calc (1 : ℝ) = 1.0 := by norm_num
    _ ≤ _ := h

/-- **β is at least every channel's 99.5 % point**: on the base raised to `β f`, each channel's
99.5 % point of `(X v)_c / (β f_c)` is at most 1. -/
theorem printFloorFactor_ge_channel (x : Mat3 ℝ) (f : Vec3 ℝ) (pixel : ℕ → ℕ → Vec3 ℕ)
    (crop : ℕ × ℕ × ℕ × ℕ) {v : ℝ}
    (h : floorChannel x.row0 f.x pixel crop = some v ∨ floorChannel x.row1 f.y pixel crop = some v ∨
      floorChannel x.row2 f.z pixel crop = some v) :
    v ≤ printFloorFactor x f pixel crop := by
  simp only [printFloorFactor]
  rcases h with h | h | h
  · rw [h]; exact (raiseTo_ge_some _ _).trans ((raiseTo_ge _ _).trans (raiseTo_ge _ _))
  · rw [h]; exact (raiseTo_ge_some _ _).trans (raiseTo_ge _ _)
  · rw [h]; exact raiseTo_ge_some _ _

/-! ## The search -/

/-- **The held exposure is the result's 1**: the film base reads `n0`, whatever the scan's exposure. -/
theorem resultExposure_held (wl : Bool) {e : ℝ} (he : e ≠ 0) :
    resultExposure wl (searchExposure wl e) e = 1 := by
  cases wl
  · simp [resultExposure, searchExposure, he]
  · simp [resultExposure, searchExposure]; norm_num

/-- **The metering bisection** keeps its interval inside the range, halves it at each step, and
keeps the end above the level (`L*` 50) as `lo` and the end at or below it as `hi`. -/
theorem bisectLevel_spec (f : ℝ → ℝ) (level : ℝ) (n : ℕ) :
    ∀ lo hi : ℝ, lo ≤ hi →
      lo ≤ (bisectLevel f level n lo hi).1 ∧ (bisectLevel f level n lo hi).1 ≤ (bisectLevel f level n lo hi).2 ∧
      (bisectLevel f level n lo hi).2 ≤ hi ∧
      (bisectLevel f level n lo hi).2 - (bisectLevel f level n lo hi).1 = (hi - lo) / 2 ^ n ∧
      (level < f lo → level < f (bisectLevel f level n lo hi).1) ∧
      (f hi ≤ level → f (bisectLevel f level n lo hi).2 ≤ level) := by
  induction n with
  | zero => intro lo hi h; simp [bisectLevel, h]
  | succ n ih =>
    intro lo hi h
    have hm1 : lo ≤ 0.5 * (lo + hi) := by norm_num; linarith
    have hm2 : 0.5 * (lo + hi) ≤ hi := by norm_num; linarith
    simp only [bisectLevel]
    split_ifs with hc
    · obtain ⟨a1, a2, a3, a4, a5, a6⟩ := ih _ hi hm2
      refine ⟨hm1.trans a1, a2, a3, ?_, fun _ => a5 hc, a6⟩
      rw [a4, pow_succ]; field_simp; norm_num; ring
    · obtain ⟨a1, a2, a3, a4, a5, a6⟩ := ih lo _ hm1
      refine ⟨a1, a2, a3.trans hm2, ?_, a5, fun _ => a6 (not_lt.mp hc)⟩
      rw [a4, pow_succ]; field_simp; norm_num; ring

/-- A0 lies in the range it is metered over. -/
theorem meterA0_mem (lstar : ℝ → ℝ) {aLo aHi : ℝ} (h : aLo ≤ aHi) :
    aLo ≤ meterA0 lstar aLo aHi ∧ meterA0 lstar aLo aHi ≤ aHi := by
  obtain ⟨a1, a2, a3, -⟩ := bisectLevel_spec lstar 50.0 60 aLo aHi h
  simp only [meterA0]
  constructor <;> norm_num <;> linarith

theorem clamp_mem {lo hi : ℝ} (h : lo ≤ hi) (x : ℝ) : lo ≤ clamp x lo hi ∧ clamp x lo hi ≤ hi := by
  simp only [clamp_real]
  exact ⟨le_min (le_max_right _ _) h, min_le_right _ _⟩

theorem goldenPhi_mem : (0 : ℝ) ≤ goldenPhi ∧ (goldenPhi : ℝ) ≤ 1 := by
  simp only [goldenPhi]; norm_num

/-- The golden-section loop's points stay in any interval that holds its starting points: each new
point is a convex combination of two points it has. -/
theorem goldenLoop_mem (loss : ℝ → ℝ) {L H : ℝ} (n : ℕ) :
    ∀ lo hi x1 x2 f1 f2 : ℝ, L ≤ lo → lo ≤ H → L ≤ hi → hi ≤ H → L ≤ x1 → x1 ≤ H → L ≤ x2 → x2 ≤ H →
      L ≤ (goldenLoop loss n lo hi x1 x2 f1 f2).1 ∧ (goldenLoop loss n lo hi x1 x2 f1 f2).1 ≤ H ∧
      L ≤ (goldenLoop loss n lo hi x1 x2 f1 f2).2 ∧ (goldenLoop loss n lo hi x1 x2 f1 f2).2 ≤ H := by
  obtain ⟨p0, p1⟩ := goldenPhi_mem
  induction n with
  | zero => intro lo hi x1 x2 f1 f2 a b c d _ _ _ _; exact ⟨a, b, c, d⟩
  | succ n ih =>
    intro lo hi x1 x2 f1 f2 a b c d e f g h
    simp only [goldenLoop]
    split_ifs
    · exact ih _ _ _ _ _ _ a b g h (by nlinarith) (by nlinarith) e f
    · exact ih _ _ _ _ _ _ e f c d g h (by nlinarith) (by nlinarith)
    · exact ⟨a, b, c, d⟩

theorem goldenSection_mem (loss : ℝ → ℝ) {L H lo hi : ℝ} (n : ℕ) (a : L ≤ lo) (b : lo ≤ H) (c : L ≤ hi)
    (d : hi ≤ H) : L ≤ goldenSection loss lo hi n ∧ goldenSection loss lo hi n ≤ H := by
  obtain ⟨p0, p1⟩ := goldenPhi_mem
  obtain ⟨r1, r2, r3, r4⟩ := goldenLoop_mem loss n lo hi (hi - goldenPhi * (hi - lo))
    (lo + goldenPhi * (hi - lo)) (loss (hi - goldenPhi * (hi - lo))) (loss (lo + goldenPhi * (hi - lo)))
    a b c d (by nlinarith) (by nlinarith) (by nlinarith) (by nlinarith)
  simp only [goldenSection]
  constructor <;> norm_num <;> linarith

theorem marchLoop_mem (loss : ℝ → ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (dir : ℝ) (n : ℕ) :
    ∀ xo xm fm xn fn : ℝ, aLo ≤ xo → xo ≤ aHi → aLo ≤ xm → xm ≤ aHi → aLo ≤ xn → xn ≤ aHi →
      aLo ≤ (marchLoop loss aLo aHi dir n xo xm fm xn fn).lo ∧ (marchLoop loss aLo aHi dir n xo xm fm xn fn).lo ≤ aHi ∧
      aLo ≤ (marchLoop loss aLo aHi dir n xo xm fm xn fn).hi ∧ (marchLoop loss aLo aHi dir n xo xm fm xn fn).hi ≤ aHi := by
  induction n with
  | zero =>
    intro xo xm fm xn fn a b _ _ e f
    simp only [marchLoop]
    exact ⟨le_min a e, (min_le_left _ _).trans b, a.trans (le_max_left _ _), max_le b f⟩
  | succ n ih =>
    intro xo xm fm xn fn a b c d e f
    obtain ⟨g, h⟩ := clamp_mem hr (xn + dir * continuousStep)
    simp only [marchLoop]
    split_ifs
    · exact ⟨le_min c g, (min_le_left _ _).trans d, c.trans (le_max_left _ _), max_le d h⟩
    · exact ih _ _ _ _ _ c d e f g h
    · exact ⟨le_min a e, (min_le_left _ _).trans b, a.trans (le_max_left _ _), max_le b f⟩

theorem march_mem (loss : ℝ → ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (dir x0 f0 : ℝ) (n : ℕ) (a : aLo ≤ x0)
    (b : x0 ≤ aHi) :
    aLo ≤ (march loss aLo aHi dir x0 f0 n).lo ∧ (march loss aLo aHi dir x0 f0 n).lo ≤ aHi ∧
      aLo ≤ (march loss aLo aHi dir x0 f0 n).hi ∧ (march loss aLo aHi dir x0 f0 n).hi ≤ aHi := by
  obtain ⟨c, d⟩ := clamp_mem hr (x0 - dir * continuousStep)
  obtain ⟨e, f⟩ := clamp_mem hr (x0 + dir * continuousStep)
  exact marchLoop_mem loss hr dir n _ _ _ _ _ c d a b e f

theorem bracketA_mem (loss : ℝ → ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (a : ℝ) (n : ℕ) :
    aLo ≤ (bracketA loss aLo aHi a n).lo ∧ (bracketA loss aLo aHi a n).lo ≤ aHi ∧
      aLo ≤ (bracketA loss aLo aHi a n).hi ∧ (bracketA loss aLo aHi a n).hi ≤ aHi := by
  obtain ⟨c0, d0⟩ := clamp_mem hr a
  obtain ⟨cl, dl⟩ := clamp_mem hr (clamp a aLo aHi - continuousStep)
  obtain ⟨cr, dr⟩ := clamp_mem hr (clamp a aLo aHi + continuousStep)
  simp only [bracketA]
  split_ifs <;> first
    | exact march_mem loss hr _ _ _ n cl dl
    | exact march_mem loss hr _ _ _ n cr dr
    | exact ⟨cl, dl, cr, dr⟩

/-- **The line search's `A` lies in the range.** -/
theorem lineSearchA_mem (loss : ℝ → ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (a : ℝ) :
    aLo ≤ lineSearchA loss aLo aHi a ∧ lineSearchA loss aLo aHi a ≤ aHi := by
  obtain ⟨a1, a2, a3, a4⟩ := bracketA_mem loss hr a lineSearchFuel
  exact goldenSection_mem loss lineSearchFuel a1 a2 a3 a4

theorem printSearchLoop_mem {σ : Type} (o : PrintSearchOracle σ ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi)
    (maxSteps : ℕ) (fuel : ℕ) :
    ∀ (it : ℕ) (s : PrintSearchState σ ℝ), aLo ≤ s.a → s.a ≤ aHi →
      (∀ b, s.best = some b → aLo ≤ b.a ∧ b.a ≤ aHi) →
      ∀ r, printSearchLoop o aLo aHi maxSteps fuel it s = some r → aLo ≤ r.a ∧ r.a ≤ aHi := by
  induction fuel with
  | zero => intro it s _ _ hb r h; exact hb r h
  | succ n ih =>
    intro it s ha1 ha2 hb r h
    simp only [printSearchLoop] at h
    have hbest : ∀ here : PrintSearchResult ℝ, here.a = s.a →
        aLo ≤ (keepBest s.best here).a ∧ (keepBest s.best here).a ≤ aHi := by
      intro here hh
      unfold keepBest
      split
      · rw [hh]; exact ⟨ha1, ha2⟩
      · rename_i b hsb
        split_ifs
        · rw [hh]; exact ⟨ha1, ha2⟩
        · exact hb b hsb
    obtain ⟨n1, n2⟩ := lineSearchA_mem (fun x => ((o.step s.oracle s.a s.g s.b).2.2 x).1) hr s.a
    split_ifs at h with h1 h2
    all_goals first
      | (injection h with h; subst h; dsimp only; exact hbest _ rfl)
      | (refine ih _ _ n1 n2 ?_ r h
         intro b hb'
         injection hb' with hb'
         subst hb'
         exact hbest _ rfl)

/-- **The print's search keeps an `A` in the curve's range.** -/
theorem printSearch_a_mem {σ : Type} (o : PrintSearchOracle σ ℝ) (s0 : σ) {aLo aHi a0 g0 b0 : ℝ}
    (hr : aLo ≤ aHi) (ha1 : aLo ≤ a0) (ha2 : a0 ≤ aHi) (maxSteps : ℕ) {r : PrintSearchResult ℝ}
    (h : printSearch o s0 aLo aHi a0 g0 b0 maxSteps = some r) : aLo ≤ r.a ∧ r.a ≤ aHi :=
  printSearchLoop_mem o hr _ _ 0 _ ha1 ha2 (fun _ hb => by simp at hb) r h

theorem knobNewton_mem (render : Vec3 ℝ → Vec3 ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (target : Vec3 ℝ)
    (n : ℕ) : ∀ x lab : Vec3 ℝ, aLo ≤ x.x → x.x ≤ aHi →
      aLo ≤ (knobNewton render aLo aHi target n x lab).x ∧ (knobNewton render aLo aHi target n x lab).x ≤ aHi := by
  induction n with
  | zero => intro x lab a b; exact ⟨a, b⟩
  | succ n ih =>
    intro x lab a b
    simp only [knobNewton]
    split_ifs
    all_goals first
      | exact ⟨a, b⟩
      | (apply ih
         · exact (clamp_mem hr _).1
         · exact (clamp_mem hr _).2)

theorem knobSolve_mem (render : Vec3 ℝ → Vec3 ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (x0 target : Vec3 ℝ)
    (a : aLo ≤ x0.x) (b : x0.x ≤ aHi) :
    aLo ≤ (knobSolve render aLo aHi x0 target).1.x ∧ (knobSolve render aLo aHi x0 target).1.x ≤ aHi :=
  knobNewton_mem render hr target 30 x0 _ a b

theorem knobLargest_mem (render : Vec3 ℝ → Vec3 ℝ) {aLo aHi : ℝ} (hr : aLo ≤ aHi) (x0 : Vec3 ℝ)
    (a0 : aLo ≤ x0.x) (b0 : x0.x ≤ aHi) (make : ℝ → Vec3 ℝ) (n : ℕ) :
    ∀ lo hi : ℝ, ∀ best : Vec3 ℝ, aLo ≤ best.x → best.x ≤ aHi →
      aLo ≤ (knobLargest render aLo aHi x0 make n lo hi best).x ∧
        (knobLargest render aLo aHi x0 make n lo hi best).x ≤ aHi := by
  induction n with
  | zero => intro lo hi best a b; exact ⟨a, b⟩
  | succ n ih =>
    intro lo hi best a b
    simp only [knobLargest]
    obtain ⟨c, d⟩ := knobSolve_mem render hr x0 (make (0.5 * (lo + hi))) a0 b0
    split_ifs
    · apply ih
      · exact c
      · exact d
    · apply ih
      · exact a
      · exact b

/-- **The knobs keep the print's `A` in the curve's range.** -/
theorem solvePrintKnobs_a_mem (render : Vec3 ℝ → Vec3 ℝ) {aLo aHi a0 g0 b0 fullL : ℝ} {knobs : Vec3 ℝ}
    {v : Vec3 ℝ} {lim : Bool} (h : solvePrintKnobs render aLo aHi a0 g0 b0 knobs fullL = some (v, lim)) :
    aLo ≤ v.x ∧ v.x ≤ aHi := by
  unfold solvePrintKnobs at h
  dsimp only at h
  split_ifs at h with hv hk h1 h2
  all_goals
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    have hr : aLo ≤ aHi := hv.2.2.1.le
    have a0' : aLo ≤ (⟨a0, log g0, log b0⟩ : Vec3 ℝ).x := hv.2.2.2.1
    have b0' : (⟨a0, log g0, log b0⟩ : Vec3 ℝ).x ≤ aHi := hv.2.2.2.2
    first
    | exact ⟨a0', b0'⟩
    | exact knobSolve_mem render hr _ _ a0' b0'
    | exact knobLargest_mem render hr _ a0' b0' _ 7 _ _ _ (knobSolve_mem render hr _ _ a0' b0').1
        (knobSolve_mem render hr _ _ a0' b0').2
    | exact knobLargest_mem render hr _ a0' b0' _ 7 _ _ _ a0' b0'

/-! ## Aim -/

/-- **Aim prints an 18 % grey** within `1e-6` in each of X, Y and Z, whenever it is found. -/
theorem aimSolve_residual {m : PrintMaterial ℝ} {e : Vec3 ℝ} (h : aimSolve m = some e) :
    |(aimResidual m e).x| < 1.0e-6 ∧ |(aimResidual m e).y| < 1.0e-6 ∧ |(aimResidual m e).z| < 1.0e-6 := by
  unfold aimSolve at h
  split at h
  · exact absurd h (by simp)
  · rename_i e' _
    dsimp only at h
    split_ifs at h with hc
    injection h with h
    subst h
    simpa only [fabs_real] using hc

/-- **The mid grey prints as an 18 % grey**: a loaded material prints the printing density 0.70 in
every channel within `1e-6` of 18 % of its viewing light, in each of X, Y and Z. -/
theorem loadMaterial_grey {p : SpektrafilmProfile ℝ} {m : PrintMaterial ℝ} (h : loadMaterial p = some m) :
    |(printOfDensity m ⟨printGreyDensity, printGreyDensity, printGreyDensity⟩).x - 0.18 * m.white.x| < 1.0e-6 ∧
    |(printOfDensity m ⟨printGreyDensity, printGreyDensity, printGreyDensity⟩).y - 0.18 * m.white.y| < 1.0e-6 ∧
    |(printOfDensity m ⟨printGreyDensity, printGreyDensity, printGreyDensity⟩).z - 0.18 * m.white.z| < 1.0e-6 := by
  unfold loadMaterial at h
  dsimp only at h
  split at h
  · rw [Option.map_eq_some_iff] at h
    obtain ⟨e, ha, rfl⟩ := h
    have := aimSolve_residual ha
    simp only [printOfDensity, exposureOf_grey]
    exact this
  · exact absurd h (by simp)

end Negicc
