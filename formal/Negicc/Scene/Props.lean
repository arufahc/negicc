/-
Properties of the per-pixel conversion over the real numbers.
-/
module

public import Negicc.Base
public import Negicc.Scene.Defs

public section

/-!
# Properties of the per-pixel conversion

Proved over `ℝ`.

* How `E`, `g` and `b` enter: they scale the channels of the corrected RGB, and nothing else
  (`candidateMatrix_mulVec`). The exposure is a common factor (`candidateMatrix_mul_exposure`), an
  offset in density (`density_exposure`), and the search's reference units are the frame's units
  divided by `e_norm` (`candidateMatrix_divScalar`).
* The film base normalisation puts bare film on one level (`mergedMatrix_mulVec_base`).
* The knee and the shoulder are the identity below their knees, and a conversion with the shoulder
  off or without a tone curve reduces to the simpler formula.
* The colour table's interpolation is a weighted mean with weights that add up to one
  (`clutEval_const`).
* Between its knee and `x_hi` the shoulder is a parabola in the *encoded* value
  (`srgbEncode_applyShoulderChannel`).
-/

namespace Negicc

open Scalar

/-! ## Exposure and gains -/

/-- `E` scales the three channels of the corrected RGB (before the knee), `g` the green one and
`b` the blue one: the candidate matrix is linear in them, as `candidate_matrix` claims. -/
theorem candidateMatrix_mulVec (M : Mat3 ℝ) (e g b : ℝ) (v : Vec3 ℝ) :
    (candidateMatrix M e g b).mulVec v =
      ⟨e * (M.mulVec v).x, e * g * (M.mulVec v).y, e * b * (M.mulVec v).z⟩ := by
  rw [candidateMatrix, Mat3.mulVec_scaleRows]
  rfl

/-- With `E = g = b = 1` the candidate matrix is the merged matrix itself. -/
theorem candidateMatrix_one (M : Mat3 ℝ) : candidateMatrix M 1 1 1 = M := by
  ext <;> simp [candidateMatrix, Mat3.scaleRows, Vec3.smul]

/-- The exposure is a common factor: `k` times the exposure gives `k` times the corrected RGB
before the knee, at the same gains. -/
theorem candidateMatrix_mul_exposure (M : Mat3 ℝ) (k e g b : ℝ) (v : Vec3 ℝ) :
    (candidateMatrix M (k * e) g b).mulVec v = ((candidateMatrix M e g b).mulVec v).smul k := by
  rw [candidateMatrix_mulVec, candidateMatrix_mulVec]
  ext <;> simp only [Vec3.smul] <;> ring

/-- The merged matrix divided by `k` at the exposure `E` is the merged matrix at `E / k`. On a
narrow-band scan the search works on the merged matrix divided by `e_norm`; this identity is what
makes its exposure the conversion's divided by `e_norm` (the search itself is specified in #211). -/
theorem candidateMatrix_divScalar (M : Mat3 ℝ) (k e g b : ℝ) :
    candidateMatrix (M.divScalar k) e g b = candidateMatrix M (e / k) g b := by
  ext <;> simp only [candidateMatrix, Mat3.scaleRows, Mat3.divScalar, Vec3.smul, Vec3.map] <;> ring

/-- In density the exposure is an offset: film seen at `E` times the exposure reads `log10 E`
less dense. (So the film that renders as a given grey is denser by `log10 E`, the statement the
print emulation's frame matrix rests on.) -/
theorem density_exposure {e t : ℝ} (he : 0 < e) (ht : 0 < t) :
    density (e * t) = density t - log10 e := by
  rw [density_mul he ht]
  simp only [density]
  ring

/-! ## Film base normalisation -/

/-- The merged matrix puts bare film on one level: a pixel whose crosstalk-corrected value is the
film base `f` comes out at `n * ratio` in every channel. -/
theorem mergedMatrix_mulVec_base (X : Mat3 ℝ) (base f : Vec3 ℝ) (n ratio : ℝ)
    (hf : X.mulVec base = f) (hx : 0 < f.x) (hy : 0 < f.y) (hz : 0 < f.z) :
    (mergedMatrix X f n ratio).mulVec base = ⟨n * ratio, n * ratio, n * ratio⟩ := by
  have hx' : (0.0 : ℝ) < f.x := by norm_num; exact hx
  have hy' : (0.0 : ℝ) < f.y := by norm_num; exact hy
  have hz' : (0.0 : ℝ) < f.z := by norm_num; exact hz
  rw [mergedMatrix, Mat3.mulVec_scaleRows, hf]
  ext
  · simp only [Vec3.mul, Vec3.map]
    rw [ite_eq_left hx']
    field_simp
  · simp only [Vec3.mul, Vec3.map]
    rw [ite_eq_left hy']
    field_simp
  · simp only [Vec3.mul, Vec3.map]
    rw [ite_eq_left hz']
    field_simp

/-- **Bare film in the scan lands on the code `n`.** A scan pixel of bare film reads, through the
crosstalk correction, the film base `f` divided by `ratio` (the base was measured at `ratio` times
the scan's exposure); the merged matrix puts it on `n` in every channel. After the load that is
`n / 65535` (`0.839` for `n = 55000`), the level that enters the knee at `E = g = b = 1`. -/
theorem mergedMatrix_mulVec_scan_base (X : Mat3 ℝ) (base f : Vec3 ℝ) (n ratio : ℝ) (hr : ratio ≠ 0)
    (hf : X.mulVec base = f.map (· / ratio)) (hx : 0 < f.x) (hy : 0 < f.y) (hz : 0 < f.z) :
    (mergedMatrix X f n ratio).mulVec base = ⟨n, n, n⟩ := by
  have hx' : (0.0 : ℝ) < f.x := by norm_num; exact hx
  have hy' : (0.0 : ℝ) < f.y := by norm_num; exact hy
  have hz' : (0.0 : ℝ) < f.z := by norm_num; exact hz
  rw [mergedMatrix, Mat3.mulVec_scaleRows, hf]
  ext
  · simp only [Vec3.mul, Vec3.map]
    rw [ite_eq_left hx']
    field_simp
  · simp only [Vec3.mul, Vec3.map]
    rw [ite_eq_left hy']
    field_simp
  · simp only [Vec3.mul, Vec3.map]
    rw [ite_eq_left hz']
    field_simp

/-! ## The knee -/

/-- The knee sends a non-positive value to zero. -/
theorem kneeCompress_of_nonpos {knee v : ℝ} (hv : v ≤ 0) : kneeCompress knee v = 0 := by
  have hv' : v ≤ (0.0 : ℝ) := by norm_num; exact hv
  unfold kneeCompress
  rw [ite_eq_left hv']
  norm_num

/-- Up to the knee the knee is the identity: the corrected RGB stays proportional to the film's
transmission. -/
theorem kneeCompress_of_le {knee v : ℝ} (hk : knee < 1) (h0 : 0 < v) (hv : v ≤ knee) :
    kneeCompress knee v = v := by
  have h0' : ¬ v ≤ (0.0 : ℝ) := by norm_num; exact h0
  have hk' : ¬ (1.0 : ℝ) ≤ knee := by norm_num; exact hk
  unfold kneeCompress
  rw [ite_eq_right h0', ite_eq_right hk', ite_eq_left hv]
  have : v ≤ (1.0 : ℝ) := by norm_num; linarith
  exact min_eq_right this

/-- The knee never gives more than one. -/
theorem kneeCompress_le_one (knee v : ℝ) : kneeCompress knee v ≤ 1 := by
  unfold kneeCompress
  split_ifs
  · norm_num
  · exact le_trans (min_le_left _ _) (by norm_num)
  · exact le_trans (min_le_left _ _) (by norm_num)
  · exact le_trans (min_le_left _ _) (by norm_num)

/-- The hyperbolic tangent is monotone. -/
theorem tanh_mono' {x y : ℝ} (h : x ≤ y) : Real.tanh x ≤ Real.tanh y := by
  have hx := Real.neg_one_lt_tanh x
  have hy := Real.tanh_lt_one y
  have hx1 := Real.tanh_lt_one x
  have hy1 := Real.neg_one_lt_tanh y
  rw [← Real.artanh_le_artanh_iff ⟨hx, hx1⟩ ⟨hy1, hy⟩, Real.artanh_tanh, Real.artanh_tanh]
  exact h

/-- **The knee is monotone** for a knee in `(0, 0.9999]` (the conversion's is `0.95`): a thinner film
never gives a smaller corrected value. For a knee in `(0.9999, 1)` it is not: the curve steps down at
the knee, from `knee` to at most `0.9999` (`kneeCompress_step`). -/
theorem kneeCompress_mono {knee v w : ℝ} (hk0 : 0 < knee) (hk : knee ≤ 0.9999) (hvw : v ≤ w) :
    kneeCompress knee v ≤ kneeCompress knee w := by
  have hk1 : ¬ (1.0 : ℝ) ≤ knee := by norm_num at hk ⊢; linarith
  have hd : (0 : ℝ) < 1.0 - knee := by norm_num at hk ⊢; linarith
  have hk9 : knee ≤ (0.9999 : ℝ) := hk
  -- The roll-off branch is at least the knee.
  have roll : ∀ z : ℝ, knee < z → knee ≤ min 0.9999 (knee + (1.0 - knee) * Real.tanh ((z - knee) / (1.0 - knee))) := by
    intro z hz
    have ht := tanh_mono' (le_of_lt (div_pos (by linarith : (0 : ℝ) < z - knee) hd))
    rw [Real.tanh_zero] at ht
    exact le_min hk9 (by nlinarith)
  have one : ∀ z : ℝ, z ≤ knee → min 1.0 z = z := fun z hz => min_eq_right (by norm_num at hk ⊢; linarith)
  unfold kneeCompress
  simp only [tanh_real]
  rw [ite_eq_right hk1, ite_eq_right hk1]
  by_cases hw0 : w ≤ 0.0
  · have hv0 : v ≤ (0.0 : ℝ) := le_trans hvw hw0
    rw [ite_eq_left hv0, ite_eq_left hw0]
  rw [ite_eq_right hw0]
  have hw0' : (0 : ℝ) < w := by norm_num at hw0; exact hw0
  by_cases hwk : w ≤ knee
  · rw [ite_eq_left hwk, one w hwk]
    by_cases hv0 : v ≤ 0.0
    · rw [ite_eq_left hv0]; exact le_trans (le_of_eq (by norm_num)) (le_of_lt hw0')
    · rw [ite_eq_right hv0, ite_eq_left (le_trans hvw hwk), one v (le_trans hvw hwk)]; exact hvw
  rw [ite_eq_right hwk]
  have hwk' : knee < w := lt_of_not_ge hwk
  by_cases hv0 : v ≤ 0.0
  · rw [ite_eq_left hv0]; exact le_trans (le_of_eq (by norm_num)) (le_trans (le_of_lt hk0) (roll w hwk'))
  rw [ite_eq_right hv0]
  by_cases hvk : v ≤ knee
  · rw [ite_eq_left hvk, one v hvk]; exact le_trans hvk (roll w hwk')
  rw [ite_eq_right hvk]
  have hq : (v - knee) / (1.0 - knee) ≤ (w - knee) / (1.0 - knee) :=
    div_le_div_of_nonneg_right (by linarith) hd.le
  have := tanh_mono' hq
  exact min_le_min_left _ (by nlinarith)

/-- With a knee in `(0.9999, 1)` the knee steps down: the value at the knee is the knee, and every
larger value gives at most `0.9999`. -/
theorem kneeCompress_step {knee w : ℝ} (hk : 0.9999 < knee) (hk1 : knee < 1) (hw : knee < w) :
    kneeCompress knee knee = knee ∧ kneeCompress knee w ≤ 0.9999 := by
  have h1 : ¬ (1.0 : ℝ) ≤ knee := by norm_num; exact hk1
  have h0 : ¬ knee ≤ (0.0 : ℝ) := by norm_num; linarith
  have hw0 : ¬ w ≤ (0.0 : ℝ) := by norm_num; linarith
  constructor
  · unfold kneeCompress
    rw [ite_eq_right h0, ite_eq_right h1, ite_eq_left (le_refl knee)]
    exact min_eq_right (by norm_num; linarith)
  · unfold kneeCompress
    rw [ite_eq_right hw0, ite_eq_right h1, ite_eq_right (not_le.mpr hw)]
    exact min_le_left _ _

/-! ## The colour table -/

/-- A blend of four equal nodes with weights that add up to one is that node. -/
theorem tetraBlend_const (c : Vec3 ℝ) (w0 w1 w2 w3 : ℝ) (h : w0 + w1 + w2 + w3 = 1) :
    tetraBlend c c c c w0 w1 w2 w3 = c := by
  ext <;> simp only [tetraBlend]
  · calc c.x * w0 + c.x * w1 + c.x * w2 + c.x * w3 = c.x * (w0 + w1 + w2 + w3) := by ring
      _ = c.x := by rw [h, mul_one]
  · calc c.y * w0 + c.y * w1 + c.y * w2 + c.y * w3 = c.y * (w0 + w1 + w2 + w3) := by ring
      _ = c.y := by rw [h, mul_one]
  · calc c.z * w0 + c.z * w1 + c.z * w2 + c.z * w3 = c.z * (w0 + w1 + w2 + w3) := by ring
      _ = c.z := by rw [h, mul_one]

/-- A colour table whose nodes all hold the same value gives that value everywhere: in each of
the six tetrahedra the four weights add up to one. -/
theorem clutEval_const (p : Profile ℝ) (c : Vec3 ℝ) (h : ∀ i j k, p.node i j k = c) (t : Vec3 ℝ) :
    clutEval p t = c := by
  unfold clutEval
  simp only [h]
  split_ifs <;> exact tetraBlend_const c _ _ _ _ (by norm_num)

/-- Four values in `[lo, hi]` blended with non-negative weights that add up to one stay in
`[lo, hi]`. -/
theorem blend4_mem {a b c d w0 w1 w2 w3 lo hi : ℝ} (ha : lo ≤ a ∧ a ≤ hi) (hb : lo ≤ b ∧ b ≤ hi)
    (hc : lo ≤ c ∧ c ≤ hi) (hd : lo ≤ d ∧ d ≤ hi) (h0 : 0 ≤ w0) (h1 : 0 ≤ w1) (h2 : 0 ≤ w2)
    (h3 : 0 ≤ w3) (hs : w0 + w1 + w2 + w3 = 1) :
    lo ≤ a * w0 + b * w1 + c * w2 + d * w3 ∧ a * w0 + b * w1 + c * w2 + d * w3 ≤ hi := by
  have el : lo * w0 + lo * w1 + lo * w2 + lo * w3 = lo := by
    rw [← mul_add, ← mul_add, ← mul_add, hs, mul_one]
  have eh : hi * w0 + hi * w1 + hi * w2 + hi * w3 = hi := by
    rw [← mul_add, ← mul_add, ← mul_add, hs, mul_one]
  constructor
  · linarith [mul_le_mul_of_nonneg_right ha.1 h0, mul_le_mul_of_nonneg_right hb.1 h1,
      mul_le_mul_of_nonneg_right hc.1 h2, mul_le_mul_of_nonneg_right hd.1 h3]
  · linarith [mul_le_mul_of_nonneg_right ha.2 h0, mul_le_mul_of_nonneg_right hb.2 h1,
      mul_le_mul_of_nonneg_right hc.2 h2, mul_le_mul_of_nonneg_right hd.2 h3]

/-- The position inside a grid cell: for a coordinate in `[0, 1]` on a grid of `dim ≥ 1` nodes, the
share `s - i` of the way from the cell's lower node lies in `[0, 1]`, and both nodes are on the
grid. -/
theorem clut_cell_mem {x : ℝ} {dim : ℕ} (hd : 1 ≤ dim) (hx0 : 0 ≤ x) (hx1 : x ≤ 1) :
    let s := x * ((dim - 1 : ℕ) : ℝ)
    let i := min (natFloor s) (dim - 1)
    0 ≤ s - (i : ℝ) ∧ s - (i : ℝ) ≤ 1 ∧ i < dim ∧ min (i + 1) (dim - 1) < dim := by
  intro s i
  have hn1 : ((dim - 1 : ℕ) : ℝ) = (dim : ℝ) - 1 := by rw [Nat.cast_sub hd]; simp
  have hs0 : 0 ≤ s := by
    simp only [s, hn1]; have : (1 : ℝ) ≤ dim := by exact_mod_cast hd
    nlinarith
  have hs1 : s ≤ (dim : ℝ) - 1 := by
    simp only [s, hn1]; have : (1 : ℝ) ≤ dim := by exact_mod_cast hd
    nlinarith
  have hfl : (natFloor s : ℝ) ≤ s := by rw [natFloor_real]; exact Nat.floor_le hs0
  refine ⟨?_, ?_, by omega, by omega⟩
  · have : (i : ℝ) ≤ (natFloor s : ℝ) := by exact_mod_cast min_le_left _ _
    linarith
  · by_cases hc : natFloor s ≤ dim - 1
    · have : i = natFloor s := min_eq_left hc
      rw [this, natFloor_real]; linarith [Nat.lt_floor_add_one s]
    · have : i = dim - 1 := min_eq_right (by omega)
      rw [this, Nat.cast_sub hd]; push_cast; linarith

/-- In the sixth case of `clutEval` (none of the first five orderings holds), the positions are
ordered `dr ≤ dg ≤ db`. -/
theorem clut_last_case {dr dg db : ℝ} (h1 : ¬ (dg ≤ dr ∧ db ≤ dg)) (h2 : ¬ (db ≤ dr ∧ dg < db))
    (h3 : ¬ (dr < db ∧ dg ≤ dr)) (h4 : ¬ (dr < dg ∧ db ≤ dr)) (h5 : ¬ (db ≤ dg ∧ dr < db)) :
    dr ≤ dg ∧ dg ≤ db := by
  have a : dr ≤ dg := by
    by_contra h
    push Not at h
    by_cases hb : db ≤ dg
    · exact h1 ⟨le_of_lt h, hb⟩
    · push Not at hb
      by_cases hr : db ≤ dr
      · exact h2 ⟨hr, hb⟩
      · push Not at hr
        exact h3 ⟨hr, le_of_lt h⟩
  refine ⟨a, ?_⟩
  by_contra h
  push Not at h
  by_cases hr : db ≤ dr
  · rcases lt_or_eq_of_le a with ha | ha
    · exact h4 ⟨ha, hr⟩
    · exact h1 ⟨le_of_eq ha.symm, le_of_lt h⟩
  · push Not at hr
    exact h5 ⟨le_of_lt h, hr⟩

/-- **The colour table never leaves the range of its nodes.** If one component (`π`: the first,
second or third value of a node) of every node of the grid lies in `[lo, hi]`, that component of
`clutEval` at a point of `[0, 1]³` lies in `[lo, hi]`: in each of the six tetrahedra the four
weights are non-negative and add up to one. `applyProfile` clamps its coordinates to `[0, 1]`, so
this holds for every pixel. -/
theorem clutEval_mem (π : Vec3 ℝ → ℝ)
    (hπ : ∀ v0 va vp vq w0 w1 w2 w3,
      π (tetraBlend v0 va vp vq w0 w1 w2 w3) = π v0 * w0 + π va * w1 + π vp * w2 + π vq * w3)
    (p : Profile ℝ) (hd : 1 ≤ p.dim.x ∧ 1 ≤ p.dim.y ∧ 1 ≤ p.dim.z) {lo hi : ℝ}
    (hn : ∀ i j k, i < p.dim.x → j < p.dim.y → k < p.dim.z → lo ≤ π (p.node i j k) ∧ π (p.node i j k) ≤ hi)
    {t : Vec3 ℝ} (hx : 0 ≤ t.x ∧ t.x ≤ 1) (hy : 0 ≤ t.y ∧ t.y ≤ 1) (hz : 0 ≤ t.z ∧ t.z ≤ 1) :
    lo ≤ π (clutEval p t) ∧ π (clutEval p t) ≤ hi := by
  obtain ⟨r0, r1, ri, ri1⟩ := clut_cell_mem hd.1 hx.1 hx.2
  obtain ⟨g0, g1, gi, gi1⟩ := clut_cell_mem hd.2.1 hy.1 hy.2
  obtain ⟨b0, b1, bi, bi1⟩ := clut_cell_mem hd.2.2 hz.1 hz.2
  unfold clutEval
  simp only at r0 r1 g0 g1 b0 b1 ri ri1 gi gi1 bi bi1 ⊢
  split_ifs with c1 c2 c3 c4 c5
  · obtain ⟨h1, h2⟩ := c1
    rw [hπ]
    exact blend4_mem (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption))
      (by linarith) (by linarith) (by linarith) (by linarith) (by norm_num)
  · obtain ⟨h1, h2⟩ := c2
    rw [hπ]
    exact blend4_mem (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption))
      (by linarith) (by linarith) (by linarith) (by linarith) (by norm_num)
  · obtain ⟨h1, h2⟩ := c3
    rw [hπ]
    exact blend4_mem (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption))
      (by linarith) (by linarith) (by linarith) (by linarith) (by norm_num)
  · obtain ⟨h1, h2⟩ := c4
    rw [hπ]
    exact blend4_mem (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption))
      (by linarith) (by linarith) (by linarith) (by linarith) (by norm_num)
  · obtain ⟨h1, h2⟩ := c5
    rw [hπ]
    exact blend4_mem (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption))
      (by linarith) (by linarith) (by linarith) (by linarith) (by norm_num)
  · obtain ⟨h1, h2⟩ := clut_last_case c1 c2 c3 c4 c5
    rw [hπ]
    exact blend4_mem (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption)) (hn _ _ _ (by assumption) (by assumption) (by assumption))
      (by linarith) (by linarith) (by linarith) (by linarith) (by norm_num)

/-! ## The shoulder -/

/-- At and below its knee the shoulder is the identity. -/
theorem applyShoulderChannel_of_le {s : Shoulder ℝ} {x : ℝ} (h : x ≤ s.xK) :
    applyShoulderChannel s x = x := by
  unfold applyShoulderChannel
  exact ite_eq_right (not_lt.mpr h)

/-- From `x_hi` on the shoulder stays at its ceiling. -/
theorem applyShoulderChannel_of_ge {s : Shoulder ℝ} {x : ℝ} (h1 : s.xK < x) (h2 : s.tHi ≤ log2 x) :
    applyShoulderChannel s x = s.yC := by
  unfold applyShoulderChannel
  rw [ite_eq_left h1]
  exact ite_eq_left h2

/-- Between the knee and `x_hi` the shoulder is a parabola in the encoded value: the picture's
code is `C - (C - s_k) u^2` with `u = 1 - (t - t_k) / (t_hi - t_k)` and `t = log2 x`, as long as
that code lies on the sRGB curve's power segment (above `0.04045`; the knee's own code is far
above it). -/
theorem srgbEncode_applyShoulderChannel {s : Shoulder ℝ} {x : ℝ} (h1 : s.xK < x)
    (h2 : log2 x < s.tHi)
    (h3 : 0.04045 < s.c - (s.c - s.sK) *
      ((1 - (log2 x - s.tK) / (s.tHi - s.tK)) * (1 - (log2 x - s.tK) / (s.tHi - s.tK)))) :
    srgbEncode (applyShoulderChannel s x) =
      s.c - (s.c - s.sK) *
        ((1 - (log2 x - s.tK) / (s.tHi - s.tK)) * (1 - (log2 x - s.tK) / (s.tHi - s.tK))) := by
  have e1 : applyShoulderChannel s x = srgbDecode (s.c - (s.c - s.sK) *
      ((1 - (log2 x - s.tK) / (s.tHi - s.tK)) * (1 - (log2 x - s.tK) / (s.tHi - s.tK)))) := by
    unfold applyShoulderChannel
    rw [ite_eq_left h1, ite_eq_right (not_le.mpr h2)]
    norm_num
  rw [e1, srgbEncode_srgbDecode_of_gt h3]

/-- With the shoulder off, a pixel passes through it unchanged. -/
theorem applyShoulder_of_off {s : Shoulder ℝ} (h : s.xHi ≤ 1.0001) (v : Vec3 ℝ) :
    applyShoulder s v = v := by
  unfold applyShoulder
  exact ite_eq_right (not_lt.mpr h)

/-- An `x_hi` at or below 1.0001 switches the shoulder off. -/
theorem setShoulder_of_le {x : ℝ} (h : x ≤ 1.0001) : setShoulder x = Shoulder.off := by
  unfold setShoulder
  exact ite_eq_left h

/-- With the shoulder off, the display value is the pre-shoulder value held to `[0, 1]`. -/
theorem toDisplay_off (v : Vec3 ℝ) : toDisplay (Shoulder.off : Shoulder ℝ) v = v.map clamp01 := by
  have h : (Shoulder.off : Shoulder ℝ).xHi ≤ 1.0001 := by
    simp only [Shoulder.off]
    norm_num
  unfold toDisplay
  rw [applyShoulder_of_off h]

/-- Every channel of a display value is in `[0, 1]`. -/
theorem toDisplay_mem (s : Shoulder ℝ) (v : Vec3 ℝ) :
    (0 ≤ (toDisplay s v).x ∧ (toDisplay s v).x ≤ 1) ∧
    (0 ≤ (toDisplay s v).y ∧ (toDisplay s v).y ≤ 1) ∧
    (0 ≤ (toDisplay s v).z ∧ (toDisplay s v).z ≤ 1) :=
  ⟨clamp01_mem _, clamp01_mem _, clamp01_mem _⟩

/-! ## The whole pixel -/

/-- Without a tone curve the output is the encoded display value. -/
theorem convertPixel_of_no_tone (c : PipelineConfig ℝ) (h : c.tone = none) (m : Mat3 ℝ) (v : Vec3 ℝ) :
    convertPixel c m v = encodeOutputWith c.encoding (convertPixelDisplay c m v) := by
  unfold convertPixel PipelineConfig.activeTone
  rw [h]

/-- A tone table of fewer than two entries is no tone stage, as `convert_pixel`'s test
`cfg.tone.n > 0` makes it. -/
theorem convertPixel_of_short_tone (c : PipelineConfig ℝ) (lut : Array ℝ) (h : c.tone = some lut)
    (hs : lut.size < 2) (m : Mat3 ℝ) (v : Vec3 ℝ) :
    convertPixel c m v = encodeOutputWith c.encoding (convertPixelDisplay c m v) := by
  have : c.activeTone = none := by
    unfold PipelineConfig.activeTone
    rw [h]
    exact ite_eq_right (fun hc => absurd hc.2 (by omega))
  unfold convertPixel
  rw [this]

/-- With the shoulder off the display value is the pre-shoulder value held to `[0, 1]`. -/
theorem convertPixelDisplay_of_off (c : PipelineConfig ℝ) (h : c.shoulder = Shoulder.off) (m : Mat3 ℝ)
    (v : Vec3 ℝ) :
    convertPixelDisplay c m v = (convertPixelLinear c.profile c.knee m v).map clamp01 := by
  have : c.activeShoulder = Shoulder.off := by
    unfold PipelineConfig.activeShoulder
    split_ifs <;> simp [h]
  unfold convertPixelDisplay
  rw [this, toDisplay_off]

/-- **The linear output** (`-P srgb-g10`) is the pre-shoulder linear sRGB held to `[0, 1]`: no
shoulder, no tone curve, no sRGB curve. -/
theorem convertPixel_linear (c : PipelineConfig ℝ) (h : c.encoding = .linear) (m : Mat3 ℝ) (v : Vec3 ℝ) :
    convertPixel c m v = (convertPixelLinear c.profile c.knee m v).map clamp01 := by
  have hs : c.activeShoulder = Shoulder.off := by
    unfold PipelineConfig.activeShoulder; rw [h]; rfl
  have ht : c.activeTone = none := by
    unfold PipelineConfig.activeTone
    split <;> simp [h]
  have hd : convertPixelDisplay c m v = (convertPixelLinear c.profile c.knee m v).map clamp01 := by
    unfold convertPixelDisplay; rw [hs, toDisplay_off]
  unfold convertPixel
  rw [ht, h, hd]
  simp only [encodeOutputWith, Vec3.map]
  ext <;> simp only <;> exact Scalar.clamp01_of_mem (clamp01_mem _).1 (clamp01_mem _).2

/-- Every channel of the encoded output is in `[0, 1]`. -/
theorem encodeOutput_mem (v : Vec3 ℝ) :
    (0 ≤ (encodeOutput v).x ∧ (encodeOutput v).x ≤ 1) ∧
    (0 ≤ (encodeOutput v).y ∧ (encodeOutput v).y ≤ 1) ∧
    (0 ≤ (encodeOutput v).z ∧ (encodeOutput v).z ≤ 1) :=
  ⟨clamp01_mem _, clamp01_mem _, clamp01_mem _⟩

/-! ## Linear sRGB -/

/-- **The PCS white maps to `(1, 1, 1)`**, to within `1e-8` per channel: the matrix is the inverse of
the embedded profile's colorants, whose sum is the PCS white, rounded to nine significant digits. -/
theorem pcsToLinearSrgb_white :
    |(pcsToLinearSrgbUnclamped (pcsWhite : Vec3 ℝ)).x - 1| ≤ 1e-8 ∧
    |(pcsToLinearSrgbUnclamped (pcsWhite : Vec3 ℝ)).y - 1| ≤ 1e-8 ∧
    |(pcsToLinearSrgbUnclamped (pcsWhite : Vec3 ℝ)).z - 1| ≤ 1e-8 := by
  simp only [pcsToLinearSrgbUnclamped, pcsToLinearSrgbMatrix, pcsWhite, Mat3.mulVec, Vec3.dot]
  refine ⟨?_, ?_, ?_⟩ <;> rw [abs_le] <;> constructor <;> norm_num

end Negicc
