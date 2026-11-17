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

/-- The search's units. On a narrow-band scan the search works on the merged matrix divided by
`e_norm` and its exposure is in reference units; the conversion works on the merged matrix with
the exposure divided by `e_norm`. Both are the same matrix, so both render the same picture. -/
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
    convertPixel c m v = encodeOutput (convertPixelDisplay c m v) := by
  unfold convertPixel
  rw [h]

/-- With the shoulder off the display value is the pre-shoulder value held to `[0, 1]`. -/
theorem convertPixelDisplay_of_off (c : PipelineConfig ℝ) (h : c.shoulder = Shoulder.off) (m : Mat3 ℝ)
    (v : Vec3 ℝ) :
    convertPixelDisplay c m v = (convertPixelLinear c.profile c.knee m v).map clamp01 := by
  unfold convertPixelDisplay
  rw [h, toDisplay_off]

/-- Every channel of the encoded output is in `[0, 1]`. -/
theorem encodeOutput_mem (v : Vec3 ℝ) :
    (0 ≤ (encodeOutput v).x ∧ (encodeOutput v).x ≤ 1) ∧
    (0 ≤ (encodeOutput v).y ∧ (encodeOutput v).y ≤ 1) ∧
    (0 ≤ (encodeOutput v).z ∧ (encodeOutput v).z ≤ 1) :=
  ⟨clamp01_mem _, clamp01_mem _, clamp01_mem _⟩

end Negicc
