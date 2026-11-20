/-
Proved properties of the Auto search over the real numbers.
-/
module

public import Negicc.Base.Real
public import Negicc.Auto.Defs

@[expose] public section

/-!
# Properties of the Auto search

What a caller of the search relies on, proved over the real numbers (`Scalar ℝ`). Nothing here
covers the `Float` reading: rounding is not modelled (see the README).

* **Geometry.** The inner crop is centred and keeps at least one pixel; the network's grid has at
  least one cell per side and its input is a whole number of patches.
* **The candidates' bounds.** Every exposure the optimiser returns lies in its window when `E` is
  searched (in `[0.20, 3.0]` from a start without a window), and is the start's when `E` is pinned; from a start, `g` moves by at most 0.06 and `b`
  by at most 0.08 in one optimisation; the trust region lies inside the window; the metered
  exposure lies in `[2^-6, 2^6]`; the highlight pullback's nudge raises `E` by at most 0.15 stop.
* **The objective.** Its z-scores are invariant to a common shift of the render and the field,
  which is what aiming at `μ` minus the correction means; the residual is never negative and is
  zero for a render equal to the field; the lightness wish fixes `L*` whatever the network
  predicted; the penalties are never negative and vanish inside the calibrated range.
* **The brackets.** A bracket's usable range is inverted exactly when the bracket is inadmissible,
  and its calibrated range and flags do not depend on the scene's exposure except through a
  common factor.
* **Termination.** The fixed-point loop runs at least one and at most `maxSteps` steps; the
  winner is a run of the tournament with the lowest score, converged when one converged and that
  is preferred.
* **Tone weights** sum to 1.
-/

namespace Negicc

open Scalar

/-! ## Geometry -/

/-- The inner crop is centred and keeps at least one pixel each way: its two margins and its width
make up the frame, and so do the height's. -/
theorem innerCrop_spec (W H : Nat) (r : ℝ) (hW : 0 < W) (hH : 0 < H) :
    2 * (innerCrop W H r).1 + (innerCrop W H r).2.2.1 = W ∧ 2 * (innerCrop W H r).2.1 + (innerCrop W H r).2.2.2 = H ∧
    0 < (innerCrop W H r).2.2.1 ∧ 0 < (innerCrop W H r).2.2.2 := by
  unfold innerCrop
  have : (W = 0 || H = 0) = false := by simp; omega
  simp only [this, Bool.false_eq_true, ↓reduceIte]
  split_ifs
  · simp; omega
  · simp only
    have h1 : min (roundNat ((W : ℝ) * (1.0 - r) * 0.5)) ((W - 1) / 2) ≤ (W - 1) / 2 := min_le_right _ _
    have h2 : min (roundNat ((H : ℝ) * (1.0 - r) * 0.5)) ((H - 1) / 2) ≤ (H - 1) / 2 := min_le_right _ _
    omega

/-- The network's grid has at least one cell each way, and its input is a whole number of
patches. -/
theorem aspectShape_grid (w h m p : Nat) :
    1 ≤ (aspectShape (α := ℝ) w h m p).gridW ∧ 1 ≤ (aspectShape (α := ℝ) w h m p).gridH ∧
    (aspectShape (α := ℝ) w h m p).targetW = (aspectShape (α := ℝ) w h m p).gridW * max p 1 ∧
    (aspectShape (α := ℝ) w h m p).targetH = (aspectShape (α := ℝ) w h m p).gridH * max p 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [aspectShape, gridSide]

/-! ## Tone weights -/

/-- A cell's three tone weights sum to 1: the shadow and the highlight ramps do not overlap. -/
theorem toneBins_sum (L : ℝ) : (toneBins L).x + (toneBins L).y + (toneBins L).z = 1 := by
  simp only [toneBins, clamp_real]
  set sh := min (max ((30.0 + 10.0 / 2.0 - L) / 10.0) 0.0) 1.0 with hsh
  set hi := min (max ((L - (80.0 - 10.0 / 2.0)) / 10.0) 0.0) 1.0 with hhi
  have s0 : 0 ≤ sh := le_min (by norm_num) (by norm_num)
  have s1 : sh ≤ 1 := by rw [hsh]; exact (min_le_right _ _).trans (by norm_num)
  have h0 : 0 ≤ hi := le_min (by norm_num) (by norm_num)
  have h1 : hi ≤ 1 := by rw [hhi]; exact (min_le_right _ _).trans (by norm_num)
  have hsum : sh + hi ≤ 1 := by
    by_cases hL : L ≤ 35
    · have : hi = 0 := by
        rw [hhi]
        have hm : max ((L - (80.0 - 10.0 / 2.0)) / 10.0) (0.0 : ℝ) = 0.0 := max_eq_right (by norm_num; linarith)
        rw [hm]; norm_num
      linarith
    · have : sh = 0 := by
        rw [hsh]
        have hm : max ((30.0 + 10.0 / 2.0 - L) / 10.0) (0.0 : ℝ) = 0.0 := max_eq_right (by norm_num; linarith)
        rw [hm]; norm_num
      linarith
  have : min (max (1.0 - sh - hi) 0.0) 1.0 = 1 - sh - hi := by
    norm_num
    rw [max_eq_left (by linarith), min_eq_left (by linarith)]
  rw [this]
  ring

/-! ## Penalties -/

theorem clipPenalty_nonneg (clip : ℝ) : 0 ≤ clipPenalty clip := by
  simp only [clipPenalty, sq_real]
  positivity

/-- No clip penalty while at most 2.5 % of the channels clip. -/
theorem clipPenalty_eq_zero {clip : ℝ} (h : clip ≤ 2.5) : clipPenalty clip = 0 := by
  simp only [clipPenalty, sq_real]
  rw [max_eq_left (by norm_num; linarith)]
  norm_num

theorem centrePenalty_nonneg (e c : ℝ) : 0 ≤ centrePenalty e c := by
  simp only [centrePenalty, sq_real]
  split_ifs <;> positivity

/-- No pull at the centre itself. -/
theorem centrePenalty_self {e : ℝ} (he : 0 < e) : centrePenalty e e = 0 := by
  simp only [centrePenalty, sq_real, log2_real]
  have he' : (0.0 : ℝ) < e := by norm_num; exact he
  rw [ite_eq_left he', div_self he.ne']
  simp

theorem enclosurePenalty_nonneg (a b e : ℝ) : 0 ≤ enclosurePenalty a b e := by
  simp only [enclosurePenalty, sq_real]
  positivity

/-- The pullback's penalties vanish for an exposure inside the calibrated range of a feasible
bracket. -/
theorem enclosurePenalty_eq_zero {a b e : ℝ} (ha : 0 < a) (hae : a ≤ e) (heb : e ≤ b) :
    enclosurePenalty a b e = 0 := by
  have he : 0 < e := lt_of_lt_of_le ha hae
  have hb : 0 < b := lt_of_lt_of_le he heb
  simp only [enclosurePenalty, sq_real, log2_real]
  have h1 : Real.logb 2 (a / e) ≤ 0 :=
    Real.logb_nonpos (by norm_num) (div_nonneg ha.le he.le) ((div_le_one he).2 hae)
  have h2 : Real.logb 2 (e / b) ≤ 0 :=
    Real.logb_nonpos (by norm_num) (div_nonneg he.le hb.le) ((div_le_one hb).2 heb)
  have h3 : Real.logb 2 (a / b) ≤ 0 :=
    Real.logb_nonpos (by norm_num) (div_nonneg ha.le hb.le) ((div_le_one hb).2 (hae.trans heb))
  rw [max_eq_left (by norm_num; exact h1), max_eq_left (by norm_num; exact h2),
    max_eq_left (by norm_num; exact h3)]
  norm_num

theorem penalty_nonneg (cfg : OptConfig ℝ) (e clip c : ℝ) : 0 ≤ penalty cfg e clip c := by
  unfold penalty
  split_ifs
  · exact add_nonneg (clipPenalty_nonneg _) (enclosurePenalty_nonneg _ _ _)
  · exact add_nonneg (clipPenalty_nonneg _) (centrePenalty_nonneg _ _)

/-! ## The objective -/

/-- The z-scores are invariant to a common shift of the render and the field. -/
theorem zsq_shift (x m s c : Lab ℝ) : zsq (x.add c) (m.add c) s = zsq x m s := by
  simp only [zsq, Lab.add]
  congr 1 <;> ring_nf

/-- Aiming the optimiser at `μ` minus a correction `c` scores a cell exactly as scoring the cell
plus `c` against `μ`: at the optimiser's optimum the corrected cells, the picture's, match the
field. -/
theorem zsq_aim (x m s c : Lab ℝ) : zsq x (m.sub c) s = zsq (x.add c) m s := by
  simp only [zsq, Lab.add, Lab.sub]
  congr 1 <;> ring_nf

/-- A cell's squared z-scores are not negative. -/
theorem zsq_nonneg (x m s : Lab ℝ) : 0 ≤ (zsq x m s).L ∧ 0 ≤ (zsq x m s).a ∧ 0 ≤ (zsq x m s).b := by
  simp only [zsq, sq_real]
  exact ⟨sq_nonneg _, sq_nonneg _, sq_nonneg _⟩

/-- A cell equal to the field's mean has zero z-scores. -/
theorem zsq_self (m s : Lab ℝ) : zsq m m s = ⟨0, 0, 0⟩ := by
  simp [zsq, sq_real]

/-- A sum built by adding non-negative terms to a non-negative start is not negative. -/
theorem foldl_add_nonneg {β : Type} (f : β → ℝ) (hf : ∀ i, 0 ≤ f i) (l : List β) (a : ℝ) (ha : 0 ≤ a) :
    0 ≤ l.foldl (fun acc i => acc + f i) a := by
  induction l generalizing a with
  | nil => simpa using ha
  | cons i rest ih => exact ih _ (add_nonneg ha (hf i))

/-- Adding zero terms leaves the start. -/
theorem foldl_add_zero {β : Type} (f : β → ℝ) (hf : ∀ i, f i = 0) (l : List β) (a : ℝ) :
    l.foldl (fun acc i => acc + f i) a = a := by
  induction l generalizing a with
  | nil => rfl
  | cons i rest ih => rw [List.foldl_cons, hf i, add_zero]; exact ih a

/-- The residual is never negative. -/
theorem z2_nonneg (cells mu sigma : Array (Lab ℝ)) : 0 ≤ z2 cells mu sigma := by
  simp only [z2]
  refine div_nonneg (foldl_add_nonneg _ (fun i => ?_) _ _ (by norm_num)) (Nat.cast_nonneg _)
  obtain ⟨h1, h2, h3⟩ := zsq_nonneg (labAt cells i) (labAt mu i) (labAt sigma i)
  have : (3.0 : ℝ) = 3 := by norm_num
  rw [this]
  positivity

/-- A render equal to the field has zero residual. -/
theorem z2_self (mu sigma : Array (Lab ℝ)) : z2 mu mu sigma = 0 := by
  simp only [z2]
  rw [foldl_add_zero _ (fun i => by simp [zsq_self])]
  norm_num

/-- With the lightness wish in place, a field's `L*` is the wish's, whatever the network
predicted, for every cell. -/
theorem withWish_L (wm ws : Array ℝ) (f : IntentField ℝ) (h : wm.size = f.mu.size) (i : Nat)
    (hi : i < wm.size) :
    ((withWish wm ws f).mu[i]?.map (·.L)) = some wm[i] := by
  simp only [withWish, h, ↓reduceIte]
  rw [Array.getElem?_zipWith]
  simp [Array.getElem?_eq_getElem hi, Array.getElem?_eq_getElem (h ▸ hi)]

/-- With the lightness wish in place, two predictions that agree in `a*` and `b*` give the same
`μ`. -/
theorem withWish_mu_congr (wm ws : Array ℝ) (f g : IntentField ℝ) (hs : f.mu.size = g.mu.size)
    (hab : f.mu.map (fun m => (m.a, m.b)) = g.mu.map (fun m => (m.a, m.b))) (h : wm.size = f.mu.size) :
    (withWish wm ws f).mu = (withWish wm ws g).mu := by
  simp only [withWish, h, ← hs, ↓reduceIte]
  apply Array.ext
  · simp [hs]
  · intro i h1 h2
    simp only [Array.getElem_zipWith]
    have := congrArg (fun a => a[i]?) hab
    simp only [Array.getElem?_map] at this
    have hf : i < f.mu.size := by simp at h1; omega
    have hg : i < g.mu.size := by omega
    rw [Array.getElem?_eq_getElem hf, Array.getElem?_eq_getElem hg] at this
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at this
    obtain ⟨ha, hb⟩ := this
    cases hfm : f.mu[i]
    cases hgm : g.mu[i]
    rw [hfm, hgm] at ha hb
    simp_all

/-! ## Trust region and nudge -/

/-- The tournament's trust region lies inside the window and is not empty, for a screened exposure
in the window. -/
theorem trustBounds_mem {lo hi e c : ℝ} (he : 0 < e) (hlo : lo ≤ e) (hhi : e ≤ hi) :
    lo ≤ (trustBounds lo hi e c).1 ∧ (trustBounds lo hi e c).1 ≤ (trustBounds lo hi e c).2.1 ∧
    (trustBounds lo hi e c).2.1 ≤ hi := by
  simp only [trustBounds, pow_real]
  have h1 : (2.0 : ℝ) ^ (-0.5 : ℝ) ≤ 1 := Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (by norm_num)
  have h2 : (1 : ℝ) ≤ 2.0 ^ (0.5 : ℝ) := Real.one_le_rpow (by norm_num) (by norm_num)
  have a1 : e * 2.0 ^ (-0.5 : ℝ) ≤ e := by nlinarith
  have a2 : e ≤ e * 2.0 ^ (0.5 : ℝ) := by nlinarith
  refine ⟨le_max_left _ _, ?_, min_le_left _ _⟩
  apply max_le
  · exact le_min (hlo.trans hhi) (hlo.trans a2)
  · exact le_min (a1.trans hhi) (a1.trans a2)

/-- The highlight pullback's nudge raises `E` by at most 0.15 stop and never past the calibrated
range's ends. -/
theorem pullbackNudge_bound {a b e ep : ℝ} (h : pullbackNudge a b e = some ep) :
    e < ep ∧ ep ≤ e * 2.0 ^ (0.15 : ℝ) ∧ ep ≤ a ∧ ep ≤ b := by
  unfold pullbackNudge at h
  simp only [pow_real] at h
  split_ifs at h with hlt
  cases h
  refine ⟨hlt, ?_, ?_, min_le_left _ _⟩
  · exact (min_le_right _ _).trans (min_le_right _ _)
  · exact (min_le_right _ _).trans (min_le_left _ _)

/-- The clip fallback takes the next bracket only when it at least halves the clipping, moves the
median `L*` by −4.5 to +0.5 and keeps the residual within 10 %. -/
theorem clipFallbackVerdict_taken {c0 c1 dl r1 r0 : ℝ}
    (h : clipFallbackVerdict c0 c1 dl r1 r0 = .taken) :
    c1 ≤ 0.5 * c0 ∧ -4.5 ≤ dl ∧ dl ≤ 0.5 ∧ r1 ≤ (1.0 + 0.10) * r0 := by
  unfold clipFallbackVerdict at h
  split_ifs at h with h1 h2 h3
  have h2' : -4.5 ≤ dl ∧ dl ≤ 0.5 := by simpa using h2
  exact ⟨le_of_not_gt h1, h2'.1, h2'.2, le_of_not_gt h3⟩

/-! ## The first smallest element and the winner -/

theorem minFirst_fold_spec {β : Type} (score : β → ℝ) (xs : List β) (x0 : β) :
    let r := xs.foldl (fun best y => if decide (score y < score best) then y else best) x0
    r ∈ x0 :: xs ∧ ∀ y ∈ x0 :: xs, score r ≤ score y := by
  induction xs generalizing x0 with
  | nil => simp
  | cons z rest ih =>
    simp only [List.foldl_cons]
    by_cases h : score z < score x0
    · simp only [h, decide_true, ↓reduceIte]
      obtain ⟨hm, hle⟩ := ih z
      refine ⟨List.mem_cons_of_mem _ hm, fun y hy => ?_⟩
      rcases List.mem_cons.1 hy with rfl | hy
      · exact (hle z (List.mem_cons_self)).trans h.le
      · exact hle y hy
    · simp only [h, decide_false, Bool.false_eq_true, ↓reduceIte]
      obtain ⟨hm, hle⟩ := ih x0
      refine ⟨?_, fun y hy => ?_⟩
      · rcases List.mem_cons.1 hm with h1 | h1
        · rw [h1]; exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h1)
      · rcases List.mem_cons.1 hy with rfl | hy
        · exact hle _ List.mem_cons_self
        · rcases List.mem_cons.1 hy with rfl | hy
          · exact (hle x0 List.mem_cons_self).trans (not_lt.1 h)
          · exact hle y (List.mem_cons_of_mem _ hy)

/-- The first smallest element under a score is an element of the list and scores no more than
any element. -/
theorem minFirst_spec {β : Type} (score : β → ℝ) (l : List β) (x : β)
    (h : minFirst (fun a b => decide (score a < score b)) l = some x) :
    x ∈ l ∧ ∀ y ∈ l, score x ≤ score y := by
  cases l with
  | nil => simp [minFirst] at h
  | cons x0 xs =>
    simp only [minFirst, Option.some.injEq] at h
    subst h
    exact minFirst_fold_spec score xs x0

/-- The tournament's winner is one of its runs, scores no more than any run it was chosen among,
and, when converged runs are preferred and one converged, is converged. -/
theorem winner_spec {R : Type} (runs : List R) (pref : Bool) (conv : R → Bool) (score : R → ℝ) (w : R)
    (h : winner runs pref conv score = some w) :
    w ∈ runs ∧ (∀ r ∈ runs, (!(pref && runs.any conv) || conv r) = true → score w ≤ score r) ∧
    ((pref && runs.any conv) = true → conv w = true) := by
  unfold winner at h
  obtain ⟨hm, hle⟩ := minFirst_spec score _ w h
  rw [List.mem_filter] at hm
  refine ⟨hm.1, fun r hr hc => hle r (List.mem_filter.2 ⟨hr, hc⟩), fun hp => ?_⟩
  have := hm.2
  simp only [hp, Bool.not_true, Bool.false_or] at this
  exact this

/-! ## Termination of the fixed-point loop -/

theorem fpGo_steps {α P F : Type} [Scalar α] (m : P → Option (P × α × F)) (s : F → P → P) (d : P → P → α)
    (c : ConvRule α) (fuel it : Nat) (curr : P) (prev : α) (acc : Option ((P × α × F) × F)) (steps : Nat)
    (r : FPResult α P F) (h : fpGo m s d c it fuel curr prev acc steps = some r) :
    steps ≤ r.steps ∧ r.steps ≤ steps + fuel := by
  induction fuel generalizing it curr prev acc steps with
  | zero =>
    simp only [fpGo] at h
    cases acc with
    | none => simp at h
    | some b => simp only [Option.map_some, Option.some.injEq] at h; subst h; simp
  | succ n ih =>
    simp only [fpGo] at h
    split at h
    · simp at h
    · split_ifs at h
      · simp only [Option.some.injEq] at h; subst h; simp
      · simp only [Option.some.injEq] at h; subst h; simp
      · have := ih _ _ _ _ _ h
        omega

theorem fpGo_converged {α P F : Type} [Scalar α] (m : P → Option (P × α × F)) (s : F → P → P) (d : P → P → α)
    (c : ConvRule α) (fuel it : Nat) (curr : P) (prev : α) (acc : Option ((P × α × F) × F)) (steps : Nat)
    (r : FPResult α P F) (hit : it = steps) (h : fpGo m s d c it fuel curr prev acc steps = some r)
    (hc : r.status = .converged) : 2 ≤ r.steps := by
  induction fuel generalizing it curr prev acc steps with
  | zero =>
    simp only [fpGo] at h
    cases acc with
    | none => simp at h
    | some b => simp only [Option.map_some, Option.some.injEq] at h; subst h; simp at hc
  | succ n ih =>
    simp only [fpGo] at h
    split at h
    · simp at h
    · split_ifs at h with h1 h2
      · simp only [Option.some.injEq] at h; subst h; simp at hc
      · simp only [Option.some.injEq] at h; subst h
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h2
        simp only
        omega
      · exact ih _ _ _ _ _ (by omega) h

/-- The fixed-point loop ends: it runs at least one step and at most `max 1 maxSteps`, and a loop
that converged ran at least two (convergence is judged from the second step on). -/
theorem fixedPoint_steps {α P F : Type} [Scalar α] (m : P → Option (P × α × F)) (s : F → P → P) (d : P → P → α)
    (c : ConvRule α) (maxSteps : Nat) (start : P) (r : FPResult α P F)
    (h : fixedPoint m s d c maxSteps start = some r) :
    1 ≤ r.steps ∧ r.steps ≤ max 1 maxSteps ∧ (r.status = .converged → 2 ≤ r.steps) := by
  unfold fixedPoint at h
  refine ⟨?_, by simpa using (fpGo_steps m s d c _ 0 start 0.0 none 0 r h).2, fun hc =>
    fpGo_converged m s d c _ 0 start 0.0 none 0 r rfl h hc⟩
  obtain ⟨k, hk⟩ : ∃ k, max 1 maxSteps = k + 1 := ⟨max 1 maxSteps - 1, by omega⟩
  rw [hk] at h
  simp only [fpGo] at h
  split at h
  · simp at h
  · split_ifs at h
    · simp only [Option.some.injEq] at h; subst h; simp
    · simp only [Option.some.injEq] at h; subst h; simp
    · exact (fpGo_steps m s d c _ _ _ _ _ _ r h).1

/-! ## The optimiser's candidates -/

theorem firstMin_fold_spec {β : Type} (score : β → ℝ) (xs : List β) (b : β × ℝ) (hb : b.2 = score b.1) :
    let r := xs.foldl (fun best y => let s := score y; if s < best.2 then (y, s) else best) b
    (r.1 = b.1 ∨ r.1 ∈ xs) ∧ r.2 = score r.1 := by
  induction xs generalizing b with
  | nil => simp [hb]
  | cons z rest ih =>
    simp only [List.foldl_cons]
    split_ifs with h
    · obtain ⟨h1, h2⟩ := ih (z, score z) rfl
      refine ⟨?_, h2⟩
      rcases h1 with h1 | h1
      · exact Or.inr (by rw [h1]; exact List.mem_cons_self)
      · exact Or.inr (List.mem_cons_of_mem _ h1)
    · obtain ⟨h1, h2⟩ := ih b hb
      refine ⟨?_, h2⟩
      rcases h1 with h1 | h1
      · exact Or.inl h1
      · exact Or.inr (List.mem_cons_of_mem _ h1)

/-- The first lowest element is an element of the list, with its score. -/
theorem firstMin_mem {β : Type} (score : β → ℝ) (l : List β) (x : β) (v : ℝ) (h : firstMin score l = some (x, v)) :
    x ∈ l ∧ v = score x := by
  cases l with
  | nil => simp [firstMin] at h
  | cons x0 xs =>
    simp only [firstMin, Option.some.injEq] at h
    obtain ⟨h1, h2⟩ := firstMin_fold_spec score xs (x0, score x0) rfl
    rw [h] at h1 h2
    simp only at h1 h2
    refine ⟨?_, h2⟩
    rcases h1 with h1 | h1
    · rw [h1]; exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ h1

theorem sortUnique_sub (xs : List ℝ) (x : ℝ) (h : x ∈ sortUnique xs) : x ∈ xs := by
  have key : ∀ (l : List ℝ), x ∈ l.foldr dedupStep [] → x ∈ l := by
    intro l
    induction l with
    | nil => simp
    | cons a rest ih =>
      intro hx
      simp only [List.foldr_cons, dedupStep] at hx
      split at hx
      · rename_i y t heq
        split_ifs at hx
        · exact List.mem_cons_of_mem _ (ih (heq ▸ hx))
        · rcases List.mem_cons.1 hx with rfl | hx
          · exact List.mem_cons_self
          · exact List.mem_cons_of_mem _ (ih (heq ▸ hx))
      · simp at hx; rw [hx]; exact List.mem_cons_self
  exact (List.mem_mergeSort).1 (key _ h)

/-- With a window and `E` searched, every exposure candidate lies in the window. -/
theorem eCandidates_mem (cfg : OptConfig ℝ) (pins : Pins) (init : Option (Gains ℝ)) (bd : EBounds ℝ)
    (hp : pins.e = false) (hb : bd.lo ≤ bd.hi) :
    (eCandidates cfg pins init (some bd)).2 = bd ∧
    ∀ e ∈ (eCandidates cfg pins init (some bd)).1, bd.lo ≤ e ∧ e ≤ bd.hi := by
  simp only [eCandidates, hp, Bool.false_eq_true, ↓reduceIte]
  refine ⟨by trivial, fun e he => ?_⟩
  have he' := sortUnique_sub _ _ he
  rcases List.mem_append.1 he' with h1 | h1
  · rcases List.mem_append.1 h1 with h2 | h2
    · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 h2
      have hi' : (i : ℝ) < 12 := by exact_mod_cast List.mem_range.1 hi
      have hstep : 0 ≤ (bd.hi - bd.lo) / 12.0 := div_nonneg (by linarith) (by norm_num)
      constructor
      · nlinarith [Nat.cast_nonneg (α := ℝ) i]
      · have : (i : ℝ) * ((bd.hi - bd.lo) / 12.0) ≤ 12 * ((bd.hi - bd.lo) / 12.0) :=
          mul_le_mul_of_nonneg_right hi'.le hstep
        have : (12 : ℝ) * ((bd.hi - bd.lo) / 12.0) = bd.hi - bd.lo := by norm_num; ring
        linarith
    · simp at h2; subst h2; constructor <;> linarith
  · cases init with
    | none => simp at h1
    | some i =>
      simp only [List.mem_filterMap] at h1
      obtain ⟨d, _, hd⟩ := h1
      split_ifs at hd with hc
      cases hd
      exact hc

theorem eRefine_mem (score : ℝ → ℝ) (bd : EBounds ℝ) (base : ℝ) (l : List ℝ) (st : ℝ × ℝ)
    (h : bd.lo ≤ st.1 ∧ st.1 ≤ bd.hi) :
    bd.lo ≤ (l.foldl (eRefineStep score bd base) st).1 ∧ (l.foldl (eRefineStep score bd base) st).1 ≤ bd.hi := by
  induction l generalizing st with
  | nil => exact h
  | cons de rest ih =>
    apply ih
    simp only [eRefineStep]
    split_ifs with h1 h2
    · exact h
    · push Not at h1; exact h1
    · exact h

/-- Step 1 keeps `E` in the window when every candidate is in it. -/
theorem ePass_mem (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (refine : Bool) (cands : List ℝ)
    (bd : EBounds ℝ) (g b : ℝ) (hb : bd.lo ≤ bd.hi) (hc : ∀ e ∈ cands, bd.lo ≤ e ∧ e ≤ bd.hi) :
    bd.lo ≤ (ePass loss cfg refine cands bd g b).1 ∧ (ePass loss cfg refine cands bd g b).1 ≤ bd.hi := by
  let score := fun (e : ℝ) => let r := loss ⟨e, g, b⟩; r.L + penalty cfg e r.clip bd.centre
  have h0 : bd.lo ≤ ((firstMin score cands).getD (bd.lo, score bd.lo)).1 ∧
      ((firstMin score cands).getD (bd.lo, score bd.lo)).1 ≤ bd.hi := by
    cases hf : firstMin score cands with
    | none => exact ⟨le_refl _, hb⟩
    | some p => exact hc p.1 (firstMin_mem score cands p.1 p.2 hf).1
  show bd.lo ≤ (if !refine then _ else _ : ℝ × ℝ).1 ∧ (if !refine then _ else _ : ℝ × ℝ).1 ≤ bd.hi
  cases refine with
  | false => exact eRefine_mem score bd 0 [] _ h0
  | true => exact eRefine_mem score bd _ _ _ h0

/-- `E` stays in the window through the pullback's second pass. -/
theorem secondPass_e_mem (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (pins : Pins) (cands : List ℝ)
    (bd : EBounds ℝ) (gs bs : List ℝ) (e1 g3 b3 : ℝ) (hb : bd.lo ≤ bd.hi) (hc : ∀ e ∈ cands, bd.lo ≤ e ∧ e ≤ bd.hi)
    (he1 : bd.lo ≤ e1 ∧ e1 ≤ bd.hi) :
    bd.lo ≤ (secondPass loss cfg pins (cands, bd) gs bs e1 g3 b3).e ∧
    (secondPass loss cfg pins (cands, bd) gs bs e1 g3 b3).e ≤ bd.hi := by
  simp only [secondPass]
  split_ifs
  · exact he1
  · exact ePass_mem loss cfg true cands bd g3 b3 hb hc

/-- The optimiser's `E` lies in the range of its candidates' window when every candidate does. -/
theorem optimize_e_mem_of (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (init : Option (Gains ℝ))
    (bounds : Option (EBounds ℝ)) (pins : Pins) (bd : EBounds ℝ)
    (h2 : (eCandidates cfg pins init bounds).2 = bd) (hb : bd.lo ≤ bd.hi)
    (hc : ∀ e ∈ (eCandidates cfg pins init bounds).1, bd.lo ≤ e ∧ e ≤ bd.hi) :
    bd.lo ≤ (optimize loss cfg init bounds pins).1.e ∧ (optimize loss cfg init bounds pins).1.e ≤ bd.hi := by
  have hec : eCandidates cfg pins init bounds = ((eCandidates cfg pins init bounds).1, bd) :=
    Prod.ext rfl h2
  generalize hcs : (eCandidates cfg pins init bounds).1 = cs at hc hec
  have he1 := fun g b => ePass_mem loss cfg (!pins.e) cs bd g b hb hc
  have he2 := fun g b => ePass_mem loss cfg true cs bd g b hb hc
  simp only [optimize, hec]
  split_ifs <;> first
    | exact he2 _ _
    | exact secondPass_e_mem loss cfg pins cs bd _ _ _ _ _ hb hc (he1 _ _)
    | exact he1 _ _

/-- **The exposure's bounds.** With a window `[lo, hi]` (`lo ≤ hi`) and `E` searched, the optimiser
returns an `E` in the window, whatever the loss. -/
theorem optimize_e_mem (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (init : Option (Gains ℝ))
    (bd : EBounds ℝ) (pins : Pins) (hp : pins.e = false) (hb : bd.lo ≤ bd.hi) :
    bd.lo ≤ (optimize loss cfg init (some bd) pins).1.e ∧ (optimize loss cfg init (some bd) pins).1.e ≤ bd.hi := by
  obtain ⟨h2, hc⟩ := eCandidates_mem cfg pins init bd hp hb
  exact optimize_e_mem_of loss cfg init (some bd) pins bd h2 hb hc

/-- **A start without a window.** From a start and with no window, `E` searched, the optimiser
returns an `E` in `[0.20, 3.0]`, wherever the start is. -/
theorem optimize_e_mem_start (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (i : Gains ℝ) (pins : Pins)
    (hp : pins.e = false) :
    0.20 ≤ (optimize loss cfg (some i) none pins).1.e ∧ (optimize loss cfg (some i) none pins).1.e ≤ 3.0 := by
  apply optimize_e_mem_of loss cfg (some i) none pins ⟨0.20, 3.0, clamp i.e 0.20 3.0⟩
  · simp [eCandidates, hp]
  · norm_num
  · intro e he
    simp only [eCandidates, hp, Bool.false_eq_true, ↓reduceIte, List.mem_filter, Bool.and_eq_true,
      decide_eq_true_eq] at he
    exact he.2

theorem optimize_e_pinned_aux (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (init : Option (Gains ℝ))
    (bounds : Option (EBounds ℝ)) (pins : Pins) (hp : pins.e = true) (E0 : ℝ)
    (hcand : eCandidates cfg pins init bounds = ([E0], ⟨E0, E0, E0⟩)) :
    (optimize loss cfg init bounds pins).1.e = E0 := by
  have he1 : ∀ g b, (ePass loss cfg false [E0] ⟨E0, E0, E0⟩ g b).1 = E0 := by
    intro g b; simp [ePass, firstMin]
  simp only [optimize, hcand, hp, Bool.not_true, Bool.and_false, Bool.false_and, Bool.false_eq_true,
    ↓reduceIte]
  split_ifs <;> simp [secondPass, hp, he1]

/-- **A pinned exposure stays.** With `E` pinned the optimiser returns the start's `E`, or the
configured one without a start. -/
theorem optimize_e_pinned (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (init : Option (Gains ℝ))
    (bounds : Option (EBounds ℝ)) (pins : Pins) (hp : pins.e = true) :
    (optimize loss cfg init bounds pins).1.e = (match init with | some i => i.e | none => cfg.pinE) := by
  cases init with
  | none => exact optimize_e_pinned_aux loss cfg none bounds pins hp _ (by simp [eCandidates, hp])
  | some i => exact optimize_e_pinned_aux loss cfg (some i) bounds pins hp _ (by simp [eCandidates, hp])

/-- One refinement move of step 3 changes `g` by `0` or `dg`. -/
theorem chain_fold_bound (loss : Gains ℝ → IntentLoss ℝ) (pins : Pins) (e : ℝ) (moves : List (ℝ × ℝ))
    (st : ℝ × ℝ × ℝ) :
    st.1 + (moves.map (fun d => min 0 d.1)).sum ≤ (moves.foldl (chainStep loss pins e) st).1 ∧
    (moves.foldl (chainStep loss pins e) st).1 ≤ st.1 + (moves.map (fun d => max 0 d.1)).sum ∧
    st.2.1 + (moves.map (fun d => min 0 d.2)).sum ≤ (moves.foldl (chainStep loss pins e) st).2.1 ∧
    (moves.foldl (chainStep loss pins e) st).2.1 ≤ st.2.1 + (moves.map (fun d => max 0 d.2)).sum := by
  induction moves generalizing st with
  | nil => simp
  | cons d rest ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    obtain ⟨h1, h2, h3, h4⟩ := ih (chainStep loss pins e st d)
    have hm1 := min_le_left (0 : ℝ) d.1
    have hm2 := min_le_right (0 : ℝ) d.1
    have hM1 := le_max_left (0 : ℝ) d.1
    have hM2 := le_max_right (0 : ℝ) d.1
    have hn1 := min_le_left (0 : ℝ) d.2
    have hn2 := min_le_right (0 : ℝ) d.2
    have hN1 := le_max_left (0 : ℝ) d.2
    have hN2 := le_max_right (0 : ℝ) d.2
    have key : (st.1 + min 0 d.1 ≤ (chainStep loss pins e st d).1 ∧ (chainStep loss pins e st d).1 ≤ st.1 + max 0 d.1) ∧
        (st.2.1 + min 0 d.2 ≤ (chainStep loss pins e st d).2.1 ∧
          (chainStep loss pins e st d).2.1 ≤ st.2.1 + max 0 d.2) := by
      simp only [chainStep]
      split_ifs <;> (try simp only) <;> refine ⟨⟨?_, ?_⟩, ?_, ?_⟩ <;> linarith
    obtain ⟨⟨k1, k2⟩, k3, k4⟩ := key
    refine ⟨?_, ?_, ?_, ?_⟩ <;> linarith

/-- Step 3 moves each gain by at most 0.02. -/
theorem chain_bound (loss : Gains ℝ → IntentLoss ℝ) (pins : Pins) (e g b best : ℝ) :
    |(chain loss pins e g b best).1 - g| ≤ 0.02 ∧ |(chain loss pins e g b best).2.1 - b| ≤ 0.02 := by
  obtain ⟨h1, h2, h3, h4⟩ := chain_fold_bound loss pins e chainMoves (g, b, best)
  have s1 : ((chainMoves (α := ℝ)).map (fun d => min 0 d.1)).sum = -0.02 := by simp [chainMoves]; norm_num
  have s2 : ((chainMoves (α := ℝ)).map (fun d => max 0 d.1)).sum = 0.02 := by simp [chainMoves]; norm_num
  have s3 : ((chainMoves (α := ℝ)).map (fun d => min 0 d.2)).sum = -0.02 := by simp [chainMoves]; norm_num
  have s4 : ((chainMoves (α := ℝ)).map (fun d => max 0 d.2)).sum = 0.02 := by simp [chainMoves]; norm_num
  rw [s1] at h1
  rw [s2] at h2
  rw [s3] at h3
  rw [s4] at h4
  unfold chain
  simp only at h1 h2 h3 h4
  constructor <;> rw [abs_le] <;> constructor <;> linarith

/-- The first lowest of a non-empty grid is one of its points. -/
theorem gSweep_mem (loss : Gains ℝ → IntentLoss ℝ) (grid : List ℝ) (e b : ℝ) (h : grid ≠ []) :
    (gSweep loss grid e b).1 ∈ grid := by
  unfold gSweep
  cases hf : firstMin (fun g => (loss ⟨e, g, b⟩).a) grid with
  | none => cases grid with
    | nil => exact absurd rfl h
    | cons x xs => simp [firstMin] at hf
  | some p => simpa using (firstMin_mem _ grid p.1 p.2 hf).1

theorem bSweep_mem (loss : Gains ℝ → IntentLoss ℝ) (grid : List ℝ) (e g : ℝ) (h : grid ≠ []) :
    (bSweep loss grid e g).1 ∈ grid := by
  unfold bSweep
  cases hf : firstMin (fun b => (loss ⟨e, g, b⟩).b) grid with
  | none => cases grid with
    | nil => exact absurd rfl h
    | cons x xs => simp [firstMin] at hf
  | some p => simpa using (firstMin_mem _ grid p.1 p.2 hf).1

theorem sweep_chain_near (loss : Gains ℝ → IntentLoss ℝ) (pins : Pins) (e1 g0 b0 b1 : ℝ) (gs bs : List ℝ)
    (hgs : gs ≠ []) (hg : ∀ x ∈ gs, |x - g0| ≤ 0.04) (hbs : bs ≠ []) (hb : ∀ x ∈ bs, |x - b0| ≤ 0.06) :
    |(chain loss pins e1 (gSweep loss gs e1 b1).1 (bSweep loss bs e1 (gSweep loss gs e1 b1).1).1
        ((gSweep loss gs e1 b1).2 + (bSweep loss bs e1 (gSweep loss gs e1 b1).1).2)).1 - g0| ≤ 0.06 ∧
    |(chain loss pins e1 (gSweep loss gs e1 b1).1 (bSweep loss bs e1 (gSweep loss gs e1 b1).1).1
        ((gSweep loss gs e1 b1).2 + (bSweep loss bs e1 (gSweep loss gs e1 b1).1).2)).2.1 - b0| ≤ 0.08 := by
  have mg := hg _ (gSweep_mem loss gs e1 b1 hgs)
  have mb := hb _ (bSweep_mem loss bs e1 (gSweep loss gs e1 b1).1 hbs)
  obtain ⟨c1, c2⟩ := chain_bound loss pins e1 (gSweep loss gs e1 b1).1 (bSweep loss bs e1 (gSweep loss gs e1 b1).1).1
    ((gSweep loss gs e1 b1).2 + (bSweep loss bs e1 (gSweep loss gs e1 b1).1).2)
  constructor
  · calc _ ≤ _ := abs_sub_le _ (gSweep loss gs e1 b1).1 g0
      _ ≤ 0.02 + 0.04 := add_le_add c1 mg
      _ = 0.06 := by norm_num
  · calc _ ≤ _ := abs_sub_le _ (bSweep loss bs e1 (gSweep loss gs e1 b1).1).1 b0
      _ ≤ 0.02 + 0.06 := add_le_add c2 mb
      _ = 0.08 := by norm_num

/-- **The gains' moves from a start.** From a start `(E, g, b)` with `g` and `b` searched, one
optimisation moves `g` by at most 0.06 and `b` by at most 0.08: the grids reach ±0.04 and ±0.06,
the refinement ±0.02 more. In the fixed-point loop this bounds each step's move of the gains. -/
theorem optimize_gains_near (loss : Gains ℝ → IntentLoss ℝ) (cfg : OptConfig ℝ) (i : Gains ℝ)
    (bounds : Option (EBounds ℝ)) (pins : Pins) (hg : pins.g = false) (hb : pins.b = false) :
    |(optimize loss cfg (some i) bounds pins).1.g - i.g| ≤ 0.06 ∧
    |(optimize loss cfg (some i) bounds pins).1.b - i.b| ≤ 0.08 := by
  simp only [optimize, Option.isNone_some, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  apply sweep_chain_near
  · simp [gGrid, hg]
  · intro x hx
    simp only [gGrid, hg, Bool.false_eq_true, ↓reduceIte, List.map_cons, List.map_nil, List.mem_cons,
      List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl <;> norm_num [abs_le]
  · simp [bGrid, hb]
  · intro x hx
    simp only [bGrid, hb, Bool.false_eq_true, ↓reduceIte, List.map_cons, List.map_nil, List.mem_cons,
      List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl <;> norm_num [abs_le]

/-! ## Metering -/

/-- The metering bisection keeps `log2 E` in its interval: the metered exposure lies in
`[2^lo, 2^hi]`. -/
theorem meterGo_mem (lstar : ℝ → ℝ) (n : Nat) (lo hi : ℝ) (h : lo ≤ hi) :
    (2.0 : ℝ) ^ lo ≤ (meterGo lstar n lo hi).1 ∧ (meterGo lstar n lo hi).1 ≤ (2.0 : ℝ) ^ hi := by
  have mono : ∀ x y : ℝ, x ≤ y → (2.0 : ℝ) ^ x ≤ (2.0 : ℝ) ^ y := fun x y hxy =>
    Real.rpow_le_rpow_of_exponent_le (by norm_num) hxy
  induction n generalizing lo hi with
  | zero =>
    simp only [meterGo, pow_real]
    exact ⟨mono _ _ (by linarith), mono _ _ (by linarith)⟩
  | succ k ih =>
    simp only [meterGo, pow_real]
    have hm1 : lo ≤ 0.5 * (lo + hi) := by linarith
    have hm2 : 0.5 * (lo + hi) ≤ hi := by linarith
    split_ifs
    · exact ⟨mono _ _ hm1, mono _ _ hm2⟩
    · obtain ⟨a, b⟩ := ih _ _ hm2
      exact ⟨(mono _ _ hm1).trans a, b⟩
    · obtain ⟨a, b⟩ := ih _ _ hm1
      exact ⟨a, b.trans (mono _ _ hm2)⟩

/-- **The metered exposure's bounds**: `2^-6 ≤ E ≤ 2^6`. -/
theorem exposureForLightness_mem (prof : Profile ℝ) (p : Vec3 ℝ) (g b : ℝ) :
    (2.0 : ℝ) ^ (-6.0 : ℝ) ≤ (exposureForLightness prof p g b).1 ∧
    (exposureForLightness prof p g b).1 ≤ (2.0 : ℝ) ^ (6.0 : ℝ) :=
  meterGo_mem _ 25 _ _ (by norm_num)

/-! ## The brackets -/

/-- **Inverted exactly when inadmissible.** Over the real numbers, with positive patch values,
exposure and scene range, a bracket's usable range is inverted exactly when the scene's span
exceeds the bracket's by more than the two tolerances together (2.7 stops). -/
theorem usableInverted_iff_not_admissible (gMin gMax x gLo gHi : ℝ)
    (hmin : 0 < gMin) (hmax : 0 < gMax) (hx : 0 < x) (hlo : 0 < gLo) (hhi : 0 < gHi) :
    (bracketEnclosure gMin gMax x gLo gHi).usableInverted = !(bracketEnclosure gMin gMax x gLo gHi).admissible := by
  simp only [bracketEnclosure, usableHiEv, usableShEv, pow_real, log2_real]
  have kt : 0 < referenceExposure / x := by
    simp only [referenceExposure, captureExposure]; norm_num; positivity
  set k := referenceExposure / x with hk
  have r1 : Real.logb 2 ((2.0 : ℝ) ^ (1.2 : ℝ)) = 1.2 := by
    rw [show (2.0 : ℝ) = 2 by norm_num, Real.logb_rpow (by norm_num) (by norm_num)]
  have r2 : Real.logb 2 ((2.0 : ℝ) ^ (-1.5 : ℝ)) = -1.5 := by
    rw [show (2.0 : ℝ) = 2 by norm_num, Real.logb_rpow (by norm_num) (by norm_num)]
  have L2 : Real.logb 2 (gMax / gHi * k * (2.0 : ℝ) ^ (1.2 : ℝ)) =
      Real.logb 2 gMax - Real.logb 2 gHi + Real.logb 2 k + 1.2 := by
    rw [Real.logb_mul (by positivity) (by positivity), Real.logb_mul (by positivity) (by positivity),
      Real.logb_div (x := gMax) (y := gHi) (by positivity) (by positivity), r1]
  have L1 : Real.logb 2 (gMin / gLo * k * (2.0 : ℝ) ^ (-1.5 : ℝ)) =
      Real.logb 2 gMin - Real.logb 2 gLo + Real.logb 2 k - 1.5 := by
    rw [Real.logb_mul (by positivity) (by positivity), Real.logb_mul (by positivity) (by positivity),
      Real.logb_div (x := gMin) (y := gLo) (by positivity) (by positivity), r2]
    ring
  have lhs : gMax / gHi * k * (2.0 : ℝ) ^ (1.2 : ℝ) < gMin / gLo * k * (2.0 : ℝ) ^ (-1.5 : ℝ) ↔
      Real.logb 2 gMax - Real.logb 2 gHi + Real.logb 2 k + 1.2 <
        Real.logb 2 gMin - Real.logb 2 gLo + Real.logb 2 k - 1.5 := by
    rw [← L1, ← L2, Real.logb_lt_logb_iff (b := 2) (by norm_num) (by positivity) (by positivity)]
  have l1 : Real.logb 2 (gHi / gLo) = Real.logb 2 gHi - Real.logb 2 gLo :=
    Real.logb_div (by positivity) (by positivity)
  have l2 : Real.logb 2 (gMax / gMin) = Real.logb 2 gMax - Real.logb 2 gMin :=
    Real.logb_div (by positivity) (by positivity)
  rw [l1, l2]
  by_cases hc : gMax / gHi * k * (2.0 : ℝ) ^ (1.2 : ℝ) < gMin / gLo * k * (2.0 : ℝ) ^ (-1.5 : ℝ)
  · have h' := lhs.1 hc
    have hn : ¬ (Real.logb 2 gHi - Real.logb 2 gLo ≤ Real.logb 2 gMax - Real.logb 2 gMin + 1.5 + 1.2) := by
      intro hh; linarith
    simp only [hn, decide_false, Bool.not_false]
    exact decide_eq_true hc
  · have hn : Real.logb 2 gHi - Real.logb 2 gLo ≤ Real.logb 2 gMax - Real.logb 2 gMin + 1.5 + 1.2 := by
      by_contra hh; push Not at hh; exact hc (lhs.2 (by linarith))
    simp only [hn, decide_true, Bool.not_true]
    exact decide_eq_false hc

/-- **The scene's exposure only scales the calibrated range.** Scaling the scene's green range by
`k > 0` divides `e_min`, `e_max` by `k` and changes neither the flags nor the fill ratio. This is
why the white-light search may scale `g_lo`, `g_hi` by `e_norm` instead of dividing the matrix by
it. -/
theorem bracketEnclosure_scale (gMin gMax x gLo gHi k : ℝ) (hk : 0 < k) (hlo : 0 < gLo) (hhi : 0 < gHi) :
    let e := bracketEnclosure gMin gMax x gLo gHi
    let e' := bracketEnclosure gMin gMax x (k * gLo) (k * gHi)
    e'.eMin = e.eMin / k ∧ e'.eMax = e.eMax / k ∧ e'.feasible = e.feasible ∧ e'.admissible = e.admissible ∧
    e'.usableInverted = e.usableInverted ∧ e'.fillRatio = e.fillRatio := by
  simp only [bracketEnclosure]
  have m1 : gMin / (k * gLo) * (referenceExposure / x) = gMin / gLo * (referenceExposure / x) / k := by
    field_simp
  have m2 : gMax / (k * gHi) * (referenceExposure / x) = gMax / gHi * (referenceExposure / x) / k := by
    field_simp
  have r : k * gHi / (k * gLo) = gHi / gLo := mul_div_mul_left _ _ hk.ne'
  rw [m1, m2, r]
  refine ⟨rfl, rfl, ?_, rfl, ?_, rfl⟩
  · congr 1
    exact propext (div_le_div_iff_of_pos_right hk)
  · congr 1
    apply propext
    generalize gMax / gHi * (referenceExposure / x) = A
    generalize gMin / gLo * (referenceExposure / x) = B
    rw [div_mul_eq_mul_div, div_mul_eq_mul_div]
    exact div_lt_div_iff_of_pos_right hk

end Negicc
