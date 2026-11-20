/-
The brackets: which are searched, their exposure ranges, and how they are ranked.
-/
module

public import Negicc.Scene.Defs
public import Negicc.Auto.Cells

@[expose] public section

/-!
# The brackets and their exposure ranges

**A bracket's calibrated range** is `bracketEnclosure` of `Negicc.Scene.Bracket` (DESIGN.md 3.3);
`BracketRange` adds the bracket's index. The search window is the usable range, ordered. Over the
real numbers a bracket is inadmissible exactly when its usable range is inverted
(`usableInverted_iff_not_admissible` in `Negicc.Auto.Props`).

**Which brackets are searched** (`filter_admissible_targets`): in a set of five or more, those
whose grey scale spans at least 0.75 of the widest one's, at least three.

**The rankings**: the best-fit bracket (lowest `|fill - 0.85|` plus 0.5 per stop of overflow,
admissible ones first), the depth of a metered exposure inside its window, the trust region of the
tournament (±0.5 stop around the screened `E`, inside the window), and the score that ranks the
highlight pullback's runs.

**The frame's scale** (`frame_scale`): the merged matrix of the scan, the reference merged matrix
(1/8 s, ISO 100, the bundle's own film base) and `e_norm`, the ratio of their green-green entries.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- What the search reads of a bracket (`BundleTarget`, as `profile_bundle.cc` reads it from the
bundle): the smallest and largest green value of its patches, its capture exposure (shutter times
ISO over 100) and its tonal capacity, the span of its grey scale `gs0` … `gs23` in stops.
`hasPatches` is false for a bracket without patch readings. -/
structure Bracket (α : Type) where
  hasPatches : Bool
  tgtMinG : α
  tgtMaxG : α
  exposure : α
  tonalCapacity : α

/-- A bracket's range on a frame with the bracket's index: `Enclosure` of `Negicc.Scene.Bracket`
(`bracket_enclosure`) for one bracket of the bundle. -/
structure BracketRange (α : Type) where
  idx : Nat
  range : Enclosure α

namespace BracketRange
variable (c : BracketRange α)
/-- `e_min ≤ e_max`. -/
def feasible : Bool := c.range.feasible
/-- The scene's span fits the bracket's plus the two tolerances. -/
def admissible : Bool := c.range.admissible
/-- The calibrated range's lower end. -/
def eMin : α := c.range.eMin
/-- The calibrated range's upper end. -/
def eMax : α := c.range.eMax
/-- The search window's lower end. -/
def searchLo : α := c.range.searchLo
/-- The search window's upper end. -/
def searchHi : α := c.range.searchHi
/-- `sqrt(e_min e_max)`. -/
def eCenter : α := c.range.eCentre
/-- The usable range is inverted. -/
def usableInverted : Bool := c.range.usableInverted
/-- `log2(e_max / e_min)`. -/
def slackEv : α := c.range.slackEv
/-- The scene's contrast over the bracket's. -/
def fillRatio : α := c.range.fillRatio
/-- The centre of the exposure penalty (`pull_centre`). -/
def centre : α := pullCentre c.range
end BracketRange

/-- Bracket `idx`'s range on a scene with green range `[g_lo, g_hi]` (`bracket_enclosure`); none
for a bracket without patch readings. -/
def bracketRange (b : Bracket α) (idx : Nat) (gLo gHi : α) : Option (BracketRange α) :=
  if !b.hasPatches then none else some ⟨idx, bracketEnclosure b.tgtMinG b.tgtMaxG b.exposure gLo gHi⟩

/-- The brackets that are searched, by index (`filter_admissible_targets`): all of a set of fewer
than five; otherwise those whose tonal capacity is at least 0.75 of the largest, and when fewer
than three are left, the three of largest capacity (ties in the bundle's order), in index order. -/
def filterAdmissible (caps : List α) : List Nat :=
  let n := caps.length
  let idx := List.range n
  if n < 5 then idx
  else
    let maxC := caps.foldl max 0.0
    let cap (i : Nat) : α := caps.getD i 0.0
    let keep := idx.filter fun i => decide (0.75 * maxC ≤ cap i)
    if 3 ≤ keep.length then keep
    else ((idx.mergeSort fun a b => decide (cap b ≤ cap a)).take 3).mergeSort (· ≤ ·)

/-- The best-fit score of a bracket: `|fill - 0.85|` plus 0.5 per stop by which the scene
overflows an infeasible bracket (`rank` in `search_conversion_params`; `SLACK_PENALTY_SCALE`). -/
def fitRank (c : BracketRange α) : α := fabs (c.fillRatio - 0.85) + 0.5 * max 0.0 (-c.slackEv)

/-- `a` comes before `b` in the best-fit order: an admissible bracket before an inadmissible one,
then the lower `fitRank`. -/
def fitBefore (a b : BracketRange α) : Bool :=
  if a.admissible != b.admissible then a.admissible else decide (fitRank a < fitRank b)

/-- The first smallest element of a non-empty list under a strict order `lt`, as
`std::min_element` finds it: a later element replaces the current one only when it is strictly
before it. -/
def minFirst {β : Type} (lt : β → β → Bool) : List β → Option β
  | [] => none
  | x :: xs => some (xs.foldl (fun best y => if lt y best then y else best) x)

/-- The bracket whose calibrated range the scene fits best (`best_fit`). -/
def bestFit (cs : List (BracketRange α)) : Option (BracketRange α) := minFirst fitBefore cs

/-- A metered bracket: its exposure for `L*` 50, the window it is searched in and the depth of the
metered exposure inside that window (`Depth`). -/
structure Metered (α : Type) where
  enc : BracketRange α
  eMet : α
  depth : α
  lo : α
  hi : α

/-- The depth of a metered exposure: its distance in stops from the nearer end of the window
(ends held at least `1e-6`), less 0.5 per stop of overflow of an infeasible bracket
(`metered`). -/
def metered (c : BracketRange α) (eMet lo hi : α) : Metered α :=
  let d := min (log2 (eMet / max 0.000001 lo)) (log2 (max 0.000001 hi / eMet)) - 0.5 * max 0.0 (-c.slackEv)
  ⟨c, eMet, d, lo, hi⟩

/-- Metered brackets, deepest first; equal depths keep their order (`deepest_first`, a stable
sort). -/
def deepestFirst (ms : List (Metered α)) : List (Metered α) :=
  ms.mergeSort fun a b => decide (b.depth ≤ a.depth)

/-- The tournament's trust region around a screened exposure `e` in the window `[lo, hi]`: at
most half a stop either way, inside the window, with the bracket's pull centre (`trust_bounds`).
-/
def trustBounds (lo hi e centre : α) : α × α × α :=
  (max lo (e * pow 2.0 (-0.5)), min hi (e * pow 2.0 0.5), centre)

/-- The pullback's penalties on an exposure `e` of a bracket with calibrated range
`[e_min, e_max]`: `4.5 d_hi² + 1.2 d_sh² + 0.8 slack²`, `d_hi` the stops of `e` below `e_min`,
`d_sh` above `e_max`, `slack` the stops by which `e_min` exceeds `e_max`. -/
def enclosurePenalty (eMin eMax e : α) : α :=
  let dHi := max 0.0 (log2 (eMin / e))
  let dSh := max 0.0 (log2 (e / eMax))
  let neg := max 0.0 (log2 (eMin / eMax))
  4.5 * sq dHi + 1.2 * sq dSh + 0.8 * sq neg

/-- The score that ranks the pullback's tournament: the residual plus the enclosure penalties at the
run's exposure (`tourney_score`); the residual alone without an enclosure or for `e ≤ 0`. The
overflow is read as `-slack_ev`, which over the real numbers is `log2(e_min / e_max)`. -/
def tourneyScore (enc : Option (BracketRange α)) (resid e : α) : α :=
  match enc with
  | some c =>
    if 0.0 < e then
      let dHi := max 0.0 (log2 (c.eMin / e))
      let dSh := max 0.0 (log2 (e / c.eMax))
      let neg := max 0.0 (-c.slackEv)
      resid + 4.5 * sq dHi + 1.2 * sq dSh + 0.8 * sq neg
    else resid
  | none => resid

/-! ## The frame's scale -/

/-- What `frame_scale` needs of a bundle: its crosstalk matrix, normalisation target, film base
and the film base capture's exposure (shutter time, from the bundle's shutter string, times ISO
over 100). -/
structure BundleScale (α : Type) where
  crosstalk : Mat3 α
  normTarget : α
  filmBase : Vec3 α
  filmBaseExposure : α

/-- The frame's scale (`FrameScale`): the film base used, the exposure ratio of the base capture
over the scan, the merged matrix `M_frame` and `e_norm`. -/
structure FrameScale (α : Type) where
  fb : Vec3 α
  ratioFrame : α
  frameMatrix : Mat3 α
  eNorm : α

/-- The frame's scale (`frame_scale`): the film base is the one measured on this capture when given,
else the bundle's; a base read from the scan itself has no exposure ratio. `e_norm` is the
green-green entry of `M_frame` over that of the reference matrix (1 when that entry is 0). -/
def frameScale (b : BundleScale α) (shutter iso : α) (customFb : Option (Vec3 α)) (fromScan : Bool) :
    FrameScale α :=
  let fb := customFb.getD b.filmBase
  let ratio := if customFb.isSome && fromScan then 1.0 else b.filmBaseExposure / captureExposure shutter iso
  let ratioRef := b.filmBaseExposure / referenceExposure
  let frameM := mergedMatrix b.crosstalk fb b.normTarget ratio
  let refM := mergedMatrix b.crosstalk b.filmBase b.normTarget ratioRef
  let eNorm := if scalarEq refM.row1.y 0.0 then 1.0 else exposureNorm frameM refM
  ⟨fb, ratio, frameM, eNorm⟩

end Negicc
