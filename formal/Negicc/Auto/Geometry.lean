/-
The frame's inner crop and the network's input size.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The inner crop and the network's grid

The search reads a frame's centred inner crop (90 % of each side by default), so that the
frame's border does not count (`compute_inner_crop` in `color_pipeline.h`).

The network takes an image whose longer side is scaled to `max_size` (512) and whose two sides
are rounded to whole patches of `patch_size` (16) pixels; each patch is one cell of the intent
grid (`compute_aspect_preserved_shape` in `color_pipeline.h`). A 3:2 crop gives 32 × 21 cells.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- A rectangle of a frame: its top left corner and its size, in pixels (`InnerCrop`). -/
structure InnerCrop where
  x0 : Nat
  y0 : Nat
  w : Nat
  h : Nat
deriving Repr, BEq

/-- The margin of one side of the inner crop: `round(n (1 - ratio) / 2)`, held to at most
`(n - 1) / 2` so that at least one pixel remains. -/
def cropMargin (n : Nat) (ratio : α) : Nat :=
  min (roundNat ((n : α) * (1.0 - ratio) * 0.5)) ((n - 1) / 2)

/-- The centred `ratio` share of a `W × H` frame (`compute_inner_crop`): the whole frame for a
ratio of 1 or more, nothing for an empty frame. The margins are rounded halves upward, as C
`roundf` does for the non-negative values a ratio in `[0, 1]` gives. -/
def innerCrop (W H : Nat) (ratio : α) : InnerCrop :=
  if W = 0 ∨ H = 0 then ⟨0, 0, 0, 0⟩
  else if 1.0 ≤ ratio then ⟨0, 0, W, H⟩
  else
    let x0 := cropMargin W ratio
    let y0 := cropMargin H ratio
    ⟨x0, y0, W - 2 * x0, H - 2 * y0⟩

/-- The network's input size for a `w × h` image: `(target_w, target_h, grid_w, grid_h)`
(`AspectPreservedShape`). -/
structure NetShape where
  targetW : Nat
  targetH : Nat
  gridW : Nat
  gridH : Nat
deriving Repr, BEq

/-- The number of cells of the grid. -/
def NetShape.cells (s : NetShape) : Nat := s.gridW * s.gridH

/-- One side's number of patches: `nearbyint(side · max_size / longer side / patch)`, at least
one. -/
def gridSide (side longer m p : Nat) : Nat :=
  max 1 (roundEvenNat ((side : α) * ((m : α) / (longer : α)) / (p : α)))

/-- The network's input size of a `w × h` image (`compute_aspect_preserved_shape`): the longer
side scaled to `max_size`, each side rounded to the nearest whole number of patches (a half to
the even neighbour, C `nearbyint`), at least one. A patch size below 1 is taken as 1, a
`max_size` below the patch size as the patch size, and an empty side as 1 pixel. -/
def aspectShape (w h maxSize patch : Nat) : NetShape :=
  let p := max patch 1
  let m := if maxSize > p then maxSize else p
  let mw := max w 1
  let mh := max h 1
  let longer := max mw mh
  let gw := gridSide (α := α) mw longer m p
  let gh := gridSide (α := α) mh longer m p
  ⟨gw * p, gh * p, gw, gh⟩

end Negicc
