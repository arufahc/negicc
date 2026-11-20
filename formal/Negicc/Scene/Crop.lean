/-
The inner crop the statistics read.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The inner crop

The statistics and previews of a frame leave out its border: they read the centred `crop_ratio`
share of the frame (`compute_inner_crop` in `color_pipeline.h`), with `crop_ratio = 0.90`
(`ParamSearchOptions::crop_ratio`).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The share of the frame the statistics and previews read, its centred inner crop
(`ParamSearchOptions::crop_ratio`, `0.90`). -/
def searchCropRatio : α := 0.90

/-- The centred `ratio` share of a `W x H` frame, at least one pixel: its left, top, width and height
(`compute_inner_crop`). -/
def innerCrop (W H : Nat) (ratio : α) : Nat × Nat × Nat × Nat :=
  if W = 0 || H = 0 then (0, 0, 0, 0)
  else if 1.0 ≤ ratio then (0, 0, W, H)
  else
    let x0 := min (roundNat ((W : α) * (1.0 - ratio) * 0.5)) ((W - 1) / 2)
    let y0 := min (roundNat ((H : α) * (1.0 - ratio) * 0.5)) ((H - 1) / 2)
    (x0, y0, W - 2 * x0, H - 2 * y0)

end Negicc
