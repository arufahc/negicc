/-
Resampling an image to the network's grid.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# Resampling

Three resamplings bring a frame or a preview to the network's size:

* **Area average** (`resample_grid_rgb16`, `resample_view_rgb16` in `rgb_view.cc`, the first branch
  of `compute_frame_stats` and both passes of `trained_cells_lab`): output pixel `d` of a side
  covers the source interval `[d s, (d + 1) s]`, `s` the ratio of the source size to the output
  size, and is the mean of the source pixels weighted by how much of each the interval covers.
* **Bilinear point sample** (the enlarging branch of the same functions): the source read at
  `(d + 0.5) s - 0.5` between its two neighbours, the index held to the image.
* **Pillow's antialiased bilinear filter** (`antialiased_bilinear_resize_rgb` in
  `dinov3_preprocess.cc`), which the network's input goes through: a triangle filter widened by
  the reduction factor, its weights in fixed point with 22 fraction bits, an 8-bit image between
  the horizontal and the vertical pass.

Images are row-major arrays of pixels; a 16-bit or 8-bit image holds natural numbers.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The smallest natural number not below `x ≥ 0` (C `ceil` for a non-negative argument). -/
def natCeil (x : α) : Nat :=
  let f := natFloor x
  if (f : α) < x then f + 1 else f

/-- Equality of two scalars, decided by the two comparisons (`==` in the C++). -/
def scalarEq (a b : α) : Bool := decide (a ≤ b) && decide (b ≤ a)

/-- The pixel at column `x`, row `y` of a row-major image `w` pixels wide (black outside). -/
def pixelAt (img : Array (Vec3 α)) (w x y : Nat) : Vec3 α :=
  img.getD (y * w + x) (Vec3.fill 0.0)

/-- The pixel at column `x`, row `y` of a row-major image of codes `w` pixels wide (black
outside). -/
def pixelAtNat (img : Array (Vec3 Nat)) (w x y : Nat) : Vec3 Nat :=
  img.getD (y * w + x) ⟨0, 0, 0⟩

/-! ## Area average -/

/-- The source pixels output pixel `d` covers when `srcN` pixels go to `dstN`, with the length of
each one's overlap with `[d s, (d + 1) s]`; pixels with no overlap are left out. -/
def areaTaps (srcN dstN d : Nat) : List (Nat × α) :=
  let s : α := (srcN : α) / (dstN : α)
  let x0 : α := (d : α) * s
  let x1 : α := ((d + 1 : Nat) : α) * s
  let xs := natFloor x0
  let xe := min srcN (natCeil x1)
  (List.range (xe - xs)).filterMap fun k =>
    let xx := xs + k
    let wt := min x1 ((xx : α) + 1.0) - max x0 (xx : α)
    if wt ≤ 0.0 then none else some (xx, wt)

/-- The weighted mean of a list of (value, weight) pairs: the sum of value times weight, times the
reciprocal of the weights' sum; zero when the weights sum to zero or less. -/
def weightedMean (xs : List (Vec3 α × α)) : Vec3 α :=
  let acc := xs.foldl (fun a p => a.add (p.1.map (· * p.2))) (Vec3.fill 0.0)
  let wsum := xs.foldl (fun a p => a + p.2) 0.0
  let inv : α := if 0.0 < wsum then 1.0 / wsum else 0.0
  acc.map (· * inv)

/-- Output pixel `(dx, dy)` of the area average of a `sw × sh` image to `dw × dh`: the source
pixels under it weighted by the area each one shares with it. -/
def areaPixel (img : Array (Vec3 α)) (sw sh dw dh dx dy : Nat) : Vec3 α :=
  let ty := areaTaps (α := α) sh dh dy
  let tx := areaTaps (α := α) sw dw dx
  weightedMean (ty.flatMap fun (yy, wy) => tx.map fun (xx, wx) =>
    (pixelAt img sw xx yy, wx * wy))

/-- Output pixel `(dx, dy)` of the bilinear point sample of a `sw × sh` image at `dw × dh`: the
source read at `(d + 0.5) s - 0.5` on each axis from the pixel at or below it (held to the image)
and the next one. Before the first pixel the weights extrapolate from the first two. -/
def bilinearPixel (img : Array (Vec3 α)) (sw sh dw dh dx dy : Nat) : Vec3 α :=
  let sx : α := (sw : α) / (dw : α)
  let sy : α := (sh : α) / (dh : α)
  let fy : α := ((dy : α) + 0.5) * sy - 0.5
  let y0 := min (natFloor fy) (sh - 1)
  let y1 := min (y0 + 1) (sh - 1)
  let wy1 := fy - (y0 : α)
  let wy0 := 1.0 - wy1
  let fx : α := ((dx : α) + 0.5) * sx - 0.5
  let x0 := min (natFloor fx) (sw - 1)
  let x1 := min (x0 + 1) (sw - 1)
  let wx1 := fx - (x0 : α)
  let wx0 := 1.0 - wx1
  let p00 : Vec3 α := pixelAt img sw x0 y0
  let p01 : Vec3 α := pixelAt img sw x1 y0
  let p10 : Vec3 α := pixelAt img sw x0 y1
  let p11 : Vec3 α := pixelAt img sw x1 y1
  let top := Vec3.zipWith (fun a b => a * wx0 + b * wx1) p00 p01
  let bot := Vec3.zipWith (fun a b => a * wx0 + b * wx1) p10 p11
  Vec3.zipWith (fun t b => t * wy0 + b * wy1) top bot

/-- A `sw × sh` image resampled to `dw × dh`: the area average when neither side is enlarged, the
bilinear point sample otherwise (the branch rule of `compute_frame_stats` and
`resample_view_rgb16`). -/
def resampleImage (img : Array (Vec3 α)) (sw sh dw dh : Nat) : Array (Vec3 α) :=
  let area := decide (1.0 ≤ (sw : α) / (dw : α)) && decide (1.0 ≤ (sh : α) / (dh : α))
  ((List.range dh).flatMap fun dy => (List.range dw).map fun dx =>
    if area then areaPixel img sw sh dw dh dx dy else bilinearPixel img sw sh dw dh dx dy).toArray

/-- A value rounded to a 16-bit code: halves upward, held to `[0, 65535]`. -/
def toU16 (v : α) : Nat := min (roundNat v) 65535

/-- A 16-bit image resampled to `dw × dh` and rounded back to 16-bit codes
(`resample_view_rgb16`; an image of the same size is copied). -/
def resampleU16 (img : Array (Vec3 Nat)) (sw sh dw dh : Nat) : Array (Vec3 Nat) :=
  if sw = dw ∧ sh = dh then img
  else (resampleImage (img.map (·.map (fun n : Nat => (n : α)))) sw sh dw dh).map (·.map toU16)

/-! ## Pillow's antialiased bilinear filter -/

/-- The fixed-point precision of the filter's weights: 22 fraction bits. -/
def filterBits : Nat := 22

/-- The taps of output index `xx` of one pass from `inN` to `outN` samples: the first source index
and the weights in fixed point (`compute_antialiased_bilinear_coeffs`). The triangle filter has
support 1, widened by the reduction factor when shrinking; its weights are normalised to sum to 1
before they are rounded. -/
def aaTaps (inN outN xx : Nat) : Nat × List Nat :=
  let scale : α := (inN : α) / (outN : α)
  let filterscale : α := max scale 1.0
  let support : α := 1.0 * filterscale
  let center : α := ((xx : α) + 0.5) * scale
  let ss : α := 1.0 / filterscale
  let xmin := natFloor (center - support + 0.5)
  let xmax := min (natFloor (center + support + 0.5)) inN
  let count := xmax - xmin
  let ws : List α := (List.range count).map fun x =>
    let d := fabs ((((x + xmin : Nat) : α) - center + 0.5) * ss)
    if d < 1.0 then 1.0 - d else 0.0
  let ww := ws.foldl (· + ·) 0.0
  let wn := ws.map fun w => if scalarEq ww 0.0 then w else w / ww
  (xmin, wn.map fun w => roundNat (w * ((2 ^ filterBits : Nat) : α)))

/-- A fixed-point sum back to an 8-bit value: rounded (the sum starts at half a unit), held to
`[0, 255]` (`clip_u8`). -/
def clipU8 (s : Nat) : Nat := if 2 ^ (filterBits + 8) ≤ s then 255 else s / 2 ^ filterBits

/-- One sample of a pass: the taps of output index `o` applied to the samples `get i`. -/
def aaSample (inN outN o : Nat) (get : Nat → Nat) : Nat :=
  let (xmin, ks) := aaTaps (α := α) inN outN o
  clipU8 ((ks.zipIdx.foldl (fun s (k, i) => s + get (xmin + i) * k)) (2 ^ (filterBits - 1)))

/-- An 8-bit `sw × sh` image resized to `dw × dh` (`antialiased_bilinear_resize_rgb`, Pillow's
`Image.resize` with `BILINEAR`): a horizontal pass when the width changes, then a vertical pass
when the height changes, each rounded to 8 bits. An image of the same size is copied. -/
def aaResize (img : Array (Vec3 Nat)) (sw sh dw dh : Nat) : Array (Vec3 Nat) :=
  let mid : Array (Vec3 Nat) :=
    if sw = dw then img
    else ((List.range sh).flatMap fun y => (List.range dw).map fun x =>
      let ch (c : Vec3 Nat → Nat) := aaSample (α := α) sw dw x fun i => c (pixelAtNat img sw i y)
      (⟨ch (·.x), ch (·.y), ch (·.z)⟩ : Vec3 Nat)).toArray
  if sh = dh then mid
  else ((List.range dh).flatMap fun y => (List.range dw).map fun x =>
    let ch (c : Vec3 Nat → Nat) := aaSample (α := α) sh dh y fun i => c (pixelAtNat mid dw x i)
    (⟨ch (·.x), ch (·.y), ch (·.z)⟩ : Vec3 Nat)).toArray

end Negicc
