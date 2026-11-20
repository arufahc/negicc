/-
Reading the scan: black level, the half-size image, the debayers, pixel shift, three captures.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# Reading the scan

The arithmetic of how a RAW's mosaic becomes RGB (`np_input.cc`, `rgb_view.cc`):

* **Black level** (`subtract_cfa_black_in_place`): each photosite less its colour's black level
  (LibRaw's `black + cblack[colour]`, plus the pattern entry when the RAW has a black pattern), held
  at 0.
* **The half-size image** under the RGB LED (`cpu_preview_debayer`, the view `from_cfa_half`): one
  pixel per 2x2 Bayer quad, `(R, (G1 + G2) / 2, B)` with the green sum halved by a shift.
* **The grain-aware debayer** of a single Sony RGGB RAW (`cpu_grain_aware_debayer`): green from the
  vertical and horizontal Hamilton-Adams estimates blended with weights `1 / (4096 + gradient^2)`
  (`reconstruct_green_soft_cpu`), red and blue as ratios `(C + 16) / (G + 16)` of their neighbours
  carried to the pixel's green, and plain neighbour means where a channel is within 64 of white.
* **Pixel shift** (`cpu_pixel_shift_merge`): four captures shifted by one photosite; each pixel
  takes its R and B from the capture that saw them and the mean of its two greens, held at white.
* **Three captures** (`load_rgb_input`): the green and blue captures scaled to the first's exposure
  (shutter times ISO), rounded and held at 65535; a TIFF's half-size image is the rounded mean of
  each 2x2 block.

A photosite outside the mosaic reads its mirror image about the edge photosite (`get_px_refl_cpu`).
The geometry (crop, rotation, flips) of the output relative to the mosaic is index bookkeeping and
is taken as the identity here: output pixel `(x, y)` is mosaic position `(rx + x, ry + y)`.
LibRaw's own demosaic (`dcraw_process`), used for other RAWs, for `-q 0` and more and for the
white-light decode, is not part of this specification.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- A Bayer mosaic of 16-bit samples, `w` photosites a row, row-major. -/
structure Mosaic where
  w : Nat
  h : Nat
  data : Array Nat

/-- The colour of photosite `(r, c)` of an RGGB mosaic: 0 red, 1 green, 2 blue
(`bayer_color_at` for the pattern the RGB-LED debayers accept; a second green reads as 1). -/
def rggbColour (r c : Int) : Nat :=
  if r % 2 = 0 then (if c % 2 = 0 then 0 else 1) else (if c % 2 = 0 then 1 else 2)

/-- An index mirrored about the edge into `[0, n - 1]` (`get_px_refl_cpu`). -/
def reflect (i : Int) (n : Nat) : Nat :=
  let r : Int := if i < 0 then -i else if (n : Int) ≤ i then 2 * ((n : Int) - 1) - i else i
  (max 0 (min ((n : Int) - 1) r)).toNat

/-- The sample at `(r, c)`, mirrored outside the mosaic (`get_px_refl_cpu`). -/
def Mosaic.at (m : Mosaic) (r c : Int) : Nat := m.data.getD (reflect r m.h * m.w + reflect c m.w) 0

/-- The sample at `(r, c)` as a number. -/
def Mosaic.atF (m : Mosaic) (r c : Int) : α := (m.at r c : α)

/-! ## Black level -/

/-- A photosite less its black level, held at 0 (`subtract_cfa_black_in_place`). -/
def subtractBlack (v black : Nat) : Nat := v - black

/-- The white level left after the black subtraction: the RAW's maximum less the largest black level
of a colour (with its pattern's mean), at least 1. -/
def whiteAfterBlack (maximum maxBlack : Nat) : Nat := max 1 (maximum - maxBlack)

/-! ## The half-size image -/

/-- The half-size pixel of the 2x2 quad at `(r, c)` (its top left photosite) of an RGGB mosaic:
`(R, (G1 + G2) / 2, B)`, the green sum halved by a shift (`cpu_preview_debayer`). -/
def halfSizePixel (m : Mosaic) (r c : Int) : Vec3 Nat :=
  let s := [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
  let pick := fun (col : Nat) => (s.filter fun p => rggbColour p.1 p.2 = col).map fun p => m.at p.1 p.2
  ⟨(pick 0).headD 0, ((pick 1).foldl (· + ·) 0) / 2, (pick 2).headD 0⟩

/-! ## The grain-aware debayer -/

/-- Green at photosite `(r, c)` of an RGGB mosaic (`reconstruct_green_soft_cpu`): the sample on a
green site; elsewhere the vertical and horizontal Hamilton-Adams estimates (neighbour mean plus a
quarter of the centre channel's Laplacian) blended with weights `1 / (eps + d^2)`, `d` the green
gradient plus that Laplacian, held to `[0, white]`. -/
def greenSoft (m : Mosaic) (white eps : α) (r c : Int) : α :=
  if (r + c) % 2 = 1 then m.atF r c
  else
    let cc : α := m.atF r c
    let gu : α := m.atF (r - 1) c
    let gd : α := m.atF (r + 1) c
    let gl : α := m.atF r (c - 1)
    let gr : α := m.atF r (c + 1)
    let cu2 : α := m.atF (r - 2) c
    let cd2 : α := m.atF (r + 2) c
    let cl2 : α := m.atF r (c - 2)
    let cr2 : α := m.atF r (c + 2)
    let dv := fabs (gu - gd) + fabs (2.0 * cc - cu2 - cd2)
    let dh := fabs (gl - gr) + fabs (2.0 * cc - cl2 - cr2)
    let gv := 0.5 * (gu + gd) + 0.25 * (2.0 * cc - cu2 - cd2)
    let gh := 0.5 * (gl + gr) + 0.25 * (2.0 * cc - cl2 - cr2)
    let wv := 1.0 / (eps + dv * dv)
    let wh := 1.0 / (eps + dh * dh)
    max 0.0 (min white ((wv * gv + wh * gh) / (wv + wh)))

/-- The ratio debayer's offset `k = 16` and the default blend `eps = 4096`. -/
def debayerK : α := 16.0
@[inherit_doc debayerK] def debayerEps : α := 4096.0

/-- A chroma channel at `(r, c)` from neighbours `ns`: `(G + k) mean((C_n + k) / (G_n + k)) - k`. -/
def ratioFromNeighbours (m : Mosaic) (white eps G : α) (ns : List (Int × Int)) : α :=
  let k : α := debayerK
  let rs := ns.map fun p => (m.atF p.1 p.2 + k) / (greenSoft m white eps p.1 p.2 + k)
  (G + k) * ((rs.foldl (· + ·) 0.0) * (1.0 / (ns.length : α))) - k

/-- The mean of the samples at the positions `ns`, with the C++'s factor `0.5` or `0.25`. -/
def meanAt (m : Mosaic) (ns : List (Int × Int)) : α :=
  (if ns.length = 2 then 0.5 else 0.25) * (ns.map fun p => (m.atF p.1 p.2 : α)).foldl (· + ·) 0.0

/-- One pixel of the grain-aware debayer at photosite `(r, c)` of an RGGB mosaic
(`cpu_grain_aware_debayer`), before the rounding and the flat field. Where the centre sample or
green is within 64 of `white`, red and blue are plain neighbour means; elsewhere the ratio
debayer. -/
def grainAwarePixel (m : Mosaic) (white eps : α) (r c : Int) : Vec3 α :=
  let G := greenSoft m white eps r c
  let rEven := r % 2 = 0
  let cEven := c % 2 = 0
  let cc : α := m.atF r c
  let diag := [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
  let horiz := [(r, c - 1), (r, c + 1)]
  let vert := [(r - 1, c), (r + 1, c)]
  if white - 64.0 ≤ cc || white - 64.0 ≤ G then
    if rEven && cEven then ⟨cc, G, meanAt m diag⟩
    else if !rEven && !cEven then ⟨meanAt m diag, G, cc⟩
    else if rEven && !cEven then ⟨meanAt m horiz, G, meanAt m vert⟩
    else ⟨meanAt m vert, G, meanAt m horiz⟩
  else
    let R := if rEven && cEven then cc
      else if rEven && !cEven then ratioFromNeighbours m white eps G horiz
      else if !rEven && cEven then ratioFromNeighbours m white eps G vert
      else ratioFromNeighbours m white eps G diag
    let B := if !rEven && !cEven then cc
      else if !rEven && cEven then ratioFromNeighbours m white eps G horiz
      else if rEven && !cEven then ratioFromNeighbours m white eps G vert
      else ratioFromNeighbours m white eps G diag
    ⟨R, G, B⟩

/-- A debayered value as a 16-bit code: held to `[0, 65535]`, rounded halves away from zero. -/
def toU16 (v : α) : Nat := roundNat (max 0.0 (min 65535.0 v))

/-- The grain-aware debayer's 16-bit pixel (before the flat field). -/
def grainAwarePixelU16 (m : Mosaic) (white eps : α) (r c : Int) : Vec3 Nat :=
  (grainAwarePixel m white eps r c).map toU16

/-! ## Pixel shift -/

/-- One pixel of the pixel-shift merge at `(r, c)` (`cpu_pixel_shift_merge`): the four captures
`f0 .. f3` read at `(r, c)`, `(r - 1, c)`, `(r - 1, c + 1)`, `(r, c + 1)`; red and blue from the
capture whose site has that colour, green the mean of the two green sites (integer division); each
held at `white`, green also when either green site reached it. -/
def pixelShiftPixel (f0 f1 f2 f3 : Mosaic) (white : Nat) (r c : Int) : Vec3 Nat :=
  let s := [(f0.at r c, r, c), (f1.at (r - 1) c, r - 1, c), (f2.at (r - 1) (c + 1), r - 1, c + 1),
            (f3.at r (c + 1), r, c + 1)]
  let cols := s.map fun (v, rr, cc) => (v, rggbColour rr cc)
  let gs := (cols.filter (·.2 = 1)).map (·.1)
  let R := ((cols.filter (·.2 = 0)).map (·.1)).headD 0
  let B := ((cols.filter (·.2 = 2)).map (·.1)).headD 0
  let G := if gs.length > 0 then gs.foldl (· + ·) 0 / gs.length else 0
  let gClipped := gs.any (white ≤ ·)
  ⟨if white ≤ R then white else R, if gClipped || white ≤ G then white else G, if white ≤ B then white else B⟩

/-! ## Three captures -/

/-- A channel of a later capture scaled to the first capture's exposure: `min(65535, round(v e0 / ek))`
(`load_rgb_input`; unchanged when the scale is 1). -/
def captureScale (v : Nat) (e0 ek : α) : Nat :=
  let s := e0 / ek
  if decide (s ≤ 1.0) && decide (1.0 ≤ s) then v else min 65535 (roundNat ((v : α) * s))

/-- The half-size value of a 2x2 block of a TIFF capture: `(a + b + c + d + 2) / 4` in integers. -/
def tiffHalf (a b c d : Nat) : Nat := (a + b + c + d + 2) / 4

end Negicc
