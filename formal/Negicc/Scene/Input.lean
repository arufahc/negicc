/-
From the camera's RGB of the negative to the corrected RGB the film profile reads.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The input side of the conversion

One pixel of a narrow-band (TriLED) scan arrives as 16-bit camera RGB: the sensor's values after
the black level is subtracted and the mosaic is resolved to three channels (the demosaicing is
not part of this specification). It then goes through

1. the **flat field** (`flatDivide`): each channel divided by the gain of the scanner's light and
   lens at that place, `flat_divide` in `flat_field.h`;
2. the **load** (`loadU16`): the 16-bit code over 65535;
3. the **input matrix** (`candidateMatrix` of `mergedMatrix`): the crosstalk correction, which
   unmixes the three LEDs, with each row scaled so that the film base lands on one level
   (`compute_station_merged_matrix` in `profile_bundle.cc`), and then by the exposure `E` and the
   gains `g` and `b` (`candidate_matrix` in `color_pipeline.h`);
4. the **soft knee** (`kneeCompress`): the identity up to 0.95, then a roll-off toward 1, so that
   values past the profile's input range compress instead of clipping.

The result, `correctedRgb`, is in `[0, 1]` and is proportional, below the knee, to the film's
transmission in each LED band: `convert_pixel_corrected` in `color_pipeline.h`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-! ## Flat field -/

/-- Where pixel `k` of an axis falls on the flat field's gain grid: the index `i` of the grid cell
and the share `t` of the way to the next one. `block` is the grid's block size in half-size sensor
pixels and `gridN` the number of blocks along the axis; a gain sits at the centre of its block.
`half` says whether `k` counts half-size pixels (the analysis decode) or full-size ones (the
export). `axis_table` in `flat_field.cc`. -/
def flatAxis (half : Bool) (block gridN k : Nat) : Nat × α :=
  let h : α := if half then (k : α) else ((k : α) + 0.5) / 2.0 - 0.5
  let f := clamp ((h + 0.5) / (block : α) - 0.5) 0.0 ((gridN - 1 : Nat) : α)
  let i := min (natFloor f) (gridN - 2)
  (i, f - (i : α))

/-- The gain at a pixel, bilinear between four neighbouring grid gains: `g00` and `g01` in the
upper grid row at columns `i` and `i + 1`, `g10` and `g11` in the row below, with `tx` and `ty`
the shares from `flatAxis`. First along `y` in each column, then along `x`, as `flat_divide`
does. -/
def flatGain (g00 g01 g10 g11 tx ty : α) : α :=
  let lo := g00 + (g10 - g00) * ty
  let hi := g01 + (g11 - g01) * ty
  lo + (hi - lo) * tx

/-- A 16-bit channel value divided by its gain, rounded to the nearest whole number (a half to
the even one) and held at 65535 (`flat_divide`).

The C++ forms the gain and the quotient in single precision, which near 50000 has steps of 1/256,
and then rounds to a whole number. Where the exact quotient lies within about 0.004 of a half, the
two roundings together can give the neighbouring code. This is the one place where the program
differs from the real-number formula by more than rounding noise: by one 16-bit code, on about
one value in a hundred. -/
def flatDivide (gain : α) (px : Nat) : Nat :=
  min (roundEvenNat ((px : α) / gain)) 65535

/-! ## Load -/

/-- A 16-bit code as a number in `[0, 1]` (`load_u16`). -/
def ofU16 (n : Nat) : α := (n : α) / 65535.0

/-- A 16-bit RGB pixel as three numbers in `[0, 1]` (`load_u16`). -/
def loadU16 (px : Vec3 Nat) : Vec3 α := px.map ofU16

/-! ## The input matrix -/

/-- The exposure of a capture: shutter time in seconds times ISO over 100. -/
def captureExposure (shutter iso : α) : α := shutter * (iso / 100.0)

/-- The merged input matrix of a frame: row `r` of the crosstalk correction `X` scaled by
`n / f_r * ratio`. `f` is the film base (after the crosstalk correction), `n` the bundle's
normalisation target (55000), and `ratio` the exposure of the film base's capture over the
scan's, or `1` when the base was read from the scan itself. A row whose base is not positive is
left unscaled. With it, bare film lands on the level `n` in every channel.
`compute_station_merged_matrix` in `profile_bundle.cc`. -/
def mergedMatrix (X : Mat3 α) (f : Vec3 α) (n ratio : α) : Mat3 α :=
  Mat3.scaleRows (f.map fun fr => if 0.0 < fr then n / fr * ratio else 1.0) X

/-- The input matrix of one candidate `(E, g, b)`: the merged matrix with its rows scaled by `E`,
`E g` and `E b`. `E` is the exposure, a gain on all three channels; `g` and `b` are the gains of
green and blue relative to red. `candidate_matrix` in `color_pipeline.h`. -/
def candidateMatrix (merged : Mat3 α) (e g b : α) : Mat3 α :=
  Mat3.scaleRows ⟨e, e * g, e * b⟩ merged

/-- A matrix with every entry divided by `k`. The search works on the merged matrix divided by
the frame's exposure normalisation `e_norm` (`SearchFrame::set_frame_matrix`). -/
def Mat3.divScalar (m : Mat3 α) (k : α) : Mat3 α :=
  ⟨m.row0.map (· / k), m.row1.map (· / k), m.row2.map (· / k)⟩

/-- The exposure normalisation of a frame: the green-green entry of its merged matrix over that
of the reference merged matrix (the same bundle scanned at 1/8 s, ISO 100 with the bundle's own
film base). `frame_scale` in `multi_target_optimizer.cc`. -/
def exposureNorm (frame reference : Mat3 α) : α := frame.row1.y / reference.row1.y

/-! ## The soft knee -/

/-- The soft knee on one channel: `0` for a non-positive value, the identity up to `knee`, then
`knee + (1 - knee) tanh((v - knee) / (1 - knee))`, held at `0.9999`. With `knee ≥ 1` it only
clamps to `[0, 1]`. `knee_compress` in `color_pipeline.h`; the conversion uses `knee = 0.95`. -/
def kneeCompress (knee v : α) : α :=
  if v ≤ 0.0 then 0.0
  else if 1.0 ≤ knee then min 1.0 v
  else if v ≤ knee then min 1.0 v
  else min 0.9999 (knee + (1.0 - knee) * tanh ((v - knee) / (1.0 - knee)))

/-- The knee of the conversion and of every preview of the search (`kKnee`, `AUTO_KNEE`). -/
def autoKnee : α := 0.95

/-! ## The corrected RGB -/

/-- The corrected RGB of a loaded pixel `v`: the input matrix `m`, then the knee on each channel
(`convert_pixel_corrected`, after the load). -/
def correctedRgb (knee : α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  (m.mulVec v).map (kneeCompress knee)

end Negicc
