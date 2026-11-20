/-
The whole conversion of one pixel.
-/
module

public import Negicc.Scene.Linear
public import Negicc.Scene.Shoulder
public import Negicc.Tone.Apply

@[expose] public section

/-!
# The conversion of one pixel

The stages of `color_pipeline.h`, in order, for one 16-bit camera RGB pixel and one input matrix
(`candidateMatrix` of the frame's merged matrix at the chosen `E`, `g`, `b`):

| Value | Definition | C++ |
|---|---|---|
| corrected RGB in `[0, 1]` | `correctedRgb` | `convert_pixel_corrected` |
| PCS XYZ (D50) | `convertPixelPcs` | `convert_pixel_pcs` |
| pre-shoulder linear sRGB | `convertPixelLinear` | `convert_pixel_linear` |
| display linear sRGB in `[0, 1]` | `convertPixelDisplay` | `convert_pixel_display` |
| encoded sRGB in `[0, 1]` | `convertPixel` | `convert_pixel` |

The tone curve, when the tone stage runs, sits between the display value and the encoding. The
search scores the CIELAB of the display value (`pixelLab`); it never applies the tone curve.

The output encoding (`OutputEncoding`) decides three things. The sRGB picture (`-P srgb`,
`Encoding.srgb`) applies the shoulder, the tone curve and the sRGB curve. The linear output
(`-P srgb-g10`, `Encoding.linear`) applies none of them: it is `convertPixelLinear` held to
`[0, 1]` (`convertPixel_linear`). Either is quantised to 16 bits (`quantizeU16`). The two other
outputs of a conversion are the corrected RGB alone (no `-P`: `correctedRgb`, written as 16-bit
codes with the bracket's profile embedded) and the PCS through an ICC output profile (`-P
<profile>`: `convertPixelPcs`, then Little-CMS, which is not part of this specification).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The output encoding (`OutputEncoding`): the sRGB curve (`ENCODE_SRGB`) or linear light with
sRGB primaries (`ENCODE_SRGB_LINEAR`, `-P srgb-g10`). -/
inductive Encoding where
  | srgb
  | linear
  deriving DecidableEq, Repr

/-- What a conversion needs besides the input matrix (`PipelineConfig`): the bracket's film
profile, the knee, the frame's shoulder, the tone curve's table when the tone stage runs, and the
output encoding (sRGB unless given). A tone table needs at least two entries (`n ≥ 1` in
`ToneTable`); a shorter one is no tone stage, as in `convert_pixel`. -/
structure PipelineConfig (α : Type) where
  profile : Profile α
  knee : α
  shoulder : Shoulder α
  tone : Option (Array α)
  encoding : Encoding := .srgb

/-- The shoulder the conversion applies: the frame's on the sRGB output, none on the linear one
(`apply_shoulder` tests the encoding). -/
def PipelineConfig.activeShoulder (c : PipelineConfig α) : Shoulder α :=
  if c.encoding = .srgb then c.shoulder else Shoulder.off

/-- The tone table the conversion applies: on the sRGB output, when there is one with at least two
entries (`cfg.encoding == ENCODE_SRGB && cfg.tone.lut && cfg.tone.n > 0`). -/
def PipelineConfig.activeTone (c : PipelineConfig α) : Option (Array α) :=
  match c.tone with
  | some lut => if c.encoding = .srgb ∧ 2 ≤ lut.size then some lut else none
  | none => none

/-- The display value of a loaded pixel: linear sRGB (D65) in `[0, 1]`, after the shoulder
(`convert_pixel_display`). -/
def convertPixelDisplay (c : PipelineConfig α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  toDisplay c.activeShoulder (convertPixelLinear c.profile c.knee m v)

/-- The sRGB encoding of a pixel, held to `[0, 1]` (`encode_output` with `ENCODE_SRGB`). -/
def encodeOutput (v : Vec3 α) : Vec3 α := (v.map srgbEncode).map clamp01

/-- A pixel in an output encoding, held to `[0, 1]` (`encode_output`). -/
def encodeOutputWith (e : Encoding) (v : Vec3 α) : Vec3 α :=
  match e with
  | .srgb => encodeOutput v
  | .linear => v.map clamp01

/-- The encoded output of a loaded pixel, in `[0, 1]` (`convert_pixel`). -/
def convertPixel (c : PipelineConfig α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  let d := convertPixelDisplay c m v
  encodeOutputWith c.encoding (match c.activeTone with
    | some lut => applyTone lut d
    | none => d)

/-- The CIELAB (D65) of a loaded pixel's display value: what every statistic and the search's
losses read (`linear_srgb_to_lab(to_display(...))`). -/
def pixelLab (c : PipelineConfig α) (m : Mat3 α) (v : Vec3 α) : Lab α :=
  linearSrgbToLab (convertPixelDisplay c m v)

/-- An encoded value in `[0, 1]` as an 8-bit code (`quantize_u8`). Over the real numbers the
C++'s `roundf` and `roundNat` agree on `[0, 255]` (see `roundNat`). -/
def quantizeU8 (v : α) : Nat := roundNat (v * 255.0)

/-- An encoded value in `[0, 1]` as a 16-bit code (`quantize_u16`): the 16-bit output's sample. -/
def quantizeU16 (v : α) : Nat := roundNat (v * 65535.0)

/-- The 16-bit output pixel of a 16-bit camera pixel (the picture `-P srgb` or `-P srgb-g10`
writes). -/
def outputPixel (c : PipelineConfig α) (m : Mat3 α) (px : Vec3 Nat) : Vec3 Nat :=
  (convertPixel c m (loadU16 px)).map quantizeU16

/-- The 16-bit corrected RGB of a 16-bit camera pixel: the output without `-P`, which embeds the
bracket's profile (`convert_pixel_corrected`, quantised). -/
def correctedPixel (c : PipelineConfig α) (m : Mat3 α) (px : Vec3 Nat) : Vec3 Nat :=
  (correctedRgb c.knee m (loadU16 px)).map quantizeU16

/-- The 8-bit sRGB preview pixel of a 16-bit camera pixel: what the intent network is shown. -/
def previewPixel (c : PipelineConfig α) (m : Mat3 α) (px : Vec3 Nat) : Vec3 Nat :=
  (convertPixel c m (loadU16 px)).map quantizeU8

end Negicc
