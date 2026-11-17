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

This is the sRGB picture (`-P srgb`). The linear output (`-P srgb-g10`) leaves out the shoulder,
the tone curve and the sRGB curve; it is `convertPixelLinear` held to `[0, 1]`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- What a conversion needs besides the input matrix (`PipelineConfig`): the bracket's film
profile, the knee, the frame's shoulder and, when the tone stage runs, the tone curve's table. -/
structure PipelineConfig (α : Type) where
  profile : Profile α
  knee : α
  shoulder : Shoulder α
  tone : Option (Array α)

/-- The display value of a loaded pixel: linear sRGB (D65) in `[0, 1]`, after the shoulder
(`convert_pixel_display`). -/
def convertPixelDisplay (c : PipelineConfig α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  toDisplay c.shoulder (convertPixelLinear c.profile c.knee m v)

/-- The sRGB encoding of a pixel, held to `[0, 1]` (`encode_output`). -/
def encodeOutput (v : Vec3 α) : Vec3 α := (v.map srgbEncode).map clamp01

/-- The encoded sRGB output of a loaded pixel, in `[0, 1]` (`convert_pixel`). -/
def convertPixel (c : PipelineConfig α) (m : Mat3 α) (v : Vec3 α) : Vec3 α :=
  let d := convertPixelDisplay c m v
  encodeOutput (match c.tone with
    | some lut => applyTone lut d
    | none => d)

/-- The CIELAB (D65) of a loaded pixel's display value: what every statistic and the search's
losses read (`linear_srgb_to_lab(to_display(...))`). -/
def pixelLab (c : PipelineConfig α) (m : Mat3 α) (v : Vec3 α) : Lab α :=
  linearSrgbToLab (convertPixelDisplay c m v)

/-- An encoded value in `[0, 1]` as an 8-bit code (`quantize_u8`). -/
def quantizeU8 (v : α) : Nat := roundNat (v * 255.0)

/-- An encoded value in `[0, 1]` as a 16-bit code (`quantize_u16`). -/
def quantizeU16 (v : α) : Nat := roundNat (v * 65535.0)

/-- The 8-bit sRGB preview pixel of a 16-bit camera pixel: what the intent network is shown. -/
def previewPixel (c : PipelineConfig α) (m : Mat3 α) (px : Vec3 Nat) : Vec3 Nat :=
  (convertPixel c m (loadU16 px)).map quantizeU8

end Negicc
