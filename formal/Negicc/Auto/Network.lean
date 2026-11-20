/-
The networks: how their input is made from a preview, what they give back, what is checked.
-/
module

public import Negicc.Auto.Geometry
public import Negicc.Auto.Resample

@[expose] public section

/-!
# The networks

The search judges a render with a vision network (`DinoV3Engine::infer` in `dinov3_engine.cc`).
Its input is made from the 8-bit sRGB preview of the frame's inner crop:

1. the preview is resized to the network's size (`aspectShape`) with Pillow's antialiased
   bilinear filter (`aaResize`);
2. each 8-bit value becomes `(v / 255 - mean_c) / std_c` with ImageNet's statistics, in planes
   of red, green and blue (`pixelValues`);
3. each cell gets three tone weights, shadow, midtone and highlight, from the `L*` of its mean
   colour (`toneWeights`, `compute_tone_weights` in `dinov3_preprocess.cc`).

The network itself (the frozen backbone and the stock's intent head) is a function of stated
shape, `IntentNet`: from the pixel values of a `w × h` image and the `N` cells' tone weights it
gives, per cell, the intent field: a CIELAB mean `μ` and a spread `σ`. Its weights are data, not
formulas. The search uses a field only when it has exactly one `μ` and one `σ` per cell of the
frame's grid and every `σ` is positive (`fieldValid`; `infer_checked` in
`dinov3_auto_solver.cc`, `SearchFrame::predict` in `multi_target_optimizer.cc`); otherwise the
search fails.

The neutral head (per cell `p`) and the tone heads (from the backbone's pooled features and the
spread of its tokens) read the same inference; they are networks of stated shape as well
(`NeutralHead`, `ToneHead` in `dinov3_engine.h`) and their users are the curve fit (4.3) and the
tone stage (4.4).
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- An intent field: per cell of the grid, the predicted CIELAB mean `μ` and its spread `σ`, cells
in row-major order. -/
structure IntentField (α : Type) where
  mu : Array (Lab α)
  sigma : Array (Lab α)

/-- The network's input image: pixel values in three planes (red, green, blue), each row-major. -/
structure NetInput (α : Type) where
  width : Nat
  height : Nat
  /-- `3 · width · height` values: the red plane, then green, then blue. -/
  pixels : Array α
  /-- One triple (shadow, midtone, highlight) per cell. -/
  poolWeights : Array (Vec3 α)

/-- The intent network as a function of stated shape: from the input of a `w × h` image whose grid
has `N` cells it gives one `μ` and one `σ` per cell. The weights of the backbone and of the
stock's head are data; nothing here depends on them. -/
structure IntentNet (α : Type) where
  run : NetInput α → IntentField α

/-- ImageNet's per-channel mean, which the backbone's weights expect. -/
def imagenetMean : Vec3 α := ⟨0.485, 0.456, 0.406⟩

/-- ImageNet's per-channel standard deviation. -/
def imagenetStd : Vec3 α := ⟨0.229, 0.224, 0.225⟩

/-- An 8-bit value as the network reads it in channel `c`: `(v / 255 - mean_c) / std_c`. -/
def normalizeU8 (mean std : α) (v : Nat) : α := ((v : α) / 255.0 - mean) / std

/-- The pixel values of an 8-bit image, plane by plane (step 2 of `DinoV3Engine::infer`). -/
def pixelValues (img : Array (Vec3 Nat)) : Array α :=
  img.map (fun p => normalizeU8 imagenetMean.x imagenetStd.x p.x) ++
  img.map (fun p => normalizeU8 imagenetMean.y imagenetStd.y p.y) ++
  img.map (fun p => normalizeU8 imagenetMean.z imagenetStd.z p.z)

/-- The shadow, midtone and highlight weights of a cell of lightness `L`: the shadow weight falls
from 1 to 0 over `L*` 25 to 35, the highlight weight rises over 75 to 85, the midtone weight is
what is left. -/
def toneBins (L : α) : Vec3 α :=
  let wsh := clamp ((30.0 + 10.0 / 2.0 - L) / 10.0) 0.0 1.0
  let whi := clamp ((L - (80.0 - 10.0 / 2.0)) / 10.0) 0.0 1.0
  ⟨wsh, clamp (1.0 - wsh - whi) 0.0 1.0, whi⟩

/-- The tone weights of each cell of an image `gw · patch` pixels wide (`compute_tone_weights`):
the cell's mean 8-bit value per channel, decoded through the sRGB curve, its `L*`, and the
weights of `toneBins`. -/
def toneWeights (img : Array (Vec3 Nat)) (gw gh patch : Nat) : Array (Vec3 α) :=
  let tw := gw * patch
  ((List.range gh).flatMap fun gy => (List.range gw).map fun gx =>
    let px := (List.range patch).flatMap fun py => (List.range patch).map fun qx =>
      pixelAtNat img tw (gx * patch + qx) (gy * patch + py)
    let total : Vec3 Nat := px.foldl (fun s p => ⟨s.x + p.x, s.y + p.y, s.z + p.z⟩) ⟨0, 0, 0⟩
    let inv : α := 1.0 / ((patch * patch : Nat) : α)
    let lin : Vec3 α := total.map fun (n : Nat) => srgbDecode ((n : α) * inv / 255.0)
    toneBins (lstarOfY (linearSrgbLuminance lin))).toArray

/-- The network's input of an 8-bit `w × h` preview (steps 1 to 3 of `DinoV3Engine::infer`). -/
def netInput (preview : Array (Vec3 Nat)) (w h maxSize patch : Nat) : NetInput α :=
  let s := aspectShape (α := α) w h maxSize patch
  let resized := aaResize (α := α) preview w h s.targetW s.targetH
  ⟨s.targetW, s.targetH, pixelValues resized, toneWeights resized s.gridW s.gridH patch⟩

/-- The intent field a network predicts from an 8-bit preview (`DinoV3Engine::infer`). -/
def infer (net : IntentNet α) (preview : Array (Vec3 Nat)) (w h maxSize patch : Nat) :
    IntentField α :=
  net.run (netInput preview w h maxSize patch)

/-- A field the search can use on a grid of `n` cells: one `μ` and one `σ` per cell, every `σ`
positive (`infer_checked`, `SearchFrame::predict`; finiteness is not modelled). -/
def fieldValid (f : IntentField α) (n : Nat) : Bool :=
  f.mu.size == n && f.sigma.size == n &&
    f.sigma.all fun s => decide (0.0 < s.L) && decide (0.0 < s.a) && decide (0.0 < s.b)

end Negicc
