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

The neutral head and the tone heads read the same inference (`NeutralHead::run`, `ToneHead::run`
in `dinov3_engine.h`). Their forward passes are stated here over their weights, which are data:

* **neutral head**, per cell, from the cell's backbone token `t` (768 values): standardised
  `x = (t - mean) / std`, one hidden layer `a = W1 x + b1`, `z = b2 + w2 · GELU(a)`, and
  `p = 1 / (1 + exp(-z))`, the probability that the cell shows a neutral surface (read by the
  curve fit, 4.3);
* **tone heads**, from the 1920 features (the pooled features and the tokens' spread): standardised,
  two hidden layers with GELU, then per target a mean and a log-variance `o`, the mean
  `o_mean ys + ym` and the spread `exp(0.5 clamp(o, -6, 4)) ys` (read by the tone stage, 4.4).

GELU is the exact one, `0.5 a (1 + erf(a / √2))`; the error function is a parameter of the
definitions, as `Scalar` does not carry it.
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

/-! ## The heads -/

/-- The exact GELU (Hendrycks and Gimpel, 2016) over an error function `erf`:
`0.5 a (1 + erf(a / √2))`. -/
def gelu (erf : α → α) (a : α) : α := 0.5 * a * (1.0 + erf (a * 0.70710678118654752440))

/-- The dot product of two arrays, summed in order from the first entry, from `init`. -/
def dotFrom (init : α) (w x : Array α) : α :=
  (List.range (min w.size x.size)).foldl (fun s k => s + w.getD k 0.0 * x.getD k 0.0) init

/-- The neutral head's weights (`neutral_head.bin`): the standardisation (`mean`, `std`, one per
input), the hidden layer (`w1`, one row of inputs per hidden unit, and `b1`) and the output
(`w2`, `b2`). -/
structure NeutralWeights (α : Type) where
  mean : Array α
  std : Array α
  w1 : Array (Array α)
  b1 : Array α
  w2 : Array α
  b2 : α

/-- The neutral head's `p` for one cell's token (`NeutralHead::run`). -/
def neutralP (erf : α → α) (W : NeutralWeights α) (token : Array α) : α :=
  let x := (List.range token.size).toArray.map fun k => (token.getD k 0.0 - W.mean.getD k 0.0) / W.std.getD k 1.0
  let a := (List.range W.b1.size).toArray.map fun j => dotFrom (W.b1.getD j 0.0) (W.w1.getD j #[]) x
  let z := dotFrom W.b2 W.w2 (a.map (gelu erf))
  1.0 / (1.0 + exp (-z))

/-- The tone heads' weights (`tone_head.bin`): the input's standardisation (`xm`, `xs`), two hidden
layers (`w1`, `b1`, `w2`, `b2`), the output layer (`w3`, `b3`: rows `t` for the means and `11 + t`
for the log-variances) and the targets' scale (`ym`, `ys`). -/
structure ToneWeights (α : Type) where
  xm : Array α
  xs : Array α
  w1 : Array (Array α)
  b1 : Array α
  w2 : Array (Array α)
  b2 : Array α
  w3 : Array (Array α)
  b3 : Array α
  ym : Array α
  ys : Array α

/-- The tone heads' mean and spread for each of the 11 targets (L2, L5, L10, L50, L90, L95, L98,
IQR, R9010, Cmean, C90) from the 1920 features (`ToneHead::run`). -/
def toneHeads (erf : α → α) (W : ToneWeights α) (features : Array α) : Array (α × α) :=
  let x := (List.range features.size).toArray.map fun k => (features.getD k 0.0 - W.xm.getD k 0.0) / W.xs.getD k 1.0
  let layer (w : Array (Array α)) (b : Array α) (v : Array α) : Array α :=
    (List.range b.size).toArray.map fun j => gelu erf (dotFrom (b.getD j 0.0) (w.getD j #[]) v)
  let a2 := layer W.w2 W.b2 (layer W.w1 W.b1 x)
  let out (j : Nat) : α := dotFrom (W.b3.getD j 0.0) (W.w3.getD j #[]) a2
  (List.range 11).toArray.map fun t =>
    (out t * W.ys.getD t 0.0 + W.ym.getD t 0.0, exp (0.5 * clamp (out (11 + t)) (-6.0) 4.0) * W.ys.getD t 0.0)

/-! ## The intent network's input -/

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
