# Formulas of the scene conversion

Every formula of the scene conversion that the Lean specification covers, in ordinary notation, with
the C++ function it mirrors and its Lean name. The sections follow `DESIGN.md`: reading the scan and
the film base (2), the inner crop the statistics read (3.3), the per-pixel conversion and the outputs
(4.1), given input curves on a white-light profile (4.3) and the tone stage (4.4). Two parts that the
code is still changing are left out (see "Not in this specification" at the end): a bracket's
calibrated range on a frame (3.3), and the white-light curve fit with the re-anchoring after the
knobs (4.3). The coverage table at the end says for each function whether properties are proved and
whether it is checked against the program; the lists after it name the definitions the check does not
evaluate and what is not formalized, with reasons.

Notation: `clamp(x, lo, hi) = min(max(x, lo), hi)`, `clamp01(x) = clamp(x, 0, 1)`, `log2`, `exp2`,
`round` (halves away from zero), `rint` (halves to even). A 16-bit code `v` loads as `v / 65535`.

## 2.1 Reading the scan

**Black level** (`np_input.cc`: `subtract_cfa_black_in_place`; Lean `subtractBlack`,
`whiteAfterBlack`). Each photosite of colour `c`:

```
v' = max(0, v − (black + cblack[c] + pattern[r mod dr, c mod dc]))
white' = max(1, maximum − max_c(black + cblack[c] + mean pattern of c))
```

**Mirror at the edge** (`rgb_view.cc`: `get_px_refl_cpu`; Lean `reflect`, `Mosaic.at`): an index `i`
outside `[0, n − 1]` reads `−i` below and `2(n − 1) − i` above, held to `[0, n − 1]`.

**Half-size image** (`cpu_preview_debayer`, the view of `RgbView::from_cfa_half`; Lean
`halfSizePixel`). The 2×2 quad at `(2y, 2x)` of an RGGB mosaic:

```
(R, G, B) = (R, ⌊(G1 + G2) / 2⌋, B)
```

**Grain-aware debayer** (`reconstruct_green_soft_cpu`, `cpu_grain_aware_debayer`; Lean `greenSoft`,
`grainAwarePixel`, `grainAwarePixelU16`). Green at a red or blue site `c` with neighbours `u, d, l, r`
(green) and `u2, d2, l2, r2` (same colour, two away):

```
Lv = 2c − u2 − d2          Lh = 2c − l2 − r2
dv = |u − d| + |Lv|        dh = |l − r| + |Lh|
gv = (u + d)/2 + Lv/4      gh = (l + r)/2 + Lh/4
wv = 1 / (4096 + dv²)      wh = 1 / (4096 + dh²)
G  = clamp((wv gv + wh gh) / (wv + wh), 0, white)
```

Red and blue where they are not sampled, from neighbours `n` of that colour (two horizontal, two
vertical or four diagonal), with `k = 16`:

```
C = (G + k) · mean_n((C_n + k) / (G_n + k)) − k
```

Where the centre sample or `G` is at `white − 64` or above, `C` is the plain mean of the same
neighbours. Each channel is held to `[0, 65535]` and rounded.

**Pixel shift** (`cpu_pixel_shift_merge`; Lean `pixelShiftPixel`). Captures `f0 … f3` read at
`(r, c)`, `(r − 1, c)`, `(r − 1, c + 1)`, `(r, c + 1)`; red and blue from the capture whose site has
that colour, `G = ⌊(G1 + G2) / 2⌋`; each channel held at `white`, green also when either green site
reached it.

**Three captures** (`np_input.cc`: `load_rgb_input`; Lean `captureScale`, `tiffHalf`). When all three
captures record their shutter and ISO, with `e_k` the shutter × ISO of capture `k`, the green and
blue captures and their half-size images are scaled to the first's exposure; otherwise (three TIFF
captures, for instance) nothing is scaled:

```
v_k' = min(65535, round(v_k e_0 / e_k))       (k = 1, 2; unchanged when e_0 = e_k)
half = ⌊(a + b + c + d + 2) / 4⌋              (a TIFF's 2×2 block, before its scaling; and the
                                               assembled image's)
```

The analysis reads the three captures' own half-size images when all three have the image's half
size, whatever the frame (Lean `capturePlanesUsed`); otherwise, and for an RGB file, the half-size
image of the assembled image. The frame is cut from the full image after the analysis; the analysis
itself is told where the frame lies (`--roi`, `--roi auto`'s detection, or the first RAW capture's
inset crop, which it searches as it searches `--roi`).

**White-light film-base balance** (`white_light_balance.h`: `film_base_balance_gains`; Lean
`filmBaseBalanceGains`, `rec2020Rgb`, `Mat3.adjugate`, `Mat3.det`). With `C = rec2020_rgb · rgb_cam`:

```
K_c = (adj(C) f_profile)_c / (adj(C) f)_c,   then K_c ← K_c / max(K)
```

refused (gains unchanged) when `(adj(C) f)_c / det C` or `(adj(C) f_profile)_c / det C` is not
positive for some `c`. The program also refuses a `K_c`, before or after the division by the largest,
that is not a finite positive number, which only floating point's overflow and underflow can give (over
`ℝ` the conditions above imply it).

## 2.2 Flat field

(`flat_field.cc`: `axis_table`; `flat_field.h`: `flat_divide`; Lean `flatAxis`, `flatGain`,
`flatDivide`.) Pixel `k` of an axis, `h = k` on the half-size decode and `h = (k + 0.5)/2 − 0.5` on
the full-size one, block `B`, `N` blocks:

```
f = clamp((h + 0.5)/B − 0.5, 0, N − 1)     i = min(⌊f⌋, N − 2)     t = f − i
gain = lerp(lerp(g00, g10, ty), lerp(g01, g11, ty), tx)        (first along y, then along x)
v' = min(rint(v / gain), 65535)
```

## 2.3 Film base

**Merged matrix** (`profile_bundle.cc`: `compute_station_merged_matrix`; Lean `mergedMatrix`,
`captureExposure`). Base `f` after the crosstalk correction, `n = 55000`:

```
s_c = n / f_c · ρ   (1 when f_c ≤ 0)        ρ = exposure(base capture) / exposure(scan)
M_frame = diag(s) X          exposure = shutter · ISO / 100
```

`ρ = 1` for a base read on the scan. A scan pixel of bare film comes out at the code `n`, `n / 65535`
after the load (`mergedMatrix_mulVec_scan_base`).

**Base from a capture** (`np_film_base.cc`; Lean `baseFromCapture`): `X v · exposure(scan) /
exposure(capture)`; refused when the film it reads is at the sensor's clip before the flat field.

**Rebate** (`film_base_detector.cc`: `detect_film_base`; Lean `rebatePlaced`, `rebateBox`,
`patchClipped`, `outshoneStops`, `rebateAccepted`, `medianU16`; checked through `detect_film_base` on
synthetic decodes, with the thinnest film, which the segmentation gives, as input). Placed from a detection of confidence
≥ 0.5 on a decode with `|sx − sy| ≤ 0.02 max(sx, sy)`; the box inset by 15 % on each side, scaled and
rounded inward, at least 32 samples; refused when more than 0.1 % of its samples reach `0.98 white`
in a channel before the flat field. With `dec` its per-channel median:

```
outshone = max_c log2(thinnest_c / max(dec_c, 1))
taken when min(dec) > 0, min(X dec) > 0 and outshone ≤ 0.5
```

**Thinnest film** (`percentile_f32`; Lean `percentileF32`, `thinnestAccepted`): the 99.5th
percentile per channel of the film pixels, numpy's linear method: rank `r = (n − 1) q / 100`,
`γ = r − ⌊r⌋`, `lo + (hi − lo) γ` (from `hi` when `γ ≥ 0.5`). Taken when every channel is above 1 and
`X` of it is positive. **Order** (Lean `filmBaseSource`): capture, rebate, thinnest film, the
bundle's base.

## 2.4 Crosstalk correction

`X` is the bundle's crosstalk correction (`crosstalk_profile.crosstalk_correction_matrix`, or `-r`,
`-g`, `-b`), given data here; it is measured when a profile set is built, outside this repository.
White light has `X = I`. It enters through `M_frame = diag(s) X`.

## 2.5 The input matrix and the soft knee

(`color_pipeline.h`: `load_u16`, `candidate_matrix`, `knee_compress`, `convert_pixel_corrected`;
Lean `loadU16`, `candidateMatrix`, `kneeCompress`, `autoKnee`, `correctedRgb`.)

```
x = knee(diag(E, E g, E b) M_frame v / 65535)
knee(x) = 0                                        x ≤ 0
        = min(1, x)                                x ≤ k  (or k ≥ 1)
        = min(0.9999, k + (1 − k) tanh((x − k)/(1 − k)))   otherwise;   k = 0.95
```

## 3.3 The inner crop

(`color_pipeline.h`: `compute_inner_crop`; `multi_target_optimizer.h`: `ParamSearchOptions::crop_ratio`;
Lean `innerCrop`, `searchCropRatio`.)

```
x0 = min(round(W (1 − r) / 2), ⌊(W − 1)/2⌋), width W − 2 x0 (likewise y);  r = 0.90
```

## 3.4 The profiles' prediction errors

The conversion does not use them: the bundle keeps each bracket's `profcheck` report
(`profcheck_output`), and nothing reads it.

## 4.1 The per-pixel conversion

**Profile** (`apply_trc`, `interpolate_clut`, `apply_profile`; Lean `tableEval`, `tableAt`,
`clutEval`, `tetraBlend`, `applyProfile`, `Profile.hasClut`, `pcsScale`).

```
table(v) = lerp(T[k], T[k+1], s − k),   s = v (n − 1),   k = clamp(⌊s⌋, 0, n − 2)   (n ≥ 2;
           a table of one sample is that sample, an empty one the identity)
the parser takes only input curves, a cLUT, an identity matrix at most, output curves
           (profileLayoutAccepted), and sets M = I, o = 0 (Profile.matrixAccepted); it also requires
           an RGB to XYZ profile and every stage three channels to three, conditions the definition
           does not carry
t = clamp01(M in(x) + o)
clut: the value passes through without a table; else
      s = t (dim − 1), (i, j, k) the lower corner held inside the grid, the upper one from it,
      (dr, dg, db) the position in the cell;
      with d1 ≥ d2 ≥ d3 the sorted positions:
      value = (1 − d1) v000 + (d1 − d2) v_a + (d2 − d3) v_b + d3 v111
XYZ = 1.99996948 · out(clut(t))       (out(in(x)) when there is no colour table)
```

**Linear sRGB** (`pcs_to_linear_srgb_unclamped`, `pcs_to_linear_srgb_pre_shoulder`; Lean
`pcsToLinearSrgbMatrix`, `pcsWhite`, `pcsToLinearSrgbUnclamped`, `pcsToLinearSrgbPreShoulder`):

```
rgb = max(0, P XYZ),   P = [[3.13427485, −1.61727573, −0.490724277],
                            [−0.978795611, 1.91616171, 0.0334533327],
                            [0.0719770074, −0.228984979, 1.40571822]]
```

`P w = (1, 1, 1)` within `1e-8` (`pcsToLinearSrgb_white`).

**Shoulder** (`set_shoulder`, `apply_shoulder_channel`, `apply_shoulder`, `to_display`,
`shoulder_x_hi_anchored`, `max_hist_bin`, `max_hist_quantile`, `shoulder_x_hi_from_hist`; Lean
`shoulderCeiling`, `shoulderSlope`, `shoulderKneeGap`, `bisect`, `setShoulder`, `applyShoulderChannel`,
`applyShoulder`, `toDisplay`, `shoulderXHiAnchored`, `maxHistBin`, `maxHistQuantile`, `maxHistOf`,
`shoulderXHiFromHist`):

```
off for x_hi ≤ 1.0001;  x_hi ← min(x_hi, 2.5)
C = 1 − 0.012 · min(1, (x_hi − 1) / 0.1),   t_hi = log2 x_hi
m(t) = (1.055 / 2.4) (2^t)^(1/2.4) ln 2
t_k: root of t_hi − t − 2 (C − sRGB(2^t)) / m(t) on [−3, 0] by 100 bisections
s(x) = x                                         x ≤ 2^t_k
     = sRGB⁻¹(C − (C − s_k) u²),  u = 1 − (t − t_k)/(t_hi − t_k),  t = log2 x
     = sRGB⁻¹(C)                                  t ≥ t_hi
display = clamp01(s(rgb))
x_hi: the 99.5 % point q of max(R, G, B) (4096 bins over [0, 2], linear inside a bin) mapped to
      code 242 by 60 bisections of x_hi on [1.0002, 2.5]
```

**Outputs** (`encode_output`, `convert_pixel`, `quantize_u8`, `quantize_u16`, the clipping rules;
Lean `Encoding`, `PipelineConfig.activeShoulder`, `PipelineConfig.activeTone`, `encodeOutputWith`,
`convertPixel`, `outputPixel`, `correctedPixel`, `previewPixel`, `quantizeU8`, `quantizeU16`,
`linearChannelWhiteClipped`, `linearChannelBlackClipped`, `pixelLab`):

```
-P srgb:      code = round(65535 · clamp01(sRGB(tone(display))))
-P srgb-g10:  code = round(65535 · clamp01(rgb))                (no shoulder, tone or curve)
no -P:        code = round(65535 · x)                            (corrected RGB, profile embedded)
preview:      code = round(255 · clamp01(sRGB(display)))         (what the network is shown)
white-clipped channel: rgb ≥ 0.9953029;  black-clipped: rgb < 0.00023679829 (all three)
```

**sRGB and CIELAB** (`srgb_encode`, `srgb_decode`, `lab_f`, `linear_srgb_to_lab`,
`linear_srgb_to_L`, `srgb_to_lab`; Lean `srgbEncode`, `srgbDecode`, `labEpsilon`, `labKappa`, `labF`,
`labFInv`, `linearSrgbLuminance`, `linearSrgbToXyzRel`, `lstarOfY`, `xyzRelToLab`, `linearSrgbToLab`,
`linearSrgbToL`, `srgbToLab`):

```
sRGB(l) = 12.92 l (l ≤ 0.0031308),  1.055 l^(1/2.4) − 0.055 otherwise
f(t) = t^(1/3) (t > 216/24389),  ((24389/27) t + 16)/116 otherwise
Y = 0.212671 R + 0.715160 G + 0.072169 B
L* = 116 f(Y) − 16,   a* = 500 (f(X/0.950456) − f(Y)),   b* = 200 (f(Y) − f(Z/1.088754))
```

## 4.3 Given input curves on a white-light profile

(`input_curves.h`: `curve_stops`, `input_curve`, `compose_input_table`, `CombinedProfile::build`; Lean
`curveStops`, `inputCurve`, `composeInputTable`, `combinedProfile`.) Curves the fit found or a run
recorded, composed into the bracket's green and blue input tables:

```
u = min(log2(max(v, 1e-12) / ref), 0)
f(u) = u (1 + a + k tanh(u − u_med))                       held: f(u_lo) + (u − u_lo) below u_lo
v' = ref 2^f(u) for v < ref (and a curve that is not the identity), v otherwise
table_i = in(clamp01(curve(i / (n − 1))))                  (green and blue; red and the rest unchanged)
```

## 4.4 The tone stage

(`tone_stage.cc`: `tone_curve_L`, `tone_table`, `tone_curve_ok`, `tone_descriptors`, `tone_targets`,
`tone_solve`, `tone_plan`, `tone_pick_clip_share`, `tone_clip_sample`, `tone_clip_shares`, `tone_plan_for_image`;
`color_pipeline.h`: `tone_lab_fprime`, `tone_table_Y`, `apply_tone`. Lean `ToneCurve`, `toneCurveL`,
`toneTable`, `toneCurveOk`, `toneLstarOfU8`, `percentileSorted`, `toneDescriptors`, `toneTargets`,
`toneResiduals`, `toneCost`, `toneLM`, `toneSolve`, `tonePlan`, `tonePickClipShare`, `toneClipSample`,
`tonePlanForImage`, `toneLabFprime`, `toneLuminance`, `toneTableY`, `applyTone`.)

```
x = clamp(L/100, 1e-5, 1 − 1e-5),   g = x^γ
f(L) = black + (white − black) g^c / (g^c + (1 − g)^c)
table_i = finv((clamp(f(116 f(t²) − 16), 0, 100) + 16)/116),   t = i / 8192
descriptors: L* per pixel of the 8-bit preview; P(p) = L[i] + (r − i)(L[i+1] − L[i]), r = p/100 (n − 1)
targets: L2' = L2 + (L2_pred − L2),  L98' = L98 + ½ s (L98_pred − L98),
         R' = R + ½ s (R_pred − R),  L50' = L50          (s the highlight share)
residuals: f(L2) − L2', f(L98) − L98', f(L50) − L50', f(L90) − f(L10) − R',
           0.02 black/5, 0.02 (white − 100)/5, 0.02 ln γ / 0.1, 0.02 ln c / 0.1,
           0.07 (f(100) − min(100, f(L98) + 100 − L98))
bounds: black ±15, white 80 … 110, γ 0.7 … 1.45, c 0.7 … 1.5; starts at white 100, 110, 85
share: 1 if the curve clips within 0.5 % more than without it, else the first of 0.75, 0.5, 0.25, 0
       that does, else the one that clips least (sample: the 2 × 2 at each 8 × 8 block's origin inside
       the inner crop, 0.90; 16-bit outputs; white: a channel at 65535, black: a channel at 0)
Y' = table(sqrt(clamp01(Y)) · 8192) (· Y above 1),   Y = 0.212671 R + 0.715160 G + 0.072169 B
out = Y' + (v − Y) min(f'(max(Y, 0)) / f'(max(Y', 1e-9)), 8)
f'(t) = 1 / (3 t^(2/3)) above 216/24389, 841/108 below
```

The Levenberg-Marquardt of `tone_solve` (Lean `toneLM`, `solve4`): central differences of step
`1e-6 max(1, |q|)` kept inside the bounds, damping `λ max(A_aa, 1e-12)`, a number at a bound that the
gradient pushes outward held, the step clamped into the bounds, `λ/3` on success (at least `1e-12`),
`4λ` on failure, at most 20 tries and 200 iterations.

## Coverage

"Proved" lists the theorems about the definition (over `ℝ`); "checked" means compared over `Float`
with the program's value on the committed vectors (`lake exe check`). "Read" means compared with the
code by reading only.

| C++ function | Lean | Proved | Checked |
|---|---|---|---|
| `srgb_encode` | `srgbEncode` | round trip, `srgbEncode_not_monotone`, `srgbEncode_le_of_le` | yes |
| `srgb_decode` | `srgbDecode` | `srgbDecode_mono`, round trips | yes |
| `lab_f`, `kLabEpsilon`, `kLabKappa` | `labF`, `labEpsilon`, `labKappa` | `labF_mono`, `labF_line_at_break`, `cbrt_labEpsilon` | yes |
| `tone_lab_finv` | `labFInv` | `labFInv_labF` | yes (in `toneTable`) |
| `linear_srgb_to_lab`, `linear_srgb_to_L`, `srgb_to_lab` | `linearSrgbToLab`, `linearSrgbToL`, `srgbToLab` | grey, white, `linearSrgbToLab_L`, `lstarOfY_mono` | yes |
| `apply_trc` | `tableEval`, `tableAt` | `tableEval_mem`, `tableAt_natCast` | yes |
| `parse_film_profile_pipeline` (the layout taken, the identity matrix set) | `profileLayoutAccepted`, `Profile.matrixAccepted` | | `matrixAccepted` on the bundles' profiles; the layout read |
| `apply_matrix`, `candidate_matrix` | `Mat3.mulVec`, `candidateMatrix` | `candidateMatrix_mulVec`, `_one`, `_mul_exposure` | yes |
| `subtract_cfa_black_in_place` | `subtractBlack`, `whiteAfterBlack` | | read |
| `cpu_preview_debayer` | `halfSizePixel` | | yes |
| `reconstruct_green_soft_cpu`, `cpu_grain_aware_debayer` | `greenSoft`, `grainAwarePixel`, `grainAwarePixelU16` | | yes |
| `cpu_pixel_shift_merge` | `pixelShiftPixel` | | yes |
| `load_rgb_input` (scaling, the half-size images) | `captureScale`, `tiffHalf`, `capturePlanesUsed` | | read |
| `film_base_balance_gains` | `filmBaseBalanceGains` | | yes |
| `axis_table`, `flat_divide` | `flatAxis`, `flatGain`, `flatDivide` | | yes |
| `compute_station_merged_matrix` | `mergedMatrix`, `captureExposure` | `mergedMatrix_mulVec_base`, `_scan_base` | yes |
| `detect_film_base` (rules) | `rebatePlaced`, `rebateBox`, `patchClipped`, `outshoneStops`, `rebateAccepted`, `thinnestAccepted`, `filmBaseSource` | | yes |
| `percentile_f32`, `median_u16` | `percentileF32`, `medianU16` | | yes |
| film base capture scaling | `baseFromCapture` | | read |
| `knee_compress` | `kneeCompress`, `autoKnee` | `kneeCompress_mono`, `_step`, `_of_le`, `_of_nonpos`, `_le_one` | yes |
| `load_u16`, `convert_pixel_corrected` | `loadU16`, `correctedRgb` | | yes |
| `compute_inner_crop` | `innerCrop` | | yes |
| `ParamSearchOptions::crop_ratio` | `searchCropRatio` | | yes |
| `interpolate_clut` | `clutEval`, `tetraBlend` | `clutEval_const`, `clutEval_mem` | yes |
| `apply_profile` | `applyProfile`, `Profile.hasClut` | | yes |
| `kPcsWhite`, `pcs_to_linear_srgb_unclamped`, `_pre_shoulder` | `pcsWhite`, `pcsToLinearSrgbUnclamped`, `pcsToLinearSrgbPreShoulder` | `pcsToLinearSrgb_white` | yes |
| `convert_pixel_pcs`, `convert_pixel_linear`, `convert_pixel_display` | `convertPixelPcs`, `convertPixelLinear`, `convertPixelDisplay` | `convertPixelDisplay_of_off` | yes |
| `set_shoulder` | `setShoulder`, `shoulderCeiling`, `shoulderKneeGap`, `bisect` | `setShoulder_of_le` | yes |
| `apply_shoulder_channel`, `apply_shoulder`, `to_display` | `applyShoulderChannel`, `applyShoulder`, `toDisplay` | identity below the knee, ceiling, parabola, `toDisplay_mem` | yes |
| `shoulder_x_hi_anchored` | `shoulderXHiAnchored` | | yes |
| `max_hist_bin`, `max_hist_quantile`, the histogram's fill (`cpu_highlight_clip`), `shoulder_x_hi_from_hist` | `maxHistBin`, `maxHistQuantile`, `maxHistOf`, `shoulderXHiFromHist` | | yes |
| `encode_output`, `convert_pixel` | `encodeOutputWith`, `encodeOutput`, `convertPixel` | `convertPixel_of_no_tone`, `_of_short_tone`, `convertPixel_linear`, `encodeOutput_mem` | yes |
| `quantize_u8`, `quantize_u16` and the outputs' codes | `quantizeU8`, `quantizeU16`, `outputPixel`, `correctedPixel`, `previewPixel` | | yes (exact) |
| `linear_channel_white_clipped`, `_black_clipped` | `linearChannelWhiteClipped`, `linearChannelBlackClipped`, `pixelWhiteClipped`, `pixelBlackClipped` | | yes |
| `curve_stops`, `input_curve` | `curveStops`, `inputCurve` | `inputCurve_of_ge`, `inputCurve_zero`, `curveStops_zero`, `curveStops_hold` | yes |
| `compose_input_table`, `CombinedProfile::build` | `composeInputTable`, `combinedProfile` | | yes |
| `tone_curve_L` | `toneCurveL` | `toneCurveL_mono`, `toneCurveL_identity` | yes (in table and plan) |
| `tone_table`, `tone_curve_ok` | `toneTable`, `toneCurveOk` | | yes |
| `tone_descriptors` | `toneDescriptors`, `percentileSorted`, `toneLstarOfU8` | | yes |
| `tone_targets` | `toneTargets` | `toneTargets_share_zero` | yes (through `tone_plan`) |
| `tone_solve`, `tone_plan` | `toneResiduals`, `toneCost`, `toneLM`, `solve4`, `toneSolve`, `tonePlan` | | yes |
| `tone_pick_clip_share`, `tone_clip_sample` with `tone_clip_shares`, `tone_plan_for_image` | `tonePickClipShare`, `toneClipSample`, `tonePlanForImage` | | yes |
| `tone_lab_fprime`, `tone_table_Y`, `apply_tone` | `toneLabFprime`, `toneTableY`, `applyTone` | `applyTone_grey` | yes |

## Defined but not evaluated by the check

- `subtractBlack`, `whiteAfterBlack`, `captureScale`, `tiffHalf`, `capturePlanesUsed` (`np_input.cc`)
  and `baseFromCapture` (`np_film_base.cc`): those files are not in the library the generator links,
  and the functions work on LibRaw's state or on files.
- `profileLayoutAccepted`: the parser reads Little-CMS's pipeline of an ICC file; the check's own
  reader takes only the bundles' layout, and every bundle profile is taken. The parser's two other
  conditions, an RGB to XYZ profile and every stage three channels to three, are not in the
  definition.
- `density`, `transmission`, `sq`, `mean`, the componentwise `add` and `sub`: helpers of the base for
  the other parts (the print emulation's density); the scene conversion does not use them.

## Not formalized

- **The Auto search** (4.2) and the networks (3.5): specified separately, in #211.
- **Where the passes run.** On a CUDA build with a device the search's previews and clip passes run
  on the GPU (`gpu_passes.cu`), through the same per-pixel functions of `color_pipeline.h`; the
  specification is over the real numbers and covers both, but the vectors are computed on the CPU
  (the generator runs without a device), so the GPU's last-place rounding of `powf`, `tanhf` and
  `cbrtf` is not checked.

- **LibRaw and Little-CMS.** LibRaw's decoding (its demosaic for other cameras, for `-q 0` and more
  and for the white-light decode, its white balance and camera matrix to linear Rec.2020) and the
  colour-managed conversions through Little-CMS (a TIFF or JPEG with an embedded profile to
  Rec.2020, the output through an ICC profile with `-P <profile>`): third-party code; their results
  are the inputs and outputs of the formulas here.
- **Image analysis whose result is a region or a mask**: the frame detector, and the film-base
  segmentation (bare light, the holder, their borders, OpenCV's area reduction to 512 pixels). The
  formulas take the region and the film pixels as given.
- **Geometry**: the crop, rotation and flips that map output pixels to mosaic positions
  (`map_inverse_geometry_cpu`, `crop_and_orient`) are index bookkeeping; the specification uses the
  identity mapping.
- **Parsing and records**: ICC profiles (the check reads the bundles' `lut16Type` itself), the
  bundle's JSON, the flat field's `.npz`, the sidecar.
- **NaN, infinities and finiteness tests**, which the C++ uses to skip cells or values, and the
  guards for absent data (no profile, a scan exposure of zero, a missing base shutter): the definitions assume finite values and present data.
- **The network heads** that give the tone stage its predictions: an
  opaque function whose outputs are inputs here (specified with the Auto search in #211).

## Not in this specification

Two parts the code is still changing are left out of this specification. Their definitions, theorems,
vectors and checks as they stood against `47a187d` are in this branch's history at `21b63a4`
(`formal/Negicc/Scene/Bracket.lean`, `formal/Negicc/WhiteLight/Fit.lean` and the anchoring in
`formal/Negicc/WhiteLight/Curve.lean`):

- **A bracket's calibrated range on a frame** (DESIGN 3.3; `multi_target_optimizer.cc`: the scene's
  green points `compute_frame_stats`, `u16_percentile_linear`, `bracket_enclosure`, `pull_centre`,
  `frame_scale`'s `e_norm`; a bracket's green range and exposure as the loader derives them).
- **The white-light curve fit and the re-anchoring after the knobs** (DESIGN 4.3; `input_curves.h`:
  the anchors `input_curve_reference`, the shape test `curve_shape_ok`, the per-cell model, the `E`
  solve, the moments, the never-greener guard, the Levenberg-Marquardt solve, the re-meter;
  `multi_target_optimizer.cc`: the fit's target clamp, `fallback_picture_cells`, `curve_frame_of`,
  `reanchor_input_curves`, `shape_scale_of`).

The refusals of the code that this document leaves out (each is a check on the input before any
formula here runs):

- a `--roi` outside the image is refused, and one past its right or bottom edge is clamped to it; so is a
  RAW's inset crop;
- a RAW's inset crop, placed in the image after the sensor's margins, counts as a crop unless it is
  absent (LibRaw's `0xffff` corner or a zero size) or is the whole image at the origin;
- the analysis needs an image at least 2 pixels wide and high;
- `-E`, `--gain_g` and `--gain_b`, when given, must be finite and above 0;
- recorded input curves are refused unless they keep the shape the fit holds them to
  (`curve_shape_ok`: a hold point at most 0, monotone, within 2.2 stops of the bracket's curve), a
  test cut with the fit.
