# Formulas of the scene conversion

Every formula of the scene conversion that the Lean specification covers, in ordinary notation, with
the C++ function it mirrors and its Lean name. The sections follow `DESIGN.md`: reading the scan and
the film base (2), a bracket's range (3.3), the per-pixel conversion and the outputs (4.1), the
white-light curve fit (4.3) and the tone stage (4.4). The Auto search (4.2) and the networks (3.5)
are specified in #211, stacked on this pull request. The coverage table at the end says for each
function whether properties are proved and whether it is checked against the program; the lists
after it name the definitions the check does not evaluate and what is not formalized, with reasons.

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

**Three captures** (`np_input.cc`: `load_rgb_input`; Lean `captureScale`, `tiffHalf`). With `e_k` the
shutter × ISO of capture `k`:

```
v_k' = min(65535, round(v_k e_0 / e_k))       (k = 1, 2; unchanged when e_0 = e_k)
half = ⌊(a + b + c + d + 2) / 4⌋              (a TIFF's 2×2 block)
```

**White-light film-base balance** (`white_light_balance.h`: `film_base_balance_gains`; Lean
`filmBaseBalanceGains`, `rec2020Rgb`, `Mat3.adjugate`, `Mat3.det`). `rec2020_rgb` is
`M_2020⁻¹ M_709`, each `M` the RGB to XYZ matrix of its primaries with Lindbloom's D65; LibRaw's six
decimals are within `5.3e-7` of it (`rec2020Rgb_literal`). With `C = rec2020_rgb · rgb_cam`:

```
K_c = (adj(C) f_profile)_c / (adj(C) f)_c,   then K_c ← K_c / max(K)
```

refused (gains unchanged) when `(adj(C) f)_c / det C` or `(adj(C) f_profile)_c / det C` is not
positive for some `c`.

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
exposure(capture)`.

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

**Exposure normalisation** (`multi_target_optimizer.cc`: `frame_scale`; Lean `exposureNorm`,
`Mat3.divScalar`): `e_norm = M_frame[1][1] / M_ref[1][1]`, `M_ref` the bundle's base at 1/8 s,
ISO 100. `candidateMatrix(M / k, E, g, b) = candidateMatrix(M, E / k, g, b)`
(`candidateMatrix_divScalar`).

## 3.3 A bracket's calibrated range

(`multi_target_optimizer.cc`: `compute_inner_crop`, `u16_percentile_linear`, `compute_frame_stats`,
`bracket_enclosure`, `pull_centre`; `profile_bundle.cc`: `load_profile_bundle`; Lean `searchCropRatio`,
`innerCrop`, `u16PercentileLinear`, `greenHistogram`, `sceneGreenRange`, `patchGreenRange`,
`bracketExposure`, `bracketEnclosure`, `pullCentre`.)

```
inner crop: x0 = min(round(W (1 − r) / 2), ⌊(W − 1)/2⌋), width W − 2 x0 (likewise y);  r = 0.90
g_min, g_max = smallest, largest green value of the bracket's patches
exposure(bracket) = max(10⁻⁶, shutter · ISO / 100)
pos = q (n − 1);  v_lo, v_hi = first values whose cumulative count exceeds ⌊pos⌋, ⌈pos⌉
percentile = (1 − (pos − ⌊pos⌋)) v_lo + (pos − ⌊pos⌋) v_hi
g_lo = max(1, P(0.005)),  g_hi = max(g_lo + 1, P(0.995))        (green, inner crop)
ρ_b = (0.125 · 1) / exposure(bracket)
e_min = g_min / g_lo · ρ_b,   e_max = g_max / g_hi · ρ_b,   e_centre = sqrt(e_min e_max)
u_min = e_min · 2^−1.5,  u_max = e_max · 2^1.2,  search range [min(u_min, u_max), max(u_min, u_max)]
feasible: e_min ≤ e_max;  admissible: log2(g_hi / g_lo) ≤ log2(g_max / g_min) + 2.7;  inverted: u_min > u_max
slack = log2(e_max / e_min),   fill = (g_hi / g_lo) / (g_max / g_min)
pull centre = 0 when inverted, else e_centre
```

On a white-light bundle the search multiplies `g_lo`, `g_hi` by `e_norm` before computing the range;
on TriLED it works in the reference unit.

## 3.4 The profiles' prediction errors

The conversion does not use them: the bundle keeps each bracket's `profcheck` report
(`profcheck_output`), and nothing reads it.

## 4.1 The per-pixel conversion

**Profile** (`apply_trc`, `interpolate_clut`, `apply_profile`; Lean `tableEval`, `tableAt`,
`clutEval`, `tetraBlend`, `applyProfile`, `Profile.hasClut`, `pcsScale`).

```
table(v) = lerp(T[k], T[k+1], s − k),   s = v (n − 1),   k = clamp(⌊s⌋, 0, n − 2)   (n ≥ 2;
           a table of one sample is that sample, an empty one the identity)
M = I, o = 0: the parser refuses a profile whose AToB0 has another matrix (Profile.matrixAccepted)
t = clamp01(M in(x) + o)
clut: s = t (dim − 1), (i, j, k) the lower corner held inside the grid, the upper one from it,
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
C   = [rXYZ gXYZ bXYZ] of the embedded sRGB profile, s15Fixed16 (elleSrgbColorants)
P   = C⁻¹ correctly rounded to nine significant digits   (pcsToLinearSrgbMatrix_literal)
w   = s15Fixed16(0.9642, 1, 0.8249) = (63190, 65536, 54061) / 65536   (pcsWhite_eq)
```

`C⁻¹ w = (1, 1, 1)` exactly (`pcsToLinearSrgbMatrix_white`) and `P w = (1, 1, 1)` within `1e-8`
(`pcsToLinearSrgb_white`). `C` is BT.709 adapted by Bradford from D65 `(0.3127, 0.3290)` to the PCS
illuminant within 1.03 units of `1/65536` (`elleSrgbColorants_bradford`). The conversion computes
with `P` and with `1.99996948`, which is `1 + 32767/32768` within `2.5e-9` (`pcsScale_literal`).

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

The XYZ rows are OpenCV's matrix to six decimals, the BT.709 primaries with OpenCV's D65
`(0.950456, 1, 1.088754)` correctly rounded (`srgbToXyzOpenCV_literal`); the rows sum to that white
exactly. The encoding's break `0.0031308` is IEC 61966-2-1's, `0.04045 / 12.92` within `5e-9`
(`srgb_break_literal`).

## 4.3 The white-light curve fit

**The curve** (`input_curves.h`: `curve_stops`, `input_curve`, `compose_input_table`,
`CombinedProfile::build`, `input_curve_reference`, `curve_shape_ok`; Lean `curveStops`, `inputCurve`,
`composeInputTable`, `combinedProfile`, `inputCurveReference`, `cellMedian`, `cellMinimum`, `curveShapeOk`):

```
u = min(log2(max(v, 1e-12) / ref), 0)
f(u) = u (1 + a + k tanh(u − u_med))                       held: f(u_lo) + (u − u_lo) below u_lo
v' = ref 2^f(u) for v < ref (and a curve that is not the identity), v otherwise
table_i = in(clamp01(curve(i / (n − 1))))
ref = knee(E g (M_frame f)_G / 65535),  u_i = min(log2(max(knee(cell_i), 1e-12)/max(ref, 1e-12)), 0)
u_med = median_i u_i,  u_lo = min_i u_i
shape: f strictly decreasing from ref down over the table, |f(u) − u| ≤ 2.2, and at 1e-12 below the last
```

**The per-cell model** (`JointCurveModel`: `anchored`, `cell_lab`, `median_L`, `solve_e`; Lean
`CurveFrame`, `anchored`, `cellLab`, `cellLabs`, `upperMedian`, `medianL`, `solveE`): each cell's rows
`r = M_frame cell / 65535` are scaled by `(E, E g, E b)`, the knee, the curves on green and blue,
the profile, the display value, CIELAB. `E` with the median `L*` at the target: secant steps on
`log E` inside a bracket, then bisection, to `2e-5 L*`.

**The anchors** (`joint_moments`, `fit_input_curves_joint`; Lean `jointMoments`, `shrinkWeight`,
`FitProblem.setup`):

```
B = cells with 2 < L* < 92 at Auto;   p_i² from the neutral head;   m = Σ_B p_i²
ω_i = (p_i² + 10⁻³ / |B|) / (m + 10⁻³)
mean, var_ω(L*), k_a = 50 cov_ω(L*, a*) / (var + 25), k_b likewise
w = m / (m + 12),   s = var / (var + 36),   target k = (1 − w s) k_Auto
σ_anc = sqrt((144 + m) / (m + 12))
```

**The intent term** (Lean `FitProblem.setup`, `FitProblem.r4`):

```
ν_i = (1/max(σ_a,i, 10⁻³)²) / Σ_j (1/max(σ_a,j, 10⁻³)²)
L_f = Σ ν_i L*_i,   d_i = μ_a,i − a*_i (at Auto),   d̄ = Σ ν_i d_i
z_i = (L*_i − L_f)/50,   k_d = Σ ν z (d − d̄) / Σ ν z²
SE = sqrt(Σ_blocks (Σ_{i in block} ν_i ((d_i − d̄) − k_d z_i))²)        (4×4 blocks of cells)
dL = (L_f − L_anchors)/50 · (var + 25) / max(var, 25)
r4 = (1/1.5) (Σ ν_i (a*_i − μ_a,i) − (k_a − target k_a) dL − SE) / sqrt(SE² + 1)
```

**The residuals** (Lean `FitProblem.residuals`, `FitProblem.loss`):

```
r0 = (1/1.5)(ā − (1 − w) ā_Auto) / σ_anc       r1 = (1/1.5)(b̄ − (1 − w) b̄_Auto)
r2 = ½ sqrt(var/(var + 100)) (k_a − target k_a)   r3 likewise for b
r4 (above)     r5, r6 the guard's hinges (guarded pass)     r7, r8 = sqrt(1.5) a_G, sqrt(1.5) a_B
loss = r0² + … + r4²
```

**The guard** (`green_guard_cell_a`, `JointFitGuard`; Lean `greenGuardCellA`, `guardOf`, `guardMeans`,
`guardHinge`, `guardBinds`):

```
d = a − a_Auto;  counted a = a if d ≥ 0, else a_Auto + (1 − w_s) d
w_s = 1 − (3x² − 2x³),  x = clamp((L_Auto − 15)/10 + 0.5, 0, 1)
means over the guard's cells (2 < L* < 92 at Auto), plain and p²-weighted; they count with ≥ 8 cells
and Σ p² ≥ 2;  hinge = 4 max(0, ref − 0.5 − mean);  binds: ref − 0.5 − mean > 0
```

**The solve** (`JointLM`, `fit_input_curves_joint`; Lean `FitProblem.solvePoint`, `FitProblem.run`,
`FitProblem.cutBack`, `FitProblem.fit`, `solve4`): parameters `(g, b, a_G, a_B)` within
`g_0 2^±0.75`, `b_0 2^±1.25` (at least `10⁻³`) and `|a| ≤ 0.35`; forward differences of `±10⁻³`;
`(JᵀJ + λ diag(max(JᵀJ_aa, 10⁻⁶))) δ = −Jᵀr`, a parameter at a bound that the gradient pushes outward
held; steps at shares `1, 0.5, 0.25, 0.1, 0.05` (the full one cut back to the shape's limit by ten
bisections); accepted when the cost falls (`λ/3`, at least `10⁻⁹`), else `4λ`; at most 25 iterations;
settled when the step is below `10⁻⁵` or the gain below `10⁻⁶ max(1, cost)`. The free run first; the
guarded run from the start when the free result binds.

**The fit's target** (`WhiteLightFitCore`: `fit_target_L_`, `kIntentMinL`, `kIntentMaxL`; Lean
`fitTargetL`, `intentMinL`, `intentMaxL`): Auto's grid median `L*` held to `[18, 85]`, or Auto's own
when `E` was given.

**Re-anchoring after the knobs** (`reanchor_input_curves`, `shape_scale_of`; Lean `reanchorCurves`,
`reanchorChannel`, `shapeScaleOf`). With `a0`, `a1` the anchors (`inputCurveReference`) at the curves'
own `(E, g, b)` and at the knobbed ones, on the frame's grid (its picture cells on a fallback crop):

```
ref' = ref · a1.ref / a0.ref (both positive),  u_med' = u_med + (a1.u_med − a0.u_med),  u_lo' likewise
s = 1 if the whole bend keeps the shape, else 24 bisections of [0, 1];   a' = s a,  k' = s k
```

**Re-meter and cells** (`grid_median_L`, `solve_e_for_median_L`, `fallback_picture_cells`,
`curve_frame_of`; Lean `gridMedianL`, `solveEForMedianL`, `fallbackPictureCells`, `curveFrameBase`):
40 bisections of `log E` on the combined profile over `[E_c/2, 2E_c]` within the bounds when the
target lies there; on a fallback crop the cells within 0.25 stop of a rebate base or 9 stops below
it are left out when at least `max(16, n/4)` remain.

## 4.4 The tone stage

(`tone_stage.cc`: `tone_curve_L`, `tone_table`, `tone_curve_ok`, `tone_descriptors`, `tone_targets`,
`tone_solve`, `tone_plan`, `tone_pick_clip_share`, `tone_clip_sample`, `tone_plan_for_image`;
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
       that does, else the one that clips least (sample: every 8th pixel each way, 16-bit outputs)
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
| `parse_film_profile_pipeline` (the matrix refused) | `Profile.matrixAccepted` | | yes (on the bundles' profiles) |
| `apply_matrix`, `candidate_matrix` | `Mat3.mulVec`, `candidateMatrix` | `candidateMatrix_mulVec`, `_one`, `_mul_exposure`, `_divScalar` | yes |
| `subtract_cfa_black_in_place` | `subtractBlack`, `whiteAfterBlack` | | read |
| `cpu_preview_debayer` | `halfSizePixel` | | yes |
| `reconstruct_green_soft_cpu`, `cpu_grain_aware_debayer` | `greenSoft`, `grainAwarePixel`, `grainAwarePixelU16` | | yes |
| `cpu_pixel_shift_merge` | `pixelShiftPixel` | | yes |
| `load_rgb_input` (scaling, TIFF half) | `captureScale`, `tiffHalf` | | read |
| `film_base_balance_gains` | `filmBaseBalanceGains` | | yes |
| `axis_table`, `flat_divide` | `flatAxis`, `flatGain`, `flatDivide` | | yes |
| `compute_station_merged_matrix` | `mergedMatrix`, `captureExposure` | `mergedMatrix_mulVec_base`, `_scan_base` | yes |
| `detect_film_base` (rules) | `rebatePlaced`, `rebateBox`, `patchClipped`, `outshoneStops`, `rebateAccepted`, `thinnestAccepted`, `filmBaseSource` | | yes |
| `percentile_f32`, `median_u16` | `percentileF32`, `medianU16` | | yes |
| film base capture scaling | `baseFromCapture` | | read |
| `knee_compress` | `kneeCompress`, `autoKnee` | `kneeCompress_mono`, `_step`, `_of_le`, `_of_nonpos`, `_le_one` | yes |
| `load_u16`, `convert_pixel_corrected` | `loadU16`, `correctedRgb` | | yes |
| `frame_scale` (`e_norm`) | `exposureNorm`, `Mat3.divScalar` | `candidateMatrix_divScalar` | yes |
| `compute_inner_crop` | `innerCrop` | | yes |
| `u16_percentile_linear`, `compute_frame_stats` (`g_lo`, `g_hi`) | `u16PercentileLinear`, `greenHistogram`, `sceneGreenRange` | | yes |
| `load_profile_bundle` (a bracket's green range and exposure) | `patchGreenRange`, `bracketExposure` | | yes |
| `ParamSearchOptions::crop_ratio` | `searchCropRatio` | | yes |
| `bracket_enclosure` | `bracketEnclosure` | | yes |
| `pull_centre` | `pullCentre` | | yes |
| `interpolate_clut` | `clutEval`, `tetraBlend` | `clutEval_const`, `clutEval_mem` | yes |
| `apply_profile` | `applyProfile`, `Profile.hasClut` | | yes |
| `kPcsWhite`, `pcs_to_linear_srgb_unclamped`, `_pre_shoulder` | `pcsWhite`, `pcsToLinearSrgbUnclamped`, `pcsToLinearSrgbPreShoulder`, `pcsToLinearSrgbMatrix` | `pcsWhite_eq`, `pcsToLinearSrgb_white`, `pcsToLinearSrgbMatrix_white`, `pcsToLinearSrgbMatrix_literal` | yes |
| `convert_pixel_pcs`, `convert_pixel_linear`, `convert_pixel_display` | `convertPixelPcs`, `convertPixelLinear`, `convertPixelDisplay` | `convertPixelDisplay_of_off` | yes |
| `set_shoulder` | `setShoulder`, `shoulderCeiling`, `shoulderKneeGap`, `bisect` | `setShoulder_of_le` | yes |
| `apply_shoulder_channel`, `apply_shoulder`, `to_display` | `applyShoulderChannel`, `applyShoulder`, `toDisplay` | identity below the knee, ceiling, parabola, `toDisplay_mem` | yes |
| `shoulder_x_hi_anchored` | `shoulderXHiAnchored` | | yes |
| `max_hist_bin`, `max_hist_quantile`, the histogram's fill, `shoulder_x_hi_from_hist`, `measure_rolloff_x_hi` | `maxHistBin`, `maxHistQuantile`, `maxHistOf`, `shoulderXHiFromHist` | | yes |
| `encode_output`, `convert_pixel` | `encodeOutputWith`, `encodeOutput`, `convertPixel` | `convertPixel_of_no_tone`, `_of_short_tone`, `convertPixel_linear`, `encodeOutput_mem` | yes |
| `quantize_u8`, `quantize_u16` and the outputs' codes | `quantizeU8`, `quantizeU16`, `outputPixel`, `correctedPixel`, `previewPixel` | | yes (exact) |
| `linear_channel_white_clipped`, `_black_clipped` | `linearChannelWhiteClipped`, `linearChannelBlackClipped`, `pixelWhiteClipped`, `pixelBlackClipped` | | yes |
| `curve_stops`, `input_curve` | `curveStops`, `inputCurve` | `inputCurve_of_ge`, `inputCurve_zero`, `curveStops_zero`, `curveStops_hold` | yes |
| `compose_input_table`, `CombinedProfile::build` | `composeInputTable`, `combinedProfile` | | yes |
| `input_curve_reference`, `JointCurveModel::anchored` | `inputCurveReference`, `anchored` | | yes |
| `curve_shape_ok` | `curveShapeOk` | | yes |
| `JointCurveModel::cell_lab`, `median_L`, `solve_e` | `cellLab`, `medianL`, `solveE` | | yes |
| `joint_moments` | `jointMoments` | | yes |
| `green_guard_cell_a`, `JointFitGuard` | `greenGuardCellA`, `guardOf`, `guardMeans`, `guardHinge`, `guardBinds` | `greenGuardCellA_ge`, `_of_ge`, `_shadow` | yes |
| `shrink_weight` | `shrinkWeight` | `shrinkWeight_mem` | yes |
| `fit_input_curves_joint` (setup, residuals, loss), `JointLM` | `FitProblem.setup`, `.r4`, `.residuals`, `.loss`, `.run`, `.cutBack`, `.fit` | | yes |
| `grid_median_L`, `solve_e_for_median_L` | `gridMedianL`, `solveEForMedianL` | | yes |
| `fallback_picture_cells` | `fallbackPictureCells` | | yes |
| `curve_frame_of` (base) | `curveFrameBase` | | yes |
| `fit_target_L_` (the clamp) | `fitTargetL`, `intentMinL`, `intentMaxL` | | the bounds |
| `shape_scale_of`, `anchor_curves` | `shapeScaleOf`, `inputCurveReference` | | yes |
| `reanchor_input_curves` | `reanchorCurves`, `reanchorChannel` | | read (its parts checked) |
| `tone_curve_L` | `toneCurveL` | `toneCurveL_mono`, `toneCurveL_identity` | yes (in table and plan) |
| `tone_table`, `tone_curve_ok` | `toneTable`, `toneCurveOk` | | yes |
| `tone_descriptors` | `toneDescriptors`, `percentileSorted`, `toneLstarOfU8` | | yes |
| `tone_targets` | `toneTargets` | `toneTargets_share_zero` | yes (through `tone_plan`) |
| `tone_solve`, `tone_plan` | `toneResiduals`, `toneCost`, `toneLM`, `solve4`, `toneSolve`, `tonePlan` | | yes |
| `tone_pick_clip_share`, `tone_clip_sample`, `tone_plan_for_image` | `tonePickClipShare`, `toneClipSample`, `tonePlanForImage` | | yes |
| `tone_lab_fprime`, `tone_table_Y`, `apply_tone` | `toneLabFprime`, `toneTableY`, `applyTone` | `applyTone_grey` | yes |

## Defined but not evaluated by the check

- `subtractBlack`, `whiteAfterBlack`, `captureScale`, `tiffHalf` (`np_input.cc`) and `baseFromCapture`
  (`np_film_base.cc`): those files are not in the library the generator links, and the functions
  work on LibRaw's state or on files.
- `reanchorCurves`, `reanchorChannel`: `reanchor_input_curves` needs the network engine; its parts
  (the anchors, the shape test, the shape's scale) are checked.
- `fitTargetL`: the clamp sits inside the fit's class; its bounds are checked.
- `density`, `transmission`, `sq`, `mean`, the componentwise `add` and `sub`: helpers of the base for
  the other parts (the print emulation's density); the scene conversion does not use them.

## Not formalized

- **The Auto search** (4.2) and the networks (3.5): specified in #211, stacked on this pull request.
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
  guards for absent data (no profile, a zero reference entry in `frame_scale`, a scan exposure of
  zero, a missing base shutter): the definitions assume finite values and present data.
- **The network heads** that give the tone stage its predictions, the fit its `p`, `μ` and `σ`: an
  opaque function whose outputs are inputs here (specified with the Auto search in #211).
