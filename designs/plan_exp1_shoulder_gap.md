# Exp 1 shoulder: closing the gap between what the solver scores and what is written

Conventions of both repositories apply (AGENTS.md in each): shared pipeline stages are written once in
`color_pipeline.h` and compiled by g++ and nvcc; CPU and CUDA must agree; baked hashes that move name the commit and
the reason; no absolute machine paths in tests.

Status: **proposed**. Line numbers refer to `exp-fit3-dinov3-hybrid` after merging `b2e160e`.

## 1. What exp1 is and what is wrong with it

Branch `exp-fit3-dinov3-hybrid`, commit `b7aa105`, has two parts and a design doc.

**Part 1A: the shoulder.** `pcs_to_linear_srgb_unclamped` (`neg_pipeline.h`) decodes PCS XYZ to linear sRGB with the
upper clamp removed. `apply_fit3_highlight_shoulder_rgb` (`profile_bundle.cc`) applies a per-channel power-8 curve
from linear 0.42 onward to a ceiling of encoded 0.988, through a 6144-entry LUT. It is called in exactly two places:
the sRGB export loops of `neg_process.cc` (the CFA path and the libraw path), for `ENCODE_SRGB` and
`ENCODE_SRGB_LINEAR`.

**Part 1B: the bracket rescue.** `multi_target_optimizer.cc`, after Step 5: when the chosen bracket is not feasible
and more than 2 to 11% of the render sits above sRGB 242 or 235, the pool is re-screened with an asymmetric
enclosure penalty (`dinov3_auto_solver.cc`, `compute_penalties`) and a wider g, b grid, and the result is taken if it
clips no more and keeps the median L* within 4.5 to 7.5. Steps 1 to 4 are unchanged: the winner selection still uses
the plain residual (`a.residual`), although the comment above the penalty lambda suggests otherwise.

**The gap.** Everything that decides the parameters, and everything the station renders, still clamps at 1.0.

| Consumer | Code | Shouldered in exp1? |
|---|---|---|
| Preview DINOv3 predicts the intent field from | `SearchFrame::render_preview` (`multi_target_optimizer.cc`) through `cpu_render` and `convert_pixel` (`color_pipeline.h`) | no |
| `intent_loss`, the coordinate descent's objective | `dinov3_auto_solver.cc`, `intent_loss` | no |
| `highlight_clip`, `lab_stats`, Step 5 and Part 1B gates | `cfa_pipeline.cc`, `cpu_intent_loss` to `cpu_lab_stats` | no |
| single-target solver `solve_dinov3_intent` | `dinov3_auto_solver.cc` | no |
| station CPU and CUDA kernels, station export and UI previews | `negicc-station/src/pipeline/kernels_cuda.cu`, `converter.cpp` | no |
| `neg_process` sRGB export | `neg_process.cc`, both export loops | yes |

So the fixed point converges on an image that is not the one written out: the solver places the highlights for a
clipped output and the export then darkens them by 10 to 16 levels without the loss knowing. The station's export
will differ from `neg_process`'s as soon as the submodule is bumped, and `tests/test_neg_process_parity.py` will say
so.

Two more facts about the curve itself, both relevant to the replacement:

* **The profile's headroom.** The station builds each bracket's profile so that, on the grey axis, PCS Y = H below
  the chart white (H = exposure relative to the chart white, Y = 1 at the chart white) and
  Y = 1 + (H-1)/(3(H-1)+2) above it (`film_profile.py` `highlight_rolloff_curve`, white_u 0.75, rolloff 0.5, and the
  neutral ramp rows). Everything above the chart white is squeezed into Y in (1, 1.333). Today `pcs_to_linear_srgb`
  clamps at 1, so the chart white and everything brighter land on 255. That is the ceiling exp1 set out to fix, and
  it is right that the headroom is there.
* **exp1's curve reads the headroom in the wrong variable.** Its Michaelis-Menten inverse puts the roll-off's white
  at linear sRGB 0.75, but after the cLUT the white is at 1.0 (measured on the grey axis). The whole infinite range
  above the chart white gets 8 code values and the range from the knee to the white loses 12: a contrast reduction
  from sRGB 173 up on every frame, which is why the branch's own regression table shows Y90 failing by 15 levels on
  all 55 TriLED frames, not only on the 9 overblown ones. The "exact inverse and non-destructive on calibrated
  frames" claim in the design doc does not hold.

Also: the station's own FIT3 reverse engineering (`negicc-station/designs/epson_fit3_specification.md`, the negative
highlight shoulder, `hlCurveType == 2`) says the shoulder is quadratic in linear-1.8 space from `hlOut` = 200 to 255,
and that the 9th power is the shadow toe. exp1's doc calls the shoulder power-8 and cites a `MakeLUT_Highlight2` the
spec does not list. `fit3.cc` is the author's local file, outside both repos.

## 2. The loss colour space, for the record

The solver's Lab goes: PCS XYZ, Bradford D50 to D65, XYZ to linear sRGB, clamp to [0,1], Lab with D65 white and
OpenCV's constants (`linear_srgb_to_lab`). The "direct" step (station commit `752c8bb`) only removed an sRGB encode
followed by its exact inverse; measured against the full round trip over 225k points the difference is at most
0.00006. This matches how the intent field's targets were made (`cv2.COLOR_RGB2LAB` on sRGB images,
`negicc-station/training/extract.py`). Do not change it, and do not compute Lab from PCS XYZ directly (D50 white, no
gamut clamp, up to dE 5 in gamut, unbounded above white). What changes is only that the Lab must be taken **after
the shoulder**.

## 3. Target design

One shoulder stage in the shared pipeline, with one per-frame parameter, applied identically by every consumer,
identity when the frame has nothing above the chart white, bit-identical to today's output when off.

Definitions:

* **pre-shoulder value:** linear sRGB after Bradford and the XYZ to sRGB matrix, clamped at 0 only. Chart white is
  1.0; values up to about 1.33 on the grey axis and 1.5 off axis are real.
* **x_hi:** the parameter: the frame's 99.5% point of the pre-shoulder max channel over the inner crop at the current
  (bracket, E, g, b), clamped to [1.0, 1.5]. 1.0 means off.
* **post-shoulder value:** `shoulder(x, x_hi)` clamped to [0,1], then encoded as today.

### 3.1 The per-pixel pipeline

Stage order for one pixel, every consumer:

1. camera RGB u16, load to [0,1]
2. matrix: crosstalk x base scale x diag(E, E g, E b)
3. soft knee
4. profile: in_trc, cLUT, out_TRC, PCS XYZ D50
5. Bradford D50 to D65, XYZ to linear sRGB, clamp at 0 only. The result is the pre-shoulder value, 0 to about
   1.33. Two things read it here and nowhere else: the clip statistics against the chart white (`pen_clip`,
   `highlight_clip`, the `lab_stats` clipped counts, threshold 0.9953) and the max-channel histogram that gives x_hi.
6. `shoulder(x_hi)` per channel, identity where x_hi = 1
7. clamp to [0,1], sRGB encode
8. either quantise to u8 / u16 (preview, export) or Lab D65 with OpenCV's constants (intent loss against mu, sigma;
   `lab_stats`; the L* histogram of `highlight_clip`)

### 3.2 Computing x_hi inside the fixed-point loop

One iteration of `fixed_point` for one bracket:

1. Incoming params (E, g, b) and x_hi from the previous iteration (1.0 at the start).
2. Render the inner crop pre-shoulder at (E, g, b) and take the histogram of max(R, G, B).
3. x_hi = clamp(P99.5, 1.0, 1.5). Cross-check only, printed in verbose mode: log2(E / e_max) from the bracket
   enclosure is the same quantity for green.
4. Preview with `shoulder(x_hi)`, 8-bit sRGB; DINOv3 predicts mu, sigma per patch from it.
5. E, g, b coordinate descent: Lab post-shoulder against mu, sigma; `pen_clip` pre-shoulder; x_hi fixed.
6. Converged (dp <= TOL_PARAM or rel <= TOL_REL_ERR)? No: back to 1 with the new params. Yes: freeze (E, g, b, x_hi)
   in `FixedPointResult`.

After the loop: Step 5 headroom and Part 1B rescue as today (the rescue reruns the loop on another bracket); the
sidecar records E, g, b, knee and shoulder x_hi, and `cli_equivalent` carries `--shoulder`; the `neg_process`
full-resolution export and the station's export and previews apply the stored x_hi.

One x_hi per iteration, from the render at the iteration's incoming params, held constant through the E, g, b sweeps
so the loss stays smooth in E. Frozen when the loop converges. Never recomputed from the full-resolution image at
export.

### 3.3 The curve

Per channel, in stops t = log2(x). Constants: ceiling C = 0.988 in encoded sRGB (about 252); exponent n = 2 by
default (see the FIT3 note above; n = 8 is a variant to evaluate, never a per-frame parameter).

1. If x_hi <= 1.0001: y = min(x, 1). Today's behaviour, bit for bit.
2. Otherwise t_hi = log2(x_hi). Solve for the knee t_k < 0 such that the power curve joins the identity with a
   continuous first derivative: `t_hi - t_k = n (C - s(t_k)) / m(t_k)`, with `s(t) = srgb_code(2^t)` and
   `m(t) = ds/dt = srgb_encode'(2^t) 2^t ln 2`. Bisection on t_k in [-3, 0]; the difference changes sign once.
   * `t <= t_k`: y = x.
   * `t_k < t < t_hi`: `s = C - (C - s_k) (1 - (t - t_k) / (t_hi - t_k))^n`, `y = srgb_decode(s)`.
   * `t >= t_hi`: `y = srgb_decode(C)`.

Measured values (double precision, bisection as above):

| x_hi | n | knee x_k | code at knee | chart white (x = 1) lands at code |
|---|---|---|---|---|
| 1.01 | 2 | 0.938 | 248 | 251.9 |
| 1.1 | 2 | 0.866 | 239 | 249.9 |
| 1.333 | 2 | 0.735 | 223 | 245.1 |
| 1.5 | 2 | 0.671 | 214 | 242.2 |
| 1.333 | 8 | 0.931 | 247 | 251.1 |

So with n = 2 and x_hi = 1.333 the knee is near x = 0.735 (sRGB 223) and the chart white lands at 245, compressed but
not clipped. Everything below the knee, which includes the grey axis the metering uses (GreyPoint,
`exposure_for_lightness`, the e_center bisection), is untouched. With n = 8 the C1 join forces the knee **higher**
(x about 0.93), a harder and later shoulder; that is a reason to start with n = 2. The step from "off" (white at
255) to the smallest x_hi (white at about 252) is a discontinuity in x_hi; it is accepted, and the x_hi <= 1.0001
gate is what keeps frames without headroom content exactly as they are.

Evaluate it analytically on both backends (a few flops above the knee, nothing below). No LUT: a LUT built on the
host and on the device is one more thing to keep in parity. Precompute t_k and s_k once per `PipelineConfig`, not per
pixel.

## 4. Changes by layer

### 4.1 shared pipeline (`negicc/color_pipeline.h`)

* `PipelineConfig` gains `float shoulder_x_hi = 1.0f; float shoulder_t_k; float shoulder_s_k;` (and the derived
  constants the per-pixel code needs) and a host function `set_shoulder(PipelineConfig&, float x_hi)` that solves the
  knee.
* Keep `pcs_to_linear_srgb` for callers that need today's semantics; add `pcs_to_linear_srgb_pre_shoulder`, clamped
  at 0 only (exp1's `pcs_to_linear_srgb_unclamped`, moved here).
* `convert_pixel_linear` returns the pre-shoulder value. New `apply_shoulder(const PipelineConfig&, Rgb)` and
  `convert_pixel_display(cfg, m, px) = clamp01(apply_shoulder(convert_pixel_linear(...)))`. `convert_pixel` encodes
  `convert_pixel_display`.
* Clip rules unchanged and applied to the pre-shoulder value. `linear_srgb_to_lab` always runs on the **post-shoulder**
  clamped value (`convert_pixel_display`).
* The shoulder applies to `ENCODE_SRGB` only; with any other encoding `apply_shoulder` is the identity.
* Update the header comment's stage list and the stage table in `negicc-station/src/pipeline/README.md`.

### 4.2 negicc kernels and solver

* `cfa_pipeline.cc` `cpu_render`, `cpu_intent_loss`, `cpu_highlight_clip`, `cpu_lab_stats`: clip statistics from
  `convert_pixel_linear`; render and Lab from `convert_pixel_display`. `HighlightClipArgs` gains an optional 4096-bin
  histogram of the pre-shoulder max channel over [0, 2] (`uint64_t* max_hist`), filled when non-null. This is where
  x_hi comes from.
* `neg_pipeline.h` is a second copy of the pipeline, used by `neg_process.cc`, `dinov3_auto_solver.cc`
  (`intent_loss`, `render_crop_preview`) and GreyPoint (`multi_target_optimizer.cc`). Do not add a second shoulder
  there: make `convert_pixel_srgb` and the solver's `intent_loss` call the `color_pipeline.h` functions with a
  `PipelineConfig`. One implementation.
* `ConversionParams` (`multi_target_optimizer.h`) gains `float shoulder = 1.0f`. `SolverConfig` gains
  `float shoulder = 1.0f; bool shoulder_auto = false;`.
* `SearchFrame::fixed_point`: at the top of each iteration, when `shoulder_auto`, run `highlight_clip` on the inner
  crop at the incoming params with `max_hist`, take the 99.5% point, clamp to [1.0, 1.5], set `curr.shoulder`, and
  pass it through `search()` into `SolverConfig` so `render_preview`, `predict` and the coordinate descent use it.
  Hold it constant inside the iteration. The convergence test is unchanged. The screening pass (`search` with
  `initial == nullptr`) runs with the preview value or 1.0.
* Single-target path `solve_dinov3_intent`: the same, with the histogram taken from the preview render.
  `render_crop_preview` already builds a max-channel histogram of the *camera-linear* value for the dynamic knee; add
  the pre-shoulder one next to it. The 32x21 grid of patch means underestimates the 99.5% point; do not use it for
  x_hi.
* Step 5 and Part 1B gates: `highlight_clip` compares the pre-shoulder value against `ceil_lin`, so their meaning is
  unchanged; `median_L` is post-shoulder (say so in the comment).
* Verbose line in the style of the dynamic knee:
  `Shoulder: q995=1.212, x_hi = 1.212, knee=0.781, enclosure log2(E/e_max)=+0.27`. The enclosure number (`e_max`
  from `bracket_enclosure`) is the cross-check; print it, do not use it for the value.
* Guard, same pattern as `evaluate_dynamic_knee`: after convergence, re-score the final params with x_hi and with
  1.0 against the selected field; if the objective with the shoulder is worse by more than 2%, keep x_hi but set
  `shoulder_guard = true` in the result. Do not revert silently; the evaluation needs to see how often the intent
  loss disagrees.

### 4.3 neg_process, clip and sidecar

* `--shoulder off|auto|<x_hi in [1,1.5]>`, default `off` in this round, so every existing golden and parity record
  stays valid; `auto` is opt-in until the ledger says otherwise. `auto` without the parameter search is an error,
  like `--knee auto`.
* `ENCODE_SRGB_LINEAR` and the ICC output paths never apply the shoulder; a linear export is scene referred. Say so
  in the help text.
* Sidecar `tone` block adds `shoulder` (x_hi), `shoulder_source` (`off|cli|auto`), `shoulder_q995` (measured value
  before clamping, or null), `shoulder_knee` (x_k or null), `shoulder_guard` (bool or null). `cli_equivalent` appends
  `--shoulder <x_hi %.9g>` so a replay reproduces the export exactly.
* Export applies the stored x_hi, both export loops.
* Remove exp1's `apply_fit3_highlight_shoulder_rgb`, its LUT, the two export-loop edits and
  `pcs_to_linear_srgb_unclamped` from `neg_pipeline.h`. Keep Part 1B. `git merge-tree` reports no conflicts between
  `b7aa105` and `b2e160e`.

### 4.4 negicc-station

* Bump the submodule. `ConverterOptions` gains `float shoulder = 1.0f`; `Converter::config_for` calls
  `set_shoulder`. The gains are per candidate and the shoulder is per frame, so it lives in the options; the
  fixed-point loop in `per_target_conversion.py` rebuilds the options per iteration as it does for the knee.
* CUDA kernels pick up the shared functions; add the `max_hist` output to `gpu_highlight_clip` with a shared-memory
  histogram like `lab_stats_kernel`.
* Bindings and Python: `make_converter(..., shoulder=1.0)` in `src/pipeline_api.py`; the fixed-point loop and the
  export in `src/per_target_conversion.py`; the sidecar reader for replay; the UI preview path, which must use the
  same converter options as the export.

## 5. Tests

1. Bit identity when off: `make test` produces the same TIFF hashes as `b2e160e`. Add a hash check for one sRGB and
   one sRGB-g10 export.
2. Self-consistency: with shoulder 1.25 the Lab the solver scores for the final params equals `srgb_to_lab` of the
   exported preview at the grid points within float noise (compare `render_preview` through the same grid
   resampling, since the grid is patch means). This is the test that the gap is closed.
3. CPU vs CUDA parity (station): `intent_loss`, `highlight_clip` including `max_hist`, `lab_stats` and render with
   shoulder = 1.3 under the existing parity rules; a new nd0 baseline next to the current ones, not replacing them.
4. Station vs `neg_process` parity: `tests/test_neg_process_parity.py` gets a case with `--shoulder 1.25` in
   `cli_equivalent` replayed through the station export; existing tolerances.
5. Sidecar: `tone.shoulder*` fields present, `cli_equivalent` replay reproduces the TIFF. `--shoulder auto` with
   an explicit-parameter (no search) run errors out.
6. Unit tests in negicc-station. Keep unit tests of negicc in negicc.
