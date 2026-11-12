# Design: the white-light conversion

This document describes how `neg_process` converts a negative scanned under white light (`--light_source d65`) with
`-P srgb`, stage by stage, and lists every tunable number each stage carries. Its purpose is to make it possible to
see the whole algorithm at once and to check that no stage rests on arbitrary adjustments.

Tri-LED scans run stage 1 only; stages 2 to 5 are white-light only. `--fit_curves off` also stops after stage 1.

## The stages

| # | Stage | Takes | Decides | Hands on |
|---|---|---|---|---|
| 1 | Auto search | the half-size scan, the film stock's profile bundle (one profile per exposure bracket), the film-base reading | which bracket's profile to use, and the exposure E and green and blue gains g, b that bring the render closest to the intent network's prediction of the picture | the bracket, (E, g, b), the highlight shoulder, and the network's outputs for the chosen render: the intent field, the neutral-surface probability p of each grid cell, the backbone features |
| 2 | Curve fit | stage 1's result and p | a correction of the gains and of the tone slope of the green and blue input curves that removes the colour of the surfaces the network reads as neutral | fitted curves and (E, g, b) |
| 3 | Tint trim | the fitted render | one uniform magenta-green offset, chosen by scoring candidate offsets against the intent network | the offset |
| 4 | Profile tables | the bracket's profile, the fitted curves, the offset | nothing: composes them into the profile's input tables, so the pixel pipeline is unchanged | one ordinary profile |
| 5 | Tone stage | the finished sRGB image and the backbone features | one monotone lightness curve toward the shadow level, highlight level and tonal range the tone head predicts for the frame | the final image |

What each stage is for, in one sentence:

1. **Auto search** finds the exposure bracket and the overall balance.
2. **Curve fit** removes a cast that differs between shadows and highlights, which gains alone cannot.
3. **Tint trim** settles the remaining overall magenta-green balance.
4. **Profile tables** bakes 2 and 3 into the profile.
5. **Tone stage** sets contrast and the black and white levels.

## How the knobs are classified

- **Policy**: a choice about what the picture should look like, or how much to trust an estimate. These are the
  numbers that deserve scrutiny. Each has a stated reason and a measured effect.
- **Safety limit**: a bound that only acts when something has gone wrong (a runaway curve). It should be inactive
  on ordinary frames.
- **Numerical detail**: iteration counts, step sizes, tolerances, table sizes. Changing them within reason does not
  change the picture. They are kept inside the function that uses them and are not listed one by one.
- **Calibration or convention**: values fixed by the profile data, the colour space or a reference condition.

"Effect" below is the number of frames, out of a 140-frame test set, whose render changes by a mean ΔE above 1 when
the knob's rule is switched to its no-op (or the value halved or doubled, where it says so). It was measured on the
plain fit before the evidence hand-over was added, except for the hand-over's own row. It says how much work the rule
does, not whether the result is better. A dash means it was not measured.

Numerical details are not listed one by one, but the ones that could look like choices are named in a line under each
table.

## Stage 2: curve fit

The fit solves one least-squares problem. Every grid cell Auto renders between L\* 2 and 92 is an anchor, weighted by
p². Rendered through the profile's curves, the anchors' weighted mean a\* and b\*, and their trend from shadows to
highlights, are driven to neutral. The fit varies the green and blue gains and the slope term of the green and blue
input curves. E is re-solved at every step so that the median lightness stays where Auto put it. The solved correction
is then scaled by the evidence hand-over, and E is solved once more. There is no separate target, no lightness bands
and no confidence product. On a fallback crop (the whole image, no detected frame), cells at the film base's level or
far below it are left out first. After the fit, E is re-metered on the real tables, and the fit runs again (up to three
fits in all) if the highlight roll-off measured there differs from the one it ran with.

| Knob | Value | Kind | What it does | Effect |
|---|---|---|---|---|
| Evidence half-mass | 12 (sum of p²) | Policy | The fitted correction is applied at strength m / (m + 12), m the neutral evidence in the frame, so a correction resting on almost no neutral surface is not trusted. Not derived: the derivation from the noise model gives full strength at any evidence. | 24 (a second audit count gives 46) |
| Anchor range | L\* 2 to 92 at Auto | Policy | The cells that can be anchors; also the guard's cells. Clipped blacks and whites say nothing about the cast. | — |
| Slope span | 6 L\* | Policy | A tone slope measured over anchors that span little lightness is shrunk, sd² / (sd² + 6²), so a narrow range is not extrapolated across the whole curve. The principle is stated; the 6 is not derived. | 34 |
| Slope hold | on | Policy | Beyond the frame's densest cell the fitted curve stops bending and keeps the offset it has there, instead of extrapolating its bend to the end of the table. | 38 |
| Curve bound | 0.35 | Safety limit | Limit on each curve's slope term. | 4 when doubled, 32 when halved |
| Maximum deviation | 2.2 stops | Safety limit | The fitted curve may not leave the profile's own curve by more than this anywhere. | 0 when doubled, 21 when halved |
| Green gain bound | 0.75 stops | Safety limit | Range of the green gain around Auto's. | 0 when doubled, 2 when halved |
| Blue gain bound | 1.25 stops | Safety limit | Range of the blue gain around Auto's. | 2 when doubled, 16 when halved |
| Plausible lightness | L\* 18 to 85 | Safety limit | If Auto's grid median L\* is outside this range, the search failed (a nearly black or blown-out render, as when the frame holds rebate or holder), and the fit holds the nearest end instead. | — |
| Fallback-crop cells | within 0.25 stop of the film base, or more than 9 stops below it | Safety limit | On a fallback crop these cells are rebate, gaps, bare light or holder, not picture, and are left out (only while at least 16 cells and a quarter of the grid remain). | — |
| Minimum anchors | 8 cells in range | Safety limit | Fewer cells in the anchor range and the fit keeps Auto's colour. | — |
| Curve damping | 1.5 | Numerical detail with a visible effect | Penalty on the curves' slope terms in the solve; without it the gains and the slope terms are nearly interchangeable. | 2 at half, 3 at double |

Four policy knobs, seven safety limits.

Numerical details: the weights of the mean and trend residuals (1/1.5 and a weight that grows with the anchors'
lightness spread) only matter where a bound, the damping or the guard competes, because four residuals in four
unknowns are otherwise solved exactly. The gain floor (10⁻³), the E solve's tolerance, the iteration limits, at most
three fit passes and six re-meter passes, and four shape-check rounds also stay out of the table.

### The never-greener guard (inside stages 2 and 3)

The fit may move colour anywhere the neutral surfaces ask, except greener than Auto's render: a picture slightly too
magenta is preferred to one slightly too green. The fit runs freely first; only if its result is greener than Auto, on
the plain mean a\* or on the p²-weighted mean a\*, does it run again with the excess as a penalty.

| Knob | Value | Kind | What it does | Effect |
|---|---|---|---|---|
| The guard itself | on | Policy | As above. Without it the mid-tones go greener on 38 of the 140 frames. On emulated scans with known colour it costs accuracy, because Auto's render is itself magenta there; on real frames the guarded render was preferred. | 41 |
| Tolerance | 0.5 a\* | Policy | How much greener than Auto still counts as "not greener". | 7 |
| Shadow exemption | below L\* 15, 10 L\* wide | Policy | A shadow cell that ends greener than at Auto does not count against the fit, so a magenta cast in the shadows can be corrected. | 10 |
| Penalty weight | 4 | Numerical detail with a visible effect | Strength of the penalty in the second run. It has no unit and no derivation. | 11 at half or double |
| Counting minimums | 8 cells; evidence 2 (sum of p²) | Safety limit | A mean counts only with this many cells, or this much evidence for the weighted one. | 0 (never triggers) |

Three policy knobs, one safety limit, and the weight, which is the least principled.

## Stage 3: tint trim

A ladder of uniform magenta-green offsets is scored against the intent network's field; the best few are rendered and
scored again on their own fields; the winner is applied. Offsets that would make the picture greener than Auto are
not candidates. The stage's own guard check counts Auto's cells above L\* 2 up to and including 92, with the fit's
tolerance plus the slack below and no shadow exemption.

| Knob | Value | Kind | What it does | Effect |
|---|---|---|---|---|
| The stage itself | on | Policy | A uniform offset of this kind is also something a user can set afterwards. | 41 |
| Ladder | −8 to +8 a\*, step 0.5 | Policy | The offsets considered. No frame reaches the ends. | — |
| Candidates re-rendered | 3 | Policy | How many of the best offsets get their own network inference. | 0 above ΔE 1; 19 above 0.5 |
| Read-out | +1 standard error toward magenta | Policy | The pick is moved toward magenta by its own uncertainty. A systematic lean of about 0.25 a\*. | 0 above ΔE 1; 10 above 0.5 |
| Cap tolerance | 1 a\* | Policy | Where the frame has confident neutral surfaces, the offset may not push them more than this from neutral. | 18 (with the next row) |
| Cap evidence | 100 (sum of p²) | Policy | Evidence at which the neutral surfaces count as fully confident for the cap. Halving or doubling changes 3 to 4 frames. | — |
| Guard slack | 0.1 a\* | Policy | Extra tolerance of the guard inside this stage. | 0 |
| Density change bounds | offset ±0.3 D, contrast ±0.6 | Safety limit | The limits on the green and blue density offset and contrast change one offset may ask for. | — |

Seven policy knobs, one safety limit. The read-out, the candidate count and the guard slack are small uniform effects
and are candidates for removal; the cap evidence is close to a plain literal. With the evidence hand-over in place,
removing the read-out and the candidate count together moves 22 of the 140 frames by a mean ΔE above 0.5, and 6
above 1 (at most 1.8).

Numerical details: the cells the offset's direction is measured on (L\* 5 to 95), the direction solve's probe steps
and ridge, and the bootstrap (300 resamples of 4 × 4-cell blocks) that gives the read-out's standard error. The cap
does not act at a confidence of 0.5 or less; that is part of its tolerance formula, 1 / ((c − 0.5) / 0.5) a\*.

## Stage 5: tone stage

Tone heads on the backbone features predict, for the frame, the shadow level L2, the highlight level L98 and the tonal
range L90 − L10 a lab scan would have. One monotone L\* curve moves the image toward them while holding the mid-tone
and the mean chroma.

| Knob | Value | Kind | What it does | Effect |
|---|---|---|---|---|
| Shadow move | 1.0 | Policy | The image moves all the way to the predicted shadow level. | 60 at half |
| Highlight move | 0.5 | Policy | Half way to the predicted highlight level. | 16 to 30 |
| Range move | 0.5 | Policy | Half way to the predicted tonal range. | 38 to 61 |
| Clip allowance | 0.5 % of pixels | Policy | If the curve would clip more white than this, the highlight and range moves are scaled back through 0.75, 0.5, 0.25, 0. | 14 |
| Curve bounds | black −15 to 15, white 80 to 110, gamma 0.7 to 1.45, contrast 0.7 to 1.5 | Safety limit | The range of the lightness curve's four parameters. | — |
| Top slope and white starts | 0.07; 100, 110, 85 | Numerical detail with a visible effect | A floor on the curve's slope at the top, and three starting points for the solve. | 3 |
| Prior | 0.02 | Numerical detail | Regularises the solve. | 0 |
| Chroma limits | 0.5 to 1.6 | Safety limit | Range of the chroma factor; it stays between 1.00 and 1.03 on every test frame. | 0 |
| Neutral chroma ramp | C\* 3 to 8 | Policy | Colours below C\* 3 keep their a\* and b\* as they are, those above 8 take the whole chroma factor, with a smooth step between, so near-neutral colours keep their balance. | — |

Five policy knobs, two safety limits. The three moves are the stage's real policy: the full shadow move and the half
moves were chosen by blind comparison, not derived.

## Stage 1: Auto search

Stage 1 is unchanged by the white-light stages and is shared with tri-LED scans. Its knobs:

| Knob | Value | Kind | What it does |
|---|---|---|---|
| Admissible brackets | at least 0.75 of the largest tonal capacity | Policy | Which brackets may be chosen at all. |
| Slack penalty | 0.5 per stop | Policy | The cost of a scene that overflows a bracket's calibrated range. |
| Enclosure tolerances | 1.5 stops of highlight, 1.2 stops of shadow; window 0.6 to 0.85 stop, 0.6 stop per stop of overflow | Policy | How far a scene may overflow a bracket's calibration, and how wide E is searched when it does. |
| Trust region | ±0.5 stop around the screened E | Policy | The fixed-point loop refines E rather than moving it far from where screening put it. |
| Clip fallback | clip share 3·10⁻⁴, ratio 0.5, lightness window −4.5 to +0.5, residual tolerance 0.10 | Policy | When the chosen bracket clips highlights, whether to move to the next denser one. |
| Highlight pullback | median L\* ≥ 58 with ≥ 2.2 % above sRGB 242, or median L\* ≥ 48 with ≥ 7.5 % above 242 (≥ 11 % above 235 on the fourth bracket and later); E raised by at most 0.15 stop; a re-search kept only if it clips no more, keeps the mid-tones, and raises E by more than 2 % or moves to a later bracket | Policy | Darkens a bright frame whose highlights crowd the top, leaving dark and backlit frames alone. |
| Highlight roll-off | anchored at sRGB 242, ceiling 0.988 (about code 252), ramp 0.10, exponent 2, at most 2.5 | Policy | The shoulder that rolls highlights off instead of clipping them; measured per frame from its 99.5th percentile, and used by every render, the export included. |
| Scene tail | 0.5 % | Numerical detail | The scene's range is taken between its 0.5 % and 99.5 % points. |
| Fixed-point tolerances | 0.02; 1.5 % | Numerical detail | When the search over E, g, b has converged. |
| Bootstrap gains | 1.10, 1.15 | Convention | The gains the first render uses before any intent field exists. |
| Knee | 0.95 | Convention | The soft knee of the previews and the final render. |
| Reference exposure | 1/8 s, ISO 100 | Convention | The search runs as if the frame were scanned at this exposure. |
| Metering grey | L\* 50 | Convention | The grey each bracket is metered to before screening. |

Seven policy knobs. Most of the thresholds are kept from the earlier station implementation, which records no reason
for them. Their ablation is not complete.

## Count

| Stage | Policy | Safety limits | Notes |
|---|---|---|---|
| 1 Auto search | 7 | 0 | shared with tri-LED; ablation pending |
| 2 Curve fit | 4 | 7 | plus the curve damping, a numerical detail with a small visible effect |
| Guard | 3 | 1 | plus the penalty weight, which has no derivation |
| 3 Tint trim | 7 | 1 | three are small uniform effects |
| 4 Profile tables | 0 | 0 | |
| 5 Tone stage | 5 | 2 | plus two numerical details with a small visible effect |
| **White light, stages 2 to 5** | **19** | **11** | |

## What is not principled yet

- **The evidence half-mass (12)** is a stated policy value. On the held-out measures 12, 24 and 48 score alike. 12 was
  chosen over 24 by blind comparison: 24 also damped well-evidenced frames, while 12 still protects the few frames
  with almost no neutral surface (evidence below 12 on 4 of the 140). To do: re-check the value on the scans that
  arrive through the web portal as they accumulate (their evidence distribution, and the corrections of the scans
  with little evidence at 6, 12, 24 and 48).
- **The slope span (6 L\*)** and **the guard's penalty weight (4)** have a stated purpose but no derived value.
- **The tint trim's read-out, candidate count and guard slack** make small uniform changes. A uniform offset can be set
  after conversion, so they add code without fixing anything only the conversion can fix.
- **The tone moves (1.0, 0.5, 0.5)** come from blind comparison.
- **The shadow end.** The neutral head finds few neutral surfaces in deep shadows, so the fit has little to anchor
  them, and a dark neutral object can end a few a\* off while brighter neutrals are right.
