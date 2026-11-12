# Design: the white-light conversion

How `neg_process` converts a negative scanned under white light (`--light_source d65`, `-P srgb`), and every
choice it makes along the way. Tri-LED scans and `--fit_curves off` run stage 1 only.

## The stages

| # | Stage | What it decides | Where it acts |
|---|---|---|---|
| 1 | Auto search | which exposure bracket's profile to use, and the exposure and green and blue gains | scores small previews; no pixel path of its own |
| 2 | Curve fit | a gain and tone-slope correction that makes neutral surfaces neutral and sets the overall magenta-green balance | the profile's green and blue input tables |
| 3 | Profile tables | nothing: bakes 2 into the profile | the input tables |
| 4 | Tone stage | one lightness curve for contrast and the black and white levels | linear sRGB inside the conversion, before encoding |

Every pixel goes through the conversion once. No stage converts finished pixels to another colour space and back.

## Reading the tables

Each table lists the **choices** a stage makes: numbers that decide how the picture looks or how far an estimate is
trusted. **Effect** is how many frames of a 140-frame test set change visibly (mean ΔE above 1) when the choice is
switched off, or halved or doubled where it says so. It shows how much work the choice does, not whether the result
is better. Bounds that only act when something goes wrong are listed in one line under each table. Iteration
counts, step sizes and table sizes are left to the code.

## Stage 1: Auto search

Renders each bracket's profile at a metered exposure, asks the intent network what the picture should look like,
and searches exposure and gains for the closest match. Shared with tri-LED.

| Choice | Value | What it does |
|---|---|---|
| Admissible brackets | within 0.75 of the widest | Which brackets may be chosen at all. |
| Bracket ranking | fill 0.85 of the range; 0.5 per stop of overflow | Prefers the bracket the scene fits best. |
| Tournament | best 2 | The two best-ranked brackets get the full search; the lower residual wins. |
| Overflow allowed | 1.5 stops of highlight, 1.2 of shadow | How far a scene may exceed a bracket's calibrated range. |
| Exposure trust region | ±0.5 stop | The search refines the metered exposure instead of leaving it. |
| Search start | gains 1.22, 1.15 | Where each search over exposure and gains begins (the first, bootstrap render uses 1.10, 1.15). |
| Search penalties | clipping above 2.5 %; exposure far from the bracket's centre | Added to the intent residual. |
| Scene range | 0.5 % to 99.5 % | The tails ignored when measuring the scene's range. |
| Clip fallback | several thresholds | Moves to the next denser bracket when highlights clip. |
| Highlight pullback | several thresholds | Darkens a bright frame whose highlights crowd the top. |
| Highlight roll-off | anchored at sRGB 242 | Rolls highlights off instead of clipping them. |

Eleven choices. Most thresholds come from the earlier station implementation, which records no reason for them, and
their effect has not been fully measured. The search does not always converge: its result depends on where it
starts.

## Stage 2: Curve fit

One least-squares solve. Every grid cell Auto renders between L\* 2 and 92 is an anchor, weighted by the square of
the neutral head's probability that it shows a neutral surface. The fit moves the green and blue gains and the
slope of the green and blue input curves until the anchors' colour, and its trend from shadows to highlights, is
neutral, pulled at strength m / (m + 12), where m is the neutral evidence in the frame. In the same solve, the
overall magenta-green balance is pulled toward what the search's intent field says, so the two estimates are pooled:
the anchors count for more the more evidence there is, but never past each one's systematic error. The intent term
leans one of its standard errors toward magenta. Exposure is re-solved at every step so the median lightness stays
where Auto put it.

| Choice | Value | What it does | Effect |
|---|---|---|---|
| Evidence half-mass | 12 | The anchors pull at strength m / (m + 12), so a correction resting on almost no neutral surface is not trusted. | — |
| Head systematic error | 1 a\* | Each estimate of the overall a\* (neutral anchors, intent field) is trusted no further than this, so at high evidence their disagreement is split. A stated value, not derived. | — |
| Magenta lean | 1 standard error | The intent term's target leans toward magenta by its own uncertainty. A stated value, not derived. | — |
| Anchor range | L\* 2 to 92 | Clipped blacks and whites say nothing about a cast. | — |
| Slope span | 6 L\* | A tone slope measured over a narrow range of lightness is shrunk, not extrapolated. | 34 |
| Slope hold | on | Beyond the densest cell the curve stops bending. | 38 |

Bounds: curve slope 0.35; deviation from the profile's curve 2.2 stops; green gain 0.75 stops and blue gain 1.25
stops around Auto's; a plausible median lightness (L\* 18 to 85); at least 8 cells in range; rebate and holder
cells left out on an uncropped frame.

### The never-greener guard

The fit may not end greener than Auto's render: slightly magenta is preferred to slightly green. The fit runs
freely first, and only if its result is greener than Auto does it run again with the excess penalised.

| Choice | Value | What it does | Effect |
|---|---|---|---|
| The guard | on | As above. Without it mid-tones go greener on 38 of 140 frames. | 41 |
| Tolerance | 0.5 a\* | How much greener still counts as not greener. | 7 |
| Shadow exemption | full below L\* 10, fading out by 20 | A magenta cast in the shadows may be corrected. | 10 |

## Stage 4: Tone stage

The tone head predicts the shadow level, highlight level and tonal range a lab scan of the frame would have. One
lightness curve moves the image toward them while holding the mid-tone. The head reads one network inference on
the final render's preview, the only one after the search. It is applied to each pixel's linear sRGB
value inside the conversion: the pixel's brightness follows the curve and its colour is carried along unchanged.

| Choice | Value | What it does | Effect |
|---|---|---|---|
| Shadow move | 1.0 | All the way to the predicted shadow level. | 60 at half |
| Highlight move | 0.5 | Half way to the predicted highlight level. | 16 to 30 |
| Range move | 0.5 | Half way to the predicted tonal range. | 38 to 61 |
| Clip allowance | 0.5 % of pixels | Scales the highlight and range moves back if the curve would clip more white. | 14 |
| Curve limits | black ±15, white 80 to 110, gamma 0.7 to 1.45, contrast 0.7 to 1.5 | How far the curve may go. It sits on a limit on most frames. | — |

## Count

| Stage | Choices |
|---|---|
| 1 Auto search | 11 |
| 2 Curve fit | 6 |
| Guard | 3 |
| 3 Profile tables | 0 |
| 4 Tone stage | 5 |
| **White light, stages 2 to 4** | **14** |

## What is not principled yet

- **Evidence half-mass (12).** A stated value, not a derived one. To do: re-check it on the scans that arrive
  through the web portal.
- **Slope span (6 L\*)** and **the guard's penalty strength.** Each has a purpose but no derived value.
- **Tone moves (1.0, 0.5, 0.5).** Chosen by blind comparison.
- **The intent term's two constants** (head systematic error 1 a\*, magenta lean 1 standard error) are stated, not
  derived. Folding the overall balance into the fit costs a little where the true colour is known: on the emulated
  scans the grey a\* error is 8.03 against 7.62 with the separate trim stage. The guard is a penalty, not an exact
  bound: results end up to 0.18 a\* past it on 39 of 140 frames (with the trim stage, up to 0.75 on 18).
- **The intent term's cells.** Its weights run over every fit cell, including those Auto renders above L\* 92, which
  carry up to 42 % of its precision on one upload. That is the form that was blind-checked; restricting it to the
  anchor range is not done.
- **The guard.** It costs accuracy where the true colour is known, because Auto's render can itself be magenta. It
  stays because guarded renders were preferred on real frames.
- **Tone-dependent error.** The fit removes most of the uniform cast and little of the difference between shadows
  and highlights; what remains comes from the profile and the shape of the curve. Deep shadows have few neutral
  anchors, so a dark neutral object can end a few a\* off.
- **The clip check** measures the whole export, film borders included, while the tone curve is planned on the
  inner crop.
- **Stage 1.** Its thresholds are inherited and largely unmeasured.
