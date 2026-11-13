# Design: the white-light conversion

How `neg_process` converts a negative scanned under white light (`--light_source d65`, `-P srgb`), and every
choice it makes along the way. Tri-LED scans run stages 1 and 4: their curves are the measured ones and are not
fitted. `--fit_curves off` leaves out stages 2 and 3, and `--tone_stage off` stage 4.

## The stages

| # | Stage | What it decides | Where it acts |
|---|---|---|---|
| 1 | Auto search | which exposure bracket's profile to use, and the exposure and green and blue gains | scores small previews; no pixel path of its own |
| 2 | Curve fit | a gain and tone-slope correction that makes neutral surfaces neutral and sets the overall magenta-green balance | the profile's green and blue input tables |
| 3 | Profile tables | nothing: bakes 2 into the profile | the input tables |
| 4 | Tone stage | one lightness curve for contrast and the black and white levels | linear sRGB inside the conversion, before encoding |

Every pixel goes through the conversion once. No stage converts finished pixels to another colour space and back.

## How the estimates work

**The intent field.** The search renders a small preview and the intent network predicts, for each cell i of a grid
over it, what a finished scan would show there: a mean mu_i = (mu_L, mu_a, mu_b) and an uncertainty sigma_i per
channel (small sigma: confident). A render with Lab values (L_i, a_i, b_i) is scored per channel:

```
z_L,i = (L_i - mu_L,i) / sigma_L,i        (likewise z_a, z_b)
loss_L = mean_i z_L,i^2                  (likewise loss_a, loss_b)
residual = (loss_L + loss_a + loss_b) / 3 + clip penalty + exposure penalty
```

Under a Gaussian model per cell this is the negative log-likelihood: uncertain cells count for little. The penalties
grow with white clipping above 2.5 % and with exposure off the bracket's range. Within a step, exposure is chosen on
loss_L plus the penalties, the green gain on loss_a, the blue gain on loss_b; the residual ranks steps and brackets.

```
render at (E, g, b) -> network predicts mu, sigma -> search (E, g, b) against them -> repeat
stop when |dE| + |dg| + |db| <= 0.02, or once the residual has passed its minimum (the best step is kept)
```

**The neutral prediction.** A second head gives each cell p_i in [0, 1], the probability that it shows a neutral
surface. The anchors are the cells Auto renders with 2 < L\* < 92; each is weighted by p squared:

```
m = sum_i p_i^2                 w_i = p_i^2 / m    (plus 10^-3 spread evenly, so m = 0 is defined)
a_bar = sum_i w_i a_i           b_bar = sum_i w_i b_i
k_a = 50 cov_w(L, a) / (var_w(L) + 25)          (likewise k_b)
```

Squaring sets no threshold but makes p = 0.9 count 81 times p = 0.1; m is the frame's neutral evidence. Neutral
surfaces have a\* = b\* = 0 at every lightness: the gains move the means, the curve slopes the trend (k, per 50 L\*).

**Trust.** The pull toward neutral grows with the evidence:

```
w = m / (m + 12)
target a_bar = (1 - w) a_bar_Auto          target b_bar = (1 - w) b_bar_Auto
target k_a = (1 - w s) k_a_Auto,  s = var_w(L) / (var_w(L) + 36)         (likewise k_b)
```

With no evidence the anchors hold Auto's colour; s keeps a trend measured over a narrow lightness range from being
extrapolated.

**Pooling the two estimates of the overall a\*.** The intent field gives a second estimate:

```
alpha_i = (1 / sigma_a,i^2) / sum_j (1 / sigma_a,j^2)        over every fit cell
s = sum_i alpha_i (a_i - mu_a,i)
intent term = (s - (k_a - target k_a) dL - SE) / sqrt(SE^2 + 1)
anchor term = (a_bar - target a_bar) / sigma_anc,     sigma_anc^2 = (144 + m) / (m + 12)
```

SE is the standard error of s from sums over 4 x 4 blocks of cells (neighbouring cells are not independent),
detrended in L\*. dL, the gap between the field's and the anchors' mean lightness per 50 L\*, keeps a slope change
from reading as a shift in a\*. Each term is divided by its own uncertainty (and both, like the b\* term, by 1.5), so
the solve is a precision-weighted compromise: with much neutral evidence the two count about equally and a
disagreement is split; with little, the intent field decides. The 1 under the root, and the limit 1 of sigma_anc as m
grows, are each estimate's systematic error in a\*.

| What rests on what | Basis |
|---|---|
| The z-score loss and the inverse-variance weights | maximum likelihood under a Gaussian model per cell |
| mu, sigma and p | learned networks, checked on held-out data and by blind comparison |
| p² weighting | a choice |
| 12, the 1 a\* systematic error, the one-standard-error lean | stated values, not derived |

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

One least-squares solve over the green and blue gains and the slopes of the green and blue input curves, on the
anchor and intent terms of "How the estimates work". Exposure is re-solved at every step so the median lightness
stays where Auto put it.

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

The stage runs after every bundle search with `-P srgb`, on white light and tri-LED alike. Where no curve fit runs
(tri-LED, or `--fit_curves off`) it is planned on the search's own result, which it leaves as it is. In a blind
comparison on 20 tri-LED frames the stage was preferred on 17 and the conversion without it on 2.

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

Not counted as choices: the solver's constants, which set how the terms are scaled and solved rather than what they
aim at — the guard's penalty weight 4 and counting minimums, the 4 x 4 blocks of the standard error, the slope weight
and the slope moment's 25, the residual scale 1/1.5, the slope ridge 1.5 and the lever factor. They live in
`input_curves.h`; "How the estimates work" shows where most of them enter.

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
