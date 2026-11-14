# Design

`neg_process` was built in three steps, and this document follows them.

1. **[The narrow-band LED conversion](#the-narrow-band-led-conversion).** The original design. Under a red, green
   and blue LED light the film's curves are measured on a chart, at several exposure brackets. Converting a frame
   is then a search: which bracket's profile, at what exposure and gains. The tone stage, added later for both
   lights, sets the contrast.
2. **[White light: the curve fit](#white-light-the-curve-fit).** Added on the first: scans under a white light,
   and with them two more stages, which fit for each frame a correction of the green and blue gains and curves of
   the profile.
3. **[Print emulation](#print-emulation).** Added last, for narrow-band scans: the negative as a density file, and a
   CLF beside it that prints the file on a paper or a print film.

The first two make a picture (`-P srgb`) in four stages, numbered in the order they run:

| # | Stage | What it decides | Where it acts |
|---|---|---|---|
| 1 | Auto search | Which exposure bracket's profile converts the frame, and the exposure and the green and blue gains to use with it. | Only on small preview renders, which it scores. It has no pixel path of its own: its result is a set of parameters for the conversion. |
| 2 | Curve fit | A correction of the green and blue gains and of the slopes of the green and blue tone curves. It makes surfaces the neutral head picked out come out grey, and sets the overall magenta-green balance. | The green and blue input tables of the profile. |
| 3 | Profile tables | Nothing new. It writes stage 2's correction into the profile, so the conversion applies it. | The input tables. |
| 4 | Tone stage | One lightness curve that sets the contrast and the black and white levels. | Linear sRGB inside the conversion, before the output is encoded. |

Every pixel goes through the conversion once. No stage converts finished pixels to another colour space and back.

A narrow-band scan (`--light_source tri`) runs stages 1 and 4. A white-light scan (`--light_source d65`) runs all
four; `--fit_curves off` leaves out stages 2 and 3, and `--tone_stage off` leaves out stage 4.

# The narrow-band LED conversion

Stages 1 and 4. The profile's curves are the measured ones and are not fitted.

## How the search scores a render

**The intent field.** The search renders a small preview, and the intent network predicts, for each cell i of a
grid over it, what a finished scan would show there: a mean mu_i = (mu_L, mu_a, mu_b) and an uncertainty sigma_i per
channel (small sigma means the network is confident). A render with Lab values (L_i, a_i, b_i) is scored per
channel:

```
z_L,i = (L_i - mu_L,i) / sigma_L,i        (likewise z_a, z_b)
loss_L = mean_i z_L,i^2                  (likewise loss_a, loss_b)
residual = (loss_L + loss_a + loss_b) / 3 + clip penalty + exposure penalty
```

Under a Gaussian model per cell this is the negative log-likelihood, so cells the network is unsure of count for
little. The clip penalty grows once more than 2.5 % of the preview clips white. The exposure penalty grows as the
exposure moves away from the centre of the bracket's calibrated range. Within one step, exposure is chosen on loss_L
plus the penalties, the green gain on loss_a and the blue gain on loss_b. The residual ranks steps and brackets.

```
render at (E, g, b) -> network predicts mu, sigma -> search (E, g, b) against them -> repeat
stop when |dE| + |dg| + |db| <= 0.02, or once the residual has passed its minimum (the best step is kept)
```

## Reading the tables

Each table lists the **choices** a stage makes: numbers that decide how the picture looks or how far an estimate is
trusted. **Effect** is the number of frames, out of a 140-frame test set, that change visibly (mean ΔE above 1) when
the choice is switched off, or halved or doubled where the table says so. It shows how much work a choice does, not
whether the result is better. Bounds that only act when something goes wrong are listed in one line under each
table. Iteration counts, step sizes and table sizes are left to the code.

## Stage 1: Auto search

For each candidate bracket, Auto renders the bracket's profile at a metered exposure, asks the intent network what
the finished picture should look like, and then searches exposure and gains for the closest match. White-light scans
use the same stage.

| Choice | Value | What it does |
|---|---|---|
| Admissible brackets | Tonal range at least 0.75 of the widest bracket's | In a set of five or more brackets, a bracket that calibrates a much shorter range than the best ones cannot hold a whole scene, so it is not considered. At least three are always kept. |
| Bracket ranking | Scene fills 0.85 of the bracket's range; 0.5 per stop of overflow | Ranks the brackets by how well the scene's contrast fits each one's calibrated range: ideally filling 85 % of it. A scene wider than a bracket costs 0.5 per stop it overflows by. |
| Tournament | Best 2 | The two best brackets from a first, single-render screening each get the full search. The one that ends with the lower residual wins. |
| Overflow allowed | 1.5 stops of highlight, 1.2 of shadow | How far a scene may run past a bracket's calibrated range, at either end, before the bracket is ruled out. |
| Exposure trust region | ±0.5 stop | The full search may move exposure at most half a stop from the screening's value. It refines the screened exposure instead of relocating it. |
| Search start | Gains 1.22 (green), 1.15 (blue) | Where each search over exposure and gains begins. The very first render, used only to get the network's first prediction, uses 1.10 and 1.15. |
| Search penalties | White clipping above 2.5 %; exposure away from the bracket's centre | Added to the intent residual, so the search avoids renders that clip or that leave the bracket's calibrated range. |
| Scene range | 0.5 % to 99.5 % | The scene's range is measured between these points, so a few specular highlights or dust specks do not count. |
| Clip fallback | Over 0.03 % white clip | Tries the next denser bracket. It is taken only if it halves the clipping, shifts the median lightness by no more than −4.5 to +0.5 L\*, and its residual is no more than 10 % worse. |
| Highlight pullback | Median L\* ≥ 58 with ≥ 2.2 % above sRGB 242, or ≥ 48 with ≥ 7.5 % | A bright frame whose highlights crowd the top was rendered too bright. Auto darkens it by up to 0.15 stop, or re-runs the search with wider limits when the scene does not fit the bracket. Dark or backlit frames are left alone, so the subject is not sunk. |
| Highlight roll-off | The scene's 99.5 % point mapped to sRGB 242 | Highlights above that point are compressed smoothly toward white instead of clipping. sRGB output only. |

That makes eleven choices. None of them has been switched off on the 140-frame test set yet, so the table has no
Effect column. The search also does not always converge to the same answer: where it ends depends on where it
starts.

## Stage 4: Tone stage

The tone head predicts the shadow level, highlight level and tonal range that a lab scan of the frame would have.
One lightness curve moves the image toward those values while holding the mid-tone where it is. The head needs one
network inference, on the preview of the final render: the only inference after the search. The curve is applied to
each pixel's linear sRGB value inside the conversion. The pixel's brightness follows the curve, and its colour is
carried along unchanged.

The stage runs after every bracket search with `-P srgb`, on white light and narrow-band alike. Where no curve fit
runs (narrow-band, or `--fit_curves off`), it is planned on the search's own result and leaves that result's colour as it is.
In a blind comparison on 20 narrow-band frames, the stage was preferred on 17 and the conversion without it on 2.

| Choice | Value | What it does | Effect |
|---|---|---|---|
| Shadow move | 1.0 | Moves the shadows all the way to the predicted shadow level. | 60 at half |
| Highlight move | 0.5 | Moves the highlights half way to the predicted highlight level. | 16 to 30 |
| Range move | 0.5 | Moves the tonal range half way to the predicted range. | 38 to 61 |
| Clip allowance | 0.5 % of pixels | If the curve would clip more than this share of pixels to white, the highlight and range moves are scaled back until it does not. | 14 |
| Curve limits | Black ±15, white 80 to 110, gamma 0.7 to 1.45, contrast 0.7 to 1.5 | How far each of the curve's four parameters may go. On most frames at least one of them sits on its limit. | — |

# White light: the curve fit

Stages 2 and 3, between the search and the tone stage. They exist for white light only. A white-light conversion leaves
a cast that differs from frame to frame, more than a narrow-band one does: the search has chosen a bracket, an
exposure and gains, and the fit then corrects the green and blue gains and the slopes of the green and blue curves,
so that what the frame shows as neutral comes out grey.

What the two sets of profiles show of that, on the chart they were built from (CIEDE2000, mean over a stock's seven
brackets; each bundle's README has the brackets):

| Stock | Fitted patches: narrow band | white light | Held-out patches: narrow band | white light |
|---|---|---|---|---|
| Portra 400 | 0.61 | 0.78 | 2.64 | 2.17 |
| Portra 160 | 0.72 | 0.89 | 2.41 | 2.04 |
| Ektar 100 | 0.72 | 0.90 | 3.29 | 2.42 |

On the patches a profile was fitted to, white light is the less exact by about a quarter. On patches held out of
the fit it is the more exact, by 15 to 26 %. So the chart does not show the larger error of a white-light
conversion. That error is seen on converted frames, and this document has no figure for it.

## How the fit estimates a cast

**The neutral prediction.** A second head gives each cell p_i in [0, 1], the probability that it shows a neutral
surface. The anchors are the cells that Auto's render puts between L\* 2 and L\* 92, and each is weighted by p squared:

```
m = sum_i p_i^2                 w_i = p_i^2 / m    (plus 10^-3 spread evenly, so m = 0 is defined)
a_bar = sum_i w_i a_i           b_bar = sum_i w_i b_i
k_a = 50 cov_w(L, a) / (var_w(L) + 25)          (likewise k_b)
```

Squaring sets no threshold, but it makes a cell with p = 0.9 count 81 times as much as one with p = 0.1. m is the
frame's total neutral evidence. A neutral surface has a\* = b\* = 0 at every lightness, so two things are corrected:
the gains move the weighted means a_bar and b_bar, and the curve slopes remove the trend k (how far a\* or b\* drifts
per 50 L\*).

**Trust.** The pull toward neutral grows with the evidence:

```
w = m / (m + 12)
target a_bar = (1 - w) a_bar_Auto          target b_bar = (1 - w) b_bar_Auto
target k_a = (1 - w s) k_a_Auto,  s = var_w(L) / (var_w(L) + 36)         (likewise k_b)
```

With no evidence (m near 0) the targets are Auto's own values, so the anchors leave Auto's colour alone. s shrinks
the slope correction when the anchors cover only a narrow range of lightness, so a trend measured there is not
extrapolated over the whole curve.

**Pooling the two estimates of the overall a\*.** The intent field gives a second estimate of the overall
magenta-green balance, besides the anchors:

```
alpha_i = (1 / sigma_a,i^2) / sum_j (1 / sigma_a,j^2)        over every fit cell
s = sum_i alpha_i (a_i - mu_a,i)
intent term = (s - (k_a - target k_a) dL - SE) / sqrt(SE^2 + 1)
anchor term = (a_bar - target a_bar) / sigma_anc,     sigma_anc^2 = (144 + m) / (m + 12)
```

s is the precision-weighted mean gap between the render's a\* and what the intent field predicts. SE is the standard
error of s, computed from sums over 4 x 4 blocks of cells because neighbouring cells are not independent, and
detrended in L\*. dL is the gap between the intent field's and the anchors' mean lightness, per 50 L\*. It keeps a
change of slope from being read as a shift of a\*. Subtracting SE leans the intent term's target toward magenta by
one standard error.

Each term is divided by its own uncertainty, and both, like the b\* term, by 1.5, so the solve is a
precision-weighted compromise. With much neutral evidence the two estimates count about equally and a disagreement
between them is split. With little, the intent field decides. The 1 under the square root, and the value 1 that
sigma_anc approaches as m grows, are each estimate's systematic error in a\*: however much evidence there is, neither
is trusted closer than that.

| What rests on what | Basis |
|---|---|
| The z-score loss and the inverse-variance weights | Maximum likelihood under a Gaussian model for each cell. |
| mu, sigma and p | Learned networks, checked on held-out data and in blind comparisons. |
| The p² weighting | A choice: it is not derived from anything. |
| The evidence half-mass 12, the systematic error of 1 a\* and the one-standard-error lean | Stated values. None of them is derived or fitted. |

## Stage 2: Curve fit

One least-squares solve over four numbers: the green and blue gains, and the slopes of the green and blue input
curves. It works on the anchor and intent terms described in "How the fit estimates a cast". Exposure is re-solved at
every step, so the median lightness stays where Auto put it.

| Choice | Value | What it does | Effect |
|---|---|---|---|
| Evidence half-mass | 12 | The anchors pull toward neutral with strength m / (m + 12): half strength when the neutral evidence m is 12. A correction resting on very little neutral surface is mostly ignored. | — |
| Head systematic error | 1 a\* | Neither estimate of the overall a\* (the neutral anchors or the intent field) is trusted more precisely than this. So when there is plenty of evidence and they disagree, the difference is split. A stated value, not derived. | — |
| Magenta lean | 1 standard error | The intent term's target leans toward magenta by its own uncertainty, because slightly magenta looks better than slightly green. A stated value, not derived. | — |
| Anchor range | L\* 2 to 92 | Only cells between these lightnesses are anchors. Clipped blacks and whites carry no colour, so they say nothing about a cast. | — |
| Slope span | 6 L\* | A tone slope measured over a narrow range of lightness is shrunk toward Auto's instead of being extrapolated. | 34 |
| Slope hold | On | Beyond the densest cell in the frame the curve stops bending, so the slope fitted in the measured range is not carried into tones the frame does not contain. | 38 |

Bounds: the curve slope is held within 0.35; the curve may deviate at most 2.2 stops from the profile's own; the
green gain may move 0.75 stop and the blue gain 1.25 stops from Auto's; the median lightness must stay plausible (L\*
18 to 85); at least 8 cells must be in range; on an uncropped frame, cells on the film rebate or the holder are left
out.

### The never-greener guard

The fit may not end greener than Auto's render: slightly magenta is preferred to slightly green. The fit first runs
freely. Only if its result is greener than Auto's does it run again with the excess green penalised.

| Choice | Value | What it does | Effect |
|---|---|---|---|
| The guard | On | As above. Without it, mid-tones end up greener on 38 of the 140 frames. | 41 |
| Tolerance | 0.5 a\* | A result up to this much greener than Auto's still counts as not greener. | 7 |
| Shadow exemption | Full below L\* 10, fading out by L\* 20 | The guard does not apply in the deep shadows, so a magenta cast there can still be corrected toward green. | 10 |

## Stage 3: Profile tables

Nothing is decided here. The correction of stage 2 is written into the profile's green and blue input tables, so
the one conversion of the pixels applies it.

## Count

| Stage | Choices |
|---|---|
| 1 Auto search | 11 |
| 2 Curve fit | 6 |
| Guard | 3 |
| 3 Profile tables | 0 |
| 4 Tone stage | 5 |
| **White light, stages 2 to 4** | **14** |

Not counted as choices: the solver's constants. They set how the terms are scaled and solved rather than what the
solve aims at: the guard's penalty weight 4 and its counting minimums, the 4 x 4 blocks of the standard error, the
slope weight and the 25 in the slope moment, the residual scale 1/1.5, the slope ridge 1.5 and the lever factor.
They live in `input_curves.h`, and "How the fit estimates a cast" shows where most of them enter.

## What is not principled yet

- **Evidence half-mass (12).** A stated value, not a derived one. To do: re-check it on the scans that arrive
  through the web portal.
- **Slope span (6 L\*)** and **the guard's penalty strength.** Each has a clear purpose, but neither value is
  derived.
- **Tone moves (1.0, 0.5, 0.5).** Chosen by blind comparison, not derived.
- **The intent term's two constants** (head systematic error 1 a\*, magenta lean 1 standard error) are stated, not
  derived. Folding the overall balance into the fit costs a little where the true colour is known: on the emulated
  scans the grey a\* error is 8.03, against 7.62 with the separate trim stage it replaced. The guard is a penalty,
  not an exact bound: results end up to 0.18 a\* past it on 39 of the 140 frames (with the trim stage, up to 0.75 on
  18).
- **The intent term's cells.** Its weights run over every fit cell, including those Auto renders above L\* 92. On
  one upload those cells carry up to 42 % of its precision. That is the form that was checked in blind comparison;
  restricting it to the anchor range has not been tried.
- **The guard.** It costs accuracy where the true colour is known, because Auto's render can itself be magenta. It
  stays because guarded renders were preferred on real frames.
- **Tone-dependent error.** The fit removes most of a uniform cast but little of a cast that differs between shadows
  and highlights. What remains comes from the profile and the shape of the curve. Deep shadows have few neutral
  anchors, so a dark neutral object can end a few a\* off.
- **The clip check** measures the whole export, film borders included, while the tone curve is planned on the inner
  crop.
- **Stage 1 thresholds that need confirming:** the admissible-bracket cut (0.75), the overflow tolerances (1.5 and
  1.2 stops), the clip fallback's trigger and acceptance limits (0.03 %, half the clipping, −4.5 to +0.5 L\*, 10 %)
  and the highlight pullback's tiers (L\* 58 with 2.2 %, L\* 48 with 7.5 %).

# Print emulation

How `neg_process` writes a narrow-band scan as density (`-P statusm`, `-P rp180`, `-P apd`) and, beside the file, a
CLF that prints it on a paper or a print film, and every choice it makes along the way. The density file is the film
as measured. The CLF holds everything that turns it into a picture, and `neg_process` only looks its numbers up and
multiplies them: nothing is fitted when a file is written.

## The stages

| # | Stage | What it does | Where its numbers come from |
|---|---|---|---|
| 1 | Density | Camera RGB to the density of the film above its base, in one standard measure. Writes the density file. | The profile set: the crosstalk correction and the density matrix. |
| 2 | Search | Stage 1 of the picture conversion, unchanged: the exposure bracket, the exposure and the green and blue gains. Nothing is applied to the file. | The frame. |
| 3 | Frame matrix | Places the frame on the print: its mid grey on the print's mid grey, its balance, and the slopes of its grey scale. One matrix, the first node of the CLF. | The search, the bracket's entry in the profile set, the print material's matrix. |
| 4 | Print | The print material's response: its curve per layer, then its dyes under a viewing light, as sRGB. The other nodes of the CLF. | The print material's file. |

A profile set holds nothing of a print material, and a print material's file (`prints/<print>.json`) holds nothing
measured on a film. `-P` picks one measure for a run. The code is `np_print.cc`.

## How the numbers are found

**Density.** `v` is the camera RGB after the flat field, `C` the crosstalk correction and `B` the scale that puts
the film base on one level:

```
T        = B C v / base level                 transmittance against bare film, per LED channel
D_light  = -log10(max(T, 0.0001))
D        = M D_light                          the measure's density
code     = offset + scale k D                 10 bits: 95 + 500 k D     16 bits: 1520 + 8000 k D
```

**The density matrix, in two parts.** `M = F L`. `L` takes the density the LED sees to the film's three dye amounts
and depends on the light's spectrum and the film's dyes. `F` takes the dye amounts to the measure and depends on the
film's dyes and the measure's response curves, not on the light. The two are kept apart in `bundle.json`, so a
scanner with another light replaces `L` alone. Each is a least-squares 3×3 with no offset, on 3000 sample colours of
a model of the film: dye amounts along its neutral scale between log exposure −1.5 and 1.0, with a log-normal colour
variation.

**The print matrix.** `P` takes a measure's density to the density the print material's own layers read: its
spectral sensitivity under a printer lamp. It is fitted the same way, for each measure and each film model, and is
in the print material's file.

**Mid grey.** Each bracket of a profile set is one shot of the chart, and the profile set keeps the chart's 18 %
grey patch of each bracket as the density the LED sees. The bracket's profile renders that density as mid grey at an
exposure of 1. The search's exposure `E` multiplies the LED channels, so for this frame the film that renders as mid
grey is denser by `log10 E` in each:

```
grey = M (bracket's mid grey + log10 E)
```

**Balance.** The gains `g` and `b` multiply the green and blue LED channels, which in density is an offset:

```
balance = M [0, -log10 g, -log10 b]
```

**Slopes.** Over a bracket's 24 grey steps, the red and the blue density do not change as the green does. With
`r_i` a step's density less the mid grey's, in Academy Printing Density:

```
k_red = sum_i r_i,red r_i,green / sum_i r_i,green^2          (likewise k_blue)
S     = (1 / k_red, 1, 1 / k_blue)
```

Multiplied by `S`, the film's grey scale has equal contrast in the three channels. `S` is computed when the profile
set is built and kept per bracket; a frame takes that of its bracket.

![The grey steps of a bracket in APD, and its slopes](samples/slopes_grey_steps.png)

**The frame matrix.** On the file's code `x` (the code over the largest code):

```
D             = (x largest code - offset) / (scale k)
print density = 0.70 + S P (D + balance - grey)
```

The film that the picture conversion renders as the chart's mid grey is denser than the chart's by `log10 E`,
`log10 E g` and `log10 E b` in the LED channels. In this formula it comes out at 0.70 in all three channels, so the
print and the picture agree on which film is mid grey and neutral.

**The print.** For each layer `c` and wavelength `w`:

```
logE_c = aim_c - (print density_c - 0.70)       aim: the exposure that prints an 18 % grey
a_c    = curve_c(logE_c)                        dye amount, held at the ends of the curve
t(w)   = 10^-(sum_c a_c dye_c(w) + base(w))     the print's transmittance or reflectance
XYZ    = sum_w t(w) light(w) observer(w) / sum_w light(w) y(w)
rgb    = X XYZ                                  linear sRGB, then the sRGB curve
```

| What rests on what | Basis |
|---|---|
| `C` | Measured: each LED captured alone. |
| `L`, `F`, `P` | Fitted on a model of the film, from spectra alone. None is fitted to the film's chart. |
| The mid grey and the slopes of a bracket | The film's chart, as the profile set was built from it. |
| The bracket, `E`, `g`, `b` | The picture conversion's search, which uses the ICC profiles. |
| The print's curves, dyes and sensitivity | spektrafilm's print profiles. A model, not compared with a real print. |

## Stage 1: the density file

| Choice | Value | What it does |
|---|---|---|
| Measures | Status M, RP 180, APD | `rp180` is SMPTE RP 180 printing density, the Cineon convention; `apd` is Academy Printing Density, SMPTE ST 2065-2. |
| Container | `.dpx`: 10 bits, `95 + 500 k D`. `.tif`: 16 bits, `1520 + 8000 k D` | The output's extension picks it. `rp180` in a DPX is a Cineon file; `apd` is ADX10 or ADX16. A DPX clips at 1.86 above the base. |
| ADX factors `k` | 1.00, 0.92, 0.95 for `apd`; 1 otherwise | SMPTE ST 2065-3, so an ADX reader gets APD back. With them a DPX clips at 1.86, 2.02 and 1.95. |
| Zero | The film base | From `--film_base`, a capture of unexposed film, else from the scan's rebate, else estimated from the thinnest film in the frame, with a warning. |
| Density floor | 4.0 | A transmittance below 0.0001 is held there, and counted. |
| Flat field | As for a picture | It corrects the scan and the base capture before the density is taken. |
| Not applied | Exposure, gains, curves, tone stage, look | The file is the film as measured. The search's result is recorded (`suggested_balance`). |

The file carries the matrix, the LED's spectrum, the film base and the encoding as JSON, in a TIFF's
`ImageDescription` or a DPX's user data, and so does the sidecar.

How well a 3×3 reproduces each measure on the film model (`fit_rms_density`, `fit_max_density` in the bundle), over
the three stocks and the densities a scan reaches, about 2.2 above the base:

| Measure | RMS error (density) | Worst case (density) |
|---|---|---|
| Status M | up to 0.004 | up to 0.04, in green |
| RP 180 | up to 0.004 in red and green, 0.010 in blue | up to 0.10, in blue |
| APD | up to 0.010 in red, 0.002 in green and blue | up to 0.08, in red |

These are errors against the model, not against the film.

- **Film data that does not exist.** The datasheets give no base density below about 400 nm or above about 700 nm,
  and the nearest measured value is held there. RP 180 blue and APD red have part of their response outside that
  range (`response_outside_film_data`). Take those two rows as approximate.
- **Status M is calibrated on the chart**, the printing densities are not. One factor on the green and one on the
  blue LED density hold the chart's grey scale to the film model's Status M: 1.02 and 1.06 for Portra 400, 0.96 and
  0.97 for Portra 160. Ektar 100's, 0.83 and 1.08, lie more than 0.10 from 1 and are recorded, not applied.
- **Each LED band is taken as one channel.** What the camera adds to that is sized below.

## Stage 3: the frame matrix

| Choice | Value | What it does |
|---|---|---|
| Mid grey patch | `gs10` | The chart's step nearest 18 % reflectance. |
| Mid grey lands on | 0.70 | Code 445 of a Cineon file, where Cineon holds the LAD patch. The constant cancels inside the CLF. |
| Slopes measured in | APD | The standard measure in which equal steps are meant to print neutral. The same slopes serve a file in any measure. |
| Slopes taken from | The frame's bracket | One pair per bracket, since the film's channels shift with exposure. |
| Slopes applied | After `P` | They scale two rows of the print matrix. |

The slopes are the film's alone: no print material enters them, so a print material keeps its own grey scale. On
paper the bracket's grey steps then print near neutral. On Vision 2383, which in the model prints a neutral scale
from green in the shadows to yellow in the highlights, that look stays.

![The grey steps printed without and with the bracket's slopes](samples/slopes_grey_steps_printed.png)

The brackets are seven exposures of one chart. Portra 400's slopes agree within 0.04. Portra 160's red runs from
1.08 to 0.89 and Ektar 100's from 0.83 to 1.18, with its blue from 0.83 to 1.35: on those two the film's channels
change their relation with exposure. The squares are what the same formula gives the film's datasheet model.

![The slopes in the bundle, bracket by bracket](samples/slopes_across_brackets.png)

Two frames of one roll on different brackets are printed with different slopes. One Portra 400 frame, whose bracket
is Target 3, with the slopes of the two brackets furthest from it: against its own they move the picture by 0.8 and
1.4 of 255 levels on average. For Ektar 100 the brackets differ ten times as much.

![One frame with the slopes of three brackets](samples/slopes_of_other_brackets.jpg)

The same frame without slopes and with its own, on each print material, beside the picture conversion:

![One frame without and with its bracket's slopes](samples/slopes_on_a_picture.jpg)

## Stage 4: the print

| Choice | Value | What it does |
|---|---|---|
| Print materials | Kodak Portra Endura (paper), Kodak Vision 2383 (print film) | One CLF each, `<output>.<print>.clf`. |
| Printer lamp | 3200 K blackbody, no heat filter | With the material's sensitivity it defines the density the material reads. |
| Viewing light | D50 for paper, D55 for print film | The print's colour is computed under it and adapted to D65 (CAT02) for sRGB. |
| Curve table | 4096 points | The curve of each layer, as a 1D table. |
| Colour table | 45 points a side | Dye amounts to linear sRGB, as a 3D table. |

A CLF fits the one file it is named after: its first matrix holds that file's code scale, measure and channel
factors. To print lighter or darker, add a constant to the three offsets of the first matrix; 0.30 is a stop. The
balance, the exposure and the slopes are applied there and nowhere else. The CLF's two tables derive from
spektrafilm's print profiles and carry their licence notice (CC BY-SA 4.0); the matrices are computed.

## The spectral data

All of it is on 380 to 780 nm at 1 nm, and all of it is used ahead of time, except that the print's colour table
is computed from the print material's file when a CLF is written.

| Data | Specific to | Used for | Kept in |
|---|---|---|---|
| The scanning light's spectrum, split into three LED bands at 495 and 578 nm | The light | `L` | `bundle.json`, `light_source` |
| The film's three dye spectra and base, and its neutral scale | The film stock | `L`, `F`, `P` | Not shipped |
| The response curves of Status M, RP 180 and APD | The standards | `F`, `P` | Not shipped |
| The print material's spectral sensitivity | The print material | `P` | Not shipped |
| The print material's curve per layer, dye spectra and base | The print material | The print | `prints/<print>.json` |
| The CIE 1931 2° observer, D50 and D55 | The CIE tables | The print | `prints/<print>.json`, as their product |

The camera has no spectral data here. The crosstalk correction stands in for its response curves.

## The crosstalk correction, measured and computed

Each camera pixel sees all three LEDs: the green pixel takes in nearly half as much of the blue LED as the blue
pixel does. `C` undoes that. It is measured by capturing each LED alone. It could instead be computed from the
camera's response curves and the light's spectrum, and this section compares the two.

![The camera's response curves and the light's three LEDs](samples/crosstalk_spectra.png)

The computed one uses the published curves of the same camera model (Sony ILCE-7RM4, `rawtoaces-data` at commit
`28801d1ad6`) and the three LEDs separated out of the all-on spectrum, each fitted as two skewed Gaussians.

![Crosstalk terms, measured against computed](samples/crosstalk_measured_vs_computed.png)

| Leak | Measured | Computed | Computed / measured |
|---|---|---|---|
| Green pixel, red LED | 0.168 | 0.170 | 1.01 |
| Blue pixel, red LED | 0.017 | 0.016 | 0.95 |
| Red pixel, green LED | 0.089 | 0.079 | 0.89 |
| Blue pixel, green LED | 0.280 | 0.269 | 0.96 |
| Red pixel, blue LED | 0.063 | 0.041 | 0.66 |
| Green pixel, blue LED | 0.457 | 0.421 | 0.92 |

Four of the six agree within 8 %. The two of the red pixel do not. What that does to density, on a model of Portra
400 seen through the published curves, up to a density of 1.5 (red / green / blue):

| | RMS | Largest |
|---|---|---|
| Camera with the measured correction, against the LED bands alone | 0.015 / 0.021 / 0.009 | 0.049 / 0.055 / 0.076 |
| Camera with the computed correction, against the LED bands alone | 0.012 / 0.013 / 0.007 | 0.050 / 0.038 / 0.062 |
| The computed correction against the measured one | 0.003 / 0.009 / 0.002 | 0.008 / 0.044 / 0.014 |

Most of the first two rows is a scale: against the LED band alone the model's camera reads red and green 2 to 3 %
flatter and blue 1 % steeper. Above a density of 1.5 the corrected green and blue signals near zero and the
comparison stops meaning much. The measured matrix is kept: it is of this camera and this light, and needs no
spectral data of the camera.

## What depends on what

| | The frame | The film stock | The print material | The scanner and its light |
|---|---|---|---|---|
| `C`, the flat field | | | | yes |
| `B` | yes: the roll's film base | | | |
| `L` | | yes: its dye spectra | | yes: the light's spectrum |
| `F` | | yes: its dye spectra | | |
| `P` | | a little: its dye spectra | yes | |
| The mid grey and slopes of a bracket | | yes: the chart | | |
| balance, `grey`, `S` | yes: the search's bracket, exposure and gains | | | |
| CLF nodes 2 to 7 | | | yes | |

## What is not principled yet

- **The densities are a model's.** `L` and `F` come from datasheet spectra and have not been checked against a
  densitometer.
- **The print material is a model.** Its curves, dye spectra and sensitivity are spektrafilm's, the lamp is a
  blackbody, and no result has been compared with a real print.
- **The slopes belong to one chart.** They hold for the film batch, the development and the scan the profile set
  was built from. Film from another lab may differ.
- **The slopes step between brackets.** For Ektar 100 by enough to show between two frames of one roll.
  Interpolating them at the frame's own mid grey has not been tried.
- **The slopes remove part of the film's look.** A film's own drift of its greys from shadows to highlights is
  taken out with them. Whether a print should keep it has not been judged on pictures.
- **Mid grey at 0.70, printed as an 18 % grey.** A Cineon scan has an 18 % card at 0.75, and a print to the LAD aim
  is about a third of a stop of exposure darker. Both are stated values.
