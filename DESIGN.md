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
| Highlight pullback | Median L\* ≥ 58 with ≥ 2.2 % above sRGB 242, or ≥ 48 with ≥ 7.5 % | A bright frame whose highlights crowd the top was rendered too bright. Auto nudges exposure down by up to 0.15 stop, or re-runs the search with wider limits when the scene does not fit the bracket. Dark or backlit frames are left alone, so the subject is not sunk. |
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
- **Stage 1's thresholds have no recorded basis.** The admissible-bracket rule, the overflow tolerances, the clip
  fallback and the highlight pullback have no recorded reason for their values. None of them has been switched off
  on the test set, so how many frames each one changes is unknown. To do: measure the Effect of each, as for stages
  2 and 4.

# Print emulation

With a narrow-band LED scan, `-P statusm`, `-P rp180` or `-P apd` writes the negative's density above the film base in
place of a picture, for grading tools and film-print LUTs. Beside a printing density file (`rp180`, `apd`)
`neg_process` also writes a CLF per print material, which renders the file as a print on that material. This part
gives the stages from a scan to that print, what the density file holds and how good it is, the spectral data and
the matrices behind it, and what is multiplied in what order. The README has the commands.

Two things are prepared ahead of time: the film's profile set and the print material's file. Two are computed when
`neg_process` runs: the density of each pixel, and the CLF for the frame. A profile set holds nothing of a print
material, and a print material's file holds nothing measured on a film.

## The stages

| # | Stage | When | Input | Output |
|---|---|---|---|---|
| A | Density matrices of a film and a light | When the profile set is built | Three inputs: (1) the light's spectrum; (2) the film's three dye spectra and base; (3) the response curves of one density measure, a red, a green and a blue curve. There are three measures (Status M, RP 180, APD), each with its own three curves, and the fit is run once for each | `M`, one 3×3 per measure, in `bundle.json` |
| B | A print material's file | Once per print material, ahead of time | The print material's sensitivity, curves and dye spectra, a printer lamp, the film's dye spectra, the measures' response curves, the CIE observer and a viewing light | `prints/<print>.json` |
| 1 | Scan to light density | Per pixel | Camera RGB, the flat field, the crosstalk correction, the film base | `D_light`: density in the red, green and blue of the LED |
| 2 | Light density to a density measure | Per pixel | `D_light`, `M` | The density file: Status M, RP 180 or APD |
| 3 | Balance and exposure | Per frame: stage 1 of the conversion, the Auto search | The frame | The bracket, the exposure `E`, the gains `g` and `b` |
| 4 | The frame's mid grey `grey` and the slopes `S` (defined below the table) | Per frame | The 24 grey steps of that bracket from the profile set, `M`, the print material's file | `grey`: three densities. `S`: three ratios |
| 5 | Density to the print's printing density | CLF node 1 | The file's code values, `P`, `S`, the balance, `grey` | The density the print material reads |
| 6 | Printing density to the print's exposure | CLF node 2 | The exposure that prints an 18 % grey | Log exposure per layer |
| 7 | Exposure to dye amounts | CLF nodes 3 and 4: a range and a 1D table of 4096 points | The print's curve per layer | Three dye amounts |
| 8 | Dye amounts to colour | CLF nodes 5 and 6: a scale and a 3D table of 45 points a side | The print's dye spectra and base, the observer under the viewing light | Linear sRGB |
| 9 | Display | CLF node 7 | | sRGB |

The two results of stage 4 are each a vector of three numbers, one per channel (red, green, blue):

- **`grey`, the frame's mid grey.** Unit: density (log10, above the film base), in the file's measure (RP 180 or
  APD). It is the density of the chart's 18 % grey step on the frame's bracket, moved by the exposure the search
  found. For Portra 400's bracket Target 6 at an exposure of 1, as APD: 1.23, 1.05, 1.08. The gains the search found
  are the balance, kept apart from it, and the CLF prints `grey - balance` as the print's mid grey: that is the
  density of the film where an 18 % grey object of the scene lies on this negative.
- **`S`, the slopes.** No unit: a ratio, density per density. Each multiplies, in one channel of the print
  material's printing density, the difference between a pixel's density and the mid grey's, so it is a contrast
  scale per channel. Green is 1 by definition, and red and blue are fitted. For the same bracket on Portra Endura:
  0.90, 1.00, 0.97, which lowers the red contrast by 10 % and the blue by 3 % against green.

Stages 1 and 2 write the density file, and nothing after them changes a pixel of it. Stages 5 to 9 are the CLF
(`<output>.<print>.clf`); stage 4 fills its first node. `np_density.cc` holds all of it: the density file
(stages 1 and 2) and the print (stages 4 to 9: the print material, the fit of the slopes and the CLF).

The sections below follow that order: the density file, then the spectral data and the matrices that the stages
rest on, then the arithmetic of each stage, then how to use the CLF.

## The density file

Stages 1 and 2. Each pixel is divided by the film base, its logarithm is taken, and the 3×3 matrix `M` turns the
densities the LED sees into the measure asked for.

| `-P` | Density | In a `.dpx` (10 bits) | In a `.tif` / `.tiff` (16 bits) |
| :--- | :--- | :--- | :--- |
| `statusm` | Status M | Status M at the Cineon scaling | Status M at 8000 codes per unit density |
| `rp180` | SMPTE RP 180 printing density | Cineon | RP 180 at 8000 codes per unit density |
| `apd` | Academy Printing Density (SMPTE ST 2065-2) | ADX10 | ADX16 |

**The encoding.** The output's extension picks the file. A DPX holds `code = 95 + 500 × k × D` and a TIFF
`code = 1520 + 8000 × k × D`. `k` is 1 for `statusm` and `rp180`. For `apd` it is the channel factor of ADX (SMPTE
ST 2065-3), 1.00 for red, 0.92 for green and 0.95 for blue, so a tool that reads the file as ADX gets Academy
Printing Density back. A DPX clips at 1.86 density above the base; for `apd`, with its factors, at 1.86 in red, 2.02
in green and 1.95 in blue. The sidecar counts the pixels that ran off either end of the code range, and those too
dense to measure.

**The zero is the film base.** `--film_base` takes it from a capture of unexposed film. Without it the base is read
from the scan: from the rebate when the frame shows one, else estimated from the thinnest film in the frame, which is
too dark wherever the frame has no unexposed film. `neg_process` warns in that case, and when the base is the
profile's own roll's. `--flat` applies as it does to a picture: the flat field corrects the scan and the film base
capture before the density is taken, so the light's fall-off towards the corners does not read as density.

**Nothing of a picture is applied.** Density takes no bracket, exposure, gain, curve fit, tone stage or look option,
and no white-light scan. The search that balances a picture still runs (stage 3), and what it finds is recorded, not
applied: `suggested_balance` holds one density offset per channel, summing to zero, that a grading tool can add.

**What the file carries.** The matrix that was applied, the LED's spectrum, the film base and the encoding, as JSON:
a TIFF in its `ImageDescription` tag, a DPX in its user-defined data. The sidecar has the same record.

### How good the densities are

The matrices are computed from datasheet spectra and have not been checked against a densitometer. How well a 3×3
reproduces each measure on the film model, as each bundle records it (`fit_rms_density`, `fit_max_density`), across
the three stocks, over the densities a scan reaches (the film's neutral scale between log exposure −1.5 and 1.0,
about 2.2 above the base):

| Measure | RMS error (density) | Worst case (density) |
| :--- | :--- | :--- |
| Status M | up to 0.004 | up to 0.04, in green |
| RP 180 printing density | up to 0.004 in red and green, up to 0.010 in blue | up to 0.10, in blue |
| Academy Printing Density | up to 0.010 in red, 0.002 in green and blue | up to 0.08, in red |

These are errors against the model, not against the film. Three things limit them further:

- **Film data that does not exist.** The datasheets give no base density below about 400 nm or above about 700 nm,
  and the fit holds the base's nearest measured value there. RP 180 blue depends on that base below 400 nm (up to
  6 % of its response lies there), and Academy Printing Density red on the cyan dye above 700 nm, where the film
  data is a reconstruction (20 %, 30 % and 59 % of its response for Portra 400, Portra 160 and Ektar 100; RP 180
  red has 4 %, 7 % and 26 %). Take those rows as approximate (`response_outside_film_data` in the bundle).
- **Each LED band is taken as one channel.** How the sensor's sensitivity varies inside a band is not modelled;
  "The crosstalk correction, measured and computed" sizes what the camera adds, on a model.
- **Only Status M is calibrated on the film.** The grey scale of the profile's chart, shot at seven exposures, is
  held to the Status M density of the film model's neutral scale: one factor on the green and one on the blue
  density the LED sees (`density_matrices.neutral_calibration`; the matrix before the factors is
  `spectral_fit_matrix`). The factors are 1.02 and 1.06 for Portra 400 and 0.96 and 0.97 for Portra 160. Ektar
  100's (0.83 and 1.08) lie more than 0.10 from 1 and are recorded, not applied: its grey scale differs from the
  model's in shape, which no factor mends. A resampling of the seven negatives in a review of this work, not
  recomputed here, puts Portra 400's blue and Portra 160's green about six standard deviations from 1, Portra 400's
  green two, and Portra 160's blue 1.4, so that last factor is not determined by the grey scale. The reference is
  spektrafilm's film model, not the datasheet. The two printing densities have no such reference and are the fit
  from the spectra alone.

## The spectral data

All of it is on 380 to 780 nm at 1 nm. Nothing spectral is read per frame except from the print material's file:
the fit of the slopes (stage 4) and the colour table (stage 8) compute with its curves, its dye spectra and its
observer. "Specific to" says what a change of equipment or material
replaces.

| Data | Specific to | Source | Used by | Kept in |
|---|---|---|---|---|
| The scanning light's spectrum, split into its three LED bands at 495 and 578 nm | The light | A spectrometer reading of the light | A | `bundle.json`, `light_source` |
| The film's three dye spectra and base, and its neutral scale as dye amounts against log exposure | The film stock | spektrafilm's film profile | A, B | Not shipped: used when the profile set and the print material's file are built |
| The response curves of Status M, RP 180 (SMPTE RP 180, Table 1) and APD (SMPTE ST 2065-2) | Nothing: the standards | The standards | A, B | Not shipped: used when the profile set and the print material's file are built |
| The print material's spectral sensitivity per layer | The print material | spektrafilm's print profile | B | Not shipped: used when the print material's file is built |
| The printer lamp | Nothing: a choice | A 3200 K blackbody, no heat filter | B | A constant |
| The print material's curve per layer, its three dye spectra and base | The print material | spektrafilm's print profile | B, then 4, 7 and 8 | `prints/<print>.json` |
| The CIE 1931 2° observer, and D50 (paper) or D55 (print film) as the viewing light | Nothing: the CIE tables and a choice of light | The CIE tables | B, then 4 and 8 | `prints/<print>.json`, as their product |

Of these, `bundle.json` holds the light's spectrum and nothing else spectral: not the film's dye spectra and not
the response curves. It names the film model its matrices were fitted on (`density_matrices.film_dye_spectra`) and
holds the fitted matrices.

**The camera has no spectral data here.** Its response curves are not used. What stands in for them is the crosstalk
correction `C`, which is specific to the camera and the light together: it is measured by capturing each LED alone,
and it turns the camera's RGB into the three LED channels. After it, each channel is taken to be one LED band, and
the density matrices are fitted on the bands, with no camera in them. `C` could instead be computed from the
camera's response curves and the light's spectrum; the next section compares the two.

Measured on the scanner and not spectral, all specific to the camera and the light: the crosstalk correction, the
flat field, the film base (also specific to the roll), and the chart's grey steps at each of the seven exposure
brackets (also specific to the film stock), which are in `bundle.json`.

## The crosstalk correction, measured and computed

Each camera pixel sees all three LEDs: the green pixel takes in nearly half as much of the blue LED as the blue
pixel does. The correction `C` undoes that.

![The camera's response curves and the light's three LEDs](samples/crosstalk_spectra.png)

- **Measured**, as `neg_process` uses it: each LED is captured alone, the response of the three pixels to it is
  read, each LED's column is scaled to its own pixel, and the matrix is inverted. It does not depend on how bright
  each LED is.
- **Computed**: the camera's published response curves (Sony ILCE-7RM4, from the Academy's `rawtoaces-data`) times
  the spectrum of each LED. The light was measured with all three LEDs on, so the three are first separated by
  fitting each as two skewed Gaussians (the sum fits the measurement to 0.27 % of its peak, 1.26 % at worst).

![Crosstalk terms, measured against computed](samples/crosstalk_measured_vs_computed.png)

| Leak | Measured | Computed | Computed / measured |
|---|---|---|---|
| Green pixel, red LED | 0.168 | 0.170 | 1.01 |
| Blue pixel, red LED | 0.017 | 0.016 | 0.95 |
| Red pixel, green LED | 0.089 | 0.079 | 0.89 |
| Blue pixel, green LED | 0.280 | 0.269 | 0.96 |
| Red pixel, blue LED | 0.063 | 0.041 | 0.66 |
| Green pixel, blue LED | 0.457 | 0.421 | 0.92 |

Four of the six leaks agree within 8 %. The two that involve the red pixel do not: the published curves give it 11 %
less of the green LED and 34 % less of the blue LED than this camera shows. The two correction matrices differ by
0.046 at most (0.045 from the three decimals shown):

```
measured                      computed
 1.014  -0.082  -0.027         1.013  -0.077  -0.010
-0.186   1.162  -0.519        -0.186   1.142  -0.474
 0.035  -0.324   1.146         0.034  -0.306   1.128
```

What that does to density, on a model of Portra 400 seen through the published camera curves (3000 sample
colours, light density up to 1.5 above the base). The first two rows are what the camera adds to the assumption the
density matrices make, that each corrected channel is one LED band: the density from the camera's corrected signal
against the density through the LED band alone. The third is the two corrections against each other.

| Light density, red / green / blue | RMS | Largest |
|---|---|---|
| Camera with the measured correction, against the LED bands alone | 0.015 / 0.021 / 0.009 | 0.049 / 0.055 / 0.076 |
| Camera with the computed correction, against the LED bands alone | 0.012 / 0.013 / 0.007 | 0.050 / 0.038 / 0.062 |
| The computed correction against the measured one | 0.003 / 0.009 / 0.002 | 0.008 / 0.044 / 0.014 |

Most of the first two rows is a scale, which RMS and largest do not show: against the LED band alone the model's
camera reads red and green 2 to 3 % flatter and blue 1 % steeper (slopes 0.967, 0.974 and 1.010 with the measured
correction, 0.972, 0.980 and 1.007 with the computed one), and a scale goes through `M` into every density of the
file. The first row pairs the model's camera, which has the published curves, with the correction measured on this
camera, so the second row is the cleaner size of what a camera adds.

Above a density of 1.5 the comparison stops meaning much: under dense film the green and blue pixels' signal is
mostly leakage from the other LEDs, the corrected signal nears zero, and with either matrix it goes below zero on a
few of the samples.

So the measured matrix is kept. It is of this camera and this light, where the published curves are of another body
of the same model and the LED spectra are a fit; and it needs no spectral data of the camera at all. The computed
one would serve a camera whose curves are known and whose light cannot be switched one LED at a time; against the
measured one it differs by up to 0.009 RMS and 0.044 at most, up to a density of 1.5. All of this is on a model:
the camera in it has the published curves, not this camera's.

The camera curves are `data/camera/Sony_ILCE-7RM4_380_780_5.json` of AcademySoftwareFoundation/rawtoaces-data at its
commit `28801d1ad6` (Apache-2.0). Each LED is fitted as the sum of two Gaussians with separate widths on either
side, the second one free to sit up to 25 nm from the first; the film model is spektrafilm's Portra 400. The
script that computes the tables and draws the two figures, and its output, are kept with the film model and the
light's spectrum, beside the tools that build the profile sets, outside this repository.

## The matrices

Densities are above the film base. "Fitted on the film model" means: 3000 sample colours of the film are drawn, dye
amounts along its neutral scale between log exposure −1.5 and 1.0 with a log-normal colour variation; the spectral
density of each sample is read with two sets of weights; the 3×3 is the least-squares fit from the first reading to
the second, with no offset.

| Symbol | From, to | How it is computed | Kept in |
|---|---|---|---|
| `C` | Camera RGB to the LED's channels | The inverse of the camera's response to each LED alone, each column scaled to its own channel | `bundle.json`, `crosstalk_profile` |
| `B` | Diagonal: the film base to one level | The base level over the film base, per channel | Per frame |
| `M` | LED density to Status M, RP 180 or APD | Fitted on the film model: the reading through each LED band, to the reading through the measure's response curves. Status M is then calibrated on the chart's grey steps | `bundle.json`, `density_matrices` |
| `k` | Diagonal: the ADX channel factors 1.00, 0.92, 0.95 | SMPTE ST 2065-3. 1 for Status M and RP 180 | The file's encoding |
| `P` | RP 180 or APD to the print material's own printing density | Fitted on the film model: the reading through the measure's response curves, to the reading through the print material's sensitivity under the printer lamp. One per film model, and one over all film models for any other film | `prints/<print>.json`, `to_print_density` |
| `S` | Diagonal: the slopes on red and blue; green is 1 | Fitted per frame, below | The file's metadata, `print` |
| `X` | The print's XYZ under its viewing light to linear sRGB | The sRGB matrix times a CAT02 adaptation from the viewing light's white to D65 | `prints/<print>.json`, `xyz_to_linear_srgb` |

For Portra 400, `M` for APD is

```
 1.488  -0.074   0.013
-0.052   1.113  -0.045
 0.018  -0.029   1.009
```

and `P` from APD to Kodak Portra Endura is

```
 0.896   0.058   0.013
 0.012   0.918   0.065
 0.025   0.055   0.876
```

`P` differs between the three stocks by at most 0.03 for APD and 0.12 for RP 180 on Portra Endura, and by 0.02 and
0.09 on Vision 2383.

## What is multiplied, in order

The arithmetic of each stage, with the symbols of "The matrices".

**Per pixel**, the density file. `v` is the camera RGB after the flat field:

```
T        = B C v / base level                 transmittance relative to bare film, per LED channel
D_light  = -log10(max(T, 0.0001))
D        = M D_light                          the measure's density
code     = offset + scale k D                 10 bits: 95 + 500 k D     16 bits: 1520 + 8000 k D
```

**Per frame.** The search gives the bracket and `E`, `g`, `b`. The profile set gives the bracket's grey steps as the
profile read them, `r_i`, at the bracket's exposure:

```
d_i      = -log10(r_i (base exposure / bracket exposure) / profile film base)      light density of step i
balance  = M [0, -log10 g, -log10 b]          what the gains do to the density, level included
grey     = M d_mid + log10(E) M [1, 1, 1]     the frame's mid grey; d_mid is the chart's mid grey step
pd_i     = P M d_i                            the steps as the print material reads them
```

**The slopes.** A film whose channels differ in contrast prints a grey scale that drifts in colour from shadows to
highlights, and no printer-light setting removes that. `S` is fitted on `pd_i`: the steps are printed about the
bracket's mid grey, through nodes 2 to 6 below with `S (pd_i - pd_mid)` as the density difference and one
printer-light offset on red and one on blue. `S` and the two offsets minimise the CIELAB a\* and b\* of the steps.
The offsets belong to the fit and are not used afterwards.

| Choice | Value | What it does |
|---|---|---|
| Steps judged | Those that print between L\* 12 and 92 before the slopes | A step printed near black or near white has no colour to judge. |
| Fewest steps | 12 | With fewer the slopes stay 1. |
| Bounds of a slope | 0.6 to 1.5 | A slope that ends on a bound is recorded as such. |
| Mid grey step | `gs10` | The chart's step nearest 18 % reflectance. |

**The CLF**, on the file's code value `x` (the code over the largest code):

```
node 1    pd    = 0.70 + S P (D + balance - grey),    D = (x largest code - offset) / (scale k)
node 2    logE  = aim - (pd - 0.70)                   aim: the exposure per layer that prints an 18 % grey
node 3-4  a_c   = curve_c(logE_c)                     the print's curve per layer, held at the ends of its range
node 5-6  t(w)  = 10^-(sum_c a_c dye_c(w) + base(w))  the print's transmittance or reflectance
          XYZ   = sum_w t(w) light(w) observer(w) / sum_w light(w) y(w)
          rgb   = X XYZ                               the 3D table holds rgb on a grid of (a_1, a_2, a_3)
node 7    the sRGB curve
```

Node 1 is one 3×4 matrix. Its 3×3 part is `S P diag(largest code / (scale k))` and its offsets are
`0.70 + S P (balance - grey - offset / (scale k))`. So between a camera pixel and the print's exposure the matrices
multiplied are, in order, `C`, `B`, the logarithm, `M`, `k` into the file and out of it again, `P` and `S`; after the
print's curves comes `X`. The frame's mid grey, balanced, lands on the printing density 0.70, code 445 of a 10-bit Cineon file, where
Cineon holds the LAD patch (an 18 % grey card is at 470). The print exposes that density as an 18 % grey. Both are
choices: a Cineon scan has an 18 % card at 0.75, and a print to the LAD aim, a visual density of 1.0, takes about
a third of a stop more exposure (0.10 less on each of the three offsets of the first Matrix); its grey has Y = 0.10
where this one has 0.18. Inside the CLF the constant
cancels, so the picture does not depend on it; it matters to whoever takes the output of node 1 to a print LUT made
for Cineon densities.

## Using the CLF

The density file holds none of the balance, the exposure or the slopes; its metadata records what was computed for
the frame (`print`), and the CLF applies it. Apply them once: through the CLF, or by hand from the metadata, not
both. To print lighter or darker, add a constant to the three offsets of the first matrix (0.30 is a stop). A CLF
fits the one file it is named after, `<output>.<print>.clf`: its first matrix holds that file's code scale, measure
and channel factors.

## What depends on what

| | The frame | The film stock | The print material | The scanner and its light |
|---|---|---|---|---|
| `C`, the flat field | | | | yes |
| `B` | yes: the roll's film base | | | |
| `M` | | yes: its dye spectra | | yes: the light's spectrum |
| `P` | | a little: its dye spectra | yes | |
| balance, `grey` | yes: the search | yes: the chart's mid grey per bracket | | |
| `S` | through its bracket | yes: the chart's grey steps | yes | |
| CLF nodes 2 to 7 | | | yes | |

## What is not principled yet

- **The densities are a model's.** `M` comes from datasheet spectra and has not been checked against a
  densitometer; "How good the densities are" lists what limits it.

- **The print material is a model.** Its curves, dye spectra and sensitivity are spektrafilm's, the printer lamp is
  a 3200 K blackbody with no heat filter, and no result has been compared with a real print.
- **The slopes correct the print material too.** They make the grey steps neutral on the print, so they take out the
  print material's own grey scale along with the film's.
- **The slopes change between brackets.** Each bracket's are fitted on its own 24 steps, and frames on neighbouring
  brackets are printed with different slopes. One fit per stock, or slopes interpolated at the frame's own mid grey,
  have not been tried on pictures.
- **A 3×3 for each density measure.** `M` and `P` are least-squares fits on a model of the film. Where the film's
  data ends (below about 400 nm, above about 700 nm) the nearest measured value is held.
- **The values in the table of the slopes** (L\* 12 and 92, 12 steps, 0.6 and 1.5) are stated, not derived.
