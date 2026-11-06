# White-light conversion with a film model on the camera's own channels

Experiment report, 2026-11-06. Code: `white_film.h`, `white_film.cc` (`neg_process --white_model film`);
constants: `tools/white_film/fit_calibration.py`. This report is written so that each step of the code maps to one
numbered step below, and each constant to the experiment that produced it.

## 1. Result in short

- On white-light scans from other people's cameras and lights, today's path (film profile in linear Rec.2020, with
  exposure and balance from the intent model) fails on whole rolls: blue shadows, yellow or cyan highlights, crushed
  contrast. The first cause sits before the profile: the camera-to-Rec.2020 matrix drives blue below zero in the
  dense, very orange parts of some negatives, and the frame clips it (section 3).
- The new path uses the camera's raw channels and a three-line film model with one fixed matrix, fitted on the
  existing IT8 calibration readings. The intent model still chooses exposure and balance, and it adjusts the green
  and blue exponents half-way. No profile, no brackets, no curves, no roll-off (section 4).
- Judged frame by frame against today's best mode (`--fit_curves auto`) on 137 scans from 30 camera models:

  | set | new path better | same | today's curves better | not comparable |
  |---|---|---|---|---|
  | development, 106 frames | 35 | 58 | 5 | 8 |
  | held out (never used for a decision), 31 frames | 13 | 15 | 2 | 1 |

  The intent model's own colour score (lower is closer to what it expects; section 2 says how far it can be trusted),
  medians for today's default, today's curves mode and the new path: 3.37, 1.78, 1.81 on the development frames and
  2.15, 1.47, 1.41 on the held-out frames. On the frames Alpha rated, the new path is closer in colour to the version
  that was preferred on 29 of 53 verdicts, the same count as today's curves mode (29) and above the default (19).
- Two detector changes came out of the same frames (section 8): 7 of the 137 crops change, all for the better, and the
  crop evaluation set does not regress.
- What is not solved: section 9.

## 2. Data and how results were judged

**Frames.** The 137 white-light colour negatives of the corpus catalogue (`Film Negatives/meta/catalog.jsonl`,
`light_source d65`): the 41 frames of the blind sessions of Iterations 229 and 231, and 96 others (86 added on
2026-11-05), from 30 camera models (Canon, Nikon, Sony, Fujifilm, Olympus, Panasonic, two film scanners). Frames are
grouped by uploader folder, a proxy for one camera, light and usually one roll.

**Split.** Groups holding a voted frame are development frames. The other groups were sorted by size and alternately
assigned to development and held-out, so that both hold a large foreign roll. After the first look at the held-out
set (Iteration 4) one folder (Canon R6 II, 11 frames) and one single frame were declared seen and moved to
development. The remaining 31 held-out frames were looked at once, with the final algorithm.

**Orientation.** The corpus records no orientation for the 96 frames outside the blind sessions. It was set by eye
for each of them (56 turned, 12 of two rolls mirrored, where edge print or shop signs read mirrored) and every final
run, master's included, renders upright: the intent model reads the picture's content.

**Judges, and what each is worth.**
1. *Alpha's blind votes* (Iterations 229, 231): 53 directional verdicts between Auto and fitted curves on the 41
   frames, with the previews that were rated. They show what was preferred: curves on frames with a cast (9:4:2),
   no preference elsewhere, and on four frames Auto in both sessions because the curves had dulled real colours
   (blue sky, foliage, skin).
2. *The intent model's residual*: the mean squared z-score of a render against the field the model predicts from that
   render. It prefers the voted version on 32 of the 53 verdicts (60%); its a\*b\* part on 35 (66%); answering
   "curves" every time scores 33. So a lower residual is evidence, not a verdict, and a method that fits more numbers
   against it will score lower without being better. It is reported per setup, never alone.
3. *The eye*: every frame of both sets, the new path beside today's curves mode (the stronger of today's two
   modes). The full pass was made on the prototype's renders, as scanned; the frames whose verdict could change
   (every "curves better", every "not comparable", every changed crop) were looked at again on the final build,
   upright. Verdicts are recorded per frame in the lab (`runs/w7a/review.json`, amended in `LEDGER.md`).
4. *Calibration targets*: the IT8 readings of the three A7R4 white-light bundles, for the form of the film model
   only (section 5, Iteration 1).

## 3. What was wrong

- **Blue clipped to zero.** In the linear Rec.2020 frame of scan 1cc48b9a44 (Canon R5) blue is exactly 0 on 1.15% of
  the pixels and 8 counts at the 5th percentile, while red is 650. The raw sensor blue there is 22 to 32 counts above
  black. The matrix's blue row has negative red and green terms; on a dense orange negative they exceed the blue
  signal. The density information of the highlights is gone before any profile sees it. Working in Rec.2020 with the
  new film model as well turns the same skies saturated magenta (Iteration 4).
- **One profile for every setup.** The profile's tone curves and its 45x45x45 table are those of Portra 400 on the
  A7R4 under one LED. On the IT8 readings, taking another stock through Portra 400's model costs 7 to 10 dE00 at the
  median; giving it its own per-channel exponent and balance brings that back to 3.6 and 5.3 (Iteration 1). Red's
  curve is fixed in today's path, so overall contrast cannot adapt either.
- **The intent score as the only guide.** Fitted against it, curves win where a cast exists and otherwise move scene
  colours toward the model's average. The new path gives the per-frame fit two numbers for slope and takes them
  half-way.

## 4. The algorithm

Input: the scan, the frame crop and the film-base patch that `neg_process` already finds.

1. **Camera channels.** Decode to the camera's linear RGB: black subtracted, no white balance, no colour matrix.
2. **Film base.** The median of the base patch in those channels (without a patch: the 99.9th percentile of the
   frame).
3. **Film model**, per pixel and channel c:

       d_c = max(log2(base_c / v_c), 0)        stops below the film base
       x_c = w_c * (2^(p_c * d_c) - 1)         exponent p_c, balance w_c; -1 is the base's own share (the toe)
       XYZ = M x                               one fixed matrix
       out = sRGB(tone(e * XYZ))               exposure e; tone(s) = s^c / (s^c + 0.18^c (1-g)/g), c = 1.4, g = 0.18

   Six numbers describe a frame: three exponents, two balances (w_R is 1), one exposure.
4. **Starting values** from the calibration: exponents `P_CAL = (2.33, 1.90, 1.71)`; the balance starts at 1 (the
   calibration's own balance is inside `M`).
5. **Exponent ratios from the frame's own densities.** The ratio of the channels' density ranges (2nd to 98th
   percentile over the picture cells) against the calibration's, to the power 0.5 for green and 0.6 for blue.
6. **Exposure and balance from the intent model.** Up to four rounds: render, predict the field, then (a) balance by
   coordinate descent on the a\*b\* z-scores with the exposure re-metered so that the median L\* holds, and (b) the
   exposure that minimises the L\* z-scores.
7. **Contrast.** The exponents are scaled half-way toward a slope of 0.9 of rendered L\* against the field's L\*
   (limits 0.75 to 1.2); step 6 runs again for two rounds.
8. **Slope.** The green and blue exponents are fitted against the a\*b\* z-scores with a pivot (a channel keeps its
   value at the frame's median density, so only slope is judged), limited to 0.75-1.33 and taken half-way; step 6
   runs again for two rounds.
9. **Lightness offset.** The exposure is lowered by 0.7 stop.

"Picture cells" are cells at least 0.1 stop below the base in some channel; rebate and bare light inside the crop do
not take part in any statistic.

### Where each constant comes from

| constant | value | from |
|---|---|---|
| exponents `P_CAL`, matrix `M` | see code | fit on the Portra 400 IT8 readings, 7 brackets (Iteration 1; `fit_calibration.py`) |
| toe term | -1 | Iteration 1: no parameter, cuts the 90th-percentile error from 9.8 to 7.3 dE00 |
| ratio weights | 0.5 (G), 0.6 (B) | Iteration 3: variance of the range ratio within a setup against between setups |
| contrast target and step | 0.9, half-way, limits 0.75-1.2 | Iteration 2-3: slope separates setups (Canon R5 1.06-1.16, Fuji 0.81-0.94, ZV-E1 0.61-0.78) |
| slope step | half-way, limits 0.75-1.33 | Iterations 4 and 6: one frame scatters by 0.14-0.40 (log2) within a setup; the full fit is what the votes did not prefer on cast-free frames |
| lightness offset | 0.7 stop | Iteration 2: the intent optimum is 7.5 to 12.5 L\* brighter than the rated renders (correlation 0.87) |
| tone curve c | 1.4 | set by eye in Iteration 2 and not tuned further |
| picture cell | 0.1 stop | Iteration 3 |

## 5. Experiments

Every run and number is in the lab's ledger (`whitelab/LEDGER.md`, not in this repository). In order:

| # | question | what was run | result |
|---|---|---|---|
| 0 | How does master do on all 137 frames? | Auto and `--fit_curves auto` | 0 render failures. Whole foreign rolls with crossover and harsh contrast (Canon R5, Canon R6 II), a dark muddy roll (ZV-E1), crop failures on 35mm strips. Curves reduce casts but cannot change contrast. |
| - | Can the intent score judge? | residuals of the rated previews against the 53 verdicts | 60% (all), 66% (a\*b\*), 62% for a constant answer. Lower is not reliably better. |
| - | Can scene statistics calibrate a frame? | main axis of the log-RGB cloud of 1497 lab scans; tone slope of a\*, b\* on 2000 | Main-axis slopes scatter (G 0.89-1.14, B 0.92-1.37 quartiles): too noisy for one frame. |
| 1 | Which film model? | forms with and without toe, density mix, curvature, on the IT8 readings of 3 stocks x 7 brackets | Toe form: 3.66 dE00 median on its own stock; 3.55 and 5.27 on the other two once exponents and balance are theirs (7.4 and 10.1 with Portra 400's). A density-domain mix gains 0.6; curvature nothing. One curve per channel spans -3 to +3 EV. |
| 2 | Does it work on real scans? | prototype W1: calibration exponents, intent balance and exposure | Natural tones on the rolls master fails on. Density-range ratios between camera channels are nearly the same on every camera make (R/G 0.85, R/B 0.72; calibration 0.82, 0.73). Fixed lightness targets miss the rated lightness by 11-13 L\*; the intent exposure correlates 0.87 with it, one stop brighter. |
| 3 | Contrast and ratios | W2-W4: picture cells, contrast step, shrunk range ratios, camera normalisation | Contrast scale is consistent within a setup (Canon R5 0.85-0.93). A mapping of each camera's channels to the calibration camera's (LibRaw matrices) looked right on Fuji. |
| 4 | Held-out, first look (W4) | 2 of 9 pages | Failed on the Canon R6 II folder: lilac highlights, violet skies. That folder became a development group. A pivoted intent fit of the exponents is too noisy per frame (0.74 to 1.69 within one roll) but its roll medians point the right way. |
| 5 | Why? | intent score per setup for W1-W4 | The camera normalisation made every Canon setup worse (R5 3.35 to 5.93). The camera matrices are fitted to scene colours, not to film dyes under an LED. |
| 6 | W7 | no camera normalisation; slope step at half and full strength | Canon R6 II 1.33 (W4 3.38, master curves 3.00); all frames 1.66 at half strength (master curves 1.83). Half strength chosen. |
| 7 | Held-out, final (W7a) | the 31 unseen frames | Table in section 1. The held-out Canon R5 Portra 400 roll: master dark, purple and crushed on all 7 frames; the new path natural on all 7. |

### Tried and rejected

| idea | why it was dropped |
|---|---|
| Rec.2020 as the working space of the film model | fits the IT8 readings slightly better (3.34) but clips blue on the Canon rolls: magenta skies |
| camera normalisation through LibRaw's camera matrices | worse on every Canon setup; the cause of the first held-out failure |
| density-domain mix before the exponents | 0.6 dE00 on the calibration, no gain on the Canon skies it was tried for |
| per-channel curvature | no gain across stocks |
| grey-world balance | magenta on scenes dominated by foliage |
| the calibration's own balance on every frame | magenta on other cameras: balance has to be found per frame |
| the L\* z-score as a contrast selector | falls monotonically with more contrast |
| exposure from a fixed median or 95th-percentile L\* | 11-13 L\* from the rated lightness on average |
| per-frame exponent ratios from the intent fit at full strength | scatter of 0.14-0.40 (log2) within a roll |
| a stray-light term | the opaque holder reads 0.05% of the base on the Canon R6 II scans: no stray light to remove there |

### A result that points beyond one frame

Estimates of a setup's exponents scatter from frame to frame but their medians over a roll are stable and in the
direction the eye confirms (leave-one-out roll medians cleared the Canon highlights before the true cause was found).
A roll-level calibration (`--white_slopes fG,fB` carries one into `neg_process`) is the natural next step for the
station, which scans rolls.

## 6. Results by setup

Final build (`neg_process --white_model film`, with the crop changes of section 8), all frames upright. Median a\*b\*
intent residual:

| setup (frames) | master Auto | master curves | new path |
|---|---|---|---|
| development, all (106) | 3.37 | 1.78 | 1.81 |
| Canon R5 folder (15) | 4.98 | 3.16 | 2.87 |
| Fuji X-T2 (11) | 3.70 | 1.60 | 2.86 |
| Canon R6 II folder (11) | 8.10 | 3.11 | 1.15 |
| Sony ZV-E1 (10) | 1.64 | 1.13 | 1.04 |
| Nikon Z6 II (5) | 1.46 | 0.70 | 1.04 |
| Canon EOS R (4) | 2.05 | 0.87 | 1.25 |
| Canon T3i (4) | 4.27 | 3.72 | 3.28 |
| Sony A7 II, one night frame four times (4) | 5.59 | 3.09 | 8.86 |
| singles and pairs (36) | 1.89 | 1.08 | 1.60 |
| held out, all (31) | 2.15 | 1.47 | 1.41 |
| held out: Canon R5 Portra 400 roll (7) | 3.95 | 2.21 | 1.41 |
| held out: Sony A7 (5) | 2.06 | 1.29 | 0.87 |
| held out: singles and pairs (13) | 2.41 | 1.63 | 1.80 |

The new path is lower than master's curves on 37 development frames, higher on 41, within 0.3 on 28; held out: 8, 11
and 12.

Against the rated previews of the 41 voted frames: median a\*b\* distance 5.2 to the rated Auto and 3.1 to the rated
curves version (those two are 3.2 apart); closer to the preferred version on 29 of 53 verdicts (25 of the 33 for
curves, 4 of the 20 for Auto); median L50 1.2 L\* below the rated Auto, mean absolute difference 5.9 L\* (master: 1.1,
being the algorithm that made the rated renders).

Does each step earn its place? Without the density-range ratios (step 5) the development median rises from 1.77 to
2.12 (prototype, upright), higher on 24 frames and lower on 5. Without the slope step it is 1.96 against 1.66
(Iteration 6). The contrast step is what brings the Canon R5 folder from "harsh" to natural (scale 0.85-0.93 on all
15 frames).

## 7. Where the new path is worse than master's curves mode

- df43650632: a cast frame that the curves neutralise; the new path leaves a pink-to-blue sky and a cyan building.
- fafe459624 and the fog scenes 2e2cae59c6, fe052ac1bb: darker and cooler than master. Their contrast scale is 0.98
  to 1.03, so this is the exposure: the fixed 0.7 stop offset is too much for high-key scenes.
- 5813cc06f1 (a scanner TIFF): yellow highlights.
- Held out: aadaaf5ad4 (much too dark, purple shadows) and e74e8d948f (lilac sky, darker).
- The Sony A7 II night frame (four captures): the film-base patch is bare light, a known fault of the base detector;
  both paths are wrong, and the score of the new path is the worse of the two.
- Fuji X-T2 frames score higher on the intent residual (2.86 against 1.60) although they look equal or better.
- Lightness differs from the rated renders by 6 L\* on average, in both directions.

## 8. Crop

The same 96 new frames were used to look at the frame detector (53 development frames looked at; the held-out frames
only after the changes). An independent measure classed the cells inside each crop as picture, film base, bare light or
holder; every overlay was looked at. 14 of the 53 crops were wrong. Two causes were shared by several frames:

| change | cause | effect |
|---|---|---|
| Keep 35mm perforation rows out of the frame band | a row through the sprocket holes is still 58% picture, so the band took both hole rows and the hole search, which looks outside the band, found nothing | two frames went from the whole strip (read as 645) to the 35mm frame; held out, one more strip frame (7e318af49b) |
| Stop the frame's transverse edges at a full row of holder or light | stray light on the holder reads as picture and the image border counts as a clean edge | three frames lose a 140-180 px holder band; held out, ce34e25826 goes from the whole holder to the frame |

Of the 137 crops 7 change (5 development, 2 held out), all for the better by eye. On the crop evaluation set
(405 truth frames with the TriLED rolls, 13 excluded): given format 387 pass, auto crops 369, auto formats 354, all
unchanged; no frame goes from pass to fail; one frame moves (56f30d06a9, 0.9995 to 0.9306, still passing: the new
crop ends at the picture where the truth box includes the sprocket-hole row).

Still wrong, each a single picture or needing new detection: a night frame whose shadows sit at base density (four
captures), two 35mm strips whose holes are cut by the image border, one 645 reading laid across a visible gap, one
scanner output, one masked picture.

## 9. Open

- Exposure on high-key and low-contrast scenes; aadaaf5ad4.
- Borders: where the crop holds rebate, holder edge or bare light, the new path shows a coloured edge.
- The film-base detector's bare-light case.
- Roll-level calibration of the exponents: the lab shows roll medians are stable; `--white_slopes` only carries the
  numbers in.
- One film stock's constants. Portra 160 and Ektar 100 would give exponents within 8% and another starting balance;
  the search absorbs the balance and the slope step part of the exponents.
- Peak memory and time were measured on x86 only (61 MP frame: 8.1 s and 1.28 GB against 9.1 s and 1.32 GB for the
  default path); the Orin Nano is not measured.
- The default is unchanged: `--white_model profile`. Making `film` the default is a decision for after the gallery.

## 10. The port, and how to reproduce

- `white_film.cc` against the Python prototype on 14 frames of 9 camera models: median dE00 of the output 0.30 to 1.15
  (about half of it from the decode: the prototype uses another LibRaw build), parameters within a few percent, the
  same base source on all 14. The default path is pixel-identical to master on the two frames checked; the sidecar's
  `cli_equivalent` replays the film path exactly. The search takes its steps on thresholds, so a change in the last
  digits of a constant can move a frame by up to 2 dE (median 0.0, 95th percentile 0.8 over the 137 frames when the
  starting balance was folded into `M`); two runs of one build are identical.
- Constants: `python3 tools/white_film/fit_calibration.py --ref R190808_ref.json` prints `P_CAL` and `M` as
  the code holds them (the chart reference is the vendor's file, not in this repository).
- A conversion: `neg_process --film_stock portra400 --light_source d65 -P srgb --white_model film --roi auto
  --film_format auto [--rot 90] -o out.tiff scan.raw`. The sidecar's `white_film` block holds the six numbers, the
  factors of steps 5, 7 and 8 and the search log.
