# Portra400-A7R4-D65

Profiles for Kodak Portra 400 scanned with Sony ILCE-7RM4 (Linear Rec.2020) under a white (D65) light, for `neg_process --film_stock portra400 --light_source d65`.

How they were built: an IT8 target (R190808) was photographed on this film stock at 7 exposures. Each developed
frame (a bracket) was scanned with this camera and light, and its patches were matched to the target's reference
colours to make one ICC profile per bracket (`Target_1.icc` to `Target_7.icc`). Before matching, the values were
taken to linear Rec.2020 through the camera's daylight white balance and colour matrix and scaled to the film base
(the unexposed film), which scans are normalised to as well. The captures were flat-field corrected.

The bundle's `input_space` is `rec2020_linear`: `neg_process` converts a scan through LibRaw's white-light decode
(the camera's daylight white balance and LibRaw's colour matrix, to linear Rec.2020) and
applies no crosstalk correction, so `-r`, `-g` and `-b` cannot be given with it. The camera matrix the patches were
measured through is kept under `profiling_input_transform` as a record only; nothing applies it.

`bundle.json` holds the measurements, the film base and an `export` record of the source file.

## Accuracy

How closely each bracket's profile reproduces the IT8 chart, as CIEDE2000 colour differences (ΔE₀₀; under 2 is
barely visible).

| Bracket | Profile | Fitted: mean | Fitted: median | Held out: mean | Held out: median |
| :--- | :--- | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1.08 | 0.71 | 2.65 | 2.08 |
| −2 EV | `Target_2.icc` | 0.74 | 0.53 | 1.95 | 1.63 |
| −1 EV | `Target_3.icc` | 0.69 | 0.53 | 2.05 | 1.49 |
| 0 EV | `Target_4.icc` | 0.75 | 0.50 | 2.10 | 1.42 |
| +1 EV | `Target_5.icc` | 0.68 | 0.47 | 1.80 | 1.30 |
| +2 EV | `Target_6.icc` | 0.73 | 0.50 | 2.21 | 1.42 |
| +3 EV | `Target_7.icc` | 0.80 | 0.57 | 2.44 | 1.53 |

**Fitted** is ArgyllCMS `profcheck` on all 288 chart patches the profile was built from (the output is stored in
`bundle.json`). It shows how well the profile fits its own data, so it flatters the profile. **Held out** is the
error on patches the profile never saw: each bracket's profile was rebuilt twice, each time leaving out alternate
halves of the 264 colour patches, and checked on the half it left out. The 24 grey patches, which set the tone curves, stay in every build.
That is the better guide to the error on colours in a real photograph. The profiles are not de-glared: removing the
chart's veiling glare before fitting was tried, did not help overall, and the bundles ship as measured.

## Dynamic range

Range across each bracket's 24 grey patches (`gs0` to `gs23`) before the camera matrix (`bundle.json`'s patch values
multiplied by the inverse of `profiling_input_transform.matrix` to recover raw sensor RGB), as 14-bit ADC bit depth
(`log2(max - min)` out of `16384`) and contrast in stops (`log2(max / min)`).

| Bracket | Profile | Shutter | Bit depth: R / G / B | Bit depth: mean | Stops: R / G / B | Stops: mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1/15s | 12.55 / 13.19 / 12.11 | 12.62 | 1.49 / 1.88 / 2.09 | 1.82 |
| −2 EV | `Target_2.icc` | 1/15s | 12.66 / 13.27 / 12.17 | 12.70 | 1.77 / 2.25 / 2.51 | 2.18 |
| −1 EV | `Target_3.icc` | 1/15s | 12.76 / 13.38 / 12.28 | 12.81 | 2.02 / 2.56 / 2.82 | 2.47 |
| 0 EV | `Target_4.icc` | 1/10s | 13.30 / 13.76 / 12.57 | 13.21 | 2.25 / 2.74 / 3.00 | 2.66 |
| +1 EV | `Target_5.icc` | 1/8s | 13.43 / 13.75 / 12.54 | 13.24 | 2.38 / 2.83 / 3.16 | 2.79 |
| +2 EV | `Target_6.icc` | 1/5s | 13.63 / 13.80 / 12.60 | 13.34 | 2.29 / 2.68 / 3.07 | 2.68 |
| +3 EV | `Target_7.icc` | 1/4s | 13.50 / 13.65 / 12.34 | 13.16 | 2.22 / 2.64 / 2.96 | 2.61 |
| **Mean** | — | — | **13.12 / 13.54 / 12.38** | **13.01** | **2.06 / 2.51 / 2.80** | **2.46** |
