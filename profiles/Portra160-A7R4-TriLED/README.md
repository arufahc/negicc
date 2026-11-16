# Portra160-A7R4-TriLED

Profiles for Kodak Portra 160 scanned with ILCE-7RM4 under a narrowband red, green and blue LED light, for `neg_process --film_stock portra160 --light_source tri`.

How they were built: an IT8 target (R190808) was photographed on this film stock at 7 exposures. Each developed
frame (a bracket) was scanned with this camera and light, and its patches were matched to the target's reference
colours to make one ICC profile per bracket (`Target_1.icc` to `Target_7.icc`). Before matching, the camera's
channel crosstalk was removed and the values were scaled to the film base (the unexposed film), which scans are
normalised to as well. The captures were flat-field corrected.

`bundle.json` holds the measurements, the crosstalk matrix, the film base and an `export` record of the source file.

## Accuracy

How closely each bracket's profile reproduces the IT8 chart, as CIEDE2000 colour differences (ΔE₀₀; under 2 is
barely visible).

| Bracket | Profile | Fitted: mean | Fitted: median | Held out: mean | Held out: median |
| :--- | :--- | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1.09 | 0.47 | 2.35 | 1.61 |
| −2 EV | `Target_2.icc` | 0.82 | 0.42 | 2.09 | 1.55 |
| −1 EV | `Target_3.icc` | 0.59 | 0.39 | 1.96 | 1.40 |
| 0 EV | `Target_4.icc` | 0.56 | 0.44 | 2.30 | 1.37 |
| +1 EV | `Target_5.icc` | 0.62 | 0.43 | 2.65 | 1.58 |
| +2 EV | `Target_6.icc` | 0.67 | 0.49 | 2.67 | 1.62 |
| +3 EV | `Target_7.icc` | 0.70 | 0.53 | 2.82 | 1.78 |

**Fitted** is ArgyllCMS `profcheck` on all 288 chart patches the profile was built from (the output is stored in
`bundle.json`). It shows how well the profile fits its own data, so it flatters the profile. **Held out** is the
error on patches the profile never saw: each bracket's profile was rebuilt twice, each time leaving out alternate
halves of the 264 colour patches, and checked on the half it left out. The 24 grey patches, which set the tone curves, stay in every build.
That is the better guide to the error on colours in a real photograph. The profiles are not de-glared: removing the
chart's veiling glare before fitting was tried, did not help overall, and the bundles ship as measured.

## Dynamic range

Range across each bracket's 24 grey patches (`gs0` to `gs23`) before crosstalk correction (`bundle.json`'s patch values
multiplied by the inverse of `crosstalk_correction_matrix`), as 14-bit ADC bit depth (`log2(max - min)` out of `16384`)
and contrast in stops (`log2(max / min)`).

| Bracket | Profile | Shutter | Bit depth: R / G / B | Bit depth: mean | Stops: R / G / B | Stops: mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1/8s | 12.78 / 13.23 / 13.10 | 13.04 | 1.29 / 1.75 / 1.87 | 1.63 |
| −2 EV | `Target_2.icc` | 1/8s | 12.97 / 13.36 / 13.21 | 13.18 | 1.63 / 2.17 / 2.32 | 2.04 |
| −1 EV | `Target_3.icc` | 1/8s | 13.07 / 13.38 / 13.22 | 13.22 | 1.93 / 2.52 / 2.73 | 2.40 |
| 0 EV | `Target_4.icc` | 1/6s | 13.45 / 13.71 / 13.48 | 13.55 | 2.29 / 2.96 / 3.18 | 2.81 |
| +1 EV | `Target_5.icc` | 1/5s | 13.72 / 13.78 / 13.55 | 13.68 | 2.63 / 3.20 / 3.41 | 3.08 |
| +2 EV | `Target_6.icc` | 1/4s | 13.74 / 13.64 / 13.40 | 13.59 | 2.71 / 3.26 / 3.49 | 3.15 |
| +3 EV | `Target_7.icc` | 1/3s | 13.63 / 13.46 / 13.22 | 13.44 | 2.64 / 3.13 / 3.35 | 3.04 |
| **Mean** | — | — | **13.34 / 13.51 / 13.31** | **13.39** | **2.16 / 2.71 / 2.91** | **2.59** |
