# Portra400-A7R4-TriLED

Profiles for Kodak Portra 400 scanned with ILCE-7RM4 under a narrowband red, green and blue LED light, for `neg_process --film_stock portra400 --light_source tri`.

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
| −3 EV | `Target_1.icc` | 0.79 | 0.50 | 2.70 | 1.91 |
| −2 EV | `Target_2.icc` | 0.56 | 0.41 | 2.04 | 1.38 |
| −1 EV | `Target_3.icc` | 0.53 | 0.40 | 2.38 | 1.40 |
| 0 EV | `Target_4.icc` | 0.55 | 0.42 | 2.62 | 1.48 |
| +1 EV | `Target_5.icc` | 0.54 | 0.40 | 2.75 | 1.46 |
| +2 EV | `Target_6.icc` | 0.59 | 0.43 | 2.92 | 1.58 |
| +3 EV | `Target_7.icc` | 0.69 | 0.54 | 3.08 | 1.77 |

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
| −3 EV | `Target_1.icc` | 1/8s | 13.08 / 13.26 / 13.11 | 13.15 | 1.67 / 2.03 / 2.17 | 1.96 |
| −2 EV | `Target_2.icc` | 1/8s | 13.16 / 13.33 / 13.16 | 13.22 | 1.98 / 2.44 / 2.60 | 2.34 |
| −1 EV | `Target_3.icc` | 1/8s | 13.23 / 13.44 / 13.28 | 13.31 | 2.25 / 2.76 / 2.93 | 2.65 |
| 0 EV | `Target_4.icc` | 1/6s | 13.44 / 13.45 / 13.21 | 13.37 | 2.55 / 2.97 / 3.11 | 2.88 |
| +1 EV | `Target_5.icc` | 1/5s | 13.52 / 13.43 / 13.18 | 13.38 | 2.73 / 3.09 / 3.27 | 3.03 |
| +2 EV | `Target_6.icc` | 1/3s | 13.66 / 13.53 / 13.23 | 13.47 | 2.62 / 3.01 / 3.18 | 2.93 |
| +3 EV | `Target_7.icc` | 0.4s | 13.44 / 13.25 / 12.93 | 13.21 | 2.52 / 2.89 / 3.04 | 2.82 |
| **Mean** | — | — | **13.36 / 13.38 / 13.16** | **13.30** | **2.33 / 2.74 / 2.90** | **2.66** |
