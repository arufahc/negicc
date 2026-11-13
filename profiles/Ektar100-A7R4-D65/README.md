# Ektar100-A7R4-D65

Profiles for Kodak Ektar 100 scanned with Sony ILCE-7RM4 (Linear Rec.2020) under a white (D65) light, for `neg_process --film_stock ektar100 --light_source d65`.

How they were built: an IT8 target (R190808) was photographed on this film stock at 7 exposures. Each developed
frame (a bracket) was scanned with this camera and light, and its patches were matched to the target's reference
colours to make one ICC profile per bracket (`Target_1.icc` to `Target_7.icc`). Before matching, the values were
taken to linear Rec.2020 through the camera's daylight white balance and colour matrix and scaled to the film base
(the unexposed film), which scans are normalised to as well. The captures were flat-field corrected.

The bundle's `input_space` is `rec2020_linear`: `neg_process` converts a scan through LibRaw's white-light decode
(the camera's daylight white balance and LibRaw's colour matrix, to linear Rec.2020), as negicc-station does, and
applies no crosstalk correction, so `-r`, `-g` and `-b` cannot be given with it. The camera matrix the patches were
measured through is kept under `profiling_input_transform` as a record only; nothing applies it.

negicc-station built the bundle and its `tools/export_negicc_data.py` split it into these files. `bundle.json`
holds the measurements, the film base and an `export` record of the source file.

## Accuracy

How closely each bracket's profile reproduces the IT8 chart, as CIEDE2000 colour differences (ΔE₀₀; under 2 is
barely visible).

| Bracket | Profile | Fitted: mean | Fitted: median | Held out: mean | Held out: median |
| :--- | :--- | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1.45 | 0.66 | 3.18 | 2.00 |
| −2 EV | `Target_2.icc` | 0.80 | 0.47 | 2.13 | 1.41 |
| −1 EV | `Target_3.icc` | 0.63 | 0.42 | 1.78 | 1.41 |
| 0 EV | `Target_4.icc` | 0.65 | 0.50 | 1.84 | 1.44 |
| +1 EV | `Target_5.icc` | 0.79 | 0.64 | 2.18 | 1.70 |
| +2 EV | `Target_6.icc` | 0.92 | 0.79 | 2.76 | 1.79 |
| +3 EV | `Target_7.icc` | 1.05 | 0.94 | 3.05 | 2.19 |

**Fitted** is ArgyllCMS `profcheck` on all 288 chart patches the profile was built from (the output is stored in
`bundle.json`). It shows how well the profile fits its own data, so it flatters the profile. **Held out** is the
error on patches the profile never saw: each bracket's profile was rebuilt twice, each time leaving out alternate
halves of the 264 colour patches, and checked on the half it left out (negicc-station's
`src/audit/profile_holdout.py`, 2 folds). The 24 grey patches, which set the tone curves, stay in every build.
That is the better guide to the error on colours in a real photograph. The profiles are not de-glared: removing the
chart's veiling glare before fitting was tried, did not help overall, and the bundles ship as measured.
