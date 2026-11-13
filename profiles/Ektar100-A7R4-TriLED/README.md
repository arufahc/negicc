# Ektar100-A7R4-TriLED

Profiles for Kodak Ektar 100 scanned with ILCE-7RM4 under a narrowband red, green and blue LED light, for `neg_process --film_stock ektar100 --light_source tri`.

How they were built: an IT8 target (R190808) was photographed on this film stock at 7 exposures. Each developed
frame (a bracket) was scanned with this camera and light, and its patches were matched to the target's reference
colours to make one ICC profile per bracket (`Target_1.icc` to `Target_7.icc`). Before matching, the camera's
channel crosstalk was removed and the values were scaled to the film base (the unexposed film), which scans are
normalised to as well. The captures were flat-field corrected.

negicc-station built the bundle and its `tools/export_negicc_data.py` split it into these files. `bundle.json`
holds the measurements, the crosstalk matrix, the film base and an `export` record of the source file.

## Accuracy

How closely each bracket's profile reproduces the IT8 chart, as CIEDE2000 colour differences (ΔE₀₀; under 2 is
barely visible).

| Bracket | Profile | Fitted: mean | Fitted: median | Held out: mean | Held out: median |
| :--- | :--- | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1.11 | 0.49 | 3.05 | 1.87 |
| −2 EV | `Target_2.icc` | 0.61 | 0.39 | 2.68 | 1.52 |
| −1 EV | `Target_3.icc` | 0.51 | 0.36 | 2.79 | 1.55 |
| 0 EV | `Target_4.icc` | 0.62 | 0.49 | 3.25 | 1.84 |
| +1 EV | `Target_5.icc` | 0.63 | 0.49 | 3.52 | 1.75 |
| +2 EV | `Target_6.icc` | 0.72 | 0.59 | 4.01 | 2.15 |
| +3 EV | `Target_7.icc` | 0.83 | 0.65 | 3.72 | 2.26 |

**Fitted** is ArgyllCMS `profcheck` on all 288 chart patches the profile was built from (the output is stored in
`bundle.json`). It shows how well the profile fits its own data, so it flatters the profile. **Held out** is the
error on patches the profile never saw: each bracket's profile was rebuilt twice, each time leaving out alternate
halves of the 264 colour patches, and checked on the half it left out. The 24 grey patches, which set the tone curves, stay in every build.
That is the better guide to the error on colours in a real photograph. The profiles are not de-glared: removing the
chart's veiling glare before fitting was tried, did not help overall, and the bundles ship as measured.
