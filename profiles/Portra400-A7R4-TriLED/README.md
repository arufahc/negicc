# Portra400-A7R4-TriLED

Profiles for Kodak Portra 400 scanned with ILCE-7RM4 under a narrowband red, green and blue LED light, for
`neg_process --film_stock portra400 --light_source tri`.

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
`bundle.json`). It shows how well the profile fits its own data, so it flatters the profile. **Held out** is the error
on patches the profile never saw: each bracket's profile was rebuilt twice, each time leaving out alternate halves of
the 264 colour patches, and checked on the half it left out. The 24 grey patches, which set the tone curves, stay in
every build. That is the better guide to the error on colours in a real photograph. The profiles are not de-glared:
removing the chart's veiling glare before fitting was tried, did not help overall, and the bundles ship as measured.

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

<!-- print profiles: written by tools/export_negicc_data.py -->
## Print profiles

One ICC profile per print material: the film printed on that paper or print film, for `neg_process --print <id>`.
The print materials are spektrafilm's print profiles (Andrea Volpato, CC BY-SA 4.0), built into negicc from
`3rd_party/spektrafilm/profiles/`; the tables of these ICCs derive from them and carry that notice.

| File | Print material | Viewed under |
|---|---|---|
| `print_fujifilm_crystal_archive_typeii.icc` | Fujifilm Crystal Archive Type II | D50 |
| `print_kodak_2383.icc` | Kodak Vision 2383 | D55 |
| `print_kodak_2393.icc` | Kodak Vision Premier 2393 | D55 |
| `print_kodak_ektacolor_edge.icc` | Kodak Ektacolor Edge | D50 |
| `print_kodak_endura_premier.icc` | Kodak Professional Endura Premier | D50 |
| `print_kodak_portra_endura.icc` | Kodak Professional Portra Endura | D50 |
| `print_kodak_supra_endura.icc` | Kodak Professional Supra Endura | D50 |
| `print_kodak_ultra_endura.icc` | Kodak Professional Ultra Endura | D50 |

Each is an ICC v2 input profile (RGB to PCS XYZ D50) shaped like the chart profiles beside it: an A2B0 with three
4096-entry input curves, a 45 x 45 x 45 table and identity output curves. Its own curves are for the middle bracket
(Target 4). neg_process keeps the table and writes the curves for the A the print's search finds when it converts.

### What they compute

For the profile's input v (the scan after the crosstalk correction, the film base at n0 =
normalization_target / 65535, after neg_process's soft knee at 0.95), in each channel c:

```
light density    dL_c  = -log10(unknee(v_c) / n0)
grey balance     alpha_c = (G_c(dL_c) - A) (1 + b_c)
print density    pd = 0.70 + Q L (w alpha),   w = M^-1 (1, 1, 1)
the print        log E_c = aim_c - (pd_c - 0.70)  ->  the material's dye amounts (its curve per layer)
                 ->  its colour under its viewing light (CIE 1931 2 degree)  ->  XYZ, adapted to D50 (Bradford)
                 ->  relative to the paper's white when neg_process renders it (the paper base at D50)
```

- **G_c**, the film's grey balance curve (`bundle.json` `density_matrices.grey_balance_curves`, 501 points per
  channel from light density -0.5 in steps of 0.01): the APD of the film's
  grey that has light density dL_c in channel c. It comes from the chart's grey steps of every bracket: the red and
  the blue light density as a cubic in the green (least squares, carried on by its tangent past the end steps),
  then through M. The steps run from green -0.0092 to 1.4511
  above the film base; past them the tangents' slopes (red and blue against green) are red 0.8036
  and 0.8584, blue 1.0998 and 1.1225 at the first and the last step
  (`grey_balance_curves.tangents`), and most of the table past the last step comes from them.
- **A**, the print's mid grey: a bracket's own `A_mid` (the green APD of its 18 % grey step, gs10, from
  `density_matrices.brackets.targets.<bracket>.grey_light_density`), or the A the print's search finds.
- **b_c**, each bracket's tilt relative to the curve
  (`density_matrices.brackets.targets.<bracket>.grey_relative.b`): neg_process takes it linear in A between the two
  brackets whose A_mid enclose A, held beyond the end ones, so at a bracket's own A_mid it is that bracket's tilt.
- **M**, the bundle's matrix for Academy Printing Density (`film_to_measure` times `light_to_film`).
- **Q L**, from the light's density to the density the print material reads: Q from the film model's dye amounts
  to the print's own density under a 3200 K printer lamp, least squares over the film's sample colours
  (negicc `DESIGN.md`, "The print matrix").
- **aim**, the exposure of each layer that prints an 18 % grey; the bracket's mid grey prints there.

The profile splits this where a table samples it well:

```
input curve_c(v)  = Paper_c((G_c(dL_c) - A) (1 + b_c))
table(a)          = the chain above, from alpha_c = Paper_c^-1(a_c)
Paper_c(alpha)    = curve_c(aim_c - k_c alpha) / top,   k = P (1, 1, 1)
```

Paper_c is the dye amount layer c takes on a grey alpha above the mid grey (top: the curves' largest value, rounded
up to 0.1; P: the print matrix for Academy Printing Density, so P M = Q L). The paper's S-curve is then in the input
curves, and the table holds only the dyes' colour and the print matrix's mixing off the grey axis. The table has no
bracket in it: every bracket's profile is this table with its own curves.
<!-- end of print profiles -->
