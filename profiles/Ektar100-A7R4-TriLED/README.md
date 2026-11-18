# Ektar100-A7R4-TriLED

Profiles for Kodak Ektar 100 scanned with ILCE-7RM4 under a narrowband red, green and blue LED light, for
`neg_process --film_stock ektar100 --light_source tri`.

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
| −3 EV | `Target_1.icc` | 1.11 | 0.49 | 3.05 | 1.87 |
| −2 EV | `Target_2.icc` | 0.61 | 0.39 | 2.68 | 1.52 |
| −1 EV | `Target_3.icc` | 0.51 | 0.36 | 2.79 | 1.55 |
| 0 EV | `Target_4.icc` | 0.62 | 0.49 | 3.25 | 1.84 |
| +1 EV | `Target_5.icc` | 0.63 | 0.49 | 3.52 | 1.75 |
| +2 EV | `Target_6.icc` | 0.72 | 0.59 | 4.01 | 2.15 |
| +3 EV | `Target_7.icc` | 0.83 | 0.65 | 3.72 | 2.26 |

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
| −3 EV | `Target_1.icc` | 1/10s | 12.83 / 13.00 / 13.00 | 12.94 | 1.65 / 1.93 / 2.19 | 1.92 |
| −2 EV | `Target_2.icc` | 1/8s | 13.25 / 13.36 / 13.25 | 13.29 | 2.01 / 2.32 / 2.47 | 2.27 |
| −1 EV | `Target_3.icc` | 1/8s | 13.30 / 13.33 / 13.07 | 13.23 | 2.34 / 2.70 / 2.70 | 2.58 |
| 0 EV | `Target_4.icc` | 1/6s | 13.56 / 13.53 / 13.15 | 13.42 | 2.63 / 3.02 / 2.90 | 2.85 |
| +1 EV | `Target_5.icc` | 1/5s | 13.50 / 13.37 / 13.00 | 13.29 | 2.66 / 3.11 / 3.02 | 2.93 |
| +2 EV | `Target_6.icc` | 1/4s | 13.46 / 13.33 / 12.96 | 13.25 | 2.58 / 3.12 / 3.09 | 2.93 |
| +3 EV | `Target_7.icc` | 1/3s | 13.24 / 13.13 / 12.77 | 13.05 | 2.39 / 3.00 / 3.01 | 2.80 |
| **Mean** | — | — | **13.30 / 13.30 / 13.03** | **13.21** | **2.32 / 2.74 / 2.77** | **2.61** |

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

For the profile's input x (the scan after the crosstalk correction; the film base at
n0 = normalization_target / 65535, after neg_process's soft knee at 0.95), in each channel c:

```
light density    dL_c  = -log10(unknee(x_c) / n0)
grey balance     alpha_c = (G_c(dL_c) - A) (1 + t_c)
print density    pd = 0.70 + Q L (psi alpha),   psi = M^-1 (1, 1, 1)
the print        log E_c = aim_c - (pd_c - 0.70)  ->  the material's dye amounts (its curve per layer)
                 ->  its colour under its viewing light (CIE 1931 2 degree)  ->  XYZ, adapted to D50 (Bradford)
                 ->  relative to the paper's white when neg_process renders it (the paper base at D50)
```

- **G_c**, the film's grey balance curve (`bundle.json` `density_matrices.grey_balance_curves`, 501 points per
  channel from light density -0.5 in steps of 0.01): the APD of the film's
  grey that has light density dL_c in channel c. It comes from the chart's grey steps of every bracket: the red and
  the blue light density as a cubic in the green (least squares, carried on by its tangent past the end steps),
  then through M. The steps run from green -0.0082 to 1.5185
  above the film base; past them the tangents' slopes (red and blue against green) are red 1.0874
  and 0.7023, blue 1.6077 and 1.5405 at the first and the last step
  (`grey_balance_curves.tangents`), and most of the table past the last step comes from them.
- **A**, the print's mid grey: a bracket's own `A_mid` (the green APD of its 18 % grey step, gs10, from
  `density_matrices.brackets.targets.<bracket>.grey_light_density`), or the A the print's search finds.
- **t_c**, each bracket's tilt relative to the curve
  (`density_matrices.brackets.targets.<bracket>.grey_relative.b`): neg_process takes it linear in A between the two
  brackets whose A_mid enclose A, held beyond the end ones, so at a bracket's own A_mid it is that bracket's tilt.
- **M**, the bundle's matrix for Academy Printing Density (`film_to_measure` times `light_to_film`).
- **Q L**, from the light's density to the density the print material reads: Q from the film model's dye amounts
  to the print's own density under a 3200 K printer lamp, least squares over the film's sample colours
  (negicc `DESIGN.md`, "The print matrix").
- **aim**, the exposure of each layer that prints an 18 % grey; the bracket's mid grey prints there.

The profile splits this where a table samples it well:

```
input curve_c(x)  = Paper_c((G_c(dL_c) - A) (1 + t_c))
table(a)          = the chain above, from alpha_c = Paper_c^-1(a_c)
Paper_c(alpha_i)  = 0.75 D_c(aim_c - eta_c alpha_i) / top_c + 0.25 (1 - i / (N - 1)),   eta = P (1, 1, 1)
```

Paper_c is 0.75 of the dye amount layer c takes on a grey alpha above the mid grey, scaled to [0, 1] per channel, plus
0.25 of a linear ramp that falls from 1 to 0 across the grid alpha_i = -1.5 + 0.005 i (N = 661, -1.5 .. 1.8). D_c(log E)
is the material's dye amount of layer c, held non-increasing in alpha_i; top_c is its value at alpha_0 = -1.5, the
largest on the grid; P is the print matrix for Academy Printing Density, so P M = Q L. An alpha_c outside the grid is
held at its end. The paper's S-curve is in the input curves, each channel's curve spans [0, 1] with a positive slope
across the toe and shoulder, and the table holds the dyes' colour and the print matrix's mixing off the grey axis. The
table has no bracket in it: every bracket's profile is this table with its own curves.
<!-- end of print profiles -->
