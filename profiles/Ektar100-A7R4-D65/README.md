# Ektar100-A7R4-D65

Profiles for Kodak Ektar 100 scanned with Sony ILCE-7RM4 (Linear Rec.2020) under a white (D65) light, for
`neg_process --film_stock ektar100 --light_source d65`.

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
| −3 EV | `Target_1.icc` | 1.45 | 0.66 | 3.18 | 2.00 |
| −2 EV | `Target_2.icc` | 0.80 | 0.47 | 2.13 | 1.41 |
| −1 EV | `Target_3.icc` | 0.63 | 0.42 | 1.78 | 1.41 |
| 0 EV | `Target_4.icc` | 0.65 | 0.50 | 1.84 | 1.44 |
| +1 EV | `Target_5.icc` | 0.79 | 0.64 | 2.18 | 1.70 |
| +2 EV | `Target_6.icc` | 0.92 | 0.79 | 2.76 | 1.79 |
| +3 EV | `Target_7.icc` | 1.05 | 0.94 | 3.05 | 2.19 |

**Fitted** is ArgyllCMS `profcheck` on all 288 chart patches the profile was built from (the output is stored in
`bundle.json`). It shows how well the profile fits its own data, so it flatters the profile. **Held out** is the error
on patches the profile never saw: each bracket's profile was rebuilt twice, each time leaving out alternate halves of
the 264 colour patches, and checked on the half it left out. The 24 grey patches, which set the tone curves, stay in
every build. That is the better guide to the error on colours in a real photograph. The profiles are not de-glared:
removing the chart's veiling glare before fitting was tried, did not help overall, and the bundles ship as measured.

## Dynamic range

Range across each bracket's 24 grey patches (`gs0` to `gs23`) before the camera matrix (`bundle.json`'s patch values
multiplied by the inverse of `profiling_input_transform.matrix` to recover raw sensor RGB), as bits
(`log2(max - min)`, on the scale of the camera's 14-bit raw values; the values are flat-fielded, so they can exceed
16383) and contrast in stops (`log2(max / min)`).

| Bracket | Profile | Shutter | Bit depth: R / G / B | Bit depth: mean | Stops: R / G / B | Stops: mean |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| −3 EV | `Target_1.icc` | 1/15s | 12.59 / 13.20 / 12.30 | 12.69 | 1.45 / 1.75 / 2.08 | 1.76 |
| −2 EV | `Target_2.icc` | 1/15s | 12.72 / 13.27 / 12.23 | 12.74 | 1.77 / 2.15 / 2.37 | 2.10 |
| −1 EV | `Target_3.icc` | 1/13s | 13.13 / 13.64 / 12.44 | 13.07 | 2.08 / 2.55 / 2.63 | 2.42 |
| 0 EV | `Target_4.icc` | 1/13s | 13.10 / 13.56 / 12.23 | 12.97 | 2.33 / 2.87 / 2.87 | 2.69 |
| +1 EV | `Target_5.icc` | 1/8s | 13.41 / 13.74 / 12.41 | 13.19 | 2.35 / 2.90 / 2.97 | 2.74 |
| +2 EV | `Target_6.icc` | 1/6s | 13.43 / 13.72 / 12.38 | 13.18 | 2.30 / 2.88 / 3.02 | 2.73 |
| +3 EV | `Target_7.icc` | 1/5s | 13.29 / 13.53 / 12.20 | 13.01 | 2.14 / 2.73 / 2.94 | 2.60 |
| **Mean** | — | — | **13.10 / 13.52 / 12.31** | **12.98** | **2.06 / 2.55 / 2.70** | **2.44** |

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
(Target 4) with gains 1. neg_process keeps the table and writes the curves for the A, g and b the print's search
finds when it converts.

### What they compute

For the profile's input x (the scan decoded to linear Rec.2020 through the camera's colour matrix; the film base at
n0 = normalization_target / 65535, after neg_process's soft knee at 0.95), in each channel c:

```
light density    dL_c  = -log10(unknee(x_c) g_c / n0)
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
  then through M. The steps run from green -0.0151 to 1.2527
  above the film base; past them the tangents' slopes (red and blue against green) are red 0.9412
  and 0.7992, blue 1.5868 and 1.7433 at the first and the last step
  (`grey_balance_curves.tangents`), and most of the table past the last step comes from them.
- **A**, the print's mid grey: a bracket's own `A_mid` (the green APD of its 18 % grey step, gs10, from
  `density_matrices.brackets.targets.<bracket>.grey_light_density`), or the A the print's search finds.
- **g_c**, the print's colour balance (1, g, b), applied after the knee: 1 in the profile's own curves, the g and b
  the print's search finds when neg_process writes them.
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
input in_c(x)     = Paper_c((G_c(dL_c) - A) (1 + t_c))
table(a)          = the chain above, from alpha_c = Paper_c^-1(a_c)
Paper_c(alpha_i)  = 0.75 curve_c(aim_c - eta_c alpha_i) / top_c + 0.25 (1 - i / (N - 1)),   eta = P (1, 1, 1)
```

Paper_c is 0.75 of the dye amount layer c takes on a grey alpha above the mid grey, scaled to [0, 1] per channel, plus
0.25 of a linear ramp that falls from 1 to 0 across the grid alpha_i = -1.5 + 0.005 i (N = 661, -1.5 .. 1.8).
curve_c(log E) is the material's dye amount of layer c, held non-increasing in alpha_i; top_c is its value at alpha_0 =
-1.5, the largest on the grid; P is the print matrix for Academy Printing Density, so P M = Q L. An alpha_c outside the
grid is held at its end. The paper's S-curve is in the input curves, each channel's curve spans [0, 1] with a positive
slope across the toe and shoulder, and the table holds the dyes' colour and the print matrix's mixing off the grey axis.
The table has no bracket in it: every bracket's profile is this table with its own curves.
<!-- end of print profiles -->
