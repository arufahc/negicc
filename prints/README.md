# Print materials

One file per print material, a paper or a print film (`negicc.print_material/1`). `neg_process` builds them in and
computes from them, for each printing density file it writes, a CLF that prints the file on that material
(`np_print.h`, and "the print of a density file" in `DESIGN.md`).

A file holds two kinds of numbers:

| Fields | What they are | Licence |
|---|---|---|
| `log_exposure`, `curves`, `dye_density`, `base_density` | The material's curve per layer, its three dye spectra and its base, from the print profile of [spektrafilm](https://github.com/andreavolpato/spektrafilm) by Andrea Volpato, resampled to 1 nm with gaps held at the nearest measured value (by Alpha Lam; see `CHANGELOG.txt`) | CC BY-SA 4.0, as spektrafilm's profiles ([licence text](https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt)) |
| `to_print_density`, `aim_log_exposure`, `observer`, `white_xyz`, `xyz_to_linear_srgb` | Computed numbers: the 3x3 from RP 180 and from Academy Printing Density to the density the material reads (one per film model, and `any` for other films), the exposure that prints an 18 % grey, the CIE observer under the viewing light, and the matrix from that XYZ to linear sRGB | Results, not spektrafilm's data: part of negicc, under negicc's licence |

A file holds nothing measured on a film; a film's chart readings are in its profile set. The print materials are a
model and have not been compared with a real print.
