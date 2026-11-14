# Print materials

One file per print material, a paper or a print film (`negicc.print_material/1`). `neg_process` builds them in and
computes from them, for each printing density file it writes, a CLF that prints the file on that material
(`np_print.h`, and "Print emulation" in `DESIGN.md`).

A file holds the material's curve per layer, its three dye spectra and its base, from the print profile of
[spektrafilm](https://github.com/andreavolpato/spektrafilm) by Andrea Volpato, resampled to 1 nm; and numbers
computed from them: the 3x3 from RP 180 and from Academy Printing Density to the density the material reads (one per
film model, and `any` for other films), the exposure that prints an 18 % grey, the CIE observer under the viewing
light, and the matrix from that XYZ to linear sRGB.

**Licence.** These JSON files are under CC BY-SA 4.0, as the spektrafilm profiles they derive from
([licence text](https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt)); they were modified
by Alpha Lam as `CHANGELOG.txt` records. The CLF files that `neg_process` writes with them, and the pictures made
through those, are output of `neg_process` and are not under that licence.

A file holds nothing measured on a film; a film's chart readings are in its profile set. The print materials are a
model and have not been compared with a real print.
