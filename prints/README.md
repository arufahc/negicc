# Print materials

One Common LUT Format transform per print material: its printing density (as density, 0.70 exposed as mid grey) to
the print under its viewing light, as sRGB. `neg_process` builds them in and writes, beside each printing density
file, the frame's matrix followed by these nodes.

Written by negicc-station (`tools/export_print_clfs.py`, `src/print_slopes.py`). The print curves, dye spectra and
spectral sensitivities come from [spektrafilm](https://github.com/andreavolpato/spektrafilm) by Andrea Volpato
(print profiles, CC BY-SA 4.0); these files derive from them and carry the same licence. They are a model and have
not been compared with a real print.
