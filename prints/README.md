# Print materials

One JSON per print material (`negicc.print_material/1`): the curve of each layer, the dye spectra and base, the
observer under the viewing light, and the 3x3 from each printing density measure (RP 180, Academy Printing Density)
to the density the material reads itself. That matrix comes from spectra alone; there is one per film model, by
name, and one for any other film. A file holds nothing measured on a film. `neg_process` builds them in and computes
from them, for each printing density file it writes, a CLF that prints the file (np_print.h).

Written by negicc-station (`tools/export_print_materials.py`, `src/print_slopes.py`). The print curves, dye spectra
and spectral sensitivities come from [spektrafilm](https://github.com/andreavolpato/spektrafilm) by Andrea Volpato
(print profiles, CC BY-SA 4.0, [licence text](https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt)).
These files are derived from them by negicc-station (Alpha Lam) and carry the same licence, as does every CLF that
`neg_process` writes from them; `CHANGELOG.txt` records the changes. They are a model and have not been compared
with a real print.
