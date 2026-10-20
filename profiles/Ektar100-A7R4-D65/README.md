# Ektar100-A7R4-D65

Profiles for Kodak Ektar 100 scanned with Sony ILCE-7RM4 (Linear Rec.2020) under a white (D65) light, for `neg_process --film_stock ektar100 --light_source d65`.

How they were built: an IT8 target (R190808) was photographed on this film stock at 7 exposures. Each developed
frame (a bracket) was scanned with this camera and light, and its patches were matched to the target's reference
colours to make one ICC profile per bracket (`Target_1.icc` to `Target_7.icc`). Before matching, the camera's
channel crosstalk was removed and the values were scaled to the film base (the unexposed film), which scans are
normalised to as well. The captures were flat-field corrected.

negicc-station built the bundle and its `tools/export_negicc_data.py` split it into these files. `bundle.json`
holds the measurements, the crosstalk matrix, the film base and an `export` record of the source file.
