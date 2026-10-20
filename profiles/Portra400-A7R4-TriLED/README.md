# Portra400-A7R4-TriLED

Profiles for Kodak Portra 400 scanned with ILCE-7RM4 under a narrowband red, green and blue LED light, for `neg_process --film_stock portra400 --light_source tri`.

How they were built: an IT8 target (R190808) was photographed on this film stock at 7 exposures. Each developed
frame (a bracket) was scanned with this camera and light, and its patches were matched to the target's reference
colours to make one ICC profile per bracket (`Target_1.icc` to `Target_7.icc`). Before matching, the camera's
channel crosstalk was removed and the values were scaled to the film base (the unexposed film), which scans are
normalised to as well. The captures were flat-field corrected.

negicc-station built the bundle and its `tools/export_negicc_data.py` split it into these files. `bundle.json`
holds the measurements, the crosstalk matrix, the film base and an `export` record of the source file.
