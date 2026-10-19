# NegICC

NegICC converts color negatives scanned with a digital camera into positive images. `neg_process` reads the RAW
scan, removes the camera's channel crosstalk, normalises to the film base and applies a film profile: an ICC
profile of the film stock, built from an IT8 target recorded on that film.

With `--auto` it does the whole conversion by itself: it finds the film frame, picks the calibration bracket of a
profile bundle and solves the exposure and color balance with a DINOv3 model of how the film stock should look.

![sample conversion](samples/sample.png)

Profiling, capture and the research behind the automatic conversion live in
**[negicc-station](https://github.com/arufahc/negicc-station)**, which keeps its own Python implementation of the
automatic conversion for prototyping and links this repository's C++ code. The profiles and models here are exported
from it.

# Building

```
sudo apt-get install build-essential libraw-dev liblcms2-dev zlib1g-dev
make neg_process            # bin_out/neg_process
```

ONNX Runtime (CPU) is expected under `3rd_party/onnxruntime` (see its README). On a machine with CUDA and TensorRT
(e.g. a Jetson with JetPack 6) the build uses them automatically; `make USE_CUDA=0` forces the CPU build.

The profile bundles and the intent adapters are built into `neg_process`, so the binary needs no data files except
the DINOv3 backbone that `--dino` and `--auto` run, in `models/dinov3-small/`. To export it from the pretrained
weights (needs Python with torch, transformers and onnx):

```
make dinov3_model           # models/dinov3-small/backbone.onnx, with the weights' licence files
make dinov3_engine          # optional: its TensorRT engine, on the device that runs it
```

`neg_process` looks for the backbone in `$NEGICC_DINOV3_MODEL`, then `models/dinov3-small` next to the
executable's directory, then `models/dinov3-small`, or takes a path: `--dino_model PATH`.

# Usage

```
# Automatic: find the frame, pick the bracket, solve exposure and color balance.
bin_out/neg_process --auto --film_stock portra400 -P srgb -o out.tif scan.ARW

# Parameter search on a frame you give (sensor pixels: x y w h [rot_cw [hflip [vflip]]]).
bin_out/neg_process --dino --film_stock portra400 --roi 1700 0 6686 6376 -P srgb -o out.tif scan.ARW

# Manual: one bracket with exposure and gains given.
bin_out/neg_process --film_stock portra400 --target 2 -E 2.83 --gain_g 1.0 --gain_b 0.54 -P srgb -o out.tif scan.ARW

# The profile bundles built in.
bin_out/neg_process --list_profiles
```

Picking the profile bundle and intent adapter:
- `--film_stock ektar100|portra160|portra400` with `--light_source tri|d65` (default `tri`) picks the built-in
  bundle and the stock's adapter.
- `--profile` (`-p`) overrides the bundle: a key from `--list_profiles`, a bundle directory, a negicc-station bundle
  JSON, or a single ICC profile.

Automatic steps:
- `--auto_crop` finds the film frame (the lit area of the scan), narrowed to a film format with
  `--format 6x6|35mm|645|6x7|6x9`.
- `--dino` searches the bracket, exposure and gains with DINOv3. `--dino PATH` uses another intent adapter
  (`adapter.onnx` or its directory). With `--target`, or with a single ICC profile, it solves only the exposure and
  gains of that profile.
- `--auto` is `--auto_crop` and `--dino`.

Manual overrides:
- `--roi`, `--rot`, `--hflip` and `--vflip` set the frame and its orientation.
- `-E`, `--gain_g` and `--gain_b` set the exposure and gains; with `--dino` they scale the solved values.
- `--crosstalk_correction r1 r2 r3 g1 g2 g3 b1 b2 b3` replaces the bundle's crosstalk matrix (not with `--dino` on a
  bundle).

Other behaviour:
- Four RAW files are merged as a Sony 4-shot pixel shift capture.
- Each output gets a JSON sidecar (`out.tif.json`) with the settings, the bracket chosen, the solved values and a
  `cli_equivalent` that reproduces the result without `--dino`.

# Data

- `profiles/<Key>/`: a profile bundle, one ICC profile per calibration bracket (`Target_N.icc`) and `bundle.json`
  with the measurements, crosstalk matrix, film base and export record. Its README says how it was built.
- `models/dinov3-small_<stock>/`: the intent adapter of a film stock (`adapter.onnx`, `metadata.json`) and its
  model card.
- `models/dinov3-small/`: the DINOv3 backbone shared by every stock (`make dinov3_model`), with its licence.
- `profiles/Sony A7RM4 Portra400* R190808 cLUT.icc`: the earlier single profiles, usable with `-p`.

They are updated from negicc-station with its `tools/export_negicc_data.py`, never edited here except for the
READMEs.

# Code

- `neg_process.cc`: the command line tool.
- `auto_convert.{h,cc}`: the automatic conversion (bracket search, exposure, highlights), a port of
  negicc-station's `PerTargetConverter.convert`.
- `frame_analysis.{h,cc}`: the half-size analysis decode, film frame, statistics, previews and highlight clipping,
  shared with negicc-station.
- `profile_bundle.{h,cc}`, `dinov3_*`: profile bundles, the DINOv3 engine and the intent solver.
- `embedded_data.{h,cc}`, `embed_data.sh`: the data files built into the binary.

`make lib` builds `bin_out/libnegicc_dinov3.a`, the library negicc-station links.
