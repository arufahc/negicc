# NegICC

NegICC converts color negatives scanned with a digital camera into positive images. `neg_process` reads the RAW
scan, removes the camera's channel crosstalk, normalises to the film base and applies a film profile: an ICC
profile of the film stock, built from an IT8 target recorded on that film.

Given the film stock and the scanning light, it does the whole conversion by itself: it picks the calibration bracket
of the built-in profile bundle and solves the exposure and color balance with a DINOv3 model of how the film stock
should look.

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

The profile bundles, the intent adapters and the DINOv3 backbone of the parameter search are built into
`neg_process`, so the binary needs no data files. Export the backbone from the pretrained weights before building
(needs Python with torch, transformers and onnx):

```
make dinov3_model           # models/dinov3-small/backbone.onnx, with the weights' licence files
make dinov3_engine          # USE_CUDA=1 builds: its TensorRT engine, on the machine that runs neg_process
make neg_process
```

A `USE_CUDA=1` build packs the TensorRT engine, a `USE_CUDA=0` build packs `backbone.onnx`. Without the backbone
`neg_process` still builds, but converts only with `--gain_g` or `--gain_b` given.

# Usage

```
# Automatic: pick the bracket, solve exposure and color balance (frame: x y w h [rot_cw [hflip [vflip]]]).
bin_out/neg_process --film_stock portra400 --light_source tri --roi 1700 0 6686 6376 -P srgb -o out.tif scan.ARW

# Manual: a bracket of the bundle with the exposure and gains given.
bin_out/neg_process --film_stock portra400 --light_source tri --target 3 -E 0.6 --gain_g 0.9 --gain_b 0.8 \
    -P srgb -o out.tif scan.ARW

# The film stocks and light sources built in.
bin_out/neg_process --list_profiles
```

- `--film_stock` is `ektar100`, `portra160` or `portra400`; `--light_source` is `tri` (narrowband red, green and
  blue LED) or `d65` (white).
- The parameter search runs unless `--gain_g` or `--gain_b` is given. `-E` keeps the exposure and solves only the
  gains, on the bracket `--target` picks or the bundle's default.
- `-r`, `-g` and `-b` replace rows of the bundle's crosstalk correction with ones measured for the capture.
- `-p` takes a negicc-station profile bundle JSON (e.g. to test parity with it) or a single ICC profile (for
  research) in place of the built-in bundle; `--dino` an intent adapter (`adapter.onnx` or its directory) in place of
  the film stock's.
- `--roi auto` finds the film frame, but not yet with the bracket search.
- Four RAW files are merged as a Sony 4-shot pixel shift capture.
- Each output gets a JSON sidecar (`out.tif.json`) with the settings, the bracket chosen and the solved values; its
  `cli_equivalent` reproduces the conversion.

# Data

- `profiles/<Key>/`: a profile bundle, one ICC profile per calibration bracket (`Target_N.icc`) and `bundle.json`
  with the measurements, crosstalk matrix, film base and export record. Its README says how it was built.
- `models/dinov3-small_<stock>/`: the intent adapter of a film stock (`adapter.onnx`, `metadata.json`) and its
  model card.
- `models/dinov3-small/`: the DINOv3 backbone shared by every stock (`make dinov3_model`), with its licence.

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
