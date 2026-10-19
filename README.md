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
executable's directory, then `models/dinov3-small`, or takes a path: `--dino PATH`.

# Usage

```
# Automatic: find the frame, pick the bracket, solve exposure and color balance.
bin_out/neg_process --auto -p Portra400-A7R4-TriLED -P srgb -o out.tif scan.ARW

# The same with the frame given (sensor pixels: x y w h [rot_cw [hflip [vflip]]]).
bin_out/neg_process --dino -p Portra400-A7R4-TriLED -P srgb --roi 1700 0 6686 6376 -o out.tif scan.ARW

# Manual: one bracket of a bundle, or a single ICC profile, with exposure and gains given.
bin_out/neg_process -p Portra400-A7R4-TriLED --target 3 -E 0.6 --gain_g 0.9 --gain_b 0.8 -P srgb -o out.tif scan.ARW

# The profile bundles built in.
bin_out/neg_process --list_profiles
```

- `--auto` narrows the frame to a film format with `--format 6x6|35mm|645|6x7|6x9`, and orients it with `--rot`,
  `--hflip` and `--vflip`.
- With `--dino` or `--auto`, `-E`, `--gain_g` and `--gain_b` scale the solved values.
- With `--target`, or with a single ICC profile, `--dino` solves only the exposure and gains of that profile.
- Four RAW files are merged as a Sony 4-shot pixel shift capture.
- Each output gets a JSON sidecar (`out.tif.json`) with the settings, the bracket chosen and the solved values.

`-p` takes a bundle key (`--list_profiles`), a bundle directory (`profiles/<Key>/`), an all-in-one bundle JSON from
negicc-station, or an ICC profile.

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
