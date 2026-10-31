# Objective

NegICC creates ICC profiles and converts color negatives captured with a digital camera into consistent, color-accurate positive images. The ICC profiles and conversion tool (`neg_process`) have the following properties:
* Allow color balance and exposure adjustments while preserving film dye characteristics
* Produce neutral, accurate color ready for display or further editing in standard photo editors
* Include **3 manually profiled film stocks** (**Kodak Portra 400**, **Portra 160**, and **Ektar 100**), each photographed and calibrated across **7 exposure brackets** under **2 light sources** (**42 ICC profiles total** in `profiles/<Key>/`, `Target_1.icc`..`Target_7.icc` under narrowband `TriLED` and white `D65` light)

![sample conversion](samples/sample.png)

# Introduction

Typical color negative conversion software operates in sRGB colorspace. However, color negative film records color information as dye density designed to be extracted with [Status M](https://law.resource.org/pub/us/cfr/ibr/001/aimm.it2.18.1996.pdf) densitometry rather than D50 or 5000K light.

The first step of scanning film with a digital camera is to produce a response that resembles Status M densitometry with good separation between the channels (see the original [guide](https://docs.google.com/document/d/1NsfFPx5c7kxNRUhuGQBKRNnaN0c52SjLsZsyhLCj3OE/edit?usp=sharing)).

# How does it work?

With narrowband RGB LED light (or a triband filter) and crosstalk correction, the RGB response is still not linear to density. From this diagram, the B and G responses of the profiled film closely resemble each other, while the R response has a different slope:

![Characteristic Curves](samples/characteristic_curves.png)

An IT8.7/2 reflective target (`R190808` from coloraid.de) photographed onto each film stock across **7 exposure brackets** is used to correct the different channel responses and invert them into linear RGB corresponding to original scene luminance:

![IT8 used for profiling](samples/it8.png)

### Simple Trichromatic (`TriLED`) & White-Light (`D65`) Capture

Capturing negatives is much simpler now and supports camera RAWs, DNGs, and 16-bit linear TIFFs without optical filter wheels:
* **Trichromatic narrowband RGB LED (`--light_source tri`):**
  * *Single-shot RAW/DNG or 4-shot Sony Pixel Shift:* Uses the bundle's $3\times 3$ sensor crosstalk correction matrix (or custom `-r`, `-g`, `-b` rows) to remove Bayer CFA channel overlap.
  * *3-shot monochromatic RGB (`r.raw g.raw b.raw`) or pre-separated TIFF/DNG:* Capturing 3 separate frames under Red, Green, and Blue LEDs isolates the channels physically with zero sensor crosstalk—use the **identity color correction** (`-r "1 0 0" -g "0 1 0" -b "0 0 1"`).
* **White D65 light (`--light_source d65`):** Single-shot capture under a standard D65 white LED panel. `neg_process` decodes the scan through LibRaw's daylight white balance and camera matrix into linear Rec.2020 with **no crosstalk matrix needed**.

### 7-Bracket Tone Curves (TRC) & Highlight Roll-Off

A single IT8 target only calibrates tones across its own **7.22-stop** range up to the chart's white patch (`gs0` at 0.75). Fitting all **7 brackets** (`Target_1`..`Target_7`, spanning **4.6 to 5.4 stops** from under- to over-exposed) together measures each bracket's exact exposure offset and traces the film's true highlight shoulder—giving **11.9 to 12.7 stops of total scene dynamic range** and **+2.6 to +4.0 stops of calibrated highlight headroom** above normal white:

![TRC Curves](samples/trc_curves.png)

The final step converts linearized RGB values into CIE XYZ / $L^*a^*b^*$ using a 16-bit Curve + **$45^3$ 3D-cLUT** encoded as an ICC profile:

![ICC profile](samples/icc.png)

# ICC Profile Description & Expanded Highlight Gamut

While early versions generated Curve+Matrix and V4 MPET profiles, the built-in bundles provide **7 brackets of 16-bit Curve + $45^3$ cLUT ICC profiles** (`profiles/<Key>/Target_1.icc`..`Target_7.icc`). Standard single-target cLUT profiles lose saturation or clip in bright highlights because a single chart has no patches above its white patch ($L^* \approx 89.4$). By injecting the brighter patches from the denser, more over-exposed brackets (`Target 4`–`Target 7`, adding +567 highlight color samples on `Target_3.icc`) into the cLUT, the calibrated 3D CIELAB gamut volume and highlight chroma extend smoothly all the way to pure white ($L^* = 100$) without amplifying film grain:

![Gamut Volume](samples/gamut_volume.png)

### DINOv3 Automatic Color Grader & Iterative Convergence

Instead of relying on simple grey-world heuristics, `neg_process` uses Meta's **DINOv3** vision model as an **automatic color grader** trained on real **Fujifilm Frontier and Noritsu lab scans** collected from Flickr and Photohito (`models/dinov3-small_<stock>/`). Because DINOv3 understands scene semantics, it recognizes objects in context (sky, foliage, skin tones, stone, shadows) and predicts the **target Lab color** ($\mu_i$) and **tolerance** ($\sigma_i$) that each $16\times 16$ patch of the photo should have based on learned lab-scan examples:

![DINOv3 Intent Field and Convergence](samples/dinov3_intent.png)

To convert a frame automatically:
1. **Auto Film-Base & Frame Detection:** Automatically samples the unexposed orange film base from the scan border (rebate or thinnest film; *TODO: add support for reading the film base from a separate unexposed film capture*) and crops the film frame (`--roi auto --film_format 35mm|645|6x6|6x7|6x9`).
2. **Bracket Screening & Iterative Error Minimization:** Screens the 7 brackets against the predicted target Lab patch field, then iteratively adjusts exposure (`E`) and green/blue color balance (`g`, `b`) on the top brackets—re-evaluating the DINOv3 target Lab field at each step—until the patch Lab error converges to a minimum (with automatic highlight pullback and bracket step-up if highlights clip).

# Requirements & Hardware

`neg_process` builds LibRaw (`3rd_party/libraw` submodule; `make LIBRAW_JPEG=1` adds lossy DNG support), Little-CMS 2 (`3rd_party/lcms2`), and zlib from source, and uses ONNX Runtime (`3rd_party/onnxruntime`) with the DINOv3 model (`make dinov3_model`).
* **Hardware:** Sony A7RM4 (`ILCE-7RM4`, single-shot or 4-shot pixel shift) + Rayfact QVM05041MF; narrowband RGB LED light ([Shoebox Photos BCS](https://www.shoeboxphotosbcs.com) or Olafus RGB flood light with diffuser—**no optical filters needed**) or D65 white LED panel; IT8.7/2 target (`R190808` from coloraid.de, 7 bracketed exposures per stock).

```bash
sudo apt-get install clang make autoconf automake libtool pkg-config libtiff-dev libjpeg-dev curl python3
git clone --recursive https://github.com/arufahc/negicc.git && cd negicc
make dinov3_model && make -j$(nproc) neg_process   # add USE_CUDA=1 for NVIDIA GPU acceleration
```

# Usage Example

```bash
bin_out/neg_process --list_profiles   # list built-in 7-bracket bundles (portra400, portra160, ektar100 for tri/d65)

# 1. Single-shot Trichromatic LED scan (auto film-base, auto bracket & DINOv3 color grading -> 16-bit sRGB TIFF)
bin_out/neg_process --film_stock portra400 --light_source tri -P srgb -o out.tif scan.ARW

# 2. Single-shot D65 white-light scan with automatic 6x7 frame crop
bin_out/neg_process --film_stock portra400 --light_source d65 --roi auto --film_format 6x7 -P srgb -o out.tif scan.ARW

# 3. 4-shot Sony Pixel Shift mode (pass 4 RAW files; merges sub-pixel RGB without Bayer demosaicing)
bin_out/neg_process --film_stock ektar100 --light_source tri -P srgb -o out.tif ps1.ARW ps2.ARW ps3.ARW ps4.ARW

# 4. 3-shot sequential R, G, B LED capture (use identity color correction — zero sensor crosstalk)
bin_out/neg_process --film_stock portra160 --light_source tri -r "1 0 0" -g "0 1 0" -b "0 0 1" -P srgb -o out.tif r.ARW g.ARW b.ARW

# 5. Manual override: pick a bracket (0..6), exposure (-E), or green/blue gains (--gain_g, --gain_b)
bin_out/neg_process --film_stock portra400 --light_source tri --target 4 -E 1.15 --gain_g 0.98 --gain_b 1.04 -P srgb -o out.tif scan.ARW
```

For manual darkroom-style grading in Capture One, omit `-P srgb` and adjust *only* the Levels tool before exporting to AdobeRGB/sRGB. See [`neg_process.cc`](neg_process.cc), [`multi_target_optimizer.cc`](multi_target_optimizer.cc), and [`dinov3_auto_solver.cc`](dinov3_auto_solver.cc) for implementation details.

# Licences

negicc is GPL-3.0 (`COPYING`). The DINOv3 backbone in `models/dinov3-small/backbone.onnx`, which `neg_process` packs into its binary, is Meta's DINOv3 and is under the DINOv3 License (`models/dinov3-small/LICENSE.md`): distribute it, or a `neg_process` built with it, only under that licence and with a copy of it.
