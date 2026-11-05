# Objective

NegICC creates ICC profiles and converts color negatives captured with a digital camera into consistent, color-accurate positive images with **no manual editing required**, combining physical multi-bracket film profiling with a **Vision Transformer (DINOv3) as an automatic color grader**. The ICC profiles and conversion tool (`neg_process`) have the following properties:
* **Zero-touch automatic color grading:** Uses a Vision Transformer (**DINOv3**) trained on Fujifilm Frontier and Noritsu lab scans to automatically select the exposure bracket, exposure gain, and color balance so converted negatives are ready for display out of the box without manual tweaking
* **Preserves authentic film dye characteristics:** Allows automatic or manual color balance and exposure adjustments in linear scene-referred space before applying the film stock's calibrated tone curves and 3D color LUT
* **Includes 3 manually profiled film stocks** (**Kodak Portra 400**, **Portra 160**, and **Ektar 100**), each photographed and calibrated across **7 exposure brackets** (`-3 EV` to `+3 EV`) under both narrowband RGB LED and white D65 light (**42 ICC profiles total**)

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
  * *Single-shot RAW/DNG or 4-shot Sony Pixel Shift:* Single-shot Sony RGGB scans use grain-aware color-ratio demosaicing; 4-shot Pixel Shift merges the frames directly (see [Demosaicing](#demosaicing)). Both are followed by the bundle's 3×3 sensor crosstalk correction matrix (or custom `-r`, `-g`, `-b` rows) to remove Bayer CFA channel overlap.
  * *3-shot monochromatic RGB (`r.raw g.raw b.raw`) or pre-separated TIFF/DNG:* Capturing 3 separate frames under Red, Green, and Blue LEDs isolates the channels physically with zero sensor crosstalk—use the **identity color correction** (`-r "1 0 0" -g "0 1 0" -b "0 0 1"`).
* **White D65 light (`--light_source d65`) & Non-RGGB CFAs:** Single-shot capture under a standard D65 white LED panel decodes through LibRaw's daylight white balance and camera matrix into linear Rec.2020 with **no crosstalk matrix needed**. LibRaw defaults to bilinear (`-q 0`, ~1.2 s for a 61 MP RAW); pass `-q 4` (LibRaw DCB) for higher quality (see [Demosaicing](#demosaicing)).

### Demosaicing

How the sensor data is turned into RGB depends on the capture mode:

* **4-shot Sony Pixel Shift** (`ps1.ARW ps2.ARW ps3.ARW ps4.ARW`): the four shifted frames are merged directly into full RGB at every pixel. No interpolation.
* **Single-shot TriLED on Sony RGGB** (`--light_source tri`): grain-aware color-ratio demosaicing.
  * Green is interpolated along edges using soft directional weights instead of a hard horizontal/vertical switch. Hard switches (PPG/AHD) turn film grain into maze-like artifacts.
  * Red and blue are interpolated as R/G and B/G ratios instead of R-G and B-G differences. This keeps errors uniform after the -log(T) film inversion.
  * Result: +1.22 dB positive PSNR and half the R-G zippering of bilinear, and +0.53 dB over LibRaw DCB.
* **White-light (D65) and other CFAs**: decoded by LibRaw.
  * `-q 0` (bilinear, default): fastest, about 1.2 s for a 61 MP RAW.
  * `-q 4` (LibRaw DCB, recommended for quality): +0.69 dB positive PSNR over bilinear, about 6.0 s single-threaded for a 61 MP RAW.

Benchmark against the 4-shot Pixel Shift ground truth (PSNR and ΔE₀₀ are measured on the inverted positive):

| Algorithm | Raw MAE ↓ | Raw PSNR (dB) ↑ | Positive PSNR (dB) ↑ | Positive ΔE₀₀ mean / p95 ↓ | Acutance | R-G Zipper ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 4-shot Pixel Shift (ground truth) | 0.00 | ∞ | ∞ | 0.00 / 0.00 | 100.0% | 142.4 |
| **Grain-aware color-ratio** (TriLED default) | **80.81** | **39.73** | **40.38** (+1.22) | **1.69 / 6.86** | 102.0% | 56.5 |
| LibRaw DCB (`-q 4`) | 79.60 | 39.73 | 39.85 (+0.69) | 1.76 / 7.21 | 95.8% | 71.7 |
| LibRaw VNG (`-q 1`) | 80.19 | 39.72 | 39.96 (+0.80) | 1.75 / 7.12 | 102.2% | 58.6 |
| PPG / AHD (`-q 2` / `-q 3`) | 80.61 | 39.75 | 40.03 (+0.87) | 1.73 / 6.97 | 100.5% (grain maze) | 55.3 |
| Malvar-He-Cutler | 86.68 | 39.07 | 38.89 (-0.27) | 1.85 / 7.58 | 100.1% (ringing) | 73.2 |
| Bilinear (`-q 0`, baseline) | 89.15 | 38.39 | 39.16 | 1.90 / 7.87 | 107.3% (aliased) | 115.0 |

Acutance is edge sharpness relative to the 4-shot result. Raw MAE is in 14-bit LSB.

![Demosaicing Algorithm Comparison](samples/demosaic_comparison.png)

### 7-Bracket Tone Curves (TRC) & Highlight Roll-Off

A single IT8 target only calibrates tones across its own **7.2-stop** range up to the chart's white patch. To capture the full latitude of color negative film, each stock is photographed across **7 exposure brackets from `-3 EV` to `+3 EV`** in 1-stop steps. Fitting all 7 brackets together measures each bracket's exact exposure offset and traces the film's true highlight shoulder—giving **12 to 13 stops of total scene dynamic range** and **up to +3 stops of calibrated highlight headroom** above normal white:

![TRC Curves](samples/trc_curves.png)

The final step converts linearized RGB values into CIE XYZ / Lab using a 16-bit tone curve and a 45×45×45 3D color lookup table (cLUT) encoded as an ICC profile:

![ICC profile](samples/icc.png)

# ICC Profile Description & Expanded Highlight Gamut

NegICC is built around **multi-bracket physical film profiling** with separate tone curves and color lookup tables. Each film stock is photographed on an IT8.7/2 target across seven exposure brackets (`-3 EV` to `+3 EV`) under two light sources, producing seven ICC profiles per film and light source. Within each profile, 16-bit 1D tone curves first linearize the film's density response at that exposure level, and a 45×45×45 3D color lookup table (cLUT) then maps linearized RGB into CIE Lab.

Because a single target capture contains no color patches brighter than its white patch (`L* ≈ 89`), standard ICC profiling leaves the upper highlight range uncalibrated. NegICC solves this by borrowing the brighter patches from the over-exposed brackets and blending them into the cLUT, extending the calibrated 3D color gamut and highlight saturation smoothly all the way to pure white (`L* = 100`) without amplifying film grain.

This makes NegICC much more adaptable than generic histogram stretching or the **Cineon workflow**, which converts negative scans to density and applies a single fixed 3D LUT. Because a Cineon LUT bakes the tone curve and color mapping together at one fixed exposure, it does not adapt well when a frame is over- or under-exposed—for example, at **+2 stops overexposure**, shifting density into a single LUT pushes midtones and highlights into the wrong part of the film's shoulder and dye curves. With seven bracketed profiles, `neg_process` instead selects the profile calibrated near that exposure level, and for exposures between brackets it uses **gamut enclosure fitting** and the **DINOv3 vision model** to find the optimal exposure gain and color balance (see [DINOv3 Automatic Color Grader & Iterative Convergence](#dinov3-automatic-color-grader--iterative-convergence) below):

![Gamut Volume](samples/gamut_volume.png)

### Why Dedicated Per-Stock Profiles? (Own-Sample vs. Held-Out Color Accuracy)

Generic negative inversion tools apply a single curve or 3×3 matrix to every film stock, which cannot untangle non-linear dye cross-coupling and typically leaves `5–10+ ΔE₀₀` color errors across saturated colors and highlights. A dedicated 7-bracket 45×45×45 3D cLUT per film stock fits its own IT8 patches to **`0.51–0.83 ΔE₀₀` mean (`0.36–0.65` median) from `-1 EV` to `+3 EV`**, and to `0.56–0.82` and `0.79–1.11` mean at the two under-exposed brackets (`-2 EV` and `-3 EV`). On unseen held-out color patches the error is **`1.4–2.3 ΔE₀₀` median** (`< 2.0 ΔE₀₀` is barely perceptible to the human eye).

No shipped profile is de-glared. Removing the veiling glare of the outdoor IT8 shot (CS-100A spot readings of every patch, `0.27 %` of white as fitted) before fitting was tried on all six bundles on 2026-11-01: on the D65 bundles it raised held-out mean error by `0.12–1.07` from `-3 EV` to `-1 EV` and they went back on 2026-11-03; on the TriLED bundles it raised held-out mean error by `0.19–0.53` at `-3 EV`, lowered it by `0.05–0.22` at `+1 EV` to `+3 EV`, and left a step in Target 7's green TRC beyond negicc-station's TRC gate, and they went back on 2026-11-04 (the experiment was not complete). The table is for the bundles as shipped.

Own-sample errors are the profiles' fit on all 288 patches (ArgyllCMS `profcheck`, stored in each `bundle.json`); held-out errors come from 2-fold cross-validation that rebuilds each profile with half of the 264 color patches held out at a time (the grey patches and the brighter brackets' highlight samples stay in), using negicc-station's profile build. Both are for the TriLED bundles. The D65 bundles are within `0.63–1.05` own-sample mean from `-1 EV` to `+3 EV` (`0.74–1.45` at the two under-exposed brackets) and `1.3–2.2` held-out median.

| Bracket Target (Exposure) | Portra 400 Own `ΔE₀₀` avg (med) | Portra 400 Held-Out avg (med) | Portra 160 Own `ΔE₀₀` avg (med) | Portra 160 Held-Out avg (med) | Ektar 100 Own `ΔE₀₀` avg (med) | Ektar 100 Held-Out avg (med) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Target 1`** (`-3 EV`) | `0.79` (`0.50`) | `2.70` (`1.91`) | `1.09` (`0.47`) | `2.35` (`1.61`) | `1.11` (`0.49`) | `3.05` (`1.87`) |
| **`Target 2`** (`-2 EV`) | `0.56` (`0.41`) | `2.04` (`1.38`) | `0.82` (`0.42`) | `2.09` (`1.55`) | `0.61` (`0.39`) | `2.68` (`1.52`) |
| **`Target 3`** (`-1 EV`) | `0.53` (`0.40`) | `2.38` (`1.40`) | `0.59` (`0.39`) | `1.96` (`1.40`) | `0.51` (`0.36`) | `2.79` (`1.55`) |
| **`Target 4`** (`0 EV` center) | **`0.55` (`0.42`)** | **`2.62` (`1.48`)** | **`0.56` (`0.44`)** | **`2.30` (`1.37`)** | **`0.62` (`0.49`)** | **`3.25` (`1.84`)** |
| **`Target 5`** (`+1 EV`) | `0.54` (`0.40`) | `2.75` (`1.46`) | `0.62` (`0.43`) | `2.65` (`1.58`) | `0.63` (`0.49`) | `3.52` (`1.75`) |
| **`Target 6`** (`+2 EV`) | `0.59` (`0.43`) | `2.92` (`1.58`) | `0.67` (`0.49`) | `2.67` (`1.62`) | `0.72` (`0.59`) | `4.01` (`2.15`) |
| **`Target 7`** (`+3 EV`) | `0.69` (`0.54`) | `3.08` (`1.77`) | `0.70` (`0.53`) | `2.82` (`1.78`) | `0.83` (`0.65`) | `3.72` (`2.26`) |

### DINOv3 Automatic Color Grader & Iterative Convergence

Instead of relying on simple grey-world heuristics, `neg_process` uses Meta's **DINOv3** vision model as an **automatic color grader** trained on real **Fujifilm Frontier and Noritsu lab scans** collected from Flickr and Photohito. Because DINOv3 understands scene semantics, it recognizes objects in context (sky, foliage, skin tones, stone, shadows) and predicts the **target Lab color** and **confidence weight** that each 16×16 patch of the photo should have based on learned lab-scan examples:

![DINOv3 Intent Field and Convergence](samples/dinov3_intent.png)

To convert a frame automatically:
1. **Auto Film-Base & Frame Detection:** Automatically samples the unexposed orange film base from the scan border (rebate or thinnest film; *TODO: add support for reading the film base from a separate unexposed film capture*) and crops the film frame (`--roi auto --film_format 35mm|645|6x6|6x7|6x9|auto`; `auto` tries every format). With `--roi auto` the format defaults to 35mm; with a given frame (`--roi x y w h`, or none) the film base's frame detection tries every format unless `--film_format` names one, as negicc-station does.
2. **Bracket Screening & Iterative Error Minimization:** Screens the 7 brackets against the predicted target Lab patch field, then iteratively adjusts exposure (`E`) and green/blue color balance (`g`, `b`) on the top brackets—re-evaluating the DINOv3 target Lab field at each step—until the patch Lab residual error converges to a minimum (with automatic highlight pullback and bracket step-up if highlights clip). As shown below on a second scene (Lake Ashi & Mt. Fuji), an unbalanced initial render with a heavy cyan/green cast produces large patch Lab residuals across the sky and water (`21.6 ΔE` median patch error, score `6.30`); DINOv3 drives the residual error down by **72%** (`8.7 ΔE` median, score `1.79`) to recover natural daylight balance:

![DINOv3 Patch Residual Error Minimization](samples/dinov3_residuals.png)

# Requirements & Hardware

`neg_process` builds LibRaw, Little-CMS 2, zlib, libtiff, and libjpeg-turbo directly from the pinned submodules in `3rd_party/` (see [`3rd_party/README.md`](3rd_party/README.md)), and uses ONNX Runtime (`3rd_party/onnxruntime`) with the DINOv3 model (`make dinov3_model`, which requires Python `torch`, `transformers`, and `onnx`).
* **Hardware & Profile Calibration:** The built-in profiles were created with a Sony A7RM4 (`ILCE-7RM4`), [Shoebox Photos BCS](https://www.shoeboxphotosbcs.com) RGB LED light, and an IT8.7/2 target (`R190808` from coloraid.de, 7 bracketed exposures per film stock). They can also be used with color negatives captured under D65 white light (`--light_source d65`; *TODO: add samples from web images*).

```bash
sudo apt-get install build-essential git python3 python3-pip   # plus libjpeg-dev if building with LIBRAW_JPEG=1
pip install torch transformers onnx
git clone --recursive https://github.com/arufahc/negicc.git && cd negicc
make dinov3_model && make -j$(nproc) neg_process               # add USE_CUDA=1 for NVIDIA GPU acceleration
```

# Usage Example

```bash
bin_out/neg_process --list_profiles   # list built-in 7-bracket bundles (portra400, portra160, ektar100 for tri/d65)

# 1. Single-shot Trichromatic LED scan (auto film-base, auto bracket & DINOv3 color grading -> 16-bit sRGB TIFF)
bin_out/neg_process --film_stock portra400 --light_source tri -P srgb -o out.tif scan.ARW

# 2. Single-shot D65 white-light scan with automatic 6x7 frame crop (add -q 4 for higher-quality LibRaw DCB demosaicing)
bin_out/neg_process --film_stock portra400 --light_source d65 --roi auto --film_format 6x7 -P srgb -o out.tif scan.ARW
bin_out/neg_process --film_stock portra400 --light_source d65 -q 4 --roi auto --film_format 6x7 -P srgb -o out_dcb.tif scan.ARW

# 3. 4-shot Sony Pixel Shift mode (pass 4 RAW files; merges the shifted frames directly, no demosaicing)
bin_out/neg_process --film_stock ektar100 --light_source tri -P srgb -o out.tif ps1.ARW ps2.ARW ps3.ARW ps4.ARW

# 4. 3-shot sequential R, G, B LED capture (use identity color correction — zero sensor crosstalk)
bin_out/neg_process --film_stock portra160 --light_source tri -r "1 0 0" -g "0 1 0" -b "0 0 1" -P srgb -o out.tif r.ARW g.ARW b.ARW

# 5. Manual override: pick a bracket (0..6), exposure (-E), or green/blue gains (--gain_g, --gain_b)
bin_out/neg_process --film_stock portra400 --light_source tri --target 4 -E 1.15 --gain_g 0.98 --gain_b 1.04 -P srgb -o out.tif scan.ARW

# 6. White light: fit g, b and G/B input curves jointly after the search, keeping Auto's lightness (-P srgb or
#    srgb-g10; the sidecar's input_curves records them; its cli_equivalent replays them with
#    --fit_curves ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB (u_loG,u_loB appended for a held curve), --target, -E,
#    --gain_g and --gain_b, all four required with recorded curves; the older aG,kG,aB,kB,refG,refB,u_medG,u_medB
#    order without the prefix is still read when aG is not negative)
bin_out/neg_process --film_stock portra400 --light_source d65 --fit_curves on -P srgb -o out.tif scan.ARW
```

## Options

`bin_out/neg_process --help` prints every option. In short:

| Option | Meaning |
| :--- | :--- |
| `--film_stock`, `--light_source tri\|d65` | A built-in bundle (`--list_profiles` lists them). |
| `-p`, `--film_profile` | A bundle JSON (or its directory), or a single ICC profile. |
| `-P`, `--colorspace srgb\|srgb-g10\|<icc>` | The output encoding. Only `srgb` applies the highlight roll-off; the input curves need `srgb` or `srgb-g10`. |
| `-o`, `--output` | The TIFF; `<output>.json` is its sidecar unless `--no-metadata` is given. |
| `--target N` | The bracket (0-based index or name). Without `-E` and the gains: the single-bracket DINOv3 search on that bracket. |
| `-E`, `--exposure_comp` | E as negicc-station gives it. With a bundle and given gains it defaults to the centre of the bracket's exposure range as the bundle search meters it; with an ICC profile, to 1. Given alone, the search keeps it. |
| `--gain_g`, `--gain_b` | G and B gains relative to R (default 1). Either given turns the DINOv3 search off. |
| `--brightness`, `--tint`, `--warmth` | Grey-point knobs after the search (or on given values); reuse `<output>.json` when it was written by this build from the same inputs and arguments. |
| `--fit_curves off\|auto\|on\|<numbers>` | G/B input curves (white light, `-P srgb` or `srgb-g10`). `auto`: on wherever on applies. `on`: fit after the bundle search. Eight or ten comma-separated numbers: replay a sidecar's curves, with `--target`, `-E`, `--gain_g` and `--gain_b`. |
| `--roi auto [h\|v]` or `--roi x y w h [rot [hflip [vflip]]]` | The frame: detected, or given in sensor pixels. |
| `--film_format 35mm\|645\|6x6\|6x7\|6x9\|auto` | The format `--roi auto` finds (default 35mm); with a given frame, the format the film base is read around (default: every format). |
| `--rot`, `--hflip`, `--vflip` | Orientation of the output. |
| `-q`, `--quality` | Demosaic: -1 auto (grain-aware on Sony CFA, else bilinear), or a LibRaw quality (4: DCB). |
| `-r`, `-g`, `-b` | Rows of the crosstalk correction (default: the bundle's). |
| `-C`, `--no_crop` | Keep the RAW's full area instead of its default crop. |
| `--flat` | A flat-field `.npz` measured on the empty gate. |
| `--dino` | An intent adapter (`.onnx` or its directory) instead of the built-in one for the stock. |
| `--dino_backend auto\|cpu\|gpu`, `--dino_iters` | DINOv3's backend and the search's iterations. |

# Licences

negicc is GPL-3.0 (`COPYING`). The DINOv3 backbone in `models/dinov3-small/backbone.onnx`, which `neg_process` packs into its binary, is Meta's DINOv3 and is under the DINOv3 License (`models/dinov3-small/LICENSE.md`): distribute it, or a `neg_process` built with it, only under that licence and with a copy of it.
