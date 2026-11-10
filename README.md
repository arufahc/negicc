# negicc

Converts colour negatives photographed with a digital camera into finished positives, with no manual editing. It
combines film profiles measured on an IT8 target at seven exposures with a vision model (DINOv3) that sets exposure
and colour balance the way a lab scanner's operator would.

* **Hands-off.** `neg_process` picks the exposure bracket, the exposure gain and the colour balance by itself. The
  result is ready to look at; the knobs are there if you want them.
* **Film-accurate colour.** Balance and exposure are applied in linear, scene-referred space, and the stock's own
  calibrated tone curves and 3D colour table do the rest. Nothing is stretched or guessed from a histogram.
* **Three stocks built in:** Kodak Portra 400, Portra 160 and Ektar 100, each profiled at 7 exposures from −3 to
  +3 EV under both a narrow-band RGB LED and a white D65 LED (42 ICC profiles in total).

![sample conversion](samples/sample.png)

**Try it in the browser:** [negicc-web](https://negicc-web-dmin4o4cra-uc.a.run.app) runs `neg_process` on a scan you
upload (a camera RAW, DNG, TIFF or JPEG) with the built-in profiles, and lets you regrade it with the brightness,
tint and warmth knobs. The background and the history of the method are in the original
[guide](https://docs.google.com/document/d/1NsfFPx5c7kxNRUhuGQBKRNnaN0c52SjLsZsyhLCj3OE/edit?usp=sharing) (Google Doc).

## Quick start

```bash
sudo apt-get install build-essential git python3 python3-pip   # plus libjpeg-dev for LIBRAW_JPEG=1
pip install torch transformers onnx                             # only for `make dinov3_model`
git clone --recursive https://github.com/arufahc/negicc.git && cd negicc
make dinov3_model && make -j$(nproc) neg_process                # add USE_CUDA=1 for an NVIDIA GPU
```

```bash
bin_out/neg_process --list_profiles          # the built-in bundles: portra400, portra160, ektar100, each for tri and d65

# A scan under a white LED (the common case): auto frame crop, auto film base, Auto's bracket, exposure and balance,
# fitted G/B curves -> 16-bit sRGB TIFF, plus out.tif.json with everything the conversion decided
bin_out/neg_process --film_stock portra400 --light_source d65 --roi auto --film_format 35mm -P srgb -o out.tif scan.ARW

# The same under a narrow-band RGB LED
bin_out/neg_process --film_stock portra400 --light_source tri -P srgb -o out.tif scan.ARW
```

LibRaw, Little-CMS 2, zlib, libtiff and libjpeg-turbo are built from the pinned submodules in `3rd_party/`
(see [3rd_party/README.md](3rd_party/README.md)); the DINOv3 model runs through ONNX Runtime
(`3rd_party/onnxruntime`) or TensorRT.

## How it works

### Capturing

Colour negative film stores its picture as dye densities meant to be read with Status M densitometry, not under D50
light, so the first job is to get a camera response that resembles Status M with good separation between the
channels (the [guide](https://docs.google.com/document/d/1NsfFPx5c7kxNRUhuGQBKRNnaN0c52SjLsZsyhLCj3OE/edit?usp=sharing)
above has the background). `neg_process` takes camera RAWs, DNGs and 16-bit linear TIFFs, captured two ways:

* **Narrow-band RGB LED** (`--light_source tri`). One RAW (or Sony's 4-shot pixel shift), followed by the bundle's
  3×3 crosstalk matrix that removes the Bayer filters' overlap. Three separate frames under red, green and blue LEDs
  work too (`-r "1 0 0" -g "0 1 0" -b "0 0 1"`: no crosstalk to remove).
* **White D65 LED** (`--light_source d65`), any CFA. LibRaw decodes with daylight white balance and the camera's
  matrix into linear Rec.2020. Before the matrix, the camera's channels are scaled so the scan's film base lands on
  the profile's (`film_base.balance_gains` in the sidecar): a warmer or cooler lamp is corrected before the channels
  are mixed, so dense areas do not clip to blue.

Even with narrow-band light and the crosstalk removed, the channels do not respond linearly to density, and not
alike: green and blue track each other, red has a different slope. The profiles correct that.

![Characteristic Curves](samples/characteristic_curves.png)

### Demosaicing

* **4-shot Sony pixel shift:** the four frames are merged into full RGB at every pixel. No interpolation.
* **Single-shot TriLED on a Sony RGGB sensor:** a grain-aware colour-ratio demosaic. Green is interpolated along edges
  with soft directional weights (hard horizontal/vertical switches turn grain into maze patterns); red and blue as
  R/G and B/G ratios rather than differences, which keeps the error uniform after the film inversion. Against the
  pixel-shift ground truth it is +1.22 dB over bilinear on the positive and halves the R−G zippering; +0.53 dB over
  LibRaw's DCB.
* **White light and other CFAs:** LibRaw. `-q 0` (bilinear, the default) takes about 1.2 s for a 61 MP RAW; `-q 4`
  (DCB) is better by +0.69 dB and takes about 6 s.

| Algorithm | Raw MAE ↓ | Positive PSNR (dB) ↑ | Positive ΔE₀₀ mean / p95 ↓ | R−G zipper ↓ |
| :--- | :---: | :---: | :---: | :---: |
| 4-shot pixel shift (ground truth) | 0.00 | ∞ | 0.00 / 0.00 | 142.4 |
| **Grain-aware colour-ratio** (TriLED default) | **80.81** | **40.38** | **1.69 / 6.86** | 56.5 |
| LibRaw DCB (`-q 4`) | 79.60 | 39.85 | 1.76 / 7.21 | 71.7 |
| LibRaw VNG (`-q 1`) | 80.19 | 39.96 | 1.75 / 7.12 | 58.6 |
| PPG / AHD (`-q 2` / `-q 3`) | 80.61 | 40.03 | 1.73 / 6.97 | 55.3 (grain maze) |
| Bilinear (`-q 0`) | 89.15 | 39.16 | 1.90 / 7.87 | 115.0 |

Raw MAE is in 14-bit LSB; PSNR and ΔE₀₀ are measured on the inverted positive.

![Demosaicing Algorithm Comparison](samples/demosaic_comparison.png)

### The profiles: seven exposures per stock

An IT8.7/2 target (coloraid.de `R190808`) is photographed onto each stock at seven exposures, −3 to +3 EV in 1-stop
steps, and each capture is scanned under both lights.

![IT8 used for profiling](samples/it8.png)

One target only calibrates the 7.2 stops up to its white patch. Fitting the seven brackets together measures each
bracket's true exposure offset and traces the film's shoulder, so the set covers 12 to 13 stops of scene range with
up to 3 stops of calibrated headroom above white. Each bracket becomes an ICC profile: 16-bit tone curves that
linearise the density response at that exposure, then a 45×45×45 3D colour table into CIE Lab. The brighter
patches of the over-exposed brackets are blended into each table, so the colour gamut and highlight saturation run
smoothly to pure white instead of stopping at the chart's L* 89.

![TRC Curves](samples/trc_curves.png)
![ICC profile](samples/icc.png)

This is what sets negicc apart from a single fixed LUT (the Cineon-style workflow) or a histogram stretch: a fixed
LUT bakes tone and colour together at one exposure, and a frame two stops over lands on the wrong part of the
film's curves. Here the bracket near the frame's exposure is used, and between brackets the exposure and balance are
fitted:

![Gamut Volume](samples/gamut_volume.png)

## White-light conversions of scans from the web

The profiles were measured with one camera (a Sony A7R IV) and one white LED. Scans from other people, other cameras
and other lamps convert with the same bundle: the camera's channels are balanced on the scan's own film base before
the camera matrix, so the lamp's colour is taken out, and Auto does the rest. Three examples from the
[pixls.us](https://discuss.pixls.us) forum, scan on the left as the camera saw it, `neg_process` on the right
(`--film_stock portra400 --light_source d65 -P srgb`; the frame crop was given, everything else is Auto's):

![Mauritius mountain with river mouth](samples/web_5e74adc30c.jpg)

*Canon EOS 5DS R.* [Mauritius Mountain with River Mouth](https://discuss.pixls.us/t/50730/1), posted by gsinha,
2025-06-16.

![Hands holding a marble](samples/web_72a0a8b2e5.jpg)

*Nikon D810.* [Negative inversion challenge! Skin tones and greens](https://discuss.pixls.us/t/59155/1), posted by
DanielE, 2026-07-11. Converted with `--fit_curves off`: Auto's bracket, exposure and balance alone.

![Fête du Travail](samples/web_d216d498d5.jpg)

*Canon EOS R.* [Fête du Travail](https://discuss.pixls.us/t/45538/1), posted by jonathanBieler, 2024-09-13.

These scans were picked from what people have shared publicly to show the converter on cameras and lamps it was not
calibrated with. No copyright is claimed on them: the RAW files and the pictures belong to the photographers who
posted them, at the links above. If you are one of them and would rather not have your scan here, open an issue and
it will be removed.

### How accurate are the profiles


A generic single curve or 3×3 matrix leaves 5–10+ ΔE₀₀ on saturated colours, because it cannot
untangle the dyes' non-linear coupling. The per-stock tables fit their own patches to a mean of 0.5–0.8 ΔE₀₀ from
−1 to +3 EV (0.6–1.1 at −2 and −3 EV), and unseen patches, by 2-fold cross-validation on the 264 colour patches, to a
median of 1.4–2.3 (under 2 is barely visible). The TriLED bundles:

| Bracket | Portra 400 own (median) | held-out | Portra 160 own | held-out | Ektar 100 own | held-out |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| −3 EV | 0.79 (0.50) | 2.70 (1.91) | 1.09 (0.47) | 2.35 (1.61) | 1.11 (0.49) | 3.05 (1.87) |
| −2 EV | 0.56 (0.41) | 2.04 (1.38) | 0.82 (0.42) | 2.09 (1.55) | 0.61 (0.39) | 2.68 (1.52) |
| −1 EV | 0.53 (0.40) | 2.38 (1.40) | 0.59 (0.39) | 1.96 (1.40) | 0.51 (0.36) | 2.79 (1.55) |
| 0 EV | 0.55 (0.42) | 2.62 (1.48) | 0.56 (0.44) | 2.30 (1.37) | 0.62 (0.49) | 3.25 (1.84) |
| +1 EV | 0.54 (0.40) | 2.75 (1.46) | 0.62 (0.43) | 2.65 (1.58) | 0.63 (0.49) | 3.52 (1.75) |
| +2 EV | 0.59 (0.43) | 2.92 (1.58) | 0.67 (0.49) | 2.67 (1.62) | 0.72 (0.59) | 4.01 (2.15) |
| +3 EV | 0.69 (0.54) | 3.08 (1.77) | 0.70 (0.53) | 2.82 (1.78) | 0.83 (0.65) | 3.72 (2.26) |

Mean ΔE₀₀ (median in brackets), from ArgyllCMS `profcheck` on all 288 patches (own) and from the cross-validation
(held-out). The D65 bundles are at 0.6–1.1 own mean from −1 to +3 EV and 1.3–2.2 held-out median. No profile is
de-glared: removing the chart's veiling glare before fitting was tried, did not help overall, and the bundles ship
as measured.

### The parameter search

What Auto solves for a frame are a bracket and three numbers: the exposure gain E and the green and blue gains g, b,
applied to the scan's linear channels before the bracket's profile. A grey-world rule would balance a sunset to grey;
instead the target comes from Meta's DINOv3 vision model with a small head trained on Fujifilm Frontier and Noritsu
lab scans (collected from Flickr and Photohito), which says for each 16×16 patch of the picture what Lab colour it
should have and how sure it is, for sky, foliage, skin, stone and shadow in context:

![DINOv3 Intent Field and Convergence](samples/dinov3_intent.png)

The search, in brief:

1. **Frame and film base.** The frame is found on the RAW's mosaic (`--roi auto`, with `--film_format` to say which
   format, `auto` to try them all); the unexposed film base is read from the rebate next to it, or from the thinnest
   film when no rebate is in view. On white light the camera's channels are balanced on that base before the matrix.
2. **Which brackets can hold the scene.** From the frame's density range and each bracket's calibrated range the
   search knows the exposures at which the scene stays inside a bracket's calibration (its enclosure), and meters a
   starting E per bracket for a mid-grey median.
3. **Screening.** Each bracket is rendered at its metered start and scored against the target field; the best two
   go on.
4. **Fixed point.** On each of those, E, g and b are searched against the field, the frame re-rendered, the field
   re-predicted, and the search repeated until the parameters hold still or the residual stops falling. The bracket
   with the lowest residual wins. Highlights that would clip pull E back or step the bracket up.

On the frame below, the first render has a cyan-green cast and a median patch error of 21.6 ΔE; the search brings it
to 8.7 and a natural daylight balance:

![DINOv3 Patch Residual Error Minimization](samples/dinov3_residuals.png)

### The fitted curves on top of E, g, b

E, g and b are three gains: they move each channel as a whole. On white light that is not always enough, because a
lamp and a stock the profile was not measured with change the shape of a channel's response, not only its level, and
a gain then balances the mid-tones while the shadows or the highlights keep a cast. So, after the search, the green
and blue input curves of the winning bracket are fitted jointly with g and b so that the frame's scene-neutral content
renders grey, both its average and its trend from shadows to highlights, with E re-solved at every step so the
picture's median lightness stays where Auto put it.

The scene-neutral content comes from a small neutral-patch head on DINOv3's tokens
([models/neutral_head](models/neutral_head/README.md)), which gives every grid cell a p(neutral). Every cell of four
lightness bands is an anchor, weighted by p squared: a cell the head is sure of counts fully, an uncertain one a
little, and there is no threshold and no minimum count. The sum of p squared is the fit's evidence: its confidence in
the anchors grows with it, and so does the weight of each band. With no evidence (`--neutral_head off`, or a frame
where the head finds nothing neutral) there are no anchors, and the fit keeps Auto's colour: Auto's g and b, no bend.
Two rules set how far the fit follows the anchors. The shadow-to-highlight trend is only partly removed when the
anchors span a narrow range of lightness (a trend measured over an L* spread well under 6 is mostly kept, not
extrapolated into a bend). And when the anchors share an offset larger than both 6 (in a\*b\*) and their own
scatter, the fit treats it as a cast and removes it at full strength rather than scaled down by its confidence.

Each curve is a bend of the channel's density response of at most ±35 % slope and 2.2 stops of departure from the
profile's own curve, held to a plain offset below the frame's densest cell, and never above the film base; a fit that
would take E out of the bracket's range, or a curve that fails those bounds, falls back to Auto's plain result.
`--fit_curves off` keeps the plain result.

The sidecar's `input_curves` records the curves, `input_curves.remeter.neutral` the anchors' evidence and what the two
rules did (`evidence`, `span_L`, `slope_share`, `endpoints`, `auto_ab`, `spread`, `offset_gate`).

## More examples

```bash
# 4-shot Sony pixel shift (four RAWs; merged, not demosaiced)
bin_out/neg_process --film_stock ektar100 --light_source tri -P srgb -o out.tif ps1.ARW ps2.ARW ps3.ARW ps4.ARW

# Three frames under red, green and blue LEDs (identity crosstalk matrix)
bin_out/neg_process --film_stock portra160 --light_source tri -r "1 0 0" -g "0 1 0" -b "0 0 1" -P srgb -o out.tif r.ARW g.ARW b.ARW

# White light with LibRaw's DCB demosaic and a 6x7 frame
bin_out/neg_process --film_stock portra400 --light_source d65 -q 4 --roi auto --film_format 6x7 -P srgb -o out.tif scan.ARW

# Your own choices: bracket 0..6, exposure, green and blue gains (any gain given turns Auto off)
bin_out/neg_process --film_stock portra400 --light_source tri --target 4 -E 1.15 --gain_g 0.98 --gain_b 1.04 -P srgb -o out.tif scan.ARW
```

## Options

`bin_out/neg_process --help` prints every option. The ones you will meet:

| Option | Meaning |
| :--- | :--- |
| `--film_stock`, `--light_source tri\|d65` | A built-in bundle (`--list_profiles` lists them). |
| `-p`, `--film_profile` | A bundle JSON (or its directory), or a single ICC profile. |
| `-P`, `--colorspace srgb\|srgb-g10\|<icc>` | The output encoding. Only `srgb` applies the highlight roll-off; the input curves need `srgb` or `srgb-g10`. |
| `-o`, `--output` | The TIFF; `<output>.json` is its sidecar unless `--no-metadata` is given. |
| `--target N` | The bracket (0-based index or name). Without `-E` and the gains: the DINOv3 search on that bracket only. |
| `-E`, `--exposure_comp` | The exposure gain. With a bundle and given gains it defaults to the centre of the bracket's range; with an ICC profile, to 1. Given alone, the search keeps it. |
| `--gain_g`, `--gain_b` | Green and blue gains relative to red (default 1). Either given turns the search off. |
| `--brightness`, `--tint`, `--warmth` | Grey-point knobs after the search (or on given values); reuse `<output>.json` from the same inputs. |
| `--fit_curves auto\|off\|on\|<numbers>` | G/B input curves (white light, `-P srgb` or `srgb-g10`). `auto` (default): on where it applies. `off`: none. `on`: an error where it does not apply. Eight or ten numbers: replay a sidecar's curves, with `--target`, `-E`, `--gain_g` and `--gain_b`. |
| `--roi auto [h\|v]` or `--roi x y w h [rot [hflip [vflip]]]` | The frame: detected, or given in sensor pixels. |
| `--film_format 35mm\|645\|6x6\|6x7\|6x9\|auto` | The format `--roi auto` looks for (default 35mm); with a given frame, the format the film base is read around (default: every format). |
| `--rot`, `--hflip`, `--vflip` | Orientation of the output. |
| `-q`, `--quality` | Demosaic: −1 auto (grain-aware on a Sony CFA, else bilinear), or a LibRaw quality (4: DCB). |
| `-r`, `-g`, `-b` | Rows of the crosstalk matrix (default: the bundle's). |
| `-C`, `--no_crop` | Keep the RAW's full area instead of its default crop. |
| `--flat` | A flat-field `.npz` measured on the empty gate. |
| `--dino` | An intent adapter (`.onnx` or its directory) instead of the built-in one for the stock. |
| `--neutral_head on\|off\|<file>` | The neutral-patch head that weights the curve fit's neutral anchors: the built-in one (default), none (no anchors: the fit keeps Auto's colour), or another `neutral_head.bin`. |
| `--dino_backend auto\|cpu\|gpu`, `--dino_iters` | DINOv3's backend and the search's iterations. |

## Hardware the profiles were made with

Sony A7R IV (`ILCE-7RM4`), a [Shoebox Photos BCS](https://www.shoeboxphotosbcs.com) RGB LED light for the TriLED
bundles and a white D65 LED panel for the D65 ones, and the coloraid.de `R190808` IT8.7/2 target. Other cameras and
lamps use the same bundles, as the web scans above show.

## Licences

negicc is GPL-3.0 (`COPYING`). The DINOv3 backbone in `models/dinov3-small/backbone.onnx`, which `neg_process` packs
into its binary, is Meta's DINOv3 under the DINOv3 License (`models/dinov3-small/LICENSE.md`): distribute it, or a
`neg_process` built with it, only under that licence and with a copy of it. The sample scans in `samples/web_*.jpg`
are other people's photographs, credited above; no copyright is claimed on them.
