# negicc

Turns a photo of a colour negative into a finished picture, with no editing. You point a camera at the negative on
a light, give `neg_process` the file, and get back a positive that looks like a lab scan.

* **Hands-off.** It chooses exposure, colour balance and tone by itself. The result is ready to look at; brightness,
  tint and warmth knobs are there if you want them.
* **The film's own colour.** Each film stock was measured on a colour chart at seven exposures, so colour comes from
  how that film actually behaves, not from stretching a histogram.
* **A model that looks at the picture.** A vision model (DINOv3) decides how bright the picture should be and what its
  colours should look like, the way a lab operator would.
* **Three stocks built in:** Kodak Portra 400, Portra 160 and Ektar 100, for both white and RGB LED light.

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

A colour negative is three dye layers on an orange base. Turning a photo of one into a good positive takes three
things: a clean reading of each layer, a description of how this particular film responds, and a decision about
exposure and colour for this particular picture. negicc measures the first two and lets a vision model make the
third.

### Photographing the negative

`neg_process` reads camera RAWs, DNGs and 16-bit linear TIFFs, lit one of two ways:

* **A white LED** (`--light_source d65`). The common case, and it works with any camera. The scan's own film base is
  used to take the lamp's colour out before anything else happens, so a warmer or cooler light than the one the
  profiles were made with is fine.
* **A narrow-band RGB LED** (`--light_source tri`). Red, green and blue light sit on the three dyes, which separates
  the layers better. One RAW is enough (Sony's 4-shot pixel shift also works, as do three separate frames under red,
  green and blue).

Even with a clean reading the three layers do not respond in step with each other, which is why a simple inversion
looks wrong and why the profiles below exist.

![Characteristic Curves](samples/characteristic_curves.png)

For single-shot RGB-LED scans on Sony sensors negicc uses its own demosaic, built to keep film grain from turning
into maze patterns. Against a 4-shot pixel-shift capture of the same frame it is about 1.2 dB better than bilinear
and half a dB better than LibRaw's DCB. Everything else is decoded by LibRaw (`-q` picks the method).

![Demosaicing Algorithm Comparison](samples/demosaic_comparison.png)

### The film profiles

Each stock was profiled by photographing a colour chart onto it at seven exposures, from three stops under to three
stops over, and scanning every one under both lights.

![IT8 used for profiling](samples/it8.png)

A single chart exposure only describes a narrow slice of what film can hold. Seven of them, fitted together, follow
the film all the way up its shoulder: the set covers 12 to 13 stops of scene brightness. Each exposure becomes an ICC
profile, with tone curves that straighten out the film's response and a 3D colour table that untangles the dyes.

![TRC Curves](samples/trc_curves.png)
![ICC profile](samples/icc.png)

This is the difference from a single fixed LUT or a histogram stretch. A fixed LUT is right at one exposure; a frame
shot two stops over lands on the wrong part of the film's curves. negicc picks the profile nearest the frame's real
exposure and fits the remainder.

![Gamut Volume](samples/gamut_volume.png)

How good are they? On the chart's own patches the profiles are within about 0.5 to 0.8 ΔE₀₀ on average, and on
patches held out of the fit the typical error is 1.4 to 2.3, around the point where a difference becomes visible. A
generic curve or 3×3 matrix leaves 5 to 10 on saturated colours. The full tables are in
[MEASUREMENTS.md](MEASUREMENTS.md).

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

### Choosing exposure and colour: DINOv3 and three small heads

The profiles say how the film behaves. They cannot say how bright this picture should be or what its colour balance
is; a lab operator decides that by looking. negicc looks with Meta's DINOv3 vision model. The model itself is used as
it comes; on top of it sit three small heads, each trained on lab scans to answer one question about a picture.

| Head | Answers | Used for |
|---|---|---|
| Intent | What colour should each part of this picture be? | Picking the profile, exposure and colour balance |
| Neutral ([models/neutral_head](models/neutral_head/README.md)) | Which parts are neutral surfaces? | Removing casts that change from shadows to highlights |
| Tone ([models/tone_head](models/tone_head/README.md)) | How deep, bright and wide should the tones be? | Giving flat negatives proper blacks and highlights |

![DINOv3 Intent Field and Convergence](samples/dinov3_intent.png)

The conversion runs in that order:

1. **Find the frame and the film base**, and work out which exposure profiles can hold the scene.
2. **Search.** Try the candidates, render each, and ask the intent head how far the result is from what the picture
   should look like. Adjust exposure and the green and blue balance, look again, and keep the best. A grey-world rule
   would turn a sunset grey; this does not, because the target knows what sky, foliage and skin look like.
3. **Clean up casts** (white light). Balance alone cannot fix a cast that is warm in the shadows and cool in the
   highlights, so the green and blue curves are bent until the surfaces the neutral head trusts come out grey. With
   nothing neutral in the frame it leaves the picture alone, and it never pushes a picture toward green.
4. **Set the tones** (white light). The tone head says where the shadows and highlights of a finished scan of this
   picture would sit, and one lightness curve moves the image toward that. Colour is not touched.

[DESIGN.md](DESIGN.md) lists the white-light stages and every tunable number each one carries, with what it does.

On the frame below the first render is cyan-green; the search brings it to a natural daylight balance.

![DINOv3 Patch Residual Error Minimization](samples/dinov3_residuals.png)

`--fit_curves off` switches the curve fit off, and with it the tint trim and the tone stage, which run wherever it
runs; `--tint_trim` and `--tone_stage` only replay recorded numbers. Everything the conversion decided is written to a
sidecar next to the output (the tint trim under `offset`), and the brightness, tint and warmth knobs replay it from
there without running the model again.

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
| `--fit_curves auto\|off\|on\|<numbers>` | G/B input curves (white light, `-P srgb` or `srgb-g10`). `auto` (default): on where it applies. `off`: none, and no tint trim or tone stage. `on`: an error where it does not apply. Eight or ten numbers: replay a sidecar's curves, with `--target`, `-E`, `--gain_g` and `--gain_b`. |
| `--roi auto [h\|v]` or `--roi x y w h [rot [hflip [vflip]]]` | The frame: detected, or given in sensor pixels. |
| `--film_format 35mm\|645\|6x6\|6x7\|6x9\|auto` | The format `--roi auto` looks for (default 35mm); with a given frame, the format the film base is read around (default: every format). |
| `--rot`, `--hflip`, `--vflip` | Orientation of the output. |
| `-q`, `--quality` | Demosaic: −1 auto (grain-aware on a Sony CFA, else bilinear), or a LibRaw quality (4: DCB). |
| `-r`, `-g`, `-b` | Rows of the crosstalk matrix (default: the bundle's). |
| `-C`, `--no_crop` | Keep the RAW's full area instead of its default crop. |
| `--flat` | A flat-field `.npz` measured on the empty gate. |
| `--dino <head>:<file>,...` | Your own models in place of the built-in ones. The heads are `intent` (the intent adapter: an `.onnx` or its directory), `neutral` (a `neutral_head.bin`) and `tone` (a `tone_head.bin`); for example `--dino neutral:my_head.bin` or `--dino intent:adapter.onnx,tone:th.bin`. A path without a head name is the intent adapter. Heads you leave out stay built in. Paths cannot contain commas. |
| `--tint_trim theta:<numbers>` | Replays a sidecar's tint trim (10 numbers, from its `cli_equivalent`; the sidecar records it under `offset`). The tint trim, the network's setting on the magenta–green axis, runs after the curve fit wherever the fit runs (up to four extra inferences). |
| `--tone_stage curve:<numbers>` | Replays a sidecar's lightness curve (5 numbers). The tone stage itself runs on the final image wherever the curve fit runs with `-P srgb`. |
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
