# Tone heads

Predict, from a preview's frozen DINOv3 features, the tone a lab scan of the frame would have: 11 descriptors of its
pixels at network size, each with a mean and a standard deviation. `neg_process` uses three of them for the white-light
tone stage (`np_pipeline.cc`): the shadow level L2 (L* at 2 %), the highlight level L98 (L* at 98 %) and the tonal range
R9010 (L* at 90 % minus L* at 10 %). One monotone L* curve then moves the final image all the way to the predicted L2
and half way to the predicted L98 and R9010, holding L50 and the mean chroma. The heads predict tone only; colour is
the curve fit's.

- Input: 1920 features, the backbone's pooled output (the CLS token of layer 12, the mean patch token of layer 9 and
  of layer 12: 1152) and the patch tokens' standard deviation over the patches (layers 9 and 12: 768;
  `DinoV3InferenceResult::token_sd`), standardised with the stored `xm` and `xs`.
- Model: Linear(1920, 512), GELU (exact, erf), Dropout(0.2), Linear(512, 256), GELU, Linear(256, 22): the 11
  targets' standardised means and log-variances (clamped to [-6, 4]); mean = out * ys + ym, sd = exp(logvar / 2) * ys.
  1,120,534 parameters.
- Targets, in order: L2 L5 L10 L50 L90 L95 L98 (pixel L* percentiles), IQR (L75 - L25), R9010 (L90 - L10), Cmean
  (mean C*ab), C90 (C*ab at 90 %), all on CIELAB D65 of the image at DINOv3's input size.

## Training

- Data: FilmCorpus lab scans of 8 stocks (ektar100, gold200, portra, portra160, portra400, portra800, pro160ns,
  pro400h). They are split by photographer, so a photographer's rolls and near-duplicates stay on one side: pool C
  (sha1 of the photographer mod 100 < 10, from both sites: 1,107 scans) is held out for testing; of the rest, pool A
  is the photohito photographers (6,378 scans, 1,282 photographers; Japanese labs) and pool B the flickr ones (3,054
  scans). These heads are pool A's. Pool B was kept apart to grade renders independently of them.
- Each scan, cropped to its inner 94 % and resized in linear light to DINOv3's input size, gives five versions: the
  clean scan and 4 random perturbations of it (exposure ±0.8 EV, gamma, contrast, black point -3..22 L*, white
  point 72..101 L*, saturation x0.5..1.6, an a*/b* cast in 70 % of them), covering flat, lifted-black negatives and
  crushed, contrasty ones. The input is the features of one version (the clean one in 15 % of draws); the target is
  always the clean scan's descriptors, so the heads learn the scan's tone from content, not from the render's own
  levels.
- Loss: Gaussian negative log-likelihood on the standardised targets; AdamW (weight decay 0.01), one-cycle learning
  rate to 2e-3, 40 epochs, batch 256; the best epoch by mean absolute error on 5 % of pool A's photographers held out.

## Validation (pool C, held-out photographers; mean absolute error over the perturbed versions)

| target | head | corpus mean | the perturbed input's own value | within-scan sd of the prediction | between scans |
|---|---|---|---|---|---|
| L2 | 5.2 | 7.0 | 9.9 | 1.2 | 7.7 |
| L98 | 4.0 | 5.4 | 15.0 | 0.8 | 4.5 |
| R9010 | 7.5 | 12.8 | 17.0 | 2.2 | 12.6 |
| L50 | 6.6 | 14.8 | 9.1 | 3.0 | 16.9 |
| Cmean | 3.3 | 5.2 | 3.5 | 1.0 | 5.4 |

The prediction barely follows the render (within-scan spread 1-2 L* against 5-13 between scans), which is what lets
the stage read it from any preview of the frame. The errors are those of predicting one lab's scan: about 5 L* on the
shadow level, 4 on the highlight level. The heads were trained on lab scans of film, not on negicc's renders; on
negicc renders, flattening or steepening a render by 4-10 L* moves the prediction by 0.3-1.5 L*.

## Files

- `tone_head.pt`: the trained PyTorch dict `{kind: "pool", state, xm, xs, ym, ys, tgt, pool: "A"}` (sha256 in
  `metadata.json`).
- `tone_head.bin`: the same weights flattened for `neg_process`, which embeds it (Makefile `EMBED_FILES`) and runs it
  in plain C++ (`ToneHead` in `intent_head.h`), once per frame, on features an inference of the white-light fit already produced.
  Format in `export_tone_head.py`.
- `export_tone_head.py`: writes `tone_head.bin` from `tone_head.pt`; `--check` compares PyTorch with the exported
  weights on random features.

```bash
python3 models/tone_head/export_tone_head.py --check   # needs torch and numpy
```

## Using it in neg_process

Runs wherever the white-light curve fit runs with `-P srgb` (always; `--tone_stage` only replays a recorded curve;
TriLED never runs it). The engine then also gives the tokens' spread with every inference of the search
(`set_token_spread`); the heads run once, on the features of the tint trim's winning candidate (the state
closest to the export), or of the search's intent preview when that setting did not run, so the stage adds no
inference. The sidecar's `tone_stage` records which (`features`), the predictions, the final render's descriptors
before the stage, the targets, the curve and the clipping before and after.

To try retrained heads, export them with `export_tone_head.py` and pass the `.bin` with `--dino tone:<file>`, or
replace `tone_head.bin` and rebuild. The sidecar's `heads.tone` records which file ran, with its size and CRC-32.

## Provenance

Trained in the whitelab TONE arm (`tools/extract_corpus.py --K 4`, then `tools/train_heads.py A --kind pool`), with
negicc-station's environment and its DINOv3 ViT-S/16 (`dinov3-small_portra400`: the same backbone and layers as
`../dinov3-small`). `metadata.json` has the settings, metrics and checksums.
