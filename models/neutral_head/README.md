# Neutral-patch head

Predicts, for every 16x16 patch of a preview, p(neutral): the probability that the patch should render grey in the
intended positive, whatever colour cast the current render has. `neg_process` uses it, when enabled, to pick the
scene-neutral anchors of the joint curve fit (B5 + head anchors) in place of the raw-chroma selection and its
hand-tuned rules. It does not predict colour; the intent adapter's mu and sigma stay as they are.

- Input: the [N, 768] patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, the tokens the intent
  adapters of `../dinov3-small_*` read), standardised with the stored `mean` and `std` (768 each).
- Model: Linear(768, 256), GELU (exact, erf), Dropout(0.1), Linear(256, 1); p = sigmoid(output). 197,121 parameters.
- Use: cells with p >= 0.9 are the neutral anchors.
- Training data: FilmCorpus lab scans of 8 stocks pooled (ektar100, gold200, portra, portra160, portra400,
  portra800, pro160ns, pro400h): 10,539 scans x 2 cast copies, 4.18M labelled patches. Labels from the clean scan:
  neutral when the patch-mean C* < 4, the 75th-percentile pixel C* < 8 and 8 < L* < 95; non-neutral when C* > 10.
  Tokens from cast copies (per-channel offset up to +-0.7 stop, crossover gamma 0.7-1.3 for R and B about mid grey in
  linear light, +-0.5 EV), so the head has to recognise grey-intended content under a cast.
- Training: BCE with positive-class weighting, AdamW, one-cycle LR 2e-3, 4 epochs, batch 8192.

## Validation (all stocks pooled, 5-fold CV)

| split unit | head AUROC | current rule (C_mu + sigma_ab < 6) | recall at the rule's 4.3% false-pick rate |
|---|---|---|---|
| random image | 0.906 | 0.865 | 0.50 vs 0.37 |
| photographer (2,042 groups) | 0.898 | 0.865 | 0.48 vs 0.37 |

AUROC gain over the rule +0.042 [+0.036, +0.047] (image split) and +0.033 [+0.029, +0.038] (photographer split),
cluster bootstrap over photographers. Hard subset (neutral patches the cast made colourful, and colourful patches it
made grey): 0.795-0.806 vs 0.752. Leave-one-stock-out: the head beats the rule on all 8 held-out stocks.

Owner blind A/B on 12 negicc-web uploads and 20 whitelab frames: B5 with head anchors replacing its four selection
rules (low-key merge, collinearity drop, chart-like gate, shadow cluster) ties B5 (7:7:14).

## Files

- `neutral_head.pt`: the trained PyTorch dict `{features: "tok", kind: "mlp", state, mean, std}` (the same file as
  negicc-station's `models/neutral_head/neutral_head_pooled_mlp.pt`; sha256 in `metadata.json`).
- `neutral_head.bin`: the same weights flattened for `neg_process`, which embeds it (Makefile `EMBED_FILES`) and runs
  it in plain C++ (`neutral_head.h`) on the tokens the backbone already produced. Format in
  `export_neutral_head.py`.
- `export_neutral_head.py`: writes `neutral_head.bin` from `neutral_head.pt`; `--check` compares PyTorch with the
  exported weights on random tokens.

```bash
python3 models/neutral_head/export_neutral_head.py --check   # needs torch and numpy
```

## Using it in neg_process

Off by default; with it off, output is byte-identical to a build without it.

```bash
NP_PHASE2_ARM=B5 NP_NEUTRAL_HEAD=1 neg_process ...        # B5 + head anchors, built-in head
NP_NEUTRAL_HEAD=path/to/neutral_head.bin ...                # another export of the head
NP_NEUTRAL_P_THR=0.9                                        # anchor threshold (default 0.9)
```

`NP_NEUTRAL_HEAD` adds `neutral_p` to every DINOv3 inference (about 5-8 ms on 8 threads, 25 ms on one, against
about 250-350 ms for the backbone and adapter). With the backbone and adapter on TensorRT the tokens are copied to the
host once per inference for the head. `NP_B5_HEAD_P_DIR` (p from text files keyed by a hash of mu) still overrides
the native p, for experiments.

## Provenance

Trained with negicc-station's `training/neutral_head/` (branch `exp-neutral-head`, commit 1781969: extraction,
training and evaluation scripts; weights and metadata in commit 0ac462b). From the negicc-station repository root:

```bash
./venv/bin/python training/neutral_head/extract_neutral.py <out_dir> --workers 64 --per-stock 1500
./venv/bin/python training/neutral_head/train_pooled.py <out_dir> --split image --folds 5
./venv/bin/python training/neutral_head/train_pooled.py <out_dir> --split photographer
./venv/bin/python training/neutral_head/final_pooled.py <out_dir>        # writes the deployable head
```

`metadata.json` has the settings, metrics and checksums.
