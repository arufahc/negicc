# Intent adapter: Kodak Portra 160

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values per patch). The backbone is
  not trained.
- Model: a small MLP (595,590 parameters) with three outputs, mu, sigma and log variance; `neg_process` reads mu and
  sigma.
- Training data: 1845 photographs of Kodak Portra 160, and 282 by other photographers for validation.
- Validation error: 4.85 L*, 5.07 a*b* (mean absolute error over the validation photographers' patches) on the
  validation photographers the seed was chosen with.
- Acceptance gates (`metadata.json` `acceptance_gates`): validation loss at most 2.30, a*b* error at most 5.00, L* error
  at most 8.00. This adapter misses the a*b* gate (5.07; the other two seeds 5.20 and 5.09) and is shipped with
  `all_passed` false.
- Training: cosine schedule over 15 epochs, final epoch kept, loss weighted per patch; seed 43 of 42, 43, 44 by the
  calibrated validation loss. Sigma calibrated per channel on the validation photographers (scale 1.26, 1.24, 1.39 for
  L*, a*, b*, folded into the variance head's bias); the mean is not changed.
- Held-out photographers (in neither training nor validation): mean absolute error 4.18 L* per photographer; mean signed
  error -0.32 L*, -1.07 a*, -1.42 b*. These are means per photographer, not over patches as the validation error is; no
  held-out a*b* absolute error is given, and none of the held-out numbers is in `metadata.json`.

`metadata.json` has the training settings and the validation metrics. Its `val_loss`, and each seed's, is the loss
after the sigma calibration, which was fitted on the same validation photographers; the seed choice and the loss
gate therefore use the photographers the calibration was fitted on. The absolute errors do not change with the
calibration. `best_epoch` is the last of the 15 epochs (`checkpoint` is `final`). `backbone_details` describes a
square 512 x 512 input (32 x 32 = 1024 patches); the adapter takes any number of patches, and `neg_process` gives it
the preview at the frame's own aspect (672 patches for 3:2). The adapter was trained with the companion profiling
tools, which are not in this repository: `git_commit` is a commit of those tools, not of negicc, and
`exported_files` also names `model.pt` and `adapter.torchscript`, which are not shipped here.

## Files

| File | Contents |
|---|---|
| `adapter.onnx` | the adapter, SHA-256 `f2e6a2cc5cffa21cee30cc56da13d9415185bff948842986b3d6d36fd66fdf42` |
| `metadata.json` | the training settings and validation metrics |
