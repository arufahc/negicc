# Intent adapter: Kodak Portra 400

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values per patch). The backbone is
  not trained.
- Model: a small MLP (595,590 parameters) with three outputs, mu, sigma and log variance; `neg_process` reads mu and
  sigma.
- Data: 2,908 photographs of Kodak Portra 400 by 298 photographers for training, 365 by 22 others for validation, and
  373 by 24 more held out (in neither training nor validation). The photographers are split 80/10/10 (`split_hash`
  `72ded7f18932`, seed 42).
- Validation error: 4.30 L*, 4.33 a*b* (mean absolute error over the validation photographers' 287,904 patches; a*b* is
  the mean of the a* and b* errors), on the validation photographers the seed was chosen with.
- Held-out error, the same statistic: 4.81 L*, 4.53 a*b* (over the held-out photographers' 266,624 patches;
  `metadata.json` `held_out`).
- Per held-out photographer (averaged per photograph, then per photographer): mean absolute error 4.30 L*; mean signed
  error -0.02 L*, +0.04 a*, +0.18 b* (bootstrap standard errors 0.45, 0.41, 0.60).
- Acceptance gates (`metadata.json` `acceptance_gates`): validation loss at most 2.30, a*b* error at most 5.00, L* error
  at most 8.00; all three pass.
- Training: cosine schedule over 15 epochs, final epoch kept, loss weighted per patch; seed 42 of 42, 43, 44 by the
  calibrated validation loss. Sigma calibrated per channel on the validation photographers (scale 1.34, 1.36, 1.34 for
  L*, a*, b*, folded into the variance head's bias); the mean is not changed.
- Trained by `tools/build_profile_adapter.py` of the companion repository (negicc's profiling tools, which are not
  public), at commit `d66dc5e` (`git_commit`) on its `adapters-final-epoch` branch, which is not merged into its default
  branch. The command, as run (`metadata.json` `training`):

  ```
  NEGICC_CORPUS_DIR=<corpus> python tools/build_profile_adapter.py \
      --profile profiles/profile_Portra400-A7R4-TriLED_20261003_172952.json --weighting token \
      --jobs 24 --threads 16 --no-combined-onnx --force --models-dir <output>
  ```

  The rest are the script's defaults at that commit: 15 epochs, the final checkpoint, batch 64, learning rate 1e-3,
  seeds 42, 43 and 44, sigma calibration on. Only the stock is read from the profile, from its file name; the
  photographs are the corpus manifest `portra400.json` under `NEGICC_CORPUS_DIR` (`corpus_manifest_sha256`).

`metadata.json` has the training settings, the validation metrics and the held-out record (`held_out`). Its
`metrics.val_loss`, and each seed's, is the loss after the sigma calibration, which was fitted on the same validation
photographers; the seed choice and the loss gate therefore use the photographers the calibration was fitted on. The
absolute errors do not change with the calibration. `metrics.epoch` is the last of the 15 epochs (`checkpoint` is
`final`). `backbone_details` describes a square 512 x 512 input (32 x 32 = 1024 patches); the adapter takes any number
of patches, and `neg_process` gives it the preview at the frame's own aspect (672 patches for 3:2). `git_commit` and
`training.commit` are commits of the companion repository, not of negicc. `exported_files` lists the files shipped here,
and `adapter_onnx_sha256` is the hash of `adapter.onnx`. `seed_runs[*].checkpoint_sha256` are the hashes of the three
seeds' PyTorch checkpoints, which stay with the training tools. The held-out figures were computed after training, by
the statistics of the comparison that chose this recipe, from the selected seed's checkpoint on the split's test
photographers; the training script does not write them. Computed the same way, the validation errors equal
`metrics.val_mae_L` and `metrics.val_mae_ab` to three decimals, and running the shipped `adapter.onnx` on the held-out
patches gives the held-out errors above, so the two are the same statistic of the same model.

## Files

| File | Contents |
|---|---|
| `adapter.onnx` | the adapter, SHA-256 `94f9e2e76c21a8f5a2c362c5b4b3dcbfb32118424d5accd2d24c82d3a9eda152` |
| `metadata.json` | the training settings, the validation metrics and the held-out record |
