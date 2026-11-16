# Intent adapter: Kodak Portra 160

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values
  per patch). The backbone is not trained.
- Model: a small MLP (595,590 parameters) giving mu and log variance.
- Training data: 1845 photographs of Kodak Portra 160.
- Validation error: 4.85 L*, 5.07 a*b* (mean absolute) on the validation photographers the seed was chosen with.
- Training: cosine schedule over 15 epochs, final epoch kept, loss weighted per patch; seed 43 of 42, 43, 44 by the
  calibrated validation loss. Sigma calibrated per channel on the validation photographers (scale 1.26, 1.24, 1.39 for
  L*, a*, b*, folded into the variance head's bias); the mean is not changed.
- Held-out photographers (in neither training nor validation): mean absolute error 4.18 L* per photographer;
  mean signed error -0.32 L*, -1.07 a*, -1.42 b*.

`metadata.json` has the full settings and metrics.
