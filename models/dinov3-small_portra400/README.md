# Intent adapter: Kodak Portra 400

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values
  per patch). The backbone is not trained.
- Model: a small MLP (595,590 parameters) with three outputs, mu, sigma and log variance; `neg_process` reads mu and
  sigma.
- Training data: 2908 photographs of Kodak Portra 400.
- Validation error: 4.30 L*, 4.33 a*b* (mean absolute) on the validation photographers the seed was chosen with.
- Training: cosine schedule over 15 epochs, final epoch kept, loss weighted per patch; seed 42 of 42, 43, 44 by the
  calibrated validation loss. Sigma calibrated per channel on the validation photographers (scale 1.34, 1.36, 1.34 for
  L*, a*, b*, folded into the variance head's bias); the mean is not changed.
- Held-out photographers (in neither training nor validation): mean absolute error 4.30 L* per photographer;
  mean signed error -0.02 L*, +0.04 a*, +0.18 b*.

`metadata.json` has the full settings and metrics.
