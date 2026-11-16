# Intent adapter: Kodak Portra 160

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values
  per patch). The backbone is not trained.
- Model: a small MLP (595,590 parameters) giving mu and log variance.
- Training data: 1845 photographs of Kodak Portra 160.
- Validation error: 4.83 L*, 5.20 a*b* (mean absolute) on the validation photographers the seed was chosen with.
- Training: cosine schedule over 15 epochs, final epoch kept; seed 42 of 42, 43, 44 by final validation loss.
- Held-out photographers (in neither training nor validation): mean absolute error 4.12 L* per photographer;
  mean signed error -0.33 L*, -1.22 a*, -1.65 b*.

`metadata.json` has the full settings and metrics.
