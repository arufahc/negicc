# Intent adapter: Kodak Ektar 100

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values
  per patch). The backbone is not trained.
- Model: a small MLP (595,590 parameters) giving mu and log variance.
- Training data: 4351 photographs of Kodak Ektar 100.
- Validation error: 3.83 L*, 4.97 a*b* (mean absolute) on the validation photographers the seed was chosen with.
- Training: cosine schedule over 15 epochs, final epoch kept; seed 42 of 42, 43, 44 by final validation loss.
- Held-out photographers (in neither training nor validation): mean absolute error 3.83 L* per photographer;
  mean signed error -0.08 L*, -0.37 a*, -0.52 b*.

`metadata.json` has the full settings and metrics.
