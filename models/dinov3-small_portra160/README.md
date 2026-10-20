# Intent adapter: Kodak Portra 160

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values
  per patch). The backbone is not trained.
- Model: a small MLP (595,590 parameters) giving mu and log variance.
- Training data: 1845 photographs of Kodak Portra 160, trained with negicc-station's `training/`.
- Validation error: 4.59 L*, 4.92 a*b* (mean absolute).

`metadata.json` has the full settings and metrics.
