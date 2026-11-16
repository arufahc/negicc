# Intent adapter: Kodak Portra 400

Predicts, for every 16x16 patch of a preview, the colour (CIELAB mean and tolerance) a good conversion of
this film stock has there. `neg_process` searches the bracket, exposure and gains that bring
the conversion closest to it.

- Input: patch tokens of the frozen DINOv3 ViT-S/16 backbone (layers 9 and 12, 768 values
  per patch). The backbone is not trained.
- Model: a small MLP (595,590 parameters) giving mu and log variance.
- Training data: 2908 photographs of Kodak Portra 400.
- Validation error: 4.15 L*, 4.36 a*b* (mean absolute) on the photos the epoch and the seed were chosen with; 5.20 L*,
  4.74 a*b* on 373 photos by 24 photographers held out from both.

`metadata.json` has the full settings and metrics.
