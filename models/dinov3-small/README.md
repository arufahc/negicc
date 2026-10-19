# DINOv3 backbone

The frozen DINOv3 ViT-S/16 that `neg_process --dino` and `--auto` run; every film stock's intent adapter
(`models/dinov3-small_<stock>/`) reads its tokens. Export it from the pretrained weights with

```
make dinov3_model        # DINOV3_SOURCE=<Hugging Face id or local directory> to choose the weights
```

which writes `backbone.onnx`, `backbone.json` (where the weights came from) and the weights' licence files. The
weights are Meta's DINOv3, distributed under the DINOv3 License: keep its licence files next to `backbone.onnx`.
`make dinov3_engine` builds `backbone.engine` for TensorRT on the device that runs it (not checked in).
