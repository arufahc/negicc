# DINOv3 backbone

The frozen DINOv3 ViT-S/16 that `neg_process` runs in its parameter search; every film stock's intent adapter
(`models/dinov3-small_<stock>/`) reads its tokens. Export it from the pretrained weights with

```
make dinov3_model        # DINOV3_SOURCE=<Hugging Face id or local directory> to choose the weights
```

which writes `backbone.onnx`, `backbone.json` (where the weights came from) and the weights' licence files. The
weights are Meta's DINOv3, distributed under the DINOv3 License: keep its licence files next to `backbone.onnx`.

`neg_process` packs the backbone into the binary: `backbone.onnx`, and in `USE_CUDA=1` builds also `backbone.engine`,
the TensorRT engine `make dinov3_engine` builds on the machine that runs it (not checked in). The engine runs when the
machine has a CUDA device; when it cannot be loaded, `--dino_backend auto` falls back to `backbone.onnx`.
