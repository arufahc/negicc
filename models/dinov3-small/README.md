# DINOv3 backbone

The frozen DINOv3 ViT-S/16 that `neg_process` runs in its parameter search; every film stock's intent adapter
(`models/dinov3-small_<stock>/`) reads its tokens. Export it from the pretrained weights with

```
make dinov3_model        # DINOV3_SOURCE=<Hugging Face id or local directory> to choose the weights
```

which writes `backbone.onnx`, `backbone.json` (where the weights came from) and the weights' licence files. The
weights are Meta's DINOv3, distributed under the DINOv3 License: keep its licence files next to `backbone.onnx`.

`backbone.onnx` is checked in, so a clone builds `neg_process` with its parameter search. It is the export of
DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`, through the mirror and revision `backbone.json` names);
exporting it again from that source with torch 2.14 gives the same file (sha256 in `backbone.json`). `LICENSE.md`
is the DINOv3 License as Meta publishes it (github.com/facebookresearch/dinov3), unchanged. The backbone, and any
`neg_process` binary that packs it, may be distributed only under that licence and with a copy of it.

`neg_process` packs the backbone into the binary: `backbone.onnx`, and in `USE_CUDA=1` builds also `backbone.engine`,
the TensorRT engine `make dinov3_engine` builds on the machine that runs it (not checked in). The engine runs when the
machine has a CUDA device; when it cannot be loaded, `--dino_backend auto` falls back to `backbone.onnx`.
