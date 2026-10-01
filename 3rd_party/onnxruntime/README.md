# ONNX Runtime Prebuilt Libraries and Headers

This directory can hold prebuilt ONNX Runtime v1.30.0 CPU shared libraries and C/C++ headers for x86_64 and aarch64 (Linux).

## Overview
ONNX Runtime CPU is used for native C++ DINOv3 photographic intent inference (`DinoV3Engine`) when running without CUDA / TensorRT or on non-NVIDIA hosts.

## Source & License
- Upstream: https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0
- License: MIT License (https://github.com/microsoft/onnxruntime/blob/main/LICENSE)

## Discovery
When building `libnegicc_dinov3.a` or `neg_process`:
- If `3rd_party/onnxruntime` exists locally, `negicc/Makefile` uses it.
- When `negicc` is included as a submodule of `negicc_station` at `3rd_party/negicc`, `negicc/Makefile` automatically locates `../onnxruntime` (i.e. `negicc_station/3rd_party/onnxruntime`).
- Otherwise, `ORT_DIR` can be set explicitly (e.g. `make ORT_DIR=/path/to/onnxruntime`).
