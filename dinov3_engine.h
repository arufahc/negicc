#ifndef DINOV3_ENGINE_H
#define DINOV3_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "dinov3_runner.h"

class NeutralHead;

// The fused DINOv3 intent model (src/export_intent_engine.py): inputs pixel_values, pool_weights; outputs mu,
// sigma, pooled, pooled_tone. Preprocessing reproduces intent_model.predict / dino_geometry.
// py-ref: negicc_station/src/dino_geometry.py:30-74 @ 2f7ee4a (geometry & tone weights)
// py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (predict flow)
class DinoV3Engine {
public:
    DinoV3Engine();
    ~DinoV3Engine();

    DinoV3Engine(const DinoV3Engine&) = delete;
    DinoV3Engine& operator=(const DinoV3Engine&) = delete;
    DinoV3Engine(DinoV3Engine&&) noexcept;
    DinoV3Engine& operator=(DinoV3Engine&&) noexcept;

    // model_spec: a model directory, an .onnx file or an .engine file. A directory holds either the DINOv3 backbone
    // alone (backbone.onnx, backbone.engine: models/dinov3-small/export_dinov3_backbone.py) or a fused intent model
    // (model.onnx, model.engine); the backbone wins when both are present. A backbone needs an adapter
    // (set_adapter) before infer().
    // backend: "cpu" (ONNX Runtime), "gpu" (TensorRT, USE_CUDA=1 builds) or "auto" ("gpu" when this build has
    // a CUDA device and the model has a TensorRT engine, as intent_model.resolve_intent_backend).
    // Throws std::runtime_error (std::invalid_argument for a bad backend) when the model cannot be loaded.
    // py-ref: negicc_station/src/intent_model.py:320-366 @ 2f7ee4a (resolve_intent_backend & load_intent_model)
    void load(const std::string& model_spec, const std::string& backend = "auto", int num_threads = 4,
              int max_size = 512, int patch_size = 16);

    // The backbone built into neg_process, held in memory: its TensorRT engine (USE_CUDA=1 builds) and its .onnx,
    // each null when not built in. backend picks between them as load() does: "auto" runs the engine when this build
    // has a CUDA device and falls back to the .onnx when the engine is missing or fails to load. Throws as load()
    // does.
    void load_memory(const void* engine_data, size_t engine_size, const void* onnx_data, size_t onnx_size,
                     const std::string& backend = "auto", int num_threads = 4, int max_size = 512,
                     int patch_size = 16);

    // Adapter for a backbone model, after load(): its adapter.onnx and its TensorRT adapter.engine (make
    // dinov3_engine), held in memory (embedded in neg_process) or read from files. The adapter runs where the
    // backbone runs: a TensorRT backbone needs the engine (there is no CPU adapter behind a GPU backbone), an ONNX
    // Runtime backbone uses the .onnx. name is reported by adapter_name(). Ignored by fused models, which carry their
    // own adapter. Throws std::runtime_error when the adapter the backbone needs is missing or cannot be loaded.
    void set_adapter(const void* onnx_data, size_t onnx_size, const std::string& name, int num_threads = 1,
                     const void* engine_data = nullptr, size_t engine_size = 0);
    // path: adapter.onnx or adapter.engine (or a directory holding them); the other file is taken from beside it.
    void load_adapter_file(const std::string& path, int num_threads = 1);
    // The adapter runs on the GPU (TensorRT).
    bool adapter_on_device() const { return adapter_ && adapter_->on_device(); }
    // Device-to-host copies of the backbone's tokens so far (0 when the backbone and adapter both run on the GPU).
    int token_downloads() const { return runner_ ? runner_->token_downloads() : 0; }
    const std::string& adapter_name() const { return adapter_name_; }
    // The loaded model is a backbone and needs an adapter.
    bool needs_adapter() const;

    // The neutral-patch head (intent_head.h) from a neutral_head.bin held in memory: with it, infer() also gives
    // DinoV3InferenceResult::neutral_p from the backbone's tokens. When the backbone and adapter run on the GPU the
    // tokens are copied to the host once per inference for it. Null data clears it. Throws std::runtime_error on
    // a bad file or a fused model (no tokens to read).
    void set_neutral_head(const void* data, size_t size);
    bool has_neutral_head() const { return neutral_head_ != nullptr; }
    // With it on, infer() also gives DinoV3InferenceResult::token_sd, the spread of the backbone's tokens over the
    // patches: with the pooled features, the input of the white-light tone heads (intent_head.h). Like the neutral head
    // it needs the tokens on the host (one copy per inference when the backbone and adapter run on the GPU).
    void set_token_spread(bool on) { token_spread_ = on; }

    bool is_loaded() const;
    const char* backend_name() const;
    const char* backend_used() const;
    bool has_pooled() const;
    bool has_pooled_tone() const;
    bool has_pool_weights() const;
    size_t memory_bytes() const;
    int max_size() const { return max_size_; }
    int patch_size() const { return patch_size_; }

    // Run inference on 8-bit sRGB preview image. Not thread-safe (a TensorRT execution context runs one inference at
    // a time): callers that share one engine between threads serialize their calls, as negicc-station's binding does.
    // py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (predict)
    DinoV3InferenceResult infer(const uint8_t* srgb_data, int width, int height, int stride);

private:
    std::unique_ptr<DinoV3Runner> runner_;
    std::unique_ptr<DinoV3AdapterRunner> adapter_;
    std::unique_ptr<NeutralHead> neutral_head_;
    bool token_spread_ = false;
    std::vector<uint8_t> adapter_file_;
    std::string adapter_name_;
    int max_size_ = 512;
    int patch_size_ = 16;
};

#endif  // DINOV3_ENGINE_H
