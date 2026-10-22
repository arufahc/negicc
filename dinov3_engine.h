#ifndef DINOV3_ENGINE_H
#define DINOV3_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "dinov3_runner.h"

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

    // Adapter for a backbone model: an adapter.onnx held in memory (embedded in neg_process) or read from a file.
    // name is reported by adapter_name(). Ignored by fused models, which carry their own adapter. Throws
    // std::runtime_error when the adapter cannot be read or loaded.
    void set_adapter(const void* onnx_data, size_t onnx_size, const std::string& name, int num_threads = 1);
    void load_adapter_file(const std::string& path, int num_threads = 1);
    const std::string& adapter_name() const { return adapter_name_; }
    // The loaded model is a backbone and needs an adapter.
    bool needs_adapter() const;

    bool is_loaded() const;
    const char* backend_name() const;
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
    std::vector<uint8_t> adapter_file_;
    std::string adapter_name_;
    int max_size_ = 512;
    int patch_size_ = 16;
};

#endif  // DINOV3_ENGINE_H
