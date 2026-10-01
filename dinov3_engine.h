#ifndef NEGICC_DINOV3_ENGINE_H
#define NEGICC_DINOV3_ENGINE_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "dinov3_runner.h"

namespace negicc {

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

    // model_spec: a model directory (model.onnx, model.engine), an .onnx file or an .engine file.
    // backend: "cpu" (ONNX Runtime), "gpu" (TensorRT, USE_CUDA=1 builds) or "auto" ("gpu" when this build has
    // a CUDA device and the model has a TensorRT engine, as intent_model.resolve_intent_backend).
    // py-ref: negicc_station/src/intent_model.py:320-366 @ 2f7ee4a (resolve_intent_backend & load_intent_model)
    bool load(const std::string& model_spec, const std::string& backend = "auto", int num_threads = 4,
              int max_size = 512, int patch_size = 16);

    bool is_loaded() const;
    const char* backend_name() const;
    bool has_pooled() const;
    bool has_pooled_tone() const;
    bool has_pool_weights() const;
    size_t memory_bytes() const;
    int max_size() const { return m_max_size; }
    int patch_size() const { return m_patch_size; }

    // Run inference on 8-bit sRGB preview image
    // py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (predict)
    DinoV3InferenceResult infer(const uint8_t* srgb_data, int width, int height, int stride);

private:
    std::unique_ptr<DinoV3Runner> m_runner;
    int m_max_size = 512;
    int m_patch_size = 16;
};

}  // namespace negicc

#endif  // NEGICC_DINOV3_ENGINE_H
