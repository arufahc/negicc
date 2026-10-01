// Inference runners behind DinoV3Engine: one fused-model forward pass on preprocessed tensors.
//
// The ONNX Runtime CPU runner (dinov3_ort_runner.cpp) is built everywhere. The TensorRT runner
// (dinov3_trt_runner.cpp) is built only with USE_CUDA=1 and runs the device-specific model.engine that
// `make engine` builds (src/export_intent_engine.py). When USE_CUDA=0, dinov3_nocuda.cpp provides
// the fallback stubs.
//
// py-ref: negicc_station/src/export_intent_engine.py:49-70, 104-112 @ 2f7ee4a (output contract)
// py-ref: negicc_station/src/tensorrt_intent_model.py:249-273 @ 2f7ee4a (multi-head output parsing)
// py-ref: negicc_station/src/dino_geometry.py:23-27 @ 2f7ee4a (IntentPrediction namedtuple)
#ifndef NEGICC_DINOV3_RUNNER_H
#define NEGICC_DINOV3_RUNNER_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace negicc {

// Multi-head inference result matching Python IntentPrediction.
// py-ref: negicc_station/src/dino_geometry.py:23-27 @ 2f7ee4a (IntentPrediction namedtuple)
// py-ref: negicc_station/src/tensorrt_intent_model.py:249-273 @ 2f7ee4a (multi-head output parsing)
struct DinoV3InferenceResult {
    std::vector<float> mu;            // [1, N, 3] Target Lab color field
    std::vector<float> sigma;         // [1, N, 3] Tolerance field
    std::vector<float> pooled;        // [1, 1152] Exposure ranker CLS+mean features
    std::vector<float> pooled_tone;   // [1, 3, 768] Tone-stratified ranker features
    std::vector<float> tone_mass;     // [3] Shadows, mid-tones, highlights mass fractions
    std::vector<float> features_3459; // [3459] Contiguous concatenated: pooled + pooled_tone + tone_mass
    bool has_pooled = false;
    bool has_pooled_tone = false;
    int target_w = 0;
    int target_h = 0;
    int grid_w = 0;
    int grid_h = 0;
    int num_patches = 0;
    double duration_ms = 0.0;
};

// Destination buffers for multi-head DINOv3 model outputs.
// Any pointer left as nullptr will not be copied back by the runner.
// py-ref: negicc_station/src/tensorrt_intent_model.py:249-273 @ 2f7ee4a
struct DinoV3RunnerOutputs {
    float* mu = nullptr;           // [1, patches, 3] CIELAB mean field
    float* sigma = nullptr;        // [1, patches, 3] CIELAB tolerance field
    float* pooled = nullptr;       // [1, 1152] Exposure ranker CLS + patch mean
    float* pooled_tone = nullptr;  // [1, 3, 768] Tone-stratified ranker features
};

class DinoV3Runner {
public:
    virtual ~DinoV3Runner() = default;
    virtual const char* name() const = 0;
    // The model takes the tone-bin weights input "pool_weights" (exports since the tone features were added).
    virtual bool has_pool_weights() const = 0;
    virtual bool has_pooled() const { return false; }
    virtual bool has_pooled_tone() const { return false; }
    virtual size_t memory_bytes() const { return 0; }

    // Multi-head forward pass with selectable destination buffers.
    virtual void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
                     const DinoV3RunnerOutputs& outputs) = 0;

    // Legacy overload for callers only requiring intent field (mu, sigma).
    void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
             float* mu, float* sigma) {
        DinoV3RunnerOutputs outs;
        outs.mu = mu;
        outs.sigma = sigma;
        run(pixel_values, height, width, pool_weights, patches, outs);
    }
};

// Deserialises a TensorRT engine; throws std::runtime_error on failure or in CPU-only builds.
std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& engine_path);

// Creates an ONNX Runtime CPU runner.
std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads = 4);

// Checks if a CUDA device is available.
bool cuda_device_available();

}  // namespace negicc

#endif  // NEGICC_DINOV3_RUNNER_H
