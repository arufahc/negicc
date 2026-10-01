// DinoV3Engine: native C++ photographic intent inference engine.
//
// py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (predict flow)
// py-ref: negicc_station/src/intent_model.py:320-375 @ 2f7ee4a (resolve_intent_backend & load_intent_model)

#include "dinov3_engine.h"
#include "dinov3_preprocess.h"
#include "dinov3_runner.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace negicc {

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string parent_dir(const std::string& p) {
    auto pos = p.rfind('/');
    return (pos == std::string::npos) ? "." : p.substr(0, pos);
}

}  // namespace

DinoV3Engine::DinoV3Engine() = default;
DinoV3Engine::~DinoV3Engine() = default;
DinoV3Engine::DinoV3Engine(DinoV3Engine&&) noexcept = default;
DinoV3Engine& DinoV3Engine::operator=(DinoV3Engine&&) noexcept = default;

bool DinoV3Engine::is_loaded() const { return m_runner != nullptr; }

const char* DinoV3Engine::backend_name() const { return m_runner ? m_runner->name() : "none"; }
bool DinoV3Engine::has_pooled() const { return m_runner ? m_runner->has_pooled() : false; }
bool DinoV3Engine::has_pooled_tone() const { return m_runner ? m_runner->has_pooled_tone() : false; }
bool DinoV3Engine::has_pool_weights() const { return m_runner ? m_runner->has_pool_weights() : false; }
size_t DinoV3Engine::memory_bytes() const { return m_runner ? m_runner->memory_bytes() : 0; }

// py-ref: negicc_station/src/intent_model.py:320-375 @ 2f7ee4a (resolve_intent_backend & load_intent_model)
bool DinoV3Engine::load(const std::string& model_spec, const std::string& backend, int num_threads,
                        int max_size, int patch_size) {
    m_runner.reset();
    m_max_size = max_size;
    m_patch_size = patch_size;
    if (backend != "auto" && backend != "cpu" && backend != "gpu") {
        std::cerr << "DinoV3Engine: backend must be 'auto', 'cpu' or 'gpu', got '" << backend << "'" << std::endl;
        return false;
    }

    // Resolve the ONNX model and the TensorRT engine of the model spec
    std::string onnx_path, engine_path;
    if (dir_exists(model_spec)) {
        if (file_exists(model_spec + "/model.onnx")) {
            onnx_path = model_spec + "/model.onnx";
        }
        if (file_exists(model_spec + "/model.engine")) engine_path = model_spec + "/model.engine";
    } else if (ends_with(model_spec, ".engine")) {
        engine_path = model_spec;
    } else {
        onnx_path = model_spec;
        if (file_exists(parent_dir(model_spec) + "/model.engine")) engine_path = parent_dir(model_spec) + "/model.engine";
    }

    try {
        if (backend != "cpu") {
            const bool has_gpu = cuda_device_available();
            if (has_gpu && !engine_path.empty()) {
                try {
                    m_runner = make_trt_runner(engine_path);
                    return true;
                } catch (const std::exception& e) {
                    if (backend == "gpu") throw;
                    std::cerr << "DinoV3Engine: TensorRT runner failed to load (" << e.what()
                              << "), falling back to CPU ONNX Runtime" << std::endl;
                    m_runner.reset();
                }
            }
            if (backend == "gpu") {
                std::cerr << "DinoV3Engine: backend 'gpu' requested but "
                          << (has_gpu ? "the model has no TensorRT engine (run `make engine`)" : "no CUDA device")
                          << std::endl;
                return false;
            }
        }

        if (onnx_path.empty() || !file_exists(onnx_path)) {
            std::cerr << "DinoV3Engine: no ONNX model found at " << model_spec
                      << " (expected model.onnx; run `make onnx_model`)" << std::endl;
            return false;
        }
        m_runner = make_ort_runner(onnx_path, num_threads);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "DinoV3Engine load failed: " << e.what() << std::endl;
        m_runner.reset();
        return false;
    }
}

// py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (predict implementation)
DinoV3InferenceResult DinoV3Engine::infer(const uint8_t* srgb_data, int width, int height, int stride) {
    if (!is_loaded()) {
        throw std::runtime_error("DinoV3Engine::infer called before model loaded.");
    }
    if (width <= 0 || height <= 0 || !srgb_data) {
        throw std::invalid_argument("DinoV3Engine::infer: invalid image dimensions or null data pointer");
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    DinoV3InferenceResult res;
    // py-ref: negicc_station/src/dino_geometry.py:30-40 @ 2f7ee4a (compute_aspect_preserved_shape)
    compute_aspect_preserved_shape(width, height, m_max_size, m_patch_size,
                                  res.target_w, res.target_h, res.grid_w, res.grid_h);
    res.num_patches = res.grid_w * res.grid_h;

    // 1. Antialiased bilinear resize to (target_w, target_h)
    // py-ref: negicc_station/src/tensorrt_intent_model.py:189 @ 2f7ee4a (PIL Image.Resampling.BILINEAR)
    std::vector<uint8_t> resized_u8((size_t)res.target_w * res.target_h * 3);
    antialiased_bilinear_resize_rgb(srgb_data, width, height, stride,
                                    resized_u8.data(), res.target_w, res.target_h);

    // 2. pixel_values [1, 3, target_h, target_w]: (u8 / 255 - mean) / std in float32
    // py-ref: negicc_station/src/dino_geometry.py:17-18 @ 2f7ee4a (DINO_MEAN, DINO_STD)
    // py-ref: negicc_station/src/tensorrt_intent_model.py:190-192 @ 2f7ee4a (ImageNet normalization)
    std::vector<float> pixel_values((size_t)3 * res.target_h * res.target_w);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdev[3] = {0.229f, 0.224f, 0.225f};
    const size_t plane = (size_t)res.target_h * res.target_w;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < res.target_h; ++y) {
        for (int x = 0; x < res.target_w; ++x) {
            const size_t i = (size_t)y * res.target_w + x;
            for (int c = 0; c < 3; ++c)
                pixel_values[c * plane + i] = (resized_u8[i * 3 + c] / 255.0f - mean[c]) / stdev[c];
        }
    }

    // 3. pool_weights [1, N, 3]
    // py-ref: negicc_station/src/dino_geometry.py:43-74 @ 2f7ee4a (compute_tone_weights)
    std::vector<float> pool_weights((size_t)res.num_patches * 3);
    compute_tone_weights(resized_u8.data(), res.grid_w, res.grid_h, m_patch_size, pool_weights.data());

    // 4. Multi-head forward pass
    // py-ref: negicc_station/src/tensorrt_intent_model.py:202-273 @ 2f7ee4a
    res.mu.resize((size_t)res.num_patches * 3);
    res.sigma.resize((size_t)res.num_patches * 3);

    DinoV3RunnerOutputs outs;
    outs.mu = res.mu.data();
    outs.sigma = res.sigma.data();

    const bool want_ranker = m_runner->has_pooled() || m_runner->has_pooled_tone();
    if (want_ranker) {
        // py-ref: negicc_station/src/tensorrt_intent_model.py:269-270 @ 2f7ee4a (features_3459 contiguous buffer)
        res.features_3459.assign(1152 + 3 * 768 + 3, 0.0f);
        if (m_runner->has_pooled()) {
            outs.pooled = res.features_3459.data();
            res.has_pooled = true;
        }
        if (m_runner->has_pooled_tone()) {
            outs.pooled_tone = res.features_3459.data() + 1152;
            res.has_pooled_tone = true;
        }
    }

    m_runner->run(pixel_values.data(), res.target_h, res.target_w, pool_weights.data(), res.num_patches, outs);

    if (res.has_pooled_tone) {
        // py-ref: negicc_station/src/tensorrt_intent_model.py:268 @ 2f7ee4a (tone_mass = tone_weights.mean(axis=0))
        res.tone_mass.resize(3, 0.0f);
        for (int b = 0; b < 3; ++b) {
            float sum = 0.0f;
            for (int i = 0; i < res.num_patches; ++i) {
                sum += pool_weights[i * 3 + b];
            }
            const float m = sum / (float)res.num_patches;
            res.tone_mass[b] = m;
            res.features_3459[3456 + b] = m;
        }
        res.pooled_tone.assign(outs.pooled_tone, outs.pooled_tone + 3 * 768);
    }
    if (res.has_pooled) {
        res.pooled.assign(outs.pooled, outs.pooled + 1152);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return res;
}

}  // namespace negicc
