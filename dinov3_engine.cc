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
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

bool dir_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::is_directory(p, ec);
}

// dir/name with the platform's separator.
std::string join_path(const std::string& dir, const std::string& name) {
    return (std::filesystem::path(dir) / name).string();
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

DinoV3Engine::DinoV3Engine() = default;
DinoV3Engine::~DinoV3Engine() = default;
DinoV3Engine::DinoV3Engine(DinoV3Engine&&) noexcept = default;
DinoV3Engine& DinoV3Engine::operator=(DinoV3Engine&&) noexcept = default;

bool DinoV3Engine::is_loaded() const { return runner_ != nullptr; }

bool DinoV3Engine::needs_adapter() const { return runner_ && runner_->has_tokens(); }

void DinoV3Engine::set_adapter(const void* onnx_data, size_t onnx_size, const std::string& name, int num_threads) {
    adapter_ = make_ort_adapter(onnx_data, onnx_size, num_threads);
    adapter_name_ = name;
}

void DinoV3Engine::load_adapter_file(const std::string& path, int num_threads) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("DinoV3Engine: cannot open adapter " + path);
    std::vector<uint8_t> bytes((size_t)f.tellg());
    f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()), (std::streamsize)bytes.size());
    // ONNX Runtime copies what it needs while building the session, but keep the bytes alive with the session.
    adapter_file_ = std::move(bytes);
    set_adapter(adapter_file_.data(), adapter_file_.size(), path, num_threads);
}

const char* DinoV3Engine::backend_name() const { return runner_ ? runner_->name() : "none"; }
bool DinoV3Engine::has_pooled() const { return runner_ ? runner_->has_pooled() : false; }
bool DinoV3Engine::has_pooled_tone() const { return runner_ ? runner_->has_pooled_tone() : false; }
bool DinoV3Engine::has_pool_weights() const { return runner_ ? runner_->has_pool_weights() : false; }
size_t DinoV3Engine::memory_bytes() const {
    return (runner_ ? runner_->memory_bytes() : 0) + (adapter_ ? adapter_->memory_bytes() : 0);
}

// py-ref: negicc_station/src/intent_model.py:320-375 @ 2f7ee4a (resolve_intent_backend & load_intent_model)
void DinoV3Engine::load(const std::string& model_spec, const std::string& backend, int num_threads,
                        int max_size, int patch_size) {
    runner_.reset();
    max_size_ = max_size;
    patch_size_ = patch_size;
    if (backend != "auto" && backend != "cpu" && backend != "gpu") {
        throw std::invalid_argument("DinoV3Engine: backend must be 'auto', 'cpu' or 'gpu', got '" + backend + "'");
    }

    // Resolve the ONNX model and the TensorRT engine of the model spec
    std::string onnx_path, engine_path;
    if (dir_exists(model_spec)) {
        // The backbone alone (adapters supplied separately) wins over a fused model in the same directory.
        const std::string stem =
            (file_exists(join_path(model_spec, "backbone.onnx")) || file_exists(join_path(model_spec, "backbone.engine")))
                ? "backbone" : "model";
        if (file_exists(join_path(model_spec, stem + ".onnx"))) onnx_path = join_path(model_spec, stem + ".onnx");
        if (file_exists(join_path(model_spec, stem + ".engine"))) engine_path = join_path(model_spec, stem + ".engine");
    } else if (ends_with(model_spec, ".engine")) {
        engine_path = model_spec;
    } else {
        onnx_path = model_spec;
        const std::string stem = model_spec.substr(0, model_spec.size() - (ends_with(model_spec, ".onnx") ? 5 : 0));
        if (file_exists(stem + ".engine")) engine_path = stem + ".engine";
    }

    if (backend != "cpu") {
        const bool has_gpu = cuda_device_available();
        if (has_gpu && !engine_path.empty()) {
            try {
                runner_ = make_trt_runner(engine_path);
                return;
            } catch (const std::exception& e) {
                if (backend == "gpu") throw;
                std::cerr << "DinoV3Engine: TensorRT runner failed to load (" << e.what()
                          << "), falling back to CPU ONNX Runtime" << std::endl;
                runner_.reset();
            }
        }
        if (backend == "gpu") {
            throw std::runtime_error(std::string("DinoV3Engine: backend 'gpu' requested but ") +
                                     (has_gpu ? "the model has no TensorRT engine (run `make dinov3_engine`)"
                                              : "no CUDA device"));
        }
    }

    if (onnx_path.empty() || !file_exists(onnx_path)) {
        throw std::runtime_error("DinoV3Engine: no ONNX model found at " + model_spec +
                                 " (expected backbone.onnx or model.onnx; run `make dinov3_model`)");
    }
    runner_ = make_ort_runner(onnx_path, num_threads);
}

void DinoV3Engine::load_memory(const void* engine_data, size_t engine_size, const void* onnx_data, size_t onnx_size,
                               const std::string& backend, int num_threads, int max_size, int patch_size) {
    runner_.reset();
    max_size_ = max_size;
    patch_size_ = patch_size;
    if (backend != "auto" && backend != "cpu" && backend != "gpu") {
        throw std::invalid_argument("DinoV3Engine: backend must be 'auto', 'cpu' or 'gpu', got '" + backend + "'");
    }

    if (backend != "cpu") {
        const bool has_gpu = cuda_device_available();
        if (has_gpu && engine_data) {
            try {
                runner_ = make_trt_runner(engine_data, engine_size);
                return;
            } catch (const std::exception& e) {
                if (backend == "gpu" || !onnx_data) throw;
                std::cerr << "DinoV3Engine: TensorRT runner failed to load (" << e.what()
                          << "), falling back to CPU ONNX Runtime" << std::endl;
                runner_.reset();
            }
        }
        if (backend == "gpu") {
            throw std::runtime_error(std::string("DinoV3Engine: backend 'gpu' requested but ") +
                                     (has_gpu ? "the built-in backbone has no TensorRT engine (USE_CUDA=1, run "
                                                "`make dinov3_engine`)"
                                              : "no CUDA device"));
        }
    }

    if (!onnx_data) {
        throw std::runtime_error("DinoV3Engine: the built-in backbone has no ONNX model (run `make dinov3_model`)");
    }
    runner_ = make_ort_runner(onnx_data, onnx_size, num_threads);
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
    compute_aspect_preserved_shape(width, height, max_size_, patch_size_,
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
    compute_tone_weights(resized_u8.data(), res.grid_w, res.grid_h, patch_size_, pool_weights.data());

    // 4. Multi-head forward pass
    // py-ref: negicc_station/src/tensorrt_intent_model.py:202-273 @ 2f7ee4a
    res.mu.resize((size_t)res.num_patches * 3);
    res.sigma.resize((size_t)res.num_patches * 3);

    DinoV3RunnerOutputs outs;
    std::vector<float> tokens;
    if (runner_->has_tokens()) {
        if (!adapter_) throw std::runtime_error("DinoV3Engine::infer: the backbone model needs an adapter");
        tokens.resize((size_t)res.num_patches * 768);
        outs.tokens = tokens.data();
    } else {
        outs.mu = res.mu.data();
        outs.sigma = res.sigma.data();
    }

    const bool want_ranker = runner_->has_pooled() || runner_->has_pooled_tone();
    if (want_ranker) {
        // py-ref: negicc_station/src/tensorrt_intent_model.py:269-270 @ 2f7ee4a (features_3459 contiguous buffer)
        res.features_3459.assign(1152 + 3 * 768 + 3, 0.0f);
        if (runner_->has_pooled()) {
            outs.pooled = res.features_3459.data();
            res.has_pooled = true;
        }
        if (runner_->has_pooled_tone()) {
            outs.pooled_tone = res.features_3459.data() + 1152;
            res.has_pooled_tone = true;
        }
    }

    runner_->run(pixel_values.data(), res.target_h, res.target_w, pool_weights.data(), res.num_patches, outs);
    // py-ref: negicc_station/src/export_intent_engine.py @ 93be1ec (FusedIntentNetwork: adapter(cat(h_photo, h_sem)))
    if (!tokens.empty()) adapter_->run(tokens.data(), res.num_patches, res.mu.data(), res.sigma.data());

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
