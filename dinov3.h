// Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


// The DINOv3 intent model and the Auto search that runs on it, in the pipeline's order: the preprocessing of a
// preview, the inference runners and the token heads, DinoV3Engine, and the Auto solver. dinov3.cc holds the code.

#ifndef DINOV3_H
#define DINOV3_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rgb_view.h"
#include "profile_bundle.h"

// ====================================================================================================================
// Preprocessing
// Takes an 8-bit RGB preview; gives the patch-aligned network size, the resized image and the tone-bin weights of
// each patch, as the network was trained on them.
// ====================================================================================================================

// Antialiased 2-pass separable bilinear resize on 8-bit RGB with kernel support widening;
// bit-exact with Pillow's Image.Resampling.BILINEAR (fixed point, 8-bit intermediate).
void antialiased_bilinear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                                     uint8_t* dst, int dst_w, int dst_h);

// cv2.resize(..., interpolation=cv2.INTER_LINEAR) on 8-bit RGB (the fixed-point SIMD path): a bilinear point sample.
// The search's residual does not use it (it reads trained_cells_lab, below); it is kept for
// negicc-station's parity test of this resize (src/test_dinov3_preprocess.cpp).
void cv_linear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int dst_w, int dst_h);

// intent_model.compute_tone_weights on a (grid_w * patch) x (grid_h * patch) RGB image: [N, 3] weights.
void compute_tone_weights(const uint8_t* img_u8, int grid_w, int grid_h, int patch_size, float* out_weights);

// Computes patch-aligned target dimensions preserving aspect ratio (intent_model.compute_aspect_preserved_shape)
void compute_aspect_preserved_shape(int w, int h, int max_size, int patch_size,
                                    int& target_w, int& target_h, int& grid_w, int& grid_h);

// ====================================================================================================================
// Engine
// The DINOv3 intent model: the inference runners (ONNX Runtime on the CPU, TensorRT on CUDA), the small heads that
// read the backbone's tokens (NeutralHead, ToneHead), and DinoV3Engine, which preprocesses a frame and runs them.
// Takes an 8-bit sRGB preview and the backbone, its per-stock adapter and the token heads; gives the intent field
// (mu, sigma per patch), the ranker features and the heads' outputs (DinoV3InferenceResult).
// ====================================================================================================================

// --------------------------------------------------------------------------------------------------------------------
// Runners
// Inference runners behind DinoV3Engine: one forward pass on preprocessed tensors.
//
// A runner holds either a fused intent model (backbone + adapter: outputs "mu" and "sigma") or the DINOv3 backbone
// alone (output "tokens", the layer 9 + 12 patch tokens [1, N, 768]). A backbone runner is paired with a
// DinoV3AdapterRunner, the per-stock adapter that maps the tokens to the intent field, so one backbone file serves
// every film stock and the adapters ship inside neg_process (embedded_data.h).
//
// The adapter runs where the backbone runs: ONNX Runtime with an ONNX Runtime backbone, TensorRT with a TensorRT
// backbone, reading the backbone's tokens from device memory (DinoV3Runner::device_tokens) so they never leave the GPU.
//
// The ONNX Runtime CPU runners (dinov3.cc) are built everywhere. The TensorRT runners (dinov3.cc, after them)
// are built only with USE_CUDA=1 and run the device-specific backbone.engine and adapter.engine files
// `make dinov3_engine` builds. When USE_CUDA=0, dinov3.cc holds fallback stubs instead.
// These sources live in negicc and are consumed by negicc-station via git submodule at 3rd_party/negicc.
// --------------------------------------------------------------------------------------------------------------------

// Multi-head inference result matching Python IntentPrediction.
struct DinoV3InferenceResult {
    std::vector<float> mu;            // [1, N, 3] Target Lab color field
    std::vector<float> sigma;         // [1, N, 3] Tolerance field
    std::vector<float> pooled;        // [1, 1152] Exposure ranker CLS+mean features
    std::vector<float> pooled_tone;   // [1, 3, 768] Tone-stratified ranker features
    std::vector<float> tone_mass;     // [3] Shadows, mid-tones, highlights mass fractions
    std::vector<float> features_3459; // [3459] Contiguous concatenated: pooled + pooled_tone + tone_mass
    std::vector<float> neutral_p;     // [N] p(neutral grey) per patch, when a neutral head is set (else empty)
    std::vector<float> token_sd;      // [768] the tokens' standard deviation over the patches (N - 1 in the
                                      // denominator), with DinoV3Engine::set_token_spread(true) (else empty)
    double neutral_ms = 0.0;          // time of the neutral head, included in duration_ms
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
struct DinoV3RunnerOutputs {
    float* tokens = nullptr;       // [1, patches, 768] layer 9 + 12 patch tokens (backbone models)
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
    // The model ends at the backbone ("tokens") rather than the intent field ("mu", "sigma").
    virtual bool has_tokens() const { return false; }
    virtual bool has_pooled() const { return false; }
    virtual bool has_pooled_tone() const { return false; }
    virtual size_t memory_bytes() const { return 0; }
    // Device address of the last run's "tokens" output when the backbone runs on TensorRT (valid until the next run),
    // so a TensorRT adapter reads the tokens where they are; null for other runners.
    virtual const void* device_tokens() const { return nullptr; }
    // Device-to-host copies of the tokens made so far (tests: a GPU backbone paired with a GPU adapter makes none).
    virtual int token_downloads() const { return 0; }

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

// Per-stock adapter: tokens [1, N, 768] -> mu, sigma [1, N, 3].
class DinoV3AdapterRunner {
public:
    virtual ~DinoV3AdapterRunner() = default;
    // Tokens in host memory.
    virtual void run(const float* tokens, int patches, float* mu, float* sigma) = 0;
    // The adapter runs on the GPU and takes its tokens from device memory (run_device).
    virtual bool on_device() const { return false; }
    // Tokens in device memory (DinoV3Runner::device_tokens of a TensorRT backbone); mu and sigma land on the host.
    virtual void run_device(const void* /*device_tokens*/, int /*patches*/, float* /*mu*/, float* /*sigma*/);
    virtual size_t memory_bytes() const { return 0; }
};

// ONNX Runtime adapter from an in-memory adapter.onnx (input "tokens", outputs "mu", "sigma").
std::unique_ptr<DinoV3AdapterRunner> make_ort_adapter(const void* onnx_data, size_t onnx_size, int num_threads = 1);

// TensorRT adapter from an in-memory adapter.engine (make dinov3_engine); throws std::runtime_error on failure or in
// CPU-only builds.
std::unique_ptr<DinoV3AdapterRunner> make_trt_adapter(const void* engine_data, size_t engine_size);

// Deserialises a TensorRT engine; throws std::runtime_error on failure or in CPU-only builds.
std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& engine_path);
// The same from a serialized engine held in memory.
std::unique_ptr<DinoV3Runner> make_trt_runner(const void* engine_data, size_t engine_size);

// Creates an ONNX Runtime CPU runner. num_threads: the runtime's own threads; 0 takes the program's thread count
// (omp_get_max_threads(): see default_thread_count in rgb_view.h), so the network runs with the count every parallel
// region has.
std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads = 0);
// The same from an .onnx held in memory.
std::unique_ptr<DinoV3Runner> make_ort_runner(const void* onnx_data, size_t onnx_size, int num_threads = 0);

// Checks if a CUDA device is available.
bool cuda_device_available();

// --------------------------------------------------------------------------------------------------------------------
// Token heads
// The small heads that read the DINOv3 tokens alongside the intent adapter, each a plain C++ implementation that
// needs no runtime of its own: the neutral-patch head (NeutralHead), which gives the white-light curve fit its neutral
// anchors, and the tone heads (ToneHead), which give the tone stage its targets on both lights.
// --------------------------------------------------------------------------------------------------------------------

// The exact GELU (Hendrycks and Gimpel, 2016; torch.nn.GELU's default), x Phi(x), in double: the heads' activation.
// 0.7071... is 1 / sqrt(2).
inline double exact_gelu(double a) { return 0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440)); }

// The neutral-patch head: per DINOv3 patch, the probability that the patch is neutral grey in a good rendering of
// the frame, from the backbone's layer 9 + 12 tokens [N, 768] (DinoV3Runner "tokens", the adapters' input).
// p = sigmoid(w2 . gelu(W1 ((t - mean) / std) + b1) + b2), W1 [256][768], exact (erf) GELU, as the PyTorch model it
// was trained as. Weights: models/neutral_head/neutral_head.bin (models/neutral_head/export_neutral_head.py writes
// it from neutral_head.pt; README.md there has the training and validation).
// A plain C++ implementation (about 0.2 GFLOP per 672-patch preview), so the head needs no runtime of its own and
// runs wherever the tokens are on the host.
class NeutralHead {
public:
    // Parses neutral_head.bin (magic "NPH1", uint32 in_dim and hidden, then float32 mean, std, w1, b1, w2, b2).
    // False, with the reason in error, when the data is not such a file.
    bool load(const void* data, size_t size, std::string& error) {
        in_dim_ = hidden_ = 0;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (!p || size < 12 || std::memcmp(p, "NPH1", 4) != 0) {
            error = "not a neutral head file (magic NPH1)";
            return false;
        }
        uint32_t d = 0, h = 0;
        std::memcpy(&d, p + 4, 4);
        std::memcpy(&h, p + 8, 4);
        const size_t floats = 2 * (size_t)d + (size_t)h * d + 2 * (size_t)h + 1;
        if (d == 0 || h == 0 || d > 65536 || h > 65536 || size != 12 + 4 * floats) {
            error = "neutral head file has the wrong size for in_dim " + std::to_string(d) + ", hidden " +
                    std::to_string(h);
            return false;
        }
        std::vector<float> v(floats);
        std::memcpy(v.data(), p + 12, 4 * floats);
        const float* f = v.data();
        mean_.assign(f, f + d);
        f += d;
        std_.assign(f, f + d);  // divided by, as in PyTorch (not multiplied by a reciprocal)
        f += d;
        // w1 transposed to [in][hidden], so the inner loop runs over contiguous hidden units (vectorises without
        // reassociating the sums, which -ffp-contract=off and the absence of -ffast-math keep in order).
        w1t_.resize((size_t)d * h);
        for (uint32_t j = 0; j < h; ++j)
            for (uint32_t k = 0; k < d; ++k) w1t_[(size_t)k * h + j] = f[(size_t)j * d + k];
        f += (size_t)h * d;
        b1_.assign(f, f + h);
        f += h;
        w2_.assign(f, f + h);
        f += h;
        b2_ = *f;
        for (const float x : v) {
            if (!std::isfinite(x)) {
                error = "neutral head file has a weight that is not finite";
                return false;
            }
        }
        for (const float s : std_) {
            if (!(s > 0.0f)) {
                error = "neutral head file has a standard deviation that is not positive";
                return false;
            }
        }
        in_dim_ = (int)d;
        hidden_ = (int)h;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }
    int in_dim() const { return in_dim_; }

    // p[i] for the n patches of tokens (n x in_dim, row-major).
    void run(const float* tokens, int n, float* p) const {
        const int d = in_dim_, h = hidden_;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::vector<float> x(d), acc(b1_);
            const float* t = tokens + (size_t)i * d;
            for (int k = 0; k < d; ++k) x[k] = (t[k] - mean_[k]) / std_[k];
            for (int k = 0; k < d; ++k) {
                const float xk = x[k];
                const float* w = w1t_.data() + (size_t)k * h;
                for (int j = 0; j < h; ++j) acc[j] += xk * w[j];
            }
            double z = b2_;
            for (int j = 0; j < h; ++j) {
                const double a = acc[j];
                z += (double)w2_[j] * exact_gelu(a);
            }
            p[i] = (float)(1.0 / (1.0 + std::exp(-z)));
        }
    }

private:
    int in_dim_ = 0, hidden_ = 0;
    std::vector<float> mean_, std_, w1t_, b1_, w2_;
    float b2_ = 0.0f;
};

// The tone heads of the tone stage (both lights): from the frozen DINOv3 features of a preview, the tone a lab scan of
// the frame would have, as 11 descriptors of its pixel L* and chroma at network size (mean and standard deviation of
// each):
//   L2 L5 L10 L50 L90 L95 L98 (L* percentiles), IQR (L75 - L25), R9010 (L90 - L10), Cmean (mean C*ab), C90.
// Input: the 1920 pooled features [CLS of layer 12, mean patch token of layer 9, mean patch token of layer 12 (the
// backbone's "pooled" output, 1152), the patch tokens' standard deviation (DinoV3InferenceResult::token_sd, 768)],
// standardised with the stored xm and xs. Model: Linear(1920, 512), GELU (exact, erf), Linear(512, 256), GELU,
// Linear(256, 22); mean = out[j] * ys[j] + ym[j], sd = exp(clamp(out[11 + j], -6, 4) / 2) * ys[j], as the PyTorch
// model it was trained as. Weights: models/tone_head/tone_head.bin (models/tone_head/export_tone_head.py writes it
// from tone_head.pt; README.md there has the training and validation). Plain C++, about 1.1 M multiply-adds per run.
class ToneHead {
public:
    enum Target { kL2, kL5, kL10, kL50, kL90, kL95, kL98, kIQR, kR9010, kCmean, kC90, kTargets };
    static constexpr int kInDim = 1920;
    static const char* target_name(int t) {
        static const char* names[kTargets] = {"L2", "L5", "L10", "L50", "L90", "L95", "L98", "IQR", "R9010",
                                              "Cmean", "C90"};
        return t >= 0 && t < kTargets ? names[t] : "";
    }

    // Parses tone_head.bin (magic "TNH1", uint32 in_dim, hidden1, hidden2, targets, then float32 xm, xs, w1, b1, w2,
    // b2, w3, b3, ym, ys). False, with the reason in error, when the data is not such a file.
    bool load(const void* data, size_t size, std::string& error) {
        in_dim_ = 0;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (!p || size < 20 || std::memcmp(p, "TNH1", 4) != 0) {
            error = "not a tone head file (magic TNH1)";
            return false;
        }
        uint32_t d = 0, h1 = 0, h2 = 0, n = 0;
        std::memcpy(&d, p + 4, 4);
        std::memcpy(&h1, p + 8, 4);
        std::memcpy(&h2, p + 12, 4);
        std::memcpy(&n, p + 16, 4);
        if (d != (uint32_t)kInDim || n != (uint32_t)kTargets || h1 == 0 || h2 == 0 || h1 > 65536 || h2 > 65536) {
            error = "tone head file is not 1920 features to 11 targets";
            return false;
        }
        const size_t floats = 2 * (size_t)d + (size_t)h1 * d + h1 + (size_t)h2 * h1 + h2 + 2 * (size_t)n * h2 +
                              2 * (size_t)n + 2 * (size_t)n;
        if (size != 20 + 4 * floats) {
            error = "tone head file has the wrong size";
            return false;
        }
        std::vector<float> v(floats);
        std::memcpy(v.data(), p + 20, 4 * floats);
        const float* f = v.data();
        auto take = [&f](std::vector<float>& dst, size_t k) {
            dst.assign(f, f + k);
            f += k;
        };
        take(xm_, d);
        take(xs_, d);
        take(w1_, (size_t)h1 * d);
        take(b1_, h1);
        take(w2_, (size_t)h2 * h1);
        take(b2_, h2);
        take(w3_, 2 * (size_t)n * h2);
        take(b3_, 2 * (size_t)n);
        take(ym_, n);
        take(ys_, n);
        for (const float x : v) {
            if (!std::isfinite(x)) {
                error = "tone head file has a weight that is not finite";
                return false;
            }
        }
        for (const float s : xs_) {
            if (!(s > 0.0f)) {
                error = "tone head file has a feature scale that is not positive";
                return false;
            }
        }
        in_dim_ = (int)d;
        h1_ = (int)h1;
        h2_ = (int)h2;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }

    // mean[kTargets] and sd[kTargets] from the 1920 features. False when the head is not loaded.
    bool run(const float* features, double* mean, double* sd) const {
        if (!loaded()) return false;
        std::vector<double> x(in_dim_), a1(h1_), a2(h2_);
        for (int k = 0; k < in_dim_; ++k) x[k] = (double)((features[k] - xm_[k]) / xs_[k]);  // float, as in training
        for (int j = 0; j < h1_; ++j) {
            const float* w = w1_.data() + (size_t)j * in_dim_;
            double s = b1_[j];
            for (int k = 0; k < in_dim_; ++k) s += (double)w[k] * x[k];
            a1[j] = exact_gelu(s);
        }
        for (int j = 0; j < h2_; ++j) {
            const float* w = w2_.data() + (size_t)j * h1_;
            double s = b2_[j];
            for (int k = 0; k < h1_; ++k) s += (double)w[k] * a1[k];
            a2[j] = exact_gelu(s);
        }
        for (int t = 0; t < kTargets; ++t) {
            double o[2];
            for (int half = 0; half < 2; ++half) {
                const int j = half * kTargets + t;
                const float* w = w3_.data() + (size_t)j * h2_;
                double s = b3_[j];
                for (int k = 0; k < h2_; ++k) s += (double)w[k] * a2[k];
                o[half] = s;
            }
            mean[t] = o[0] * ys_[t] + ym_[t];
            sd[t] = std::exp(0.5 * std::clamp(o[1], -6.0, 4.0)) * ys_[t];
        }
        return true;
    }

private:
    int in_dim_ = 0, h1_ = 0, h2_ = 0;
    std::vector<float> xm_, xs_, w1_, b1_, w2_, b2_, w3_, b3_, ym_, ys_;
};

// --------------------------------------------------------------------------------------------------------------------
// Engine
// --------------------------------------------------------------------------------------------------------------------

// The fused DINOv3 intent model (src/export_intent_engine.py): inputs pixel_values, pool_weights; outputs mu,
// sigma, pooled, pooled_tone. Preprocessing reproduces intent_model.predict / dino_geometry.
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
    // num_threads: ONNX Runtime's threads for the backbone on the CPU; 0 takes the program's thread count (see
    // make_ort_runner).
    // Throws std::runtime_error (std::invalid_argument for a bad backend) when the model cannot be loaded.
    void load(const std::string& model_spec, const std::string& backend = "auto", int num_threads = 0,
              int max_size = 512, int patch_size = 16);

    // The backbone built into neg_process, held in memory: its TensorRT engine (USE_CUDA=1 builds) and its .onnx,
    // each null when not built in. backend picks between them as load() does: "auto" runs the engine when this build
    // has a CUDA device and falls back to the .onnx when the engine is missing or fails to load. Throws as load()
    // does.
    void load_memory(const void* engine_data, size_t engine_size, const void* onnx_data, size_t onnx_size,
                     const std::string& backend = "auto", int num_threads = 0, int max_size = 512,
                     int patch_size = 16);

    // Adapter for a backbone model, after load(): its adapter.onnx and its TensorRT adapter.engine (make
    // dinov3_engine), held in memory (embedded in neg_process) or read from files. The adapter runs where the
    // backbone runs: a TensorRT backbone needs the engine (there is no CPU adapter behind a GPU backbone), an ONNX
    // Runtime backbone uses the .onnx. name is reported by adapter_name(). Ignored by fused models, which carry their
    // own adapter. num_threads: ONNX Runtime's threads for the adapter on the CPU; 0 takes the program's thread
    // count (see make_ort_runner). Throws std::runtime_error when the adapter the backbone needs is missing or cannot
    // be loaded.
    void set_adapter(const void* onnx_data, size_t onnx_size, const std::string& name, int num_threads = 1,
                     const void* engine_data = nullptr, size_t engine_size = 0);
    // path: adapter.onnx or adapter.engine (or a directory holding them); the other file is taken from beside it. Only
    // the file the backbone's runtime takes is read: adapter.engine behind a TensorRT backbone (refused when it is
    // older than the adapter.onnx beside it, whose earlier weights it holds), adapter.onnx behind ONNX Runtime, neither
    // for a fused model. adapter_path() names the file read.
    void load_adapter_file(const std::string& path, int num_threads = 1);
    // The adapter file load_adapter_file read (empty after set_adapter, or for a fused model).
    const std::string& adapter_path() const { return adapter_path_; }
    // The adapter runs on the GPU (TensorRT).
    bool adapter_on_device() const { return adapter_ && adapter_->on_device(); }
    // Device-to-host copies of the backbone's tokens so far (0 when the backbone and adapter both run on the GPU).
    int token_downloads() const { return runner_ ? runner_->token_downloads() : 0; }
    const std::string& adapter_name() const { return adapter_name_; }
    // The loaded model is a backbone and needs an adapter.
    bool needs_adapter() const;

    // The neutral-patch head (NeutralHead) from a neutral_head.bin held in memory: with it, infer() also gives
    // DinoV3InferenceResult::neutral_p from the backbone's tokens. When the backbone and adapter run on the GPU the
    // tokens are copied to the host once per inference for it. Null data clears it. Throws std::runtime_error on
    // a bad file or a fused model (no tokens to read).
    void set_neutral_head(const void* data, size_t size);
    bool has_neutral_head() const { return neutral_head_ != nullptr; }
    // With it on, infer() also gives DinoV3InferenceResult::token_sd, the spread of the backbone's tokens over the
    // patches: with the pooled features, the input of the tone heads (ToneHead). Like the neutral head
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
    DinoV3InferenceResult infer(const uint8_t* srgb_data, int width, int height, int stride);

private:
    std::unique_ptr<DinoV3Runner> runner_;
    std::unique_ptr<DinoV3AdapterRunner> adapter_;
    std::unique_ptr<NeutralHead> neutral_head_;
    bool token_spread_ = false;
    std::vector<uint8_t> adapter_file_;
    std::string adapter_name_;
    std::string adapter_path_;
    int max_size_ = 512;
    int patch_size_ = 16;
};

// ====================================================================================================================
// Auto solver
// Takes the frame (an RgbView in camera RGB), its profile and matrix, and the engine; gives the (E, g, b) the intent
// field asks for, with the search's log (DinoSolveResult).
// ====================================================================================================================

// Flags for pin_mask (manual argument overrides)
constexpr unsigned PIN_E = 1 << 0;
constexpr unsigned PIN_G = 1 << 1;
constexpr unsigned PIN_B = 1 << 2;

struct IntentGains {
  float e = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
};

struct IntentLoss {
  float L = 0.0f;
  float a = 0.0f;
  float b = 0.0f;
  float clip = 0.0f;  // in percent [0, 100]
};

// The input curves of a profile with the gains g and b in them, after the knee (the print, np_print.h PrintContinuum):
// three tables of trc_size entries each, as ParsedFilmProfile::in_trc holds them.
using GainCurves = std::function<void(float g, float b, std::vector<float> trc[3])>;

struct SolverConfig {
  const ProfileData* prof = nullptr;
  // When set, a candidate's g and b go into prof's input curves, after the knee, and its matrix carries E alone, as
  // the print's conversion renders it; intent_cells and intent_loss (the search's optimiser) read it.
  const GainCurves* gain_curves = nullptr;
  float knee = 0.95f;
  // Highlight shoulder x_hi (color_pipeline.h set_shoulder); 1.0 is off.
  float shoulder = 1.0f;
  float enc_e_min = 0.0f;
  float enc_e_max = 0.0f;
  // The values pinned parameters (pin_mask) take in optimize_target_for_intent when it is given no start; with a
  // start they stay at the start's.
  float pin_e = 1.0f, pin_g = 1.0f, pin_b = 1.0f;
};

struct DinoIterationLog {
  char tag = ' ';  // 'b' for bootstrap, '0', '1', etc.
  int step = 0;
  float e = 1.0f;  // normalized E (relative to E_c)
  float g = 1.0f;
  float b = 1.0f;
  float residual = -1.0f;  // -1 if unmeasured (bootstrap)
  float loss_L = 0.0f;
  float loss_a = 0.0f;
  float loss_b = 0.0f;
  float clip_pct = 0.0f;
  float penalty_clip = 0.0f;
  float penalty_anchor = 0.0f;
  float objective = 0.0f;
  float dp = 0.0f;
  float rel = 0.0f;
};

struct DinoSolveResult {
  std::string status = "CONVERGED";  // "CONVERGED", "DIVERGED", "MAX_ITERS", "PINNED"
  int best_iteration = -1;
  float best_residual = 0.0f;
  IntentGains solved_relative;  // relative to user baseline (E_c * e_best, g_best, b_best)
  IntentGains solved_absolute;  // E_user * E_c * e_best, g_user * g_best, b_user * b_best
  float e_center = 1.0f;
  bool anchor_clamped = false;  // E_c bisection did not straddle median L* 50
  // The default e_bounds, as ratios to e_center: 2^-1.5 and 2^1.5, 1.5 stops either way.
  float bounds_lo = 0.35355339f;
  float bounds_hi = 2.82842712f;
  std::vector<DinoIterationLog> iterations;
  std::vector<float> best_mu;
  std::vector<float> best_sigma;
  std::vector<float> tone_mass;
  int grid_w = 0;
  int grid_h = 0;
  int target_w = 0;
  int target_h = 0;
  int total_inferences = 0;
  double timing_views_ms = 0.0;
  double timing_render_ms = 0.0;
  double timing_infer_ms = 0.0;
  double timing_residual_ms = 0.0;
  double timing_search_ms = 0.0;
  double timing_total_ms = 0.0;
};

// The L*a*b* intent_loss scores for one candidate: each pixel of the camera-RGB grid rendered at cand as the loss
// renders it (displayed value, after the shoulder), n_px x 3 values into lab_out.
void intent_cells(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                  const IntentGains& cand, float* lab_out);

// The pipeline configuration and the matrix (written to m) cfg renders candidate cand with: the gains in the matrix,
// or, with cfg.gain_curves, in prof's input curves (trc holds the tables the configuration points into).
negicc::PipelineConfig candidate_config(const SolverConfig& cfg, const float* merged, const IntentGains& cand,
                                        float m[9], std::vector<float> trc[3]);

// Batched loss calculation across candidate matrices
void intent_loss(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                 const IntentGains* cand, int n_cand, const float* mu, const float* sigma, IntentLoss* out);

// The shoulder x_hi (color_pipeline.h) of a max-channel histogram of the pre-shoulder render (max_hist_bin layout):
// the x_hi that maps its 99.5% point to sRGB kShoulderAnchorCode (shoulder_x_hi_anchored, the roll-off); q995_out
// gets the point.
float shoulder_x_hi_from_hist(const uint64_t* hist, float* q995_out = nullptr);

// Coordinate descent parameter search against a frozen intent field (mu, sigma). Without e_bounds, E moves within
// [0.40, 1.80] from no start, or within [0.20, 3.0] around a start (one outside that range is moved to its nearer end).
float optimize_target_for_intent(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                                 int gw, int gh, const float* mu, const float* sigma,
                                 const IntentGains* initial, const float* e_bounds /* lo, hi, center or null */,
                                 unsigned pin_mask, IntentGains& out);

// The preview's L*a*b* per intent-grid cell, made as the intent adapter's training targets are made
// (negicc-station training/extract.py, patch_lab_targets): the encoded preview (w x h, 3 bytes per pixel) resampled in
// linear light by area average to gw*patch x gh*patch pixels, CIELAB per pixel, averaged over each patch. gw*gh*3
// values, row-major, into lab_out.
void trained_cells_lab(const uint8_t* preview_u8, int w, int h, int gw, int gh, int patch, float* lab_out);

// Mean squared z-score of the preview's trained cells (trained_cells_lab) against the intent field predicted from it.
float self_consistency_residual(const uint8_t* preview_u8, int w, int h, int gw, int gh, int patch,
                                const float* mu, const float* sigma);

// The same residual from trained cells already computed (trained_cells_lab's gw*gh*3 values, n = gw*gh):
// self_consistency_residual is trained_cells_lab followed by this.
float residual_from_cells(const float* lab, int n, const float* mu, const float* sigma);

// Solves optimal (E, g, b) parameters against active DINOv3 intent field
bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const uint16_t (*image)[4], int w, int h,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out);
bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const negicc::RgbView& view,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out);

#endif  // DINOV3_H
