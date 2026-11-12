// The DINOv3 intent model: the inference runners (ONNX Runtime on the CPU, TensorRT on CUDA), the small heads that
// read the backbone's tokens (NeutralHead, ToneHead), and DinoV3Engine, which preprocesses a frame and runs them.

#ifndef DINOV3_ENGINE_H
#define DINOV3_ENGINE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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
// The ONNX Runtime CPU runners (dinov3_engine.cc) are built everywhere. The TensorRT runners (dinov3_trt_runner.cc)
// are built only with USE_CUDA=1 and run the device-specific backbone.engine and adapter.engine files
// `make dinov3_engine` builds. When USE_CUDA=0, dinov3_trt_runner.cc holds fallback stubs instead.
// These sources live in negicc and are consumed by negicc-station via git submodule at 3rd_party/negicc.
//
// py-ref: negicc_station/src/export_intent_engine.py:49-70, 104-112 @ 2f7ee4a (output contract)
// py-ref: negicc_station/src/tensorrt_intent_model.py:249-273 @ 2f7ee4a (multi-head output parsing)
// py-ref: negicc_station/src/dino_geometry.py:23-27 @ 2f7ee4a (IntentPrediction namedtuple)
// --------------------------------------------------------------------------------------------------------------------

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
// py-ref: negicc_station/src/tensorrt_intent_model.py:249-273 @ 2f7ee4a
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

// Creates an ONNX Runtime CPU runner.
std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads = 4);
// The same from an .onnx held in memory.
std::unique_ptr<DinoV3Runner> make_ort_runner(const void* onnx_data, size_t onnx_size, int num_threads = 4);

// Checks if a CUDA device is available.
bool cuda_device_available();

// --------------------------------------------------------------------------------------------------------------------
// ONNX Runtime CPU runners
// --------------------------------------------------------------------------------------------------------------------

// Creates an ONNX Runtime CPU runner.
std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads);
std::unique_ptr<DinoV3Runner> make_ort_runner(const void* onnx_data, size_t onnx_size, int num_threads);

// --------------------------------------------------------------------------------------------------------------------
// Token heads
// The small heads that read the DINOv3 tokens alongside the intent adapter, each a plain C++ implementation that
// needs no runtime of its own: the neutral-patch head (NeutralHead), which gives the white-light curve fit its neutral
// anchors, and the tone heads (ToneHead), which give the white-light tone stage its targets.
// --------------------------------------------------------------------------------------------------------------------

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
                z += (double)w2_[j] * (0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440)));
            }
            p[i] = (float)(1.0 / (1.0 + std::exp(-z)));
        }
    }

private:
    int in_dim_ = 0, hidden_ = 0;
    std::vector<float> mean_, std_, w1t_, b1_, w2_;
    float b2_ = 0.0f;
};

// The white-light tone heads: from the frozen DINOv3 features of a preview, the tone a lab scan of the frame would
// have, as 11 descriptors of its pixel L* and chroma at network size (mean and standard deviation of each):
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

    // Parses tone_head.bin (magic "TNH1", uint32 in_dim, hidden1, hidden2, targets, then float32 xm, xs, w1, b1, w2, b2,
    // w3, b3, ym, ys). False, with the reason in error, when the data is not such a file.
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
        in_dim_ = (int)d;
        h1_ = (int)h1;
        h2_ = (int)h2;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }

    // mean[kTargets] and sd[kTargets] from the 1920 features. False when the head is not loaded.
    bool run(const float* features, double* mean, double* sd) const {
        if (!loaded()) return false;
        auto gelu = [](double a) { return 0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440)); };
        std::vector<double> x(in_dim_), a1(h1_), a2(h2_);
        for (int k = 0; k < in_dim_; ++k) x[k] = (double)((features[k] - xm_[k]) / xs_[k]);  // float, as in training
        for (int j = 0; j < h1_; ++j) {
            const float* w = w1_.data() + (size_t)j * in_dim_;
            double s = b1_[j];
            for (int k = 0; k < in_dim_; ++k) s += (double)w[k] * x[k];
            a1[j] = gelu(s);
        }
        for (int j = 0; j < h2_; ++j) {
            const float* w = w2_.data() + (size_t)j * h1_;
            double s = b2_[j];
            for (int k = 0; k < h1_; ++k) s += (double)w[k] * a1[k];
            a2[j] = gelu(s);
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

    // The neutral-patch head (NeutralHead) from a neutral_head.bin held in memory: with it, infer() also gives
    // DinoV3InferenceResult::neutral_p from the backbone's tokens. When the backbone and adapter run on the GPU the
    // tokens are copied to the host once per inference for it. Null data clears it. Throws std::runtime_error on
    // a bad file or a fused model (no tokens to read).
    void set_neutral_head(const void* data, size_t size);
    bool has_neutral_head() const { return neutral_head_ != nullptr; }
    // With it on, infer() also gives DinoV3InferenceResult::token_sd, the spread of the backbone's tokens over the
    // patches: with the pooled features, the input of the white-light tone heads (ToneHead). Like the neutral head
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
