// TensorRT runners (USE_CUDA=1 builds, which define NEGICC_USE_CUDA): the DINOv3 backbone (backbone.engine) or a
// fused intent model (model.engine), and the per-stock adapter (adapter.engine), all strict-FP32 engines
// `make dinov3_engine` builds on the device. A backbone keeps its tokens on the device; the adapter binds that buffer
// as its input, so only the intent field and the ranker features come back to the host.
//
// py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (TensorRTPhotographicIntentModel)
//
// A USE_CUDA=0 build compiles the stubs at the end of this file instead: no CUDA device, and runners that throw.
#include "dinov3_engine.h"

#include <stdexcept>

#ifdef NEGICC_USE_CUDA

#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

TrtLogger g_logger;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Device buffer that grows to the largest size requested and is reused across calls.
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void* reserve(size_t n) {
        if (n > bytes) {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
            check_cuda(cudaMalloc(&ptr, n), "cudaMalloc");
            bytes = n;
        }
        return ptr;
    }
};

// py-ref: negicc_station/src/tensorrt_intent_model.py:65-170 @ 2f7ee4a (TrtRunner lifecycle and buffer management)
class TrtRunner : public DinoV3Runner {
public:
    explicit TrtRunner(const std::string& engine_path) {
        std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open TensorRT engine " + engine_path);
        std::vector<char> blob((size_t)f.tellg());
        f.seekg(0);
        f.read(blob.data(), (std::streamsize)blob.size());
        init(blob.data(), blob.size(), engine_path);
    }

    // From a serialized engine held in memory (the backbone built into neg_process).
    TrtRunner(const void* engine_data, size_t engine_size) { init(engine_data, engine_size, "[built-in]"); }

    void init(const void* engine_data, size_t engine_size, const std::string& name) {
        engine_bytes_ = engine_size;
        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(engine_data, engine_size));
        if (!engine_) throw std::runtime_error("cannot deserialise TensorRT engine " + name);
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");

        // py-ref: negicc_station/src/tensorrt_intent_model.py:118-122 @ 2f7ee4a (detecting optional multi-head outputs)
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (std::strcmp(name, "pool_weights") == 0) has_pool_weights_ = true;
            if (std::strcmp(name, "tokens") == 0) has_tokens_ = true;
            if (std::strcmp(name, "pooled") == 0) has_pooled_ = true;
            if (std::strcmp(name, "pooled_tone") == 0) has_pooled_tone_ = true;
        }
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~TrtRunner() override {
        if (stream_) cudaStreamDestroy(stream_);
    }

    const char* name() const override { return "gpu (TensorRT)"; }
    bool has_pool_weights() const override { return has_pool_weights_; }
    bool has_tokens() const override { return has_tokens_; }
    bool has_pooled() const override { return has_pooled_; }
    bool has_pooled_tone() const override { return has_pooled_tone_; }
    const void* device_tokens() const override { return has_tokens_ ? tok_.ptr : nullptr; }
    int token_downloads() const override { return token_downloads_; }

    // py-ref: negicc_station/src/tensorrt_intent_model.py:153-169 @ 2f7ee4a (memory_bytes)
    size_t memory_bytes() const override {
        return engine_bytes_ + pv_.bytes + pw_.bytes + tok_.bytes + mu_.bytes + sigma_.bytes +
               pooled_.bytes + pooled_tone_.bytes;
    }

    // py-ref: negicc_station/src/tensorrt_intent_model.py:200-274 @ 2f7ee4a (predict / execution loop)
    void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
             const DinoV3RunnerOutputs& outputs) override {
        const size_t pv_bytes = (size_t)3 * height * width * sizeof(float);
        const size_t field_bytes = (size_t)patches * 3 * sizeof(float);

        // py-ref: negicc_station/src/tensorrt_intent_model.py:202-206 @ 2f7ee4a (input shapes must be set before output shapes)
        if (!context_->setInputShape("pixel_values", nvinfer1::Dims4{1, 3, height, width}))
            throw std::runtime_error("TensorRT engine rejects input shape (profile range)");
        context_->setTensorAddress("pixel_values", pv_.reserve(pv_bytes));
        check_cuda(cudaMemcpyAsync(pv_.ptr, pixel_values, pv_bytes, cudaMemcpyHostToDevice, stream_), "H2D");
        if (has_pool_weights_) {
            context_->setInputShape("pool_weights", nvinfer1::Dims3{1, patches, 3});
            context_->setTensorAddress("pool_weights", pw_.reserve(field_bytes));
            check_cuda(cudaMemcpyAsync(pw_.ptr, pool_weights, field_bytes, cudaMemcpyHostToDevice, stream_), "H2D");
        }
        const size_t tok_bytes = (size_t)patches * 768 * sizeof(float);
        if (has_tokens_) {
            context_->setTensorAddress("tokens", tok_.reserve(tok_bytes));
        } else {
            context_->setTensorAddress("mu", mu_.reserve(field_bytes));
            context_->setTensorAddress("sigma", sigma_.reserve(field_bytes));
        }
        if (has_pooled_) context_->setTensorAddress("pooled", pooled_.reserve(1152 * sizeof(float)));
        if (has_pooled_tone_) context_->setTensorAddress("pooled_tone", pooled_tone_.reserve(3 * 768 * sizeof(float)));

        // py-ref: negicc_station/src/tensorrt_intent_model.py:245 @ 2f7ee4a (enqueue execution)
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");

        // py-ref: negicc_station/src/tensorrt_intent_model.py:252-266 @ 2f7ee4a (D2H output copy on stream)
        if (outputs.tokens && has_tokens_) {
            check_cuda(cudaMemcpyAsync(outputs.tokens, tok_.ptr, tok_bytes, cudaMemcpyDeviceToHost, stream_), "D2H tokens");
            ++token_downloads_;
        }
        if (outputs.mu && !has_tokens_)
            check_cuda(cudaMemcpyAsync(outputs.mu, mu_.ptr, field_bytes, cudaMemcpyDeviceToHost, stream_), "D2H mu");
        if (outputs.sigma && !has_tokens_)
            check_cuda(cudaMemcpyAsync(outputs.sigma, sigma_.ptr, field_bytes, cudaMemcpyDeviceToHost, stream_), "D2H sigma");
        if (outputs.pooled && has_pooled_)
            check_cuda(cudaMemcpyAsync(outputs.pooled, pooled_.ptr, 1152 * sizeof(float), cudaMemcpyDeviceToHost, stream_), "D2H pooled");
        if (outputs.pooled_tone && has_pooled_tone_)
            check_cuda(cudaMemcpyAsync(outputs.pooled_tone, pooled_tone_.ptr, 3 * 768 * sizeof(float), cudaMemcpyDeviceToHost, stream_), "D2H pooled_tone");

        // py-ref: negicc_station/src/tensorrt_intent_model.py:246 @ 2f7ee4a (sync stream before host access)
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    size_t engine_bytes_ = 0;
    bool has_pool_weights_ = false, has_tokens_ = false, has_pooled_ = false, has_pooled_tone_ = false;
    int token_downloads_ = 0;
    DeviceBuffer pv_, pw_, tok_, mu_, sigma_, pooled_, pooled_tone_;
};

// The per-stock adapter engine: input "tokens" [1, N, 768], outputs "mu", "sigma" (and "log_var", bound but not
// copied back) [1, N, 3]. run_device reads the tokens from the backbone's device buffer; run copies host tokens in
// first (tests, and backbones whose tokens are on the host).
class TrtAdapterRunner : public DinoV3AdapterRunner {
public:
    TrtAdapterRunner(const void* engine_data, size_t engine_size) : engine_bytes_(engine_size) {
        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(engine_data, engine_size));
        if (!engine_) throw std::runtime_error("cannot deserialise the TensorRT adapter engine");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        bool tokens = false, mu = false, sigma = false;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (std::strcmp(name, "tokens") == 0) tokens = true;
            else if (std::strcmp(name, "mu") == 0) mu = true;
            else if (std::strcmp(name, "sigma") == 0) sigma = true;
            else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) extra_outputs_.push_back(name);
        }
        if (!tokens || !mu || !sigma)
            throw std::runtime_error("TensorRT adapter engine lacks tokens, mu or sigma (not an adapter.engine)");
        extra_.resize(extra_outputs_.size());
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~TrtAdapterRunner() override {
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool on_device() const override { return true; }

    size_t memory_bytes() const override {
        size_t n = engine_bytes_ + tok_in_.bytes + mu_.bytes + sigma_.bytes;
        for (const DeviceBuffer& b : extra_) n += b.bytes;
        return n;
    }

    void run(const float* tokens, int patches, float* mu, float* sigma) override {
        const size_t tok_bytes = (size_t)patches * 768 * sizeof(float);
        check_cuda(cudaMemcpyAsync(tok_in_.reserve(tok_bytes), tokens, tok_bytes, cudaMemcpyHostToDevice, stream_),
                   "H2D tokens");
        run_device(tok_in_.ptr, patches, mu, sigma);
    }

    void run_device(const void* device_tokens, int patches, float* mu, float* sigma) override {
        if (!device_tokens) throw std::runtime_error("TensorRT adapter: null device tokens");
        const size_t field_bytes = (size_t)patches * 3 * sizeof(float);
        if (!context_->setInputShape("tokens", nvinfer1::Dims3{1, patches, 768}))
            throw std::runtime_error("TensorRT adapter engine rejects the token shape (profile range)");
        context_->setTensorAddress("tokens", const_cast<void*>(device_tokens));
        context_->setTensorAddress("mu", mu_.reserve(field_bytes));
        context_->setTensorAddress("sigma", sigma_.reserve(field_bytes));
        for (size_t i = 0; i < extra_outputs_.size(); ++i)
            context_->setTensorAddress(extra_outputs_[i].c_str(), extra_[i].reserve(field_bytes));
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT adapter enqueueV3 failed");
        check_cuda(cudaMemcpyAsync(mu, mu_.ptr, field_bytes, cudaMemcpyDeviceToHost, stream_), "D2H mu");
        check_cuda(cudaMemcpyAsync(sigma, sigma_.ptr, field_bytes, cudaMemcpyDeviceToHost, stream_), "D2H sigma");
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    size_t engine_bytes_ = 0;
    std::vector<std::string> extra_outputs_;
    std::vector<DeviceBuffer> extra_;
    DeviceBuffer tok_in_, mu_, sigma_;
};

}  // namespace

std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& engine_path) {
    return std::make_unique<TrtRunner>(engine_path);
}

std::unique_ptr<DinoV3Runner> make_trt_runner(const void* engine_data, size_t engine_size) {
    return std::make_unique<TrtRunner>(engine_data, engine_size);
}

std::unique_ptr<DinoV3AdapterRunner> make_trt_adapter(const void* engine_data, size_t engine_size) {
    return std::make_unique<TrtAdapterRunner>(engine_data, engine_size);
}

bool cuda_device_available() {
    static bool available = false;
    static std::once_flag once;
    std::call_once(once, [] {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err == cudaSuccess && count > 0) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.canMapHostMemory) {
                available = true;
            } else {
                cudaGetLastError(); // Clear error state if prop check failed
                available = false;
            }
        } else {
            cudaGetLastError(); // Clear error state if no device
            available = false;
        }
    });
    return available;
}

#else  // !NEGICC_USE_CUDA

bool cuda_device_available() {
    return false;
}

std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& /*engine_path*/) {
    throw std::runtime_error("TensorRT runner is unavailable in CPU-only build (USE_CUDA=0)");
}

std::unique_ptr<DinoV3Runner> make_trt_runner(const void* /*engine_data*/, size_t /*engine_size*/) {
    throw std::runtime_error("TensorRT runner is unavailable in CPU-only build (USE_CUDA=0)");
}

std::unique_ptr<DinoV3AdapterRunner> make_trt_adapter(const void* /*engine_data*/, size_t /*engine_size*/) {
    throw std::runtime_error("TensorRT adapter is unavailable in CPU-only build (USE_CUDA=0)");
}

#endif  // NEGICC_USE_CUDA
