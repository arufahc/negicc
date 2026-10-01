// TensorRT runner for the fused DINOv3 intent model (USE_CUDA=1 builds only). Runs model.engine, the
// strict-FP32 engine `make engine` builds on the device; the same engine the Python 'gpu' intent backend runs.
//
// py-ref: negicc_station/src/tensorrt_intent_model.py:180-274 @ 2f7ee4a (TensorRTPhotographicIntentModel)
#include "dinov3_runner.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace negicc {

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
        m_engine_bytes = (size_t)f.tellg();
        std::vector<char> blob(m_engine_bytes);
        f.seekg(0);
        f.read(blob.data(), (std::streamsize)blob.size());

        m_runtime.reset(nvinfer1::createInferRuntime(g_logger));
        if (!m_runtime) throw std::runtime_error("createInferRuntime failed");
        m_engine.reset(m_runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!m_engine) throw std::runtime_error("cannot deserialise TensorRT engine " + engine_path);
        m_context.reset(m_engine->createExecutionContext());
        if (!m_context) throw std::runtime_error("createExecutionContext failed");

        // py-ref: negicc_station/src/tensorrt_intent_model.py:118-122 @ 2f7ee4a (detecting optional multi-head outputs)
        for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
            const char* name = m_engine->getIOTensorName(i);
            if (std::strcmp(name, "pool_weights") == 0) m_has_pool_weights = true;
            if (std::strcmp(name, "pooled") == 0) m_has_pooled = true;
            if (std::strcmp(name, "pooled_tone") == 0) m_has_pooled_tone = true;
        }
        check_cuda(cudaStreamCreate(&m_stream), "cudaStreamCreate");
    }

    ~TrtRunner() override {
        if (m_stream) cudaStreamDestroy(m_stream);
    }

    const char* name() const override { return "gpu (TensorRT)"; }
    bool has_pool_weights() const override { return m_has_pool_weights; }
    bool has_pooled() const override { return m_has_pooled; }
    bool has_pooled_tone() const override { return m_has_pooled_tone; }

    // py-ref: negicc_station/src/tensorrt_intent_model.py:153-169 @ 2f7ee4a (memory_bytes)
    size_t memory_bytes() const override {
        return m_engine_bytes + m_pv.bytes + m_pw.bytes + m_mu.bytes + m_sigma.bytes +
               m_pooled.bytes + m_pooled_tone.bytes;
    }

    // py-ref: negicc_station/src/tensorrt_intent_model.py:200-274 @ 2f7ee4a (predict / execution loop)
    void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
             const DinoV3RunnerOutputs& outputs) override {
        const size_t pv_bytes = (size_t)3 * height * width * sizeof(float);
        const size_t field_bytes = (size_t)patches * 3 * sizeof(float);

        // py-ref: negicc_station/src/tensorrt_intent_model.py:202-206 @ 2f7ee4a (input shapes must be set before output shapes)
        if (!m_context->setInputShape("pixel_values", nvinfer1::Dims4{1, 3, height, width}))
            throw std::runtime_error("TensorRT engine rejects input shape (profile range)");
        m_context->setTensorAddress("pixel_values", m_pv.reserve(pv_bytes));
        check_cuda(cudaMemcpyAsync(m_pv.ptr, pixel_values, pv_bytes, cudaMemcpyHostToDevice, m_stream), "H2D");
        if (m_has_pool_weights) {
            m_context->setInputShape("pool_weights", nvinfer1::Dims3{1, patches, 3});
            m_context->setTensorAddress("pool_weights", m_pw.reserve(field_bytes));
            check_cuda(cudaMemcpyAsync(m_pw.ptr, pool_weights, field_bytes, cudaMemcpyHostToDevice, m_stream), "H2D");
        }
        m_context->setTensorAddress("mu", m_mu.reserve(field_bytes));
        m_context->setTensorAddress("sigma", m_sigma.reserve(field_bytes));
        if (m_has_pooled) m_context->setTensorAddress("pooled", m_pooled.reserve(1152 * sizeof(float)));
        if (m_has_pooled_tone) m_context->setTensorAddress("pooled_tone", m_pooled_tone.reserve(3 * 768 * sizeof(float)));

        // py-ref: negicc_station/src/tensorrt_intent_model.py:245 @ 2f7ee4a (enqueue execution)
        if (!m_context->enqueueV3(m_stream)) throw std::runtime_error("TensorRT enqueueV3 failed");

        // py-ref: negicc_station/src/tensorrt_intent_model.py:252-266 @ 2f7ee4a (D2H output copy on stream)
        if (outputs.mu)
            check_cuda(cudaMemcpyAsync(outputs.mu, m_mu.ptr, field_bytes, cudaMemcpyDeviceToHost, m_stream), "D2H mu");
        if (outputs.sigma)
            check_cuda(cudaMemcpyAsync(outputs.sigma, m_sigma.ptr, field_bytes, cudaMemcpyDeviceToHost, m_stream), "D2H sigma");
        if (outputs.pooled && m_has_pooled)
            check_cuda(cudaMemcpyAsync(outputs.pooled, m_pooled.ptr, 1152 * sizeof(float), cudaMemcpyDeviceToHost, m_stream), "D2H pooled");
        if (outputs.pooled_tone && m_has_pooled_tone)
            check_cuda(cudaMemcpyAsync(outputs.pooled_tone, m_pooled_tone.ptr, 3 * 768 * sizeof(float), cudaMemcpyDeviceToHost, m_stream), "D2H pooled_tone");

        // py-ref: negicc_station/src/tensorrt_intent_model.py:246 @ 2f7ee4a (sync stream before host access)
        check_cuda(cudaStreamSynchronize(m_stream), "cudaStreamSynchronize");
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    cudaStream_t m_stream = nullptr;
    size_t m_engine_bytes = 0;
    bool m_has_pool_weights = false, m_has_pooled = false, m_has_pooled_tone = false;
    DeviceBuffer m_pv, m_pw, m_mu, m_sigma, m_pooled, m_pooled_tone;
};

}  // namespace

std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& engine_path) {
    return std::make_unique<TrtRunner>(engine_path);
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
                available = true;
            }
        } else {
            cudaGetLastError(); // Clear error state if no device
            available = false;
        }
    });
    return available;
}

}  // namespace negicc
