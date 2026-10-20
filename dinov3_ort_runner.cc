// ONNX Runtime CPU execution provider for DINOv3 photographic intent.
//
// py-ref: negicc_station/src/export_intent_engine.py:49-70 @ 2f7ee4a (model graph inputs/outputs)
// py-ref: negicc_station/src/bench_fused_vs_separate.py:163-209 @ 2f7ee4a (ORT execution session)

#include "dinov3_ort_runner.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

namespace {

class OrtCpuRunner : public DinoV3Runner {
public:
    OrtCpuRunner(const std::string& model_path, int num_threads) {
        set_options(num_threads);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts_);
        read_io();
    }

    // From an .onnx held in memory (the backbone built into neg_process).
    OrtCpuRunner(const void* onnx_data, size_t onnx_size, int num_threads) {
        set_options(num_threads);
        session_ = std::make_unique<Ort::Session>(env_, onnx_data, onnx_size, opts_);
        read_io();
    }

    void set_options(int num_threads) {
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        opts_.SetIntraOpNumThreads(std::min(num_threads, hw));
        opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    void read_io() {
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            if (std::string(session_->GetInputNameAllocated(i, alloc).get()) == "pool_weights")
                has_pool_weights_ = true;
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            std::string out_name = session_->GetOutputNameAllocated(i, alloc).get();
            if (out_name == "tokens") has_tokens_ = true;
            if (out_name == "pooled") has_pooled_ = true;
            if (out_name == "pooled_tone") has_pooled_tone_ = true;
        }
    }

    const char* name() const override { return "cpu (ONNX Runtime)"; }
    bool has_pool_weights() const override { return has_pool_weights_; }
    bool has_tokens() const override { return has_tokens_; }
    bool has_pooled() const override { return has_pooled_; }
    bool has_pooled_tone() const override { return has_pooled_tone_; }

    void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
             const DinoV3RunnerOutputs& outputs) override {
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t pv_shape[] = {1, 3, height, width};
        const int64_t field_shape[] = {1, patches, 3};
        const size_t field_n = (size_t)patches * 3;

        std::vector<const char*> in_names = {"pixel_values"};
        std::vector<Ort::Value> in_tensors;
        in_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(pixel_values),
                                                             (size_t)3 * height * width, pv_shape, 4));
        if (has_pool_weights_) {
            in_names.push_back("pool_weights");
            in_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(pool_weights),
                                                                 field_n, field_shape, 3));
        }

        std::vector<const char*> out_names;
        std::vector<Ort::Value> out_tensors;

        if (outputs.tokens && has_tokens_) {
            const int64_t tok_shape[] = {1, patches, 768};
            out_names.push_back("tokens");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.tokens, (size_t)patches * 768,
                                                                  tok_shape, 3));
        }
        if (outputs.mu && !has_tokens_) {
            out_names.push_back("mu");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.mu, field_n, field_shape, 3));
        }
        if (outputs.sigma && !has_tokens_) {
            out_names.push_back("sigma");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.sigma, field_n, field_shape, 3));
        }
        if (outputs.pooled && has_pooled_) {
            const int64_t pooled_shape[] = {1, 1152};
            out_names.push_back("pooled");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.pooled, 1152, pooled_shape, 2));
        }
        if (outputs.pooled_tone && has_pooled_tone_) {
            const int64_t pt_shape[] = {1, 3, 768};
            out_names.push_back("pooled_tone");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.pooled_tone, 3 * 768, pt_shape, 3));
        }

        session_->Run(Ort::RunOptions{nullptr}, in_names.data(), in_tensors.data(), in_tensors.size(),
                       out_names.data(), out_tensors.data(), out_names.size());
    }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "negicc_dinov3"};
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    bool has_pool_weights_ = false;
    bool has_tokens_ = false;
    bool has_pooled_ = false;
    bool has_pooled_tone_ = false;
};

class OrtAdapterRunner : public DinoV3AdapterRunner {
public:
    OrtAdapterRunner(const void* onnx_data, size_t onnx_size, int num_threads) : bytes_(onnx_size) {
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        opts_.SetIntraOpNumThreads(std::min(std::max(1, num_threads), hw));
        opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, onnx_data, onnx_size, opts_);
    }

    size_t memory_bytes() const override { return bytes_; }

    void run(const float* tokens, int patches, float* mu, float* sigma) override {
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t tok_shape[] = {1, patches, 768};
        const int64_t field_shape[] = {1, patches, 3};
        const size_t field_n = (size_t)patches * 3;
        const char* in_names[] = {"tokens"};
        Ort::Value in = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(tokens), (size_t)patches * 768,
                                                        tok_shape, 3);
        const char* out_names[] = {"mu", "sigma"};
        Ort::Value outs[] = {Ort::Value::CreateTensor<float>(mem_info, mu, field_n, field_shape, 3),
                             Ort::Value::CreateTensor<float>(mem_info, sigma, field_n, field_shape, 3)};
        session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, outs, 2);
    }

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "negicc_adapter"};
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    size_t bytes_ = 0;
};

}  // namespace

std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads) {
    return std::make_unique<OrtCpuRunner>(model_path, num_threads);
}

std::unique_ptr<DinoV3Runner> make_ort_runner(const void* onnx_data, size_t onnx_size, int num_threads) {
    return std::make_unique<OrtCpuRunner>(onnx_data, onnx_size, num_threads);
}

std::unique_ptr<DinoV3AdapterRunner> make_ort_adapter(const void* onnx_data, size_t onnx_size, int num_threads) {
    return std::make_unique<OrtAdapterRunner>(onnx_data, onnx_size, num_threads);
}
