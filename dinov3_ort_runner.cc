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
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        m_opts.SetIntraOpNumThreads(std::min(num_threads, hw));
        m_opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
        m_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), m_opts);

        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < m_session->GetInputCount(); ++i) {
            if (std::string(m_session->GetInputNameAllocated(i, alloc).get()) == "pool_weights")
                m_has_pool_weights = true;
        }
        for (size_t i = 0; i < m_session->GetOutputCount(); ++i) {
            std::string out_name = m_session->GetOutputNameAllocated(i, alloc).get();
            if (out_name == "pooled") m_has_pooled = true;
            if (out_name == "pooled_tone") m_has_pooled_tone = true;
        }
    }

    const char* name() const override { return "cpu (ONNX Runtime)"; }
    bool has_pool_weights() const override { return m_has_pool_weights; }
    bool has_pooled() const override { return m_has_pooled; }
    bool has_pooled_tone() const override { return m_has_pooled_tone; }

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
        if (m_has_pool_weights) {
            in_names.push_back("pool_weights");
            in_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(pool_weights),
                                                                 field_n, field_shape, 3));
        }

        std::vector<const char*> out_names;
        std::vector<Ort::Value> out_tensors;

        if (outputs.mu) {
            out_names.push_back("mu");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.mu, field_n, field_shape, 3));
        }
        if (outputs.sigma) {
            out_names.push_back("sigma");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.sigma, field_n, field_shape, 3));
        }
        if (outputs.pooled && m_has_pooled) {
            const int64_t pooled_shape[] = {1, 1152};
            out_names.push_back("pooled");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.pooled, 1152, pooled_shape, 2));
        }
        if (outputs.pooled_tone && m_has_pooled_tone) {
            const int64_t pt_shape[] = {1, 3, 768};
            out_names.push_back("pooled_tone");
            out_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, outputs.pooled_tone, 3 * 768, pt_shape, 3));
        }

        m_session->Run(Ort::RunOptions{nullptr}, in_names.data(), in_tensors.data(), in_tensors.size(),
                       out_names.data(), out_tensors.data(), out_names.size());
    }

private:
    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "negicc_dinov3"};
    Ort::SessionOptions m_opts;
    std::unique_ptr<Ort::Session> m_session;
    bool m_has_pool_weights = false;
    bool m_has_pooled = false;
    bool m_has_pooled_tone = false;
};

}  // namespace

std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads) {
    return std::make_unique<OrtCpuRunner>(model_path, num_threads);
}
