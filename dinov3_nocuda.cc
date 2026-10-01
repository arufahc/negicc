// CPU-only fallback stubs for TensorRT runner and CUDA device checks (USE_CUDA=0 builds).

#include "dinov3_runner.h"
#include <stdexcept>

namespace negicc {

bool cuda_device_available() {
    return false;
}

std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& /*engine_path*/) {
    throw std::runtime_error("TensorRT runner is unavailable in CPU-only build (USE_CUDA=0)");
}

}  // namespace negicc
