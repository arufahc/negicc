#ifndef DINOV3_ORT_RUNNER_H
#define DINOV3_ORT_RUNNER_H

#include "dinov3_runner.h"
#include <memory>
#include <string>

// Creates an ONNX Runtime CPU runner.
std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads);

#endif  // DINOV3_ORT_RUNNER_H
