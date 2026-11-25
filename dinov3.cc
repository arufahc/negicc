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


// The DINOv3 intent model and the Auto search (the DINOv3 part of negicc.h), in the pipeline's order: the
// preprocessing, DinoV3Engine with the ONNX Runtime CPU runners, the TensorRT runners (or their stubs in a USE_CUDA=0
// build), and the Auto solver.

#include "negicc.h"

#include <omp.h>
#include <pthread.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <onnxruntime_cxx_api.h>

// ====================================================================================================================
// Preprocessing
// PIL-exact and OpenCV-compatible preprocessing for DINOv3 photographic intent.
// Takes an 8-bit RGB preview; gives the patch-aligned network size, the resized image and the tone-bin weights of
// each patch.
// ====================================================================================================================

namespace {

// ---- Antialiased Bilinear Resampling with Support Widening ----------------------------------------------------
// 2-pass separable filter bit-exact with Pillow's Image.Resampling.BILINEAR: coefficients in double,
// normalised per output pixel, then fixed point with PRECISION_BITS; the horizontal pass writes an 8-bit
// intermediate image that the vertical pass reads.
constexpr int kFilterPrecisionBits = 32 - 8 - 2;

struct FilterCoeffs {
    int ksize = 0;
    std::vector<int> xmin, count;
    std::vector<int> kk;   // [out_size * ksize]
};

FilterCoeffs compute_antialiased_bilinear_coeffs(int in_size, int out_size) {
    FilterCoeffs c;
    const double scale = (double)in_size / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 1.0 * filterscale;   // bilinear filter support 1.0
    c.ksize = (int)std::ceil(support) * 2 + 1;
    c.xmin.resize(out_size);
    c.count.resize(out_size);
    c.kk.assign((size_t)out_size * c.ksize, 0);
    std::vector<double> k(c.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            double d = std::abs((x + xmin - center + 0.5) * ss);
            double w = d < 1.0 ? 1.0 - d : 0.0;
            k[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x) {
            if (ww != 0.0) k[x] /= ww;
            const double v = k[x] * (1 << kFilterPrecisionBits);
            c.kk[(size_t)xx * c.ksize + x] = v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v);
        }
        c.xmin[xx] = xmin;
        c.count[xx] = xmax;
    }
    return c;
}

inline uint8_t clip_u8(int in) {
    if (in >= (1 << kFilterPrecisionBits << 8)) return 255;
    if (in <= 0) return 0;
    return (uint8_t)(in >> kFilterPrecisionBits);
}

// ---- cv2.resize INTER_LINEAR (OpenCV modules/imgproc/src/resize.cpp) --------------------------------------------
// 11-bit fixed-point coefficients; the vertical pass is the SIMD form (VResizeLinearVec_32s8u).
constexpr int kCvCoefBits = 11;
constexpr int kCvCoefScale = 1 << kCvCoefBits;

inline short cv_coef(float v) {
    return (short)std::clamp((int)std::lrint(v * kCvCoefScale), -32768, 32767);
}

void cv_linear_coeffs(int in_size, int out_size, std::vector<int>& ofs, std::vector<short>& coef) {
    const double scale = 1.0 / ((double)out_size / in_size);
    ofs.resize(out_size);
    coef.resize((size_t)out_size * 2);
    for (int d = 0; d < out_size; ++d) {
        float f = (float)((d + 0.5) * scale - 0.5);
        int s = (int)std::floor(f);
        f -= s;
        if (s < 0) { f = 0.f; s = 0; }
        if (s >= in_size - 1) { f = 0.f; s = in_size - 1; }
        ofs[d] = s;
        coef[d * 2 + 0] = cv_coef(1.f - f);
        coef[d * 2 + 1] = cv_coef(f);
    }
}

}  // namespace

void compute_aspect_preserved_shape(int w, int h, int max_size, int patch_size,
                                    int& target_w, int& target_h, int& grid_w, int& grid_h) {
    const negicc::AspectPreservedShape s = negicc::compute_aspect_preserved_shape(w, h, max_size, patch_size);
    target_w = s.target_w;
    target_h = s.target_h;
    grid_w = s.grid_w;
    grid_h = s.grid_h;
}

void antialiased_bilinear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                                     uint8_t* dst, int dst_w, int dst_h) {
    const bool need_h = dst_w != src_w, need_v = dst_h != src_h;
    if (!need_h && !need_v) {
        for (int y = 0; y < src_h; ++y)
            std::copy_n(src + (size_t)y * src_stride, (size_t)src_w * 3, dst + (size_t)y * dst_w * 3);
        return;
    }

    // Horizontal pass into an 8-bit image [src_h, dst_w] (or the source itself when the width is unchanged)
    std::unique_ptr<uint8_t[]> tmp;  // not zero-filled: the horizontal pass writes all of it
    const uint8_t* mid = src;
    int mid_stride = src_stride;
    if (need_h) {
        const FilterCoeffs cx = compute_antialiased_bilinear_coeffs(src_w, dst_w);
        if (need_v) tmp.reset(new uint8_t[(size_t)src_h * dst_w * 3]);
        uint8_t* out = need_v ? tmp.get() : dst;
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < src_h; ++y) {
            const uint8_t* row = src + (size_t)y * src_stride;
            uint8_t* orow = out + (size_t)y * dst_w * 3;
            for (int x = 0; x < dst_w; ++x) {
                const int* k = &cx.kk[(size_t)x * cx.ksize];
                const uint8_t* p = row + (size_t)cx.xmin[x] * 3;
                int s0 = 1 << (kFilterPrecisionBits - 1), s1 = s0, s2 = s0;
                for (int i = 0; i < cx.count[x]; ++i) {
                    s0 += p[i * 3 + 0] * k[i];
                    s1 += p[i * 3 + 1] * k[i];
                    s2 += p[i * 3 + 2] * k[i];
                }
                orow[x * 3 + 0] = clip_u8(s0);
                orow[x * 3 + 1] = clip_u8(s1);
                orow[x * 3 + 2] = clip_u8(s2);
            }
        }
        if (!need_v) return;
        mid = tmp.get();
        mid_stride = dst_w * 3;
    }

    const FilterCoeffs cy = compute_antialiased_bilinear_coeffs(src_h, dst_h);
    // Each output row's sums are accumulated source row by source row: the same integer terms as a loop over the
    // output bytes first, added in another order (integer sums do not depend on it).
    #pragma omp parallel
    {
        std::vector<int> acc((size_t)dst_w * 3);
        #pragma omp for schedule(static)
        for (int y = 0; y < dst_h; ++y) {
            const int* k = &cy.kk[(size_t)y * cy.ksize];
            uint8_t* orow = dst + (size_t)y * dst_w * 3;
            std::fill(acc.begin(), acc.end(), 1 << (kFilterPrecisionBits - 1));
            for (int i = 0; i < cy.count[y]; ++i) {
                const uint8_t* r = mid + (size_t)(cy.xmin[y] + i) * mid_stride;
                const int ki = k[i];
                for (int x = 0; x < dst_w * 3; ++x) acc[x] += r[x] * ki;
            }
            for (int x = 0; x < dst_w * 3; ++x) orow[x] = clip_u8(acc[x]);
        }
    }
}

void cv_linear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int dst_w, int dst_h) {
    std::vector<int> xofs, yofs;
    std::vector<short> alpha, beta;
    cv_linear_coeffs(src_w, dst_w, xofs, alpha);
    cv_linear_coeffs(src_h, dst_h, yofs, beta);

    auto hresize = [&](const uint8_t* row, int* out) {
        for (int dx = 0; dx < dst_w; ++dx) {
            const int sx = xofs[dx];
            const int sx1 = std::min(sx + 1, src_w - 1);
            const int a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
            for (int c = 0; c < 3; ++c) out[dx * 3 + c] = row[sx * 3 + c] * a0 + row[sx1 * 3 + c] * a1;
        }
    };

    #pragma omp parallel for schedule(static)
    for (int dy = 0; dy < dst_h; ++dy) {
        std::vector<int> r0((size_t)dst_w * 3), r1((size_t)dst_w * 3);
        const int sy0 = yofs[dy], sy1 = std::min(sy0 + 1, src_h - 1);
        hresize(src + (size_t)sy0 * src_stride, r0.data());
        hresize(src + (size_t)sy1 * src_stride, r1.data());
        const int b0 = beta[dy * 2], b1 = beta[dy * 2 + 1];
        uint8_t* orow = dst + (size_t)dy * dst_w * 3;
        for (int x = 0; x < dst_w * 3; ++x) {
            const int v = ((((r0[x] >> 4) * b0) >> 16) + (((r1[x] >> 4) * b1) >> 16) + 2) >> 2;
            orow[x] = (uint8_t)std::clamp(v, 0, 255);
        }
    }
}

void compute_tone_weights(const uint8_t* img_u8, int grid_w, int grid_h, int patch_size, float* out_weights) {
    // Tone bins of intent_model.py: TONE_SHADOW_L 30, TONE_HIGHLIGHT_L 80, TONE_RAMP 10
    constexpr double kShadowL = 30.0, kHighlightL = 80.0, kRamp = 10.0;
    const int target_w = grid_w * patch_size;
    const double inv_pixels = 1.0 / (double)(patch_size * patch_size);

    #pragma omp parallel for schedule(static)
    for (int gy = 0; gy < grid_h; ++gy) {
        for (int gx = 0; gx < grid_w; ++gx) {
            long r_sum = 0, g_sum = 0, b_sum = 0;
            for (int py = 0; py < patch_size; ++py) {
                const uint8_t* row = img_u8 + ((size_t)(gy * patch_size + py) * target_w + gx * patch_size) * 3;
                for (int px = 0; px < patch_size; ++px) {
                    r_sum += row[px * 3 + 0];
                    g_sum += row[px * 3 + 1];
                    b_sum += row[px * 3 + 2];
                }
            }

            const double cr = negicc::srgb_decode(r_sum * inv_pixels / 255.0);
            const double cg = negicc::srgb_decode(g_sum * inv_pixels / 255.0);
            const double cb = negicc::srgb_decode(b_sum * inv_pixels / 255.0);

            // L* of the displayed sRGB against its own white, as negicc_core.h linear_srgb_to_lab (independent of
            // the PCS -> sRGB conversion): OpenCV's cvtColor Y row and CIE 15's exact break and slope (kLabEpsilon,
            // kLabKappa).
            const double Y = 0.212671 * cr + 0.715160 * cg + 0.072169 * cb;
            const double L = (Y > negicc::kLabEpsilon) ? (116.0 * std::cbrt(Y) - 16.0) : (negicc::kLabKappa * Y);

            const double w_sh = std::clamp((kShadowL + kRamp / 2 - L) / kRamp, 0.0, 1.0);
            const double w_hi = std::clamp((L - (kHighlightL - kRamp / 2)) / kRamp, 0.0, 1.0);
            const double w_mid = std::clamp(1.0 - w_sh - w_hi, 0.0, 1.0);

            const int p_idx = gy * grid_w + gx;
            out_weights[p_idx * 3 + 0] = (float)w_sh;
            out_weights[p_idx * 3 + 1] = (float)w_mid;
            out_weights[p_idx * 3 + 2] = (float)w_hi;
        }
    }
}

// ====================================================================================================================
// Engine
// DinoV3Engine and the ONNX Runtime CPU runners: takes an 8-bit sRGB preview, preprocesses it (above) and runs the
// backbone, its adapter and the token heads; gives DinoV3InferenceResult. The TensorRT runners (or their stubs in a
// USE_CUDA=0 build) follow.
// ====================================================================================================================

// --------------------------------------------------------------------------------------------------------------------
// Engine
// DinoV3Engine: native C++ photographic intent inference engine.
// --------------------------------------------------------------------------------------------------------------------

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

void DinoV3Engine::set_neutral_head(const void* data, size_t size) {
    if (!data) {
        neutral_head_.reset();
        return;
    }
    if (runner_ && !runner_->has_tokens()) {
        throw std::runtime_error("DinoV3Engine: the neutral head reads the backbone's tokens; a fused model has none");
    }
    auto head = std::make_unique<NeutralHead>();
    std::string err;
    if (!head->load(data, size, err)) throw std::runtime_error("DinoV3Engine: " + err);
    if (head->in_dim() != 768) throw std::runtime_error("DinoV3Engine: the neutral head does not take 768-D tokens");
    neutral_head_ = std::move(head);
}

void DinoV3AdapterRunner::run_device(const void*, int, float*, float*) {
    throw std::runtime_error("this intent adapter runs on the CPU and takes host tokens");
}

void DinoV3Engine::set_adapter(const void* onnx_data, size_t onnx_size, const std::string& name, int num_threads,
                               const void* engine_data, size_t engine_size) {
    adapter_path_.clear();
    if (runner_ && !runner_->has_tokens()) {  // a fused model carries its own adapter
        adapter_.reset();
        adapter_name_ = name;
        return;
    }
    const bool gpu = runner_ && std::strncmp(runner_->name(), "gpu", 3) == 0;
    std::unique_ptr<DinoV3AdapterRunner> a;
    if (gpu) {
        if (!engine_data || engine_size == 0) {
            throw std::runtime_error("DinoV3Engine: the backbone runs on TensorRT but intent adapter " + name +
                                     " has no TensorRT engine (run `make dinov3_engine` on this device); the adapter "
                                     "does not run on the CPU behind a GPU backbone");
        }
        a = make_trt_adapter(engine_data, engine_size);
    } else {
        if (!onnx_data || onnx_size == 0) {
            throw std::runtime_error("DinoV3Engine: intent adapter " + name +
                                     " has no adapter.onnx for the CPU backbone");
        }
        a = make_ort_adapter(onnx_data, onnx_size, num_threads);
    }
    adapter_ = std::move(a);
    adapter_name_ = name;
}

namespace {

// A model file read whole: a regular, non-empty file, every byte of it (a directory, an empty file or a short read
// throws, naming what).
std::vector<uint8_t> read_file(const std::string& path, const char* what) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        throw std::runtime_error(std::string("DinoV3Engine: ") + what + " " + path + " is not a file");
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("DinoV3Engine: cannot open ") + what + " " + path);
    const std::streamoff size = f.tellg();
    if (size <= 0) throw std::runtime_error(std::string("DinoV3Engine: ") + what + " " + path + " is empty");
    std::vector<uint8_t> bytes((size_t)size);
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(bytes.data()), (std::streamsize)bytes.size()))
        throw std::runtime_error(std::string("DinoV3Engine: cannot read all of ") + what + " " + path);
    return bytes;
}

}  // namespace

void DinoV3Engine::load_adapter_file(const std::string& path, int num_threads) {
    std::filesystem::path p(path);
    if (dir_exists(path)) p /= "adapter.onnx";
    std::filesystem::path onnx_p = p, engine_p = p;
    if (p.extension() == ".engine") onnx_p.replace_extension(".onnx");
    else engine_p.replace_extension(".engine");
    adapter_path_.clear();
    adapter_file_.clear();
    // A fused model carries its own adapter: no file is read.
    if (runner_ && !runner_->has_tokens()) {
        set_adapter(nullptr, 0, path, num_threads);
        return;
    }
    // Only the file the backbone's runtime takes is read: the engine behind a TensorRT backbone, the .onnx behind ONNX
    // Runtime (set_adapter).
    const bool gpu = runner_ && std::strncmp(runner_->name(), "gpu", 3) == 0;
    if (gpu) {
        if (!file_exists(engine_p.string())) {
            set_adapter(nullptr, 0, path, num_threads);   // throws: the engine the GPU backbone needs is missing
            return;
        }
        // An engine older than its .onnx holds the earlier weights.
        if (file_exists(onnx_p.string()) &&
            std::filesystem::last_write_time(engine_p) < std::filesystem::last_write_time(onnx_p)) {
            throw std::runtime_error("DinoV3Engine: intent adapter engine " + engine_p.string() + " is older than " +
                                     onnx_p.string() + ", so it holds earlier weights (run `make dinov3_engine` on "
                                     "this device to rebuild it)");
        }
        const std::vector<uint8_t> engine = read_file(engine_p.string(), "intent adapter engine");
        set_adapter(nullptr, 0, path, num_threads, engine.data(), engine.size());
        adapter_path_ = engine_p.string();
    } else {
        if (!file_exists(onnx_p.string()))
            throw std::runtime_error("DinoV3Engine: intent adapter " + path + " has no adapter.onnx for the CPU backbone");
        // ONNX Runtime copies what it needs while building the session, but keep the bytes alive with the session.
        adapter_file_ = read_file(onnx_p.string(), "intent adapter");
        set_adapter(adapter_file_.data(), adapter_file_.size(), path, num_threads);
        adapter_path_ = onnx_p.string();
    }
}

const char* DinoV3Engine::backend_name() const { return runner_ ? runner_->name() : "none"; }
const char* DinoV3Engine::backend_used() const {
    if (!runner_) return "none";
    return std::strncmp(runner_->name(), "gpu", 3) == 0 ? "gpu" : "cpu";
}
bool DinoV3Engine::has_pooled() const { return runner_ ? runner_->has_pooled() : false; }
bool DinoV3Engine::has_pooled_tone() const { return runner_ ? runner_->has_pooled_tone() : false; }
bool DinoV3Engine::has_pool_weights() const { return runner_ ? runner_->has_pool_weights() : false; }
size_t DinoV3Engine::memory_bytes() const {
    return (runner_ ? runner_->memory_bytes() : 0) + (adapter_ ? adapter_->memory_bytes() : 0);
}

void DinoV3Engine::load(const std::string& model_spec, const std::string& backend, int num_threads,
                        int max_size, int patch_size) {
    runner_.reset();
    adapter_.reset();
    adapter_name_.clear();
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
            (file_exists(join_path(model_spec, "backbone.onnx")) ||
             file_exists(join_path(model_spec, "backbone.engine")))
                ? "backbone" : "model";
        if (file_exists(join_path(model_spec, stem + ".onnx"))) onnx_path = join_path(model_spec, stem + ".onnx");
        if (file_exists(join_path(model_spec, stem + ".engine"))) engine_path = join_path(model_spec, stem + ".engine");
    } else if (ends_with(model_spec, ".engine")) {
        if (!file_exists(model_spec)) throw std::runtime_error("DinoV3Engine: TensorRT engine " + model_spec + " is not a file");
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
    adapter_.reset();
    adapter_name_.clear();
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

DinoV3InferenceResult DinoV3Engine::infer(const uint8_t* srgb_data, int width, int height, int stride) {
    if (!is_loaded()) {
        throw std::runtime_error("DinoV3Engine::infer called before model loaded.");
    }
    if (width <= 0 || height <= 0 || !srgb_data) {
        throw std::invalid_argument("DinoV3Engine::infer: invalid image dimensions or null data pointer");
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    DinoV3InferenceResult res;
    compute_aspect_preserved_shape(width, height, max_size_, patch_size_,
                                  res.target_w, res.target_h, res.grid_w, res.grid_h);
    res.num_patches = res.grid_w * res.grid_h;

    // 1. Antialiased bilinear resize to (target_w, target_h)
    std::vector<uint8_t> resized_u8((size_t)res.target_w * res.target_h * 3);
    antialiased_bilinear_resize_rgb(srgb_data, width, height, stride,
                                    resized_u8.data(), res.target_w, res.target_h);

    // 2. pixel_values [1, 3, target_h, target_w]: (u8 / 255 - mean) / std in float32
    std::vector<float> pixel_values((size_t)3 * res.target_h * res.target_w);
    // The ImageNet statistics DINOv3's LVD-1689M weights take (facebookresearch/dinov3 README).
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
    std::vector<float> pool_weights((size_t)res.num_patches * 3);
    compute_tone_weights(resized_u8.data(), res.grid_w, res.grid_h, patch_size_, pool_weights.data());

    // 4. Multi-head forward pass
    res.mu.resize((size_t)res.num_patches * 3);
    res.sigma.resize((size_t)res.num_patches * 3);

    DinoV3RunnerOutputs outs;
    std::vector<float> tokens;
    const bool device_adapter = runner_->has_tokens() && adapter_ && adapter_->on_device();
    if (runner_->has_tokens()) {
        if (!adapter_) throw std::runtime_error("DinoV3Engine::infer: the backbone model needs an adapter");
        // A TensorRT backbone with a TensorRT adapter: the tokens stay on the device, unless the neutral head or the
        // token spread reads them (one device-to-host copy per inference).
        if (!device_adapter || neutral_head_ || token_spread_) {
            tokens.resize((size_t)res.num_patches * 768);
            outs.tokens = tokens.data();
        }
    } else {
        outs.mu = res.mu.data();
        outs.sigma = res.sigma.data();
    }

    const bool want_ranker = runner_->has_pooled() || runner_->has_pooled_tone();
    if (want_ranker) {
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
    if (device_adapter) {
        adapter_->run_device(runner_->device_tokens(), res.num_patches, res.mu.data(), res.sigma.data());
    } else if (!tokens.empty()) {
        adapter_->run(tokens.data(), res.num_patches, res.mu.data(), res.sigma.data());
    }

    if (neutral_head_ && !tokens.empty()) {
        const auto tn0 = std::chrono::high_resolution_clock::now();
        res.neutral_p.resize((size_t)res.num_patches);
        neutral_head_->run(tokens.data(), res.num_patches, res.neutral_p.data());
        res.neutral_ms =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tn0).count();
    }

    if (token_spread_ && !tokens.empty() && res.num_patches > 1) {
        const int n = res.num_patches;
        res.token_sd.resize(768);
        // Every dimension's sums, patch by patch: each dimension adds its patches in order, in double, as a loop
        // over the dimensions first would.
        double mean[768], var[768];
        for (int d = 0; d < 768; ++d) mean[d] = var[d] = 0.0;
        for (int i = 0; i < n; ++i) {
            const float* t = &tokens[(size_t)i * 768];
            for (int d = 0; d < 768; ++d) mean[d] += t[d];
        }
        for (int d = 0; d < 768; ++d) mean[d] = mean[d] / n;
        for (int i = 0; i < n; ++i) {
            const float* t = &tokens[(size_t)i * 768];
            for (int d = 0; d < 768; ++d) {
                const double x = t[d] - mean[d];
                var[d] += x * x;
            }
        }
        for (int d = 0; d < 768; ++d) res.token_sd[d] = (float)std::sqrt(var[d] / (n - 1));
    }

    if (res.has_pooled_tone) {
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

// --------------------------------------------------------------------------------------------------------------------
// ONNX Runtime CPU runners
// ONNX Runtime CPU execution provider for DINOv3 photographic intent.
// --------------------------------------------------------------------------------------------------------------------

namespace {

// ONNX Runtime's environment, with the library's telemetry off. ONNX Runtime 1.30 runs telemetry by default: when
// its environment is created it writes a device id and a database under ~/.cache/Microsoft/DeveloperTools/ and opens
// a TLS connection to a telemetry host. It reads ORT_DISABLE_TELEMETRY then, so the variable is set here, before any
// environment exists; a value the caller set is kept.
Ort::Env make_ort_env(const char* log_id) {
    setenv("ORT_DISABLE_TELEMETRY", "1", 0);
    return Ort::Env(ORT_LOGGING_LEVEL_WARNING, log_id);
}

// ONNX Runtime's worker threads, started here (the sessions' custom thread functions) so that a thread that cannot be
// started ends the program with its reason. ONNX Runtime 1.30 does not survive a worker that fails to start: its pool
// throws with the workers already started still waiting for work, and then waits forever to join them. A process
// whose address space ran out while its pools started hung so, sleeping, instead of ending.
struct OrtWorker {
    OrtThreadWorkerFn fn;
    void* param;
    pthread_t thread;
};

void* run_ort_worker(void* p) {
    const OrtWorker* w = static_cast<const OrtWorker*>(p);
    w->fn(w->param);
    return nullptr;
}

OrtCustomThreadHandle start_ort_worker(void*, OrtThreadWorkerFn fn, void* param) {
    int err = ENOMEM;
    if (OrtWorker* w = static_cast<OrtWorker*>(std::malloc(sizeof(OrtWorker)))) {
        w->fn = fn;
        w->param = param;
        err = pthread_create(&w->thread, nullptr, run_ort_worker, w);
        if (err == 0) return reinterpret_cast<OrtCustomThreadHandle>(w);
        std::free(w);
    }
    // Ended at once: the workers already started cannot be stopped, so nothing that would wait for them may run.
    std::fprintf(stderr, "ERROR! cannot start a thread for ONNX Runtime: %s\n", std::strerror(err));
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(1);
}

void join_ort_worker(OrtCustomThreadHandle h) {
    OrtWorker* w = reinterpret_cast<OrtWorker*>(const_cast<OrtCustomHandleType*>(h));
    pthread_join(w->thread, nullptr);
    std::free(w);
}

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
        if (num_threads <= 0) num_threads = omp_get_max_threads();  // the program's thread count
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        opts_.SetIntraOpNumThreads(std::min(std::max(1, num_threads), hw));
        opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts_.SetCustomCreateThreadFn(start_ort_worker);
        opts_.SetCustomJoinThreadFn(join_ort_worker);
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
    Ort::Env env_ = make_ort_env("negicc_dinov3");
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
        if (num_threads <= 0) num_threads = omp_get_max_threads();  // the program's thread count
        const int hw = std::max(1, (int)std::thread::hardware_concurrency());
        opts_.SetIntraOpNumThreads(std::min(std::max(1, num_threads), hw));
        opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts_.SetCustomCreateThreadFn(start_ort_worker);
        opts_.SetCustomJoinThreadFn(join_ort_worker);
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
    Ort::Env env_ = make_ort_env("negicc_adapter");
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

// ====================================================================================================================
// TensorRT runners
// Take the preprocessed tensors (or the backbone's tokens on the device); give the same outputs as the ONNX Runtime
// runners above.
// ====================================================================================================================

// TensorRT runners (USE_CUDA=1 builds, which define NEGICC_USE_CUDA): the DINOv3 backbone (backbone.engine) or a
// fused intent model (model.engine), and the per-stock adapter (adapter.engine), all strict-FP32 engines
// `make dinov3_engine` builds on the device. A backbone keeps its tokens on the device; the adapter binds that buffer
// as its input, so only the intent field and the ranker features come back to the host.
//
// A USE_CUDA=0 build compiles the stubs at the end of this part instead: no CUDA device, and runners that throw.

#ifdef NEGICC_USE_CUDA

#include <mutex>

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

class TrtRunner : public DinoV3Runner {
public:
    explicit TrtRunner(const std::string& engine_path) {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(engine_path, ec))
            throw std::runtime_error("TensorRT engine " + engine_path + " is not a file");
        std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open TensorRT engine " + engine_path);
        const std::streamoff size = f.tellg();
        if (size <= 0) throw std::runtime_error("TensorRT engine " + engine_path + " is empty");
        std::vector<char> blob((size_t)size);
        f.seekg(0);
        if (!f.read(blob.data(), (std::streamsize)blob.size()))
            throw std::runtime_error("cannot read all of TensorRT engine " + engine_path);
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

    size_t memory_bytes() const override {
        return engine_bytes_ + pv_.bytes + pw_.bytes + tok_.bytes + mu_.bytes + sigma_.bytes +
               pooled_.bytes + pooled_tone_.bytes;
    }

    void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
             const DinoV3RunnerOutputs& outputs) override {
        const size_t pv_bytes = (size_t)3 * height * width * sizeof(float);
        const size_t field_bytes = (size_t)patches * 3 * sizeof(float);

        // Input shapes must be set before output shapes.
        if (!context_->setInputShape("pixel_values", nvinfer1::Dims4{1, 3, height, width}))
            throw std::runtime_error("TensorRT engine rejects input shape (profile range)");
        context_->setTensorAddress("pixel_values", pv_.reserve(pv_bytes));
        check_cuda(cudaMemcpyAsync(pv_.ptr, pixel_values, pv_bytes, cudaMemcpyHostToDevice, stream_), "H2D");
        if (has_pool_weights_) {
            if (!pool_weights) throw std::runtime_error("TensorRT engine takes pool_weights, and none were given");
            if (!context_->setInputShape("pool_weights", nvinfer1::Dims3{1, patches, 3}))
                throw std::runtime_error("TensorRT engine rejects the pool_weights shape (profile range)");
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

        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");

        if (outputs.tokens && has_tokens_) {
            check_cuda(cudaMemcpyAsync(outputs.tokens, tok_.ptr, tok_bytes, cudaMemcpyDeviceToHost, stream_),
                       "D2H tokens");
            ++token_downloads_;
        }
        if (outputs.mu && !has_tokens_)
            check_cuda(cudaMemcpyAsync(outputs.mu, mu_.ptr, field_bytes, cudaMemcpyDeviceToHost, stream_), "D2H mu");
        if (outputs.sigma && !has_tokens_)
            check_cuda(cudaMemcpyAsync(outputs.sigma, sigma_.ptr, field_bytes, cudaMemcpyDeviceToHost, stream_),
                       "D2H sigma");
        if (outputs.pooled && has_pooled_)
            check_cuda(cudaMemcpyAsync(outputs.pooled, pooled_.ptr, 1152 * sizeof(float), cudaMemcpyDeviceToHost,
                                       stream_), "D2H pooled");
        if (outputs.pooled_tone && has_pooled_tone_)
            check_cuda(cudaMemcpyAsync(outputs.pooled_tone, pooled_tone_.ptr, 3 * 768 * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_), "D2H pooled_tone");

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

// ====================================================================================================================
// Auto solver
// Takes the frame (an RgbView in camera RGB), its profile and matrix, and the engine (above); renders a preview at
// candidate (E, g, b), runs the engine on it and searches against the intent field; gives the (E, g, b) the field
// asks for, with the search's log (DinoSolveResult).
// ====================================================================================================================

// The fixed-point loop stops when |dE| + |dg| + |db| <= TOL_PARAM or the relative residual change <= TOL_REL_ERR.
static constexpr double TOL_PARAM = 0.02;
static constexpr double TOL_REL_ERR = 0.015;

negicc::PipelineConfig candidate_config(const SolverConfig& cfg, const float* merged, const IntentGains& cand,
                                        float m[9], std::vector<float> trc[3]) {
  if (!cfg.gain_curves) {
    negicc::candidate_matrix(merged, cand.e, cand.g, cand.b, m);
    return make_pipeline_config(*cfg.prof, cfg.knee, cfg.shoulder);
  }
  (*cfg.gain_curves)(cand.g, cand.b, trc);
  ProfileData prof = *cfg.prof;
  for (int c = 0; c < 3; ++c) {
    prof.in_trc[c] = trc[c].data();
    prof.in_trc_size[c] = (int)trc[c].size();
  }
  negicc::candidate_matrix(merged, cand.e, 1.0f, 1.0f, m);
  return make_pipeline_config(prof, cfg.knee, cfg.shoulder);
}

void intent_cells(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                  const IntentGains& cand, float* lab_out) {
  float m[9];
  std::vector<float> trc[3];
  const negicc::PipelineConfig pc = candidate_config(cfg, merged, cand, m, trc);
  for (int i = 0; i < n_px; ++i) {
    const negicc::Rgb lin = negicc::convert_pixel_linear(pc, m, grid_rgb16 + i * 3);
    const negicc::Lab lab = negicc::linear_srgb_to_lab(negicc::to_display(pc, lin));
    lab_out[i * 3] = lab.L;
    lab_out[i * 3 + 1] = lab.a;
    lab_out[i * 3 + 2] = lab.b;
  }
}

void intent_loss(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                 const IntentGains* cand, int n_cand, const float* mu, const float* sigma, IntentLoss* out) {
  const negicc::PipelineConfig pc_matrix = make_pipeline_config(*cfg.prof, cfg.knee, cfg.shoulder);
  #pragma omp parallel for schedule(dynamic)
  for (int c = 0; c < n_cand; ++c) {
    float m[9];
    std::vector<float> trc[3];
    negicc::PipelineConfig pc_curves;
    if (cfg.gain_curves) pc_curves = candidate_config(cfg, merged, cand[c], m, trc);
    else negicc::candidate_matrix(merged, cand[c].e, cand[c].g, cand[c].b, m);
    const negicc::PipelineConfig& pc = cfg.gain_curves ? pc_curves : pc_matrix;
    double sL = 0.0, sa = 0.0, sb = 0.0, clip = 0.0;
    for (int i = 0; i < n_px; ++i) {
      const uint16_t* px = grid_rgb16 + i * 3;
      // Scored in linear light, as the station's pipeline (the colour pipeline part of negicc_core.h): the clip rule
      // reads the pre-shoulder value and L*a*b* the displayed one (after the shoulder), the linear equivalents of the
      // encoded ones.
      const negicc::Rgb lin = negicc::convert_pixel_linear(pc, m, px);
      clip += negicc::linear_channel_white_clipped(lin.r) + negicc::linear_channel_white_clipped(lin.g) +
              negicc::linear_channel_white_clipped(lin.b);
      const negicc::Lab lab = negicc::linear_srgb_to_lab(negicc::to_display(pc, lin));
      const float dL = (lab.L - mu[i * 3]) / sigma[i * 3];
      const float da = (lab.a - mu[i * 3 + 1]) / sigma[i * 3 + 1];
      const float db = (lab.b - mu[i * 3 + 2]) / sigma[i * 3 + 2];
      sL += dL * dL;
      sa += da * da;
      sb += db * db;
    }
    out[c].L = (float)(sL / n_px);
    out[c].a = (float)(sa / n_px);
    out[c].b = (float)(sb / n_px);
    out[c].clip = (float)(clip / (3.0 * n_px) * 100.0);
  }
}

// Search state in double, as the reference's Python scalars; candidates are rounded to float only at intent_loss.
struct GainsD {
  double e, g, b;
};

static inline double compute_penalties(double e, float hi_clip, double e_center, double& pen_clip, double& pen_e) {
  const double clip_excess = std::max(0.0, (double)hi_clip - 2.5) / 2.0;
  pen_clip = 15.0 * (clip_excess * clip_excess);
  if (e_center > 0.0) {
    const double log2_ratio = std::log2(e / e_center);
    pen_e = 5.0 * log2_ratio * log2_ratio;
  } else {
    pen_e = 0.0;
  }
  return pen_clip + pen_e;
}

static inline double compute_penalties(const SolverConfig& cfg, double e, float hi_clip, double e_center,
                                       double& pen_clip, double& pen_e) {
  if (cfg.enc_e_min > 0.0f && cfg.enc_e_max > 0.0f && e > 0.0) {
    const double clip_excess = std::max(0.0, (double)hi_clip - 2.5) / 2.0;
    pen_clip = 15.0 * (clip_excess * clip_excess);
    const double d_hi = std::max(0.0, std::log2((double)cfg.enc_e_min / e));
    const double d_sh = std::max(0.0, std::log2(e / (double)cfg.enc_e_max));
    const double neg_slack = std::max(0.0, std::log2((double)cfg.enc_e_min / (double)cfg.enc_e_max));
    pen_e = 4.5 * d_hi * d_hi + 1.2 * d_sh * d_sh + 0.8 * neg_slack * neg_slack;
    return pen_clip + pen_e;
  }
  return compute_penalties(e, hi_clip, e_center, pen_clip, pen_e);
}

static inline IntentLoss eval_one(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                                  double e, double g, double b, const float* mu, const float* sigma) {
  const IntentGains cand{(float)e, (float)g, (float)b};
  IntentLoss r;
  intent_loss(cfg, merged, grid_rgb16, n_px, &cand, 1, mu, sigma, &r);
  return r;
}

static double optimize_impl(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                            int gw, int gh, const float* mu, const float* sigma,
                            const GainsD* initial, const double* e_bounds,
                            unsigned pin_mask, GainsD& out) {
  const int n_cells = gw * gh;
  // A pinned gain stays at its start value (cfg.pin_g and pin_b without initial, relative 1 by default), never at the
  // reference's free defaults.
  double curr_g = initial ? initial->g : ((pin_mask & PIN_G) ? (double)cfg.pin_g : 1.22);
  double curr_b = initial ? initial->b : ((pin_mask & PIN_B) ? (double)cfg.pin_b : 1.15);

  double e_min, e_max, e_center;
  std::vector<double> e_cands_d;

  if (pin_mask & PIN_E) {
    const double fixed_e = initial ? initial->e : (double)cfg.pin_e;
    e_cands_d.push_back(fixed_e);
    e_min = e_max = e_center = fixed_e;
  } else if (e_bounds != nullptr) {
    e_min = e_bounds[0];
    e_max = e_bounds[1];
    e_center = e_bounds[2];
    // np.linspace: start + i * step, endpoint pinned to stop
    const double lin_step = (e_max - e_min) / 12.0;
    for (int i = 0; i < 12; ++i) e_cands_d.push_back(e_min + i * lin_step);
    e_cands_d.push_back(e_max);
    if (initial) {
      for (double de : {-0.06, -0.03, 0.0, 0.03, 0.06}) {
        const double c = initial->e + de;
        if (c >= e_min && c <= e_max) e_cands_d.push_back(c);
      }
    }
    std::sort(e_cands_d.begin(), e_cands_d.end());
    e_cands_d.erase(std::unique(e_cands_d.begin(), e_cands_d.end()), e_cands_d.end());
  } else if (initial) {
    // E stays within [0.20, 3.0] as in the refinement: a start outside it is moved to its nearer end, so the offset 0
    // is always a candidate.
    e_min = 0.20;
    e_max = 3.0;
    e_center = std::clamp(initial->e, e_min, e_max);
    for (double d : {-0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15}) {
      const double c = e_center + d;
      if (c >= e_min && c <= e_max) e_cands_d.push_back(c);
    }
  } else {
    e_min = 0.40;
    e_max = 1.80;
    e_center = 1.0;
    e_cands_d = {0.40, 0.50, 0.60, 0.70, 0.80, 0.95, 1.10, 1.25, 1.40, 1.55, 1.70};
  }

  // 1. Exposure sweep (first minimum)
  std::vector<IntentGains> cands(e_cands_d.size());
  for (size_t i = 0; i < e_cands_d.size(); ++i) cands[i] = {(float)e_cands_d[i], (float)curr_g, (float)curr_b};
  std::vector<IntentLoss> losses(cands.size());
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());

  double p_clip, p_e;
  size_t best_k = 0;
  double best_score_e = 0.0;
  for (size_t i = 0; i < e_cands_d.size(); ++i) {
    const double score =
        (double)losses[i].L + compute_penalties(cfg, e_cands_d[i], losses[i].clip, e_center, p_clip, p_e);
    if (i == 0 || score < best_score_e) {
      best_score_e = score;
      best_k = i;
    }
  }
  double best_e = e_cands_d[best_k];

  // Exposure refinement (sequential, fixed base)
  if (!(pin_mask & PIN_E)) {
    const double step = (e_max > e_min) ? std::max(0.01, (e_max - e_min) / 40.0) : 0.02;
    const double base_e = best_e;
    for (double de : {-2.0 * step, -step, step, 2.0 * step}) {
      const double e_test = base_e + de;
      if (e_test < e_min || e_test > e_max) continue;
      const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, e_test, curr_g, curr_b, mu, sigma);
      const double s = (double)r.L + compute_penalties(cfg, e_test, r.clip, e_center, p_clip, p_e);
      if (s < best_score_e) {
        best_score_e = s;
        best_e = e_test;
      }
    }
  }

  // 2. Chromatic gains: g sweep, then b sweep
  // The highlight pullback's re-search sets the enclosure bounds and sweeps wider g and b grids.
  const bool pullback_search = (cfg.enc_e_min > 0.0f && cfg.enc_e_max > 0.0f);
  std::vector<double> g_cands_d, b_cands_d;
  if (pin_mask & PIN_G) {
    g_cands_d = {curr_g};
  } else if (initial) {
    for (double d : {-0.04, -0.02, 0.0, 0.02, 0.04}) g_cands_d.push_back(initial->g + d);
  } else if (pullback_search) {
    g_cands_d = {0.74, 0.80, 0.86, 0.92, 0.98, 1.04, 1.10, 1.16, 1.22, 1.28};
  } else {
    g_cands_d = {0.98, 1.04, 1.10, 1.16, 1.22, 1.28};
  }
  if (pin_mask & PIN_B) {
    b_cands_d = {curr_b};
  } else if (initial) {
    for (double d : {-0.06, -0.03, 0.0, 0.03, 0.06}) b_cands_d.push_back(initial->b + d);
  } else if (pullback_search) {
    b_cands_d = {0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35};
  } else {
    b_cands_d = {0.85, 0.95, 1.05, 1.15, 1.25, 1.35};
  }

  cands.resize(g_cands_d.size());
  losses.resize(g_cands_d.size());
  for (size_t i = 0; i < g_cands_d.size(); ++i) cands[i] = {(float)best_e, (float)g_cands_d[i], (float)curr_b};
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
  size_t k = 0;
  for (size_t i = 1; i < g_cands_d.size(); ++i) {
    if (losses[i].a < losses[k].a) k = i;
  }
  double best_g = g_cands_d[k];

  cands.resize(b_cands_d.size());
  losses.resize(b_cands_d.size());
  for (size_t i = 0; i < b_cands_d.size(); ++i) cands[i] = {(float)best_e, (float)best_g, (float)b_cands_d[i]};
  intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
  k = 0;
  for (size_t i = 1; i < b_cands_d.size(); ++i) {
    if (losses[i].b < losses[k].b) k = i;
  }
  double best_b = b_cands_d[k];

  // 3. g/b fine-tune (sequential, chained), starting from the sum at (best_e, best_g, best_b): the b sweep's own
  // record there, the number eval_one returns at that point.
  double best_gb = (double)losses[k].a + (double)losses[k].b;
  const double chain[4][2] = {{-0.02, 0.0}, {0.02, 0.0}, {0.0, -0.02}, {0.0, 0.02}};
  for (const auto& d : chain) {
    if ((pin_mask & PIN_G) && d[0] != 0.0) continue;
    if ((pin_mask & PIN_B) && d[1] != 0.0) continue;
    const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, best_e, best_g + d[0], best_b + d[1], mu, sigma);
    if ((double)r.a + (double)r.b < best_gb) {
      best_gb = (double)r.a + (double)r.b;
      best_g += d[0];
      best_b += d[1];
    }
  }

  // If the pullback's screening (no initial) shifted g or b from defaults, run a second coordinate-descent pass on
  // (e, g, b).
  if (pullback_search && !initial) {
    if (!(pin_mask & PIN_E)) {
      cands.resize(e_cands_d.size());
      losses.resize(e_cands_d.size());
      for (size_t i = 0; i < e_cands_d.size(); ++i) cands[i] = {(float)e_cands_d[i], (float)best_g, (float)best_b};
      intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
      best_k = 0;
      for (size_t i = 0; i < e_cands_d.size(); ++i) {
        const double score =
            (double)losses[i].L + compute_penalties(cfg, e_cands_d[i], losses[i].clip, e_center, p_clip, p_e);
        if (i == 0 || score < best_score_e) {
          best_score_e = score;
          best_k = i;
        }
      }
      best_e = e_cands_d[best_k];
      const double step = (e_max > e_min) ? std::max(0.01, (e_max - e_min) / 40.0) : 0.02;
      const double base_e = best_e;
      for (double de : {-2.0 * step, -step, step, 2.0 * step}) {
        const double e_test = base_e + de;
        if (e_test < e_min || e_test > e_max) continue;
        const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, e_test, best_g, best_b, mu, sigma);
        const double s = (double)r.L + compute_penalties(cfg, e_test, r.clip, e_center, p_clip, p_e);
        if (s < best_score_e) {
          best_score_e = s;
          best_e = e_test;
        }
      }
    }
    if (!(pin_mask & PIN_G)) {
      cands.resize(g_cands_d.size());
      losses.resize(g_cands_d.size());
      for (size_t i = 0; i < g_cands_d.size(); ++i) cands[i] = {(float)best_e, (float)g_cands_d[i], (float)best_b};
      intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
      k = 0;
      for (size_t i = 1; i < g_cands_d.size(); ++i) {
        if (losses[i].a < losses[k].a) k = i;
      }
      best_g = g_cands_d[k];
    }
    if (!(pin_mask & PIN_B)) {
      cands.resize(b_cands_d.size());
      losses.resize(b_cands_d.size());
      for (size_t i = 0; i < b_cands_d.size(); ++i) cands[i] = {(float)best_e, (float)best_g, (float)b_cands_d[i]};
      intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
      k = 0;
      for (size_t i = 1; i < b_cands_d.size(); ++i) {
        if (losses[i].b < losses[k].b) k = i;
      }
      best_b = b_cands_d[k];
    }
    const IntentLoss r0 = eval_one(cfg, merged, grid_rgb16, n_cells, best_e, best_g, best_b, mu, sigma);
    best_gb = (double)r0.a + (double)r0.b;
    for (const auto& d : chain) {
      if ((pin_mask & PIN_G) && d[0] != 0.0) continue;
      if ((pin_mask & PIN_B) && d[1] != 0.0) continue;
      const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, best_e, best_g + d[0], best_b + d[1], mu, sigma);
      if ((double)r.a + (double)r.b < best_gb) {
        best_gb = (double)r.a + (double)r.b;
        best_g += d[0];
        best_b += d[1];
      }
    }
  }

  // Re-align E at the final (best_g, best_b) so L* matches mu_L after chromatic gain updates.
  if (pullback_search && !(pin_mask & PIN_E) &&
      (std::fabs(best_g - curr_g) > 1e-6 || std::fabs(best_b - curr_b) > 1e-6)) {
    cands.resize(e_cands_d.size());
    losses.resize(e_cands_d.size());
    for (size_t i = 0; i < e_cands_d.size(); ++i) cands[i] = {(float)e_cands_d[i], (float)best_g, (float)best_b};
    intent_loss(cfg, merged, grid_rgb16, n_cells, cands.data(), (int)cands.size(), mu, sigma, losses.data());
    best_k = 0;
    for (size_t i = 0; i < e_cands_d.size(); ++i) {
      const double score =
          (double)losses[i].L + compute_penalties(cfg, e_cands_d[i], losses[i].clip, e_center, p_clip, p_e);
      if (i == 0 || score < best_score_e) {
        best_score_e = score;
        best_k = i;
      }
    }
    best_e = e_cands_d[best_k];
    const double step = (e_max > e_min) ? std::max(0.01, (e_max - e_min) / 40.0) : 0.02;
    const double base_e = best_e;
    for (double de : {-2.0 * step, -step, step, 2.0 * step}) {
      const double e_test = base_e + de;
      if (e_test < e_min || e_test > e_max) continue;
      const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, e_test, best_g, best_b, mu, sigma);
      const double s = (double)r.L + compute_penalties(cfg, e_test, r.clip, e_center, p_clip, p_e);
      if (s < best_score_e) {
        best_score_e = s;
        best_e = e_test;
      }
    }
  }

  // 4. Final balanced intent loss
  const IntentLoss r = eval_one(cfg, merged, grid_rgb16, n_cells, best_e, best_g, best_b, mu, sigma);
  const double loss_intent = ((double)r.L + (double)r.a + (double)r.b) / 3.0;
  const double pen = compute_penalties(cfg, best_e, r.clip, e_center, p_clip, p_e);
  out = {best_e, best_g, best_b};
  return loss_intent + pen;
}

float optimize_target_for_intent(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                                 int gw, int gh, const float* mu, const float* sigma,
                                 const IntentGains* initial, const float* e_bounds,
                                 unsigned pin_mask, IntentGains& out) {
  GainsD init_d, out_d;
  if (initial) init_d = {initial->e, initial->g, initial->b};
  double bounds_d[3];
  if (e_bounds) {
    for (int i = 0; i < 3; ++i) bounds_d[i] = e_bounds[i];
  }
  const double score = optimize_impl(cfg, merged, grid_rgb16, gw, gh, mu, sigma, initial ? &init_d : nullptr,
                                     e_bounds ? bounds_d : nullptr, pin_mask, out_d);
  out = {(float)out_d.e, (float)out_d.g, (float)out_d.b};
  return (float)score;
}

void trained_cells_lab(const uint8_t* preview_u8, int w, int h, int gw, int gh, int patch, float* lab_out) {
  const int W = gw * patch, H = gh * patch;
  float lut[256];
  for (int v = 0; v < 256; ++v) lut[v] = negicc::srgb_decode(v / 255.0f);
  const float sx = w / float(W), sy = h / float(H);
  // Each output column's and row's source span and area weights. They depend only on the column or the row, so they
  // are tabulated once, in the order the passes add them.
  struct Span { int s = 0, n = 0; size_t w0 = 0; };
  auto spans = [](int out_n, float scale, int in_n, std::vector<Span>& sp, std::vector<float>& wts) {
    sp.assign(out_n, Span{});
    wts.clear();
    for (int d = 0; d < out_n; ++d) {
      const float p0 = d * scale, p1 = (d + 1) * scale;
      const int ps = (int)std::floor(p0), pe = std::min(in_n, (int)std::ceil(p1));
      Span& q = sp[d];
      q.w0 = wts.size();
      bool first = true;
      for (int pp = ps; pp < pe; ++pp) {
        const float wt = std::min(p1, pp + 1.0f) - std::max(p0, (float)pp);
        if (wt <= 0.0f) continue;
        if (first) { q.s = pp; first = false; }
        wts.push_back(wt);
        ++q.n;
      }
      // A weight that is not positive can only lie at a span's ends, so the kept ones are contiguous from q.s.
    }
  };
  std::vector<Span> xsp, ysp;
  std::vector<float> xw, yw;
  spans(W, sx, w, xsp, xw);
  spans(H, sy, h, ysp, yw);
  std::vector<float> tmp((size_t)h * W * 3);  // horizontal pass, linear light
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = preview_u8 + (size_t)y * w * 3;
    for (int dx = 0; dx < W; ++dx) {
      const Span& q = xsp[dx];
      const float* wt = &xw[q.w0];
      const uint8_t* px = row + (size_t)q.s * 3;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, wsum = 0.0f;
      for (int i = 0; i < q.n; ++i) {
        a0 += lut[px[i * 3 + 0]] * wt[i];
        a1 += lut[px[i * 3 + 1]] * wt[i];
        a2 += lut[px[i * 3 + 2]] * wt[i];
        wsum += wt[i];
      }
      float* o = &tmp[((size_t)y * W + dx) * 3];
      o[0] = wsum > 0.0f ? a0 / wsum : 0.0f;
      o[1] = wsum > 0.0f ? a1 / wsum : 0.0f;
      o[2] = wsum > 0.0f ? a2 / wsum : 0.0f;
    }
  }
  // Vertical pass and CIELAB per pixel, on all threads. The per-patch sums are then added on one thread, in row-major
  // order, so they do not depend on the number of threads.
  // Measured: on a 192-thread machine this pass takes 2.0, 0.5, 0.4 and 5.5 ms at 8, 32, 96 and 192 threads, and on a
  // Jetson Orin Nano 37, 27, 16 and 11 ms at 1, 2, 4 and 6. It takes the program's thread count, as every region does
  // (AGENTS.md, "Parallel regions").
  std::vector<float> labpix((size_t)H * W * 3);
  #pragma omp parallel for schedule(static)
  for (int dy = 0; dy < H; ++dy) {
    const Span& q = ysp[dy];
    const float* wt = &yw[q.w0];
    float wsum = 0.0f;
    for (int i = 0; i < q.n; ++i) wsum += wt[i];
    const float inv = wsum > 0.0f ? 1.0f / wsum : 0.0f;
    for (int dx = 0; dx < W; ++dx) {
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
      for (int i = 0; i < q.n; ++i) {
        const float* p = &tmp[((size_t)(q.s + i) * W + dx) * 3];
        a0 += p[0] * wt[i];
        a1 += p[1] * wt[i];
        a2 += p[2] * wt[i];
      }
      const negicc::Lab c = negicc::linear_srgb_to_lab({a0 * inv, a1 * inv, a2 * inv});
      float* o = &labpix[((size_t)dy * W + dx) * 3];
      o[0] = c.L;
      o[1] = c.a;
      o[2] = c.b;
    }
  }
  std::vector<double> acc((size_t)gw * gh * 3, 0.0);
  for (int dy = 0; dy < H; ++dy) {
    double* arow = &acc[(size_t)(dy / patch) * gw * 3];
    const float* o = &labpix[(size_t)dy * W * 3];
    for (int dx = 0; dx < W; ++dx, o += 3) {
      double* a = arow + (size_t)(dx / patch) * 3;
      a[0] += o[0];
      a[1] += o[1];
      a[2] += o[2];
    }
  }
  const double inv_n = 1.0 / ((double)patch * patch);
  for (size_t i = 0; i < acc.size(); ++i) lab_out[i] = (float)(acc[i] * inv_n);
}

float self_consistency_residual(const uint8_t* preview_u8, int w, int h, int gw, int gh, int patch,
                                const float* mu, const float* sigma) {
  const int n = gw * gh;
  std::vector<float> lab((size_t)n * 3);
  trained_cells_lab(preview_u8, w, h, gw, gh, patch, lab.data());
  return residual_from_cells(lab.data(), n, mu, sigma);
}

float residual_from_cells(const float* lab, int n, const float* mu, const float* sigma) {
  double sum_res = 0.0;
  for (int i = 0; i < n; ++i) {
    const float dL = (lab[i * 3 + 0] - mu[i * 3 + 0]) / sigma[i * 3 + 0];
    const float da = (lab[i * 3 + 1] - mu[i * 3 + 1]) / sigma[i * 3 + 1];
    const float db = (lab[i * 3 + 2] - mu[i * 3 + 2]) / sigma[i * 3 + 2];
    sum_res += (dL * dL + da * da + db * db) / 3.0;
  }
  return (float)(sum_res / n);
}

float shoulder_x_hi_from_hist(const uint64_t* hist, float* q995_out) {
  const float q = (float)negicc::max_hist_quantile(hist, 0.995);
  if (q995_out) *q995_out = q;
  return negicc::shoulder_x_hi_anchored(q);
}

static void render_crop_preview(const SolverConfig& cfg, const float* merged_norm,
                                const negicc::RgbView& view,
                                float e, float g, float b, uint8_t* out_u8) {
  float m[9];
  negicc::candidate_matrix(merged_norm, e, g, b, m);
  negicc::RenderArgs ra{};
  ra.in_view = &view;
  ra.width = view.width;
  ra.height = view.height;
  ra.matrix = m;
  ra.cfg = make_pipeline_config(*cfg.prof, cfg.knee, cfg.shoulder);
  ra.out_type = 1;
  ra.out = out_u8;
  negicc::cpu_render(ra);
}

// Inference with the shape and sigma checks the loss relies on (no sigma floor: the reference divides by raw sigma).
static bool infer_checked(DinoV3Engine& eng, const uint8_t* preview_u8, int w, int h, int gw, int gh,
                          DinoV3InferenceResult& res, DinoSolveResult& out) {
  const double t0 = omp_get_wtime();
  try {
    res = eng.infer(preview_u8, w, h, w * 3);
  } catch (const std::exception& e) {
    fprintf(stderr, "ERROR! DINOv3 inference failed: %s\n", e.what());
    return false;
  }
  out.timing_infer_ms += (omp_get_wtime() - t0) * 1000.0;
  out.total_inferences++;
  const size_t n = (size_t)gw * gh * 3;
  if (res.grid_w != gw || res.grid_h != gh || res.mu.size() != n || res.sigma.size() != n) {
    fprintf(stderr, "ERROR! DINOv3 intent field shape %dx%d (%zu values) does not match grid %dx%d.\n",
            res.grid_w, res.grid_h, res.mu.size(), gw, gh);
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!(res.sigma[i] > 0.0f) || !std::isfinite(res.sigma[i]) || !std::isfinite(res.mu[i])) {
      fprintf(stderr, "ERROR! DINOv3 intent field has non-positive or non-finite sigma/mu at %zu.\n", i);
      return false;
    }
  }
  return true;
}

static DinoIterationLog make_log(char tag, int step, const GainsD& p, float residual, const IntentLoss& l,
                                 double e_center) {
  DinoIterationLog log;
  log.tag = tag;
  log.step = step;
  log.e = (float)p.e;
  log.g = (float)p.g;
  log.b = (float)p.b;
  log.residual = residual;
  log.loss_L = l.L;
  log.loss_a = l.a;
  log.loss_b = l.b;
  log.clip_pct = l.clip;
  double p_clip, p_e;
  const double pen = compute_penalties(p.e, l.clip, e_center, p_clip, p_e);
  log.penalty_clip = (float)p_clip;
  log.penalty_anchor = (float)p_e;
  log.objective = (float)(((double)l.L + (double)l.a + (double)l.b) / 3.0 + pen);
  return log;
}

bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const uint16_t (*image)[4], int w, int h,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out) {
  if (!image || w <= 0 || h <= 0) return false;
  return solve_dinov3_intent(eng, cfg, user_merged,
                             negicc::RgbView::from_interleaved_rgb(&image[0][0], w, h, 4, (ptrdiff_t)w * 4),
                             user_gains, pin_mask, max_iters, out);
}

bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const negicc::RgbView& view,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out) {
  const int w = view.width, h = view.height;
  if (!view.valid() || w <= 0 || h <= 0 || !cfg.prof || !user_merged) return false;

  const double t0_total = omp_get_wtime();

  int target_w = 0, target_h = 0, grid_w = 0, grid_h = 0;
  compute_aspect_preserved_shape(w, h, eng.max_size(), eng.patch_size(), target_w, target_h, grid_w, grid_h);
  out.grid_w = grid_w;
  out.grid_h = grid_h;
  out.target_w = target_w;
  out.target_h = target_h;

  const int n_cells = grid_w * grid_h;
  if (n_cells <= 0) return false;
  std::vector<uint16_t> grid_rgb16((size_t)n_cells * 3);

  const double t0_views = omp_get_wtime();
  negicc::resample_view_rgb16(view, grid_rgb16.data(), grid_w, grid_h);
  out.timing_views_ms += (omp_get_wtime() - t0_views) * 1000.0;

  // Bootstrap gains (reference (e_c, 1.10, 1.15)); a pinned gain stays at relative 1.
  const double boot_g = (pin_mask & PIN_G) ? 1.0 : 1.10;
  const double boot_b = (pin_mask & PIN_B) ? 1.0 : 1.15;

  // 1. Exposure anchor E_c: median L* of the grid render = 50, bisection on log2 E over [-4, 4].
  double e_center = 1.0;
  if (!(pin_mask & PIN_E)) {
    const double lo0 = -4.0, hi0 = 4.0;
    double log2_lo = lo0, log2_hi = hi0;
    std::vector<float> l_vals(n_cells);
    // The anchor is the median L* of the unshouldered render: the shoulder never touches the grey axis it meters.
    const negicc::PipelineConfig anchor_pc = make_pipeline_config(*cfg.prof, cfg.knee, 1.0f);
    for (int iter = 0; iter < 16; ++iter) {
      const double mid = 0.5 * (log2_lo + log2_hi);
      float m[9];
      negicc::candidate_matrix(user_merged, (float)std::pow(2.0, mid), (float)boot_g, (float)boot_b, m);
      for (int i = 0; i < n_cells; ++i) {
        const Rgb v = negicc::convert_pixel(anchor_pc, m, grid_rgb16.data() + (size_t)i * 3);
        l_vals[i] = negicc::srgb_to_lab(v).L;
      }
      std::nth_element(l_vals.begin(), l_vals.begin() + n_cells / 2, l_vals.end());
      // Negative: more transmittance (larger E) renders a darker positive, so median L* falls with E.
      if (l_vals[n_cells / 2] < 50.0f) {
        log2_hi = mid;
      } else {
        log2_lo = mid;
      }
    }
    out.anchor_clamped = (log2_lo == lo0 || log2_hi == hi0);
    e_center = std::pow(2.0, 0.5 * (log2_lo + log2_hi));
  }
  // The centre in the caller's E (user_merged carries user_gains.e): with a pinned E that is the given E, not the
  // relative 1 the search runs at.
  out.e_center = (float)((double)user_gains.e * e_center);

  // Search in E_c-normalized units: all rows scaled once by E_c.
  float merged_norm[9];
  for (int i = 0; i < 9; ++i) merged_norm[i] = user_merged[i] * (float)e_center;

  const double e_bounds[3] = {std::pow(2.0, -1.5), std::pow(2.0, 1.5), 1.0};
  // The range the search moves E in, as ratios to e_center: with E pinned it holds E at e_center (optimize_impl), so
  // [1, 1].
  out.bounds_lo = (pin_mask & PIN_E) ? 1.0f : (float)e_bounds[0];
  out.bounds_hi = (pin_mask & PIN_E) ? 1.0f : (float)e_bounds[1];

  // TODO: `preview_u8` is allocated at full `view.width x view.height` (e.g. 61 MP = 183 MB when `solve_dinov3_intent`
  // is called on the full-size `scan.rgb3`) and downsampled on every iteration inside `eng.infer`. Pass a half-size
  // view (or pre-downsample to `target_w x target_h` once before the loop) after updating single-target solver
  // baselines.
  std::vector<uint8_t> preview_u8((size_t)w * h * 3);
  DinoV3InferenceResult inf;

  // All pinned: no search; one render + inference for the intent field at the user's parameters.
  if (pin_mask == (PIN_E | PIN_G | PIN_B)) {
    const double t0_r = omp_get_wtime();
    render_crop_preview(cfg, merged_norm, view, 1.0f, 1.0f, 1.0f, preview_u8.data());
    out.timing_render_ms += (omp_get_wtime() - t0_r) * 1000.0;
    if (!infer_checked(eng, preview_u8.data(), w, h, grid_w, grid_h, inf, out)) return false;

    out.status = "PINNED";
    out.best_iteration = 0;
    out.best_residual = self_consistency_residual(preview_u8.data(), w, h, grid_w, grid_h, eng.patch_size(),
                                                  inf.mu.data(), inf.sigma.data());
    out.solved_relative = {1.0f, 1.0f, 1.0f};
    out.solved_absolute = user_gains;
    out.best_mu = std::move(inf.mu);
    out.best_sigma = std::move(inf.sigma);
    out.tone_mass = std::move(inf.tone_mass);

    out.timing_total_ms = (omp_get_wtime() - t0_total) * 1000.0;
    return true;
  }

  // 2. Bootstrap: render at (1, boot_g, boot_b), infer, search with initial = null.
  const GainsD boot{1.0, boot_g, boot_b};
  const double t0_r_boot = omp_get_wtime();
  render_crop_preview(cfg, merged_norm, view, (float)boot.e, (float)boot.g, (float)boot.b, preview_u8.data());
  out.timing_render_ms += (omp_get_wtime() - t0_r_boot) * 1000.0;
  if (!infer_checked(eng, preview_u8.data(), w, h, grid_w, grid_h, inf, out)) return false;

  const double anchor_center = (pin_mask & PIN_E) ? 0.0 : 1.0;  // pinned E drops P_anchor
  out.iterations.push_back(make_log('b', -1, boot, -1.0f,
                                    eval_one(cfg, merged_norm, grid_rgb16.data(), n_cells, boot.e, boot.g, boot.b,
                                             inf.mu.data(), inf.sigma.data()),
                                    anchor_center));

  const double t0_search_boot = omp_get_wtime();
  GainsD curr;
  optimize_impl(cfg, merged_norm, grid_rgb16.data(), grid_w, grid_h, inf.mu.data(), inf.sigma.data(),
                nullptr, e_bounds, pin_mask, curr);
  out.timing_search_ms += (omp_get_wtime() - t0_search_boot) * 1000.0;

  // 3. Fixed-point loop (reference fixed_point).
  double prev_err = HUGE_VAL;
  double best_err = HUGE_VAL;
  GainsD best_params = curr;
  DinoV3InferenceResult best_inf;
  int best_it = -1;
  std::string status = "MAX_ITERS";
  const int loop_limit = std::max(1, max_iters);

  for (int it = 0; it < loop_limit; ++it) {
    const double t0_r = omp_get_wtime();
    render_crop_preview(cfg, merged_norm, view, (float)curr.e, (float)curr.g, (float)curr.b,
                        preview_u8.data());
    out.timing_render_ms += (omp_get_wtime() - t0_r) * 1000.0;
    if (!infer_checked(eng, preview_u8.data(), w, h, grid_w, grid_h, inf, out)) return false;

    const double t0_resid = omp_get_wtime();
    const double err = self_consistency_residual(preview_u8.data(), w, h, grid_w, grid_h, eng.patch_size(),
                                                 inf.mu.data(), inf.sigma.data());
    out.timing_residual_ms += (omp_get_wtime() - t0_resid) * 1000.0;

    const IntentLoss it_loss = eval_one(cfg, merged_norm, grid_rgb16.data(), n_cells, curr.e, curr.g, curr.b,
                                        inf.mu.data(), inf.sigma.data());
    DinoIterationLog it_log = make_log((char)('0' + it % 10), it, curr, (float)err, it_loss, anchor_center);

    if (err < best_err) {
      best_err = err;
      best_params = curr;
      best_inf = inf;
      best_it = it;
    }

    // Diverged: checked before the search, as the reference.
    if (it > 0 && err > prev_err * 1.25 && (err - prev_err) > 1.5) {
      status = "DIVERGED";
      out.iterations.push_back(it_log);
      break;
    }

    const double t0_s = omp_get_wtime();
    GainsD next;
    optimize_impl(cfg, merged_norm, grid_rgb16.data(), grid_w, grid_h, inf.mu.data(), inf.sigma.data(),
                  &curr, e_bounds, pin_mask, next);
    out.timing_search_ms += (omp_get_wtime() - t0_s) * 1000.0;

    const double dp = std::abs(next.e - curr.e) + std::abs(next.g - curr.g) + std::abs(next.b - curr.b);
    const double rel = (it > 0) ? std::abs(err - prev_err) / std::max(1e-4, prev_err) : 1.0;
    it_log.dp = (float)dp;
    it_log.rel = (float)rel;
    out.iterations.push_back(it_log);

    curr = next;
    prev_err = err;
    if (it > 0 && (dp <= TOL_PARAM || rel <= TOL_REL_ERR)) {
      status = "CONVERGED";
      break;
    }
  }

  if (best_it < 0) {
    fprintf(stderr, "ERROR! DINOv3 solver produced no finite residual.\n");
    return false;
  }

  out.status = status;
  out.best_iteration = best_it;
  out.best_residual = (float)best_err;
  out.solved_relative = {(float)(e_center * best_params.e), (float)best_params.g, (float)best_params.b};
  out.solved_absolute = {(float)((double)user_gains.e * e_center * best_params.e),
                         (float)((double)user_gains.g * best_params.g),
                         (float)((double)user_gains.b * best_params.b)};
  out.best_mu = std::move(best_inf.mu);
  out.best_sigma = std::move(best_inf.sigma);
  out.tone_mass = std::move(best_inf.tone_mass);

  out.timing_total_ms = (omp_get_wtime() - t0_total) * 1000.0;
  return true;
}
