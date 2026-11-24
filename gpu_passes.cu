// The bundle search's two passes over the frame on a CUDA device (USE_CUDA=1 builds): the preview render, with or
// without its statistics, and the highlight clip pass. Each kernel is the shared per-pixel pipeline of
// color_pipeline.h plus the epilogue of its CPU function in rgb_view.cc (cpu_render, cpu_highlight_clip), built with
// --fmad=false so the device rounds the stages as the host compiler does (-ffp-contract=off). What still differs is
// the last place of the transcendentals (powf, tanhf, cbrtf: libdevice against glibc), so a GPU pass is not bit for
// bit the CPU's; neg_process --check_renderer measures the difference (multi_target_optimizer.h CompareRenderer).
//
// Reductions are deterministic: each block writes its partial sums to its own slot and the host adds the slots in a
// fixed order; histograms use integer atomics, which are exact.
//
// The callers (GpuRenderer) hand device pointers: the input (in, or in_view's channel pointers), the tables in cfg,
// and a render's out. A clip pass's matrices, out and max_hist, and a render's stats, are host memory.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgb_view.h"

namespace negicc {

#define NEGICC_CUDA_CHECK(expr)                                                                                   \
    do {                                                                                                          \
        cudaError_t _e = (expr);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
            throw negicc::GpuError(std::string("CUDA error in ") + #expr + ": " + cudaGetErrorString(_e));       \
    } while (0)

namespace {

// Grow-only device scratch, one per purpose, under one mutex: the passes are serial.
struct Scratch {
    void* ptr = nullptr;
    size_t size = 0;
    void* get(size_t bytes) {
        if (bytes > size) {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
            size = 0;
            NEGICC_CUDA_CHECK(cudaMalloc(&ptr, bytes));
            size = bytes;
        }
        return ptr;
    }
};

std::mutex g_op_mutex;
Scratch g_matrices, g_partials, g_hist, g_max_hist;

struct Mat9 {
    float m[9];
};

constexpr int kThreads = 256;
constexpr int kStatFields = 13;  // white, black, mid_a, mid_b, mid_n, hi_a, hi_b, hi_n, sh_a, sh_b, sh_n, all_a, all_b
constexpr int kRgbHistBins = 3 * 256;
constexpr int kRenderHistBins = kRgbHistBins + kLabBins;
constexpr int kHighlightBins = kLabBins + 4;  // hist_L + exceedance R, G, B + white clip

// The input of a pass: interleaved 16-bit RGB at base + y * stride + x * pitch (uint16_t elements). A contiguous
// image has pitch 3 and stride 3 * width; a crop of a bound frame keeps the frame's pitch and stride.
struct InPix {
    const uint16_t* base;
    long long pitch, stride;
};
__device__ __forceinline__ const uint16_t* in_at(const InPix& p, long long x, long long y) {
    return p.base + y * p.stride + x * p.pitch;
}

// As the CPU passes read theirs: in_view when it is a valid view, else in (contiguous).
InPix pass_input(const uint16_t* in, const RgbView* view, int width) {
    if (!view || !view->valid()) {
        if (!in) throw GpuError("a GPU pass without an input");
        return InPix{in, 3, 3LL * width};
    }
    if (view->g2_ptr || view->flat.gain || view->g_ptr != view->r_ptr + 1 || view->b_ptr != view->r_ptr + 2)
        throw GpuError("the GPU passes read an interleaved RGB view without a flat field");
    return InPix{view->r_ptr, (long long)view->pitch, (long long)view->stride};
}

template <typename OutT>
__device__ __forceinline__ void store_pixel(OutT* out, long long i, Rgb v);
template <>
__device__ __forceinline__ void store_pixel<uint16_t>(uint16_t* out, long long i, Rgb v) {
    out[i * 3] = quantize_u16(v.r); out[i * 3 + 1] = quantize_u16(v.g); out[i * 3 + 2] = quantize_u16(v.b);
}
template <>
__device__ __forceinline__ void store_pixel<uint8_t>(uint8_t* out, long long i, Rgb v) {
    out[i * 3] = quantize_u8(v.r); out[i * 3 + 1] = quantize_u8(v.g); out[i * 3 + 2] = quantize_u8(v.b);
}
template <>
__device__ __forceinline__ void store_pixel<float>(float* out, long long i, Rgb v) {
    out[i * 3] = v.r; out[i * 3 + 1] = v.g; out[i * 3 + 2] = v.b;
}

// The display value through cfg's tone stage as convert_pixel applies it (an sRGB output with a table).
__device__ __forceinline__ Rgb through_tone(const PipelineConfig& cfg, Rgb v) {
    return (cfg.encoding == ENCODE_SRGB && cfg.tone.lut && cfg.tone.n > 0) ? apply_tone(cfg.tone, v) : v;
}

// Block-wide sum of NV doubles per thread; thread 0 writes the result to out[0..NV).
template <int NV>
__device__ void block_sum(double (&v)[NV], double* out) {
    __shared__ double sh[NV][kThreads];
    for (int k = 0; k < NV; ++k) sh[k][threadIdx.x] = v[k];
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int k = 0; k < NV; ++k) sh[k][threadIdx.x] += sh[k][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < NV; ++k) out[k] = sh[k][0];
}

// out is not __restrict__: a render over the frame's own copy (GpuRenderer::render_over_frame) passes the copy as
// out. That is sound because a thread reads its own pixel, and no other, before it writes it.
template <typename OutT>
__global__ void render_kernel(InPix in, int width, long long npx, Mat9 mat, PipelineConfig cfg, OutT* out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npx) return;
    const long long y = i / width;
    store_pixel<OutT>(out, i, convert_pixel(cfg, mat.m, in_at(in, i - y * width, y)));  // through cfg's tone stage
}

template <typename OutT>
__global__ void render_with_stats_kernel(InPix in, int width, int height, InnerCrop crop, Mat9 mat,
                                         PipelineConfig cfg, OutT* __restrict__ out,
                                         unsigned int* __restrict__ hist_out, double* __restrict__ partials) {
    __shared__ unsigned int sh_hist[kRenderHistBins];
    for (int k = threadIdx.x; k < kRenderHistBins; k += blockDim.x) sh_hist[k] = 0;
    __syncthreads();
    double acc[kStatFields];
    for (int k = 0; k < kStatFields; ++k) acc[k] = 0.0;
    const long long npx = (long long)width * height;
    const long long step = (long long)gridDim.x * blockDim.x;
    const int x0 = crop.x0, x1 = crop.x0 + crop.w;
    const int y0 = crop.y0, y1 = crop.y0 + crop.h;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < npx; i += step) {
        const int y = (int)(i / width);
        const int x = (int)(i - (long long)y * width);
        const Rgb lin = convert_pixel_linear(cfg, mat.m, in_at(in, x, y));  // pre-shoulder: the clip rules
        const Rgb disp = to_display(cfg, lin);                              // post-shoulder: Lab
        // The picture and its RGB histogram go through the tone stage; the clip rules and Lab read before it.
        const Rgb v = encode_output(through_tone(cfg, disp), cfg.encoding);
        store_pixel<OutT>(out, i, v);
        if (x >= x0 && x < x1 && y >= y0 && y < y1) {
            atomicAdd(&sh_hist[0 * 256 + quantize_u8(v.r)], 1u);
            atomicAdd(&sh_hist[1 * 256 + quantize_u8(v.g)], 1u);
            atomicAdd(&sh_hist[2 * 256 + quantize_u8(v.b)], 1u);
            if (linear_channel_white_clipped(lin.r) || linear_channel_white_clipped(lin.g) ||
                linear_channel_white_clipped(lin.b))
                acc[0] += 1.0;
            if (linear_channel_black_clipped(lin.r) && linear_channel_black_clipped(lin.g) &&
                linear_channel_black_clipped(lin.b))
                acc[1] += 1.0;
            const Lab lab = linear_srgb_to_lab(disp);
            atomicAdd(&sh_hist[kRgbHistBins + lab_bin(lab.L)], 1u);
            if (lab.L >= 20.0f && lab.L <= 75.0f) { acc[2] += lab.a; acc[3] += lab.b; acc[4] += 1.0; }
            if (lab.L >= 75.0f && lab.L <= 95.0f) { acc[5] += lab.a; acc[6] += lab.b; acc[7] += 1.0; }
            if (lab.L >= 5.0f && lab.L <= 25.0f)  { acc[8] += lab.a; acc[9] += lab.b; acc[10] += 1.0; }
            acc[11] += lab.a;
            acc[12] += lab.b;
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < kRenderHistBins; k += blockDim.x)
        if (sh_hist[k]) atomicAdd(&hist_out[k], sh_hist[k]);
    block_sum<kStatFields>(acc, partials + (size_t)blockIdx.x * kStatFields);
}

__global__ void highlight_clip_kernel(InPix in, InnerCrop crop, const float* __restrict__ mats, PipelineConfig cfg,
                                      float cr, float cg, float cb, unsigned int* __restrict__ bins_out,
                                      unsigned int* __restrict__ max_hist_out) {
    __shared__ unsigned int sh_bins[kHighlightBins];
    // The pre-shoulder max-channel histogram is dynamic shared memory, sized only when max_hist_out is set.
    extern __shared__ unsigned int sh_max[];
    for (int k = threadIdx.x; k < kHighlightBins; k += blockDim.x) sh_bins[k] = 0;
    if (max_hist_out)
        for (int k = threadIdx.x; k < kMaxHistBins; k += blockDim.x) sh_max[k] = 0;
    __syncthreads();
    const int c = blockIdx.y;
    const float* m = mats + c * 9;
    unsigned int loc_er = 0, loc_eg = 0, loc_eb = 0, loc_wc = 0;
    const long long crop_npx = (long long)crop.w * crop.h;
    const long long step = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < crop_npx; idx += step) {
        const int ry = (int)(idx / crop.w);
        const int rx = (int)(idx - (long long)ry * crop.w);
        const Rgb lin = convert_pixel_linear(cfg, m, in_at(in, crop.x0 + rx, crop.y0 + ry));
        if (lin.r >= cr) loc_er++;
        if (lin.g >= cg) loc_eg++;
        if (lin.b >= cb) loc_eb++;
        if (linear_channel_white_clipped(lin.r) || linear_channel_white_clipped(lin.g) ||
            linear_channel_white_clipped(lin.b))
            loc_wc++;
        const float L = linear_srgb_to_L(to_display(cfg, lin));
        atomicAdd(&sh_bins[lab_bin(L)], 1u);
        if (max_hist_out) atomicAdd(&sh_max[max_hist_bin(fmaxf(lin.r, fmaxf(lin.g, lin.b)))], 1u);
    }
    if (loc_er) atomicAdd(&sh_bins[kLabBins + 0], loc_er);
    if (loc_eg) atomicAdd(&sh_bins[kLabBins + 1], loc_eg);
    if (loc_eb) atomicAdd(&sh_bins[kLabBins + 2], loc_eb);
    if (loc_wc) atomicAdd(&sh_bins[kLabBins + 3], loc_wc);
    __syncthreads();
    unsigned int* dst = bins_out + (size_t)c * kHighlightBins;
    for (int k = threadIdx.x; k < kHighlightBins; k += blockDim.x)
        if (sh_bins[k]) atomicAdd(&dst[k], sh_bins[k]);
    if (max_hist_out) {
        unsigned int* mdst = max_hist_out + (size_t)c * kMaxHistBins;
        for (int k = threadIdx.x; k < kMaxHistBins; k += blockDim.x)
            if (sh_max[k]) atomicAdd(&mdst[k], sh_max[k]);
    }
}

// Blocks per candidate for the reductions: one per 16 pixels a thread, at most 256.
int reduction_chunks(long long npx) {
    const long long by_size = (npx + kThreads * 16 - 1) / (kThreads * 16);
    return (int)std::clamp<long long>(by_size, 1, 256);
}

// The median L* of a clip pass's histogram, as cpu_highlight_clip reads its own (the bin's centre).
double median_from_hist_L(const unsigned int* hist, double tot) {
    if (tot <= 0.0) return 0.0;
    const double target = tot * 0.5;
    double cum = 0.0;
    int k = 0;
    for (; k < kLabBins; ++k) {
        cum += (double)hist[k];
        if (cum >= target) break;
    }
    return (std::min(k, kLabBins - 1) + 0.5) * (100.0 / kLabBins);
}

}  // namespace

// A device is usable when this build's kernels run on it: a kernel built for another architecture (CUDA_ARCH) fails
// to launch, and the passes then stay on the CPU, which is said once.
__global__ void probe_kernel(int* out) { *out = 1; }

bool gpu_device_available() {
    static const bool available = [] {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
            cudaGetLastError();  // clear the sticky "no device" error
            return false;
        }
        int* d = nullptr;
        if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        probe_kernel<<<1, 1>>>(d);
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        cudaFree(d);
        if (e != cudaSuccess) {
            cudaGetLastError();
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, 0);
            std::fprintf(stderr,
                         "WARNING: the CUDA device %s (compute capability %d.%d) does not run this build's GPU passes, "
                         "built for CUDA_ARCH=%d (%s): the passes run on the CPU; build with CUDA_ARCH=%d%d to run "
                         "them on it\n",
                         prop.name, prop.major, prop.minor, NEGICC_CUDA_ARCH, cudaGetErrorString(e), prop.major,
                         prop.minor);
            return false;
        }
        return true;
    }();
    return available;
}

void* gpu_malloc(size_t bytes) {
    void* p = nullptr;
    NEGICC_CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}
void gpu_free(void* p) {
    if (p) cudaFree(p);
}
void gpu_upload(void* dst, const void* src, size_t bytes) {
    NEGICC_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void gpu_download(void* dst, const void* src, size_t bytes) {
    NEGICC_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void gpu_render(const RenderArgs& a) {
    std::lock_guard<std::mutex> lock(g_op_mutex);
    const long long npx = (long long)a.width * a.height;
    if (npx <= 0) return;
    const InPix in = pass_input(a.in, a.in_view, a.width);
    Mat9 mat;
    std::memcpy(mat.m, a.matrix, sizeof(mat.m));
    if (!a.stats) {
        const unsigned blocks = (unsigned)((npx + kThreads - 1) / kThreads);
        if (a.out_type == 1)
            render_kernel<uint8_t><<<blocks, kThreads>>>(in, a.width, npx, mat, a.cfg, static_cast<uint8_t*>(a.out));
        else if (a.out_type == 0)
            render_kernel<uint16_t><<<blocks, kThreads>>>(in, a.width, npx, mat, a.cfg, static_cast<uint16_t*>(a.out));
        else
            render_kernel<float><<<blocks, kThreads>>>(in, a.width, npx, mat, a.cfg, static_cast<float*>(a.out));
        NEGICC_CUDA_CHECK(cudaGetLastError());
        NEGICC_CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }
    const InnerCrop crop = pass_crop(a.stats_crop, a.width, a.height);
    const int chunks = reduction_chunks(npx);
    unsigned int* d_hist = static_cast<unsigned int*>(g_hist.get(kRenderHistBins * sizeof(unsigned int)));
    NEGICC_CUDA_CHECK(cudaMemset(d_hist, 0, kRenderHistBins * sizeof(unsigned int)));
    const size_t part_count = (size_t)chunks * kStatFields;
    double* d_part = static_cast<double*>(g_partials.get(part_count * sizeof(double)));
    if (a.out_type == 1)
        render_with_stats_kernel<uint8_t><<<chunks, kThreads>>>(in, a.width, a.height, crop, mat, a.cfg,
                                                                 static_cast<uint8_t*>(a.out), d_hist, d_part);
    else if (a.out_type == 0)
        render_with_stats_kernel<uint16_t><<<chunks, kThreads>>>(in, a.width, a.height, crop, mat, a.cfg,
                                                                  static_cast<uint16_t*>(a.out), d_hist, d_part);
    else
        render_with_stats_kernel<float><<<chunks, kThreads>>>(in, a.width, a.height, crop, mat, a.cfg,
                                                               static_cast<float*>(a.out), d_hist, d_part);
    NEGICC_CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned int> hist(kRenderHistBins);
    std::vector<double> part(part_count);
    NEGICC_CUDA_CHECK(cudaMemcpy(hist.data(), d_hist, kRenderHistBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    NEGICC_CUDA_CHECK(cudaMemcpy(part.data(), d_part, part_count * sizeof(double), cudaMemcpyDeviceToHost));
    std::memset(a.stats, 0, sizeof(PositiveStats));
    for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 256; ++k) a.stats->hist_rgb[c][k] = hist[c * 256 + k];
    for (int k = 0; k < kLabBins; ++k) a.stats->hist_L[k] = hist[kRgbHistBins + k];
    double s[kStatFields] = {};
    for (int k = 0; k < chunks; ++k)
        for (int f = 0; f < kStatFields; ++f) s[f] += part[(size_t)k * kStatFields + f];
    const double tot = (double)crop.w * crop.h;
    if (tot > 0.0) {
        a.stats->white_clip = (float)(s[0] / tot * 100.0);
        a.stats->black_clip = (float)(s[1] / tot * 100.0);
        const double ma = s[4] > 0 ? s[2] / s[4] : s[11] / tot;
        const double mb = s[4] > 0 ? s[3] / s[4] : s[12] / tot;
        a.stats->mid_a = (float)ma;
        a.stats->mid_b = (float)mb;
        a.stats->hi_a = (float)(s[7] > 0 ? s[5] / s[7] : ma);
        a.stats->hi_b = (float)(s[7] > 0 ? s[6] / s[7] : mb);
        a.stats->sh_a = (float)(s[10] > 0 ? s[8] / s[10] : ma);
        a.stats->sh_b = (float)(s[10] > 0 ? s[9] / s[10] : mb);
    }
}

void gpu_highlight_clip(const HighlightClipArgs& a) {
    if (a.n <= 0) return;
    std::lock_guard<std::mutex> lock(g_op_mutex);
    const InnerCrop crop = pass_crop(a.crop, a.width, a.height);
    const long long npx = (long long)crop.w * crop.h;
    if (npx <= 0) {
        std::memset(a.out, 0, (size_t)a.n * 5 * sizeof(float));
        return;
    }
    float* d_mats = static_cast<float*>(g_matrices.get((size_t)a.n * 9 * sizeof(float)));
    NEGICC_CUDA_CHECK(cudaMemcpy(d_mats, a.matrices, (size_t)a.n * 9 * sizeof(float), cudaMemcpyHostToDevice));
    const size_t bins_count = (size_t)a.n * kHighlightBins;
    unsigned int* d_bins = static_cast<unsigned int*>(g_hist.get(bins_count * sizeof(unsigned int)));
    NEGICC_CUDA_CHECK(cudaMemset(d_bins, 0, bins_count * sizeof(unsigned int)));
    unsigned int* d_max = nullptr;
    const size_t max_count = (size_t)a.n * kMaxHistBins;
    if (a.max_hist) {
        d_max = static_cast<unsigned int*>(g_max_hist.get(max_count * sizeof(unsigned int)));
        NEGICC_CUDA_CHECK(cudaMemset(d_max, 0, max_count * sizeof(unsigned int)));
    }
    const int chunks = reduction_chunks(npx);
    const size_t dyn_smem = a.max_hist ? kMaxHistBins * sizeof(unsigned int) : 0;
    highlight_clip_kernel<<<dim3(chunks, a.n), kThreads, dyn_smem>>>(pass_input(a.in, a.in_view, a.width), crop,
                                                                      d_mats, a.cfg, a.ceil_lin[0], a.ceil_lin[1],
                                                                      a.ceil_lin[2], d_bins, d_max);
    NEGICC_CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned int> bins(bins_count);
    NEGICC_CUDA_CHECK(cudaMemcpy(bins.data(), d_bins, bins_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (a.max_hist) {
        std::vector<unsigned int> mh(max_count);
        NEGICC_CUDA_CHECK(cudaMemcpy(mh.data(), d_max, max_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (size_t k = 0; k < max_count; ++k) a.max_hist[k] = mh[k];
    }
    const double tot = (double)npx;
    for (int c = 0; c < a.n; ++c) {
        const unsigned int* b = bins.data() + (size_t)c * kHighlightBins;
        float* o = a.out + (size_t)c * 5;
        for (int k = 0; k < 4; ++k) o[k] = (float)((double)b[kLabBins + k] / tot * 100.0);
        o[4] = (float)median_from_hist_L(b, tot);
    }
}


namespace {
void require_device() {
    if (!gpu_device_available()) throw GpuError("no CUDA device runs this build's kernels");
}
}  // namespace

// ---- The full-size decode on the device -----------------------------------------------------------------------------
// The CUDA twins of cpu_preview_debayer, cpu_grain_aware_debayer and cpu_pixel_shift_merge (rgb_view.cc): one output
// pixel per thread, the same arithmetic. Every pointer in the arguments is device memory (the mosaics, the flat
// field's tables, rgb_out).
namespace {

__device__ __forceinline__ void map_inverse_geometry_dev(
    int x_dst, int y_dst, int dst_w, int dst_h,
    int rw, int rh, int rot_cw, bool hflip, bool vflip,
    int& x_rel, int& y_rel)
{
    rot_cw = ((rot_cw % 360) + 360) % 360;
    if (rot_cw == 180) {
        hflip = !hflip;
        vflip = !vflip;
        rot_cw = 0;
    }
    int x1 = hflip ? (dst_w - 1 - x_dst) : x_dst;
    int y1 = vflip ? (dst_h - 1 - y_dst) : y_dst;

    if (rot_cw == 0) {
        x_rel = x1;
        y_rel = y1;
    } else if (rot_cw == 90) {
        x_rel = y1;
        y_rel = rh - 1 - x1;
    } else { // 270
        x_rel = rw - 1 - y1;
        y_rel = x1;
    }
}

__device__ __forceinline__ int get_bayer_color_dev(int r, int c, unsigned int filters) {
    return (filters >> (((((unsigned int)r << 1) & 14u) | ((unsigned int)c & 1u)) << 1)) & 3;
}

__device__ __forceinline__ float get_px_refl_dev(const uint16_t* bayer, int r, int c, int h, int w, size_t pitch) {
    int r_refl = (r < 0) ? -r : (r >= h ? 2 * (h - 1) - r : r);
    int c_refl = (c < 0) ? -c : (c >= w ? 2 * (w - 1) - c : c);
    r_refl = max(0, min(h - 1, r_refl));
    c_refl = max(0, min(w - 1, c_refl));
    return (float)bayer[(size_t)r_refl * pitch + c_refl];
}

__device__ __forceinline__ float reconstruct_green_soft_dev(const uint16_t* m, int r, int c, int h, int w,
                                                            size_t pitch, float white, float eps) {
    if ((r + c) & 1) {
        return get_px_refl_dev(m, r, c, h, w, pitch);
    }
    float c_center = get_px_refl_dev(m, r, c, h, w, pitch);
    float g_up     = get_px_refl_dev(m, r - 1, c, h, w, pitch);
    float g_down   = get_px_refl_dev(m, r + 1, c, h, w, pitch);
    float g_left   = get_px_refl_dev(m, r, c - 1, h, w, pitch);
    float g_right  = get_px_refl_dev(m, r, c + 1, h, w, pitch);

    float c_up2    = get_px_refl_dev(m, r - 2, c, h, w, pitch);
    float c_down2  = get_px_refl_dev(m, r + 2, c, h, w, pitch);
    float c_left2  = get_px_refl_dev(m, r, c - 2, h, w, pitch);
    float c_right2 = get_px_refl_dev(m, r, c + 2, h, w, pitch);

    float dv = fabsf(g_up - g_down) + fabsf(2.0f * c_center - c_up2 - c_down2);
    float dh = fabsf(g_left - g_right) + fabsf(2.0f * c_center - c_left2 - c_right2);

    float g_v = 0.5f * (g_up + g_down) + 0.25f * (2.0f * c_center - c_up2 - c_down2);
    float g_h = 0.5f * (g_left + g_right) + 0.25f * (2.0f * c_center - c_left2 - c_right2);

    float wv = 1.0f / (eps + dv * dv);
    float wh = 1.0f / (eps + dh * dh);
    float g_est = (wv * g_v + wh * g_h) / (wv + wh);
    return fmaxf(0.0f, fminf(white, g_est));
}

__global__ void preview_debayer_kernel(PreviewDebayerArgs a) {
    int x2 = blockIdx.x * blockDim.x + threadIdx.x;
    int y2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (x2 >= a.dst_w || y2 >= a.dst_h) return;

    const int half_rw = a.rw / 2;
    const int half_rh = a.rh / 2;
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;

    int x_rel = 0, y_rel = 0;
    map_inverse_geometry_dev(x2, y2, a.dst_w, a.dst_h, half_rw, half_rh, a.rot_cw, a.hflip, a.vflip, x_rel, y_rel);

    int r = a.ry + y_rel * 2;
    int c = a.rx + x_rel * 2;

    uint16_t s00 = (uint16_t)get_px_refl_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
    uint16_t s01 = (uint16_t)get_px_refl_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch);
    uint16_t s10 = (uint16_t)get_px_refl_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch);
    uint16_t s11 = (uint16_t)get_px_refl_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch);

    int col00 = get_bayer_color_dev(r, c, a.filters);
    int col01 = get_bayer_color_dev(r, c + 1, a.filters);
    int col10 = get_bayer_color_dev(r + 1, c, a.filters);
    int col11 = get_bayer_color_dev(r + 1, c + 1, a.filters);

    // Both greens averaged, as negicc's cpu_preview_debayer and RgbView::from_cfa_half do.
    uint16_t R = 0, B = 0;
    uint32_t G_sum = 0;
    const uint16_t samples[4] = {s00, s01, s10, s11};
    const int cols[4] = {col00, col01, col10, col11};
    for (int k = 0; k < 4; ++k) {
        if (cols[k] & 1) G_sum += samples[k];
        else if (cols[k] == 0) R = samples[k];
        else B = samples[k];
    }
    uint16_t G = (uint16_t)(G_sum >> 1);
    uint16_t px[3] = {R, G, B};
    if (a.flat.gain) {
        flat_divide(a.flat, a.flat_x0 + x_rel, a.flat_y0 + y_rel, px);
    }

    long long dst_idx = ((long long)y2 * a.dst_w + x2) * 3;
    a.rgb_out[dst_idx + 0] = px[0];
    a.rgb_out[dst_idx + 1] = px[1];
    a.rgb_out[dst_idx + 2] = px[2];
}

__global__ void grain_aware_debayer_kernel(GrainAwareDebayerArgs a) {
    int x2 = blockIdx.x * blockDim.x + threadIdx.x;
    int y2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (x2 >= a.dst_w || y2 >= a.dst_h) return;

    int x_rel = 0, y_rel = 0;
    map_inverse_geometry_dev(x2, y2, a.dst_w, a.dst_h, a.rw, a.rh, a.rot_cw, a.hflip, a.vflip, x_rel, y_rel);

    int c = a.rx + x_rel;
    int r = a.ry + y_rel;

    const float k = 16.0f;
    const float eps = (a.eps > 0.0f) ? a.eps : 4096.0f;
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;

    float G = reconstruct_green_soft_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch, a.white_level, eps);

    bool r_even = ((r & 1) == 0);
    bool c_even = ((c & 1) == 0);

    float R = 0.0f;
    float B = 0.0f;

    float c_center = get_px_refl_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
    bool near_clip = (c_center >= (a.white_level - 64.0f)) || (G >= (a.white_level - 64.0f));

    if (near_clip) {
        if (r_even && c_even) {
            R = get_px_refl_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
            float b00 = get_px_refl_dev(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch);
            float b01 = get_px_refl_dev(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch);
            float b10 = get_px_refl_dev(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch);
            float b11 = get_px_refl_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch);
            B = 0.25f * (b00 + b01 + b10 + b11);
        } else if (!r_even && !c_even) {
            B = get_px_refl_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
            float r00 = get_px_refl_dev(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch);
            float r01 = get_px_refl_dev(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch);
            float r10 = get_px_refl_dev(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch);
            float r11 = get_px_refl_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch);
            R = 0.25f * (r00 + r01 + r10 + r11);
        } else if (r_even && !c_even) {
            R = 0.5f * (get_px_refl_dev(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) +
                        get_px_refl_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch));
            B = 0.5f * (get_px_refl_dev(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) +
                        get_px_refl_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch));
        } else {
            R = 0.5f * (get_px_refl_dev(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) +
                        get_px_refl_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch));
            B = 0.5f * (get_px_refl_dev(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) +
                        get_px_refl_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch));
        }
    } else {
        if (r_even && c_even) {
            R = get_px_refl_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
        } else if (r_even && !c_even) {
            float g_l = reconstruct_green_soft_dev(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch, a.white_level, eps);
            float g_r = reconstruct_green_soft_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch, a.white_level, eps);
            float rl = (get_px_refl_dev(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) + k) / (g_l + k);
            float rr = (get_px_refl_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch) + k) / (g_r + k);
            R = (G + k) * (0.5f * (rl + rr)) - k;
        } else if (!r_even && c_even) {
            float g_u = reconstruct_green_soft_dev(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch, a.white_level, eps);
            float g_d = reconstruct_green_soft_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch, a.white_level, eps);
            float ru = (get_px_refl_dev(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) + k) / (g_u + k);
            float rd = (get_px_refl_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch) + k) / (g_d + k);
            R = (G + k) * (0.5f * (ru + rd)) - k;
        } else {
            float g00 = reconstruct_green_soft_dev(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float g01 = reconstruct_green_soft_dev(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float g10 = reconstruct_green_soft_dev(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float g11 = reconstruct_green_soft_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float r00 = (get_px_refl_dev(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch) + k) / (g00 + k);
            float r01 = (get_px_refl_dev(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch) + k) / (g01 + k);
            float r10 = (get_px_refl_dev(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch) + k) / (g10 + k);
            float r11 = (get_px_refl_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch) + k) / (g11 + k);
            R = (G + k) * (0.25f * (r00 + r01 + r10 + r11)) - k;
        }

        if (!r_even && !c_even) {
            B = get_px_refl_dev(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
        } else if (!r_even && c_even) {
            float g_l = reconstruct_green_soft_dev(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch, a.white_level, eps);
            float g_r = reconstruct_green_soft_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch, a.white_level, eps);
            float bl = (get_px_refl_dev(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) + k) / (g_l + k);
            float br = (get_px_refl_dev(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch) + k) / (g_r + k);
            B = (G + k) * (0.5f * (bl + br)) - k;
        } else if (r_even && !c_even) {
            float g_u = reconstruct_green_soft_dev(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch, a.white_level, eps);
            float g_d = reconstruct_green_soft_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch, a.white_level, eps);
            float bu = (get_px_refl_dev(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) + k) / (g_u + k);
            float bd = (get_px_refl_dev(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch) + k) / (g_d + k);
            B = (G + k) * (0.5f * (bu + bd)) - k;
        } else {
            float g00 = reconstruct_green_soft_dev(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float g01 = reconstruct_green_soft_dev(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float g10 = reconstruct_green_soft_dev(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float g11 = reconstruct_green_soft_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w,
                                                   pitch, a.white_level, eps);
            float b00 = (get_px_refl_dev(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch) + k) / (g00 + k);
            float b01 = (get_px_refl_dev(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch) + k) / (g01 + k);
            float b10 = (get_px_refl_dev(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch) + k) / (g10 + k);
            float b11 = (get_px_refl_dev(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch) + k) / (g11 + k);
            B = (G + k) * (0.25f * (b00 + b01 + b10 + b11)) - k;
        }
    }

    uint16_t px[3] = {(uint16_t)roundf(fmaxf(0.0f, fminf(65535.0f, R))),
                      (uint16_t)roundf(fmaxf(0.0f, fminf(65535.0f, G))),
                      (uint16_t)roundf(fmaxf(0.0f, fminf(65535.0f, B)))};
    if (a.flat.gain) flat_divide(a.flat, a.flat_x0 + x_rel, a.flat_y0 + y_rel, px);
    long long dst_idx = ((long long)y2 * a.dst_w + x2) * 3;
    a.rgb_out[dst_idx + 0] = px[0];
    a.rgb_out[dst_idx + 1] = px[1];
    a.rgb_out[dst_idx + 2] = px[2];
}

__global__ void pixel_shift_merge_kernel(PixelShiftMergeArgs a) {
    int x2 = blockIdx.x * blockDim.x + threadIdx.x;
    int y2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (x2 >= a.dst_w || y2 >= a.dst_h) return;

    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;
    int x_rel = 0, y_rel = 0;
    map_inverse_geometry_dev(x2, y2, a.dst_w, a.dst_h, a.rw, a.rh, a.rot_cw, a.hflip, a.vflip, x_rel, y_rel);

    int c = a.rx + x_rel;
    int r = a.ry + y_rel;

    uint16_t s0 = (uint16_t)get_px_refl_dev(a.bayer_f0, r,     c,     a.src_h, a.src_w, pitch);
    uint16_t s1 = (uint16_t)get_px_refl_dev(a.bayer_f1, r - 1, c,     a.src_h, a.src_w, pitch);
    uint16_t s2 = (uint16_t)get_px_refl_dev(a.bayer_f2, r - 1, c + 1, a.src_h, a.src_w, pitch);
    uint16_t s3 = (uint16_t)get_px_refl_dev(a.bayer_f3, r,     c + 1, a.src_h, a.src_w, pitch);

    const uint16_t samples[4] = {s0, s1, s2, s3};
    const int coords[4][2] = {{r, c}, {r - 1, c}, {r - 1, c + 1}, {r, c + 1}};

    uint16_t R = 0, B = 0;
    uint32_t G_sum = 0;
    int g_count = 0;
    bool g_clipped = false;
    for (int i = 0; i < 4; ++i) {
        int col = get_bayer_color_dev(coords[i][0], coords[i][1], a.filters);
        if (col & 1) {
            G_sum += samples[i];
            g_count++;
            if (samples[i] >= (uint16_t)a.white_level) g_clipped = true;
        } else if (col == 0) {
            R = samples[i];
        } else {
            B = samples[i];
        }
    }

    uint16_t G = (g_count > 0) ? (uint16_t)(G_sum / g_count) : 0;
    if (g_clipped || G >= (uint16_t)a.white_level) G = (uint16_t)a.white_level;
    if (R >= (uint16_t)a.white_level) R = (uint16_t)a.white_level;
    if (B >= (uint16_t)a.white_level) B = (uint16_t)a.white_level;

    uint16_t px[3] = {R, G, B};
    if (a.flat.gain) flat_divide(a.flat, a.flat_x0 + x_rel, a.flat_y0 + y_rel, px);
    long long dst_idx = ((long long)y2 * a.dst_w + x2) * 3;
    a.rgb_out[dst_idx + 0] = px[0];
    a.rgb_out[dst_idx + 1] = px[1];
    a.rgb_out[dst_idx + 2] = px[2];
}

}  // namespace

void gpu_preview_debayer(const PreviewDebayerArgs& a) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("gpu_preview_debayer: rot_cw must be a multiple of 90");
    }
    if (a.dst_w <= 0 || a.dst_h <= 0) return;   // nothing to make, as the CPU's loops run no row
    require_device();
    std::lock_guard<std::mutex> lock(g_op_mutex);
    dim3 block(16, 16);
    dim3 grid((a.dst_w + block.x - 1) / block.x, (a.dst_h + block.y - 1) / block.y);
    preview_debayer_kernel<<<grid, block>>>(a);
    NEGICC_CUDA_CHECK(cudaGetLastError());
    NEGICC_CUDA_CHECK(cudaDeviceSynchronize());
}

inline bool is_rggb_pattern_host(unsigned int filters) {
    const int c10 = negicc::bayer_color_at(1, 0, filters);
    return (negicc::bayer_color_at(0, 0, filters) == 0 &&
            negicc::bayer_color_at(0, 1, filters) == 1 &&
            (c10 == 1 || c10 == 3) &&
            negicc::bayer_color_at(1, 1, filters) == 2);
}

void gpu_grain_aware_debayer(const GrainAwareDebayerArgs& a) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("gpu_grain_aware_debayer: rot_cw must be a multiple of 90");
    }
    if (!is_rggb_pattern_host(a.filters)) {
        throw std::invalid_argument("gpu_grain_aware_debayer: currently only supports RGGB Bayer pattern");
    }
    if (a.dst_w <= 0 || a.dst_h <= 0) return;   // nothing to make, as the CPU's loops run no row
    require_device();
    std::lock_guard<std::mutex> lock(g_op_mutex);
    dim3 block(16, 16);
    dim3 grid((a.dst_w + block.x - 1) / block.x, (a.dst_h + block.y - 1) / block.y);
    grain_aware_debayer_kernel<<<grid, block>>>(a);
    NEGICC_CUDA_CHECK(cudaGetLastError());
    NEGICC_CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_pixel_shift_merge(const PixelShiftMergeArgs& a) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("gpu_pixel_shift_merge: rot_cw must be a multiple of 90");
    }
    if (!(a.white_level >= 0.0f && a.white_level <= std::numeric_limits<uint16_t>::max())) {
        throw std::invalid_argument("gpu_pixel_shift_merge: white_level must be within a 16-bit sample's range");
    }
    if (a.dst_w <= 0 || a.dst_h <= 0) return;   // nothing to make, as the CPU's loops run no row
    require_device();
    std::lock_guard<std::mutex> lock(g_op_mutex);
    dim3 block(16, 16);
    dim3 grid((a.dst_w + block.x - 1) / block.x, (a.dst_h + block.y - 1) / block.y);
    pixel_shift_merge_kernel<<<grid, block>>>(a);
    NEGICC_CUDA_CHECK(cudaGetLastError());
    NEGICC_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace negicc
