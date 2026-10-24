// Flat-field gain lookup shared by FlatField::apply (CPU) and the fused crop + flat debayers (CPU and CUDA).
//
// One function, flat_divide, holds the arithmetic, so a decode that applies the flat while it is cut out of the
// sensor (the fused path) gives the same bits as decoding the whole frame, applying the flat, then cutting it:
// both evaluate the same float expressions in the same order. The CPU is built with -ffp-contract=off and nvcc with
// --fmad=false, and float division is IEEE round-to-nearest on both, so CPU and GPU agree bit for bit as well.
#ifndef NEGICC_FLAT_SAMPLE_H
#define NEGICC_FLAT_SAMPLE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define NEGICC_FLAT_HD __host__ __device__ __forceinline__
#else
#define NEGICC_FLAT_HD inline
#endif

namespace negicc {

// The flat's gain grid and its per-axis lookup tables for one decode size (FlatField::tables), as plain pointers
// (host or device-visible). Frame pixel (fx, fy) of that decode lies between grid rows ay_i[fy] and ay_i[fy] + 1
// (weight ay_t[fy] of the upper one) and grid columns ax_i[fx] and ax_i[fx] + 1 (weight ax_t[fx]). gain == nullptr:
// no flat field.
struct FlatSampleArgs {
    const float* gain = nullptr;  // grid_h * grid_w * 3 floats, row-major RGB
    int grid_w = 0;
    const int* ax_i = nullptr;    // frame width entries
    const float* ax_t = nullptr;
    const int* ay_i = nullptr;    // frame height entries
    const float* ay_t = nullptr;
};

// Divides pixel (fx, fy) of the decode by its gain, rounded once to the nearest integer (ties to even) and clamped to
// 65535: bilinear in the grid, first along y (each grid column), then along x.
NEGICC_FLAT_HD void flat_divide(const FlatSampleArgs& f, int fx, int fy, uint16_t px[3]) {
    const float ty = f.ay_t[fy];
    const float* g0 = f.gain + (size_t)f.ay_i[fy] * f.grid_w * 3;
    const float* g1 = g0 + (size_t)f.grid_w * 3;
    const int k = f.ax_i[fx] * 3;
    const float tx = f.ax_t[fx];
    for (int c = 0; c < 3; ++c) {
        const float lo = g0[k + c] + (g1[k + c] - g0[k + c]) * ty;
        const float hi = g0[k + 3 + c] + (g1[k + 3 + c] - g0[k + 3 + c]) * ty;
        const float gain = lo + (hi - lo) * tx;
#ifdef __CUDA_ARCH__
        const float v = rintf((float)px[c] / gain);
        px[c] = (uint16_t)fminf(v, 65535.0f);
#else
        const float v = std::nearbyint((float)px[c] / gain);
        px[c] = (uint16_t)std::min(v, 65535.0f);
#endif
    }
}

}  // namespace negicc

#endif  // NEGICC_FLAT_SAMPLE_H
