// Scan flat field: the light + lens gain measured on the empty gate, applied to decodes in sensor coordinates.
#ifndef NEGICC_FLAT_FIELD_H
#define NEGICC_FLAT_FIELD_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "flat_sample.h"

namespace negicc {

// A coarse per-channel gain grid in half-size sensor coordinates (LibRaw user_flip = 0, before any crop / rotate /
// flip): block (i, j) covers half-size pixels [i*block, (i+1)*block) x [j*block, (j+1)*block), its centre carries the
// gain, and pixels between centres are bilinear (clamped at the edges), as scan_flat_field.gain_sample does in Python.
// Gains are normalised to 1 at the sensor centre; a decode is divided by them. Immutable once built.
class FlatTables;

class FlatField {
public:
    // gain: grid_h * grid_w * 3 floats, row-major RGB. half_w / half_h: the half-size decode the grid was measured on.
    FlatField(std::vector<float> gain, int grid_h, int grid_w, int block, int half_w, int half_h, std::string id);

    int grid_w() const { return m_gw; }
    int grid_h() const { return m_gh; }
    int block() const { return m_block; }
    int half_w() const { return m_half_w; }
    int half_h() const { return m_half_h; }
    const std::string& id() const { return m_id; }
    const std::vector<float>& gain() const { return m_gain; }

    // Divides an interleaved u16 RGB decode in place (computed in float, rounded once, clamped to 65535).
    // half: the decode is half-size (else full-size, sampled at half-size coordinates (x + 0.5) / 2 - 0.5).
    // Before dividing, counts pixels with any channel >= clip_level; returns that count.
    // Throws std::invalid_argument if the decode size does not match the size the grid was measured on.
    size_t apply(uint16_t* rgb, int width, int height, bool half, uint16_t clip_level) const;

    // Divides a full-size CFA mosaic in place, before any demosaic (the white-light decode): photosite (x, y) is
    // divided by the gain of its own colour, pattern[(y % 6) * 6 + x % 6] (0 = R, 1 = G, 2 = B; a 2x2 Bayer or a
    // 6x6 X-Trans pattern), sampled at half-size coordinates as apply() samples a full-size decode. pitch: u16 per
    // row of cfa. Throws std::invalid_argument if width / 2 x height / 2 is not the size the grid was measured on.
    void apply_cfa(uint16_t* cfa, size_t pitch, int width, int height, const uint8_t pattern[36]) const;

    // The gain grid and the lookup tables of a width x height decode (half: half-size, else full-size), in
    // host or GPU-visible memory, for flat_divide in the fused crop + flat debayers. Built once per size and kept
    // while the FlatField lives (a few hundred kB). Throws std::invalid_argument as apply() for a decode of another size.
    std::shared_ptr<const FlatTables> tables(int width, int height, bool half) const;

private:
    std::vector<float> m_gain;
    int m_gh, m_gw, m_block, m_half_w, m_half_h;
    std::string m_id;
    mutable std::mutex m_tables_mutex;
    mutable std::map<std::tuple<int, int, bool>, std::shared_ptr<const FlatTables>> m_tables;
};

// Optional custom allocator hook for FlatTables (used by negicc-station to place tables in GPU-visible pinned memory).
using FlatTableAllocFn = void* (*)(size_t bytes, void** device_ptr_out);
using FlatTableFreeFn = void (*)(void* host_ptr, size_t bytes);
void set_flat_table_allocator(FlatTableAllocFn alloc_fn, FlatTableFreeFn free_fn);

// The gain grid and per-axis lookup tables of one decode size in one allocation (FlatField::tables):
// host() for the CPU, device() for CUDA kernels (the mapped alias; equal to host() without a GPU).
class FlatTables {
public:
    FlatTables(const std::vector<float>& gain, int grid_w, int grid_h, int block, int width, int height, bool half);
    ~FlatTables();
    FlatTables(const FlatTables&) = delete;
    FlatTables& operator=(const FlatTables&) = delete;
    const FlatSampleArgs& host() const { return m_host; }
    const FlatSampleArgs& device() const { return m_device; }

private:
    void* m_block = nullptr;
    size_t m_bytes = 0;
    FlatSampleArgs m_host, m_device;
};

using FlatFieldPtr = std::shared_ptr<const FlatField>;

// Pixels of an interleaved u16 RGB image with any channel >= clip_level.
size_t count_clipped(const uint16_t* rgb, size_t pixels, uint16_t clip_level);

// Loads a flat-field .npz file saved by scan_flat_field.py (containing gain.npy float32 [gh, gw, 3] and info.npy JSON).
FlatFieldPtr load_flat_field_file(const std::string& path);

}  // namespace negicc

#endif  // NEGICC_FLAT_FIELD_H
