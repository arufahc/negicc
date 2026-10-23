#include "flat_field.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace negicc {

namespace {

void* default_flat_alloc(size_t bytes, void** dev_out) {
    if (dev_out) *dev_out = nullptr;
    void* p = std::malloc(bytes);
    if (!p) throw std::bad_alloc();
    return p;
}

void default_flat_free(void* p, size_t) {
    std::free(p);
}

FlatTableAllocFn g_alloc_fn = default_flat_alloc;
FlatTableFreeFn g_free_fn = default_flat_free;

// Per output coordinate: the lower grid index and the weight of the upper one (bilinear, clamped at the edges).
struct Axis {
    std::vector<int> i;
    std::vector<float> t;
};

Axis axis_table(int n, bool half, int block, int grid_n) {
    Axis a;
    a.i.resize(n);
    a.t.resize(n);
    for (int k = 0; k < n; ++k) {
        const double h = half ? k : (k + 0.5) / 2.0 - 0.5;  // half-size coordinate of the pixel
        double f = (h + 0.5) / block - 0.5;                  // grid coordinate (block centres)
        f = std::min(std::max(f, 0.0), (double)(grid_n - 1));
        const int i = std::min((int)f, grid_n - 2);
        a.i[k] = i;
        a.t[k] = (float)(f - i);
    }
    return a;
}

void check_decode_size(int width, int height, bool half, int half_w, int half_h) {
    const bool fits = half ? (width == half_w && height == half_h) : (width / 2 == half_w && height / 2 == half_h);
    if (!fits)
        throw std::invalid_argument("flat field was measured on a " + std::to_string(half_w) + "x" +
                                    std::to_string(half_h) + " half-size decode; this decode is " +
                                    std::to_string(width) + "x" + std::to_string(height) +
                                    (half ? " (half-size)" : " (full-size)"));
}

}  // namespace

void set_flat_table_allocator(FlatTableAllocFn alloc_fn, FlatTableFreeFn free_fn) {
    g_alloc_fn = alloc_fn ? alloc_fn : default_flat_alloc;
    g_free_fn = free_fn ? free_fn : default_flat_free;
}

FlatField::FlatField(std::vector<float> gain, int grid_h, int grid_w, int block, int half_w, int half_h,
                     std::string id)
    : m_gain(std::move(gain)), m_gh(grid_h), m_gw(grid_w), m_block(block), m_half_w(half_w), m_half_h(half_h),
      m_id(std::move(id)) {
    if (m_gh < 2 || m_gw < 2 || m_block < 1 || m_half_w < 1 || m_half_h < 1)
        throw std::invalid_argument("flat field: grid must be at least 2x2, block and size positive");
    if (m_gain.size() != (size_t)m_gh * m_gw * 3) throw std::invalid_argument("flat field: gain size != grid_h*grid_w*3");
    for (float g : m_gain)
        if (!(g > 0.0f) || !std::isfinite(g)) throw std::invalid_argument("flat field: gains must be finite and > 0");
}

size_t count_clipped(const uint16_t* rgb, size_t pixels, uint16_t clip_level) {
    size_t n = 0;
    #pragma omp parallel for reduction(+ : n) schedule(static)
    for (long long p = 0; p < (long long)pixels; ++p) {
        const uint16_t* q = rgb + p * 3;
        n += (q[0] >= clip_level || q[1] >= clip_level || q[2] >= clip_level) ? 1 : 0;
    }
    return n;
}

FlatTables::FlatTables(const std::vector<float>& gain, int grid_w, int grid_h, int block, int width, int height,
                       bool half) {
    const Axis ax = axis_table(width, half, block, grid_w);
    const Axis ay = axis_table(height, half, block, grid_h);
    // One block: gain grid, then ax_i, ax_t, ay_i, ay_t (all 4-byte types, so every part stays aligned).
    const size_t n_gain = gain.size(), nx = (size_t)width, ny = (size_t)height;
    m_bytes = (n_gain + 2 * nx + 2 * ny) * 4;
    void* dev = nullptr;
    m_block = g_alloc_fn(m_bytes, &dev);
    char* h = static_cast<char*>(m_block);
    std::memcpy(h, gain.data(), n_gain * 4);
    std::memcpy(h + n_gain * 4, ax.i.data(), nx * 4);
    std::memcpy(h + (n_gain + nx) * 4, ax.t.data(), nx * 4);
    std::memcpy(h + (n_gain + 2 * nx) * 4, ay.i.data(), ny * 4);
    std::memcpy(h + (n_gain + 2 * nx + ny) * 4, ay.t.data(), ny * 4);
    auto args = [&](char* base) {
        FlatSampleArgs a;
        a.gain = reinterpret_cast<const float*>(base);
        a.grid_w = grid_w;
        a.ax_i = reinterpret_cast<const int*>(base + n_gain * 4);
        a.ax_t = reinterpret_cast<const float*>(base + (n_gain + nx) * 4);
        a.ay_i = reinterpret_cast<const int*>(base + (n_gain + 2 * nx) * 4);
        a.ay_t = reinterpret_cast<const float*>(base + (n_gain + 2 * nx + ny) * 4);
        return a;
    };
    m_host = args(h);
    m_device = dev ? args(static_cast<char*>(dev)) : m_host;
}

FlatTables::~FlatTables() {
    g_free_fn(m_block, m_bytes);
}

std::shared_ptr<const FlatTables> FlatField::tables(int width, int height, bool half) const {
    check_decode_size(width, height, half, m_half_w, m_half_h);
    std::lock_guard<std::mutex> lock(m_tables_mutex);
    auto& t = m_tables[std::make_tuple(width, height, half)];
    if (!t) t = std::make_shared<const FlatTables>(m_gain, m_gw, m_gh, m_block, width, height, half);
    return t;
}

size_t FlatField::apply(uint16_t* rgb, int width, int height, bool half, uint16_t clip_level) const {
    const FlatSampleArgs f = tables(width, height, half)->host();
    size_t clipped = 0;
    #pragma omp parallel for reduction(+ : clipped) schedule(static)
    for (int y = 0; y < height; ++y) {
        uint16_t* p = rgb + (size_t)y * width * 3;
        for (int x = 0; x < width; ++x, p += 3) {
            clipped += (p[0] >= clip_level || p[1] >= clip_level || p[2] >= clip_level) ? 1 : 0;
            flat_divide(f, x, y, p);
        }
    }
    return clipped;
}

void FlatField::apply_cfa(uint16_t* cfa, size_t pitch, int width, int height, const uint8_t pattern[36]) const {
    if (width / 2 != m_half_w || height / 2 != m_half_h)
        throw std::invalid_argument("flat field was measured on a " + std::to_string(m_half_w) + "x" +
                                    std::to_string(m_half_h) + " half-size decode; this mosaic is " +
                                    std::to_string(width) + "x" + std::to_string(height));
    for (int k = 0; k < 36; ++k)
        if (pattern[k] > 2) throw std::invalid_argument("flat field: CFA colour index must be 0, 1 or 2");
    const Axis ax = axis_table(width, false, m_block, m_gw);
    const Axis ay = axis_table(height, false, m_block, m_gh);
    #pragma omp parallel
    {
        std::vector<float> row((size_t)m_gw * 3);
        #pragma omp for schedule(static)
        for (int y = 0; y < height; ++y) {
            const float ty = ay.t[y];
            const float* g0 = &m_gain[(size_t)ay.i[y] * m_gw * 3];
            const float* g1 = g0 + (size_t)m_gw * 3;
            for (size_t k = 0; k < row.size(); ++k) row[k] = g0[k] + (g1[k] - g0[k]) * ty;
            const uint8_t* pat = pattern + (y % 6) * 6;
            uint16_t* p = cfa + (size_t)y * pitch;
            for (int x = 0; x < width; ++x) {
                const int c = pat[x % 6];
                const float* r0 = &row[(size_t)ax.i[x] * 3];
                const float gain = r0[c] + (r0[c + 3] - r0[c]) * ax.t[x];
                const float v = std::nearbyint(p[x] / gain);
                p[x] = (uint16_t)std::min(v, 65535.0f);
            }
        }
    }
}

namespace {

uint16_t rd_u16(const uint8_t* p) {
    return (uint16_t)(p[0] | (uint16_t(p[1]) << 8));
}

uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

std::map<std::string, std::vector<uint8_t>> read_zip_entries(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("cannot open flat field file: " + path);
    const std::streamsize sz = ifs.tellg();
    if (sz < 22) throw std::runtime_error("invalid zip/npz file: " + path);
    std::vector<uint8_t> buf((size_t)sz);
    ifs.seekg(0);
    if (!ifs.read(reinterpret_cast<char*>(buf.data()), sz))
        throw std::runtime_error("failed reading flat field file: " + path);

    size_t eocd = (size_t)sz - 22;
    while (true) {
        if (rd_u32(&buf[eocd]) == 0x06054b50u) break;
        if (eocd == 0 || (size_t)sz - eocd > 65557)
            throw std::runtime_error("zip EOCD not found in flat field file: " + path);
        --eocd;
    }
    const uint16_t num_entries = rd_u16(&buf[eocd + 10]);
    const uint32_t cd_offset = rd_u32(&buf[eocd + 16]);
    size_t pos = cd_offset;
    std::map<std::string, std::vector<uint8_t>> out;
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (pos + 46 > buf.size() || rd_u32(&buf[pos]) != 0x02014b50u)
            throw std::runtime_error("corrupt zip central directory in: " + path);
        const uint16_t method = rd_u16(&buf[pos + 10]);
        const uint32_t comp_sz = rd_u32(&buf[pos + 20]);
        const uint32_t uncomp_sz = rd_u32(&buf[pos + 24]);
        const uint16_t name_len = rd_u16(&buf[pos + 28]);
        const uint16_t extra_len = rd_u16(&buf[pos + 30]);
        const uint16_t comment_len = rd_u16(&buf[pos + 32]);
        const uint32_t lhoff = rd_u32(&buf[pos + 42]);
        if (pos + 46 + name_len > buf.size())
            throw std::runtime_error("corrupt zip entry name in: " + path);
        std::string name(reinterpret_cast<const char*>(&buf[pos + 46]), name_len);
        pos += 46 + name_len + extra_len + comment_len;

        if (lhoff + 30 > buf.size() || rd_u32(&buf[lhoff]) != 0x04034b50u)
            throw std::runtime_error("corrupt zip local header in: " + path);
        const uint16_t lname_len = rd_u16(&buf[lhoff + 26]);
        const uint16_t lextra_len = rd_u16(&buf[lhoff + 28]);
        const size_t data_off = (size_t)lhoff + 30 + lname_len + lextra_len;
        if (data_off + comp_sz > buf.size())
            throw std::runtime_error("corrupt zip entry payload in: " + path);

        std::vector<uint8_t> data(uncomp_sz);
        if (method == 0) {
            if (comp_sz != uncomp_sz) throw std::runtime_error("stored zip entry size mismatch in: " + path);
            std::memcpy(data.data(), &buf[data_off], uncomp_sz);
        } else if (method == 8) {
            z_stream zs{};
            zs.next_in = const_cast<Bytef*>(&buf[data_off]);
            zs.avail_in = comp_sz;
            zs.next_out = data.data();
            zs.avail_out = uncomp_sz;
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK)
                throw std::runtime_error("zlib inflateInit2 failed for: " + path);
            const int ret = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (ret != Z_STREAM_END || zs.total_out != uncomp_sz)
                throw std::runtime_error("zlib inflate failed for entry " + name + " in: " + path);
        } else {
            throw std::runtime_error("unsupported zip compression method in: " + path);
        }
        out[std::move(name)] = std::move(data);
    }
    return out;
}

struct NpyHeader {
    std::string descr;
    bool fortran_order = false;
    std::vector<int> shape;
    size_t data_offset = 0;
};

NpyHeader parse_npy_header(const std::vector<uint8_t>& raw, const std::string& name) {
    if (raw.size() < 10 || std::memcmp(raw.data(), "\x93NUMPY", 6) != 0)
        throw std::runtime_error("invalid .npy magic in " + name);
    const uint8_t major = raw[6];
    size_t hdr_len = 0, hdr_off = 0;
    if (major == 1) {
        hdr_len = rd_u16(&raw[8]);
        hdr_off = 10;
    } else if (major == 2 || major == 3) {
        if (raw.size() < 12) throw std::runtime_error("truncated .npy v2 header in " + name);
        hdr_len = rd_u32(&raw[8]);
        hdr_off = 12;
    } else {
        throw std::runtime_error("unsupported .npy version in " + name);
    }
    if (hdr_off + hdr_len > raw.size())
        throw std::runtime_error("truncated .npy header in " + name);
    std::string hdr(reinterpret_cast<const char*>(&raw[hdr_off]), hdr_len);
    NpyHeader h;
    h.data_offset = hdr_off + hdr_len;

    std::smatch m;
    if (std::regex_search(hdr, m, std::regex("'descr'\\s*:\\s*'([^']+)'"))) h.descr = m[1].str();
    if (std::regex_search(hdr, m, std::regex("'fortran_order'\\s*:\\s*(True|False)")))
        h.fortran_order = (m[1].str() == "True");
    if (std::regex_search(hdr, m, std::regex("'shape'\\s*:\\s*\\(([^\\)]*)\\)"))) {
        std::string s = m[1].str();
        std::regex num_re("(\\d+)");
        for (std::sregex_iterator it(s.begin(), s.end(), num_re), end; it != end; ++it)
            h.shape.push_back(std::stoi((*it)[1].str()));
    }
    if (h.fortran_order) throw std::runtime_error("Fortran-order .npy not supported in " + name);
    return h;
}

}  // namespace

FlatFieldPtr load_flat_field_file(const std::string& path) {
    auto entries = read_zip_entries(path);
    auto it_gain = entries.find("gain.npy");
    auto it_info = entries.find("info.npy");
    if (it_gain == entries.end() || it_info == entries.end())
        throw std::runtime_error("flat field .npz missing gain.npy or info.npy: " + path);

    const NpyHeader ghdr = parse_npy_header(it_gain->second, "gain.npy");
    if ((ghdr.descr != "<f4" && ghdr.descr != "f4") || ghdr.shape.size() != 3 || ghdr.shape[2] != 3)
        throw std::runtime_error("gain.npy must be float32 [grid_h, grid_w, 3] in: " + path);
    const int gh = ghdr.shape[0], gw = ghdr.shape[1];
    const size_t n_floats = (size_t)gh * gw * 3;
    if (ghdr.data_offset + n_floats * sizeof(float) > it_gain->second.size())
        throw std::runtime_error("gain.npy payload truncated in: " + path);
    std::vector<float> gain(n_floats);
    std::memcpy(gain.data(), it_gain->second.data() + ghdr.data_offset, n_floats * sizeof(float));

    const NpyHeader ihdr = parse_npy_header(it_info->second, "info.npy");
    const uint8_t* ip = it_info->second.data() + ihdr.data_offset;
    const size_t ibytes = it_info->second.size() - ihdr.data_offset;
    std::string info_str;
    if (ihdr.descr.find('U') != std::string::npos) {
        for (size_t k = 0; k + 4 <= ibytes; k += 4) {
            const uint32_t cp = rd_u32(ip + k);
            if (cp == 0) break;
            if (cp < 0x80) {
                info_str.push_back((char)cp);
            } else if (cp < 0x800) {
                info_str.push_back((char)(0xC0 | (cp >> 6)));
                info_str.push_back((char)(0x80 | (cp & 0x3F)));
            } else {
                info_str.push_back((char)(0xE0 | (cp >> 12)));
                info_str.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                info_str.push_back((char)(0x80 | (cp & 0x3F)));
            }
        }
    } else {
        info_str.assign(reinterpret_cast<const char*>(ip), ibytes);
    }

    const auto j = nlohmann::json::parse(info_str);
    const int block = j.at("block").get<int>();
    const auto shape = j.at("shape");
    if (!shape.is_array() || shape.size() != 2)
        throw std::runtime_error("info.npy shape must be [half_h, half_w] in: " + path);
    const int half_h = shape[0].get<int>();
    const int half_w = shape[1].get<int>();
    std::string fid = j.value("id", "");
    if (fid.empty()) {
        // Fallback to parent directory name if id is not embedded in info JSON
        size_t slash = path.find_last_of("/\\");
        if (slash != std::string::npos && slash > 0) {
            size_t prev = path.find_last_of("/\\", slash - 1);
            fid = path.substr(prev == std::string::npos ? 0 : prev + 1,
                              slash - (prev == std::string::npos ? 0 : prev + 1));
        }
    }
    return std::make_shared<const FlatField>(std::move(gain), gh, gw, block, half_w, half_h, fid);
}

}  // namespace negicc
