#include "rgb_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif

namespace negicc {

// The physical cores among the CPUs this process may run on (its affinity mask, grouped by the kernel's list of each
// core's hardware threads); where that cannot be read, OpenMP's count of available CPUs.
static int allowed_physical_cores() {
    const int cpus = omp_get_num_procs();  // the CPUs this process may run on
#ifdef __linux__
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return cpus;
    int cores = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        // The hardware threads of this CPU's core, as a list of numbers and ranges ("0,96" or "0-1"). The core is
        // counted at its lowest-numbered hardware thread that the process may run on.
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        std::FILE* f = std::fopen(path, "r");
        if (!f) return cpus;
        char list[256];
        const bool read = std::fgets(list, sizeof list, f) != nullptr;
        std::fclose(f);
        if (!read) return cpus;
        int first = -1;
        for (const char* p = list; *p && *p != '\n';) {
            char* end = nullptr;
            const long lo = std::strtol(p, &end, 10);
            if (end == p) return cpus;
            long hi = lo;
            if (*end == '-') {
                p = end + 1;
                hi = std::strtol(p, &end, 10);
                if (end == p) return cpus;
            }
            for (long s = lo; s <= hi && first < 0; ++s) {
                if (s >= 0 && s < CPU_SETSIZE && CPU_ISSET((int)s, &allowed)) first = (int)s;
            }
            p = *end == ',' ? end + 1 : end;
        }
        if (first == cpu) ++cores;
    }
    return cores > 0 ? cores : cpus;
#else
    return cpus;
#endif
}

int default_thread_count() { return std::min(kDefaultThreadCap, allowed_physical_cores()); }

int apply_default_thread_count() {
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(default_thread_count());
    return omp_get_max_threads();
}

namespace {

inline void normalize_rot_flips(int& rot_cw, bool& hflip, bool& vflip) {
    if (rot_cw % 90 != 0) {
        throw std::invalid_argument("rotation must be a multiple of 90 degrees");
    }
    rot_cw = ((rot_cw % 360) + 360) % 360;
    if (rot_cw == 180) {
        hflip = !hflip;
        vflip = !vflip;
        rot_cw = 0;
    }
}

inline void map_inverse_geometry_cpu(
    int x_dst, int y_dst, int dst_w, int dst_h,
    int rw, int rh, int rot_cw, bool hflip, bool vflip,
    int& x_rel, int& y_rel)
{
    normalize_rot_flips(rot_cw, hflip, vflip);
    const int x1 = hflip ? (dst_w - 1 - x_dst) : x_dst;
    const int y1 = vflip ? (dst_h - 1 - y_dst) : y_dst;

    if (rot_cw == 0) {
        x_rel = x1;
        y_rel = y1;
    } else if (rot_cw == 90) {
        x_rel = y1;
        y_rel = rh - 1 - x1;
    } else {  // 270
        x_rel = rw - 1 - y1;
        y_rel = x1;
    }
}

inline bool is_rggb_pattern_cpu(unsigned int filters) {
    const int c10 = bayer_color_at(1, 0, filters);
    return (bayer_color_at(0, 0, filters) == 0 &&
            bayer_color_at(0, 1, filters) == 1 &&
            (c10 == 1 || c10 == 3) &&
            bayer_color_at(1, 1, filters) == 2);
}

// Photosite (r, c) of an h x w mosaic, mirrored about the edge photosites outside it.
inline float get_px_refl_cpu(const uint16_t* bayer, int r, int c, int h, int w, size_t pitch) {
    int r_refl = (r < 0) ? -r : (r >= h ? 2 * (h - 1) - r : r);
    int c_refl = (c < 0) ? -c : (c >= w ? 2 * (w - 1) - c : c);
    r_refl = std::max(0, std::min(h - 1, r_refl));
    c_refl = std::max(0, std::min(w - 1, c_refl));
    return (float)bayer[(size_t)r_refl * pitch + c_refl];
}

// Green at photosite (r, c) of an RGGB mosaic: the sample itself on a green site; elsewhere the vertical and the
// horizontal Hamilton-Adams estimates (J. F. Hamilton and J. E. Adams, US patent 5,629,734, 1997: neighbour mean plus
// a quarter of the centre channel's Laplacian, steered by the green gradient plus that Laplacian), blended
// with weights 1 / (eps + gradient^2) instead of picking one. A large eps keeps grain-sized gradients from steering
// the blend; only real edges do.
inline float reconstruct_green_soft_cpu(const uint16_t* m, int r, int c, int h, int w, size_t pitch,
                                        float white, float eps) {
    if ((r + c) & 1) {
        return get_px_refl_cpu(m, r, c, h, w, pitch);
    }
    const float c_center = get_px_refl_cpu(m, r, c, h, w, pitch);
    const float g_up     = get_px_refl_cpu(m, r - 1, c, h, w, pitch);
    const float g_down   = get_px_refl_cpu(m, r + 1, c, h, w, pitch);
    const float g_left   = get_px_refl_cpu(m, r, c - 1, h, w, pitch);
    const float g_right  = get_px_refl_cpu(m, r, c + 1, h, w, pitch);

    const float c_up2    = get_px_refl_cpu(m, r - 2, c, h, w, pitch);
    const float c_down2  = get_px_refl_cpu(m, r + 2, c, h, w, pitch);
    const float c_left2  = get_px_refl_cpu(m, r, c - 2, h, w, pitch);
    const float c_right2 = get_px_refl_cpu(m, r, c + 2, h, w, pitch);

    const float dv = std::abs(g_up - g_down) + std::abs(2.0f * c_center - c_up2 - c_down2);
    const float dh = std::abs(g_left - g_right) + std::abs(2.0f * c_center - c_left2 - c_right2);

    const float g_v = 0.5f * (g_up + g_down) + 0.25f * (2.0f * c_center - c_up2 - c_down2);
    const float g_h = 0.5f * (g_left + g_right) + 0.25f * (2.0f * c_center - c_left2 - c_right2);

    const float wv = 1.0f / (eps + dv * dv);
    const float wh = 1.0f / (eps + dh * dh);
    const float g_est = (wv * g_v + wh * g_h) / (wv + wh);
    return std::max(0.0f, std::min(white, g_est));
}

struct RawStatTables {
    double log_lut[65536];
    uint8_t ev_bin[65536];
};

const RawStatTables& raw_stat_tables() {
    static RawStatTables t;
    static std::once_flag once;
    std::call_once(once, [] {
        for (int v = 0; v < 65536; ++v) {
            const double dv = std::max((double)v, 1.0);
            t.log_lut[v] = std::log(dv);
            // Stops below 16384 (2^14), in 160 bins of 1/16 stop over [-10, 0), the ends clamped.
            const double ev = std::log2(dv / 16384.0);
            const int bin = (int)((ev + 10.0) * 16.0);
            t.ev_bin[v] = (uint8_t)std::clamp(bin, 0, 159);
        }
    });
    return t;
}

// The q-quantile of a 16-bit histogram, interpolated linearly between ranks floor and ceil of q (n - 1): Hyndman and
// Fan's (1996) definition 7, numpy's default method "linear".
double u16_percentile_linear(const uint64_t hist[65536], uint64_t total, double q) {
    if (total == 0) return 0.0;
    const double pos = q * (double)(total - 1);
    const uint64_t idx_lo = (uint64_t)std::floor(pos);
    const uint64_t idx_hi = (uint64_t)std::ceil(pos);
    const double frac = pos - (double)idx_lo;

    uint64_t cum = 0;
    int v_lo = -1, v_hi = -1;
    for (int v = 0; v < 65536; ++v) {
        cum += hist[v];
        if (v_lo < 0 && cum > idx_lo) v_lo = v;
        if (v_hi < 0 && cum > idx_hi) {
            v_hi = v;
            break;
        }
    }
    if (v_lo < 0) v_lo = 65535;
    if (v_hi < 0) v_hi = 65535;
    return (1.0 - frac) * (double)v_lo + frac * (double)v_hi;
}

double percentile_from_hist_L(const uint32_t* hist, double tot, double q) {
    if (tot <= 0.0) return 0.0;
    const double target = tot * q;
    double cum = 0.0;
    int k = 0;
    for (; k < kLabBins; ++k) {
        cum += (double)hist[k];
        if (cum >= target) break;
    }
    return (std::min(k, kLabBins - 1) + 0.5) * (100.0 / kLabBins);
}

}  // namespace

// ---- RgbView -----------------------------------------------------------------------------------------------

RgbView RgbView::from_cfa_half(const uint16_t* cfa, size_t cfa_pitch, int sensor_w, int sensor_h,
                               unsigned int filters, const uint8_t xtrans_col36[36],
                               FlatSampleArgs flat_args,
                               std::shared_ptr<const FlatTables> flat_pin,
                               int cfa_x0, int cfa_y0) {
    RgbView v;
    v.flat = flat_args;
    v.flat_tables = std::move(flat_pin);
    // LibRaw's idata.filters: 9 is Fujifilm X-Trans (a 6x6 pattern, xtrans_col36), 0 no CFA and 1 Leaf's 16x16
    // pattern (neither read here); a Bayer mosaic is a 2x2 bitmask (above 1000, bayer_color_at).
    const int period = (filters == 9) ? 6 : 2;
    if (filters == 0 || filters == 1) return v;
    if (!cfa || sensor_w < period || sensor_h < period) return v;
    if (filters == 9 && !xtrans_col36) return v;

    v.width = sensor_w / period;
    v.height = sensor_h / period;
    v.pitch = period;
    v.stride = (ptrdiff_t)period * (ptrdiff_t)cfa_pitch;
    v.cfa_step = (uint8_t)period;
    // TODO: For X-Trans (`period == 6`), `sensor_dxx = 3, sensor_dyy = 3` with `sensor_x0 = 0, sensor_y0 = 0` samples
    // the half-size flat field at the top-left of each 6x6 cell `(3*x, 3*y)` rather than the cell center
    // `(3*x+1, 3*y+1)` or per-channel photosite offsets `(ch_x0, ch_y0)`, and reads only 1 green photosite
    // (`g2_ptr == nullptr`) per 6x6 cell. Evaluate centering the flat-field coordinate and multi-green averaging on
    // X-Trans in a follow-up.
    v.sensor_x0 = cfa_x0 / 2;
    v.sensor_y0 = cfa_y0 / 2;
    v.sensor_dxx = period / 2;
    v.sensor_dyy = period / 2;

    for (int pass = 0; pass < (period == 2 ? 2 : 1); ++pass) {
        for (int dr = 0; dr < period; ++dr) {
            for (int dc = 0; dc < period; ++dc) {
                const int col = (filters == 9) ? (int)(xtrans_col36[dr * 6 + dc] & 3)
                                               : bayer_color_at(dr, dc, filters);
                const uint16_t* p = cfa + (ptrdiff_t)dr * (ptrdiff_t)cfa_pitch + dc;
                if (pass == 0) {
                    if (col == 0 && !v.r_ptr) {
                        v.r_ptr = p;
                        v.ch_x0[0] = (uint8_t)dc;
                        v.ch_y0[0] = (uint8_t)dr;
                    } else if (col == 2 && !v.b_ptr) {
                        v.b_ptr = p;
                        v.ch_x0[2] = (uint8_t)dc;
                        v.ch_y0[2] = (uint8_t)dr;
                    } else if ((col == 1 || (period == 6 && col == 3)) && !v.g_ptr) {
                        v.g_ptr = p;
                        v.ch_x0[1] = (uint8_t)dc;
                        v.ch_y0[1] = (uint8_t)dr;
                    }
                } else if (col == 1 || col == 3) {
                    if (!v.g_ptr) {
                        v.g_ptr = p;
                        v.ch_x0[1] = (uint8_t)dc;
                        v.ch_y0[1] = (uint8_t)dr;
                    } else if (p != v.g_ptr && !v.g2_ptr) {
                        v.g2_ptr = p;
                    }
                }
            }
        }
    }
    if (!v.r_ptr || !v.g_ptr || !v.b_ptr) return {};
    return v;
}

RgbView RgbView::from_interleaved_rgb(const uint16_t* rgb, int w, int h, int channels,
                                      ptrdiff_t row_elems, FlatSampleArgs flat_args,
                                      std::shared_ptr<const FlatTables> flat_pin) {
    RgbView v;
    if (!rgb || w <= 0 || h <= 0 || channels < 3) return v;
    v.width = w;
    v.height = h;
    v.r_ptr = rgb;
    v.g_ptr = rgb + 1;
    v.g2_ptr = nullptr;
    v.b_ptr = rgb + 2;
    v.pitch = channels;
    v.stride = row_elems > 0 ? row_elems : (ptrdiff_t)w * channels;
    v.cfa_step = 1;
    v.flat = flat_args;
    v.flat_tables = std::move(flat_pin);
    return v;
}

RgbView RgbView::crop_and_orient(int rx, int ry, int rw, int rh, int rot_cw, bool hflip, bool vflip) const {
    normalize_rot_flips(rot_cw, hflip, vflip);
    if (!valid() || rw <= 0 || rh <= 0) return {};
    const int x0 = std::max(0, rx);
    const int y0 = std::max(0, ry);
    const int x1 = std::min(width, rx + rw);
    const int y1 = std::min(height, ry + rh);
    if (x1 <= x0 || y1 <= y0) return {};
    rx = x0;
    ry = y0;
    rw = x1 - x0;
    rh = y1 - y0;
    const bool swap_dims = (rot_cw == 90 || rot_cw == 270);
    const int dst_w = swap_dims ? rh : rw;
    const int dst_h = swap_dims ? rw : rh;

    int ox = rx, oy = ry;
    int ux = 1, uy = 0;
    int vx = 0, vy = 1;

    int ox0 = ox, oy0 = oy, ux0 = ux, uy0 = uy, vx0 = vx, vy0 = vy;
    if (rot_cw == 90) {
        ox0 = ox + (rh - 1) * vx;
        oy0 = oy + (rh - 1) * vy;
        ux0 = -vx;
        uy0 = -vy;
        vx0 = ux;
        vy0 = uy;
    } else if (rot_cw == 270) {
        ox0 = ox + (rw - 1) * ux;
        oy0 = oy + (rw - 1) * uy;
        ux0 = vx;
        uy0 = vy;
        vx0 = -ux;
        vy0 = -uy;
    }

    if (hflip) {
        ox0 += (dst_w - 1) * ux0;
        oy0 += (dst_w - 1) * uy0;
        ux0 = -ux0;
        uy0 = -uy0;
    }
    if (vflip) {
        ox0 += (dst_h - 1) * vx0;
        oy0 += (dst_h - 1) * vy0;
        vx0 = -vx0;
        vy0 = -vy0;
    }

    RgbView out = *this;
    out.width = dst_w;
    out.height = dst_h;
    const ptrdiff_t base_off = (ptrdiff_t)oy0 * stride + (ptrdiff_t)ox0 * pitch;
    out.r_ptr = r_ptr + base_off;
    out.g_ptr = g_ptr + base_off;
    out.g2_ptr = g2_ptr ? (g2_ptr + base_off) : nullptr;
    out.b_ptr = b_ptr + base_off;
    out.pitch = (ptrdiff_t)uy0 * stride + (ptrdiff_t)ux0 * pitch;
    out.stride = (ptrdiff_t)vy0 * stride + (ptrdiff_t)vx0 * pitch;
    out.sensor_x0 = sensor_x0 + ox0 * sensor_dxx + oy0 * sensor_dyx;
    out.sensor_y0 = sensor_y0 + ox0 * sensor_dxy + oy0 * sensor_dyy;
    out.sensor_dxx = ux0 * sensor_dxx + uy0 * sensor_dyx;
    out.sensor_dxy = ux0 * sensor_dxy + uy0 * sensor_dyy;
    out.sensor_dyx = vx0 * sensor_dxx + vy0 * sensor_dyx;
    out.sensor_dyy = vx0 * sensor_dxy + vy0 * sensor_dyy;
    return out;
}

// Used by negicc-station's _cfa_half_view test binding; returns an empty FilmImage (step == 0) when the view has an
// attached flat field or non-identity orientation/crop.
FilmImage RgbView::to_film_image(float white_level, int step) const {
    FilmImage img;
    img.step = 0;
    if (!valid()) return img;
    // FilmImage is a direct un-oriented strided view over raw channel pointers without flat-field tables or 2-green
    // averaging; refuse views with an attached flat field or non-identity crop/orientation.
    if (flat.gain != nullptr || !is_unoriented()) {
        return img;
    }
    const int s = (step >= 2) ? (int)cfa_step : 1;
    img.width = width * s;
    img.height = height * s;
    img.step = s;
    img.white = std::max(1.0f, white_level);
    const uint16_t* ptrs[3] = {r_ptr, g_ptr, b_ptr};
    for (int c = 0; c < 3; ++c) {
        img.ch[c].data = ptrs[c];
        img.ch[c].pitch = pitch;
        img.ch[c].stride = stride;
        img.ch[c].x0 = (s > 1) ? (int)ch_x0[c] : 0;
        img.ch[c].y0 = (s > 1) ? (int)ch_y0[c] : 0;
        img.ch[c].black = 0.0f;
    }
    return img;
}

void RgbView::materialize(uint16_t* dst_rgb) const {
    if (!valid() || !dst_rgb) return;
    #pragma omp parallel for schedule(static) if ((size_t)width * height > 8192)
    for (int y = 0; y < height; ++y) {
        uint16_t* d = dst_rgb + (size_t)y * width * 3;
        for (int x = 0; x < width; ++x, d += 3) {
            sample(x, y, d);
        }
    }
}

// ---- compute_raw_stats -------------------------------------------------------------------------------------

void compute_raw_stats(const RgbView& view, float crop_ratio, int max_size, int patch_size,
                       uint16_t* dino_dst, uint16_t* grid_dst, RawStatsSummary& out) {
    out = RawStatsSummary{};
    if (!view.valid()) return;

    const int W = view.width;
    const int H = view.height;
    const InnerCrop crop = compute_inner_crop(W, H, crop_ratio);
    const AspectPreservedShape shape = compute_aspect_preserved_shape(crop.w, crop.h, max_size, patch_size);
    const int dino_w = shape.target_w, dino_h = shape.target_h;
    const int grid_w = shape.grid_w, grid_h = shape.grid_h;
    const int psize = std::max(1, patch_size);

    out.dino_w = dino_w;
    out.dino_h = dino_h;
    out.grid_w = grid_w;
    out.grid_h = grid_h;

    const RawStatTables& tbl = raw_stat_tables();
    const bool is_contig = view.is_contiguous_rgb();

    std::vector<uint64_t> g_hist(65536, 0);
    struct RowLog { double r = 0, g = 0, b = 0; };
    std::vector<RowLog> row_logs(std::max(0, crop.h));
    std::mutex merge_mtx;

    #pragma omp parallel if ((size_t)crop.w * crop.h > 8192)
    {
        std::vector<uint32_t> loc_g(65536, 0);
        uint32_t loc_raw[3][160] = {};

        #pragma omp for schedule(static)
        for (int ry = 0; ry < crop.h; ++ry) {
            const int sy = crop.y0 + ry;
            double lr = 0.0, lg = 0.0, lb = 0.0;
            if (is_contig) {
                const uint16_t* row = view.r_ptr + (ptrdiff_t)sy * view.stride + (ptrdiff_t)crop.x0 * 3;
                for (int rx = 0; rx < crop.w; ++rx, row += 3) {
                    const uint16_t r = row[0], g = row[1], b = row[2];
                    loc_g[g]++;
                    lr += tbl.log_lut[r];
                    lg += tbl.log_lut[g];
                    lb += tbl.log_lut[b];
                    loc_raw[0][tbl.ev_bin[r]]++;
                    loc_raw[1][tbl.ev_bin[g]]++;
                    loc_raw[2][tbl.ev_bin[b]]++;
                }
            } else {
                for (int rx = 0; rx < crop.w; ++rx) {
                    uint16_t px[3];
                    view.sample(crop.x0 + rx, sy, px);
                    const uint16_t r = px[0], g = px[1], b = px[2];
                    loc_g[g]++;
                    lr += tbl.log_lut[r];
                    lg += tbl.log_lut[g];
                    lb += tbl.log_lut[b];
                    loc_raw[0][tbl.ev_bin[r]]++;
                    loc_raw[1][tbl.ev_bin[g]]++;
                    loc_raw[2][tbl.ev_bin[b]]++;
                }
            }
            row_logs[ry] = {lr, lg, lb};
        }

        std::lock_guard<std::mutex> mlock(merge_mtx);
        for (int v = 0; v < 65536; ++v) g_hist[v] += loc_g[v];
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 160; ++k) out.hist_raw[c][k] += loc_raw[c][k];
    }

    const uint64_t npx = (uint64_t)crop.w * (uint64_t)crop.h;
    if (npx > 0) {
        double tot_lr = 0.0, tot_lg = 0.0, tot_lb = 0.0;
        for (const auto& rl : row_logs) {
            tot_lr += rl.r;
            tot_lg += rl.g;
            tot_lb += rl.b;
        }
        const double inv_n = 1.0 / (double)npx;
        out.log_avg_rgb[0] = std::exp(tot_lr * inv_n) / 65535.0;
        out.log_avg_rgb[1] = std::exp(tot_lg * inv_n) / 65535.0;
        out.log_avg_rgb[2] = std::exp(tot_lb * inv_n) / 65535.0;

        const double p_lo = u16_percentile_linear(g_hist.data(), npx, 0.005);
        const double p_hi = u16_percentile_linear(g_hist.data(), npx, 0.995);
        out.g_lo = (float)std::max(1.0, p_lo);
        out.g_hi = (float)std::max((double)out.g_lo + 1.0, p_hi);
    }

    if (!dino_dst && !grid_dst) return;

    std::vector<double> dino_f64((size_t)dino_w * dino_h * 3, 0.0);
    const double scale_x = (double)crop.w / (double)dino_w;
    const double scale_y = (double)crop.h / (double)dino_h;

    if (scale_x >= 1.0 && scale_y >= 1.0) {
        #pragma omp parallel for schedule(static) if ((size_t)dino_w * dino_h > 4096)
        for (int dy = 0; dy < dino_h; ++dy) {
            const double y0 = dy * scale_y;
            const double y1 = (dy + 1) * scale_y;
            const int sy_start = (int)std::floor(y0);
            const int sy_end = std::min(crop.h, (int)std::ceil(y1));

            for (int dx = 0; dx < dino_w; ++dx) {
                const double x0 = dx * scale_x;
                const double x1 = (dx + 1) * scale_x;
                const int sx_start = (int)std::floor(x0);
                const int sx_end = std::min(crop.w, (int)std::ceil(x1));

                double r_acc = 0.0, g_acc = 0.0, b_acc = 0.0, w_acc = 0.0;
                for (int sy = sy_start; sy < sy_end; ++sy) {
                    const double wy = std::min(y1, (double)(sy + 1)) - std::max(y0, (double)sy);
                    if (wy <= 0.0) continue;
                    if (is_contig) {
                        const uint16_t* row = view.r_ptr + (ptrdiff_t)(crop.y0 + sy) * view.stride +
                                              (ptrdiff_t)(crop.x0 + sx_start) * 3;
                        for (int sx = sx_start; sx < sx_end; ++sx, row += 3) {
                            const double wx = std::min(x1, (double)(sx + 1)) - std::max(x0, (double)sx);
                            if (wx <= 0.0) continue;
                            const double w = wx * wy;
                            r_acc += row[0] * w;
                            g_acc += row[1] * w;
                            b_acc += row[2] * w;
                            w_acc += w;
                        }
                    } else {
                        for (int sx = sx_start; sx < sx_end; ++sx) {
                            const double wx = std::min(x1, (double)(sx + 1)) - std::max(x0, (double)sx);
                            if (wx <= 0.0) continue;
                            const double w = wx * wy;
                            uint16_t px[3];
                            view.sample(crop.x0 + sx, crop.y0 + sy, px);
                            r_acc += px[0] * w;
                            g_acc += px[1] * w;
                            b_acc += px[2] * w;
                            w_acc += w;
                        }
                    }
                }
                const double inv_w = (w_acc > 0.0) ? (1.0 / w_acc) : 0.0;
                const double rm = r_acc * inv_w, gm = g_acc * inv_w, bm = b_acc * inv_w;
                const size_t didx = ((size_t)dy * dino_w + dx) * 3;
                dino_f64[didx]     = rm;
                dino_f64[didx + 1] = gm;
                dino_f64[didx + 2] = bm;
                if (dino_dst) {
                    dino_dst[didx]     = (uint16_t)std::clamp(std::round(rm), 0.0, 65535.0);
                    dino_dst[didx + 1] = (uint16_t)std::clamp(std::round(gm), 0.0, 65535.0);
                    dino_dst[didx + 2] = (uint16_t)std::clamp(std::round(bm), 0.0, 65535.0);
                }
            }
        }
    } else {
        // An axis that enlarges (scale < 1): per axis, an area average where it shrinks and a clamped bilinear pair
        // where it enlarges (resample_taps).
        const auto ty = resample_taps<double>(crop.h, dino_h, scale_y);
        const auto tx = resample_taps<double>(crop.w, dino_w, scale_x);
        #pragma omp parallel for schedule(static) if ((size_t)dino_w * dino_h > 4096)
        for (int dy = 0; dy < dino_h; ++dy) {
            for (int dx = 0; dx < dino_w; ++dx) {
                double acc[3] = {0.0, 0.0, 0.0}, wsum = 0.0;
                for (const auto& [sy, wy] : ty[(size_t)dy]) {
                    for (const auto& [sx, wx] : tx[(size_t)dx]) {
                        uint16_t px[3];
                        view.sample(crop.x0 + sx, crop.y0 + sy, px);
                        const double w = wx * wy;
                        for (int c = 0; c < 3; ++c) acc[c] += px[c] * w;
                        wsum += w;
                    }
                }
                const double inv = wsum > 0.0 ? 1.0 / wsum : 0.0;
                const size_t didx = ((size_t)dy * dino_w + dx) * 3;
                for (int c = 0; c < 3; ++c) {
                    const double val = acc[c] * inv;
                    dino_f64[didx + c] = val;
                    if (dino_dst) dino_dst[didx + c] = (uint16_t)std::clamp(std::round(val), 0.0, 65535.0);
                }
            }
        }
    }

    if (grid_dst) {
        const double inv_patch = 1.0 / (double)(psize * psize);
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                double pr = 0.0, pg = 0.0, pb = 0.0;
                for (int py = 0; py < psize; ++py) {
                    const double* drow = &dino_f64[((size_t)(gy * psize + py) * dino_w + gx * psize) * 3];
                    for (int px = 0; px < psize; ++px, drow += 3) {
                        pr += drow[0];
                        pg += drow[1];
                        pb += drow[2];
                    }
                }
                const size_t gidx = ((size_t)gy * grid_w + gx) * 3;
                grid_dst[gidx]     = (uint16_t)std::clamp(std::round(pr * inv_patch), 0.0, 65535.0);
                grid_dst[gidx + 1] = (uint16_t)std::clamp(std::round(pg * inv_patch), 0.0, 65535.0);
                grid_dst[gidx + 2] = (uint16_t)std::clamp(std::round(pb * inv_patch), 0.0, 65535.0);
            }
        }
    }
}

// ---- Resampling & ROI operations ---------------------------------------------------------------------------

void resample_view_rgb16(const RgbView& src, uint16_t* dst, int dst_w, int dst_h) {
    if (!src.valid() || !dst || dst_w <= 0 || dst_h <= 0) return;
    const int src_w = src.width;
    const int src_h = src.height;
    const bool is_contig = src.is_contiguous_rgb();
    if (src_w == dst_w && src_h == dst_h) {
        if (is_contig && src.stride == (ptrdiff_t)src_w * 3) {
            std::memcpy(dst, src.r_ptr, (size_t)src_w * src_h * 3 * sizeof(uint16_t));
        } else {
            src.materialize(dst);
        }
        return;
    }

    const double scale_x = (double)src_w / (double)dst_w;
    const double scale_y = (double)src_h / (double)dst_h;

    if (scale_x >= 1.0 && scale_y >= 1.0) {
        #pragma omp parallel for schedule(static) if ((size_t)dst_w * dst_h > 8192)
        for (int dy = 0; dy < dst_h; ++dy) {
            const double y0 = dy * scale_y;
            const double y1 = (dy + 1) * scale_y;
            const int sy_start = (int)std::floor(y0);
            const int sy_end = std::min(src_h, (int)std::ceil(y1));

            for (int dx = 0; dx < dst_w; ++dx) {
                const double x0 = dx * scale_x;
                const double x1 = (dx + 1) * scale_x;
                const int sx_start = (int)std::floor(x0);
                const int sx_end = std::min(src_w, (int)std::ceil(x1));

                double r_acc = 0.0, g_acc = 0.0, b_acc = 0.0, w_acc = 0.0;
                for (int sy = sy_start; sy < sy_end; ++sy) {
                    const double wy = std::min(y1, (double)(sy + 1)) - std::max(y0, (double)sy);
                    if (wy <= 0.0) continue;
                    if (is_contig) {
                        const uint16_t* row = src.r_ptr + (ptrdiff_t)sy * src.stride + (ptrdiff_t)sx_start * 3;
                        for (int sx = sx_start; sx < sx_end; ++sx, row += 3) {
                            const double wx = std::min(x1, (double)(sx + 1)) - std::max(x0, (double)sx);
                            if (wx <= 0.0) continue;
                            const double w = wx * wy;
                            r_acc += row[0] * w;
                            g_acc += row[1] * w;
                            b_acc += row[2] * w;
                            w_acc += w;
                        }
                    } else {
                        for (int sx = sx_start; sx < sx_end; ++sx) {
                            const double wx = std::min(x1, (double)(sx + 1)) - std::max(x0, (double)sx);
                            if (wx <= 0.0) continue;
                            const double w = wx * wy;
                            uint16_t px[3];
                            src.sample(sx, sy, px);
                            r_acc += px[0] * w;
                            g_acc += px[1] * w;
                            b_acc += px[2] * w;
                            w_acc += w;
                        }
                    }
                }
                const double inv_w = (w_acc > 0.0) ? (1.0 / w_acc) : 0.0;
                const size_t dst_idx = ((size_t)dy * dst_w + dx) * 3;
                dst[dst_idx]     = (uint16_t)std::clamp(std::round(r_acc * inv_w), 0.0, 65535.0);
                dst[dst_idx + 1] = (uint16_t)std::clamp(std::round(g_acc * inv_w), 0.0, 65535.0);
                dst[dst_idx + 2] = (uint16_t)std::clamp(std::round(b_acc * inv_w), 0.0, 65535.0);
            }
        }
        return;
    }

    #pragma omp parallel for schedule(static) if ((size_t)dst_w * dst_h > 8192)
    for (int dy = 0; dy < dst_h; ++dy) {
        const float sy = ((float)dy + 0.5f) * (float)scale_y - 0.5f;
        const int y0 = std::clamp((int)std::floor(sy), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float wy1 = sy - (float)y0;
        const float wy0 = 1.0f - wy1;

        for (int dx = 0; dx < dst_w; ++dx) {
            const float sx = ((float)dx + 0.5f) * (float)scale_x - 0.5f;
            const int x0 = std::clamp((int)std::floor(sx), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float wx1 = sx - (float)x0;
            const float wx0 = 1.0f - wx1;

            uint16_t p00[3], p01[3], p10[3], p11[3];
            src.sample(x0, y0, p00);
            src.sample(x1, y0, p01);
            src.sample(x0, y1, p10);
            src.sample(x1, y1, p11);
            for (int c = 0; c < 3; ++c) {
                const float top = p00[c] * wx0 + p01[c] * wx1;
                const float bot = p10[c] * wx0 + p11[c] * wx1;
                const float val = top * wy0 + bot * wy1;
                dst[((size_t)dy * dst_w + dx) * 3 + c] = (uint16_t)std::clamp(std::round(val), 0.0f, 65535.0f);
            }
        }
    }
}

void extract_and_transform_roi(
    const uint16_t* src, uint16_t* dst,
    int src_w, int src_h,
    int rx, int ry, int rw, int rh,
    int rot_cw, bool hflip, bool vflip,
    int dst_w, int dst_h
) {
    RgbView v = RgbView::from_interleaved_rgb(src, src_w, src_h, 3)
        .crop_and_orient(rx, ry, rw, rh, rot_cw, hflip, vflip);
    if (!v.valid() || v.width != dst_w || v.height != dst_h) {
        throw std::invalid_argument("extract_and_transform_roi: ROI dimensions do not match dst_w x dst_h");
    }
    v.materialize(dst);
}

void resample_linear_rgb16(
    const uint16_t* src, int src_w, int src_h,
    uint16_t* dst, int dst_w, int dst_h
) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;
    if (src_w == dst_w && src_h == dst_h) {
        std::memcpy(dst, src, (size_t)src_w * src_h * 3 * sizeof(uint16_t));
        return;
    }
    resample_view_rgb16(RgbView::from_interleaved_rgb(src, src_w, src_h, 3), dst, dst_w, dst_h);
}

// ---- CPU Render & Scoring Kernels --------------------------------------------------------------------------

void (*cpu_pass_hook)(int kind) = nullptr;

#ifndef NEGICC_USE_CUDA
// A build without CUDA: no device, and the GPU passes refuse (gpu_passes.cu has them with USE_CUDA=1).
bool gpu_device_available() { return false; }
[[noreturn]] static void no_cuda() { throw GpuError("this neg_process was built without CUDA (USE_CUDA=0)"); }
void* gpu_malloc(size_t) { no_cuda(); }
void gpu_free(void*) {}
void gpu_upload(void*, const void*, size_t) { no_cuda(); }
void gpu_download(void*, const void*, size_t) { no_cuda(); }
void gpu_render(const RenderArgs&) { no_cuda(); }
void gpu_highlight_clip(const HighlightClipArgs&) { no_cuda(); }
void gpu_preview_debayer(const PreviewDebayerArgs&) { no_cuda(); }
void gpu_grain_aware_debayer(const GrainAwareDebayerArgs&) { no_cuda(); }
void gpu_pixel_shift_merge(const PixelShiftMergeArgs&) { no_cuda(); }
void gpu_grain_aware_debayer_grid(const GrainAwareDebayerArgs&, int, int, int, int, int) { no_cuda(); }
void gpu_pixel_shift_merge_grid(const PixelShiftMergeArgs&, int, int, int, int, int) { no_cuda(); }
#endif

void cpu_render(const RenderArgs& a) {
    if (cpu_pass_hook) cpu_pass_hook(0);
    const long long n = (long long)a.width * a.height;
    const bool use_view = (a.in_view != nullptr && a.in_view->valid());
    if (!a.stats) {
        if (!use_view) {
            #pragma omp parallel for schedule(static) if (n > 8192)
            for (long long i = 0; i < n; ++i) {
                const Rgb v = convert_pixel(a.cfg, a.matrix, a.in + i * 3);
                if (a.out_type == 1) {
                    uint8_t* o = static_cast<uint8_t*>(a.out) + i * 3;
                    o[0] = quantize_u8(v.r); o[1] = quantize_u8(v.g); o[2] = quantize_u8(v.b);
                } else if (a.out_type == 0) {
                    uint16_t* o = static_cast<uint16_t*>(a.out) + i * 3;
                    o[0] = quantize_u16(v.r); o[1] = quantize_u16(v.g); o[2] = quantize_u16(v.b);
                } else {
                    float* o = static_cast<float*>(a.out) + i * 3;
                    o[0] = v.r; o[1] = v.g; o[2] = v.b;
                }
            }
        } else {
            #pragma omp parallel for schedule(static) if (n > 8192)
            for (int y = 0; y < a.height; ++y) {
                for (int x = 0; x < a.width; ++x) {
                    uint16_t px[3];
                    a.in_view->sample(x, y, px);
                    const Rgb v = convert_pixel(a.cfg, a.matrix, px);
                    const long long i = (long long)y * a.width + x;
                    if (a.out_type == 1) {
                        uint8_t* o = static_cast<uint8_t*>(a.out) + i * 3;
                        o[0] = quantize_u8(v.r); o[1] = quantize_u8(v.g); o[2] = quantize_u8(v.b);
                    } else if (a.out_type == 0) {
                        uint16_t* o = static_cast<uint16_t*>(a.out) + i * 3;
                        o[0] = quantize_u16(v.r); o[1] = quantize_u16(v.g); o[2] = quantize_u16(v.b);
                    } else {
                        float* o = static_cast<float*>(a.out) + i * 3;
                        o[0] = v.r; o[1] = v.g; o[2] = v.b;
                    }
                }
            }
        }
        return;
    }

    if (a.stats) *a.stats = PositiveStats{};
    const bool tone = a.cfg.encoding == ENCODE_SRGB && a.cfg.tone.lut && a.cfg.tone.n > 0;
    const InnerCrop crop = pass_crop(a.stats_crop, a.width, a.height);
    const int x0 = crop.x0, x1 = crop.x0 + crop.w;
    const int y0 = crop.y0, y1 = crop.y0 + crop.h;

    struct RowSums {
        double mid_a = 0, mid_b = 0;
        double hi_a = 0, hi_b = 0;
        double sh_a = 0, sh_b = 0;
        double all_a = 0, all_b = 0;
    };
    std::vector<RowSums> row_sums(std::max(0, crop.h));

    uint64_t total_white_clip = 0, total_black_clip = 0;
    uint64_t total_mid_n = 0, total_hi_n = 0, total_sh_n = 0;
    std::mutex merge_mtx;

    #pragma omp parallel if (n > 8192)
    {
        uint32_t loc_hist_rgb[3][256] = {};
        uint32_t loc_hist_L[kLabBins] = {};
        uint64_t loc_wc = 0, loc_bc = 0;
        uint64_t loc_mid_n = 0, loc_hi_n = 0, loc_sh_n = 0;

        #pragma omp for schedule(static)
        for (int y = 0; y < a.height; ++y) {
            const bool in_y = (y >= y0 && y < y1);
            RowSums rsum{};
            for (int x = 0; x < a.width; ++x) {
                const long long i = (long long)y * a.width + x;
                uint16_t px_buf[3];
                const uint16_t* px = px_buf;
                if (use_view) {
                    a.in_view->sample(x, y, px_buf);
                } else {
                    px = a.in + i * 3;
                }
                const Rgb lin = convert_pixel_linear(a.cfg, a.matrix, px);  // pre-shoulder: clip rules
                const Rgb disp = to_display(a.cfg, lin);                    // post-shoulder: Lab
                // The picture and its RGB histogram go through the tone stage when cfg has one, as convert_pixel
                // applies it; the clip rules and the Lab statistics read the values before it.
                const Rgb v = encode_output(tone ? apply_tone(a.cfg.tone, disp) : disp, a.cfg.encoding);
                const uint8_t r8 = quantize_u8(v.r);
                const uint8_t g8 = quantize_u8(v.g);
                const uint8_t b8 = quantize_u8(v.b);
                if (a.out_type == 1) {
                    uint8_t* o = static_cast<uint8_t*>(a.out) + i * 3;
                    o[0] = r8; o[1] = g8; o[2] = b8;
                } else if (a.out_type == 0) {
                    uint16_t* o = static_cast<uint16_t*>(a.out) + i * 3;
                    o[0] = quantize_u16(v.r); o[1] = quantize_u16(v.g); o[2] = quantize_u16(v.b);
                } else {
                    float* o = static_cast<float*>(a.out) + i * 3;
                    o[0] = v.r; o[1] = v.g; o[2] = v.b;
                }

                if (in_y && x >= x0 && x < x1) {
                    loc_hist_rgb[0][r8]++;
                    loc_hist_rgb[1][g8]++;
                    loc_hist_rgb[2][b8]++;
                    if (linear_channel_white_clipped(lin.r) ||
                        linear_channel_white_clipped(lin.g) ||
                        linear_channel_white_clipped(lin.b)) {
                        loc_wc++;
                    }
                    if (linear_channel_black_clipped(lin.r) &&
                        linear_channel_black_clipped(lin.g) &&
                        linear_channel_black_clipped(lin.b)) {
                        loc_bc++;
                    }
                    const Lab lab = linear_srgb_to_lab(disp);
                    loc_hist_L[lab_bin(lab.L)]++;
                    rsum.all_a += lab.a; rsum.all_b += lab.b;
                    if (lab.L >= 20.0f && lab.L <= 75.0f) { rsum.mid_a += lab.a; rsum.mid_b += lab.b; loc_mid_n++; }
                    if (lab.L >= 75.0f && lab.L <= 95.0f) { rsum.hi_a += lab.a; rsum.hi_b += lab.b; loc_hi_n++; }
                    if (lab.L >= 5.0f && lab.L <= 25.0f)  { rsum.sh_a += lab.a; rsum.sh_b += lab.b; loc_sh_n++; }
                }
            }
            if (in_y) row_sums[y - y0] = rsum;
        }

        std::lock_guard<std::mutex> lock(merge_mtx);
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 256; ++k) a.stats->hist_rgb[c][k] += loc_hist_rgb[c][k];
        for (int k = 0; k < kLabBins; ++k) a.stats->hist_L[k] += loc_hist_L[k];
        total_white_clip += loc_wc;
        total_black_clip += loc_bc;
        total_mid_n += loc_mid_n;
        total_hi_n += loc_hi_n;
        total_sh_n += loc_sh_n;
    }

    RowSums tot_sums{};
    for (const auto& r : row_sums) {
        tot_sums.mid_a += r.mid_a; tot_sums.mid_b += r.mid_b;
        tot_sums.hi_a += r.hi_a;   tot_sums.hi_b += r.hi_b;
        tot_sums.sh_a += r.sh_a;   tot_sums.sh_b += r.sh_b;
        tot_sums.all_a += r.all_a; tot_sums.all_b += r.all_b;
    }
    const double tot = (double)crop.w * crop.h;
    if (tot > 0.0) {
        a.stats->white_clip = (float)((double)total_white_clip / tot * 100.0);
        a.stats->black_clip = (float)((double)total_black_clip / tot * 100.0);
        const double ma = total_mid_n > 0 ? tot_sums.mid_a / total_mid_n : tot_sums.all_a / tot;
        const double mb = total_mid_n > 0 ? tot_sums.mid_b / total_mid_n : tot_sums.all_b / tot;
        const double ha = total_hi_n > 0 ? tot_sums.hi_a / total_hi_n : ma;
        const double hb = total_hi_n > 0 ? tot_sums.hi_b / total_hi_n : mb;
        const double sa = total_sh_n > 0 ? tot_sums.sh_a / total_sh_n : ma;
        const double sb = total_sh_n > 0 ? tot_sums.sh_b / total_sh_n : mb;
        a.stats->mid_a = (float)ma; a.stats->mid_b = (float)mb;
        a.stats->hi_a  = (float)ha; a.stats->hi_b  = (float)hb;
        a.stats->sh_a  = (float)sa; a.stats->sh_b  = (float)sb;
    }
}

void cpu_intent_loss(const IntentLossArgs& a) {
    const long long npx = (long long)a.width * a.height;
    const bool use_view = (a.in_view != nullptr && a.in_view->valid());
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < a.n; ++c) {
        const float* m = a.matrices + (size_t)c * 9;
        double sL = 0.0, sa = 0.0, sb = 0.0, clip = 0.0;
        for (long long i = 0; i < npx; ++i) {
            uint16_t px_buf[3];
            const uint16_t* px = a.in ? (a.in + i * 3) : nullptr;
            if (use_view) {
                a.in_view->sample((int)(i % a.width), (int)(i / a.width), px_buf);
                px = px_buf;
            }
            const Rgb lin = convert_pixel_linear(a.cfg, m, px);
            clip += linear_channel_white_clipped(lin.r) +
                    linear_channel_white_clipped(lin.g) +
                    linear_channel_white_clipped(lin.b);
            const Lab lab = linear_srgb_to_lab(to_display(a.cfg, lin));
            const float dL = (lab.L - a.mu[i * 3]) / a.sigma[i * 3];
            const float da = (lab.a - a.mu[i * 3 + 1]) / a.sigma[i * 3 + 1];
            const float db = (lab.b - a.mu[i * 3 + 2]) / a.sigma[i * 3 + 2];
            sL += dL * dL; sa += da * da; sb += db * db;
        }
        float* o = a.out + (size_t)c * 4;
        o[0] = (float)(sL / npx);
        o[1] = (float)(sa / npx);
        o[2] = (float)(sb / npx);
        o[3] = (float)(clip / (3.0 * npx) * 100.0);
    }
}

void cpu_highlight_clip(const HighlightClipArgs& a) {
    if (cpu_pass_hook) cpu_pass_hook(1);
    if (a.n <= 0) return;
    const bool use_view = (a.in_view != nullptr && a.in_view->valid());
    const InnerCrop crop = pass_crop(a.crop, a.width, a.height);
    const long long npx = (long long)crop.w * crop.h;
    if (npx <= 0) {
        std::memset(a.out, 0, (size_t)a.n * 5 * sizeof(float));
        return;
    }

    struct CandAccum {
        uint64_t exc_r = 0, exc_g = 0, exc_b = 0, wc = 0;
        uint32_t hist_L[kLabBins] = {};
    };
    std::vector<CandAccum> totals(a.n);
    std::mutex merge_mtx;

    const float cr = a.ceil_lin[0], cg = a.ceil_lin[1], cb = a.ceil_lin[2];
    const int x0 = crop.x0, x1 = crop.x0 + crop.w;
    const int y0 = crop.y0, y1 = crop.y0 + crop.h;

    std::vector<uint64_t> total_max_hist;
    if (a.max_hist) total_max_hist.assign((size_t)a.n * kMaxHistBins, 0);

    #pragma omp parallel
    {
        std::vector<CandAccum> local(a.n);
        std::vector<uint64_t> local_max_hist;
        if (a.max_hist) local_max_hist.assign((size_t)a.n * kMaxHistBins, 0);
        #pragma omp for collapse(2) schedule(static)
        for (int c = 0; c < a.n; ++c) {
            for (int y = y0; y < y1; ++y) {
                const float* m = a.matrices + (size_t)c * 9;
                CandAccum& acc = local[c];
                if (!use_view) {
                    const uint16_t* row = a.in + ((long long)y * a.width + x0) * 3;
                    for (int x = x0; x < x1; ++x, row += 3) {
                        const Rgb lin = convert_pixel_linear(a.cfg, m, row);
                        if (lin.r >= cr) acc.exc_r++;
                        if (lin.g >= cg) acc.exc_g++;
                        if (lin.b >= cb) acc.exc_b++;
                        if (linear_channel_white_clipped(lin.r) ||
                            linear_channel_white_clipped(lin.g) ||
                            linear_channel_white_clipped(lin.b)) {
                            acc.wc++;
                        }
                        const float L = linear_srgb_to_L(to_display(a.cfg, lin));
                        acc.hist_L[lab_bin(L)]++;
                        if (a.max_hist) {
                            const float mc = std::max(lin.r, std::max(lin.g, lin.b));
                            local_max_hist[(size_t)c * kMaxHistBins + max_hist_bin(mc)]++;
                        }
                    }
                } else {
                    for (int x = x0; x < x1; ++x) {
                        uint16_t px[3];
                        a.in_view->sample(x, y, px);
                        const Rgb lin = convert_pixel_linear(a.cfg, m, px);
                        if (lin.r >= cr) acc.exc_r++;
                        if (lin.g >= cg) acc.exc_g++;
                        if (lin.b >= cb) acc.exc_b++;
                        if (linear_channel_white_clipped(lin.r) ||
                            linear_channel_white_clipped(lin.g) ||
                            linear_channel_white_clipped(lin.b)) {
                            acc.wc++;
                        }
                        const float L = linear_srgb_to_L(to_display(a.cfg, lin));
                        acc.hist_L[lab_bin(L)]++;
                        if (a.max_hist) {
                            const float mc = std::max(lin.r, std::max(lin.g, lin.b));
                            local_max_hist[(size_t)c * kMaxHistBins + max_hist_bin(mc)]++;
                        }
                    }
                }
            }
        }
        std::lock_guard<std::mutex> lock(merge_mtx);
        for (int c = 0; c < a.n; ++c) {
            totals[c].exc_r += local[c].exc_r;
            totals[c].exc_g += local[c].exc_g;
            totals[c].exc_b += local[c].exc_b;
            totals[c].wc    += local[c].wc;
            for (int k = 0; k < kLabBins; ++k) totals[c].hist_L[k] += local[c].hist_L[k];
        }
        for (size_t k = 0; k < local_max_hist.size(); ++k) total_max_hist[k] += local_max_hist[k];
    }
    if (a.max_hist) std::memcpy(a.max_hist, total_max_hist.data(), total_max_hist.size() * sizeof(uint64_t));

    const double tot = (double)npx;
    for (int c = 0; c < a.n; ++c) {
        float* o = a.out + (size_t)c * 5;
        o[0] = (float)((double)totals[c].exc_r / tot * 100.0);
        o[1] = (float)((double)totals[c].exc_g / tot * 100.0);
        o[2] = (float)((double)totals[c].exc_b / tot * 100.0);
        o[3] = (float)((double)totals[c].wc    / tot * 100.0);
        o[4] = (float)percentile_from_hist_L(totals[c].hist_L, tot, 0.50);
    }
}

void cpu_lab_stats(const LabStatsArgs& a) {
    const long long npx = (long long)a.width * a.height;
    const bool use_view = (a.in_view != nullptr && a.in_view->valid());
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < a.n; ++c) {
        const float* m = a.matrices + (size_t)c * 9;
        LabStatsRow& row = a.out[c];
        std::memset(&row, 0, sizeof(row));
        std::vector<long long> hist(kLabBins, 0);
        for (long long i = 0; i < npx; ++i) {
            uint16_t px_buf[3];
            const uint16_t* px = a.in ? (a.in + i * 3) : nullptr;
            if (use_view) {
                a.in_view->sample((int)(i % a.width), (int)(i / a.width), px_buf);
                px = px_buf;
            }
            const Rgb lin = convert_pixel_linear(a.cfg, m, px);
            if (linear_channel_white_clipped(lin.r) ||
                linear_channel_white_clipped(lin.g) ||
                linear_channel_white_clipped(lin.b)) row.white_clip += 1;
            if (linear_channel_black_clipped(lin.r) &&
                linear_channel_black_clipped(lin.g) &&
                linear_channel_black_clipped(lin.b)) row.black_clip += 1;
            const Lab lab = linear_srgb_to_lab(to_display(a.cfg, lin));
            hist[lab_bin(lab.L)]++;
            row.all_a += lab.a; row.all_b += lab.b;
            if (lab.L >= 20.0f && lab.L <= 75.0f) { row.mid_a += lab.a; row.mid_b += lab.b; row.mid_n += 1; }
            if (lab.L >= 75.0f && lab.L <= 95.0f) { row.hi_a += lab.a; row.hi_b += lab.b; row.hi_n += 1; }
            if (lab.L >= 5.0f && lab.L <= 25.0f) { row.sh_a += lab.a; row.sh_b += lab.b; row.sh_n += 1; }
        }
        for (int k = 0; k < kLabBins; ++k) row.hist[k] = (float)hist[k];
        row.total = (double)npx;
    }
}

// ---- CPU Debayer & Pixel-Shift Kernels ---------------------------------------------------------------------

void cpu_preview_debayer(const PreviewDebayerArgs& a) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("cpu_preview_debayer: rot_cw must be a multiple of 90");
    }
    const int half_rw = a.rw / 2;
    const int half_rh = a.rh / 2;
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;

    #pragma omp parallel for schedule(static)
    for (int y2 = 0; y2 < a.dst_h; ++y2) {
        for (int x2 = 0; x2 < a.dst_w; ++x2) {
            int x_rel = 0, y_rel = 0;
            map_inverse_geometry_cpu(x2, y2, a.dst_w, a.dst_h, half_rw, half_rh, a.rot_cw, a.hflip, a.vflip,
                                     x_rel, y_rel);

            const int r = a.ry + y_rel * 2;
            const int c = a.rx + x_rel * 2;

            const uint16_t s00 = (uint16_t)get_px_refl_cpu(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
            const uint16_t s01 = (uint16_t)get_px_refl_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch);
            const uint16_t s10 = (uint16_t)get_px_refl_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch);
            const uint16_t s11 = (uint16_t)get_px_refl_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch);

            const int col00 = bayer_color_at(r, c, a.filters);
            const int col01 = bayer_color_at(r, c + 1, a.filters);
            const int col10 = bayer_color_at(r + 1, c, a.filters);
            const int col11 = bayer_color_at(r + 1, c + 1, a.filters);

            uint16_t R = 0, B = 0;
            uint32_t G_sum = 0;
            const uint16_t samples[4] = {s00, s01, s10, s11};
            const int cols[4] = {col00, col01, col10, col11};
            for (int k = 0; k < 4; ++k) {
                if (cols[k] & 1) G_sum += samples[k];
                else if (cols[k] == 0) R = samples[k];
                else B = samples[k];
            }
            const uint16_t G = (uint16_t)(G_sum >> 1);

            uint16_t px[3] = {R, G, B};
            if (a.flat.gain) flat_divide(a.flat, a.flat_x0 + x_rel, a.flat_y0 + y_rel, px);

            const long long dst_idx = ((long long)y2 * a.dst_w + x2) * 3;
            a.rgb_out[dst_idx + 0] = px[0];
            a.rgb_out[dst_idx + 1] = px[1];
            a.rgb_out[dst_idx + 2] = px[2];
        }
    }
}

// One output pixel (x2, y2) of the grain-aware debayer: its photosite after the inverse crop, rotation and flips, the
// soft green and the ratio-interpolated red and blue from the 7 x 7 CFA neighbourhood (reflected at the sensor's
// edges), rounded and clamped to 16 bits, then the flat field at that pixel. cpu_grain_aware_debayer and
// cpu_grain_aware_debayer_sampled both compute their pixels with it.
static inline void grain_aware_pixel_cpu(const GrainAwareDebayerArgs& a, float k, float eps, size_t pitch, int x2,
                                         int y2, uint16_t px_out[3]) {
    int x_rel = 0, y_rel = 0;
    map_inverse_geometry_cpu(x2, y2, a.dst_w, a.dst_h, a.rw, a.rh, a.rot_cw, a.hflip, a.vflip, x_rel, y_rel);

    const int c = a.rx + x_rel;
    const int r = a.ry + y_rel;

    const float G = reconstruct_green_soft_cpu(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch, a.white_level, eps);

    const bool r_even = ((r & 1) == 0);
    const bool c_even = ((c & 1) == 0);

    float R = 0.0f;
    float B = 0.0f;

    const float c_center = get_px_refl_cpu(a.cfa_bayer, r, c, a.src_h, a.src_w, pitch);
    const bool near_clip = (c_center >= (a.white_level - 64.0f)) || (G >= (a.white_level - 64.0f));

    if (near_clip) {
        if (r_even && c_even) {
            R = c_center;
            const float b00 = get_px_refl_cpu(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch);
            const float b01 = get_px_refl_cpu(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch);
            const float b10 = get_px_refl_cpu(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch);
            const float b11 = get_px_refl_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch);
            B = 0.25f * (b00 + b01 + b10 + b11);
        } else if (!r_even && !c_even) {
            B = c_center;
            const float r00 = get_px_refl_cpu(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch);
            const float r01 = get_px_refl_cpu(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch);
            const float r10 = get_px_refl_cpu(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch);
            const float r11 = get_px_refl_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch);
            R = 0.25f * (r00 + r01 + r10 + r11);
        } else if (r_even && !c_even) {
            R = 0.5f * (get_px_refl_cpu(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) +
                        get_px_refl_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch));
            B = 0.5f * (get_px_refl_cpu(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) +
                        get_px_refl_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch));
        } else {
            R = 0.5f * (get_px_refl_cpu(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) +
                        get_px_refl_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch));
            B = 0.5f * (get_px_refl_cpu(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) +
                        get_px_refl_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch));
        }
    } else {
        if (r_even && c_even) {
            R = c_center;
        } else if (r_even && !c_even) {
            const float g_l = reconstruct_green_soft_cpu(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g_r = reconstruct_green_soft_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float rl = (get_px_refl_cpu(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) + k) / (g_l + k);
            const float rr = (get_px_refl_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch) + k) / (g_r + k);
            R = (G + k) * (0.5f * (rl + rr)) - k;
        } else if (!r_even && c_even) {
            const float g_u = reconstruct_green_soft_cpu(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g_d = reconstruct_green_soft_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float ru = (get_px_refl_cpu(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) + k) / (g_u + k);
            const float rd = (get_px_refl_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch) + k) / (g_d + k);
            R = (G + k) * (0.5f * (ru + rd)) - k;
        } else {
            const float g00 = reconstruct_green_soft_cpu(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g01 = reconstruct_green_soft_cpu(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g10 = reconstruct_green_soft_cpu(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g11 = reconstruct_green_soft_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float r00 = (get_px_refl_cpu(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch) + k) /
                              (g00 + k);
            const float r01 = (get_px_refl_cpu(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch) + k) /
                              (g01 + k);
            const float r10 = (get_px_refl_cpu(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch) + k) /
                              (g10 + k);
            const float r11 = (get_px_refl_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch) + k) /
                              (g11 + k);
            R = (G + k) * (0.25f * (r00 + r01 + r10 + r11)) - k;
        }

        if (!r_even && !c_even) {
            B = c_center;
        } else if (!r_even && c_even) {
            const float g_l = reconstruct_green_soft_cpu(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g_r = reconstruct_green_soft_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float bl = (get_px_refl_cpu(a.cfa_bayer, r, c - 1, a.src_h, a.src_w, pitch) + k) / (g_l + k);
            const float br = (get_px_refl_cpu(a.cfa_bayer, r, c + 1, a.src_h, a.src_w, pitch) + k) / (g_r + k);
            B = (G + k) * (0.5f * (bl + br)) - k;
        } else if (r_even && !c_even) {
            const float g_u = reconstruct_green_soft_cpu(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g_d = reconstruct_green_soft_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float bu = (get_px_refl_cpu(a.cfa_bayer, r - 1, c, a.src_h, a.src_w, pitch) + k) / (g_u + k);
            const float bd = (get_px_refl_cpu(a.cfa_bayer, r + 1, c, a.src_h, a.src_w, pitch) + k) / (g_d + k);
            B = (G + k) * (0.5f * (bu + bd)) - k;
        } else {
            const float g00 = reconstruct_green_soft_cpu(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g01 = reconstruct_green_soft_cpu(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g10 = reconstruct_green_soft_cpu(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float g11 = reconstruct_green_soft_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch,
                                                         a.white_level, eps);
            const float b00 = (get_px_refl_cpu(a.cfa_bayer, r - 1, c - 1, a.src_h, a.src_w, pitch) + k) /
                              (g00 + k);
            const float b01 = (get_px_refl_cpu(a.cfa_bayer, r - 1, c + 1, a.src_h, a.src_w, pitch) + k) /
                              (g01 + k);
            const float b10 = (get_px_refl_cpu(a.cfa_bayer, r + 1, c - 1, a.src_h, a.src_w, pitch) + k) /
                              (g10 + k);
            const float b11 = (get_px_refl_cpu(a.cfa_bayer, r + 1, c + 1, a.src_h, a.src_w, pitch) + k) /
                              (g11 + k);
            B = (G + k) * (0.25f * (b00 + b01 + b10 + b11)) - k;
        }
    }

    uint16_t px[3] = {(uint16_t)std::round(std::max(0.0f, std::min(65535.0f, R))),
                      (uint16_t)std::round(std::max(0.0f, std::min(65535.0f, G))),
                      (uint16_t)std::round(std::max(0.0f, std::min(65535.0f, B)))};
    if (a.flat.gain) flat_divide(a.flat, a.flat_x0 + x_rel, a.flat_y0 + y_rel, px);
    px_out[0] = px[0];
    px_out[1] = px[1];
    px_out[2] = px[2];
}

void cpu_grain_aware_debayer(const GrainAwareDebayerArgs& a) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("cpu_grain_aware_debayer: rot_cw must be a multiple of 90");
    }
    if (!is_rggb_pattern_cpu(a.filters)) {
        throw std::invalid_argument("cpu_grain_aware_debayer: currently only supports RGGB Bayer pattern");
    }
    const float k = 16.0f;
    const float eps = (a.eps > 0.0f) ? a.eps : 4096.0f;
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;

    #pragma omp parallel for schedule(static)
    for (int y2 = 0; y2 < a.dst_h; ++y2) {
        for (int x2 = 0; x2 < a.dst_w; ++x2) {
            uint16_t px[3];
            grain_aware_pixel_cpu(a, k, eps, pitch, x2, y2, px);
            const long long dst_idx = ((long long)y2 * a.dst_w + x2) * 3;
            a.rgb_out[dst_idx + 0] = px[0];
            a.rgb_out[dst_idx + 1] = px[1];
            a.rgb_out[dst_idx + 2] = px[2];
        }
    }
}

void cpu_grain_aware_debayer_sampled(const GrainAwareDebayerArgs& a, const int32_t* xy, long long n) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("cpu_grain_aware_debayer_sampled: rot_cw must be a multiple of 90");
    }
    if (!is_rggb_pattern_cpu(a.filters)) {
        throw std::invalid_argument("cpu_grain_aware_debayer_sampled: currently only supports RGGB Bayer pattern");
    }
    const float k = 16.0f;
    const float eps = (a.eps > 0.0f) ? a.eps : 4096.0f;
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i)
        grain_aware_pixel_cpu(a, k, eps, pitch, xy[2 * i], xy[2 * i + 1], a.rgb_out + i * 3);
}

// One output pixel (x2, y2) of the pixel-shift merge: the four frames' samples at its photosite and the three
// beside it, clipped to the white level, then the flat field at that pixel. cpu_pixel_shift_merge and
// cpu_pixel_shift_merge_sampled both compute their pixels with it.
static inline void pixel_shift_pixel_cpu(const PixelShiftMergeArgs& a, size_t pitch, int x2, int y2,
                                         uint16_t px_out[3]) {
    int x_rel = 0, y_rel = 0;
    map_inverse_geometry_cpu(x2, y2, a.dst_w, a.dst_h, a.rw, a.rh, a.rot_cw, a.hflip, a.vflip, x_rel, y_rel);

    const int c = a.rx + x_rel;
    const int r = a.ry + y_rel;

    const uint16_t s0 = (uint16_t)get_px_refl_cpu(a.bayer_f0, r,     c,     a.src_h, a.src_w, pitch);
    const uint16_t s1 = (uint16_t)get_px_refl_cpu(a.bayer_f1, r - 1, c,     a.src_h, a.src_w, pitch);
    const uint16_t s2 = (uint16_t)get_px_refl_cpu(a.bayer_f2, r - 1, c + 1, a.src_h, a.src_w, pitch);
    const uint16_t s3 = (uint16_t)get_px_refl_cpu(a.bayer_f3, r,     c + 1, a.src_h, a.src_w, pitch);

    const uint16_t samples[4] = {s0, s1, s2, s3};
    const int coords[4][2] = {{r, c}, {r - 1, c}, {r - 1, c + 1}, {r, c + 1}};

    uint16_t R = 0, B = 0;
    uint32_t G_sum = 0;
    int g_count = 0;
    bool g_clipped = false;
    for (int i = 0; i < 4; ++i) {
        const int col = bayer_color_at(coords[i][0], coords[i][1], a.filters);
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
    px_out[0] = px[0];
    px_out[1] = px[1];
    px_out[2] = px[2];
}

void cpu_pixel_shift_merge(const PixelShiftMergeArgs& a) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("cpu_pixel_shift_merge: rot_cw must be a multiple of 90");
    }
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;
    #pragma omp parallel for schedule(static)
    for (int y2 = 0; y2 < a.dst_h; ++y2) {
        for (int x2 = 0; x2 < a.dst_w; ++x2) {
            uint16_t px[3];
            pixel_shift_pixel_cpu(a, pitch, x2, y2, px);
            const long long dst_idx = ((long long)y2 * a.dst_w + x2) * 3;
            a.rgb_out[dst_idx + 0] = px[0];
            a.rgb_out[dst_idx + 1] = px[1];
            a.rgb_out[dst_idx + 2] = px[2];
        }
    }
}

long long sampled_grid_count(int step, int cx0, int cy0, int cx1, int cy1) {
    auto below = [step](int t) { return 2LL * (t / step) + (t % step < 2 ? t % step : 2); };  // sampled in [0, t)
    const long long cols = below(cx1) - below(cx0), rows = below(cy1) - below(cy0);
    return cols > 0 && rows > 0 ? cols * rows : 0;
}

void map_frame_to_sensor(int rx, int ry, int rw, int rh, int rot_cw, bool hflip, bool vflip, int dst_w, int dst_h,
                         const int32_t* xy, long long n, int32_t* rc) {
    for (long long i = 0; i < n; ++i) {
        int x_rel = 0, y_rel = 0;
        map_inverse_geometry_cpu(xy[2 * i], xy[2 * i + 1], dst_w, dst_h, rw, rh, rot_cw, hflip, vflip, x_rel, y_rel);
        rc[2 * i] = ry + y_rel;
        rc[2 * i + 1] = rx + x_rel;
    }
}

void cpu_pixel_shift_merge_sampled(const PixelShiftMergeArgs& a, const int32_t* xy, long long n) {
    if (a.rot_cw % 90 != 0) {
        throw std::invalid_argument("cpu_pixel_shift_merge_sampled: rot_cw must be a multiple of 90");
    }
    const size_t pitch = a.cfa_pitch > 0 ? a.cfa_pitch : (size_t)a.src_w;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) pixel_shift_pixel_cpu(a, pitch, xy[2 * i], xy[2 * i + 1], a.rgb_out + i * 3);
}

}  // namespace negicc
