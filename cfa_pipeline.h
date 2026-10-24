// Shared zero-copy CFA downsized view (RgbView) and CPU pipeline kernels used by neg_process and negicc-station.
//
// Five-step pipeline architecture:
//   Step 1 (Load & decode): InputFormat enum (CFA Bayer/X-Trans, 4xCFA Sony pixel shift, and TODOs for RGB/3xCFA/3xMono),
//     per-channel black subtraction in-place on the CFA mosaic, keeping at most 1 CFA in memory at a time.
//   Step 2 (Downsizing): Zero-copy half-size RgbView over the CFA mosaic (r_ptr, g_ptr, g2_ptr, b_ptr, pitch, stride).
//   Step 3 (Crop, rotate, flip & inline ROI search): detect_film_frame operates inline on the Step 2 view;
//     crop_and_orient adjusts pointers, pitch, stride, and sensor flat-field coordinates with zero memory copy.
//   Step 4 (Conversion parameter search): Operates inline on the oriented RgbView (compute_raw_stats, cpu_render,
//     cpu_highlight_clip) with zero large intermediate copies.
//   Step 5 (Full export): The only step that allocates a cropped full-size RGB buffer, fusing grain-aware debayer
//     (or 1-frame-at-a-time pixel-shift merge), crop/rotate/flip, and flat_divide.
//   TODO: Upstream the matching CUDA kernels (kernels_cuda.cu) from negicc-station into negicc.
#ifndef NEGICC_CFA_PIPELINE_H
#define NEGICC_CFA_PIPELINE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "color_pipeline.h"
#include "film_frame_detector.h"
#include "flat_field.h"
#include "flat_sample.h"

namespace negicc {

// Step 1 input formats:
//   1. CFA 2x2 Bayer or 3x3 (6x6) X-Trans
//   2. 4x CFA for Sony multishot (pixel shift)
//   3. TODO RGB
//   4. TODO 3xCFA -> R, G, B
//   5. TODO 3xmonochrome -> R, G, B
enum class InputFormat : int {
    CFA_BAYER_OR_XTRANS = 1,
    SONY_PIXEL_SHIFT_4 = 2,
    TODO_RGB = 3,
    TODO_3X_CFA = 4,
    TODO_3X_MONO = 5,
};

inline int bayer_color_at(int r, int c, unsigned int filters) {
    return (filters >> (((r << 1 & 14) + (c & 1)) << 1)) & 3;
}

// Step 2 & Step 3 zero-copy image view.
// Points directly into a black-subtracted CFA mosaic (half-size 2x2 cells) or an RGB buffer, with arbitrary crop,
// 90-degree rotation, and horizontal/vertical flip represented via (r_ptr, g_ptr, g2_ptr, b_ptr, pitch, stride)
// and an affine sensor coordinate mapping for inline flat-field division.
struct RgbView {
    int width = 0;
    int height = 0;
    const uint16_t* r_ptr = nullptr;
    const uint16_t* g_ptr = nullptr;
    // TODO: read only one green photosite (g_ptr) for later optimization; g2_ptr is kept to average the two greens
    // for bit-exact parity with cpu_preview_debayer.
    const uint16_t* g2_ptr = nullptr;
    const uint16_t* b_ptr = nullptr;
    ptrdiff_t pitch = 0;   // step in uint16_t elements when x increases by 1
    ptrdiff_t stride = 0;  // step in uint16_t elements when y increases by 1

    // X-Trans 6x6 half-size fallback (zero-copy over the CFA mosaic when filters == 9).
    bool is_xtrans_cfa = false;
    const uint16_t* cfa_base = nullptr;
    size_t cfa_pitch = 0;
    int cfa_w = 0, cfa_h = 0;
    uint8_t xtrans_col[36] = {};
    uint8_t ch_x0[3] = {0, 0, 0};
    uint8_t ch_y0[3] = {0, 0, 0};

    // Affine mapping from view pixel (x, y) to unoriented half-size sensor pixel (fx, fy):
    //   fx = sensor_x0 + x * sensor_dxx + y * sensor_dyx
    //   fy = sensor_y0 + x * sensor_dxy + y * sensor_dyy
    int sensor_x0 = 0, sensor_y0 = 0;
    int sensor_dxx = 1, sensor_dxy = 0;
    int sensor_dyx = 0, sensor_dyy = 1;

    FlatSampleArgs flat{};
    std::shared_ptr<const FlatTables> flat_tables;

    bool valid() const { return width > 0 && height > 0 && (r_ptr != nullptr || is_xtrans_cfa); }

    // Construct a zero-copy half-size view over a black-subtracted CFA mosaic (Bayer 2x2 or X-Trans 6x6).
    // For 4xCFA (SONY_PIXEL_SHIFT_4), pass the first CFA mosaic.
    static RgbView from_cfa_half(const uint16_t* cfa, size_t cfa_pitch, int sensor_w, int sensor_h,
                                 unsigned int filters, const uint8_t xtrans_col36[36] = nullptr,
                                 FlatSampleArgs flat = {},
                                 std::shared_ptr<const FlatTables> flat_pin = nullptr);

    static RgbView from_cfa_half(const uint16_t* cfa, int sensor_w, int sensor_h,
                                 unsigned int filters, FlatSampleArgs flat = {},
                                 std::shared_ptr<const FlatTables> flat_pin = nullptr) {
        return from_cfa_half(cfa, (size_t)sensor_w, sensor_w, sensor_h, filters, nullptr, flat, std::move(flat_pin));
    }

    // Construct a zero-copy view over an interleaved 16-bit RGB (channels=3) or RGBA (channels=4) buffer.
    // TODO RGB - Subsampled: cover InputFormat::TODO_RGB, TODO_3X_CFA, TODO_3X_MONO by subsampling with
    // pitch = 2 * channels and stride = 2 * row_elems.
    static RgbView from_interleaved_rgb(const uint16_t* rgb, int w, int h, int channels = 3,
                                        ptrdiff_t row_elems = 0, FlatSampleArgs flat = {},
                                        std::shared_ptr<const FlatTables> flat_pin = nullptr);

    // Step 3: Apply crop (rx, ry, rw, rh in this view's coordinates), clockwise rotation (0, 90, 180, 270),
    // and flips (hflip, vflip) by adjusting pointers, pitch, stride, and sensor affine coordinates (zero copy).
    RgbView crop_and_orient(int rx, int ry, int rw, int rh, int rot_cw, bool hflip, bool vflip) const;

    // Sample pixel (x, y) in [0, width) x [0, height) into px[0..2] (including inline flat_divide if flat is set).
    inline void sample(int x, int y, uint16_t px[3]) const {
        const int fx = sensor_x0 + x * sensor_dxx + y * sensor_dyx;
        const int fy = sensor_y0 + x * sensor_dxy + y * sensor_dyy;
        if (!is_xtrans_cfa) {
            const ptrdiff_t off = (ptrdiff_t)y * stride + (ptrdiff_t)x * pitch;
            px[0] = r_ptr[off];
            px[1] = g2_ptr ? (uint16_t)(((uint32_t)g_ptr[off] + (uint32_t)g2_ptr[off]) >> 1) : g_ptr[off];
            px[2] = b_ptr[off];
        } else {
            sample_xtrans_quad(fx, fy, px);
        }
        if (flat.gain) {
            flat_divide(flat, fx, fy, px);
        }
    }

    // Zero-copy conversion to FilmImage for inline detect_film_frame in Step 3.
    // step=1 returns half-size coordinates (width x height); step=2 returns sensor coordinates (2*width x 2*height).
    FilmImage to_film_image(float white_level = 65535.0f, int step = 1) const;

    // Materialize into a contiguous width * height * 3 uint16_t buffer (only when explicitly requested).
    void materialize(uint16_t* dst_rgb) const;

private:
    void sample_xtrans_quad(int hx, int hy, uint16_t px[3]) const;
};

// Pixels of an RgbView with any channel >= clip_level.
size_t count_clipped_view(const RgbView& view, uint16_t clip_level);

// Summary produced by compute_raw_stats (shared by FrameView::raw_stats and multi_target_optimizer).
struct RawStatsSummary {
    int dino_w = 0, dino_h = 0;
    int grid_w = 0, grid_h = 0;
    float g_lo = 1.0f;
    float g_hi = 65535.0f;
    double log_avg_rgb[3] = {0.0, 0.0, 0.0};
    uint32_t hist_raw[3][160] = {};
};

// Computes exact 16-bit green percentiles (0.5%, 99.5%), geometric mean RGB, 160-bin EV histogram, and resamples
// the inner crop inline from `view` into `dino_dst` (dino_w x dino_h x 3, optional: may be null if not needed)
// and `grid_dst` (grid_w x grid_h x 3).
void compute_raw_stats(const RgbView& view, float crop_ratio, int max_size, int patch_size,
                       uint16_t* dino_dst, uint16_t* grid_dst, RawStatsSummary& out);

// Resample an RgbView directly into a contiguous dst_w x dst_h x 3 uint16_t buffer (box-average downsampling
// when scale >= 1, bilinear interpolation when upsampling).
void resample_view_rgb16(const RgbView& src, uint16_t* dst, int dst_w, int dst_h);

void extract_and_transform_roi(
    const uint16_t* src, uint16_t* dst,
    int src_w, int src_h,
    int rx, int ry, int rw, int rh,
    int rot_cw, bool hflip, bool vflip,
    int dst_w, int dst_h
);

void resample_linear_rgb16(
    const uint16_t* src, int src_w, int src_h,
    uint16_t* dst, int dst_w, int dst_h
);

// Positive image statistics accumulated over an inner crop during render.
struct PositiveStats {
    uint32_t hist_rgb[3][256] = {};
    uint32_t hist_L[kLabBins] = {};
    float white_clip = 0.0f;          // % of inner-crop pixels with any channel white-clipped
    float black_clip = 0.0f;          // % of inner-crop pixels with all channels black-clipped
    float mid_a = 0.0f, mid_b = 0.0f; // zone means (with empty-zone fallback applied)
    float hi_a  = 0.0f, hi_b  = 0.0f;
    float sh_a  = 0.0f, sh_b  = 0.0f;
};

struct RenderArgs {
    const uint16_t* in = nullptr;
    const RgbView* in_view = nullptr;  // When non-null, reads directly from in_view (zero copy)
    int width = 0, height = 0;
    const float* matrix = nullptr;     // 9 floats
    PipelineConfig cfg{};
    int out_type = 0;                  // 0 = u16, 1 = u8, 2 = f32
    void* out = nullptr;
    PositiveStats* stats = nullptr;
    InnerCrop stats_crop = {};
};

struct IntentLossArgs {
    const uint16_t* in = nullptr;
    const RgbView* in_view = nullptr;  // When non-null, reads directly from in_view (zero copy)
    int width = 0, height = 0;         // == DINOv3 grid
    const float* matrices = nullptr;   // n * 9
    int n = 0;
    PipelineConfig cfg{};
    const float* mu = nullptr;         // width * height * 3
    const float* sigma = nullptr;      // width * height * 3
    float* out = nullptr;              // n * 4: loss_L, loss_a, loss_b, white_clip_ch_pct
};

struct HighlightClipArgs {
    const uint16_t* in = nullptr;
    const RgbView* in_view = nullptr;  // When non-null, reads directly from in_view (zero copy)
    int width = 0, height = 0;
    InnerCrop crop{};
    const float* matrices = nullptr;   // n * 9
    int n = 0;
    PipelineConfig cfg{};
    float ceil_lin[3] = {1.0f, 1.0f, 1.0f};
    float* out = nullptr;              // n * 5 floats
};

struct LabStatsRow {
    float hist[kLabBins];
    double white_clip, black_clip;
    double mid_a, mid_b, mid_n;
    double hi_a, hi_b, hi_n;
    double sh_a, sh_b, sh_n;
    double all_a, all_b;
    double total;
};

struct LabStatsArgs {
    const uint16_t* in = nullptr;
    const RgbView* in_view = nullptr;  // When non-null, reads directly from in_view (zero copy)
    int width = 0, height = 0;
    const float* matrices = nullptr;
    int n = 0;
    PipelineConfig cfg{};
    LabStatsRow* out = nullptr;
};

struct PreviewDebayerArgs {
    const uint16_t* cfa_bayer = nullptr;
    int src_w = 0, src_h = 0;
    int rx = 0, ry = 0, rw = 0, rh = 0;
    int rot_cw = 0;
    bool hflip = false, vflip = false;
    int dst_w = 0, dst_h = 0;
    unsigned int filters = 0;
    uint16_t* rgb_out = nullptr;
    FlatSampleArgs flat{};
    int flat_x0 = 0, flat_y0 = 0;
};

struct GrainAwareDebayerArgs {
    const uint16_t* cfa_bayer = nullptr;
    int src_w = 0, src_h = 0;
    int rx = 0, ry = 0, rw = 0, rh = 0;
    int rot_cw = 0;
    bool hflip = false, vflip = false;
    int dst_w = 0, dst_h = 0;
    unsigned int filters = 0;
    float white_level = 65535.0f;
    float eps = 4096.0f;
    uint16_t* rgb_out = nullptr;
    FlatSampleArgs flat{};
    int flat_x0 = 0, flat_y0 = 0;
    size_t cfa_pitch = 0;  // 0 defaults to src_w
};

struct PixelShiftMergeArgs {
    const uint16_t* bayer_f0 = nullptr;
    const uint16_t* bayer_f1 = nullptr;
    const uint16_t* bayer_f2 = nullptr;
    const uint16_t* bayer_f3 = nullptr;
    int src_w = 0, src_h = 0;
    int rx = 0, ry = 0, rw = 0, rh = 0;
    int rot_cw = 0;
    bool hflip = false, vflip = false;
    int dst_w = 0, dst_h = 0;
    unsigned int filters = 0;
    float white_level = 65535.0f;
    uint16_t* rgb_out = nullptr;
    FlatSampleArgs flat{};
    int flat_x0 = 0, flat_y0 = 0;
    size_t cfa_pitch = 0;  // 0 defaults to src_w
};

void cpu_render(const RenderArgs& a);
void cpu_intent_loss(const IntentLossArgs& a);
void cpu_highlight_clip(const HighlightClipArgs& a);
void cpu_lab_stats(const LabStatsArgs& a);

void cpu_preview_debayer(const PreviewDebayerArgs& a);
void cpu_grain_aware_debayer(const GrainAwareDebayerArgs& a);
void cpu_pixel_shift_merge(const PixelShiftMergeArgs& a);

// Streaming 1-CFA-at-a-time Sony 4-shot pixel shift merge: call for frame_idx = 0, 1, 2, 3 in order.
// Accumulates directly into a.rgb_out (zero extra buffer, only 1 CFA mosaic in memory at a time),
// and applies a.flat on frame_idx == 3, producing bit-identical output to cpu_pixel_shift_merge.
void cpu_pixel_shift_merge_frame(const PixelShiftMergeArgs& a, int frame_idx, const uint16_t* bayer_frame);

}  // namespace negicc

#endif  // NEGICC_CFA_PIPELINE_H
