// PIL-exact and OpenCV-compatible preprocessing for DINOv3 photographic intent.
//
// py-ref: negicc_station/src/tensorrt_intent_model.py:189 @ 2f7ee4a (PIL Image.Resampling.BILINEAR)
// py-ref: negicc_station/src/dino_geometry.py:43-74 @ 2f7ee4a (compute_tone_weights)
// py-ref: negicc_station/src/dino_geometry.py:30-41 @ 2f7ee4a (compute_aspect_preserved_shape)
// py-ref: negicc_station/src/per_target_conversion.py:217-222 @ 2f7ee4a (cv_linear_resize_rgb)

#include "dinov3_preprocess.h"

#include <algorithm>
#include <cmath>
#include <vector>

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
    double scale = (double)max_size / (double)std::max(w, h);
    target_w = std::max(patch_size, (int)std::nearbyint((w * scale) / patch_size) * patch_size);
    target_h = std::max(patch_size, (int)std::nearbyint((h * scale) / patch_size) * patch_size);
    grid_w = target_w / patch_size;
    grid_h = target_h / patch_size;
}

void antialiased_bilinear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                                     uint8_t* dst, int dst_w, int dst_h) {
    const bool need_h = dst_w != src_w, need_v = dst_h != src_h;
    if (!need_h && !need_v) {
        for (int y = 0; y < src_h; ++y) std::copy_n(src + (size_t)y * src_stride, (size_t)src_w * 3, dst + (size_t)y * dst_w * 3);
        return;
    }

    // Horizontal pass into an 8-bit image [src_h, dst_w] (or the source itself when the width is unchanged)
    std::vector<uint8_t> tmp;
    const uint8_t* mid = src;
    int mid_stride = src_stride;
    if (need_h) {
        const FilterCoeffs cx = compute_antialiased_bilinear_coeffs(src_w, dst_w);
        uint8_t* out = need_v ? (tmp.resize((size_t)src_h * dst_w * 3), tmp.data()) : dst;
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
        mid = tmp.data();
        mid_stride = dst_w * 3;
    }

    const FilterCoeffs cy = compute_antialiased_bilinear_coeffs(src_h, dst_h);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dst_h; ++y) {
        const int* k = &cy.kk[(size_t)y * cy.ksize];
        uint8_t* orow = dst + (size_t)y * dst_w * 3;
        for (int x = 0; x < dst_w * 3; ++x) {
            int s = 1 << (kFilterPrecisionBits - 1);
            for (int i = 0; i < cy.count[y]; ++i)
                s += mid[(size_t)(cy.xmin[y] + i) * mid_stride + x] * k[i];
            orow[x] = clip_u8(s);
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

    auto srgb_lin = [](double c) {
        return (c <= 0.04045) ? (c / 12.92) : std::pow((c + 0.055) / 1.055, 2.4);
    };

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

            const double cr = srgb_lin(r_sum * inv_pixels / 255.0);
            const double cg = srgb_lin(g_sum * inv_pixels / 255.0);
            const double cb = srgb_lin(b_sum * inv_pixels / 255.0);

            const double Y = 0.212671 * cr + 0.715160 * cg + 0.072169 * cb;
            const double fy = (Y > 0.008856) ? std::cbrt(Y) : (7.787 * Y + 16.0 / 116.0);
            const double L = (Y > 0.008856) ? (116.0 * fy - 16.0) : (903.3 * Y);

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
