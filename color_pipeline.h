// Shared per-pixel colour pipeline.
//
// Every conversion in negicc and negicc-station (preview render, full export, intent loss, Lab statistics) runs the
// stages below. They are written once, as inline functions compiled both by the host compiler (CPU backend,
// OpenMP) and by nvcc (CUDA backend), so the two backends cannot drift apart.
//
// Stage order for one pixel (camera-linear RGB, 16-bit, from the demosaiced negative):
//   load (u16 -> [0,1]) -> matrix (crosstalk x base scale x diag(E, E*g, E*b)) -> soft knee
//   -> [profile: input TRCs -> matrix + offset, clamp -> tetrahedral cLUT -> output TRCs -> PCS scale]
//   -> Bradford D50->D65 -> XYZ->linear sRGB, clamp at 0 only: the pre-shoulder value (clip statistics and the
//      x_hi histogram read it)
//   -> highlight shoulder (identity when cfg.shoulder_x_hi <= 1) -> clamp -> encode (sRGB curve or linear), clamp:
//      the display value (render and Lab read it)
#ifndef NEGICC_COLOR_PIPELINE_H
#define NEGICC_COLOR_PIPELINE_H

#include <stdint.h>
#include <cmath>

#if defined(__CUDACC__)
#define NEGICC_HD __host__ __device__ __forceinline__
#include <math.h>
#else
#define NEGICC_HD inline
#include <cmath>
#endif

namespace negicc {

struct Rgb {
    float r, g, b;
};

// Output encodings.
enum OutputEncoding : int {
    ENCODE_SRGB = 0,        // IEC 61966-2-1 transfer curve
    ENCODE_SRGB_LINEAR = 1  // linear-light sRGB primaries ("srgb-g10")
};

// Parsed ICC AToB0 pipeline. Pointers refer to host or device memory depending on the backend that runs the
// stages; the layout is identical. A null pointer / zero size skips that element.
struct ProfileData {
    int has_profile;
    const float* in_trc[3];
    int in_trc_size[3];
    const float* out_trc[3];
    int out_trc_size[3];
    int has_matrix;
    float matrix[9];
    float offset[3];
    const float* clut;       // [dim_r][dim_g][dim_b][3]
    int clut_dim[3];
};

// Everything a conversion needs apart from the per-candidate 3x3 input matrix.
struct PipelineConfig {
    ProfileData profile;
    float knee;              // soft-knee threshold in [0,1]; >= 1 disables the knee
    int encoding;            // OutputEncoding
    // Highlight shoulder (ENCODE_SRGB only). x_hi <= kShoulderOff disables it; otherwise set_shoulder() fills the
    // knee and the derived constants once per config, never per pixel.
    float shoulder_x_hi = 1.0f;
    float shoulder_n = 2.0f;
    float shoulder_x_k = 1.0f;  // linear knee: the curve is the identity at and below it
    float shoulder_t_k = 0.0f;  // log2(x_k)
    float shoulder_t_hi = 0.0f; // log2(x_hi)
    float shoulder_s_k = 1.0f;  // encoded sRGB value at the knee
    float shoulder_y_c = 1.0f;  // linear value of the ceiling, reached at x_hi
};

constexpr float kShoulderOff = 1.0001f;       // x_hi at or below this is "off"
constexpr float kShoulderCeiling = 0.988f;    // encoded sRGB ceiling C (about code 252)
constexpr float kShoulderExponent = 2.0f;     // default n; 8 is an evaluation variant, never a per-frame parameter
constexpr float kShoulderXHiMax = 1.5f;

// Bradford D50 -> D65 and XYZ -> linear sRGB constants are written inline in pcs_to_linear_srgb() so they are
// usable from device code.
// ICC PCS encoding scale applied after the output TRCs (1 + 32767/32768).
constexpr float kPcsScale = 1.99996948f;

NEGICC_HD float clamp01(float v) {
    return fminf(1.0f, fmaxf(0.0f, v));
}

// ---- Stage: load -----------------------------------------------------------------------------------------
NEGICC_HD Rgb load_u16(const uint16_t* px) {
    return { px[0] / 65535.0f, px[1] / 65535.0f, px[2] / 65535.0f };
}

// ---- Stage: input matrix -----------------------------------------------------------------------------------
NEGICC_HD Rgb apply_matrix(const float* m, Rgb v) {
    return { v.r * m[0] + v.g * m[1] + v.b * m[2],
             v.r * m[3] + v.g * m[4] + v.b * m[5],
             v.r * m[6] + v.g * m[7] + v.b * m[8] };
}

// Builds the per-candidate input matrix from the merged crosstalk matrix (crosstalk x base scale):
// rows are scaled by E, E*g and E*b. Same float operations as the original prepare_adjusted_crosstalk().
NEGICC_HD void candidate_matrix(const float* merged, float e, float g, float b, float* out) {
    const float sr = e;
    const float sg = 1.0f * e * g;
    const float sb = 1.0f * e * b;
    for (int i = 0; i < 3; ++i) {
        out[i] = merged[i] * sr;
        out[3 + i] = merged[3 + i] * sg;
        out[6 + i] = merged[6 + i] * sb;
    }
}

// ---- Stage: soft knee --------------------------------------------------------------------------------------
NEGICC_HD float knee_compress(float val, float knee) {
    if (val <= 0.0f) return 0.0f;
    if (knee >= 1.0f || val <= knee) return fminf(1.0f, val);
    const float delta = 1.0f - knee;
    return fminf(0.9999f, knee + delta * tanhf((val - knee) / delta));
}

NEGICC_HD Rgb apply_knee(Rgb v, float knee) {
    return { knee_compress(v.r, knee), knee_compress(v.g, knee), knee_compress(v.b, knee) };
}

// ---- Stage: tone curves ------------------------------------------------------------------------------------
NEGICC_HD float apply_trc(float val, const float* curve, int size) {
    if (size <= 0 || !curve) return val;
    const float scaled = val * (size - 1);
    int k = (int)scaled;
    k = k < 0 ? 0 : (k > size - 2 ? size - 2 : k);
    const float delta = scaled - k;
    return curve[k] * (1.0f - delta) + curve[k + 1] * delta;
}

// ---- Stage: tetrahedral cLUT -------------------------------------------------------------------------------
// Weight form of tetrahedral interpolation.
NEGICC_HD Rgb interpolate_clut(float r, float g, float b, const float* clut, int dim_r, int dim_g, int dim_b) {
    const float sr = r * (dim_r - 1);
    const float sg = g * (dim_g - 1);
    const float sb = b * (dim_b - 1);
    int rf = (int)sr, gf = (int)sg, bf = (int)sb;
    const int rc = (rf + 1 < dim_r - 1) ? rf + 1 : dim_r - 1;
    const int gc = (gf + 1 < dim_g - 1) ? gf + 1 : dim_g - 1;
    const int bc = (bf + 1 < dim_b - 1) ? bf + 1 : dim_b - 1;
    rf = rf < 0 ? 0 : (rf > dim_r - 1 ? dim_r - 1 : rf);
    gf = gf < 0 ? 0 : (gf > dim_g - 1 ? dim_g - 1 : gf);
    bf = bf < 0 ? 0 : (bf > dim_b - 1 ? dim_b - 1 : bf);
    const float dr = sr - rf, dg = sg - gf, db = sb - bf;

    const int s_r = dim_g * dim_b * 3, s_g = dim_b * 3;
    const float* v000 = clut + rf * s_r + gf * s_g + bf * 3;
    const float* v100 = clut + rc * s_r + gf * s_g + bf * 3;
    const float* v010 = clut + rf * s_r + gc * s_g + bf * 3;
    const float* v110 = clut + rc * s_r + gc * s_g + bf * 3;
    const float* v001 = clut + rf * s_r + gf * s_g + bc * 3;
    const float* v101 = clut + rc * s_r + gf * s_g + bc * 3;
    const float* v011 = clut + rf * s_r + gc * s_g + bc * 3;
    const float* v111 = clut + rc * s_r + gc * s_g + bc * 3;

    // Four vertices and weights of the tetrahedron containing (dr, dg, db).
    const float *a, *p, *q;
    float w0, w1, w2, w3;
    if (dr >= dg && dg >= db)      { a = v100; p = v110; q = v111; w0 = 1.0f - dr; w1 = dr - dg; w2 = dg - db; w3 = db; }
    else if (dr >= db && db > dg)  { a = v100; p = v101; q = v111; w0 = 1.0f - dr; w1 = dr - db; w2 = db - dg; w3 = dg; }
    else if (db > dr && dr >= dg)  { a = v001; p = v101; q = v111; w0 = 1.0f - db; w1 = db - dr; w2 = dr - dg; w3 = dg; }
    else if (dg > dr && dr >= db)  { a = v010; p = v110; q = v111; w0 = 1.0f - dg; w1 = dg - dr; w2 = dr - db; w3 = db; }
    else if (dg >= db && db > dr)  { a = v010; p = v011; q = v111; w0 = 1.0f - dg; w1 = dg - db; w2 = db - dr; w3 = dr; }
    else                           { a = v001; p = v011; q = v111; w0 = 1.0f - db; w1 = db - dg; w2 = dg - dr; w3 = dr; }
    return { v000[0] * w0 + a[0] * w1 + p[0] * w2 + q[0] * w3,
             v000[1] * w0 + a[1] * w1 + p[1] * w2 + q[1] * w3,
             v000[2] * w0 + a[2] * w1 + p[2] * w2 + q[2] * w3 };
}

// ---- Stage: film profile (ICC AToB0) -----------------------------------------------------------------------
NEGICC_HD Rgb apply_profile(const ProfileData& p, Rgb v) {
    if (!p.has_profile) return v;
    v.r = apply_trc(v.r, p.in_trc[0], p.in_trc_size[0]);
    v.g = apply_trc(v.g, p.in_trc[1], p.in_trc_size[1]);
    v.b = apply_trc(v.b, p.in_trc[2], p.in_trc_size[2]);

    const float* m = p.matrix;
    const float* o = p.offset;
    const Rgb t = { clamp01(v.r * m[0] + v.g * m[1] + v.b * m[2] + o[0]),
                    clamp01(v.r * m[3] + v.g * m[4] + v.b * m[5] + o[1]),
                    clamp01(v.r * m[6] + v.g * m[7] + v.b * m[8] + o[2]) };
    if (p.clut && p.clut_dim[0] > 0 && p.clut_dim[1] > 0 && p.clut_dim[2] > 0) {
        v = interpolate_clut(t.r, t.g, t.b, p.clut, p.clut_dim[0], p.clut_dim[1], p.clut_dim[2]);
    }
    v.r = apply_trc(v.r, p.out_trc[0], p.out_trc_size[0]) * kPcsScale;
    v.g = apply_trc(v.g, p.out_trc[1], p.out_trc_size[1]) * kPcsScale;
    v.b = apply_trc(v.b, p.out_trc[2], p.out_trc_size[2]) * kPcsScale;
    return v;
}

// ---- Stage: PCS (XYZ D50) -> linear sRGB -------------------------------------------------------------------
NEGICC_HD Rgb pcs_to_linear_srgb(Rgb v) {
    // Bradford D50 -> D65
    const float xr = v.r * 0.9555766f + v.g * -0.0230393f + v.b * 0.0631636f;
    const float xg = v.r * -0.0282895f + v.g * 1.0099416f + v.b * 0.0210077f;
    const float xb = v.r * 0.0122982f + v.g * -0.0204830f + v.b * 1.3299098f;
    // XYZ -> linear sRGB
    return { clamp01(xr * 3.2406255f + xg * -1.5372080f + xb * -0.4986286f),
             clamp01(xr * -0.9689307f + xg * 1.8757561f + xb * 0.0415175f),
             clamp01(xr * 0.0557101f + xg * -0.2040211f + xb * 1.0569959f) };
}

// Same as pcs_to_linear_srgb() but clamped at 0 only, so values above the chart white (the profile's headroom,
// up to about 1.33 on the grey axis) survive for the shoulder.
NEGICC_HD Rgb pcs_to_linear_srgb_pre_shoulder(Rgb v) {
    const float xr = v.r * 0.9555766f + v.g * -0.0230393f + v.b * 0.0631636f;
    const float xg = v.r * -0.0282895f + v.g * 1.0099416f + v.b * 0.0210077f;
    const float xb = v.r * 0.0122982f + v.g * -0.0204830f + v.b * 1.3299098f;
    return { fmaxf(0.0f, xr * 3.2406255f + xg * -1.5372080f + xb * -0.4986286f),
             fmaxf(0.0f, xr * -0.9689307f + xg * 1.8757561f + xb * 0.0415175f),
             fmaxf(0.0f, xr * 0.0557101f + xg * -0.2040211f + xb * 1.0569959f) };
}

// ---- Stage: output encoding --------------------------------------------------------------------------------
NEGICC_HD float srgb_encode(float l) {
    return (l <= 0.0031308f) ? (l * 12.92f) : (powf(l, 0.41666667f) * 1.055f - 0.055f);
}

NEGICC_HD Rgb encode_output(Rgb v, int encoding) {
    if (encoding == ENCODE_SRGB) v = { srgb_encode(v.r), srgb_encode(v.g), srgb_encode(v.b) };
    return { clamp01(v.r), clamp01(v.g), clamp01(v.b) };
}

NEGICC_HD float srgb_decode(float c) {
    return (c <= 0.04045f) ? (c * (1.0f / 12.92f)) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

// ---- Stage: highlight shoulder -----------------------------------------------------------------------------
// Per channel, in stops t = log2(x) of the pre-shoulder value: the identity up to the knee t_k, then
// s = C - (C - s_k) (1 - (t - t_k) / (t_hi - t_k))^n in encoded sRGB, reaching the ceiling C at x_hi and staying there.
// The knee joins the identity with a continuous first derivative: t_hi - t_k = n (C - s(t_k)) / m(t_k).
// Only for ENCODE_SRGB; the host solves the knee once in set_shoulder().
NEGICC_HD float apply_shoulder_channel(const PipelineConfig& cfg, float x) {
    if (!(x > cfg.shoulder_x_k)) return x;
    const float t = log2f(x);
    if (t >= cfg.shoulder_t_hi) return cfg.shoulder_y_c;
    const float u = 1.0f - (t - cfg.shoulder_t_k) / (cfg.shoulder_t_hi - cfg.shoulder_t_k);
    const float up = (cfg.shoulder_n == 2.0f) ? u * u : powf(u, cfg.shoulder_n);
    return srgb_decode(kShoulderCeiling - (kShoulderCeiling - cfg.shoulder_s_k) * up);
}

NEGICC_HD Rgb apply_shoulder(const PipelineConfig& cfg, Rgb v) {
    if (cfg.encoding != ENCODE_SRGB || !(cfg.shoulder_x_hi > kShoulderOff)) return v;
    return { apply_shoulder_channel(cfg, v.r), apply_shoulder_channel(cfg, v.g), apply_shoulder_channel(cfg, v.b) };
}

// ---- Whole pipeline ----------------------------------------------------------------------------------------
// Returns the pre-shoulder linear sRGB D65 value, clamped at 0 only (0 to about 1.33; chart white is 1).
NEGICC_HD Rgb convert_pixel_linear(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    Rgb v = load_u16(px);
    v = apply_matrix(matrix, v);
    v = apply_knee(v, cfg.knee);
    v = apply_profile(cfg.profile, v);
    return pcs_to_linear_srgb_pre_shoulder(v);
}

// Returns linear sRGB D65 in [0, 1] after the shoulder, before output gamma encoding: what is shown and scored.
NEGICC_HD Rgb to_display(const PipelineConfig& cfg, Rgb pre) {
    const Rgb v = apply_shoulder(cfg, pre);
    return { clamp01(v.r), clamp01(v.g), clamp01(v.b) };
}

NEGICC_HD Rgb convert_pixel_display(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    return to_display(cfg, convert_pixel_linear(cfg, matrix, px));
}

// Returns the encoded output in [0,1] for one camera-linear pixel and one 3x3 input matrix.
NEGICC_HD Rgb convert_pixel(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    return encode_output(convert_pixel_display(cfg, matrix, px), cfg.encoding);
}

// ---- Output quantisation -----------------------------------------------------------------------------------
NEGICC_HD uint8_t quantize_u8(float v) { return (uint8_t)roundf(v * 255.0f); }
NEGICC_HD uint16_t quantize_u16(float v) { return (uint16_t)roundf(v * 65535.0f); }

// ---- Scoring: sRGB -> CIELAB (D65) -------------------------------------------------------------------------
struct Lab {
    float L, a, b;
};

NEGICC_HD float lab_f(float t) {
    return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f);
}

// Direct linear sRGB D65 -> CIELAB D65 (skips 6x powf of srgb_encode + srgb_decode).
NEGICC_HD Lab linear_srgb_to_lab(Rgb lin) {
    const float r = lin.r, g = lin.g, b = lin.b;
    const float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
    const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    const float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
    const float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
    const float L = (Y > 0.008856f) ? (116.0f * fy - 16.0f) : (903.3f * Y);
    return { L, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

// Direct linear sRGB D65 -> CIE L* (1 cbrtf, 0 powf).
NEGICC_HD float linear_srgb_to_L(Rgb lin) {
    const float Y = 0.212671f * lin.r + 0.715160f * lin.g + 0.072169f * lin.b;
    return (Y > 0.008856f) ? (116.0f * cbrtf(Y) - 16.0f) : (903.3f * Y);
}

NEGICC_HD Lab srgb_to_lab(Rgb c) {
    return linear_srgb_to_lab({ srgb_decode(c.r), srgb_decode(c.g), srgb_decode(c.b) });
}

// ---- Scoring: clipping rules (one definition for every consumer) -------------------------------------------
NEGICC_HD bool channel_white_clipped(float v) { return v * 65535.0f >= 65399.5f; }
NEGICC_HD bool channel_black_clipped(float v) { return v * 65535.0f < 200.5f; }

NEGICC_HD bool linear_channel_white_clipped(float l) { return l >= 0.9953029f; }
NEGICC_HD bool linear_channel_black_clipped(float l) { return l < 0.00023679829f; }

constexpr int kLabBins = 1000;
NEGICC_HD int lab_bin(float L) {
    int k = (int)(L * (kLabBins / 100.0f));
    return k < 0 ? 0 : (k > kLabBins - 1 ? kLabBins - 1 : k);
}

// ---- Symmetric inner-crop & aspect-preserved DINOv3 geometry -----------------------------------------------
struct InnerCrop {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

NEGICC_HD InnerCrop compute_inner_crop(int W, int H, float crop_ratio) {
    if (W <= 0 || H <= 0) return {0, 0, 0, 0};
    if (crop_ratio >= 1.0f) return {0, 0, W, H};
    int x0 = (int)roundf(W * (1.0f - crop_ratio) * 0.5f);
    int y0 = (int)roundf(H * (1.0f - crop_ratio) * 0.5f);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    const int max_x0 = (W - 1) / 2;
    const int max_y0 = (H - 1) / 2;
    if (x0 > max_x0) x0 = max_x0;
    if (y0 > max_y0) y0 = max_y0;
    return {x0, y0, W - 2 * x0, H - 2 * y0};
}

struct AspectPreservedShape {
    int target_w = 16, target_h = 16;
    int grid_w = 1, grid_h = 1;
};

inline AspectPreservedShape compute_aspect_preserved_shape(int w, int h, int max_size = 512, int patch_size = 16) {
    const int p = patch_size > 1 ? patch_size : 1;
    const int m = max_size > p ? max_size : p;
    const int mw = w > 1 ? w : 1;
    const int mh = h > 1 ? h : 1;
    const double max_dim = mw > mh ? (double)mw : (double)mh;
    const double scale = (double)m / max_dim;
    int gw = (int)std::nearbyint((double)mw * scale / (double)p);
    int gh = (int)std::nearbyint((double)mh * scale / (double)p);
    if (gw < 1) gw = 1;
    if (gh < 1) gh = 1;
    return {gw * p, gh * p, gw, gh};
}

// ---- Host: solve the shoulder knee ------------------------------------------------------------------------
// x_hi in (kShoulderOff, kShoulderXHiMax] enables the shoulder (anything else turns it off). Bisection on the knee
// t_k in [-3, 0] in double precision; the C1 condition changes sign once.
inline void set_shoulder(PipelineConfig& cfg, float x_hi, float n = kShoulderExponent) {
    cfg.shoulder_n = n;
    if (!(x_hi > kShoulderOff)) {
        cfg.shoulder_x_hi = 1.0f;
        cfg.shoulder_x_k = 1.0f;
        cfg.shoulder_t_k = cfg.shoulder_t_hi = 0.0f;
        cfg.shoulder_s_k = 1.0f;
        cfg.shoulder_y_c = 1.0f;
        return;
    }
    if (x_hi > kShoulderXHiMax) x_hi = kShoulderXHiMax;
    const double C = kShoulderCeiling;
    const double ln2 = std::log(2.0);
    auto enc = [](double x) { return x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055; };
    auto dec = [](double s) { return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4); };
    auto slope = [&](double t) { return (1.055 / 2.4) * std::pow(std::exp2(t), 1.0 / 2.4) * ln2; };
    const double t_hi = std::log2((double)x_hi);
    auto f = [&](double t_k) { return t_hi - t_k - n * (C - enc(std::exp2(t_k))) / slope(t_k); };
    double lo = -3.0, hi = 0.0;
    const bool lo_pos = f(lo) > 0.0;
    for (int i = 0; i < 100; ++i) {
        const double mid = 0.5 * (lo + hi);
        if ((f(mid) > 0.0) == lo_pos) lo = mid; else hi = mid;
    }
    const double t_k = 0.5 * (lo + hi);
    cfg.shoulder_x_hi = x_hi;
    cfg.shoulder_t_k = (float)t_k;
    cfg.shoulder_x_k = (float)std::exp2(t_k);
    cfg.shoulder_t_hi = (float)t_hi;
    cfg.shoulder_s_k = (float)enc(std::exp2(t_k));
    cfg.shoulder_y_c = (float)dec(C);
}

}  // namespace negicc

#endif  // NEGICC_COLOR_PIPELINE_H
