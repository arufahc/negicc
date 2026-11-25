// Copyright 2024-2026 Alpha Lam <arufa.hc@gmail.com>
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

// negicc's library, the parts that need only the standard library, in this order (each needs only those before
// it): the per-pixel colour pipeline, the frame detector, the flat field, the image view and its passes, the
// white-light input curves, the tone stage, the film base detector, the index of the embedded data and
// neg_process's messages. gpu_passes.cu and film_detector.cc (the frame and film base detectors) include this header
// alone; every other source includes negicc.h.

#ifndef NEGICC_CORE_H
#define NEGICC_CORE_H

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ====================================================================================================================
// Colour pipeline: the per-pixel stages every conversion runs, on the CPU and in CUDA
// ====================================================================================================================

// Shared per-pixel colour pipeline.
//
// Every conversion in negicc and negicc-station (preview render, full export, intent loss, Lab statistics) runs the
// stages below. They are written once, as inline functions compiled both by the host compiler (CPU backend,
// OpenMP) and by nvcc (CUDA backend), so the two backends cannot drift apart.
//
// Stage order for one pixel (camera-linear RGB, 16-bit, from the demosaiced negative), with the function that
// returns each intermediate value:
//   load (u16 -> [0,1]) -> input matrix (diag(E, E*g, E*b) x film base scale x crosstalk) -> soft knee
//      convert_pixel_corrected: the corrected RGB in [0, 1]
//   -> film profile: input TRCs -> matrix + offset, clamp -> tetrahedral cLUT -> output TRCs -> PCS scale
//      convert_pixel_pcs: ICC PCS XYZ (D50)
//   -> linear sRGB (the inverse of the embedded profile's colorants), clamped at 0 only
//      convert_pixel_linear: the pre-shoulder value (the clip statistics and the x_hi histogram read it)
//   -> highlight shoulder (identity when cfg.shoulder_x_hi <= kShoulderOff) -> clamp to [0, 1]
//      convert_pixel_display: the display value (render and Lab read it)
//   -> tone curve (apply_tone; only with cfg.tone on an sRGB export: the tone stage)
//   -> encode (sRGB curve or linear), clamp
//      convert_pixel: the output code value in [0, 1]
//
// A backend that adds a stage writes it here as a NEGICC_HD function and calls it from these convert_pixel_* functions;
// per-pixel kernels (CPU loops: rgb_view.cc; CUDA kernels: gpu_passes.cu and the companion application) only call them.

#if defined(__CUDACC__)
#define NEGICC_HD __host__ __device__ __forceinline__
#include <math.h>
#else
#define NEGICC_HD inline
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
// The tone stage's curve on display linear sRGB luminance (the tone stage part below): lut holds Y' = f(Y) at
// Y = (i / n)^2, i = 0..n (n + 1 entries, linear interpolation in sqrt(Y)); above 1 Y' / Y stays at its value at 1.
// lut null or n < 1: no tone stage. The table lives in the memory of the code that runs the kernel (host memory on the
// CPU; device code needs a device copy, uploaded once per frame). A zero-initialised ToneTable is "no tone stage".
struct ToneTable {
    const float* lut = nullptr;
    int n = 0;
};

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
    float shoulder_c = 1.0f;    // encoded sRGB ceiling C reached at x_hi (set_shoulder(), from kShoulderCeiling)
    float shoulder_y_c = 1.0f;  // linear value of the ceiling
    // The tone stage (sRGB export only; null everywhere else: the search, the statistics and the shoulder measurement
    // never set it).
    ToneTable tone;
};

constexpr float kShoulderOff = 1.0001f;       // x_hi at or below this is "off"
constexpr float kShoulderCeiling = 0.988f;    // encoded sRGB ceiling C (about code 252) once x_hi is past the ramp
// The ceiling is ramped in over x_hi in (1, 1 + kShoulderRamp]: C goes linearly from 1 (code 255, the unshouldered
// white) to kShoulderCeiling. Without it the step from off (white at 255) to the smallest x_hi (white at about 252)
// would make neighbouring frames of a roll, one just above and one just below the boundary, differ by 3 codes.
constexpr float kShoulderRamp = 0.10f;
// The shoulder's exponent n: 2, a quadratic roll-off. A steeper shoulder (n = 8, which holds the identity longer
// and bends later) is still being evaluated against it. n is one constant for every frame, never a per-frame
// parameter.
constexpr float kShoulderExponent = 2.0f;
// Auto's highlight roll-off (shoulder_x_hi_anchored) maps the frame's own 99.5% point to sRGB kShoulderAnchorCode, so
// the frame's highlights, not the chart white, set where it starts: x_hi up to kShoulderXHiLimit (the largest x_hi
// set_shoulder accepts). 242 is the code above which the highlight pullback's gate counts a highlight as blown.
constexpr float kShoulderXHiLimit = 2.5f;
constexpr float kShoulderAnchorCode = 242.0f;

// ICC PCS encoding scale applied after the output TRCs: 1 + 32767/32768, the PCS XYZ value of code FFFFh, the largest
// u1Fixed15Number (ICC.1:2022 4.8 Table 6 and 6.3.4.2 Table 11), so a curve's [0, 1] output becomes PCS XYZ. A lut16's
// table holds PCS XYZ as its code over this (print_emulation.cc). Exact in single precision.
constexpr float kPcsScale = 1.0f + 32767.0f / 32768.0f;

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

// The input matrix of one candidate (E, g, b): the merged matrix (crosstalk x film base scale) with its rows
// scaled by E, E*g and E*b.
NEGICC_HD void candidate_matrix(const float* merged, float e, float g, float b, float* out) {
    const float sr = e;
    const float sg = e * g;
    const float sb = e * b;
    for (int i = 0; i < 3; ++i) {
        out[i] = merged[i] * sr;
        out[3 + i] = merged[3 + i] * sg;
        out[6 + i] = merged[6 + i] * sb;
    }
}

// ---- Stage: soft knee --------------------------------------------------------------------------------------
// The identity up to the knee, then a tanh roll-off towards 1 (capped at 0.9999), so values past the film
// profile's input range compress instead of clipping. Negative values become 0. knee >= 1 only clamps to [0, 1].
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
// A 1D curve of size samples over [0, 1], linearly interpolated (and extrapolated from the end segments). A curve of
// one sample is that value.
NEGICC_HD float apply_trc(float val, const float* curve, int size) {
    if (size <= 0 || !curve) return val;
    if (size == 1) return curve[0];
    const float scaled = val * (size - 1);
    int k = (int)scaled;
    k = k < 0 ? 0 : (k > size - 2 ? size - 2 : k);
    const float delta = scaled - k;
    return curve[k] * (1.0f - delta) + curve[k + 1] * delta;
}

// ---- Stage: tetrahedral cLUT -------------------------------------------------------------------------------
// Tetrahedral interpolation of a [dim_r][dim_g][dim_b][3] grid at (r, g, b) in [0, 1]: the unit cell is split into
// six tetrahedra by the order of the fractional parts, and the value is the weighted sum of the four corners of
// the one that contains the point. Without a table (null, or a dimension below 1) the value passes through.
NEGICC_HD Rgb interpolate_clut(float r, float g, float b, const float* clut, int dim_r, int dim_g, int dim_b) {
    if (!clut || dim_r <= 0 || dim_g <= 0 || dim_b <= 0) return { r, g, b };
    const float sr = r * (dim_r - 1);
    const float sg = g * (dim_g - 1);
    const float sb = b * (dim_b - 1);
    // The lower corner inside the grid, then the upper one from it, so a coordinate outside [0, 1] reads the edge cell.
    int rf = (int)sr, gf = (int)sg, bf = (int)sb;
    rf = rf < 0 ? 0 : (rf > dim_r - 1 ? dim_r - 1 : rf);
    gf = gf < 0 ? 0 : (gf > dim_g - 1 ? dim_g - 1 : gf);
    bf = bf < 0 ? 0 : (bf > dim_b - 1 ? dim_b - 1 : bf);
    const int rc = (rf + 1 < dim_r - 1) ? rf + 1 : dim_r - 1;
    const int gc = (gf + 1 < dim_g - 1) ? gf + 1 : dim_g - 1;
    const int bc = (bf + 1 < dim_b - 1) ? bf + 1 : dim_b - 1;
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
    if (dr >= dg && dg >= db)     { a = v100; p = v110; q = v111; w0 = 1.0f - dr; w1 = dr - dg; w2 = dg - db; w3 = db; }
    else if (dr >= db && db > dg) { a = v100; p = v101; q = v111; w0 = 1.0f - dr; w1 = dr - db; w2 = db - dg; w3 = dg; }
    else if (db > dr && dr >= dg) { a = v001; p = v101; q = v111; w0 = 1.0f - db; w1 = db - dr; w2 = dr - dg; w3 = dg; }
    else if (dg > dr && dr >= db) { a = v010; p = v110; q = v111; w0 = 1.0f - dg; w1 = dg - dr; w2 = dr - db; w3 = db; }
    else if (dg >= db && db > dr) { a = v010; p = v011; q = v111; w0 = 1.0f - dg; w1 = dg - db; w2 = db - dr; w3 = dr; }
    else                          { a = v001; p = v011; q = v111; w0 = 1.0f - db; w1 = db - dg; w2 = dg - dr; w3 = dr; }
    return { v000[0] * w0 + a[0] * w1 + p[0] * w2 + q[0] * w3,
             v000[1] * w0 + a[1] * w1 + p[1] * w2 + q[1] * w3,
             v000[2] * w0 + a[2] * w1 + p[2] * w2 + q[2] * w3 };
}

// ---- Stage: film profile (ICC AToB0) -----------------------------------------------------------------------
// The lutAtoBType elements in their ICC order (A curves, matrix, cLUT, B curves); the result is PCS XYZ (D50).
// Without a profile the value passes through.
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
// The PCS white: the ICC's D50 as an ICC profile stores it, in s15.16, (63190, 65536, 54061) / 65536: the PCS
// illuminant (ICC.1:2022 7.2.16, which gives it to four decimals) as an s15Fixed16Number (4.6). The colorants of
// the profile the sRGB outputs embed sum to exactly this, so it is the white the outputs write at full scale (a print's
// paper is scaled to it). The decimal (0.9642, 1, 0.8249) differs from it in the sixth decimal.
constexpr double kPcsWhite[3] = {63190.0 / 65536.0, 1.0, 54061.0 / 65536.0};
// PCS -> linear sRGB: the inverse of the colorants (rXYZ, gXYZ, bXYZ, D50-adapted) of the profile neg_process embeds in
// its sRGB outputs, Elle Stone's sRGB V2 (3rd_party/elle_icc_profiles; the sRGB-TRC and the gamma 1.0 profiles carry
// the same colorants). So the codes written mean, through the embedded profile, the PCS they were computed from, and
// kPcsWhite maps to (1, 1, 1): a colour-managed reader gets the PCS back within quantisation from srgb-g10, and from
// srgb within that and the difference between the profile's 4096-entry TRC table and the analytic sRGB curve the codes
// are encoded with. It replaces Bradford D50 -> D65 followed by the textbook XYZ -> sRGB matrix, which differ from it
// in the fourth decimal. The print CLF keeps the standard sRGB (print_emulation.cc print_clf_nodes), as it carries no
// profile. Unclamped; inline so device code can use it.
NEGICC_HD Rgb pcs_to_linear_srgb_unclamped(Rgb v) {
    return { v.r * 3.13427485f + v.g * -1.61727573f + v.b * -0.490724277f,
             v.r * -0.978795611f + v.g * 1.91616171f + v.b * 0.0334533327f,
             v.r * 0.0719770074f + v.g * -0.228984979f + v.b * 1.40571822f };
}

// Linear sRGB clamped to [0, 1].
NEGICC_HD Rgb pcs_to_linear_srgb(Rgb v) {
    const Rgb l = pcs_to_linear_srgb_unclamped(v);
    return { clamp01(l.r), clamp01(l.g), clamp01(l.b) };
}

// Linear sRGB clamped at 0 only, so values above the chart white (the profile's headroom, up to about 1.33 on the
// grey axis) survive for the shoulder.
NEGICC_HD Rgb pcs_to_linear_srgb_pre_shoulder(Rgb v) {
    const Rgb l = pcs_to_linear_srgb_unclamped(v);
    return { fmaxf(0.0f, l.r), fmaxf(0.0f, l.g), fmaxf(0.0f, l.b) };
}

// ---- Stage: output encoding --------------------------------------------------------------------------------
// The sRGB transfer curve of IEC 61966-2-1, one formula in single precision (the renderer, on the host and the device)
// and in double (the tone stage and the shoulder's solve); srgb_decode is its inverse. srgb_pow is powf or pow.
NEGICC_HD float srgb_pow(float x, float y) { return powf(x, y); }
NEGICC_HD double srgb_pow(double x, double y) { return ::pow(x, y); }
template <typename T>
NEGICC_HD T srgb_encode(T l) {
    return (l <= T(0.0031308)) ? (l * T(12.92)) : (srgb_pow(l, T(1) / T(2.4)) * T(1.055) - T(0.055));
}

NEGICC_HD Rgb encode_output(Rgb v, int encoding) {
    if (encoding == ENCODE_SRGB) v = { srgb_encode(v.r), srgb_encode(v.g), srgb_encode(v.b) };
    return { clamp01(v.r), clamp01(v.g), clamp01(v.b) };
}

template <typename T>
NEGICC_HD T srgb_decode(T c) {
    return (c <= T(0.04045)) ? (c * (T(1) / T(12.92))) : srgb_pow((c + T(0.055)) * (T(1) / T(1.055)), T(2.4));
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
    return srgb_decode(cfg.shoulder_c - (cfg.shoulder_c - cfg.shoulder_s_k) * up);
}

NEGICC_HD Rgb apply_shoulder(const PipelineConfig& cfg, Rgb v) {
    if (cfg.encoding != ENCODE_SRGB || !(cfg.shoulder_x_hi > kShoulderOff)) return v;
    return { apply_shoulder_channel(cfg, v.r), apply_shoulder_channel(cfg, v.g), apply_shoulder_channel(cfg, v.b) };
}

// ---- Whole pipeline ----------------------------------------------------------------------------------------
// Returns the corrected RGB in [0, 1]: the input matrix and the knee, before the film profile.
NEGICC_HD Rgb convert_pixel_corrected(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    return apply_knee(apply_matrix(matrix, load_u16(px)), cfg.knee);
}

// Returns the film profile's ICC PCS XYZ (D50); the corrected RGB when there is no profile.
NEGICC_HD Rgb convert_pixel_pcs(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    return apply_profile(cfg.profile, convert_pixel_corrected(cfg, matrix, px));
}

// Returns the pre-shoulder linear sRGB D65 value, clamped at 0 only (0 to about 1.33; chart white is 1).
NEGICC_HD Rgb convert_pixel_linear(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    return pcs_to_linear_srgb_pre_shoulder(convert_pixel_pcs(cfg, matrix, px));
}

// Returns linear sRGB D65 in [0, 1] after the shoulder, before output gamma encoding: what is shown and scored.
NEGICC_HD Rgb to_display(const PipelineConfig& cfg, Rgb pre) {
    const Rgb v = apply_shoulder(cfg, pre);
    return { clamp01(v.r), clamp01(v.g), clamp01(v.b) };
}

NEGICC_HD Rgb convert_pixel_display(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    return to_display(cfg, convert_pixel_linear(cfg, matrix, px));
}

// CIELAB's break and slope as CIE 15 gives them exactly: f(t) = t^(1/3) above kLabEpsilon = (6/29)^3, else
// (kLabKappa t + 16) / 116, with kLabKappa = (29/3)^3; L* = kLabKappa Y below the break. The two branches meet there.
constexpr double kLabEpsilon = 216.0 / 24389.0;
constexpr double kLabKappa = 24389.0 / 27.0;

// ---- Stage: tone (sRGB export only) ------------------------------------------------------------------
// The derivative of CIE Lab's f (the cube root above the linear segment). The constants are CIE 1976 L*a*b*'s exact
// ones (CIE 15; ICC.1:2022 Annex A, equations A.7 to A.12): the break (6/29)^3 = 216/24389 and the linear segment's
// slope 841/108 = 24389 / 27 / 116.
NEGICC_HD float tone_lab_fprime(float t) {
    return t > (float)kLabEpsilon ? (1.0f / 3.0f) / cbrtf(t * t) : (float)kLabKappa / 116.0f;
}

// Y' = f(Y) from the table (tt.n >= 1). Y is clamped to [0, 1] for the lookup (a NaN reads as 0), so the index is in
// [0, n - 1].
NEGICC_HD float tone_table_Y(const ToneTable& tt, float Y) {
    const float y = fminf(fmaxf(Y, 0.0f), 1.0f);
    float t = sqrtf(y) * (float)tt.n;
    int i = (int)t;
    if (i > tt.n - 1) i = tt.n - 1;
    t -= (float)i;
    const float yp = tt.lut[i] + t * (tt.lut[i + 1] - tt.lut[i]);
    return Y > 1.0f ? yp * Y : yp;
}

// The tone curve on one display linear sRGB pixel v: its luminance Y moves to Y' on the white point's neutral, and its
// colour difference from that neutral, v - Y (white 1, 1, 1), is added back scaled by f'(Y) / f'(Y') (at most 8), which
// holds its a* and b* to first order: out = Y' + (v - Y) f'(Y) / f'(Y'). No Lab conversion of the pixel.
// Y is the luminance of the displayed linear sRGB against its own white, independent of the PCS conversion: the Y row
// of OpenCV's cvtColor RGB -> XYZ matrix, the row linear_srgb_to_lab and the tone stage's descriptors read.
NEGICC_HD Rgb apply_tone(const ToneTable& tt, Rgb v) {
    const float y0 = 0.212671f * v.r + 0.715160f * v.g + 0.072169f * v.b;   // as tone_stage.cc's planner reads Y
    const float y1 = tone_table_Y(tt, y0);
    const float sc = fminf(tone_lab_fprime(fmaxf(y0, 0.0f)) / tone_lab_fprime(fmaxf(y1, 1e-9f)), 8.0f);
    return { y1 + sc * (v.r - y0), y1 + sc * (v.g - y0), y1 + sc * (v.b - y0) };
}

// Returns the encoded output in [0,1] for one camera-linear pixel and one 3x3 input matrix (with cfg.tone on an sRGB
// export, the tone curve between the display value and the encoding; any other encoding ignores it).
NEGICC_HD Rgb convert_pixel(const PipelineConfig& cfg, const float* matrix, const uint16_t* px) {
    Rgb v = convert_pixel_display(cfg, matrix, px);
    if (cfg.encoding == ENCODE_SRGB && cfg.tone.lut && cfg.tone.n > 0) v = apply_tone(cfg.tone, v);
    return encode_output(v, cfg.encoding);
}

// ---- Output quantisation -----------------------------------------------------------------------------------
// v must already be in [0, 1] (convert_pixel's output is).
NEGICC_HD uint8_t quantize_u8(float v) { return (uint8_t)roundf(v * 255.0f); }
NEGICC_HD uint16_t quantize_u16(float v) { return (uint16_t)roundf(v * 65535.0f); }

// ---- Scoring: sRGB -> CIELAB (D65) -------------------------------------------------------------------------
// The matrix and the white (0.950456, 1, 1.088754: the matrix's row sums) are those of OpenCV's cvtColor RGB -> Lab
// (its imgproc colour conversions documentation). The break and slope are CIE 15's exact kLabEpsilon and kLabKappa,
// where OpenCV rounds them to 0.008856, 7.787 (kLabKappa / 116) and 903.3.
struct Lab {
    float L, a, b;
};

NEGICC_HD float lab_f(float t) {
    return (t > (float)kLabEpsilon) ? cbrtf(t) : (((float)kLabKappa * t + 16.0f) / 116.0f);
}

// Direct linear sRGB D65 -> CIELAB D65 (skips 6x powf of srgb_encode + srgb_decode).
// Independent of the PCS conversion above: it reads a picture's linear sRGB and gives L*a*b* relative to sRGB's own
// white (the textbook sRGB -> XYZ D65, divided by D65's X and Z, so (1, 1, 1) is L* 100, a* b* 0). It is the colour of
// what is displayed, the space the intent adapters' training targets were made in by the profile-building tools, and is
// kept as it is: making it the inverse of the PCS conversion would move every trained cell. Its break and slope are
// CIE 15's exact ones where the targets' conversion rounds them (0.008856, 903.3): the two differ only below the break,
// by at most 3.3e-5 in L* and 1.6e-4 in a* or b*, far below the 8-bit step the targets were made from.
NEGICC_HD Lab linear_srgb_to_lab(Rgb lin) {
    const float r = lin.r, g = lin.g, b = lin.b;
    const float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
    const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    const float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
    const float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
    const float L = (Y > (float)kLabEpsilon) ? (116.0f * fy - 16.0f) : ((float)kLabKappa * Y);
    return { L, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

// Direct linear sRGB D65 -> CIE L* (1 cbrtf, 0 powf).
NEGICC_HD float linear_srgb_to_L(Rgb lin) {
    const float Y = 0.212671f * lin.r + 0.715160f * lin.g + 0.072169f * lin.b;
    return (Y > (float)kLabEpsilon) ? (116.0f * cbrtf(Y) - 16.0f) : ((float)kLabKappa * Y);
}

NEGICC_HD Lab srgb_to_lab(Rgb c) {
    return linear_srgb_to_lab({ srgb_decode(c.r), srgb_decode(c.g), srgb_decode(c.b) });
}

// ---- Scoring: clipping rules (one definition for every consumer) -------------------------------------------
// An encoded channel is white-clipped at 16-bit code 65400 and above, black-clipped at code 200 and below. The linear
// forms are the same thresholds decoded through the sRGB curve: srgb_decode(65399.5 / 65535) and
// srgb_decode(200.5 / 65535).
NEGICC_HD bool linear_channel_white_clipped(float l) { return l >= 0.9953029f; }
NEGICC_HD bool linear_channel_black_clipped(float l) { return l < 0.00023679829f; }

// L* histograms: kLabBins bins over [0, 100], each 0.1 L* wide.
constexpr int kLabBins = 1000;
NEGICC_HD int lab_bin(float L) {
    int k = (int)(L * (kLabBins / 100.0f));
    return k < 0 ? 0 : (k > kLabBins - 1 ? kLabBins - 1 : k);
}

// ---- Symmetric inner-crop & aspect-preserved DINOv3 geometry -----------------------------------------------
// The centred crop_ratio fraction of a W x H frame (the statistics skip its border), at least one pixel.
struct InnerCrop {
    int x0 = 0, y0 = 0, w = 0, h = 0;
};

// A pass's crop: the given one (the whole W x H image when it is empty) cut to the image, so a crop past an edge reads
// and counts only the image's pixels (it may come out empty).
NEGICC_HD InnerCrop pass_crop(InnerCrop c, int W, int H) {
    if (c.w <= 0 || c.h <= 0) return {0, 0, W > 0 ? W : 0, H > 0 ? H : 0};
    const int x0 = c.x0 < 0 ? 0 : c.x0, y0 = c.y0 < 0 ? 0 : c.y0;
    const long long x1 = (long long)c.x0 + c.w, y1 = (long long)c.y0 + c.h;
    const int xe = x1 > W ? W : (int)x1, ye = y1 > H ? H : (int)y1;
    return {x0, y0, xe > x0 ? xe - x0 : 0, ye > y0 ? ye - y0 : 0};
}

NEGICC_HD InnerCrop compute_inner_crop(int W, int H, float crop_ratio) {
    if (W <= 0 || H <= 0) return {0, 0, 0, 0};
    if (!(crop_ratio < 1.0f)) return {0, 0, W, H};   // 1 or more, or not a number: the whole frame
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

// The DINOv3 input size of a w x h image: its longer side scaled to max_size, both sides rounded to whole
// patch_size patches (grid_w x grid_h of them). The defaults are the backbone's image_size and patch_size
// (models/dinov3-small/backbone.json). A side is at most as many patches as an int holds in pixels (INT_MAX /
// patch_size): only a max_size within patch_size / 2 of INT_MAX reaches that; DinoV3Engine refuses such sizes.
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
    if (gw > INT_MAX / p) gw = INT_MAX / p;
    if (gh > INT_MAX / p) gh = INT_MAX / p;
    return {gw * p, gh * p, gw, gh};
}

// Histogram of the pre-shoulder max(R, G, B) over [0, kMaxHistRange] (the source of the shoulder's x_hi): bins 1/2048
// wide, over a range above the pre-shoulder values (up to about 1.33 on the grey axis).
constexpr int kMaxHistBins = 4096;
constexpr float kMaxHistRange = 2.0f;
NEGICC_HD int max_hist_bin(float v) {
    if (!((v - v) == 0.0f)) return 0;  // NaN or infinite
    const int k = (int)(v * (kMaxHistBins / kMaxHistRange));
    return k < 0 ? 0 : (k > kMaxHistBins - 1 ? kMaxHistBins - 1 : k);
}

// ---- Host: solve the shoulder knee ------------------------------------------------------------------------
// x_hi in (kShoulderOff, kShoulderXHiLimit] enables the shoulder (anything else turns it off). Bisection on the knee
// t_k in [-3, 0] in double precision; the C1 condition changes sign once.
inline void set_shoulder(PipelineConfig& cfg, float x_hi, float n = kShoulderExponent) {
    if (!(n >= 1.0f && n <= 16.0f)) throw std::invalid_argument("shoulder exponent must be in [1, 16]");
    cfg.shoulder_n = n;
    auto off = [&cfg] {
        cfg.shoulder_x_hi = 1.0f;
        cfg.shoulder_x_k = 1.0f;
        cfg.shoulder_t_k = cfg.shoulder_t_hi = 0.0f;
        cfg.shoulder_s_k = 1.0f;
        cfg.shoulder_c = 1.0f;
        cfg.shoulder_y_c = 1.0f;
    };
    if (!(x_hi > kShoulderOff)) {
        off();
        return;
    }
    if (x_hi > kShoulderXHiLimit) x_hi = kShoulderXHiLimit;
    // The ceiling ramps in from the unshouldered white so the curve is continuous in x_hi at the on/off boundary.
    const double ramp = std::min(1.0, ((double)x_hi - 1.0) / (double)kShoulderRamp);
    const double C = 1.0 - (1.0 - (double)kShoulderCeiling) * ramp;
    const double ln2 = std::log(2.0);
    auto slope = [&](double t) { return (1.055 / 2.4) * std::pow(std::exp2(t), 1.0 / 2.4) * ln2; };
    const double t_hi = std::log2((double)x_hi);
    auto f = [&](double t_k) { return t_hi - t_k - (double)n * (C - srgb_encode(std::exp2(t_k))) / slope(t_k); };
    double lo = -3.0, hi = 0.0;
    const bool lo_pos = f(lo) > 0.0;
    // The C1 condition changes sign once on [-3, 0] for every x_hi in range; if it ever did not, bisection would
    // converge to an end of the interval, not a knee. Switch the shoulder off rather than render that.
    if (lo_pos == (f(hi) > 0.0)) {
        off();
        return;
    }
    for (int i = 0; i < 100; ++i) {
        const double mid = 0.5 * (lo + hi);
        if ((f(mid) > 0.0) == lo_pos) lo = mid; else hi = mid;
    }
    const double t_k = 0.5 * (lo + hi);
    cfg.shoulder_x_hi = x_hi;
    cfg.shoulder_t_k = (float)t_k;
    cfg.shoulder_x_k = (float)std::exp2(t_k);
    cfg.shoulder_t_hi = (float)t_hi;
    cfg.shoulder_s_k = (float)srgb_encode(std::exp2(t_k));
    cfg.shoulder_c = (float)C;
    cfg.shoulder_y_c = (float)srgb_decode(C);
}

// The anchored shoulder's x_hi for a frame whose pre-shoulder max channel has its 99.5% point at q: the x_hi whose
// curve maps q to sRGB kShoulderAnchorCode, so the frame's own highlights, not the chart white, set where the roll-off
// starts. 1.0 (off) when q already encodes at or below the anchor code. A longer roll-off (larger x_hi) has a lower
// knee (the C1 join) and so a lower value at q: the value falls monotonically with x_hi, and a bisection on x_hi in
// (kShoulderOff, kShoulderXHiLimit] finds it. Past the limit the limit is used and q lands above the anchor code.
inline float shoulder_x_hi_anchored(float q, float n = kShoulderExponent) {
    const float anchor = kShoulderAnchorCode / 255.0f;
    if (!(q > 0.0f) || !(srgb_encode(q) > anchor)) return 1.0f;
    auto code_at_q = [&](float x_hi) {
        PipelineConfig c{};
        c.encoding = ENCODE_SRGB;
        set_shoulder(c, x_hi, n);
        return srgb_encode(apply_shoulder_channel(c, q));
    };
    float lo = 1.0002f, hi = kShoulderXHiLimit;   // lo: just above kShoulderOff, so set_shoulder turns the shoulder on
    if (code_at_q(hi) > anchor) return hi;
    if (!(code_at_q(lo) > anchor)) return lo;
    for (int i = 0; i < 60; ++i) {
        const float mid = 0.5f * (lo + hi);
        if (code_at_q(mid) > anchor) lo = mid; else hi = mid;
    }
    return hi;
}

}  // namespace negicc

// ====================================================================================================================
// Frame detector: the frame, the strip, the sprocket holes and a film base patch
// ====================================================================================================================

// LibRaw-free film frame detector for camera-scanned (or flatbed-scanned) film, in negative colour.
//
// Finds the picture area of the frame to crop (the "primary" frame), the other frames visible on the strip
// (neighbours and partial frames, the "secondaries"), the visible film strip (picture plus rebate, the "film base"
// rectangle), the 35mm sprocket holes, the film's tilt, the strip direction and the frame orientation, and a flat
// patch of unexposed film base for the base read from the scan. 35mm, 645, 6x6, 6x7 and 6x9 frames, given by the
// caller or tried in turn (FILM_FORMAT_AUTO). Film on bare light without a holder, in a strip holder or in a frame
// holder.
//
// Input: the image is read in place through FilmImage, three channel lattices that may point into one CFA mosaic
// (one sample per photosite) or into an RGB image (interleaved or planar). Nothing of the input is copied; the
// detector reads it in three ways: a box-averaged downsample to a work image of at most params.work_size pixels on
// the long side, a few thousand bilinear samples along the primary frame's edges (refinement), and every sample
// inside each candidate film base patch (film_sample_patch). Intermediate buffers are work-image sized, besides the
// patch samples (see film_detector.cc for each one).
//
// Output coordinates are image pixels: the pixel grid FilmImage::width x FilmImage::height (the sensor for a CFA
// mosaic, the decoded image for an RGB image), x right and y down, pixel (i, j) covering [i, i + 1) x [j, j + 1).

// The median absolute deviation's factor to a normal standard deviation, 1 / Phi^-1(3/4) to four decimals (Rousseeuw
// and Croux, 1993).
constexpr double kMadToSigma = 1.4826;

// One colour channel of the input: a regular lattice of uint16_t samples, read in place. Sample (i, j) of the
// channel is data[i * pitch + j * stride] and lies on image pixel (x0 + i * FilmImage::step, y0 + j * step). Its
// value is (sample - black) / white, clamped below at 0.
struct FilmChannel {
  const uint16_t* data = nullptr;
  ptrdiff_t pitch = 1;   // uint16_t elements from sample (i, j) to (i + 1, j)
  ptrdiff_t stride = 0;  // uint16_t elements from sample (i, j) to (i, j + 1)
  int x0 = 0;            // image pixel of sample (0, 0), 0 <= x0, y0 < step
  int y0 = 0;
  float black = 0.0f;    // black level of this channel, subtracted from every sample
};

// The image the detector reads: R, G and B as three lattices of the same step. For an RGB image step is 1 and the
// three channels share the pixel grid; for a CFA mosaic step is the CFA period (2 Bayer, 6 X-Trans) and each channel
// takes one photosite of its colour per period x period cell (see film_image_from_cfa). Each channel has
// samples_x() x samples_y() samples.
struct FilmImage {
  int width = 0;         // image size in pixels
  int height = 0;
  int step = 1;          // image pixels between neighbouring samples of a channel
  float white = 65535.0f;  // sample value (black subtracted) of a clipped photosite: the normalising scale
  FilmChannel ch[3];     // R, G, B
  int samples_x() const;
  int samples_y() const;
};

// An interleaved RGB image of |channels| (3 or 4, R G B first) uint16_t per pixel, |row_stride| uint16_t elements
// per row (0: width * channels); e.g. LibRaw's image[][4] after dcraw_process or a pixel-shift merge. Null pixels or
// fewer than 3 channels give an empty image (no data), which the detector refuses.
FilmImage film_image_from_interleaved(const uint16_t* pixels, int width, int height, int channels,
                                      ptrdiff_t row_stride = 0, float white = 65535.0f);

// Three planes (or three times the same plane for a monochrome image), |row_stride| elements per row each.
FilmImage film_image_from_planes(const uint16_t* r, const uint16_t* g, const uint16_t* b, int width, int height,
                                 ptrdiff_t row_stride, float white = 65535.0f);

// A CFA mosaic: one uint16_t per photosite, |row_stride| elements per row, pixel (0, 0) at |mosaic| (for LibRaw's
// rawdata.raw_image that is the first visible photosite: raw_image + top_margin * raw_pitch / 2 + left_margin).
// |period| is the CFA repeat (2 Bayer, 6 X-Trans) and |color| gives the colour of photosite (row, col) as LibRaw's
// COLOR() does (0 R, 1 G, 2 B, 3 the second G of a Bayer quad). For each of R, G and B the first photosite of that
// colour in the period x period cell (row-major, G before G2) becomes the channel lattice, with
// black[color] as its black level. False when a colour is missing from the cell.
bool film_image_from_cfa(const uint16_t* mosaic, int width, int height, ptrdiff_t row_stride, int period,
                         const std::function<int(int row, int col)>& color, const float black[4], float white,
                         FilmImage& out);

enum FilmFormat { FILM_FORMAT_AUTO = 0, FILM_35MM, FILM_645, FILM_6X6, FILM_6X7, FILM_6X9 };
enum FilmPolarity { FILM_NEGATIVE = 0, FILM_POSITIVE = 1 };

const char* film_format_name(FilmFormat f);
// "35mm" (or "135"), "645", "6x6", "6x7", "6x9"; anything else is FILM_FORMAT_AUTO.
FilmFormat film_format_from_name(const std::string& s);

struct FrameDetectionParams {
  FilmFormat format = FILM_FORMAT_AUTO;  // set by the user for the best accuracy; AUTO tries every format
  FilmPolarity polarity = FILM_NEGATIVE;
  char axis_hint = 0;            // 'h' (or 'H') the strip runs along x, 'v' (or 'V') along y, 0 detect
  float max_rotation_deg = 5.0f; // tilt search range, 0 to 90 degrees (the reported tilt stays within it)
  int work_size = 1024;          // long side of the work image, pixels (at least 256)
  bool refine = true;            // sub-pixel refinement of the primary frame's edges on the full-resolution image
  float crop_inset = 0.005f;     // FrameDetection::crop: inset per side, as a fraction of the crop's width or height
                                 // (at least 0, below 0.5)
  int crop_align = 1;            // FrameDetection::crop: x, y, x + w and y + h multiples of this (the CFA period)
};

// A rectangle turned clockwise by angle_deg about its centre: its corners are
//   (cx, cy) + R(angle) (+-w / 2, +-h / 2),  R = [[cos, -sin], [sin, cos]]
// (y down, so a positive angle turns it clockwise on screen). w is its extent along x before the turn.
struct FilmRect {
  float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f, angle_deg = 0.0f;
  void corners(float xy[8]) const;  // x0 y0 x1 y1 ..., top-left, top-right, bottom-right, bottom-left before the turn
};

struct FilmFrame {
  FilmRect rect;                 // the whole frame (a partial frame reaches past the image border)
  int visible[4] = {0, 0, 0, 0}; // x, y, w, h: bounding box of the part inside the image
  float visible_fraction = 0.0f; // visible picture area / nominal frame area
  bool complete = false;         // the whole frame is inside the image
  int index = 0;                 // position along the strip from the primary (..., -1, 0, +1, ...)
};

struct FrameDetection {
  bool ok = false;
  FilmFrame primary;                  // the frame to crop
  std::vector<FilmFrame> secondaries; // the other frames on the strip, ordered by index
  // The crop: x, y, w, h of the largest axis-aligned rectangle with the primary frame's aspect inside it (the
  // frame is tilted by rotation_deg), inset by params.crop_inset and aligned to params.crop_align.
  int crop[4] = {0, 0, 0, 0};
  bool has_film_base = false;         // the strip extends past the primary frame
  FilmRect film_base;                 // the visible strip: picture and rebate
  std::vector<FilmRect> sprockets;    // 35mm perforations found beside the primary frame
  float rotation_deg = 0.0f;          // tilt of the film (FilmRect convention)
  char axis = '?';                    // 'h' strip along x, 'v' along y, '?' a single square frame, undecided
  std::string orientation;            // "landscape", "portrait" or "square": the primary frame on the image
  FilmFormat format = FILM_FORMAT_AUTO;
  float px_per_mm = 0.0f;             // image pixels per mm of film
  // The film gaps at the primary frame's two ends (before, after it along the strip): their extent along the strip
  // axis in image pixels (x for axis 'h', y for 'v'), within [0, width or height], {-1, -1} where the end is
  // bounded otherwise or open. n_gaps counts the ends that have one.
  int n_gaps = 0;
  int gaps[2][2] = {{-1, -1}, {-1, -1}};
  // A flat patch of unexposed film next to the primary frame (an inter-frame gap or the rebate, tilted with the
  // film), the bounding box (x, y, w, h) of its part inside the image, its median per channel in sample units
  // (black subtracted, not normalised) and the spread of its green density (standard deviation of log10). has_base
  // when that spread is at most 0.025 (flat enough to be unexposed film rather than picture).
  bool has_base = false;
  FilmRect base_patch;
  int base_box[4] = {0, 0, 0, 0};
  float base_rgb[3] = {0.0f, 0.0f, 0.0f};
  float base_std_d = 0.0f;
  bool light_clipped = false;         // bare light around the film is clipped
  // 1: the frame is complete, bounded on both ends, with strong transverse edges and the film's bounds (holder,
  // light, strip end, strip edge, perforations or film base) seen; lower per doubt, each named in reason.
  float conf = 0.0f;
  std::string reason;                 // comma-separated diagnostics ("" when nothing to report)
};

// Detects the film frame in |image|. False when the image is too small or holds no frame, or when the image or params
// break their stated ranges (a white level that is not positive and finite, a black level that is not finite, a
// channel's x0 or y0 outside [0, step), max_rotation_deg or crop_inset outside its range).
bool detect_film_frame(const FilmImage& image, const FrameDetectionParams& params, FrameDetection& out);

// Median per channel (sample units, black subtracted) of the samples whose photosite centre lies inside |patch|
// (image pixels), and the standard deviation of their green log10 density. False when it holds fewer than 32
// samples of a channel, or when the image has no samples or the patch is not finite with a positive size. Used for the
// base patch on the image the caller converts (e.g. the decoded crop in its own colour space; shift the patch's
// centre by the crop's origin, and halve it for a half-size decode).
bool film_sample_patch(const FilmImage& image, const FilmRect& patch, float rgb[3], float* std_d);

namespace negicc {

// Square dilation by r pixels (separable distance scans). m is left as it is when r <= 0, W or H <= 0, or m holds
// fewer than W x H pixels.
void dilate(std::vector<uint8_t>& m, int W, int H, int r);

}  // namespace negicc

// ====================================================================================================================
// Flat field: the gain lookup of the CPU and CUDA decodes, and the gain grid
// ====================================================================================================================

// Scan flat field: the gain lookup shared by the CPU and CUDA decodes (flat_divide), and the gain grid measured on
// the empty gate (FlatField).

// --------------------------------------------------------------------------------------------------------------------
// Gain lookup
// Flat-field gain lookup shared by FlatField::apply (CPU) and the fused crop + flat debayers (CPU and CUDA).
//
// One function, flat_divide, holds the arithmetic, so a decode that applies the flat while it is cut out of the
// sensor (the fused path) gives the same bits as decoding the whole frame, applying the flat, then cutting it:
// both evaluate the same float expressions in the same order. The CPU is built with -ffp-contract=off and nvcc with
// --fmad=false, and float division is IEEE round-to-nearest on both, so CPU and GPU agree bit for bit as well.
// --------------------------------------------------------------------------------------------------------------------

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

// The range of a flat field's gains (FlatField refuses others): within it flat_divide cannot cancel a gain to 0.
constexpr float kFlatGainMin = 1.0f / 64.0f;
constexpr float kFlatGainMax = 64.0f;

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

// --------------------------------------------------------------------------------------------------------------------
// Flat field
// Scan flat field: the light + lens gain measured on the empty gate, applied to decodes in sensor coordinates.
// --------------------------------------------------------------------------------------------------------------------

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
    // Throws std::invalid_argument if the decode size does not match the size the grid was measured on, or rgb is null.
    size_t apply(uint16_t* rgb, int width, int height, bool half, uint16_t clip_level) const;

    // Divides a full-size CFA mosaic in place, before any demosaic (the white-light decode): photosite (x, y) is
    // divided by the gain of its own colour, pattern[(y % 6) * 6 + x % 6] (0 = R, 1 = G, 2 = B; a 2x2 Bayer or a
    // 6x6 X-Trans pattern), sampled at half-size coordinates as apply() samples a full-size decode. pitch: u16 per
    // row of cfa. Throws std::invalid_argument if width / 2 x height / 2 is not the size the grid was measured on,
    // if cfa or pattern is null, or if pitch is below width (rows would overlap).
    void apply_cfa(uint16_t* cfa, size_t pitch, int width, int height, const uint8_t pattern[36]) const;

    // The gain grid and the lookup tables of a width x height decode (half: half-size, else full-size), in host or
    // GPU-visible memory, for flat_divide in the fused crop + flat debayers. Built once per size and kept while the
    // FlatField lives (a few hundred kB). Throws std::invalid_argument as apply() for a decode of another size.
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
    // host()'s tables are one block of host_bytes() at host_block(), for a caller that copies them (to a device) and
    // points a FlatSampleArgs at the copy: every pointer of host() moved by the copy's offset.
    const void* host_block() const { return m_block; }
    size_t host_bytes() const { return m_bytes; }

private:
    void* m_block = nullptr;
    size_t m_bytes = 0;
    FlatSampleArgs m_host, m_device;
};

using FlatFieldPtr = std::shared_ptr<const FlatField>;

// Pixels of an interleaved u16 RGB image with any channel >= clip_level (0 for a null image).
size_t count_clipped(const uint16_t* rgb, size_t pixels, uint16_t clip_level);

// Loads a flat-field .npz file saved by scan_flat_field.py (containing gain.npy float32 [gh, gw, 3] and info.npy JSON).
FlatFieldPtr load_flat_field_file(const std::string& path);

}  // namespace negicc

// ====================================================================================================================
// Image view: the zero-copy CFA view (RgbView), its CPU passes and its CUDA passes
// ====================================================================================================================

// Shared zero-copy CFA downsized view (RgbView), its CPU passes, and the same passes on a CUDA device.
//
// Five-step pipeline architecture:
//   Step 1 (Load & decode): per-channel black subtraction in-place on the CFA mosaic (one Bayer or X-Trans mosaic, or
//     the four mosaics of a Sony pixel shift, all kept in memory).
//   Step 2 (Downsizing): Zero-copy half-size RgbView over the CFA mosaic (r_ptr, g_ptr, g2_ptr, b_ptr, pitch, stride).
//   Step 3 (Crop, rotate, flip & inline ROI search): detect_film_frame operates inline on the Step 2 view;
//     crop_and_orient adjusts pointers, pitch, stride, and sensor flat-field coordinates with zero memory copy.
//   Step 4 (Conversion parameter search): Operates inline on the oriented RgbView (compute_raw_stats, cpu_render,
//     cpu_highlight_clip) with zero large intermediate copies.
//   Step 5 (Full export): The only step that allocates a cropped full-size RGB buffer, fusing grain-aware debayer
//     (or 4x pixel-shift merge), crop/rotate/flip, and flat_divide.

namespace negicc {

// The colour (0 R, 1 G, 2 B, 3 a second green where the pattern marks one) of photosite (r, c) of a Bayer mosaic
// described by LibRaw's idata.filters: two bits per cell, as LibRaw's (and dcraw's) FC(row, col).
inline int bayer_color_at(int r, int c, unsigned int filters) {
    return (filters >> (((((unsigned int)r << 1) & 14u) | ((unsigned int)c & 1u)) << 1)) & 3;
}

// Step 2 & Step 3 zero-copy image view.
// Points directly into a black-subtracted CFA mosaic (Bayer 2x2 or X-Trans 6x6 cells) or an RGB buffer, with
// arbitrary crop, 90-degree rotation, and horizontal/vertical flip represented via (r_ptr, g_ptr, g2_ptr, b_ptr,
// pitch, stride) and a private affine sensor coordinate mapping for inline flat-field division.
struct RgbView {
    int width = 0;
    int height = 0;
    const uint16_t* r_ptr = nullptr;
    const uint16_t* g_ptr = nullptr;
    // The second green of a Bayer 2x2 cell: half-size views average it with g_ptr ((G1 + G2) >> 1), as
    // cpu_preview_debayer and gpu_preview_debayer do. Null for X-Trans and interleaved RGB.
    // film_image_from_cfa (the frame detector's input) still takes G1 only.
    const uint16_t* g2_ptr = nullptr;
    const uint16_t* b_ptr = nullptr;
    ptrdiff_t pitch = 0;   // step in uint16_t elements when x increases by 1
    ptrdiff_t stride = 0;  // step in uint16_t elements when y increases by 1

    uint8_t ch_x0[3] = {0, 0, 0};
    uint8_t ch_y0[3] = {0, 0, 0};
    uint8_t cfa_step = 2;

    FlatSampleArgs flat{};
    std::shared_ptr<const FlatTables> flat_tables;

    bool valid() const { return width > 0 && height > 0 && r_ptr != nullptr && g_ptr != nullptr && b_ptr != nullptr; }

    bool is_contiguous_rgb() const {
        return g2_ptr == nullptr && flat.gain == nullptr && pitch == 3 &&
               r_ptr != nullptr && g_ptr == r_ptr + 1 && b_ptr == r_ptr + 2;
    }

    // Construct a zero-copy strided view over a black-subtracted CFA mosaic (Bayer 2x2 period=2 or X-Trans 6x6
    // period=6). For 4xCFA (SONY_PIXEL_SHIFT_4), pass the first CFA mosaic.
    static RgbView from_cfa_half(const uint16_t* cfa, size_t cfa_pitch, int sensor_w, int sensor_h,
                                 unsigned int filters, const uint8_t xtrans_col36[36] = nullptr,
                                 FlatSampleArgs flat = {},
                                 std::shared_ptr<const FlatTables> flat_pin = nullptr,
                                 int cfa_x0 = 0, int cfa_y0 = 0);

    static RgbView from_cfa_half(const uint16_t* cfa, int sensor_w, int sensor_h,
                                 unsigned int filters, FlatSampleArgs flat = {},
                                 std::shared_ptr<const FlatTables> flat_pin = nullptr) {
        return from_cfa_half(cfa, (size_t)sensor_w, sensor_w, sensor_h, filters, nullptr, flat, std::move(flat_pin));
    }

    // Construct a zero-copy view over an interleaved 16-bit 3-channel RGB or 4-channel (LibRaw image[][4]) buffer.
    // TODO: no half-size form: a subsampling view (pitch = 2 * channels, stride = 2 * row_elems) is not written.
    static RgbView from_interleaved_rgb(const uint16_t* rgb, int w, int h, int channels = 3,
                                        ptrdiff_t row_elems = 0, FlatSampleArgs flat = {},
                                        std::shared_ptr<const FlatTables> flat_pin = nullptr);

    // Step 3: Apply crop (rx, ry, rw, rh in this view's coordinates), clockwise rotation (0, 90, 180, 270),
    // and flips (hflip, vflip) by adjusting pointers, pitch, stride, and sensor affine coordinates (zero copy).
    RgbView crop_and_orient(int rx, int ry, int rw, int rh, int rot_cw, bool hflip, bool vflip) const;

    // Sample pixel (x, y) in [0, width) x [0, height) into px[0..2] (including inline flat_divide if flat is set).
    inline void sample(int x, int y, uint16_t px[3]) const {
        const ptrdiff_t off = (ptrdiff_t)y * stride + (ptrdiff_t)x * pitch;
        px[0] = r_ptr[off];
        px[1] = g2_ptr ? (uint16_t)(((uint32_t)g_ptr[off] + (uint32_t)g2_ptr[off]) >> 1) : g_ptr[off];
        px[2] = b_ptr[off];
        if (flat.gain) {
            const int fx = sensor_x0 + x * sensor_dxx + y * sensor_dyx;
            const int fy = sensor_y0 + x * sensor_dxy + y * sensor_dyy;
            flat_divide(flat, fx, fy, px);
        }
    }

    // Zero-copy conversion to FilmImage for inline detect_film_frame in Step 3. step=1 returns view cell coordinates
    // (width x height); step>=2 returns sensor coordinates (cfa_step*width x cfa_step*height).
    FilmImage to_film_image(float white_level = 65535.0f, int step = 1) const;

    // Materialize into a contiguous width * height * 3 uint16_t buffer (only when explicitly requested).
    void materialize(uint16_t* dst_rgb) const;

    // Whether o is this same view of the same buffers: every pointer and every geometry field equal. The pixels are
    // not read.
    bool same_view(const RgbView& o) const {
        return width == o.width && height == o.height && r_ptr == o.r_ptr && g_ptr == o.g_ptr && g2_ptr == o.g2_ptr &&
               b_ptr == o.b_ptr && pitch == o.pitch && stride == o.stride && ch_x0[0] == o.ch_x0[0] &&
               ch_x0[1] == o.ch_x0[1] && ch_x0[2] == o.ch_x0[2] && ch_y0[0] == o.ch_y0[0] && ch_y0[1] == o.ch_y0[1] &&
               ch_y0[2] == o.ch_y0[2] && cfa_step == o.cfa_step && flat.gain == o.flat.gain &&
               flat.grid_w == o.flat.grid_w && flat.ax_i == o.flat.ax_i && flat.ax_t == o.flat.ax_t &&
               flat.ay_i == o.flat.ay_i && flat.ay_t == o.flat.ay_t && flat_tables == o.flat_tables &&
               sensor_x0 == o.sensor_x0 && sensor_y0 == o.sensor_y0 && sensor_dxx == o.sensor_dxx &&
               sensor_dxy == o.sensor_dxy && sensor_dyx == o.sensor_dyx && sensor_dyy == o.sensor_dyy;
    }

private:
    // Affine mapping from view pixel (x, y) to unoriented half-size sensor pixel (fx, fy) for flat_divide:
    //   fx = sensor_x0 + x * sensor_dxx + y * sensor_dyx
    //   fy = sensor_y0 + x * sensor_dxy + y * sensor_dyy
    int sensor_x0 = 0, sensor_y0 = 0;
    int sensor_dxx = 1, sensor_dxy = 0;
    int sensor_dyx = 0, sensor_dyy = 1;

    bool is_unoriented() const {
        const int d = (cfa_step == 6) ? 3 : 1;
        return sensor_x0 == 0 && sensor_y0 == 0 &&
               sensor_dxx == d && sensor_dxy == 0 && sensor_dyx == 0 && sensor_dyy == d;
    }
};

// The taps of a resample of one axis from n source samples to m (s = n / m): for output d, an area average over the
// source interval [d s, (d + 1) s) where the axis shrinks (s >= 1), else a bilinear pair at the position
// (d + 0.5) s - 0.5 held inside [0, n - 1], so the border repeats the edge sample and no weight is negative.
// The resample of a crop to the network's size takes each axis's taps (compute_raw_stats, compute_frame_stats).
template <typename T>
inline std::vector<std::vector<std::pair<int, T>>> resample_taps(int n, int m, T s) {
    std::vector<std::vector<std::pair<int, T>>> taps((size_t)m);
    for (int d = 0; d < m; ++d) {
        auto& t = taps[(size_t)d];
        if (s >= T(1)) {
            const T a = d * s, b = (d + 1) * s;
            const int i0 = (int)std::floor(a), i1 = std::min(n, (int)std::ceil(b));
            for (int i = i0; i < i1; ++i) {
                const T w = std::min(b, (T)(i + 1)) - std::max(a, (T)i);
                if (w > T(0)) t.emplace_back(i, w);
            }
        } else {
            const T f = std::clamp((d + T(0.5)) * s - T(0.5), T(0), (T)(n - 1));
            const int i0 = (int)std::floor(f), i1 = std::min(i0 + 1, n - 1);
            const T w1 = f - (T)i0;
            t.emplace_back(i0, T(1) - w1);
            if (w1 > T(0)) t.emplace_back(i1, w1);
        }
    }
    return taps;
}

// Summary produced by compute_raw_stats (called by the companion application).
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

// Positive image statistics accumulated over an inner crop during render: the RGB histogram of the picture written
// (through cfg.tone when it has one, as convert_pixel), the clip shares and the Lab statistics of the display value
// before the tone stage.
struct PositiveStats {
    uint32_t hist_rgb[3][256] = {};
    uint32_t hist_L[kLabBins] = {};
    float white_clip = 0.0f;          // % of inner-crop pixels with any channel white-clipped
    float black_clip = 0.0f;          // % of inner-crop pixels with all channels black-clipped
    float mid_a = 0.0f, mid_b = 0.0f; // zone means (with empty-zone fallback applied)
    float hi_a  = 0.0f, hi_b  = 0.0f;
    float sh_a  = 0.0f, sh_b  = 0.0f;
};

// The q point (0..1) of a max-channel histogram, linearly interpolated within its bin.
inline double max_hist_quantile(const uint64_t* hist, double q) {
    double total = 0.0;
    for (int i = 0; i < kMaxHistBins; ++i) total += (double)hist[i];
    if (total <= 0.0) return 0.0;
    const double rank = q * total;
    double accum = 0.0;
    for (int i = 0; i < kMaxHistBins; ++i) {
        if (accum + (double)hist[i] >= rank) {
            const double frac = hist[i] > 0 ? (rank - accum) / (double)hist[i] : 0.5;
            return ((double)i + (frac < 0.0 ? 0.0 : (frac > 1.0 ? 1.0 : frac))) * (kMaxHistRange / kMaxHistBins);
        }
        accum += (double)hist[i];
    }
    return kMaxHistRange;
}

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
    // Optional: n * kMaxHistBins counts of the pre-shoulder max(R, G, B) over [0, kMaxHistRange] per candidate (the
    // source of the shoulder's x_hi). Filled when non-null. The clip and exceedance counts read the pre-shoulder
    // value; the L* median (out[4]) is post-shoulder.
    uint64_t* max_hist = nullptr;
    // Optional: n floats, the percent of the crop with any channel black-clipped (linear_channel_black_clipped), for
    // the search's shadow alternative. Filled when non-null; out is the same either way.
    float* black_out = nullptr;
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
    size_t cfa_pitch = 0;  // 0 defaults to src_w
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

// Called at the start of every cpu_render and cpu_highlight_clip when set (kind 0: render, 1: clip); null by default.
// A check aid: neg_process --check_renderer counts with it the passes the bundle search runs outside its renderer
// (negicc.h PassAudit).
extern void (*cpu_pass_hook)(int kind);
void cpu_render(const RenderArgs& a);
void cpu_intent_loss(const IntentLossArgs& a);
void cpu_highlight_clip(const HighlightClipArgs& a);
void cpu_lab_stats(const LabStatsArgs& a);

// The same two passes on a CUDA device (gpu_passes.cu, USE_CUDA=1 builds). gpu_device_available() is true when a device
// is present and runs the kernels this build has (CUDA_ARCH); without CUDA it is false and the others throw. A failed
// pass or copy throws GpuError. Every pointer the pass reads per pixel is device memory: in, or in_view's channel
// pointers (an interleaved RGB view without a flat field), cfg's tables, and a render's out. A render's stats, and a
// clip pass's matrices, out and max_hist, are host memory. The GPU renderer (negicc.h GpuRenderer)
// makes those copies; gpu_malloc, gpu_free, gpu_upload and gpu_download are the device memory it uses.
struct GpuError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
bool gpu_device_available();
void* gpu_malloc(size_t bytes);
void gpu_free(void* p);
void gpu_upload(void* dst_device, const void* src_host, size_t bytes);
void gpu_download(void* dst_host, const void* src_device, size_t bytes);
void gpu_render(const RenderArgs& a);
void gpu_highlight_clip(const HighlightClipArgs& a);

void cpu_preview_debayer(const PreviewDebayerArgs& a);
// -----------------------------------------------------------------------------------------------------------------
// Grain-aware debayer (single-shot RGGB scans under a narrowband RGB light)
//
// Edge-directed demosaicing (PPG, AHD, VNG) picks an interpolation direction per pixel from local gradients. On a
// developed colour negative most high-frequency detail is dye-cloud grain, which is independent in the three
// emulsion layers, so those gradients are mostly noise: the hard per-pixel direction switch turns grain into
// maze-like artifacts, and the crosstalk matrix and the large mask-balance gains of the inversion amplify the
// resulting chroma errors. Bilinear avoids the artifacts but blurs the grain and zippers along edges.
//
// This debayer instead:
//   1. Interpolates green from the vertical and horizontal Hamilton-Adams estimates, blended with soft weights
//      1 / (eps + gradient^2). eps = 4096 (64 DN, squared) is large next to grain gradients, so grain barely steers the
//      blend while real edges still do.
//   2. Interpolates red and blue as colour ratios (R + k) / (G + k) and (B + k) / (G + k), k = 16 DN, rather than
//      as colour differences. Transmission is multiplicative, so a ratio error stays a constant relative error,
//      which the -log(T) inversion turns into a uniform offset instead of one that grows in the dense areas.
//   3. Falls back to bilinear red and blue where the photosite or its green is within 64 DN of white_level, where
//      a clipped channel makes the ratios meaningless.
//
// Against the 4-shot pixel-shift ground truth (Portra 400, Sony A7RM4; see README.md, Demosaicing): +1.22 dB
// positive PSNR over bilinear and +0.53 dB over LibRaw DCB, with half bilinear's R-G zippering.
//
// Under white light the blue channel is too noisy for the ratios; those scans decode through LibRaw instead.
// -----------------------------------------------------------------------------------------------------------------
// Throws std::invalid_argument when white_level or eps is not finite.
void cpu_grain_aware_debayer(const GrainAwareDebayerArgs& a);
// Throws std::invalid_argument when white_level is outside a 16-bit sample's range (0 to 65535).
void cpu_pixel_shift_merge(const PixelShiftMergeArgs& a);
// Every debayer and merge here and below throws std::invalid_argument when it would write a pixel (a positive dst_w and
// dst_h, or n > 0) from a mosaic without a positive src_w and src_h, or with a mosaic, the positions or rgb_out null.
// The same pixels as cpu_grain_aware_debayer and cpu_pixel_shift_merge, for the n output positions xy (x2, y2 pairs in
// a.dst_w x a.dst_h) only, written in that order to a.rgb_out (n x 3): the tone stage's clip check reads the
// pixels it samples without demosaicing the frame.
void cpu_grain_aware_debayer_sampled(const GrainAwareDebayerArgs& a, const int32_t* xy, long long n);
void cpu_pixel_shift_merge_sampled(const PixelShiftMergeArgs& a, const int32_t* xy, long long n);
// The sensor photosite (row, col) of each output position xy (x2, y2) of a frame decoded from region (rx, ry, rw, rh),
// turned rot_cw and flipped to dst_w x dst_h: the mapping the debayer kernels use, written as pairs to rc.
void map_frame_to_sensor(int rx, int ry, int rw, int rh, int rot_cw, bool hflip, bool vflip, int dst_w, int dst_h,
                         const int32_t* xy, long long n, int32_t* rc);

// The same three on a CUDA device (gpu_passes.cu, USE_CUDA=1 builds; they throw GpuError without one). Every pointer in
// the arguments is device memory: the mosaics, the flat field's tables and rgb_out. The caller puts them there.
// neg_process calls the debayer and the merge for its full-size image. It does not call gpu_preview_debayer, whose
// half-size image it reads as a view (RgbView::from_cfa_half): that one is here for an application that shows the
// half-size image from device memory, beside its CPU twin.
void gpu_preview_debayer(const PreviewDebayerArgs& a);
void gpu_grain_aware_debayer(const GrainAwareDebayerArgs& a);
void gpu_pixel_shift_merge(const PixelShiftMergeArgs& a);
// On the GPU, the pixels of cpu_grain_aware_debayer_sampled and cpu_pixel_shift_merge_sampled for the clip check's
// sample laid out by its blocks (the 2 x 2 at the origin of every step x step
// block inside the crop [cx0, cx1) x [cy0, cy1) of the a.dst_w x a.dst_h frame), in the order of that list of
// positions, the positions computed on the device: a.rgb_out device-visible, sampled_grid_count(...) x 3 (0 for a step
// not above 0; the crop's part below 0 holds no positions).
long long sampled_grid_count(int step, int cx0, int cy0, int cx1, int cy1);
void gpu_grain_aware_debayer_grid(const GrainAwareDebayerArgs& a, int step, int cx0, int cy0, int cx1, int cy1);
void gpu_pixel_shift_merge_grid(const PixelShiftMergeArgs& a, int step, int cx0, int cy0, int cx1, int cy1);

// -----------------------------------------------------------------------------------------------------------------
// The program's thread count when none is given: the smaller of kDefaultThreadCap and the physical cores this process
// may run on.
//
// OpenMP's own default is one thread per hardware thread. On a machine with many cores that takes all of them for a
// small gain, and with two hardware threads per core it is slower than one per core (AGENTS.md, "Parallel Regions").
// Every region takes the same count, so the default is set once for the program.
//
// kDefaultThreadCap is the owner's number, chosen from this measurement on a 192-thread machine (96 cores): six
// frames, medians of three runs, the whole conversion's time and the CPU seconds it used:
//    8 threads  12.1 s   57 CPU s
//   16 threads  10.3 s   62 CPU s
//   32 threads   9.1 s   73 CPU s
//   96 threads   7.7 s  155 CPU s
// A machine with fewer cores runs one thread per core (a Jetson Orin Nano: 6).
//
// default_thread_count() counts the physical cores among the CPUs the process may run on (its affinity mask, from the
// kernel's topology; where that cannot be read, OpenMP's count of available CPUs) and returns the smaller of that and
// the cap.
// apply_default_thread_count() sets it as the OpenMP thread count unless OMP_NUM_THREADS is set, and returns the count
// in effect. A program calls it once, before its first parallel region.
// -----------------------------------------------------------------------------------------------------------------
inline constexpr int kDefaultThreadCap = 16;
int default_thread_count();
int apply_default_thread_count();

}  // namespace negicc

// ====================================================================================================================
// Input curves: the fitted G and B input curves of white light
// ====================================================================================================================

// Fitted G and B input curves: per channel, on the value v entering the
// profile's input TRC (after the input matrix and the knee),
//
//   u = log2(v / ref)  for v < ref,   s = 1 + a + k tanh((u - u_med) / 1 stop),   v' = ref 2^(u s);   v >= ref: v
//
// ref is the film base at that point and u_med the median over the intent grid's cells of min(u, 0), both at the E, g
// and b the conversion renders with (anchor_curves): the fit anchors each candidate at its own g and b, and the
// re-meter each E it tries, so the recorded curves pivot on the film base of the export's parameters. A replay at the
// recorded E, g and b keeps the recorded ref and u_med. The knobs (--brightness, --tint, --warmth) move E, g and b
// after the curves were anchored: there the curves are re-anchored at the knobbed E, g and b (ref and u_med moved by
// what the frame's film base and grid move between the two anchors) and the input tables composed again from the
// bracket's un-curved ones (neg_process's reanchor_input_curves), so the curves stay on the same film densities
// instead of sliding along the table with E. The combined profile is the bracket
// profile with its G and B input tables replaced by in(curve(v)) sampled on the table's own entries; red, the matrix,
// the cLUT and the output curves are unchanged. White light only: with a crosstalk matrix the film base does not mark
// the dark end of every channel.
//
// Header-only: neg_process and negicc-station compile the same code, so both front ends fit the same curves.

namespace negicc {

struct InputCurves {
  // [0] = G, [1] = B
  float a[2] = {0.0f, 0.0f};
  float k[2] = {0.0f, 0.0f};
  float ref[2] = {1.0f, 1.0f};
  float u_med[2] = {0.0f, 0.0f};
  // The densest grid cell's u (<= 0): the low end of the frame's density range. Recorded only (the shape is checked
  // over the whole input table, curve_shape_ok); replay does not need it.
  float u_lo[2] = {0.0f, 0.0f};
  // Below u_lo the curve stops bending and keeps the offset it has there (slope 1 in stops): its departure from the
  // base TRC beyond the frame's density range is the one at u_lo. Off: the curve's form continues to the table's end.
  bool hold = false;
  float u_hold(int i) const { return hold ? u_lo[i] : -INFINITY; }
  bool active() const { return a[0] != 0.0f || k[0] != 0.0f || a[1] != 0.0f || k[1] != 0.0f; }
};

// f(u) = u s(u), held below u_hold at f(u_hold) + (u - u_hold) (u_hold = -inf: not held).
inline float curve_stops(float u, float u_med, float a, float k, float u_hold = -INFINITY) {
  if (u < u_hold) return u_hold * (1.0f + a + k * std::tanh(u_hold - u_med)) + (u - u_hold);
  return u * (1.0f + a + k * std::tanh(u - u_med));
}

inline float input_curve(float v, float ref, float u_med, float a, float k, float u_hold = -INFINITY) {
  if (!(v < ref) || (a == 0.0f && k == 0.0f)) return v;
  const float vv = std::max(v, 1e-12f);
  const float u = std::min(std::log2(vv / ref), 0.0f);
  return ref * std::exp2(curve_stops(u, u_med, a, k, u_hold));
}

// out[i] = in(curve(i / (n - 1))), in sampled with apply_trc's linear interpolation.
inline void compose_input_table(const float* in, int n, float ref, float u_med, float a, float k, float* out,
                                float u_hold = -INFINITY) {
  for (int i = 0; i < n; ++i) {
    float v = input_curve(static_cast<float>(i) / static_cast<float>(n - 1), ref, u_med, a, k, u_hold);
    v = std::min(1.0f, std::max(0.0f, v));
    out[i] = apply_trc(v, in, n);
  }
}

// The combined profile's G and B input tables (owned) and the ProfileData pointing at them; red and everything else
// point at the base profile, which must outlive it.
struct CombinedProfile {
  std::vector<float> in_g, in_b;
  ProfileData data{};

  void build(const ProfileData& base, const InputCurves& c) {
    data = base;
    if (!c.active() || !base.has_profile) return;
    const int ng = base.in_trc_size[1], nb = base.in_trc_size[2];
    if (ng > 1 && base.in_trc[1]) {
      in_g.resize(ng);
      compose_input_table(base.in_trc[1], ng, c.ref[0], c.u_med[0], c.a[0], c.k[0], in_g.data(), c.u_hold(0));
      data.in_trc[1] = in_g.data();
    }
    if (nb > 1 && base.in_trc[2]) {
      in_b.resize(nb);
      compose_input_table(base.in_trc[2], nb, c.ref[1], c.u_med[1], c.a[1], c.k[1], in_b.data(), c.u_hold(1));
      data.in_trc[2] = in_b.data();
    }
  }
};

// ref and u_med for G and B: the film base and the grid's cells through the merged input matrix at (e, g, b) and the
// knee. merged: the 3x3 input matrix before the per-candidate rows (candidate_matrix); film_base: camera units (as the
// grid). grid: n cells of 16-bit camera-linear RGB.
inline void input_curve_reference(const float* merged, float e, float g, float b, float knee, const float film_base[3],
                                  const uint16_t* grid, int n, InputCurves& c) {
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  const Rgb fb = apply_matrix(m, {film_base[0] / 65535.0f, film_base[1] / 65535.0f, film_base[2] / 65535.0f});
  const float refs[2] = {knee_compress(fb.g, knee), knee_compress(fb.b, knee)};
  std::vector<float> ug(n), ub(n);
  for (int i = 0; i < n; ++i) {
    const Rgb v = apply_matrix(m, load_u16(grid + 3 * i));
    const float vg = std::max(knee_compress(v.g, knee), 1e-12f), vb = std::max(knee_compress(v.b, knee), 1e-12f);
    ug[i] = std::min(std::log2(vg / std::max(refs[0], 1e-12f)), 0.0f);
    ub[i] = std::min(std::log2(vb / std::max(refs[1], 1e-12f)), 0.0f);
  }
  auto median = [](std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    float hi = v[h];
    if (v.size() % 2) return hi;
    const float lo = *std::max_element(v.begin(), v.begin() + h);
    return 0.5f * (lo + hi);
  };
  for (int ch = 0; ch < 2; ++ch) c.ref[ch] = refs[ch];
  c.u_lo[0] = ug.empty() ? 0.0f : *std::min_element(ug.begin(), ug.end());
  c.u_lo[1] = ub.empty() ? 0.0f : *std::min_element(ub.begin(), ub.end());
  c.u_med[0] = median(ug);
  c.u_med[1] = median(ub);
}

// What anchors a curve on a frame: the 3x3 input matrix before the per-candidate rows (candidate_matrix), the knee,
// the film base (camera units, as the grid) and the grid's n cells of 16-bit camera-linear RGB.
struct CurveFrame {
  const float* merged = nullptr;
  float knee = 0.95f;
  float film_base[3] = {0.0f, 0.0f, 0.0f};
  const uint16_t* grid = nullptr;
  int n = 0;
  // The neutral-patch head's p(neutral) per grid cell (n values, DinoV3InferenceResult::neutral_p), or null when the
  // head did not run. The joint fit's scene-neutral anchors come from it (JointCurveModel::set_scene_neutral).
  const float* neutral_p = nullptr;
};

// shape's a and k anchored at (e, g, b): ref, u_med and u_lo from the frame there (input_curve_reference).
inline InputCurves anchor_curves(const InputCurves& shape, const CurveFrame& f, float e, float g, float b) {
  InputCurves c = shape;
  input_curve_reference(f.merged, e, g, b, f.knee, f.film_base, f.grid, f.n, c);
  return c;
}

// One channel's curve over the whole input table it is composed into (compose_input_table): at every entry
// v = i / (n - 1) below ref, u = log2(v / ref), the curve is monotonic (denser film never renders lighter in this
// channel) and within max_dev stops of the base TRC (|u s(u) - u|). The whole table is checked, not only the frame's
// density range [u_lo, 0] (the densest grid cell): pixels denser than any grid cell land below it, where the curve
// could otherwise fold over or move arbitrarily far from the TRC. A hold point above 0 (above ref, where the curve does
// not act) is refused: every entry below ref is then held, and the curve meets the identity at ref with a step of
// f(u_hold) - u_hold stops. The identity (a = k = 0) always passes; any other curve needs a, k, u_med and ref finite and
// ref above 0.
inline bool curve_shape_ok(float a, float k, float u_med, float ref, int table_n, float max_dev,
                           float u_hold = -INFINITY) {
  if (a == 0.0f && k == 0.0f) return true;
  if (!(u_hold <= 0.0f)) return false;
  // A curve is numbers: a, k, its median and its reference finite, the reference above 0 (the film base's level).
  if (!(std::isfinite(a) && std::isfinite(k) && std::isfinite(u_med) && std::isfinite(ref) && ref > 0.0f)) return false;
  if (table_n < 2) return true;
  // f at the entry above; the table is walked from ref down, from f = 0 there (the identity at and above ref), so the
  // curve stays below ref as it leaves it.
  float prev = 0.0f;
  for (int i = table_n - 1; i >= 1; --i) {
    const float v = static_cast<float>(i) / static_cast<float>(table_n - 1);
    if (!(v < ref)) continue;
    const float u = std::log2(v / ref);
    const float f = curve_stops(u, u_med, a, k, u_hold);
    if (!(f < prev) || std::fabs(f - u) > max_dev) return false;
    prev = f;
  }
  // Entry 0 (v = 0, which input_curve reads at 1e-12): monotonic too. The departure bound is not applied there: tens of
  // stops below the table's first step every curve with a != k departs that far, and the entry reads 0 either way.
  return !(ref > 1e-12f) || curve_stops(std::log2(1e-12f / ref), u_med, a, k, u_hold) < prev;
}

// The smallest sigma the losses divide by. The intent field's sigma is checked positive and finite where it is
// predicted (SearchFrame::predict), but a sigma just above 0 would still let one cell's residual swamp the fit; the
// floor is far below any sigma DINOv3 predicts (a few L* or a*/b* units), so it never moves an ordinary loss.
constexpr float kSigmaFloor = 1e-3f;

// The intent loss of the grid through cfg (its profile possibly combined) at (e, g, b): mean over cells of
// |(Lab - mu) / sigma|^2, the sum of cpu_intent_loss's three terms (3 z2 in negicc.h's terms).
// The loss's lightness and colour parts: mean over cells of ((L - mu_L) / sigma_L)^2, and of the a* and b* terms.
struct LossParts {
  double L = 0.0, ab = 0.0;
  double total() const { return L + ab; }
};

inline LossParts grid_intent_loss_parts(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                                        const uint16_t* grid, int n, const float* mu, const float* sigma) {
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  LossParts s;
  for (int i = 0; i < n; ++i) {
    const Rgb lin = convert_pixel_linear(cfg, m, grid + 3 * i);
    const Lab lab = linear_srgb_to_lab(to_display(cfg, lin));
    const float dL = (lab.L - mu[i * 3]) / std::max(sigma[i * 3], kSigmaFloor);
    const float da = (lab.a - mu[i * 3 + 1]) / std::max(sigma[i * 3 + 1], kSigmaFloor);
    const float db = (lab.b - mu[i * 3 + 2]) / std::max(sigma[i * 3 + 2], kSigmaFloor);
    s.L += static_cast<double>(dL) * dL;
    s.ab += static_cast<double>(da) * da + static_cast<double>(db) * db;
  }
  if (n > 0) {
    s.L /= n;
    s.ab /= n;
  }
  return s;
}

// The lowest g or b a fit may reach (the gain bounds below g0, b0 are floored here).
constexpr float kGainFloor = 1e-3f;

struct CurveFitOptions {
  // Safety limits, which act only where a fit runs away: the largest per-channel slope change |a|, the largest
  // departure of each curve from the base TRC over the whole table (curve_shape_ok), and the gains' range around
  // Auto's, g in [g0 * 2^-s, g0 * 2^s] and the same for b.
  float curve_bound = 0.35f;
  float max_dev_stops = 2.2f;
  float g_bound_stops = 0.75f;
  float b_bound_stops = 1.25f;
  // Penalty on the curve numbers, joint_lambda (a_G^2 + a_B^2), added to the joint fit's objective. A numerical
  // constant, not a tuning knob: the gains and the curves' slope terms (1 + a) move the residuals almost alike, so
  // without it the Levenberg-Marquardt normal matrix is nearly singular along that direction.
  float joint_lambda = 1.5f;
  bool hold_slope = true;       // InputCurves::hold for the fitted curves (on: Alpha Lam, 2026-11-05)
  // The joint fit (fit_input_curves_joint): E solved for each candidate so the grid's median L* equals Auto's. Off:
  // E stays at Auto's (the measurement of the solver against the pattern search).
  bool solve_e = true;
  // The tolerance, in L*, of each candidate's E solve (JointCurveModel::solve_e). The Jacobian's probes move a
  // parameter by 1e-3, which moves the grid's median L* by about 0.01: a tolerance of 1e-3 would leave the probe's E
  // either unchanged or moved by up to the tolerance, so the derivative through E would be noise. 2e-5 is a few float
  // steps of L* near 50.
  float e_tol_L = 2e-5f;
};

// ---- The joint fit -------------------------------------------------------------------------------------------
//
// g, b and the G and B curves' bend a_G, a_B (k is 0) in one Levenberg-Marquardt solve, with E tied to the lightness
// constraint: each candidate is scored at the E that renders the grid's median L* at target_L (Auto's), so the fit
// chooses its colour at the lightness the export will have. It sets both the scene-neutral balance (the neutral head's
// anchors) and the overall magenta-green balance (the search's intent field), on nine residuals:
//   r0, r1: the anchors' mean a* and b* toward (1 - w) of Auto's, w = m / (m + kEvidenceHalfMass) the evidence
//           shrinkage (m the sum of p^2 over the cells in range); r0 is divided by the anchors' mean a*'s standard
//           deviation, sqrt((h^2 + m) / (m + h)) kHeadSystematicSd (h = kEvidenceHalfMass), so that at high evidence
//           the anchors are trusted only as far as the head's systematic error allows;
//   r2, r3: the anchors' slopes of a* and b* over L* toward (1 - w s) of Auto's, s the slope shrink's share
//           (kSlopeSpanL), weighted 1/2 sqrt(vL / (vL + 100));
//   r4:     the intent field's mean a* residual, weighted by each cell's 1 / sigma_a^2 over every fit cell, less the
//           share a slope change moves it by (the field's mean L* sits away from the anchors'), less kGreenLeanSe of
//           its standard error (block sums over 4 x 4 cells, detrended in L*); divided by sqrt(se^2 +
//           kHeadSystematicSd^2);
//   r5, r6: the never-greener guard's hinges, on the guarded pass only (JointFitGuard);
//   r7, r8: sqrt(CurveFitOptions::joint_lambda) times a_G and a_B.
// r0, r1 and r4 carry the factor 1/1.5 the residuals have always had (so the damping and the guard weight keep their
// scale). The fit is evaluated on a per-cell model:
//   - each cell's three matrix rows are taken once (merged · rgb), and a candidate's (e, g, b) only scales them;
//   - the curve is applied to each cell's G and B value analytically, in(curve(v)), in place of composing the two
//     4096-entry tables per candidate (the export's tables sample the same function at their entries);
//   - E is solved within [e_lo, e_hi] by secant steps on log E warm-started from the previous candidate's E (one to
//     three grid renders), with a bracketed bisection as the fallback; a candidate whose E would leave the bounds is
//     not one (the export could not render it at target_L there);
//   - the solver has a forward-difference Jacobian, box bounds by projection and the shape (curve_shape_ok over the
//     whole table, at each candidate's anchor) by cutting a step back to the furthest point that keeps it.
// The result's curves are anchored at its E, g and b; the caller re-meters E on the real combined tables, which moves
// it by the small difference between the model and the tables.
struct NeutralStats {
  int m = 0;
  float a_bar = 0.0f, b_bar = 0.0f;
  float k_a = 0.0f, k_b = 0.0f;
};

// The joint fit's never-greener guard (fit_input_curves_joint): the grid's mean a* at Auto and at the fit's result,
// over all cells and weighted by the neutral head's p^2, for the sidecar.
struct GreenGuard {
  bool counted[2] = {false, false};  // [all cells, p^2-weighted]: the term had enough cells / evidence to count
  float auto_a[2] = {0.0f, 0.0f};    // Auto's mean a* (g0, b0, no curves, E solved)
  float fit_a[2] = {0.0f, 0.0f};     // the fit's mean a* at its result
  bool binding = false;              // a counted term ended more than the tolerance greener than Auto: the hinge pulled
  // The fit runs first without the guard's residuals; refit: its result broke the bound, so this is the guarded re-run.
  // free_fit_a: the unguarded result's two means (equal to fit_a when it was kept).
  bool refit = false;
  float free_fit_a[2] = {0.0f, 0.0f};
};

struct JointCurveFit {
  bool ok = false;  // a fit ran (8 or more cells in range and a feasible start); else the caller keeps Auto's colour
  // Why no fit ran (ok false): "start" (Auto's g and b with no curves cannot keep target_L within the E bounds) or
  // "cells" (fewer than 8 cells in range); null when it ran.
  const char* skipped = nullptr;
  InputCurves curves;
  float e = 1.0f, g = 1.0f, b = 1.0f;
  // The data loss (the sum of the five squared residuals, without the guard and the damping) at Auto's g and b with no
  // curves (E solved) and at the result.
  double loss_before = 0.0, loss_after = 0.0;
  // What the residuals were built from: the evidence m (sum of p^2 over the cells in range), the shrink weight
  // w = m / (m + kEvidenceHalfMass), the slope shrink's share and the anchors' L* spread, the anchors' mean a*'s
  // standard deviation, the intent field's standard error, and the intent field's mean a* residual at Auto and at the
  // result less the lever share and the lean (r4 with its scale divided out); all three in a*.
  float evidence = 0.0f, shrink = 0.0f, slope_share = 0.0f, span_L = 0.0f, anchor_sd = 0.0f, intent_se = 0.0f;
  float intent_auto = 0.0f, intent_after = 0.0f;
  int evaluations = 0;  // points whose E was solved and whose shape and residuals were evaluated (every one tried)
  int renders = 0;      // grid median renders spent solving E (the evaluations' and the cut-backs')
  int iterations = 0;   // accepted Levenberg-Marquardt steps
  bool converged = false;      // every feasible run settled (a step or gain below tolerance), none stalled
  // False when no seed is feasible (not even Auto's g and b without curves keeps target_L within the E bounds): the
  // result is then Auto's g, b and e0 with no curves, which the caller should not take as a fit.
  bool feasible = true;
  GreenGuard guard;
};

class JointCurveModel {
 public:
  JointCurveModel(const PipelineConfig& cfg, const CurveFrame& f, const float* = nullptr, const float* = nullptr)
      : cfg_(cfg), f_(f), rows_(3 * f.n), L_(f.n), ug_(f.n), ub_(f.n) {
    for (int i = 0; i < f.n; ++i) {
      const Rgb x = load_u16(f.grid + 3 * i);
      for (int r = 0; r < 3; ++r)
        rows_[3 * i + r] = f.merged[3 * r] * x.r + f.merged[3 * r + 1] * x.g + f.merged[3 * r + 2] * x.b;
    }
    for (int r = 0; r < 3; ++r)
      fb_[r] = (f.merged[3 * r] * f.film_base[0] + f.merged[3 * r + 1] * f.film_base[1] +
                f.merged[3 * r + 2] * f.film_base[2]) / 65535.0f;
  }

  // The curves of shape (its a and k) anchored at (e, g, b), as anchor_curves anchors them.
  InputCurves anchored(const InputCurves& shape, float e, float g, float b) {
    InputCurves c = shape;
    const float refs[2] = {knee_compress(e * g * fb_[1], f_.knee), knee_compress(e * b * fb_[2], f_.knee)};
    for (int i = 0; i < f_.n; ++i) {
      const float vg = std::max(knee_compress(e * g * rows_[3 * i + 1], f_.knee), 1e-12f);
      const float vb = std::max(knee_compress(e * b * rows_[3 * i + 2], f_.knee), 1e-12f);
      ug_[i] = std::min(std::log2(vg / std::max(refs[0], 1e-12f)), 0.0f);
      ub_[i] = std::min(std::log2(vb / std::max(refs[1], 1e-12f)), 0.0f);
    }
    c.ref[0] = refs[0];
    c.ref[1] = refs[1];
    c.u_lo[0] = f_.n ? *std::min_element(ug_.begin(), ug_.end()) : 0.0f;
    c.u_lo[1] = f_.n ? *std::min_element(ub_.begin(), ub_.end()) : 0.0f;
    c.u_med[0] = median(ug_);
    c.u_med[1] = median(ub_);
    return c;
  }

  // One cell through the pipeline with the curves applied to its G and B value (c anchored).
  Lab cell_lab(int i, float e, float g, float b, const InputCurves& c) const {
    Rgb v = {knee_compress(e * rows_[3 * i], f_.knee), knee_compress(e * g * rows_[3 * i + 1], f_.knee),
             knee_compress(e * b * rows_[3 * i + 2], f_.knee)};
    v.g = clamp01(input_curve(v.g, c.ref[0], c.u_med[0], c.a[0], c.k[0], c.u_hold(0)));
    v.b = clamp01(input_curve(v.b, c.ref[1], c.u_med[1], c.a[1], c.k[1], c.u_hold(1)));
    return linear_srgb_to_lab(to_display(cfg_, pcs_to_linear_srgb_pre_shoulder(apply_profile(cfg_.profile, v))));
  }

  float median_L(float e, float g, float b, const InputCurves& shape) {
    const InputCurves c = anchored(shape, e, g, b);
    for (int i = 0; i < f_.n; ++i) L_[i] = cell_lab(i, e, g, b, c).L;
    std::nth_element(L_.begin(), L_.begin() + f_.n / 2, L_.end());
    return L_[f_.n / 2];
  }

  // Slope shrink: s0 of sd^2 / (sd^2 + s0^2), in L*; most of a slope measured over an L* spread well under s0 is kept.
  static constexpr float kSlopeSpanL = 6.0f;
  // Evidence shrinkage: the evidence mass (sum of p^2) at which the anchors' pull is half its full strength
  // (w = m / (m + this), inside the fit's residuals), so that a correction resting on very little neutral evidence is
  // not trusted. A policy value, not derived.
  // TODO: the value was checked on few low-evidence frames; re-check it on the scans that arrive through the web
  // portal as they accumulate (their evidence distribution, and the corrections of low-evidence scans at 6, 12, 24
  // and 48).
  static constexpr double kEvidenceHalfMass = 12.0;
  static double shrink_weight(double evidence) { return evidence / (evidence + kEvidenceHalfMass); }

  // The scene-neutral anchors for the sidecar's neutral statistics (neutral_stats_on): every cell Auto renders with
  // 2 < L* < 92 and p(neutral) > 0, weighted by p^2 normalised to sum to 1. Without the head's p, with fewer than 8
  // finite cells in range, or when every p^2 is 0, there are none (has_anchors() false).
  void set_scene_neutral(float e0, float g0, float b0) {
    neu_idx_.clear();
    neu_w_.clear();
    has_anchors_ = false;
    if (f_.n <= 0 || !f_.neutral_p) return;

    const InputCurves zero_c{};
    std::vector<Lab> lab0(f_.n);
    std::vector<float> c0(f_.n);
    for (int i = 0; i < f_.n; ++i) {
      lab0[i] = cell_lab(i, e0, g0, b0, zero_c);
      const float c = std::hypot(lab0[i].a, lab0[i].b);
      c0[i] = std::isfinite(c) && std::isfinite(lab0[i].L) ? c : INFINITY;
    }

    // The cells in range, collected by lightness band ((2, 22], (22, 45], (45, 70], (70, 92)) and used together: the
    // bands balance nothing; they only fix the order of the anchors and so of every sum over them (collecting them in
    // grid order instead moves 28 of 29 test frames by rounding).
    std::vector<int> bands[4];
    int total = 0;
    for (int i = 0; i < f_.n; ++i) {
      if (!std::isfinite(c0[i])) continue;
      const float L = lab0[i].L;
      if (L > 2.0f && L < 92.0f) {
        if (L <= 22.0f) bands[0].push_back(i);
        else if (L <= 45.0f) bands[1].push_back(i);
        else if (L <= 70.0f) bands[2].push_back(i);
        else bands[3].push_back(i);
        ++total;
      }
    }
    if (total < 8) return;

    std::vector<int> b_sel[4];
    std::vector<float> pw(f_.n, 0.0f);  // p^2 per cell
    for (int b = 0; b < 4; ++b) {
      for (int i : bands[b]) {
        const float p = std::clamp(f_.neutral_p[i], 0.0f, 1.0f);
        pw[i] = p * p;
        if (pw[i] > 0.0f) b_sel[b].push_back(i);
      }
    }

    float w_sum = 0.0f;
    for (int b = 0; b < 4; ++b) {
      for (int i : b_sel[b]) {
        neu_idx_.push_back(i);
        neu_w_.push_back(pw[i]);
        w_sum += pw[i];
      }
    }
    if (!(w_sum > 0.0f)) {
      neu_idx_.clear();
      neu_w_.clear();
      return;
    }
    for (float& w : neu_w_) w /= w_sum;
    has_anchors_ = true;
  }

  // set_scene_neutral found anchors (the head's p, with some evidence).
  bool has_anchors() const { return has_anchors_; }

  NeutralStats neutral_stats_on(const PipelineConfig& cfg, float e, float g, float b) const {
    NeutralStats st;
    st.m = (int)neu_idx_.size();
    if (st.m <= 0) return st;
    float m[9];
    candidate_matrix(f_.merged, e, g, b, m);
    std::vector<Lab> labs(st.m);
    double sL = 0.0, sa = 0.0, sb = 0.0;
    for (int k = 0; k < st.m; ++k) {
      const int i = neu_idx_[k];
      const double w = neu_w_[k];
      labs[k] = linear_srgb_to_lab(to_display(cfg, convert_pixel_linear(cfg, m, f_.grid + 3 * i)));
      sL += w * labs[k].L;
      sa += w * labs[k].a;
      sb += w * labs[k].b;
    }
    return compute_neutral_moments(labs, sL, sa, sb, st);
  }

  // E with the grid's median L* at target_L within [e_lo, e_hi] (a larger E renders darker: the median falls with log
  // E, each cell's L* does). Secant steps on log E from e_warm (clamped into the bounds), each new point kept in a
  // bracket of the two nearest points either side of the target; where the secant leaves the bracket or stalls, the
  // bounds the bracket lacks are tested and the bracket bisected. False when the target lies outside the bounds (the
  // median at one bound is already past it): e is that bound, and the point is not one the export can render at
  // target_L.
  bool solve_e(float g, float b, const InputCurves& shape, float target_L, float e_warm, float e_lo, float e_hi,
               float tol, int& renders, float& e) {
    const float lo_t = std::log(e_lo), hi_t = std::log(e_hi);
    auto f = [&](float t) {
      ++renders;
      return median_L(std::exp(t), g, b, shape) - target_L;
    };
    // The bracket: ta the largest log E seen with the median above the target (f > 0), tc the smallest below it.
    bool has_a = false, has_c = false;
    float ta = lo_t, tc = hi_t;
    auto note = [&](float t, float ft) {
      if (ft > 0.0f) {
        if (!has_a || t > ta) ta = t;
        has_a = true;
      } else {
        if (!has_c || t < tc) tc = t;
        has_c = true;
      }
    };
    auto at = [&](float t, bool reached) {
      e = std::exp(t);
      return reached;
    };
    float t0 = std::min(hi_t, std::max(lo_t, std::log(e_warm))), f0 = f(t0);
    if (std::fabs(f0) <= tol) return at(t0, true);
    note(t0, f0);
    // The second point: 0.05 in log E towards the target. Where t0 sits on the bound that step would cross, the
    // target is beyond the bound (the median is monotonic in log E): not reachable.
    const float d = f0 > 0.0f ? 0.05f : -0.05f;
    const float t_bound = d > 0.0f ? hi_t : lo_t;
    if (t0 == t_bound) return at(t0, false);
    float t1 = (d > 0.0f) ? std::min(hi_t, t0 + d) : std::max(lo_t, t0 + d), f1 = f(t1);
    if (std::fabs(f1) <= tol) return at(t1, true);
    note(t1, f1);
    if (t1 == t_bound && (f1 > 0.0f) == (f0 > 0.0f)) return at(t1, false);  // still short at the bound
    for (int it = 0; it < 8; ++it) {
      if (f1 == f0) break;
      const float t2 = t1 - f1 * (t1 - t0) / (f1 - f0);
      const float b_lo = has_a ? ta : lo_t, b_hi = has_c ? tc : hi_t;
      if (!(t2 > b_lo && t2 < b_hi)) break;  // outside the bracket (or the bounds): bisect instead
      t0 = t1;
      f0 = f1;
      t1 = t2;
      f1 = f(t1);
      if (std::fabs(f1) <= tol) return at(t1, true);
      note(t1, f1);
    }
    // The bisection needs both sides: test the bound the bracket lacks (the target beyond it: not reachable).
    if (!has_a) {
      const float fl = f(lo_t);
      if (std::fabs(fl) <= tol) return at(lo_t, true);
      if (fl < 0.0f) return at(lo_t, false);
      note(lo_t, fl);
    }
    if (!has_c) {
      const float fh = f(hi_t);
      if (std::fabs(fh) <= tol) return at(hi_t, true);
      if (fh > 0.0f) return at(hi_t, false);
      note(hi_t, fh);
    }
    for (int it = 0; it < 40 && tc - ta > 1e-6f; ++it) {
      const float mid = 0.5f * (ta + tc), fm = f(mid);
      if (std::fabs(fm) <= tol) return at(mid, true);
      note(mid, fm);
    }
    e = std::exp(0.5f * (ta + tc));
    return true;
  }

 private:
  NeutralStats compute_neutral_moments(const std::vector<Lab>& labs, double sL, double sa, double sb,
                                       NeutralStats& st) const {
    double vL = 0.0, sLa = 0.0, sLb = 0.0;
    for (int k = 0; k < st.m; ++k) {
      const double w = neu_w_[k];
      const double dL = labs[k].L - sL;
      vL += w * dL * dL;
      sLa += w * dL * (labs[k].a - sa);
      sLb += w * dL * (labs[k].b - sb);
    }
    const double ka = 50.0 * sLa / (vL + 25.0);
    const double kb = 50.0 * sLb / (vL + 25.0);
    st.a_bar = (float)sa;
    st.b_bar = (float)sb;
    st.k_a = (float)ka;
    st.k_b = (float)kb;
    return st;
  }

  static float median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    const size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const float hi = v[h];
    if (v.size() % 2) return hi;
    return 0.5f * (*std::max_element(v.begin(), v.begin() + h) + hi);
  }

  PipelineConfig cfg_;
  CurveFrame f_;
  float fb_[3] = {0.0f, 0.0f, 0.0f};
  std::vector<float> rows_, L_, ug_, ub_;
  std::vector<int> neu_idx_;
  std::vector<float> neu_w_;
  bool has_anchors_ = false;
};

// The never-greener guard: the joint fit may move the grid's colour anywhere the neutral term asks, except greener
// than Auto. Over the grid cells Auto renders at L* 2..92 (one band: the whole tone range), the candidate's mean a*
// over all cells, and its mean a* weighted by the neutral head's p^2, may not fall more than kGreenGuardTolAb below
// Auto's. The bound applies to the result, not to the solver's path: the fit runs first without the guard, and that
// result stands when neither counted mean ends more than the tolerance greener than Auto's. Otherwise the fit runs
// again from the same start with each excess in the Levenberg-Marquardt residuals as kGreenGuardWeight x
// max(0, a*_Auto - tol - a*), 0 at Auto and wherever the candidate is not greener: a penalty in the objective rather
// than a clamp afterwards. (A hinge on every evaluation also blocked steps that pass through a slightly greener point
// on the way to a less green result, stalling the fit at its start.) A term counts with at least kGreenGuardMinCells
// cells, the weighted one with an evidence mass (sum p^2) of at least kGreenGuardMinMass.
// Shadows carry no debit (green_guard_cell_a): in both means, a cell Auto renders in the deep shadows that ends greener
// than at Auto counts at Auto's a*, so the fit may correct a shadow cast toward green; a shadow cell that ends more
// magenta keeps its credit, and mid-tones and highlights count as they are.
constexpr float kGreenGuardTolAb = 0.5f;
// The shadow exemption's weight w_s = 1 - smoothstep((L*_Auto - kGreenGuardShadowL0) / kGreenGuardShadowWidth + 0.5):
// 1 in the deep shadows, 0.5 at L* 15, 0 above L* 20. smoothstep(x) = 3x^2 - 2x^3 on [0, 1], the cubic Hermite step.
constexpr double kGreenGuardShadowL0 = 15.0;
constexpr double kGreenGuardShadowWidth = 10.0;
// A cell's a* as the guard's means count it: a_Auto + d, d = a - a_Auto, with a negative d (the cell greener than at
// Auto) scaled by 1 - w_s, w_s the shadow weight of the cell's L* at Auto.
inline double green_guard_cell_a(double a, double a_auto, double L_auto) {
  if (!std::isfinite(a) || !std::isfinite(a_auto)) return a;
  const double d = a - a_auto;
  if (d >= 0.0) return a;
  const double x = std::clamp((L_auto - kGreenGuardShadowL0) / kGreenGuardShadowWidth + 0.5, 0.0, 1.0);
  const double ws = 1.0 - x * x * (3.0 - 2.0 * x);
  return a_auto + (1.0 - ws) * d;
}
constexpr float kGreenGuardWeight = 4.0f;
constexpr double kGreenGuardMinCells = 8.0;
constexpr double kGreenGuardMinMass = 2.0;

// The never-greener guard on one pass of the joint fit: its cells (Auto's render at L* 2..92), their p^2, Auto's a*
// and L* per cell, Auto's two means (ref) and which of them count. set_cells() takes the cells from Auto's render once
// Auto's E is solved; means() gives a candidate's two means; hinge() a mean's excess as the residual the guarded pass
// adds; binds() whether a result breaks the bound.
struct JointFitGuard {
  std::vector<char> in;
  std::vector<double> pw, auto_a, auto_L;  // p^2; Auto's a* (a cell a candidate renders non-finite counts at it) and L*
  double ref[2] = {0.0, 0.0};
  bool counted[2] = {false, false};

  // lab_auto(i): cell i's Lab on Auto's render; neutral_p: the head's p per cell, or null.
  template <typename AutoLab>
  void set_cells(int n, AutoLab lab_auto, const float* neutral_p) {
    in.assign(n, 0);
    pw.assign(n, 0.0);
    auto_a.assign(n, 0.0);
    auto_L.assign(n, 0.0);
    double na = 0.0, wp = 0.0;
    for (int i = 0; i < n; ++i) {
      const Lab l = lab_auto(i);
      if (!std::isfinite(l.L) || !std::isfinite(l.a) || !(l.L > 2.0f && l.L < 92.0f)) continue;
      in[i] = 1;
      auto_a[i] = l.a;
      auto_L[i] = l.L;
      const double p = neutral_p ? std::clamp((double)neutral_p[i], 0.0, 1.0) : 0.0;
      pw[i] = p * p;
      na += 1.0;
      wp += pw[i];
    }
    counted[0] = na >= kGreenGuardMinCells;
    counted[1] = wp >= kGreenGuardMinMass;
  }
  // a_of(i): the candidate's a* of cell i. v: the mean over the guard's cells and the p^2-weighted mean, each cell's
  // a* as green_guard_cell_a counts it.
  template <typename CellA>
  void means(CellA a_of, double v[2]) const {
    double sa = 0.0, na = 0.0, sp = 0.0, wp = 0.0;
    for (int i = 0; i < (int)in.size(); ++i) {
      if (!in[i]) continue;
      // The same cells for every candidate (the ones set_cells counted): a cell a candidate renders non-finite counts
      // at Auto's a*, so the two means never compare different populations.
      double a = a_of(i);
      if (!std::isfinite(a)) a = auto_a[i];
      a = green_guard_cell_a(a, auto_a[i], auto_L[i]);
      sa += a;
      na += 1.0;
      sp += pw[i] * a;
      wp += pw[i];
    }
    v[0] = na > 0 ? sa / na : 0.0;
    v[1] = wp > 0 ? sp / wp : 0.0;
  }
  double hinge(int k, const double v[2]) const {
    return counted[k] ? (double)kGreenGuardWeight * std::max(0.0, ref[k] - kGreenGuardTolAb - v[k]) : 0.0;
  }
  bool binds(int k, const double v[2]) const { return counted[k] && ref[k] - kGreenGuardTolAb - v[k] > 0.0; }
};

// The fifth residual's two stated constants (not derived, not tuned). kHeadSystematicSd: each head's frame-level
// systematic error on the overall a*, taken equal (1 a*) for the neutral head's anchors and the intent field; it caps
// how far either is trusted at high evidence, so their disagreement is split. kGreenLeanSe: the intent residual's
// target leans toward magenta by this many of its standard errors (a slightly magenta frame is preferred to a
// slightly green one).
constexpr double kHeadSystematicSd = 1.0;
constexpr double kGreenLeanSe = 1.0;

// The intent field the joint fit's fifth residual reads, over the fit's cells: the search's mu and sigma (3 per cell,
// Lab) and each cell's 4 x 4 block of the network's grid (bx, by; nbx blocks a row), for the residual's standard error.
struct JointFitField {
  const float* mu = nullptr;
  const float* sigma = nullptr;
  std::vector<int> bx, by;
  int nbx = 1;
};

// The joint fit's Levenberg-Marquardt loop on a residual function of (p, e): forward differences, box bounds by
// projection, the shape (curve_shape_ok at each candidate's anchor) by cutting a step back, E solved at each point.
using JointResid = std::function<void(const float* p, float e, std::vector<double>& r)>;
struct JointLM {
  JointCurveModel* model;
  float g0, b0, e0, target_L, e_lo, e_hi;
  const int* table_n;
  CurveFitOptions o;
  float lo[4] = {}, hi[4] = {};
  int iters = 0, evaluations = 0, renders = 0;
  bool settled_all = true;

  InputCurves shape_of(const float* p) const {
    InputCurves c;
    c.hold = o.hold_slope;
    c.a[0] = p[2];
    c.a[1] = p[3];
    return c;
  }
  // g in [g0 * 2^-s, g0 * 2^s] and b alike (floored at kGainFloor), |a| <= curve_bound.
  void bounds() {
    const float g_lo = g0 * std::exp2(-o.g_bound_stops), g_hi = g0 * std::exp2(o.g_bound_stops);
    const float b_lo = b0 * std::exp2(-o.b_bound_stops), b_hi = b0 * std::exp2(o.b_bound_stops);
    const float l[4] = {std::max(kGainFloor, g_lo), std::max(kGainFloor, b_lo), -o.curve_bound, -o.curve_bound};
    const float h[4] = {g_hi, b_hi, o.curve_bound, o.curve_bound};
    std::copy(l, l + 4, lo);
    std::copy(h, h + 4, hi);
  }
  bool solve_point(const float* p, float e_warm, float& e, bool check_shape) {
    const InputCurves shape = shape_of(p);
    if (o.solve_e) {
      if (!model->solve_e(p[0], p[1], shape, target_L, e_warm, e_lo, e_hi, o.e_tol_L, renders, e)) return false;
    } else {
      e = e0;
    }
    if (!check_shape) return true;
    const InputCurves c = model->anchored(shape, e, p[0], p[1]);
    return curve_shape_ok(c.a[0], c.k[0], c.u_med[0], c.ref[0], table_n[0], o.max_dev_stops, c.u_hold(0)) &&
           curve_shape_ok(c.a[1], c.k[1], c.u_med[1], c.ref[1], table_n[1], o.max_dev_stops, c.u_hold(1));
  }
  bool evaluate(const JointResid& R, const float* p, float e_warm, float& e, std::vector<double>& r, double& cost,
                bool check_shape) {
    ++evaluations;
    if (!solve_point(p, e_warm, e, check_shape)) return false;
    R(p, e, r);
    double s = 0.0;
    for (double v : r) s += v * v;
    cost = s;
    return std::isfinite(cost);
  }
  float cut_back(const float* p, const float* q, float e, float& e_t) {
    float t_ok = 0.0f, t_bad = 1.0f;
    e_t = e;
    for (int it = 0; it < 10; ++it) {
      const float t = 0.5f * (t_ok + t_bad);
      float qt[4];
      for (int j = 0; j < 4; ++j) qt[j] = p[j] + t * (q[j] - p[j]);
      float et = e;
      if (solve_point(qt, e, et, true)) {
        t_ok = t;
        e_t = et;
      } else {
        t_bad = t;
      }
    }
    return t_ok;
  }
  // p in/out (start / result), e in/out. False when the start is infeasible.
  bool run(const JointResid& R, float* p, float& e) {
    std::vector<double> r, rt, rj;
    double cost = 0.0;
    float e_s = e;
    if (!evaluate(R, p, e_s, e, r, cost, true)) return false;
    const int m = 4;
    double lambda = 1e-3;
    bool settled = cost <= 1e-12;
    for (int iter = 0; iter < 25 && !settled; ++iter) {
      const size_t nr = r.size();
      std::vector<double> J(nr * m, 0.0);
      for (int j = 0; j < m; ++j) {
        float q[4];
        std::copy(p, p + 4, q);
        const float h = (q[j] + 1e-3f <= hi[j]) ? 1e-3f : -1e-3f;
        q[j] += h;
        float ej = e;
        double cj = 0.0;
        if (!evaluate(R, q, e, ej, rj, cj, false) || rj.size() != nr) continue;
        for (size_t k = 0; k < nr; ++k) J[k * m + j] = (rj[k] - r[k]) / h;
      }
      double A[4][4] = {}, gvec[4] = {};
      for (size_t k = 0; k < nr; ++k)
        for (int a = 0; a < m; ++a) {
          gvec[a] += J[k * m + a] * r[k];
          for (int bb = a; bb < m; ++bb) A[a][bb] += J[k * m + a] * J[k * m + bb];
        }
      for (int a = 0; a < m; ++a)
        for (int bb = 0; bb < a; ++bb) A[a][bb] = A[bb][a];
      for (int a = 0; a < m; ++a)
        if ((p[a] <= lo[a] + 1e-5f && gvec[a] > 0.0) || (p[a] >= hi[a] - 1e-5f && gvec[a] < 0.0)) {
          for (int bb = 0; bb < m; ++bb) A[a][bb] = A[bb][a] = 0.0;
          A[a][a] = 1.0;
          gvec[a] = 0.0;
        }
      bool accepted = false;
      for (int tries = 0; tries < 10 && !accepted; ++tries) {
        double M[4][5];
        for (int a = 0; a < m; ++a) {
          for (int bb = 0; bb < m; ++bb) M[a][bb] = A[a][bb] + (a == bb ? lambda * std::max(A[a][a], 1e-6) : 0.0);
          M[a][4] = -gvec[a];
        }
        for (int c = 0; c < m; ++c) {
          int piv = c;
          for (int rr = c + 1; rr < m; ++rr)
            if (std::fabs(M[rr][c]) > std::fabs(M[piv][c])) piv = rr;
          for (int cc = 0; cc < 5; ++cc) std::swap(M[c][cc], M[piv][cc]);
          if (std::fabs(M[c][c]) < 1e-300) M[c][c] = 1e-300;
          for (int rr = c + 1; rr < m; ++rr) {
            const double fct = M[rr][c] / M[c][c];
            for (int cc = c; cc < 5; ++cc) M[rr][cc] -= fct * M[c][cc];
          }
        }
        double delta[4];
        for (int c = m - 1; c >= 0; --c) {
          double sum = M[c][4];
          for (int cc = c + 1; cc < m; ++cc) sum -= M[c][cc] * delta[cc];
          delta[c] = sum / M[c][c];
        }
        float step = 0.0f;
        for (int j = 0; j < m; ++j)
          step = std::max(step, std::fabs(std::min(hi[j], std::max(lo[j], p[j] + (float)delta[j])) - p[j]));
        if (step < 1e-5f) {
          settled = true;
          break;
        }
        float q[4], eq = e;
        double cq = 0.0;
        bool feasible = false;
        for (float step_s : {1.0f, 0.5f, 0.25f, 0.1f, 0.05f}) {
          float qt[4];
          for (int j = 0; j < m; ++j) qt[j] = std::min(hi[j], std::max(lo[j], p[j] + step_s * (float)delta[j]));
          float eq_t = e;
          double cq_t = 0.0;
          bool ok = evaluate(R, qt, e, eq_t, rt, cq_t, true);
          if (!ok && step_s == 1.0f) {
            float e_t = e;
            const float t_ok = cut_back(p, qt, e, e_t);
            if (t_ok > 0.0f) {
              for (int j = 0; j < m; ++j) qt[j] = p[j] + t_ok * (qt[j] - p[j]);
              ok = evaluate(R, qt, e_t, eq_t, rt, cq_t, true);
            }
          }
          if (ok && cq_t < cost) {
            std::copy(qt, qt + 4, q);
            eq = eq_t;
            cq = cq_t;
            feasible = true;
            break;
          }
        }
        if (feasible) {
          const double gain = cost - cq;
          std::copy(q, q + 4, p);
          e = eq;
          r.swap(rt);
          cost = cq;
          lambda = std::max(lambda / 3.0, 1e-9);
          accepted = true;
          ++iters;
          if (gain < 1e-6 * std::max(1.0, cost) || cost < 1e-8) settled = true;
        } else {
          lambda *= 4.0;
        }
      }
      if (!accepted && !settled) break;
    }
    settled_all = settled_all && settled;
    return true;
  }
};

// Weighted moments over a cell set: mean L*, a*, b*, the L* variance and the slopes 50 cov / (var + 25).
struct JointMoments {
  double L = 0, a = 0, b = 0, vL = 0, ka = 0, kb = 0;
};
inline JointMoments joint_moments(const std::vector<Lab>& labs, const std::vector<int>& idx,
                                  const std::vector<double>& w) {
  JointMoments m;
  double ws = 0;
  for (size_t k = 0; k < idx.size(); ++k) {
    const Lab& l = labs[idx[k]];
    ws += w[k];
    m.L += w[k] * l.L;
    m.a += w[k] * l.a;
    m.b += w[k] * l.b;
  }
  if (!(ws > 0)) return m;
  m.L /= ws;
  m.a /= ws;
  m.b /= ws;
  double sLa = 0, sLb = 0;
  for (size_t k = 0; k < idx.size(); ++k) {
    const Lab& l = labs[idx[k]];
    const double dL = l.L - m.L, wk = w[k] / ws;
    m.vL += wk * dL * dL;
    sLa += wk * dL * (l.a - m.a);
    sLb += wk * dL * (l.b - m.b);
  }
  m.ka = 50.0 * sLa / (m.vL + 25.0);
  m.kb = 50.0 * sLb / (m.vL + 25.0);
  return m;
}

// The joint fit: g, b, a_G and a_B in one Levenberg-Marquardt solve, E tied to Auto's median L*, on five residuals
// and the curve damping (see the section comment above). ok false: fewer than 8 cells in range, or no feasible start;
// the caller then keeps Auto's colour. The fit runs without the never-greener guard first; when its result is greener
// than Auto's (deep shadows exempt) it runs again from the start with the guard's hinges.
inline JointCurveFit fit_input_curves_joint(const PipelineConfig& cfg, const CurveFrame& f, const JointFitField& fi,
                                            float e0, float g0, float b0, float target_L, const int table_n[2],
                                            const CurveFitOptions& o = CurveFitOptions(), float e_lo = 0.0f,
                                            float e_hi = 0.0f) {
  JointCurveFit out;
  out.ok = false;
  const int n = f.n;
  JointCurveModel model(cfg, f);
  if (!(e_lo > 0.0f)) e_lo = e0 * 0.25f;
  if (!(e_hi > e_lo)) e_hi = std::max(e_lo, e0 * 4.0f);
  JointLM lm{&model, g0, b0, e0, target_L, e_lo, e_hi, table_n, o};
  lm.bounds();
  const double sqrt_lambda = std::sqrt(std::max(0.0, (double)o.joint_lambda));
  auto labs_at = [&](const float* p, float e, std::vector<Lab>& labs) {
    const InputCurves c = model.anchored(lm.shape_of(p), e, p[0], p[1]);
    labs.resize(n);
    for (int i = 0; i < n; ++i) labs[i] = model.cell_lab(i, e, p[0], p[1], c);
  };

  // Auto: E solved, its cells; the cells in range (2 < L* < 92 at Auto) and their p^2, whose sum is the evidence m.
  const float p0[4] = {g0, b0, 0.0f, 0.0f};
  float eA = e0;
  if (!lm.solve_point(p0, e0, eA, true)) {
    out.skipped = "start";
    return out;
  }
  std::vector<Lab> lab0;
  labs_at(p0, eA, lab0);
  std::vector<int> B;
  std::vector<double> p2;
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    const Lab& l = lab0[i];
    if (!std::isfinite(l.L) || !std::isfinite(l.a) || !std::isfinite(l.b) || !(l.L > 2.0f && l.L < 92.0f)) continue;
    B.push_back(i);
    const double p = f.neutral_p ? std::clamp((double)f.neutral_p[i], 0.0, 1.0) : 0.0;
    p2.push_back(p * p);
    m += p * p;
  }
  out.evidence = (float)m;
  if ((int)B.size() < 8) {
    out.skipped = "cells";
    return out;
  }

  JointFitGuard guard;
  auto guard_means = [&](const float* p, float e, double v[2]) {
    const InputCurves c = model.anchored(lm.shape_of(p), e, p[0], p[1]);
    guard.means([&](int i) { return (double)model.cell_lab(i, e, p[0], p[1], c).a; }, v);
  };
  {
    const InputCurves c = model.anchored(InputCurves{}, eA, g0, b0);
    guard.set_cells(n, [&](int i) { return model.cell_lab(i, eA, g0, b0, c); }, f.neutral_p);
    guard_means(p0, eA, guard.ref);
  }
  for (int k = 0; k < 2; ++k) {
    out.guard.counted[k] = guard.counted[k];
    out.guard.auto_a[k] = (float)guard.ref[k];
  }

  // The anchors' weights: p^2, plus kAnchorEps spread evenly so that a frame without evidence still has defined
  // moments (they then carry no weight in the residuals: w = 0).
  constexpr double kAnchorEps = 1e-3;
  const double w = JointCurveModel::shrink_weight(m);
  std::vector<double> om(B.size());
  for (size_t k = 0; k < B.size(); ++k) om[k] = (p2[k] + kAnchorEps / B.size()) / (m + kAnchorEps);
  const JointMoments M0 = joint_moments(lab0, B, om);
  const double slope_share = M0.vL / (M0.vL + (double)JointCurveModel::kSlopeSpanL * JointCurveModel::kSlopeSpanL);
  const double w_slope = 0.5 * std::sqrt(M0.vL / (M0.vL + 100.0));
  // The anchors' mean a*: a noise term over m anchors plus the head's systematic error (kHeadSystematicSd, in units
  // of the noise), sigma = sqrt((h^2 + m) / (m + h)), h = kEvidenceHalfMass.
  constexpr double h = JointCurveModel::kEvidenceHalfMass;
  const double sig_anc = std::sqrt((h * h + m) / (m + h)) * kHeadSystematicSd;
  const double kt_a = (1.0 - w * slope_share) * M0.ka, kt_b = (1.0 - w * slope_share) * M0.kb;

  // The intent field's precision weights (1 / sigma_a^2) over every fit cell, normalised. Without a field (no mu or
  // sigma, or block indices that do not cover the cells) or with no precision at all, the fifth residual is absent
  // and the fit is the four-residual fit.
  bool intent = fi.mu && fi.sigma && fi.nbx > 0 && (int)fi.bx.size() == n && (int)fi.by.size() == n;
  // Each cell's block within the field: 0 <= bx < nbx, and 0 <= by < n (a block row holds at least one cell).
  for (int i = 0; intent && i < n; ++i) intent = fi.bx[i] >= 0 && fi.bx[i] < fi.nbx && fi.by[i] >= 0 && fi.by[i] < n;
  std::vector<double> al(n, 0.0);
  double asum = 0;
  if (intent) {
    for (int i = 0; i < n; ++i) {
      const double s = std::max((double)fi.sigma[3 * i + 1], 1e-3);
      al[i] = 1.0 / (s * s);
      if (!std::isfinite(lab0[i].a)) al[i] = 0.0;
      asum += al[i];
    }
    intent = asum > 0.0;
  }
  double se = 0.0, dL50 = 0.0;
  if (intent) {
    for (double& v : al) v /= asum;
    double La = 0;
    for (int i = 0; i < n; ++i) La += al[i] * lab0[i].L;
    // The standard error of the field's mean a* residual at Auto: block sums over 4 x 4 cell blocks of the residual,
    // linearly detrended in L*.
    std::vector<double> d0(n);
    double dbar = 0;
    for (int i = 0; i < n; ++i) {
      d0[i] = al[i] > 0 ? fi.mu[3 * i + 1] - lab0[i].a : 0.0;
      dbar += al[i] * d0[i];
    }
    double szz = 0, szd = 0;
    for (int i = 0; i < n; ++i) {
      const double z = (lab0[i].L - La) / 50.0;
      szz += al[i] * z * z;
      szd += al[i] * z * (d0[i] - dbar);
    }
    const double kd = szz > 0 ? szd / szz : 0.0;
    std::vector<double> bs((size_t)fi.nbx * (1 + *std::max_element(fi.by.begin(), fi.by.end())), 0.0);
    for (int i = 0; i < n; ++i) {
      const double dt = (d0[i] - dbar) - kd * (lab0[i].L - La) / 50.0;
      bs[(size_t)fi.by[i] * fi.nbx + fi.bx[i]] += al[i] * dt;
    }
    double se2 = 0;
    for (double v : bs) se2 += v * v;
    se = std::sqrt(se2);
    // The lever arm: the field's weighted mean L* sits away from the anchors', so a slope change moves the field's
    // mean a*; (k_a - kt_a) dL50 removes that share, dL50 = (L_field - L_anchors) / 50 x (vL + 25) / max(vL, 25)
    // (k_a is the slope damped by vL / (vL + 25); below vL = 25, anchors on one object of nearly one lightness, the
    // undamping stops at 2 so a slope measured over a narrow span is not amplified).
    dL50 = (La - M0.L) / 50.0 * (M0.vL + 25.0) / std::max(M0.vL, 25.0);
  }
  constexpr double c_res = 1.0 / 1.5;   // the factor 1/1.5 of r0, r1 and r4 (the joint fit's notes above)
  const double r4_scale = c_res / std::sqrt(se * se + kHeadSystematicSd * kHeadSystematicSd);
  bool guard_path = false;
  std::vector<Lab> labs;
  auto r4_of = [&](const std::vector<Lab>& L, const JointMoments& Mp) {
    if (!intent) return 0.0;
    // A cell the candidate renders non-finite is skipped without renormalising the weights (the sum shrinks a little).
    double s = 0;
    for (int i = 0; i < n; ++i)
      if (al[i] > 0 && std::isfinite(L[i].a)) s += al[i] * (L[i].a - fi.mu[3 * i + 1]);
    return r4_scale * (s - (Mp.ka - kt_a) * dL50 - kGreenLeanSe * se);
  };
  const JointResid R = [&](const float* q, float eq, std::vector<double>& r) {
    labs_at(q, eq, labs);
    const JointMoments Mp = joint_moments(labs, B, om);
    r.assign(9, 0.0);
    r[0] = c_res / sig_anc * (Mp.a - (1.0 - w) * M0.a);
    r[1] = c_res * (Mp.b - (1.0 - w) * M0.b);
    r[2] = w_slope * (Mp.ka - kt_a);
    r[3] = w_slope * (Mp.kb - kt_b);
    r[4] = r4_of(labs, Mp);
    if (guard_path) {
      double v[2];
      guard.means([&](int i) { return (double)labs[i].a; }, v);
      r[5] = guard.hinge(0, v);
      r[6] = guard.hinge(1, v);
    }
    r[7] = sqrt_lambda * q[2];
    r[8] = sqrt_lambda * q[3];
  };
  auto loss_at = [&](const float* q, float eq) {
    std::vector<double> r;
    R(q, eq, r);
    double s = 0.0;
    for (int k = 0; k < 5; ++k) s += r[k] * r[k];
    return s;
  };
  out.shrink = (float)w;
  out.slope_share = (float)slope_share;
  out.span_L = (float)std::sqrt(M0.vL);
  out.anchor_sd = (float)sig_anc;
  out.intent_se = (float)se;
  out.intent_auto = intent ? (float)(r4_of(lab0, M0) / r4_scale) : 0.0f;
  out.loss_before = loss_at(p0, eA);

  float p[4];
  float e = eA;
  std::copy(p0, p0 + 4, p);
  lm.run(R, p, e);  // cannot fail here: its start, Auto's point, was solved above
  {
    double v[2];
    guard_means(p, e, v);
    out.guard.free_fit_a[0] = (float)v[0];
    out.guard.free_fit_a[1] = (float)v[1];
    if (guard.binds(0, v) || guard.binds(1, v)) {
      guard_path = true;
      std::copy(p0, p0 + 4, p);
      e = eA;
      lm.run(R, p, e);
      out.guard.refit = true;
    }
  }
  {
    double v[2];
    guard_means(p, e, v);
    for (int k = 0; k < 2; ++k) {
      out.guard.fit_a[k] = (float)v[k];
      if (guard.binds(k, v)) out.guard.binding = true;
    }
  }
  guard_path = false;
  labs_at(p, e, labs);
  out.intent_after = intent ? (float)(r4_of(labs, joint_moments(labs, B, om)) / r4_scale) : 0.0f;
  out.loss_after = loss_at(p, e);
  out.ok = true;
  out.feasible = true;
  out.converged = lm.settled_all;
  out.iterations = lm.iters;
  out.evaluations = lm.evaluations;
  out.renders = lm.renders;
  out.g = p[0];
  out.b = p[1];
  out.e = e;
  out.curves = model.anchored(lm.shape_of(p), e, p[0], p[1]);
  return out;
}

// The grid's median L* through cfg at (e, g, b) (nearest rank over the cells).
inline float grid_median_L(const PipelineConfig& cfg, const float* merged, float e, float g, float b,
                           const uint16_t* grid, int n) {
  if (n <= 0) return 0.0f;
  float m[9];
  candidate_matrix(merged, e, g, b, m);
  std::vector<float> L(n);
  for (int i = 0; i < n; ++i) L[i] = linear_srgb_to_lab(to_display(cfg, convert_pixel_linear(cfg, m, grid + 3 * i))).L;
  std::nth_element(L.begin(), L.begin() + n / 2, L.end());
  return L[n / 2];
}

// E re-metered so the grid's median L* through the combined profile at the fitted g, b equals target_L (Auto's): a
// bisection in log E (a larger E renders darker), the curves (shape's a and k) anchored at each E it tries, so the
// profile it measures is the one the export renders at that E. cfg carries the bracket's own profile. The bisection
// runs over [e_centre / 2, 2 e_centre] within [e_lo, e_hi] when the target lies there (both ends tested first), else
// over [e_lo, e_hi]; where the target lies outside those too, *clipped is set and E is the bound it reached. It is used
// instead of a fixed point on E, g and b after the fit, which would follow its own darker render.
inline float solve_e_for_median_L(PipelineConfig cfg, const CurveFrame& f, const InputCurves& shape, float e_centre,
                                  float e_lo, float e_hi, float g, float b, float target_L, bool* clipped = nullptr,
                                  int iters = 40) {
  const ProfileData base = cfg.profile;
  CombinedProfile comb;
  auto median_at = [&](float e) {
    comb.build(base, anchor_curves(shape, f, e, g, b));
    cfg.profile = comb.data;
    return grid_median_L(cfg, f.merged, e, g, b, f.grid, f.n);
  };
  if (clipped) *clipped = false;
  float lo = std::max(e_lo, e_centre * 0.5f), hi = std::min(e_hi, e_centre * 2.0f);
  const bool enclosed = lo < hi && median_at(lo) >= target_L && median_at(hi) <= target_L;
  if (!enclosed) {
    lo = e_lo;
    hi = e_hi;
    const float L_lo = median_at(lo), L_hi = median_at(hi);
    const bool dark_at_lo = L_lo + 0.05f < target_L;  // darker than the target even at the lowest E
                                                      // (0.05 L* LUT tolerance)
    if (dark_at_lo || L_hi - 0.05f > target_L) {      // or lighter even at the highest
      if (clipped) *clipped = true;
      return L_lo < target_L ? lo : hi;
    }
    if (L_lo <= target_L) return lo;
    if (L_hi >= target_L) return hi;
  }
  for (int i = 0; i < iters; ++i) {
    const float mid = std::sqrt(lo * hi);
    if (median_at(mid) < target_L) hi = mid;
    else lo = mid;
  }
  return std::sqrt(lo * hi);
}

}  // namespace negicc

// ====================================================================================================================
// Tone stage: the tone stage's curve, plan and clip check
// ====================================================================================================================

// The tone stage's curve, plan and clip check, shared by neg_process and negicc-station (part of libnegicc_dinov3.a;
// no LibRaw or option dependencies).

// ---- The tone stage (tone_stage.cc) ----
// The tone stage, on white-light and TriLED bundles alike: one monotone lightness curve on CIE L* (D65), set per frame
// from the tone heads
// (ToneHead, negicc.h), applied in the conversion kernel to each pixel's display linear sRGB value
// (apply_tone in the colour pipeline part above): its luminance Y goes to Y' = Y(f(L*(Y))) on the white point's
// neutral, and its colour difference from that neutral is scaled by f'(Y) / f'(Y') (Lab's f), which holds a* and b* to
// first order. Colour is not re-balanced, and no pixel is converted to Lab.
//
//   f(L) = black + (white - black) * h(g(L / 100)),  g(x) = x^gamma,
//   h(x) = x^contrast / (x^contrast + (1 - x)^contrast)
//
// The four curve numbers are solved so the render's own descriptors move toward the heads' predictions: since f is
// monotone, the toned render's percentiles are f of the render's, so L2' = f(L2), L50' = f(L50), L98' = f(L98) and
// (L90 - L10)' = f(L90) - f(L10). Each target is the render's value moved a share of the way to the prediction:
// all the way for the shadow level L2 (kToneShadowMove), half way for the highlight level L98 and the tonal range
// L90 - L10 (kToneHighlightMove, kToneRangeMove), and not at all for the mid-tone L50, which the curve holds. A weak
// prior keeps the curve near the identity where the targets leave it free. The descriptors are measured on the
// network-size preview of the final render, as the heads' training targets were on lab scans.

// The shares of the way from the render's descriptor to the heads' prediction the curve moves.
constexpr double kToneShadowMove = 1.0;     // L2
constexpr double kToneHighlightMove = 0.5;  // L98
constexpr double kToneRangeMove = 0.5;      // L90 - L10 (the mid-tone L50 is held where it is)
// The curve's bounds (black, white, gamma, contrast) and the identity it starts from.
constexpr double kToneLo[4] = {-15.0, 80.0, 0.7, 0.7};
constexpr double kToneHi[4] = {15.0, 110.0, 1.45, 1.5};
// The prior's weight, and its scales: 5 L* for the black and white points, 0.1 in log for gamma and contrast.
constexpr double kTonePrior = 0.02;
// Above the render's L98 no target says where the curve goes, and quite different white points can fit the four targets
// about equally, so a small change of the render could switch between them. This weight holds the curve's rise above
// L98 to the input's, f(100) = f(L98) + (100 - L98) (slope 1 on average there), but never above white, f(100) <= 100,
// so a lifted L98 does not push the top past white into clipping. That makes the solution unique, and the solve starts
// from several white points and keeps the lowest cost, so it does not depend on which basin a single start falls into.
constexpr double kToneTopSlope = 0.07;
constexpr double kToneWhiteStarts[3] = {100.0, 110.0, 85.0};
// Clipping at either end: a curve is kept when it puts no more than kToneClipAllowance more of the frame's inner crop
// (kToneClipCrop of each side) at white (a channel at the 16-bit maximum) than the conversion without it, and no more
// than kToneClipAllowance more at black (a channel at 0), each end against its own share before the stage. The moves
// are scaled down to the first of kToneClipShares at which both ends hold: the highlight and range moves always, the
// shadow move only when the black end fails with the full shadow move (a lift of the shadows is never scaled away).
// When no share holds, ToneClipFallback says which end gives way. The shares are measured on a sample of the image
// the export converts, through the kernel: the 2 x 2 pixels at the origin of every kToneClipSampleStep x
// kToneClipSampleStep block, so the sample covers each phase of a 2 x 2 colour mosaic the image was demosaiced from.
constexpr double kToneClipAllowance = 0.005;
constexpr double kToneClipShares[5] = {1.0, 0.75, 0.5, 0.25, 0.0};
// The curve kept when no share holds both ends (--tone_stage auto, highlights_first, least_total). Each share's
// candidate is the curve the check tried last at it: the whole shadow move, or the scaled one where the black end
// failed with it.
enum ToneClipFallback {
    // Shadows first (the default): the last share's candidate, no highlight or range move, the shadow lift kept where
    // black holds.
    kToneFallbackShadowsFirst,
    // Highlights first: the largest share whose white end holds; else, at the last share, the shadow move scaled
    // through kToneClipShares until the white end holds; else the candidate tried that clips least at white (the first
    // tried on a tie).
    kToneFallbackHighlightsFirst,
    // Least total: the candidate whose shares at white and at black together are least, of the shares' candidates and
    // the last share with the shadow move scaled through kToneClipShares (every candidate the other two rules try); on
    // a tie the larger share, then the larger shadow move.
    kToneFallbackLeastTotal,
};
constexpr int kToneClipSampleStep = 8;
// The inner crop the clip check reads: the search's (ParamSearchOptions::crop_ratio), the area the plan's preview
// shows.
constexpr float kToneClipCrop = 0.90f;

struct ToneCurve {
    double black = 0.0, white = 100.0, gamma = 1.0, contrast = 1.0;
};

// The render's descriptors the curve is solved from: pixel L* percentiles.
struct ToneDescriptors {
    double L2 = 0, L10 = 0, L25 = 0, L50 = 0, L75 = 0, L90 = 0, L98 = 0;
    double range() const { return L90 - L10; }
};

// The targets the curve is solved for (the render's descriptors moved toward the predictions).
struct ToneTargets {
    double L2 = 0, L50 = 0, L98 = 0, range = 0;
};

// The share of the frame's inner crop (kToneClipCrop) at white (a channel at the 16-bit maximum) and at black (a
// channel at 0): before the stage on the clip check's sample of the conversion without the curve, after it on every
// pixel of the export.
struct ToneClip {
    double white_before = 0, white_after = 0, black_before = 0, black_after = 0;
};

// The descriptors of an 8-bit RGB image (n pixels, 3 bytes each); percentiles interpolate linearly between ranks.
ToneDescriptors tone_descriptors(const uint8_t* rgb, size_t n);

// The plan at a highlight share: the targets and the curve solved for them.
ToneCurve tone_plan(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                    double highlight_share, ToneTargets* targets);
// The same with the shadow move scaled by its own share (tone_plan's moves it all the way).
ToneCurve tone_plan_shares(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                           double highlight_share, double shadow_share, ToneTargets* targets);

// The kernel's table of curve q (ToneTable in the colour pipeline part above: kToneTableSize + 1 entries on [0, 1]).
constexpr int kToneTableSize = 8192;
std::vector<float> tone_table(const ToneCurve& q);
// A curve the tone stage can produce: its four numbers finite and within kToneLo..kToneHi (every curve it applies is
// solved there). A recorded curve outside them (typed by hand, or edited) is refused rather than rendered: a gamma or
// contrast at or below 0, or white below black, inverts the table, and a contrast far past 1.5 makes it NaN.
bool tone_curve_ok(const ToneCurve& q);

// The white end's rule alone, for a caller that measures white clipping only: which of kToneClipShares to keep, as an
// index: 0 (the designed moves) when the designed curve clips no more than kToneClipAllowance beyond white_before (the
// conversion's own white clipping); else the largest share that does not; else the share that clips least (the
// larger on a tie). tone_plan_for_image checks both ends. white_after_of(k) gives the white clipping with the
// curve planned at kToneClipShares[k]; it is called for k = 0 first and for no share after the one kept.
int tone_pick_clip_share(double white_before, const std::function<double(int)>& white_after_of);

// The clip check's sample of the unconverted image img (the 2 x 2 pixels at the origin of every kToneClipSampleStep
// block that lie inside the inner crop, kToneClipCrop), converted through pc and matrix up to the tone curve: each
// pixel's display value (convert_pixel_display in the colour pipeline part above). It is converted once per conversion,
// and every curve the plan tries is applied to it (tone_clip_shares).
std::vector<negicc::Rgb> tone_clip_sample(const negicc::PipelineConfig& pc, const float* matrix, const uint16_t* img,
                                          int w, int h, int stride);
// The shares of the sample that the rest of the conversion through pc (the tone curve on an sRGB export, then the
// encoding) puts at white (a channel at the 16-bit maximum) and at black (a channel at 0): with pc as it is (q null),
// or with curve q's table. Either output may be null.
void tone_clip_shares(const negicc::PipelineConfig& pc, const std::vector<negicc::Rgb>& sample, const ToneCurve* q,
                      double* white, double* black);

// Whether a plan's kept shares move nothing: the highlight and range moves and the shadow move all at share 0. The
// targets are then the render's own descriptors, and the stage applies no curve: the conversion is the one without
// the stage, exactly (the solver's curve for those targets is within about 1e-4 of the identity, not equal to it, and
// a table, even of the identity, goes through Lab f and its inverse in float).
inline bool tone_moves_nothing(double share, double shadow_share) { return share == 0.0 && shadow_share == 0.0; }

// The plan's curve for the image the export converts (its clip check's sample, tone_clip_sample, through pc): the
// designed moves (designed, solved for designed_targets by tone_plan at share 1) when both ends hold against the
// conversion's own shares (white_before, black_before: tone_clip_shares without a curve), else the moves scaled as
// kToneClipAllowance says, and fallback when no share holds both ends. share gets the share of the highlight and
// range moves kept, shadow_share (when given) that of the shadow move, fell_back (when given) whether no share
// held, so that fallback chose the curve, and white_after and black_after (when given) the sample's shares at white
// and at black with the curve kept, as the check counted them.
ToneCurve tone_plan_for_image(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                              const negicc::PipelineConfig& pc, const std::vector<negicc::Rgb>& sample,
                              double white_before, double black_before, const ToneCurve& designed,
                              const ToneTargets& designed_targets, ToneTargets* targets, double* share,
                              double* shadow_share = nullptr, ToneClipFallback fallback = kToneFallbackShadowsFirst,
                              bool* fell_back = nullptr, double* white_after = nullptr, double* black_after = nullptr);

// The tone stage's plan as the conversion solves it after the curve fit (from the fit's ToneStagePlan: the heads'
// predictions and the final render's preview): the preview's descriptors, the targets and the curve between them.
struct TonePlanCurve {
  ToneDescriptors render;         // the final render's preview before the stage
  ToneTargets targets;            // the descriptors the curve aims at
  ToneCurve curve;                // the curve
};

// ====================================================================================================================
// Film base detector: the film base read from the scan itself
// ====================================================================================================================

// The film base of a scan read from the scan itself, for Auto when no film base is given. neg_process and
// negicc-station both call detect_film_base, so the two pick the same base.
//
//  1. rebate: the frame detector (run in the frame detector part above on the RAW's mosaic) finds a flat patch of
//     unexposed film next to the frame (an inter-frame gap or the rebate) and validates it (the spread of its green
//     density). The base is the median per channel of the half-size decode over the inner part of that patch. The base
//     is the thinnest film on the scan, so a patch the scan's thinnest film (2.) outshines by more than kRebateMargin
//     stops in a channel is picture, not base, and is not taken. Nor is a patch the sensor clipped (more than
//     kBaseClipFraction of its pixels at kSensorClip of the white level in a channel): clipped film reads the clip,
//     not the base.
//  2. statistics: else the thinnest film on the scan: the 99.5th percentile per channel of the film pixels of the
//     half-size decode reduced to kBaseWorkSize pixels, with bare light and black holder masked out. In sensor units
//     the same percentile read before the flat field at kSensorClip of the white level in a channel means the film's
//     thinnest part clipped: the base read from it is the clip (thinnest_clipped). On a decode in
//     sensor units (TriLED) bare light is what the sensor clipped in all three channels: light is brighter than the
//     base in every channel, while film near the clip clips first in its brightest channel, and the base is nearly
//     neutral in the camera's channels, so colour does not tell thin film from light. On a white-light decode
//     (Rec.2020, white-balanced) bare light is a part clipped in a channel (its maxima at a plateau) whose median blue
//     is above the film's 99.5th percentile of blue, with the bright unclipped light connected to it, or a neutral
//     part bluer than the orange film: film transmits no more blue than its base, so a flat stretch of film that
//     forms a plateau is not taken for light.
//
// The white-light arithmetic follows negicc-station's auto_crop.py (downsample, classify) and its film base reading,
// as numpy 2 and OpenCV compute them, float32 where they are float32.
// The base is in the conversion's film base space: the decode through the crosstalk matrix (none for a white-light
// decode, already linear Rec.2020). The scan is its own base capture (the same exposure).

namespace negicc {

constexpr double kBasePatchInset = 0.15;  // fraction of the patch's box dropped on each side (its edges blend)
constexpr int kBaseMinSamples = 32;       // pixels the inner box must hold
constexpr float kSensorClip = 0.98f;      // a sample at or above this fraction of the white level (before the flat
                                          // field, but after it on the LibRaw path with --flat) is clipped
constexpr double kBaseClipFraction = 0.001;  // a patch, tile or region with more clipped pixels is clipped
constexpr double kRebateMargin = 0.5;     // stops: the thinnest film may exceed a base patch by at most this
constexpr double kBasePercentile = 99.5;  // per-channel percentile of film pixels taken as the film base (the
                                          // statistics' step; also the print's floor on a base that is not a capture,
                                          // the rebate included: print_floor_factor)
constexpr int kBaseWorkSize = 512;        // long side (px) of the image the thinnest film is read on
constexpr float kBaseMinFrameConf = 0.5f; // the detection's confidence below which its patch is not used (the frame
                                          // detection neg_process falls back from at the same level)

constexpr const char* kFilmBaseSourceRebate = "rebate";
constexpr const char* kFilmBaseSourceStatistics = "statistics";

struct FilmBaseDetection {
  bool found = false;
  std::string source;                 // kFilmBaseSourceRebate or kFilmBaseSourceStatistics when found
  bool is_rebate() const { return found && source == kFilmBaseSourceRebate; }
  double rgb[3] = {0.0, 0.0, 0.0};    // the film base: decode_rgb through the crosstalk matrix
  double decode_rgb[3] = {0.0, 0.0, 0.0};  // in the half-size decode's units
  float thinnest[3] = {0.0f, 0.0f, 0.0f};  // the thinnest film (2.), in the decode's units
  bool rebate_clipped = false;        // the frame detector's patch was refused because the sensor clipped it
  bool thinnest_clipped = false;      // sensor units: the thinnest film (2.), read before the flat field, is at the
                                      // sensor's clip in a channel, so a base read from it is the clip
  bool has_box = false;               // source rebate: the sampled patch in the half-size decode (x, y, w, h)
  int box[4] = {0, 0, 0, 0};
  // The frame detector's record (when it ran): its reason, its patch's green density spread, its frame confidence,
  // whether the bare light is clipped, and how far the thinnest film outshines the patch (stops, max over channels).
  bool has_detector = false;
  std::string reason;
  float std_d = 0.0f;
  float frame_conf = 0.0f;
  bool light_clipped = false;
  bool has_outshone = false;
  double thinnest_over_patch_stops = 0.0;
};

// The thinnest film on the half-size decode |half| of a TriLED scan or capture (step 2. above), in its units. |white|
// is the clipping level of its samples before the flat field (black subtracted, in their units). |at_clip| (when
// given): whether that film, read on the same pixels before the flat field, is at the sensor's clip in a channel
// (FilmBaseDetection::thinnest_clipped). 0, 0, 0 (and not at the clip) when half is not valid().
void thinnest_film_base(const RgbView& half, float white, float out[3], bool* at_clip = nullptr);

// The film base of the scan whose half-size decode (flat field applied) is |half|. |det| is the frame detector's
// result on the RAW's mosaic of sensor_w x sensor_h photosites (the CFA path's, or the white-light decode's run before
// it decodes, or a linear DNG's; null when there is none: an RGB file, three captures); its patch is used only when the
// detection succeeded with confidence at least kBaseMinFrameConf. |crosstalk| the 3x3 row-major crosstalk matrix (null
// for none: a white-light decode). |white| is the clipping level of half's samples before its flat field (black
// subtracted, in their units; unused on a white-light decode). False (out.found false) when neither step gives a
// positive, finite base; out still carries the detector's record. False with nothing read when half is not valid().
bool detect_film_base(const RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h,
                      const double* crosstalk, float white, FilmBaseDetection& out);

}  // namespace negicc

// ====================================================================================================================
// Embedded data: the data files built into neg_process
// ====================================================================================================================

// Data files built into neg_process (EMBED_FILES in the Makefile): the profile bundles under profiles/<Key>/
// (bundle.json and its ICC profiles), the print profiles under 3rd_party/spektrafilm/profiles/ (print_emulation.cc),
// the DINOv3 intent adapters under models/dinov3-small_<stock>/ and the DINOv3 backbone when it was exported before the
// build (models/dinov3-small/backbone.onnx, and backbone.engine in USE_CUDA=1 builds), the neutral-patch head of the
// white-light curve fit (models/neutral_head/neutral_head.bin) and the tone heads of the tone stage
// (models/tone_head/tone_head.bin). Names are the files' paths in the repository, e.g.
// "profiles/Portra400-A7R4-TriLED/bundle.json" or "models/dinov3-small_portra400/adapter.onnx".
struct EmbeddedFile {
  const char* name;
  const uint8_t* data;  // followed by a NUL byte
  size_t size;
};

// All embedded files, in build order.
std::vector<EmbeddedFile> list_embedded_files();

// The embedded file with this exact name, or null.
const EmbeddedFile* find_embedded_file(const std::string& name);

// The embedded profile bundle with this key ("Portra400-A7R4-TriLED", ignoring case), or null. bundle_dir
// receives its directory ("profiles/Portra400-A7R4-TriLED"), which holds its ICC profiles.
const EmbeddedFile* find_embedded_bundle(const std::string& key, std::string& bundle_dir);

// The key of the embedded bundle of a film stock ("portra400") and scanning light ("tri" for the narrowband RGB
// LED, "d65" for white light): the one named "<Stock>-<camera>-<Light>", e.g. "Portra400-A7R4-TriLED". Empty, with
// the reason in error, when the light is unknown or no single bundle matches.
std::string embedded_bundle_key(const std::string& film_stock, const std::string& light_source, std::string& error);

// Keys of the embedded profile bundles.
std::vector<std::string> list_embedded_bundles();

// The embedded intent adapter for a profile bundle, or null. Resolution follows the station's
// resolve_model_for_profile: models/dinov3-small_<BundleKey>, then models/dinov3-small_<bundlekey>, then
// models/dinov3-small_<stock>, then models/dinov3-small_portra400. adapter_dir receives the directory used.
// A path (an ICC profile inside a bundle's directory, e.g. profiles/Ektar100-A7R4-D65/Target_4.icc) is tried by its
// file name and then by each directory above it, and each name also by the embedded adapters' stock names it starts
// with once lowercased with everything but letters and digits dropped, before the portra400 fallback. fell_back
// (when given) is set when only that fallback matched.
const EmbeddedFile* find_embedded_adapter(const std::string& profile_name, std::string& adapter_dir,
                                          bool* fell_back = nullptr);

// The DINOv3 backbone built in: its TensorRT engine (backbone.engine) when trt_engine, else its ONNX model
// (backbone.onnx). Null when the build has none.
const EmbeddedFile* find_embedded_backbone(bool trt_engine);

struct ProfileBundle;  // the profile bundle part of negicc.h
// Loads the built-in profile bundle with this key (embedded_bundle_key). False when none has that key.
bool load_embedded_bundle(const std::string& key, ProfileBundle& bundle);

// ====================================================================================================================
// Messages: neg_process's progress, warnings and errors
// ====================================================================================================================

// neg_process's messages: progress on stdout, warnings ("Warning: ...") and errors ("ERROR! ...") on stderr. The
// format strings are printf's and end their own lines.

__attribute__((format(printf, 1, 2))) inline void log_info(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stdout, fmt, ap);
  va_end(ap);
}

__attribute__((format(printf, 1, 2))) inline void log_warn(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("Warning: ", stderr);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

__attribute__((format(printf, 1, 2))) inline void log_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("ERROR! ", stderr);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

#endif  // NEGICC_CORE_H
