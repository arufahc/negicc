// Copyright 2024 Alpha Lam <arufa.hc@gmail.com>
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

#include "np_log.h"
#include "np_pipeline.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfa_pipeline.h"
#include "color_pipeline.h"
#include "embedded_data.h"
#include "nlohmann/json.hpp"
#include "np_conversion.h"
#include "np_geometry.h"
#include "np_input.h"
#include "np_output.h"
#include "np_profile.h"

// ---- Stage 5, the white-light tone stage (np_pipeline.h) ----
//
// One monotone L* curve on the finished 16-bit sRGB image, holding the mid-tones and the mean chroma. It takes the
// tone heads' predictions of the shadow level, highlight level and tonal range and the preview they were made on; the
// conversion solves the curve after the fit (tone_plan), scales the highlight and range moves down where the curve
// would clip the final image (tone_plan_for_image) and applies it (apply_tone_stage).
//
// sRGB (D65) <-> CIELAB in double precision, with the white point and matrices of the stage's design.
static inline double tone_srgb_decode(double c) {
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
static inline double tone_srgb_encode(double c) {
    c = c < 0 ? 0 : (c > 1 ? 1 : c);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1 / 2.4) - 0.055;
}
static inline double tone_lab_f(double t) {
    return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0;
}
static inline double tone_lab_finv(double t) {
    return t * t * t > 216.0 / 24389.0 ? t * t * t : (116.0 * t - 16.0) / (24389.0 / 27.0);
}
constexpr double kToneXn = 0.950456, kToneZn = 1.088754;
// Linear sRGB (D65) <-> XYZ, row-major; ToneStageFast uses the same numbers in single precision.
constexpr double kToneRgbToXyz[9] = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160,
                                     0.072169, 0.019334, 0.119193, 0.950227};
constexpr double kToneXyzToRgb[9] = {3.240479, -1.537150, -0.498535, -0.969256, 1.875992,
                                     0.041556, 0.055648,  -0.204043, 1.057311};
// The bisection steps of the in-gamut chroma scale (tone_lab_to_rgb_in_gamut): 2^-14 of the chroma.
constexpr int kToneGamutSteps = 14;

// Linear r, g, b -> L*, a*, b*.
static inline void tone_linear_to_lab(double r, double g, double b, double lab[3]) {
    const double* m = kToneRgbToXyz;
    const double X = m[0] * r + m[1] * g + m[2] * b;
    const double Y = m[3] * r + m[4] * g + m[5] * b;
    const double Z = m[6] * r + m[7] * g + m[8] * b;
    const double fx = tone_lab_f(X / kToneXn), fy = tone_lab_f(Y), fz = tone_lab_f(Z / kToneZn);
    lab[0] = 116.0 * fy - 16.0;
    lab[1] = 500.0 * (fx - fy);
    lab[2] = 200.0 * (fy - fz);
}

// sRGB-encoded r, g, b in 0..1 -> L*, a*, b*.
static inline void tone_rgb_to_lab(double r, double g, double b, double lab[3]) {
    tone_linear_to_lab(tone_srgb_decode(r), tone_srgb_decode(g), tone_srgb_decode(b), lab);
}

// L*, a*, b* -> linear r, g, b (unclipped).
static inline void tone_lab_to_linear_rgb(const double lab[3], double lin[3]) {
    const double fy = (lab[0] + 16.0) / 116.0, fx = fy + lab[1] / 500.0, fz = fy - lab[2] / 200.0;
    const double X = tone_lab_finv(fx) * kToneXn, Y = tone_lab_finv(fy), Z = tone_lab_finv(fz) * kToneZn;
    const double* m = kToneXyzToRgb;
    lin[0] = m[0] * X + m[1] * Y + m[2] * Z;
    lin[1] = m[3] * X + m[4] * Y + m[5] * Z;
    lin[2] = m[6] * X + m[7] * Y + m[8] * Z;
}

// L*, a*, b* -> r, g, b in 0..1 (clipped).
static inline void tone_lab_to_rgb(const double lab[3], double rgb[3]) {
    double lin[3];
    tone_lab_to_linear_rgb(lab, lin);
    for (int c = 0; c < 3; ++c) rgb[c] = tone_srgb_encode(lin[c]);
}

// L*, a*, b* -> r, g, b in 0..1, the chroma scaled toward neutral at the same L* (hue kept) as far as needed to keep
// every channel at or below the maximum: a colour the curve lifts past the sRGB gamut keeps its tone and hue instead
// of having its brightest channels clipped. Below the maximum it is tone_lab_to_rgb.
static inline void tone_lab_to_rgb_in_gamut(const double lab[3], double rgb[3]) {
    auto linear_max = [](const double l[3]) {
        double lin[3];
        tone_lab_to_linear_rgb(l, lin);
        return std::max({lin[0], lin[1], lin[2]});
    };
    if (linear_max(lab) <= 1.0) {
        tone_lab_to_rgb(lab, rgb);
        return;
    }
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < kToneGamutSteps; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double l[3] = {lab[0], lab[1] * mid, lab[2] * mid};
        if (linear_max(l) <= 1.0) lo = mid;
        else hi = mid;
    }
    const double l[3] = {lab[0], lab[1] * lo, lab[2] * lo};
    tone_lab_to_rgb(l, rgb);
}

// f(L) without the final clamp to 0..100.
static inline double tone_curve_L(double L, const ToneCurve& q) {
    double x = L / 100.0;
    x = x < 1e-5 ? 1e-5 : (x > 1 - 1e-5 ? 1 - 1e-5 : x);
    x = std::pow(x, q.gamma);
    const double xc = std::pow(x, q.contrast), yc = std::pow(1 - x, q.contrast);
    return q.black + (q.white - q.black) * xc / (xc + yc);
}

// One pixel's Lab through the curve and the chroma factor.
static inline void tone_lab_apply(double lab[3], const ToneCurve& q) {
    double L2 = tone_curve_L(lab[0], q);
    lab[0] = L2 < 0 ? 0 : (L2 > 100 ? 100 : L2);
    const double C = std::hypot(lab[1], lab[2]);
    double t = (C - 3.0) / 5.0;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    const double k = 1.0 + (q.chroma - 1.0) * t * t * (3 - 2 * t);
    lab[1] *= k;
    lab[2] *= k;
}

ToneDescriptors tone_descriptors(const uint8_t* rgb, size_t n) {
    ToneDescriptors d;
    if (n == 0) return d;
    std::vector<double> L(n);
    double c_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double lab[3];
        tone_rgb_to_lab(rgb[3 * i] / 255.0, rgb[3 * i + 1] / 255.0, rgb[3 * i + 2] / 255.0, lab);
        L[i] = lab[0];
        c_sum += std::hypot(lab[1], lab[2]);
    }
    std::sort(L.begin(), L.end());
    auto pct = [&](double p) {
        const double r = p / 100.0 * (double)(n - 1);
        const size_t i = (size_t)r;
        return i + 1 < n ? L[i] + (r - (double)i) * (L[i + 1] - L[i]) : L[n - 1];
    };
    d.L2 = pct(2);
    d.L10 = pct(10);
    d.L25 = pct(25);
    d.L50 = pct(50);
    d.L75 = pct(75);
    d.L90 = pct(90);
    d.L98 = pct(98);
    d.Cmean = c_sum / (double)n;
    return d;
}

double tone_mean_chroma_after(const uint8_t* rgb, size_t n, const ToneCurve& q) {
    if (n == 0) return 0.0;
    constexpr int kChunks = 64;
    double part[kChunks] = {};
#pragma omp parallel for schedule(static)
    for (int ch = 0; ch < kChunks; ++ch) {
        const size_t i0 = n * ch / kChunks, i1 = n * (ch + 1) / kChunks;
        double c_sum = 0.0;
        for (size_t i = i0; i < i1; ++i) {
            double lab[3], out[3];
            tone_rgb_to_lab(rgb[3 * i] / 255.0, rgb[3 * i + 1] / 255.0, rgb[3 * i + 2] / 255.0, lab);
            tone_lab_apply(lab, q);
            tone_lab_to_rgb(lab, out);
            tone_rgb_to_lab(out[0], out[1], out[2], lab);
            c_sum += std::hypot(lab[1], lab[2]);
        }
        part[ch] = c_sum;
    }
    double c_sum = 0.0;
    for (double v : part) c_sum += v;
    return c_sum / (double)n;
}

// highlight_share scales the highlight and range moves (1: as designed; less where they would clip, see above).
static inline ToneTargets tone_targets(const ToneDescriptors& cur, double pred_L2, double pred_L98, double pred_range,
                                       double highlight_share = 1.0) {
    ToneTargets t;
    t.L2 = cur.L2 + kToneShadowMove * (pred_L2 - cur.L2);
    t.L98 = cur.L98 + highlight_share * kToneHighlightMove * (pred_L98 - cur.L98);
    t.L50 = cur.L50;
    t.range = cur.range() + highlight_share * kToneRangeMove * (pred_range - cur.range());
    return t;
}

// The curve (black, white, gamma, contrast) that brings cur to t: bounded least squares on the four target residuals
// and the identity prior, by Levenberg-Marquardt with the bounds held by projection (a parameter at a bound that the
// gradient pushes outward is held there). Returns the curve with chroma 1.
static inline ToneCurve tone_solve(const ToneDescriptors& cur, const ToneTargets& t) {
    auto curve_of = [](const double* q) {
        ToneCurve c;
        c.black = q[0];
        c.white = q[1];
        c.gamma = q[2];
        c.contrast = q[3];
        return c;
    };
    auto resid = [&](const double* q, double* r) {
        const ToneCurve c = curve_of(q);
        r[0] = tone_curve_L(cur.L2, c) - t.L2;
        r[1] = tone_curve_L(cur.L98, c) - t.L98;
        r[2] = tone_curve_L(cur.L50, c) - t.L50;
        r[3] = tone_curve_L(cur.L90, c) - tone_curve_L(cur.L10, c) - t.range;
        r[4] = kTonePrior * (q[0] / 5.0);
        r[5] = kTonePrior * ((q[1] - 100.0) / 5.0);
        r[6] = kTonePrior * (std::log(q[2]) / 0.1);
        r[7] = kTonePrior * (std::log(q[3]) / 0.1);
        {
            const double f98 = tone_curve_L(cur.L98, c);
            r[8] = kToneTopSlope * (tone_curve_L(100.0, c) - std::min(100.0, f98 + (100.0 - cur.L98)));
        }
    };
    constexpr int kR = 9, kP = 4;
    auto cost_of = [&](const double* q) {
        double r[kR];
        resid(q, r);
        double s = 0.0;
        for (double v : r) s += v * v;
        return s;
    };
    double best_q[kP] = {0.0, 100.0, 1.0, 1.0}, best_cost = INFINITY;
    for (double white0 : kToneWhiteStarts) {
    double q[kP] = {0.0, white0, 1.0, 1.0};
    double cost = cost_of(q), lambda = 1e-3;
    for (int iter = 0; iter < 200; ++iter) {
        double r[kR], J[kR][kP];
        resid(q, r);
        for (int j = 0; j < kP; ++j) {  // central differences, kept inside the bounds
            const double h = 1e-6 * std::max(1.0, std::fabs(q[j]));
            double qp[kP], qm[kP], rp[kR], rm[kR];
            std::copy(q, q + kP, qp);
            std::copy(q, q + kP, qm);
            qp[j] = std::min(kToneHi[j], q[j] + h);
            qm[j] = std::max(kToneLo[j], q[j] - h);
            resid(qp, rp);
            resid(qm, rm);
            for (int k = 0; k < kR; ++k) J[k][j] = (rp[k] - rm[k]) / (qp[j] - qm[j]);
        }
        double A[kP][kP] = {}, g[kP] = {};
        for (int k = 0; k < kR; ++k)
            for (int a = 0; a < kP; ++a) {
                g[a] += J[k][a] * r[k];
                for (int b = 0; b < kP; ++b) A[a][b] += J[k][a] * J[k][b];
            }
        bool held[kP];
        for (int a = 0; a < kP; ++a)
            held[a] = (q[a] <= kToneLo[a] && g[a] > 0.0) || (q[a] >= kToneHi[a] && g[a] < 0.0);
        bool improved = false;
        double step = 0.0;
        for (int tries = 0; tries < 20 && !improved; ++tries) {
            double M[kP][kP + 1];
            for (int a = 0; a < kP; ++a) {
                for (int b = 0; b < kP; ++b)
                    M[a][b] = (held[a] || held[b]) ? (a == b ? 1.0 : 0.0)
                                                   : A[a][b] + (a == b ? lambda * std::max(A[a][a], 1e-12) : 0.0);
                M[a][kP] = held[a] ? 0.0 : -g[a];
            }
            for (int c = 0; c < kP; ++c) {  // Gaussian elimination with partial pivoting
                int piv = c;
                for (int rr = c + 1; rr < kP; ++rr)
                    if (std::fabs(M[rr][c]) > std::fabs(M[piv][c])) piv = rr;
                for (int cc = 0; cc <= kP; ++cc) std::swap(M[c][cc], M[piv][cc]);
                if (std::fabs(M[c][c]) < 1e-300) M[c][c] = 1e-300;
                for (int rr = c + 1; rr < kP; ++rr) {
                    const double f = M[rr][c] / M[c][c];
                    for (int cc = c; cc <= kP; ++cc) M[rr][cc] -= f * M[c][cc];
                }
            }
            double d[kP];
            for (int c = kP - 1; c >= 0; --c) {
                double s = M[c][kP];
                for (int cc = c + 1; cc < kP; ++cc) s -= M[c][cc] * d[cc];
                d[c] = s / M[c][c];
            }
            double qn[kP];
            step = 0.0;
            for (int j = 0; j < kP; ++j) {
                qn[j] = std::clamp(q[j] + d[j], kToneLo[j], kToneHi[j]);
                step = std::max(step, std::fabs(qn[j] - q[j]));
            }
            const double cn = cost_of(qn);
            if (cn < cost) {
                std::copy(qn, qn + kP, q);
                cost = cn;
                lambda = std::max(lambda / 3.0, 1e-12);
                improved = true;
            } else {
                lambda *= 4.0;
            }
            if (step < 1e-13) break;
        }
        if (!improved || step < 1e-12) break;
    }
    if (cost < best_cost) {
        best_cost = cost;
        std::copy(q, q + kP, best_q);
    }
    }
    return curve_of(best_q);
}

ToneCurve tone_plan(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                    const uint8_t* preview, size_t n, double highlight_share, ToneTargets* targets) {
    const ToneTargets t = tone_targets(render, pred_L2, pred_L98, pred_range, highlight_share);
    ToneCurve c = tone_solve(render, t);
    double chroma = 1.0;
    for (int it = 0; it < kToneChromaRefinements; ++it) {
        c.chroma = chroma;
        const double got = tone_mean_chroma_after(preview, n, c);
        chroma = std::clamp(chroma * (render.Cmean / std::max(got, 1e-3)), kToneChromaLo, kToneChromaHi);
    }
    c.chroma = chroma;
    if (targets) *targets = t;
    return c;
}

// The stage in single precision, for the full image: f(L) from a table on t = sqrt(L / 100) (kToneLutSize + 1
// entries, linear interpolation; in t the curve is close to linear even where it is steep at the black end), the sRGB
// decode from a table of every 16-bit code. Pixels at or beyond the gamut's edge (a few per cent at most) go through the
// double-precision stage (tone_lab_apply, tone_lab_to_rgb_in_gamut), whose output this matches elsewhere to within 1
// of 65535.
constexpr int kToneLutSize = 8192;
class ToneStageFast {
public:
    // kToneRgbToXyz and kToneXyzToRgb in single precision.
    static constexpr float kM[9] = {(float)kToneRgbToXyz[0], (float)kToneRgbToXyz[1], (float)kToneRgbToXyz[2],
                                    (float)kToneRgbToXyz[3], (float)kToneRgbToXyz[4], (float)kToneRgbToXyz[5],
                                    (float)kToneRgbToXyz[6], (float)kToneRgbToXyz[7], (float)kToneRgbToXyz[8]};
    static constexpr float kMinv[9] = {(float)kToneXyzToRgb[0], (float)kToneXyzToRgb[1], (float)kToneXyzToRgb[2],
                                       (float)kToneXyzToRgb[3], (float)kToneXyzToRgb[4], (float)kToneXyzToRgb[5],
                                       (float)kToneXyzToRgb[6], (float)kToneXyzToRgb[7], (float)kToneXyzToRgb[8]};
    explicit ToneStageFast(const ToneCurve& q) : f_(kToneLutSize + 1) {
        for (int i = 0; i <= kToneLutSize; ++i) {
            const double t = (double)i / kToneLutSize;
            const double L2 = tone_curve_L(100.0 * t * t, q);
            f_[i] = (float)(L2 < 0 ? 0 : (L2 > 100 ? 100 : L2));
        }
        chroma_ = (float)q.chroma;
        q_ = q;
    }

    // One 16-bit sRGB pixel through the stage.
    void apply(const uint16_t in[3], uint16_t out[3]) const {
        const float* dec = decode_table();
        const float r = dec[in[0]], g = dec[in[1]], b = dec[in[2]];
        const float* m = kM;
        const float X = m[0] * r + m[1] * g + m[2] * b;
        const float Y = m[3] * r + m[4] * g + m[5] * b;
        const float Z = m[6] * r + m[7] * g + m[8] * b;
        const float fx = lab_f(X / (float)kToneXn), fy = lab_f(Y), fz = lab_f(Z / (float)kToneZn);
        const float L = 116.0f * fy - 16.0f;
        float A = 500.0f * (fx - fy), B = 200.0f * (fy - fz);
        float t = std::sqrt(std::min(std::max(L, 0.0f), 100.0f) / 100.0f) * (float)kToneLutSize;
        const int i = std::min((int)t, kToneLutSize - 1);
        t -= (float)i;
        const float L2 = f_[i] + t * (f_[i + 1] - f_[i]);
        float u = (std::sqrt(A * A + B * B) - 3.0f) / 5.0f;
        u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
        const float k = 1.0f + (chroma_ - 1.0f) * u * u * (3.0f - 2.0f * u);
        A *= k;
        B *= k;
        float lin[3];
        to_linear(L2, A, B, lin);
        if (std::max({lin[0], lin[1], lin[2]}) > kGamutMargin || std::min({lin[0], lin[1], lin[2]}) < kBlackMargin) {
            // At or beyond the gamut's edges, where the chroma scale's bisection (top) or the clip to 0 (bottom)
            // decides by comparisons the rounding of single precision would move: the pixel in double precision, as
            // tone_lab_apply and tone_lab_to_rgb_in_gamut do it.
            double lab[3], o[3];
            const double* dd = decode_table_double();
            tone_linear_to_lab(dd[in[0]], dd[in[1]], dd[in[2]], lab);
            tone_lab_apply(lab, q_);
            tone_lab_to_rgb_in_gamut(lab, o);
            for (int c = 0; c < 3; ++c) out[c] = (uint16_t)std::lround(o[c] * 65535.0);
            return;
        }
        for (int c = 0; c < 3; ++c) out[c] = (uint16_t)std::lround(encode(lin[c]) * 65535.0f);
    }

private:
    static const float* decode_table() {
        static const std::vector<float> d = [] {
            std::vector<float> t(65536);
            for (int v = 0; v < 65536; ++v) t[v] = (float)tone_srgb_decode(v / 65535.0);
            return t;
        }();
        return d.data();
    }
    // A table on sqrt(x), linearly interpolated: in sqrt(x) both the sRGB encode and the cube root are close to linear
    // (the steep start of each is its linear segment, computed directly).
    struct SqrtTable {
        std::vector<float> v;
        float top = 1.0f;
        float at(float x) const {
            float t = std::sqrt(std::min(std::max(x, 0.0f), top) / top) * (float)(v.size() - 1);
            const int i = std::min((int)t, (int)v.size() - 2);
            t -= (float)i;
            return v[i] + t * (v[i + 1] - v[i]);
        }
    };
    static const SqrtTable& encode_table() {
        static const SqrtTable e = [] {
            SqrtTable t;
            t.v.resize(4097);
            for (size_t i = 0; i < t.v.size(); ++i) {
                const double s = (double)i / (t.v.size() - 1);
                t.v[i] = (float)tone_srgb_encode(s * s);
            }
            return t;
        }();
        return e;
    }
    static const SqrtTable& cbrt_table() {
        static const SqrtTable c = [] {
            SqrtTable t;
            t.top = 1.25f;
            t.v.resize(8193);
            for (size_t i = 0; i < t.v.size(); ++i) {
                const double s = (double)i / (t.v.size() - 1);
                t.v[i] = (float)std::cbrt(s * s * 1.25);
            }
            return t;
        }();
        return c;
    }
    static float encode(float v) {
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        return v <= 0.0031308f ? 12.92f * v : encode_table().at(v);
    }
    static float lab_f(float t) {
        if (t <= 216.0f / 24389.0f) return (24389.0f / 27.0f * t + 16.0f) / 116.0f;
        return t < 1.25f ? cbrt_table().at(t) : std::cbrt(t);
    }
    static float lab_finv(float t) {
        return t * t * t > 216.0f / 24389.0f ? t * t * t : (116.0f * t - 16.0f) / (24389.0f / 27.0f);
    }
    static void to_linear(float L, float A, float B, float lin[3]) {
        const float fy = (L + 16.0f) / 116.0f, fx = fy + A / 500.0f, fz = fy - B / 200.0f;
        const float X = lab_finv(fx) * (float)kToneXn, Y = lab_finv(fy), Z = lab_finv(fz) * (float)kToneZn;
        const float* m = kMinv;
        lin[0] = m[0] * X + m[1] * Y + m[2] * Z;
        lin[1] = m[3] * X + m[4] * Y + m[5] * Z;
        lin[2] = m[6] * X + m[7] * Y + m[8] * Z;
    }
    static const double* decode_table_double() {
        static const std::vector<double> d = [] {
            std::vector<double> t(65536);
            for (int v = 0; v < 65536; ++v) t[v] = tone_srgb_decode(v / 65535.0);
            return t;
        }();
        return d.data();
    }
    static constexpr float kGamutMargin = 0.999f;  // linear: within 0.1 % of the maximum goes the double way,
    static constexpr float kBlackMargin = 1e-4f;   // and so does a channel this near 0 or below it
    std::vector<float> f_;
    float chroma_ = 1.0f;
    ToneCurve q_;
};

// The share of a 16-bit sRGB image's pixels (w x h, stride uint16 per pixel, every step-th each way) with a channel
// at the maximum: as it is (q null), or after the stage with curve q (as apply_tone_stage would write it).
static inline double tone_white_clip_share(const uint16_t* rgb, int w, int h, int stride, int step,
                                           const ToneCurve* q) {
    long long n = 0, clipped = 0;
    const ToneStageFast fast(q ? *q : ToneCurve{});
#pragma omp parallel for schedule(static) reduction(+ : n, clipped)
    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
            const uint16_t* p = rgb + ((size_t)y * w + x) * stride;
            ++n;
            if (!q) {
                clipped += (p[0] == 65535 || p[1] == 65535 || p[2] == 65535);
                continue;
            }
            uint16_t o[3];
            fast.apply(p, o);
            clipped += (o[0] == 65535 || o[1] == 65535 || o[2] == 65535);
        }
    }
    return n ? (double)clipped / (double)n : 0.0;
}

ToneCurve tone_plan_for_image(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                              const uint8_t* preview, size_t n, const uint16_t* rgb, int w, int h, int stride,
                              const ToneCurve& designed, const ToneTargets& designed_targets, ToneTargets* targets,
                              double* share) {
    const double before = tone_white_clip_share(rgb, w, h, stride, kToneClipSampleStep, nullptr);
    const ToneCurve& c1 = designed;
    *share = 1.0;
    *targets = designed_targets;
    auto ok = [&](const ToneCurve& c) {
        return tone_white_clip_share(rgb, w, h, stride, kToneClipSampleStep, &c) <= before + kToneClipAllowance;
    };
    if (ok(c1)) return c1;
    // The largest of the fixed shares that passes; none: the clipping is not the highlight move's, keep 1. Fixed shares
    // need no monotone clipping (a bisection does) and cost at most four more curves.
    for (size_t k = 1; k < sizeof(kToneClipShares) / sizeof(kToneClipShares[0]); ++k) {
        ToneTargets t;
        const ToneCurve c = tone_plan(render, pred_L2, pred_L98, pred_range, preview, n, kToneClipShares[k], &t);
        if (ok(c)) {
            *share = kToneClipShares[k];
            *targets = t;
            return c;
        }
    }
    return c1;
}

ToneClip apply_tone_stage(uint16_t* rgb, size_t n, int stride, const ToneCurve& q) {
    const ToneStageFast fast(q);
    long long wb = 0, wa = 0, kb = 0, ka = 0;
#pragma omp parallel for schedule(static) reduction(+ : wb, wa, kb, ka)
    for (long long i = 0; i < (long long)n; ++i) {
        uint16_t* px = rgb + (size_t)i * stride;
        wb += (px[0] == 65535 || px[1] == 65535 || px[2] == 65535);
        kb += (px[0] == 0 && px[1] == 0 && px[2] == 0);
        const uint16_t in[3] = {px[0], px[1], px[2]};
        fast.apply(in, px);
        wa += (px[0] == 65535 || px[1] == 65535 || px[2] == 65535);
        ka += (px[0] == 0 && px[1] == 0 && px[2] == 0);
    }
    ToneClip c;
    if (n) {
        c.white_before = (double)wb / (double)n;
        c.white_after = (double)wa / (double)n;
        c.black_before = (double)kb / (double)n;
        c.black_after = (double)ka / (double)n;
    }
    return c;
}

void LibRawDeleter::operator()(LibRaw* p) const {
  if (!p) return;
  p->free_image();
  delete p;
}

void Scan::scan_exposure(const LibRaw& r, double& shutter_out, double& iso_out) {
  const bool have_shutter = (has_exif_exp && exif_shutter_d > 0.0) || r.imgdata.other.shutter > 0;
  const bool have_iso = (has_exif_exp && exif_iso_d > 0.0) || r.imgdata.other.iso_speed > 0;
  shutter_out = (has_exif_exp && exif_shutter_d > 0.0)
                    ? exif_shutter_d
                    : (r.imgdata.other.shutter > 0 ? (double)r.imgdata.other.shutter : 0.125);
  iso_out = (has_exif_exp && exif_iso_d > 0.0)
                ? std::round(exif_iso_d)
                : (r.imgdata.other.iso_speed > 0 ? std::round((double)r.imgdata.other.iso_speed) : 100.0);
  if ((!have_shutter || !have_iso) && !warned_default_exposure) {
    warned_default_exposure = true;
    log_warn("Missing EXIF exposure metadata in %s; falling back to %.4g s, ISO %.0f\n",
            exposure_file.c_str(), shutter_out, iso_out);
  }
}

RgbInput Scan::image() {
  RgbInput img;
  img.width = proc->imgdata.sizes.iwidth;
  img.height = proc->imgdata.sizes.iheight;
  if (!rgb3.empty()) {
    img.pixels = rgb3.data();
    img.channels = 3;
  } else {
    img.pixels = &proc->imgdata.image[0][0];
    img.channels = 4;
  }
  return img;
}

void CpuBackend::render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, uint16_t* rgb_out) {
  if (in.n_frames == 1) {
    negicc::GrainAwareDebayerArgs ga{};
    ga.cfa_bayer = in.frames[0];
    ga.src_w = in.width;
    ga.src_h = in.height;
    ga.rx = region.x;
    ga.ry = region.y;
    ga.rw = region.w;
    ga.rh = region.h;
    ga.rot_cw = region.rot_cw;
    ga.hflip = region.hflip;
    ga.vflip = region.vflip;
    ga.dst_w = dst_w;
    ga.dst_h = dst_h;
    ga.filters = in.filters;
    ga.white_level = in.white_level;
    ga.eps = 4096.0f;
    ga.rgb_out = rgb_out;
    ga.flat = in.flat;
    ga.flat_x0 = region.x;
    ga.flat_y0 = region.y;
    ga.cfa_pitch = in.pitch;
    negicc::cpu_grain_aware_debayer(ga);
  } else {
    negicc::PixelShiftMergeArgs pm{};
    pm.bayer_f0 = in.frames[0];
    pm.bayer_f1 = in.frames[1];
    pm.bayer_f2 = in.frames[2];
    pm.bayer_f3 = in.frames[3];
    pm.src_w = in.width;
    pm.src_h = in.height;
    pm.rx = region.x;
    pm.ry = region.y;
    pm.rw = region.w;
    pm.rh = region.h;
    pm.rot_cw = region.rot_cw;
    pm.hflip = region.hflip;
    pm.vflip = region.vflip;
    pm.dst_w = dst_w;
    pm.dst_h = dst_h;
    pm.filters = in.filters;
    pm.white_level = in.white_level;
    pm.rgb_out = rgb_out;
    pm.flat = in.flat;
    pm.flat_x0 = region.x;
    pm.flat_y0 = region.y;
    pm.cfa_pitch = in.pitch;
    negicc::cpu_pixel_shift_merge(pm);
  }
}

void CpuBackend::convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                         const RgbInput& img) {
  const int width = img.width, height = img.height, ch = img.channels;
  uint16_t* pixels = img.pixels;
  // The sRGB encodings and ENCODE_NONE (the corrected RGB, already in [0, 1]) write 16-bit codes in place.
  auto convert_in_place = [&](auto stage) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        uint16_t* px = &pixels[((size_t)j * width + i) * ch];
        const Rgb v = stage(px);
        px[0] = negicc::quantize_u16(v.r);
        px[1] = negicc::quantize_u16(v.g);
        px[2] = negicc::quantize_u16(v.b);
      }
    }
  };
  if (out.encoding == ENCODE_SRGB || out.encoding == ENCODE_SRGB_LINEAR) {
    convert_in_place([&](const uint16_t* px) { return negicc::convert_pixel(pc, merged_matrix, px); });
  } else if (out.encoding == ENCODE_CUSTOM) {
    // Through Little-CMS from XYZ, a row at a time, into the image's own layout (3 samples, or LibRaw's 4).
    #pragma omp parallel
    {
      cmsHTRANSFORM thread_xform = cmsCreateTransform(out.xyz, TYPE_XYZ_FLT, out.custom,
                                                      ch == 3 ? TYPE_RGB_16 : TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
      std::vector<float> xyz_row(width * 3);

      #pragma omp for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          const uint16_t* px = &pixels[((size_t)j * width + i) * ch];
          const Rgb v = negicc::convert_pixel_pcs(pc, merged_matrix, px);
          xyz_row[i * 3 + 0] = v.r;
          xyz_row[i * 3 + 1] = v.g;
          xyz_row[i * 3 + 2] = v.b;
        }
        cmsDoTransform(thread_xform, xyz_row.data(), &pixels[(size_t)j * width * ch], width);
      }
      cmsDeleteTransform(thread_xform);
    }
  } else {
    convert_in_place([&](const uint16_t* px) { return negicc::convert_pixel_corrected(pc, merged_matrix, px); });
  }
}

static bool paths_match(const std::string& a, const std::string& b) {
  if (a == b) return true;
  std::error_code ec;
  if (std::filesystem::exists(a, ec) && std::filesystem::exists(b, ec) && std::filesystem::equivalent(a, b, ec)) {
    return true;
  }
  return false;
}

static bool extract_cli_egb(const nlohmann::ordered_json& cli, float& e_out, float& g_out, float& b_out) {
  if (!cli.is_array()) return false;
  bool got_e = false, got_g = false, got_b = false;
  for (size_t i = 0; i + 1 < cli.size(); ++i) {
    if (!cli[i].is_string() || !cli[i + 1].is_string()) continue;
    const std::string flag = cli[i].get<std::string>();
    const std::string val_str = cli[i + 1].get<std::string>();
    char* end = nullptr;
    const float v = std::strtof(val_str.c_str(), &end);
    if (!end || *end != '\0' || !std::isfinite(v)) continue;
    if (flag == "-E" || flag == "--exposure_comp") {
      e_out = v;
      got_e = true;
      ++i;
    } else if (flag == "--gain_g") {
      g_out = v;
      got_g = true;
      ++i;
    } else if (flag == "--gain_b") {
      b_out = v;
      got_b = true;
      ++i;
    }
  }
  return got_e && got_g && got_b;
}

// The arguments of a run that decide its result, normalised so that spelling and order do not matter: argv without
// the program name, the output (-o, --output, -oFILE) and the knobs (--brightness, --tint, --warmth); each option under
// its long name (-E and --exposure_comp alike) with its values ("--flag value", "--flag=value" or a short flag's
// attached "-Evalue" alike), the options sorted by name (an option given twice keeps its order), then "--" and the
// positional arguments (the inputs) in their order. Options neg_process does not know keep their own group.
static std::vector<std::string> result_args(const std::vector<std::string>& argv) {
  struct Opt {
    const char* shrt;
    const char* lng;
    int arity;  // values; -1: --roi (up to 7 numbers, or auto [h|v])
    bool skip;  // the output and the knobs
  };
  static const Opt kOpts[] = {
      {"-C", "--no_crop", 0, false},      {"-q", "--quality", 1, false},        {"-r", "--r_coeff", 3, false},
      {"-g", "--g_coeff", 3, false},      {"-b", "--b_coeff", 3, false},        {"-E", "--exposure_comp", 1, false},
      {"", "--no-metadata", 0, false},    {"", "--gain_g", 1, false},           {"", "--gain_b", 1, false},
      {"", "--brightness", 1, true},      {"", "--tint", 1, true},              {"", "--warmth", 1, true},
      {"", "--film_stock", 1, false},     {"", "--light_source", 1, false},     {"-p", "--film_profile", 1, false},
      {"-P", "--colorspace", 1, false},   {"", "--roi", -1, false},             {"", "--film_format", 1, false},
      {"", "--rot", 1, false},            {"", "--rot_cw", 1, false},           {"", "--hflip", 0, false},
      {"", "--vflip", 0, false},          {"-o", "--output", 1, true},          {"", "--dino", 1, false},
      {"", "--list_profiles", 0, false},  {"", "--dino_backend", 1, false},     {"", "--dino_iters", 1, false},
      {"", "--target", 1, false},         {"", "--bundle_target", 1, false},    {"", "--fit_curves", 1, false},
      {"", "--flat", 1, false},           {"", "--tint_trim", 1, false},      {"", "--tone_stage", 1, false},
  };
  auto is_number = [](const std::string& t) {
    char* e = nullptr;
    std::strtod(t.c_str(), &e);
    return !t.empty() && e && *e == '\0';
  };
  auto canonical = [](const char* lng) {
    const std::string l = lng;
    return l == "--rot" ? std::string("--rot_cw") : l == "--bundle_target" ? std::string("--target") : l;
  };
  std::vector<std::vector<std::string>> groups;  // [name, values...]
  std::vector<std::string> positional;
  for (size_t i = 1; i < argv.size(); ++i) {
    const std::string& a = argv[i];
    if (a.size() < 2 || a[0] != '-' || is_number(a)) {
      positional.push_back(a);
      continue;
    }
    std::string flag = a, attached;
    bool has_attached = false;
    if (a.compare(0, 2, "--") == 0) {
      const size_t eq = a.find('=');
      if (eq != std::string::npos) {
        flag = a.substr(0, eq);
        attached = a.substr(eq + 1);
        has_attached = true;
      }
    } else if (a.size() > 2) {  // a short flag with its value attached
      flag = a.substr(0, 2);
      attached = a.substr(2);
      has_attached = true;
    }
    const Opt* opt = nullptr;
    for (const Opt& o : kOpts)
      if (flag == o.lng || (o.shrt[0] && flag == o.shrt)) opt = &o;
    std::vector<std::string> g = {opt ? canonical(opt->lng) : flag};
    if (has_attached) g.push_back(attached);
    if (opt && opt->arity < 0) {
      // --roi: auto [h|v], or x y w h [rot_cw [hflip [vflip]]].
      while (g.size() < 8 && i + 1 < argv.size()) {
        const std::string& v = argv[i + 1];
        const bool ok = g.size() == 1   ? (v == "auto" || is_number(v))
                        : g[1] == "auto" ? (g.size() == 2 && (v == "h" || v == "v" || v == "H" || v == "V"))
                                         : is_number(v);
        if (!ok) break;
        g.push_back(argv[++i]);
      }
    } else if (opt) {
      for (int k = (int)g.size() - 1; k < opt->arity && i + 1 < argv.size(); ++k) g.push_back(argv[++i]);
    }
    if (opt && opt->skip) continue;
    groups.push_back(g);
  }
  std::stable_sort(groups.begin(), groups.end(),
                   [](const std::vector<std::string>& x, const std::vector<std::string>& y) { return x[0] < y[0]; });
  std::vector<std::string> out;
  for (const auto& g : groups) out.insert(out.end(), g.begin(), g.end());
  out.push_back("--");
  out.insert(out.end(), positional.begin(), positional.end());
  return out;
}

// Whether two argument tokens are the same number (1 and 1.0, 0.5 and .5): a finite number in full on both sides.
static bool same_number(const std::string& a, const std::string& b) {
  char* ea = nullptr;
  char* eb = nullptr;
  const double va = std::strtod(a.c_str(), &ea);
  const double vb = std::strtod(b.c_str(), &eb);
  return !a.empty() && !b.empty() && ea && *ea == '\0' && eb && *eb == '\0' && std::isfinite(va) &&
         std::isfinite(vb) && va == vb;
}

// Whether the previous run (its sidecar's argv) was given the same arguments as this one, apart from the output and the
// knobs: the same normalised arguments (result_args), a number also matching the same number written otherwise (1 and
// 1.0) and a path the same existing file. Any option that changes the result (the profile, the flat field, -r/-g/-b,
// -E, --target, the gains, --roi, --film_format, -P, --fit_curves, -q, --no_crop, --dino, --dino_iters ...) must
// therefore be the same. A sidecar without an argv array matches nothing.
static bool argv_matches_except_knobs(const nlohmann::ordered_json& argv_json, const Options& o, std::string& why) {
  if (!argv_json.is_array()) {
    why = "it records no argv";
    return false;
  }
  std::vector<std::string> prev, cur;
  for (const auto& a : argv_json) {
    if (!a.is_string()) {
      why = "its argv is not a list of strings";
      return false;
    }
    prev.push_back(a.get<std::string>());
  }
  for (int i = 0; i < o.argc; ++i) cur.push_back(o.argv[i]);
  const std::vector<std::string> pa = result_args(prev), ca = result_args(cur);
  if (pa.size() != ca.size()) {
    why = "it was given other arguments";
    return false;
  }
  for (size_t i = 0; i < pa.size(); ++i) {
    if (pa[i] != ca[i] && !same_number(pa[i], ca[i]) &&
        !(is_regular_file(pa[i]) && is_regular_file(ca[i]) && paths_match(pa[i], ca[i]))) {
      why = "it was given " + pa[i] + " where this run has " + ca[i];
      return false;
    }
  }
  return true;
}

// The checks and the reading of a previous run's sidecar j (try_load_cached_sidecar); false with why when it does not
// serve this conversion.
static bool load_cached_sidecar(const nlohmann::ordered_json& j, const Options& o, Setup& s, std::string& why) {
  if (j.value("schema", "") != "negicc.neg_process.sidecar/1") return why = "it has another schema", false;
  // The same build: another neg_process may convert the same arguments differently.
  if (!j.contains("tool") || !j["tool"].is_object() ||
      j["tool"].value("version", "") != std::string(neg_process_version())) {
    return why = std::string("it was written by another neg_process build (this one is ") + neg_process_version() + ")",
           false;
  }
  if (!argv_matches_except_knobs(j.value("argv", nlohmann::ordered_json()), o, why)) return false;

  if (!j.contains("inputs") || !j["inputs"].is_object() || !j["inputs"].contains("raw_files") ||
      !j["inputs"]["raw_files"].is_array()) {
    return why = "it records no inputs", false;
  }
  const auto& raw_files = j["inputs"]["raw_files"];
  if (raw_files.size() != o.files.size()) return why = "it converted another number of files", false;
  for (size_t i = 0; i < o.files.size(); ++i) {
    if (!raw_files[i].is_string() || !paths_match(raw_files[i].get<std::string>(), o.files[i])) {
      return why = "it converted other files", false;
    }
  }
  // The same files, not only the same names: a capture saved again under the same name has another size or CRC-32.
  // (Not the modification time: a file copied or downloaded again, as negicc-web stages its inputs, keeps its content.)
  const auto& stats = j["inputs"].value("raw_file_stats", nlohmann::ordered_json());
  if (!stats.is_array() || stats.size() != o.files.size()) {
    return why = "it records no input file sizes and CRCs", false;
  }
  s.input_fingerprints.assign(o.files.size(), FileFingerprint{});
  for (size_t i = 0; i < o.files.size(); ++i) {
    FileFingerprint& fp = s.input_fingerprints[i];
    fp.ok = file_fingerprint(o.files[i], fp.size, fp.crc);
    const auto& st = stats[i];
    // Both recorded, as unsigned integers: a missing key must not read as 0 and match an empty file's CRC.
    if (!st.is_object() || !st.contains("size") || !st["size"].is_number_unsigned() || !st.contains("crc32") ||
        !st["crc32"].is_number_unsigned()) {
      return why = "it records no size and CRC-32 for " + o.files[i], false;
    }
    if (!fp.ok || st["size"].get<uintmax_t>() != fp.size || st["crc32"].get<uint32_t>() != fp.crc) {
      return why = o.files[i] + " is not the file it converted", false;
    }
  }

  if (!j.contains("profile") || !j["profile"].is_object()) return false;
  const auto& prof = j["profile"];
  const std::string prof_type = prof.value("type", "");
  const std::string prof_path = prof.value("path", "");
  if (s.is_bundle) {
    if (prof_type != "bundle") return false;
    if (!paths_match(prof_path, s.bundle.path)) return false;
    if (prof.value("input_space", "") != std::string(to_string(s.bundle.input_space))) return false;
  } else {
    if (prof_type != "icc") return false;
    if (!paths_match(prof_path, s.film_prof_arg) && !paths_match(prof_path, resolve_profile_path(s.film_prof_arg))) {
      return false;
    }
  }

  if (!j.contains("geometry") || !j["geometry"].is_object()) return false;
  const auto& geom = j["geometry"];
  if (!geom.contains("rot_cw") || !geom["rot_cw"].is_number_integer() || geom["rot_cw"].get<int>() != o.rot_cw) {
    return false;
  }
  if (!geom.contains("hflip") || !geom["hflip"].is_boolean() || geom["hflip"].get<bool>() != o.hflip) return false;
  if (!geom.contains("vflip") || !geom["vflip"].is_boolean() || geom["vflip"].get<bool>() != o.vflip) return false;
  const std::string roi_source = geom.value("roi_source", "");
  if (o.roi_auto) {
    if (roi_source != "auto" && roi_source != "auto_fallback") return false;
    if (j.contains("frame_detection") && j["frame_detection"].is_object()) {
      // The format the detector found (with --film_format auto it records what it found, not "auto").
      if (o.film_format != FILM_FORMAT_AUTO &&
          j["frame_detection"].value("film_format", "") != std::string(film_format_name(o.film_format))) {
        return false;
      }
    }
  } else if (o.has_roi) {
    if (roi_source != "cli" || !geom.contains("roi") || !geom["roi"].is_array() || geom["roi"].size() != 4) {
      return false;
    }
    for (int k = 0; k < 4; ++k) {
      if (!geom["roi"][k].is_number_integer() || geom["roi"][k].get<int>() != o.roi[k]) return false;
    }
  } else {
    if (roi_source != "none") return false;
  }

  if (!j.contains("matrix") || !j["matrix"].is_object() || !j["matrix"].contains("pins") ||
      !j["matrix"]["pins"].is_object()) {
    return false;
  }
  const bool saved_pin_e = j["matrix"]["pins"].value("e", false);
  if (saved_pin_e != s.exposure_given) return false;

  // Match either a previous bundle Auto run ("auto") or a previous single-target DINOv3 run ("dino").
  if (s.is_bundle && !s.exposure_given && j.contains("auto") && j["auto"].is_object()) {
    const auto& a = j["auto"];
    if (!a.contains("target_idx") || !a["target_idx"].is_number_integer()) return false;
    const int tgt_idx = a["target_idx"].get<int>();
    if (tgt_idx < 0 || tgt_idx >= (int)s.bundle.targets.size()) return false;
    if (s.manual_target_idx >= 0 && s.manual_target_idx != tgt_idx) return false;
    if (a.value("target", "") != s.bundle.targets[tgt_idx].name) return false;
    if (!a.contains("e") || !a["e"].is_number() || !a.contains("g") || !a["g"].is_number() || !a.contains("b") ||
        !a["b"].is_number()) {
      return false;
    }
    const float base_e = a["e"].get<float>();
    const float base_g = a["g"].get<float>();
    const float base_b = a["b"].get<float>();
    auto positive = [](float v) { return std::isfinite(v) && v > 0.0f; };
    if (!positive(base_e) || !positive(base_g) || !positive(base_b)) return why = "its Auto result is not positive", false;

    s.has_cached_sidecar = true;
    s.bundle_search = true;
    s.cached_sidecar.is_auto = true;
    s.cached_sidecar.target_idx = tgt_idx;
    s.cached_sidecar.target_name = s.bundle.targets[tgt_idx].name;
    s.cached_sidecar.base_e = base_e;
    s.cached_sidecar.base_g = base_g;
    s.cached_sidecar.base_b = base_b;
    s.cached_sidecar.auto_json_str = a.dump();
    s.dino_path = a.value("model", "");
    s.cached_sidecar.dino_backend = a.value("backend", o.dino_backend);
    // Fitted G/B input curves of the previous run (the curve fit's): the knobs act on its result (the
    // re-metered E and the fit's g, b) through the same combined profile, so the curves are replayed, not refitted.
    // A fit that fell back (applied false, channels null) has no curves to replay, but its re-metered values are the
    // base all the same.
    const bool no_curves = j.contains("input_curves") && j["input_curves"].is_object() &&
                           j["input_curves"].value("applied", true) == false &&
                           j["input_curves"].value("channels", nlohmann::ordered_json()).is_null();
    // The re-metered values replace the checked ones: checked again (a remeter.e of -1 aborted solve_grey_knobs).
    auto read_remeter = [&](const nlohmann::ordered_json& ic) {
      if (!ic.contains("remeter") || !ic["remeter"].is_object()) return true;
      const auto& rm = ic["remeter"];
      s.cached_sidecar.base_e = rm.value("e", base_e);
      s.cached_sidecar.base_g = rm.value("g", base_g);
      s.cached_sidecar.base_b = rm.value("b", base_b);
      return positive(s.cached_sidecar.base_e) && positive(s.cached_sidecar.base_g) &&
             positive(s.cached_sidecar.base_b);
    };
    if (no_curves) {
      s.cached_sidecar.input_curves_json_str = j["input_curves"].dump();
      if (!read_remeter(j["input_curves"])) return why = "its re-metered values are not positive", false;
    }
    if (j.contains("input_curves") && j["input_curves"].is_object() && !no_curves) {
      const auto& ic = j["input_curves"];
      negicc::InputCurves c;
      bool ok = ic.contains("channels") && ic["channels"].is_object() && ic.value("target_idx", -1) == tgt_idx;
      const char* names[2] = {"G", "B"};
      for (int i = 0; ok && i < 2; ++i) {
        const auto& ch = ic["channels"].value(names[i], nlohmann::ordered_json());
        ok = ch.is_object() && ch.contains("a") && ch.contains("k") && ch.contains("ref") && ch.contains("u_med");
        if (!ok) break;
        c.a[i] = ch["a"].get<float>();
        c.k[i] = ch["k"].get<float>();
        c.ref[i] = ch["ref"].get<float>();
        c.u_med[i] = ch["u_med"].get<float>();
        if (ch.contains("u_lo")) c.u_lo[i] = ch["u_lo"].get<float>();
      }
      c.hold = ic.value("hold", false);
      if (c.hold && !(ic["channels"]["G"].contains("u_lo") && ic["channels"]["B"].contains("u_lo"))) ok = false;
      if (ok && ic["channels"].contains("R")) {  // the red highlight-cliff curve
        const auto& r = ic["channels"]["R"];
        ok = r.is_object() && r.contains("k") && r.contains("ref") && r.contains("u_lo");
        if (ok) {
          c.k_r = r["k"].get<float>();
          c.ref_r = r["ref"].get<float>();
          c.u_lo_r = r["u_lo"].get<float>();
          ok = std::isfinite(c.k_r) && c.ref_r > 0.0f && std::isfinite(c.u_lo_r);
        }
      }
      for (int i = 0; ok && i < 2; ++i) {
        ok = std::isfinite(c.a[i]) && std::isfinite(c.k[i]) && std::isfinite(c.u_med[i]) && std::isfinite(c.u_lo[i]);
      }
      if (!ok || !positive(c.ref[0]) || !positive(c.ref[1])) return why = "its input curves are not valid", false;
      s.cached_sidecar.has_curves = true;
      s.cached_sidecar.curves = c;
      s.cached_sidecar.curves_source = ic.value("source", "fit");
      s.cached_sidecar.input_curves_json_str = ic.dump();
      if (!read_remeter(ic)) return why = "its re-metered values are not positive", false;
    }
    if (j.contains("intent_field") && j["intent_field"].is_object()) {
      s.cached_sidecar.intent_field_json_str = j["intent_field"].dump();
    }
    if (j.contains("heads") && j["heads"].is_object()) s.cached_sidecar.heads_json_str = j["heads"].dump();
    // The tint trim and the tone curve it applied: replayed as recorded (the knobs act on top).
    auto numbers = [](const nlohmann::ordered_json& a, int n, double* out) {
      if (!a.is_array() || (int)a.size() != n) return false;
      for (int i = 0; i < n; ++i) {
        if (!a[i].is_number()) return false;
        out[i] = a[i].get<double>();
      }
      return true;
    };
    if (j.contains("input_curves") && j["input_curves"].is_object()) {
      const auto& ic = j["input_curves"];
      // The sidecar records the tint trim under "offset".
      if (ic.contains("offset") && ic["offset"].is_object() && ic["offset"].value("applied", false)) {
        const auto& of = ic["offset"];
        TintTrimTables& t = s.cached_sidecar.tint_trim;
        double d[3], fb[3];
        if (!numbers(of.value("theta", nlohmann::ordered_json()), 4, t.theta) ||
            !numbers(of.value("piv", nlohmann::ordered_json()), 3, t.piv) ||
            !numbers(of.value("diag", nlohmann::ordered_json()), 3, d) ||
            !numbers(of.value("film_base", nlohmann::ordered_json()), 3, fb))
          return why = "its tint trim is not valid", false;
        for (int c = 0; c < 3; ++c) {
          t.diag[c] = (float)d[c];
          t.film_base[c] = (float)fb[c];
        }
        s.cached_sidecar.has_tint_trim = true;
      }
    }
    if (j.contains("tone_stage") && j["tone_stage"].is_object() && j["tone_stage"].value("applied", false)) {
      const auto& cv = j["tone_stage"].value("curve", nlohmann::ordered_json());
      double q[5];
      const char* keys[5] = {"black", "white", "gamma", "contrast", "chroma"};
      bool ok = cv.is_object();
      for (int i = 0; ok && i < 5; ++i) {
        ok = cv.contains(keys[i]) && cv[keys[i]].is_number();
        if (ok) q[i] = cv[keys[i]].get<double>();
      }
      if (!ok) return why = "its tone curve is not valid", false;
      ToneCurve& tc = s.cached_sidecar.tone_curve;
      tc.black = q[0];
      tc.white = q[1];
      tc.gamma = q[2];
      tc.contrast = q[3];
      tc.chroma = q[4];
      s.cached_sidecar.has_tone = true;
    }
    return true;
  }

  if (!s.bundle_search && j.contains("dino") && j["dino"].is_object()) {
    const auto& d = j["dino"];
    // The bracket the previous run converted on (its sidecar's): with --target it must be that one.
    int expected_idx = -1;
    if (s.is_bundle) {
      const std::string recorded = prof.value("bundle_target", "");
      for (size_t t = 0; t < s.bundle.targets.size(); ++t) {
        if (s.bundle.targets[t].name == recorded) expected_idx = (int)t;
      }
      if (expected_idx < 0) return why = "its bracket is not in this bundle", false;
      if (s.manual_target_idx >= 0 && s.manual_target_idx != expected_idx) {
        return why = "it converted another bracket", false;
      }
    }
    float base_e = 0.0f, base_g = 0.0f, base_b = 0.0f;
    if (j.contains("knobs") && j["knobs"].is_object() && j["knobs"].contains("base") &&
        j["knobs"]["base"].is_object()) {
      const auto& kb = j["knobs"]["base"];
      if (!kb.contains("e") || !kb["e"].is_number() || !kb.contains("g") || !kb["g"].is_number() ||
          !kb.contains("b") || !kb["b"].is_number()) {
        return false;
      }
      base_e = kb["e"].get<float>();
      base_g = kb["g"].get<float>();
      base_b = kb["b"].get<float>();
    } else if (!j.contains("cli_equivalent") || !extract_cli_egb(j["cli_equivalent"], base_e, base_g, base_b)) {
      return false;
    }
    if (!std::isfinite(base_e) || !std::isfinite(base_g) || !std::isfinite(base_b) || !(base_e > 0.0f) ||
        !(base_g > 0.0f) || !(base_b > 0.0f)) {
      return why = "its result is not positive", false;
    }
    if (s.exposure_given && std::fabs(base_e - o.exposure_comp) > 1e-4f * std::max(1.0f, std::fabs(o.exposure_comp))) {
      return false;
    }

    s.has_cached_sidecar = true;
    s.cached_sidecar.is_auto = false;
    s.cached_sidecar.target_idx = expected_idx;
    if (s.is_bundle && expected_idx >= 0) s.cached_sidecar.target_name = s.bundle.targets[expected_idx].name;
    s.cached_sidecar.base_e = base_e;
    s.cached_sidecar.base_g = base_g;
    s.cached_sidecar.base_b = base_b;
    s.cached_sidecar.dino_json_str = d.dump();
    if (j.contains("intent_field") && j["intent_field"].is_object()) {
      s.cached_sidecar.intent_field_json_str = j["intent_field"].dump();
    }
    s.dino_path = d.value("model", "");
    s.cached_sidecar.dino_backend = d.value("backend", o.dino_backend);
    if (j.contains("timings_ms") && j["timings_ms"].is_object()) {
      const auto& tm = j["timings_ms"];
      if (tm.contains("dino_views") && tm["dino_views"].is_number()) {
        s.cached_sidecar.t_dino_views_ms = tm["dino_views"].get<double>();
      }
      if (tm.contains("dino_render") && tm["dino_render"].is_number()) {
        s.cached_sidecar.t_dino_render_ms = tm["dino_render"].get<double>();
      }
      if (tm.contains("dino_inference") && tm["dino_inference"].is_number()) {
        s.cached_sidecar.t_dino_infer_ms = tm["dino_inference"].get<double>();
      }
      if (tm.contains("dino_search") && tm["dino_search"].is_number()) {
        s.cached_sidecar.t_dino_search_ms = tm["dino_search"].get<double>();
      }
    }
    return true;
  }

  return false;
}

// Checks whether <output>.json exists and was written by this build for the same inputs and arguments (apart from the
// output and the knobs), so --brightness, --tint and --warmth move its Auto or DINOv3 result instead of searching
// again.
// Anything unreadable or different refuses the reuse, says why, and leaves s as it was.
static bool try_load_cached_sidecar(const Options& o, Setup& s) {
  const std::string sidecar_path = o.output + ".json";
  if (!is_regular_file(sidecar_path)) return false;
  std::ifstream ifs(sidecar_path);
  if (!ifs.is_open()) return false;
  const nlohmann::ordered_json j = nlohmann::ordered_json::parse(ifs, nullptr, false);
  const bool had = s.has_cached_sidecar, had_search = s.bundle_search;
  const CachedSidecar had_cache = s.cached_sidecar;
  const std::string had_dino_path = s.dino_path;
  std::string why = "it does not match this conversion";
  bool ok = false;
  if (!j.is_object()) {
    why = "it is not JSON";
  } else {
    try {
      ok = load_cached_sidecar(j, o, s, why);
    } catch (const std::exception& e) {  // a field of the wrong type
      why = std::string("it cannot be read (") + e.what() + ")";
      ok = false;
    }
  }
  if (!ok) {
    s.has_cached_sidecar = had;
    s.bundle_search = had_search;
    s.cached_sidecar = had_cache;
    s.dino_path = had_dino_path;
    log_info("Not reusing %s: %s; converting from scratch\n", sidecar_path.c_str(), why.c_str());
  }
  return ok;
}

int prepare(const Options& o, Setup& s) {
  if (!o.flat.empty()) {
    try {
      s.flat_field = negicc::load_flat_field_file(o.flat);
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
  }

  // The profile: -p, else the built-in bundle of --film_stock and --light_source (or the light the input implies).
  ProfileBundle& bundle = s.bundle;
  if (!o.light_source.empty() && o.implied_light && o.light_source != o.implied_light && !o.tiff_input) {
    log_error("--light_source %s does not fit %s, which takes %s\n", o.light_source.c_str(),
            o.input_mode == InputMode::kLinearDng ? "a linear DNG"
            : o.input_mode == InputMode::kThreeShot ? "three captures, one per light"
                                                    : "a JPEG, converted from sRGB to linear Rec.2020",
            o.implied_light);
    return 1;
  }
  if (!o.light_source.empty() && o.film_stock.empty()) {
    log_error("--light_source requires --film_stock\n");
    return 1;
  }
  if (!o.film_profile.empty()) {
    s.film_prof_arg = o.film_profile;
    try {
      s.is_bundle = load_bundle_arg(s.film_prof_arg, bundle);
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    if (!s.is_bundle && !is_regular_file(resolve_profile_path(s.film_prof_arg))) {
      log_error("No profile bundle JSON or ICC profile '%s'\n", s.film_prof_arg.c_str());
      return 1;
    }
  } else if (!o.film_stock.empty()) {
    if (o.light_source.empty() && !o.implied_light) {
      log_error("--film_stock requires --light_source (tri or d65)\n");
      return 1;
    }
    const std::string light = !o.light_source.empty() ? o.light_source : std::string(o.implied_light);
    if (o.light_source.empty()) log_info("Light source: %s, from the input\n", light.c_str());
    std::string error;
    const std::string key = embedded_bundle_key(o.film_stock, light, error);
    if (key.empty()) {
      log_error("%s (--list_profiles lists them)\n", error.c_str());
      return 1;
    }
    try {
      if (!load_embedded_bundle(key, bundle)) {
        log_error("Cannot load built-in profile bundle %s\n", key.c_str());
        return 1;
      }
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    s.is_bundle = true;
  }
  if (s.is_bundle) {
    log_info("Profile bundle: %s (%s, %zu brackets)\n", bundle.key.c_str(), bundle.path.c_str(), bundle.targets.size());
  }
  s.has_profile = s.is_bundle || !s.film_prof_arg.empty();
  if (o.knobs_given && !s.has_profile) {
    log_error("--brightness, --tint and --warmth require a film profile (--film_stock and --light_source, or "
              "--film_profile)\n");
    return 1;
  }

  if (o.target_given) {
    if (!s.is_bundle) {
      log_error("--target requires a JSON profile bundle\n");
      return 1;
    }
    const std::string& tgt_str = o.target;
    char* end = nullptr;
    long val = strtol(tgt_str.c_str(), &end, 10);
    if (end && *end == '\0' && val >= 0 && val < (long)bundle.targets.size()) {
      s.manual_target_idx = (int)val;
    } else {
      for (size_t i = 0; i < bundle.targets.size(); ++i) {
        if (bundle.targets[i].name == tgt_str || bundle.targets[i].display_name == tgt_str) {
          s.manual_target_idx = (int)i;
          break;
        }
      }
    }
    if (s.manual_target_idx < 0) {
      log_error("Target '%s' not found in bundle\n", tgt_str.c_str());
      return 1;
    }
  }
  // The parameter search runs with a profile unless --gain_g or --gain_b is given. On a bundle it is
  // negicc-station's conversion parameter search (multi_target_optimizer.h), which picks the bracket, the exposure
  // and the gains; with --target or -E (and on a single ICC profile) it solves the gains, and the exposure unless -E
  // gives it, on one profile.
  s.exposure_given = o.exposure_given;
  s.use_dino = s.has_profile && !o.gain_g_given && !o.gain_b_given;
  s.bundle_search = s.use_dino && s.is_bundle && s.manual_target_idx < 0 && !s.exposure_given;
  if ((!o.dino_adapter.empty() || !o.neutral_head_file.empty() || !o.tone_head_file.empty()) && !s.use_dino) {
    log_error("--dino requires the parameter search: a profile (--film_stock and --light_source, or "
                    "--film_profile) without --gain_g or --gain_b\n");
    return 1;
  }

  // The crosstalk correction: the bundle's, with any rows -r, -g and -b give in place of its own (for the search too).
  s.r_coeff = o.r_coeff;
  s.g_coeff = o.g_coeff;
  s.b_coeff = o.b_coeff;
  const bool* row_given = o.row_given;
  std::vector<float>* rows[3] = {&s.r_coeff, &s.g_coeff, &s.b_coeff};
  // A rec2020_linear (white-light) bundle takes the white-light decode, which has no crosstalk to correct and, as in
  // negicc-station, decodes single-shot captures only.
  // py-ref: negicc_station/src/pipeline_api.py @ 813166c (prepare_image)
  s.white_light = s.is_bundle && bundle.input_space == InputSpace::kRec2020Linear;
  s.camera_rgb_input = o.tiff_input && s.is_bundle && !s.white_light;
  if (s.white_light) {
    if (row_given[0] || row_given[1] || row_given[2]) {
      log_error("-r, -g and -b cannot be used with the rec2020_linear profile bundle %s: its white-light "
                      "decode has no crosstalk to correct\n", bundle.key.c_str());
      return 1;
    }
    if (o.files.size() == 4) {
      log_error("The rec2020_linear profile bundle %s converts single-shot captures only, not pixel "
                      "shift\n", bundle.key.c_str());
      return 1;
    }
    if (o.input_mode == InputMode::kThreeShot) {
      log_error("The rec2020_linear profile bundle %s is for white light; three captures, one per "
                      "light, take a camera_raw_rgb (tri) bundle\n", bundle.key.c_str());
      return 1;
    }
  }
  // Three captures, one per light, have no crosstalk between the lights to correct: each channel comes from the
  // capture under its own light, so the correction is the identity unless -r, -g or -b give one.
  if (o.input_mode == InputMode::kThreeShot && s.is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) continue;
      for (int c = 0; c < 3; ++c) {
        bundle.crosstalk_matrix[r * 3 + c] = r == c ? 1.0f : 0.0f;
        bundle.crosstalk_matrix_d[r * 3 + c] = r == c ? 1.0 : 0.0;
      }
    }
    bundle.has_crosstalk = true;
  }
  if (s.is_bundle) {
    for (int r = 0; r < 3; ++r) {
      if (row_given[r]) {
        for (int c = 0; c < 3; ++c) {
          bundle.crosstalk_matrix[r * 3 + c] = (*rows[r])[c];
          bundle.crosstalk_matrix_d[r * 3 + c] = (double)(*rows[r])[c];
        }
      } else if (bundle.has_crosstalk) {
        rows[r]->assign(bundle.crosstalk_matrix + r * 3, bundle.crosstalk_matrix + r * 3 + 3);
      }
    }
  }
  s.has_crosstalk = row_given[0] || row_given[1] || row_given[2] || (s.is_bundle && bundle.has_crosstalk);
  // An RGB TIFF or JPEG is colour-managed RGB, converted to linear Rec.2020: not the camera RGB a crosstalk
  // correction (a camera_raw_rgb bundle) needs.
  if (o.input_mode == InputMode::kRgbFile && !s.camera_rgb_input && s.has_crosstalk) {
    log_error("%s is an RGB TIFF or JPEG, converted to linear Rec.2020: give a rec2020_linear (d65) "
                    "profile bundle, not one with a crosstalk correction (a TIFF of linear camera RGB takes "
                    "--light_source tri)\n", o.files[0].c_str());
    return 1;
  }
  if (o.input_mode == InputMode::kLinearDng && s.is_bundle && !s.white_light) {
    log_error("A linear DNG is converted to linear Rec.2020 with its own colour data: give a "
                    "rec2020_linear (d65) profile bundle\n");
    return 1;
  }
  s.raw_decode = s.white_light ? RawDecode::kWhiteLight
                 : s.has_crosstalk ? RawDecode::kCameraRgb
                                   : RawDecode::kRec2020;

  // Fitted G/B input curves (input_curves.h; negicc-station designs/plan_fit_curves.md): white light only, since with
  // a crosstalk matrix the film base does not mark the dark end of every channel. The output must apply the film
  // profile (-P): without it the TIFF is the corrected camera RGB with the bracket's own ICC profile, which does not
  // carry the curves.
  const bool fit_on = o.fit_curves == Options::kFitCurvesOn;
  if (fit_on || o.input_curves_given) {
    const char* opt = fit_on ? "--fit_curves on" : "--fit_curves with recorded curves";
    // TODO: the fitted curves and the tone stage on single-shot tri-LED. They need five changes: (1) let the fit run on
    // a single-shot crosstalk bundle (other tri-LED cases unchanged); (2) call the fit on the CFA path, which Sony raws
    // take and which has no fit call; (3) drop the crosstalk rejection in run_white_light_fit; (4) run the
    // fit on M_frame so its E is in the export's units (the search's matrix is M_frame / e_norm on tri-LED); (5) anchor
    // the curves at the film base in camera RGB, since the detector returns the base already crosstalk-corrected and
    // M_frame would apply the matrix a second time. Open before enabling: the fit lowers the mean chroma (by about a
    // fifth where tried), and three-shot scans and other stocks are untested.
    if (!s.is_bundle || !s.white_light || s.has_crosstalk) {
      log_error("%s needs a white-light (d65, rec2020_linear) profile bundle: with a crosstalk matrix the film base "
                "does not mark the dark end of every channel\n", opt);
      return 1;
    }
    if (o.colorspace.empty()) {
      log_error("%s needs -P (e.g. -P srgb): without it the output is the corrected camera RGB with the bracket's ICC "
                "profile, which does not carry the curves\n", opt);
      return 1;
    }
  }
  // The fit scores the grid in CIELAB through linear sRGB (input_curves.h, to_display), which is what -P srgb and
  // srgb-g10 write; a custom -P goes through its ICC profile with LittleCMS, which the fit does not model, while the
  // sidecar recorded the fit as made for it. So fitting takes -P srgb or srgb-g10 (srgb-linear); replaying recorded
  // curves takes any -P, since it renders the combined profile and fits nothing.
  const bool fit_encoding_ok = o.colorspace == "srgb" || o.colorspace == "srgb-g10" || o.colorspace == "srgb-linear";
  if (fit_on && !fit_encoding_ok) {
    log_error("--fit_curves on needs -P srgb or -P srgb-g10: the fit scores the frame through linear sRGB, not "
              "through a custom output profile\n");
    return 1;
  }
  if (fit_on && !s.bundle_search) {
    log_error("--fit_curves on fits after the bundle search: it takes no --target, -E, --gain_g or --gain_b "
              "(--fit_curves with recorded curves, ref:refG,refB,u_medG,u_medB,aG,kG,aB,kB, replays them)\n");
    return 1;
  }
  // --fit_curves auto or on: the curve fit (with its tint trim and the tone stage) wherever it applies (on has
  // checked above that it does), nothing elsewhere; off: none.
  s.fit_curves_active = o.fit_curves != Options::kFitCurvesOff && s.is_bundle && s.white_light && !s.has_crosstalk &&
                        fit_encoding_ok && s.bundle_search;
  if (o.input_curves_given) {
    // Recorded curves replay a recorded result (a sidecar's cli_equivalent): its bracket and its E, g and b on the
    // combined profile. Without -E and the gains the single-bracket DINOv3 search would run on the combined profile,
    // which the curves were not fitted with, and give other values.
    if (s.manual_target_idx < 0 || !o.exposure_given || !o.gain_g_given || !o.gain_b_given) {
      log_error("--fit_curves with recorded curves needs --target, -E, --gain_g and --gain_b: the curves belong to "
                "one bracket's profile at the recorded E, g and b (the sidecar's cli_equivalent gives all of them)\n");
      return 1;
    }
    // The shape the fit holds every curve to (monotonic and within max_dev_stops of the base TRC over the whole input
    // table, curve_shape_ok): numbers typed by hand, or edited, could fold the table over or move it arbitrarily far
    // from the bracket's TRC, and were composed unchecked. A fit's own numbers (written to 9 significant digits, which
    // round-trip a float) pass.
    {
      const negicc::InputCurves& c = o.input_curves;
      const ProfileData& pd = bundle.targets[s.manual_target_idx].film_prof.data;
      const float dev = negicc::CurveFitOptions().max_dev_stops;
      const char* names[2] = {"G", "B"};
      for (int i = 0; i < 2; ++i) {
        if (!negicc::curve_shape_ok(c.a[i], c.k[i], c.u_med[i], c.ref[i], pd.in_trc_size[i + 1], dev, c.u_hold(i))) {
          log_error("--fit_curves: the recorded %s curve is not monotonic, or departs more than %g stops from the "
                    "bracket's input TRC, over its input table\n", names[i], dev);
          return 1;
        }
      }
    }
    // The --target bracket's profile becomes the combined profile before anything renders through it (its un-curved
    // tables kept for the knobs).
    s.curves_base_idx = s.manual_target_idx;
    s.curves_base_prof = bundle.targets[s.manual_target_idx].film_prof;
    s.curves_base_prof.rebind_pointers();
    apply_input_curves(bundle.targets[s.manual_target_idx].film_prof, o.input_curves);
    s.input_curves = o.input_curves;
    s.input_curves_source = "cli";
  }

  // A recorded tint trim and tone curve (a sidecar's cli_equivalent) replay a recorded result like the
  // recorded curves: on the --target bracket at the given E, g and b, with nothing searched or predicted.
  if (o.input_curves_given || o.tint_trim_replay_given || o.tone_replay_given) s.replay_source = "cli";
  if (o.tint_trim_replay_given || o.tone_replay_given) {
    if (!s.is_bundle || !s.white_light || s.has_crosstalk || s.manual_target_idx < 0 || !o.exposure_given ||
        !o.gain_g_given || !o.gain_b_given) {
      log_error("--tint_trim theta:... and --tone_stage curve:... replay a white-light result: they need a d65 "
                "bundle, --target, -E, --gain_g and --gain_b (the sidecar's cli_equivalent gives all of them)\n");
      return 1;
    }
    if (o.tint_trim_replay_given) {
      const double* v = o.tint_trim_replay;
      TintTrimTables& t = s.tint_trim;
      for (int k = 0; k < 4; ++k) t.theta[k] = v[k];
      for (int c = 1; c < 3; ++c) {
        t.piv[c] = v[3 + c];
        t.diag[c] = (float)v[5 + c];
        t.film_base[c] = (float)v[7 + c];
      }
      s.has_tint_trim = true;
      s.tint_trim_target_idx = s.manual_target_idx;
      s.tint_trim_base_prof = bundle.targets[s.tint_trim_target_idx].film_prof;  // the recorded curves already composed
      s.tint_trim_base_prof.rebind_pointers();
    }
    if (o.tone_replay_given) {
      const double* q = o.tone_replay;
      s.tone_curve = {q[0], q[1], q[2], q[3], q[4]};
      s.has_tone = true;
    }
  }

  // --brightness, --tint and --warmth on a previous run's output (<output>.json): its Auto or DINOv3 result is the
  // base the knobs move, so neither the backbone nor the search runs again; its fitted curves are replayed on its
  // bracket.
  if (o.knobs_given && s.use_dino && try_load_cached_sidecar(o, s)) {
    log_info("Reusing cached metadata from %s (base E %.4f g %.4f b %.4f%s%s%s)\n", (o.output + ".json").c_str(),
             s.cached_sidecar.base_e, s.cached_sidecar.base_g, s.cached_sidecar.base_b,
             s.is_bundle ? ", target " : "", s.is_bundle ? s.cached_sidecar.target_name.c_str() : "",
             s.cached_sidecar.has_curves ? ", fitted curves replayed" : "");
    if (s.cached_sidecar.has_curves && s.input_curves_source.empty()) {
      s.curves_base_idx = s.cached_sidecar.target_idx;
      s.curves_base_prof = bundle.targets[s.cached_sidecar.target_idx].film_prof;
      s.curves_base_prof.rebind_pointers();
      apply_input_curves(bundle.targets[s.cached_sidecar.target_idx].film_prof, s.cached_sidecar.curves);
      s.input_curves = s.cached_sidecar.curves;
      s.input_curves_source = s.cached_sidecar.curves_source;
    }
    s.fit_curves_active = false;  // the curves (or none) come from the previous run
    s.replay_source = "knobs";
    if (s.cached_sidecar.has_tint_trim && s.cached_sidecar.target_idx >= 0) {
      s.has_tint_trim = true;
      s.tint_trim = s.cached_sidecar.tint_trim;
      s.tint_trim_target_idx = s.cached_sidecar.target_idx;
      s.tint_trim_base_prof = bundle.targets[s.tint_trim_target_idx].film_prof;
      s.tint_trim_base_prof.rebind_pointers();
    }
    if (s.cached_sidecar.has_tone) {
      s.has_tone = true;
      s.tone_curve = s.cached_sidecar.tone_curve;
    }
  } else if (s.use_dino) {
    const EmbeddedFile* trt = find_embedded_backbone(true);
    const EmbeddedFile* onnx = find_embedded_backbone(false);
    const EmbeddedFile* backbone = trt ? trt : onnx;
    if (!backbone) {
      log_error("This neg_process was built without the DINOv3 backbone of the parameter search (make "
                      "dinov3_model, or make dinov3_engine for USE_CUDA=1, then rebuild); give --gain_g or --gain_b "
                      "to convert without it\n");
      return 1;
    }
    s.dino_path = std::string("[built-in] ") + backbone->name;
    log_info("Loading DINOv3 backbone %s (backend: %s)...\n", s.dino_path.c_str(), o.dino_backend.c_str());
    try {
      s.engine.load_memory(trt ? trt->data : nullptr, trt ? trt->size : 0, onnx ? onnx->data : nullptr,
                           onnx ? onnx->size : 0, o.dino_backend);
      // The file that runs: the .onnx when the engine was not loaded.
      if (trt && std::strcmp(s.engine.backend_used(), "gpu") != 0) {
        s.dino_path = std::string("[built-in] ") + onnx->name;
      }
      // A model's file named by --dino, read whole (an error names the head).
      auto read_file = [](const std::string& path, const char* head) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error(std::string("--dino ") + head + ": cannot read " + path);
        return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
      };
      if (!o.dino_adapter.empty()) {
        std::string adapter = o.dino_adapter;
        if (is_directory(adapter)) adapter = (std::filesystem::path(adapter) / "adapter.onnx").string();
        s.engine.load_adapter_file(adapter);
        s.head_path[0] = adapter;
        s.head_fp[0].ok = file_fingerprint(adapter, s.head_fp[0].size, s.head_fp[0].crc);
      } else {
        std::string adapter_dir;
        const std::string profile_name = !o.film_stock.empty() ? o.film_stock
                                         : s.is_bundle ? bundle.key : s.film_prof_arg;
        bool fell_back = false;
        const EmbeddedFile* a = find_embedded_adapter(profile_name, adapter_dir, &fell_back);
        if (!a) {
          log_error("No intent adapter for profile '%s'\n", profile_name.c_str());
          return 1;
        }
        if (fell_back) {
          log_warn("no intent adapter matches profile '%s'; using %s (--dino gives another)\n",
                   profile_name.c_str(), adapter_dir.c_str());
        }
        // USE_CUDA=1 builds carry the adapter's TensorRT engine beside it (make dinov3_engine).
        const EmbeddedFile* ae = find_embedded_file(adapter_dir + "/adapter.engine");
        s.engine.set_adapter(a->data, a->size, adapter_dir, 1, ae ? ae->data : nullptr, ae ? ae->size : 0);
        s.head_path[0] = std::string("[built-in] ") + a->name;
        s.head_fp[0] = memory_fingerprint(a->data, a->size);
      }
      // The neutral-patch head on the backbone's tokens, for the white-light curve fit's scene-neutral anchors
      // (input_curves.h, JointCurveModel::set_scene_neutral): only where the fit runs.
      if (s.fit_curves_active) {
        std::vector<char> file;
        const void* data = nullptr;
        size_t size = 0;
        std::string name = "[built-in] models/neutral_head/neutral_head.bin";
        if (o.neutral_head_file.empty()) {
          const EmbeddedFile* f = find_embedded_file("models/neutral_head/neutral_head.bin");
          if (f) {
            data = f->data;
            size = f->size;
          }
        } else {
          file = read_file(o.neutral_head_file, "neutral");
          data = file.data();
          size = file.size();
          name = o.neutral_head_file;
        }
        if (data && s.engine.needs_adapter()) {
          try {
            s.engine.set_neutral_head(data, size);
          } catch (const std::exception& e) {  // not a neutral head file: say which file
            if (o.neutral_head_file.empty()) throw;
            throw std::runtime_error("--dino neutral: " + o.neutral_head_file + ": " + e.what());
          }
          s.neutral_head = name;
          s.head_path[1] = name;
          s.head_fp[1] = memory_fingerprint(data, size);
          log_info("Neutral head %s\n", name.c_str());
        }
      }
      // The tone heads read the search's pooled features and the tokens' spread (only where the fit runs; the stage
      // itself needs -P srgb, which the fit checks).
      if (s.fit_curves_active && s.engine.needs_adapter()) {
        std::string err;
        if (o.tone_head_file.empty()) {
          const EmbeddedFile* f = find_embedded_file("models/tone_head/tone_head.bin");
          if (!f || !s.tone_head.load(f->data, f->size, err)) {
            throw std::runtime_error("the built-in tone heads (models/tone_head/tone_head.bin) " +
                                     (f ? err : std::string("are missing")));
          }
          s.head_path[2] = "[built-in] models/tone_head/tone_head.bin";
          s.head_fp[2] = memory_fingerprint(f->data, f->size);
        } else {
          const std::vector<char> file = read_file(o.tone_head_file, "tone");
          if (!s.tone_head.load(file.data(), file.size(), err))
            throw std::runtime_error("--dino tone: " + o.tone_head_file + ": " + err);
          s.head_path[2] = o.tone_head_file;
          s.head_fp[2] = memory_fingerprint(file.data(), file.size());
        }
        s.engine.set_token_spread(true);
        log_info("Tone heads %s\n", s.head_path[2].c_str());
      }
    } catch (const std::exception& e) {
      log_error("%s\n", e.what());
      return 1;
    }
    log_info("DINOv3 engine loaded [%s] adapter %s (memory: %.1f MB)\n", s.engine.backend_name(),
           s.engine.adapter_name().c_str(), (double)s.engine.memory_bytes() / (1024.0 * 1024.0));
  }
  return 0;
}

// The station's E of given values: -E, else the centre of bracket target_idx's exposure range as the bundle search
// meters it (Scan::bracket_exposure, meter_bracket_exposures).
static float given_station_e(const Options& o, const Scan& scan, int target_idx) {
  if (o.exposure_given || target_idx < 0 || target_idx >= (int)scan.bracket_exposure.size() ||
      !scan.bracket_exposure[target_idx].valid) {
    return o.exposure_comp;
  }
  return scan.bracket_exposure[target_idx].e_center;
}

int decode(const Options& o, Setup& s, Scan& scan) {
  ProfileBundle& bundle = s.bundle;
  std::copy(o.roi, o.roi + 4, scan.roi);
  scan.has_roi = o.has_roi;
  scan.rot_cw = o.rot_cw;
  scan.hflip = o.hflip;
  scan.vflip = o.vflip;

  scan.exposure_file = o.files[0];
  scan.has_exif_exp = read_exif_exposure_d(o.files[0], scan.exif_shutter_d, scan.exif_iso_d);

  // Each bracket's exposure range (e_min, e_max, e_center, is_feasible) on the frame, which the bracket picked without
  // the search, the single-bracket search's bounds and the exposure of given values read. Measured here, on the
  // half-size frame every bundle conversion analyses, before anything picks a bracket: before, solve measured it
  // after decode had already picked one (from the defaults, always the first admissible bracket) and measured the
  // roll-off and solved the knobs there. The bundle search measures its own; a previous run's sidecar names its
  // bracket.
  const bool needs_enclosures = s.is_bundle && !s.bundle_search && !s.has_cached_sidecar &&
                                (s.use_dino || s.manual_target_idx < 0 || !s.exposure_given);
  auto measure_enclosures = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso) {
    if (!needs_enclosures) return;
    float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
    int num_feasible = 0;
    compute_bundle_enclosures(bundle, frame, g_lo, g_hi, scene_ev, num_feasible);
    // And each bracket's range in the station's E, as the bundle search meters it: the E given values default to and
    // the single-bracket search centres on.
    meter_bracket_exposures(bundle, frame, scan_shutter_s, scan_iso, ParamSearchOptions{}.crop_ratio,
                            scan.film_base_rgb(), scan.bracket_exposure);
    scan.has_enclosures = true;
  };
  // The bracket of given values (the search's is its own), once the enclosures are measured.
  auto given_target_idx = [&]() {
    return !s.is_bundle ? -1
           : s.manual_target_idx >= 0 ? s.manual_target_idx
                                      : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
  };
  // The bracket's final E, g, b: Auto's (or the curve fit's) or the given values, moved by --brightness, --tint and
  // --warmth when given (grey-point knobs, solved on the bracket's profile, combined when curves apply); then the
  // highlight roll-off measured at them.
  auto solve_knobs_and_rolloff = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso) {
    const int target_idx = scan.has_auto_res ? scan.auto_res.target_idx : given_target_idx();
    const float base_e = scan.has_auto_res ? scan.params().e
                                           : given_station_e(o, scan, target_idx);
    const float base_g = scan.has_auto_res ? scan.params().g : o.gain_g;
    const float base_b = scan.has_auto_res ? scan.params().b : o.gain_b;
    float e = base_e, g = base_g, b = base_b;
    // A replayed tint trim: composed into the bracket's tables at the base E, g and b (a fit's own run has
    // it there already), before the knobs' grey and the roll-off are measured through them.
    const bool tint_trim_here = s.has_tint_trim && target_idx == s.tint_trim_target_idx;
    if (s.has_tint_trim && !tint_trim_here) {
      // Said, not silent: the trim was made on another bracket's tables.
      s.tint_trim_other_bracket = true;
      log_warn("the recorded tint trim was made on bracket %d; this conversion renders bracket %d, so it is "
               "not applied\n", s.tint_trim_target_idx, target_idx);
    }
    if (tint_trim_here && !s.tint_trim_composed) {
      compose_tint_trim_tables(bundle.targets[target_idx].film_prof, s.tint_trim, base_e, base_g, base_b);
      s.tint_trim_composed = true;
      log_info("Tint trim replayed on bracket %d at E %.4f g %.4f b %.4f (theta %.5f %.5f %.5f %.5f)\n",
               target_idx, base_e, base_g, base_b, s.tint_trim.theta[0], s.tint_trim.theta[1], s.tint_trim.theta[2],
               s.tint_trim.theta[3]);
    }
    if (o.knobs_given && target_idx >= 0 && target_idx < (int)bundle.targets.size()) {
      scan.knob_base[0] = base_e;
      scan.knob_base[1] = base_g;
      scan.knob_base[2] = base_b;
      if (!solve_grey_knobs(bundle.targets[target_idx].film_prof.data, base_e, base_g, base_b, o.brightness, o.tint,
                            o.warmth, scan.knob_res)) {
        log_error("Failed to solve grey-point knobs for target %s\n", bundle.targets[target_idx].name.c_str());
        return false;
      }
      scan.has_knob_res = true;
      e = scan.knob_res.e;
      g = scan.knob_res.g;
      b = scan.knob_res.b;
      // The knobs moved E, g and b off the anchor of the curves the bracket renders through (fitted, recorded or a
      // previous run's): re-anchor them there and compose the bracket's input tables again from its un-curved ones, so
      // the curves stay on the film densities they were fitted to.
      //
      // The knobs were solved on the profile at the base's anchor, and the re-anchored curves change it: the grey then
      // missed its target (on two white-LED frames up to 0.08 stop and 5 a* for a 0.5 stop brightness). So, until the
      // gains hold still (each within kKnobTol of the previous pass, at most kMaxKnobPasses): the knobs solved on the
      // re-anchored profile for the same grey (the one the base's profile renders as neutral L* 50), and the curves
      // re-anchored at that result. Each pass cuts the miss about fourfold.
      //
      // The tint trim is a density change on the film, so the knobs rebuild its tables at their E, g and b
      // the same way, in the same passes.
      const bool curves_here = target_idx == s.curves_base_idx && s.input_curves.active();
      if (curves_here || tint_trim_here) {
        constexpr int kMaxKnobPasses = 8;
        constexpr float kKnobTol = 1e-4f;
        ParsedFilmProfile& prof = bundle.targets[target_idx].film_prof;
        ParsedFilmProfile grey_prof = prof;  // at the base's anchor: its neutral L* 50 is the grey
        grey_prof.rebind_pointers();
        ParamSearchOptions ro;
        ro.film_base_rgb_d = scan.film_base_rgb();
        ro.film_base_from_scan = ro.film_base_rgb_d != nullptr;
        const float from[3] = {base_e, base_g, base_b};
        float scale[2] = {1.0f, 1.0f};
        auto reanchor_at = [&](const float to[3]) {
          if (curves_here) {
            if (!reanchor_input_curves(bundle, s.engine, target_idx, frame, scan_shutter_s, scan_iso, ro,
                                       s.input_curves, from, to, negicc::CurveFitOptions().max_dev_stops,
                                       scan.knob_curves, scale)) {
              return false;
            }
            apply_input_curves(prof, s.curves_base_prof, scan.knob_curves);
          } else {
            prof = s.tint_trim_base_prof;
            prof.rebind_pointers();
          }
          if (tint_trim_here) compose_tint_trim_tables(prof, s.tint_trim, to[0], to[1], to[2]);
          return true;
        };
        const float to1[3] = {e, g, b};
        if (!reanchor_at(to1)) return false;
        float lab1[3];
        knob_grey_lab(grey_prof.data, prof.data, from, to1, lab1);
        KnobSolveResult k2 = scan.knob_res;
        int passes = 0;
        for (bool moved = true; moved && passes < kMaxKnobPasses; ++passes) {
          const float prev[3] = {k2.e, k2.g, k2.b};
          if (!solve_grey_knobs(prof.data, base_e, base_g, base_b, o.brightness, o.tint, o.warmth, k2,
                                &grey_prof.data)) {
            log_error("Failed to solve grey-point knobs on the re-anchored curves\n");
            return false;
          }
          const float to2[3] = {k2.e, k2.g, k2.b};
          if (!reanchor_at(to2)) return false;
          moved = false;
          for (int i = 0; i < 3; ++i) moved = moved || std::fabs(to2[i] - prev[i]) > kKnobTol * prev[i];
        }
        const float to2[3] = {k2.e, k2.g, k2.b};
        knob_grey_lab(grey_prof.data, prof.data, from, to2, k2.grey_lab);
        scan.knob_res = k2;
        e = k2.e;
        g = k2.g;
        b = k2.b;
        scan.has_knob_curves = curves_here;
        log_info("Input tables re-anchored at the knobs' E %.4f g %.4f b %.4f (refG %.6g refB %.6g%s%s); the grey L* %.3f "
                 "a* %.3f b* %.3f, %+.4f stop from the target after %d more solves (L* %.3f a* %.3f b* %.3f, %+.4f "
                 "stop, before them)\n",
                 e, g, b, scan.knob_curves.ref[0], scan.knob_curves.ref[1],
                 scale[0] < 1.0f || scale[1] < 1.0f ? ", scaled to keep the shape" : "",
                 tint_trim_here ? ", tint trim rebuilt" : "", k2.grey_lab[0],
                 k2.grey_lab[1], k2.grey_lab[2], knob_grey_miss_stops(o.brightness, k2.grey_lab[0]), passes, lab1[0],
                 lab1[1], lab1[2], knob_grey_miss_stops(o.brightness, lab1[0]));
      }
    }
    if (!scan.has_knob_res && scan.has_curve_params && s.curve_fit.rolloff_measured) {
      // The curve fit measured the roll-off on this frame, at these E, g and b, on this combined profile, as below
      // (its last re-meter pass): the same numbers, not measured a third time.
      scan.rolloff_x_hi = s.curve_fit.x_hi;
      scan.rolloff_q995 = s.curve_fit.x_hi_q995;
    } else {
      scan.rolloff_x_hi = measure_rolloff_x_hi(bundle, target_idx, frame, scan_shutter_s, scan_iso, e, g, b,
                                               ParamSearchOptions{}.crop_ratio, &scan.rolloff_q995,
                                               scan.film_base_rgb());
    }
    scan.has_rolloff = true;
    return true;
  };
  // The previous run's Auto result (bracket and its final E, g, b; with fitted curves, the re-metered E and the fit's
  // g, b on the combined profile the curves were replayed into) in place of the search.
  auto restore_cached_auto = [&]() {
    scan.auto_res.target_idx = s.cached_sidecar.target_idx;
    scan.auto_res.target_name = s.cached_sidecar.target_name;
    scan.auto_res.params = {s.cached_sidecar.base_e, s.cached_sidecar.base_g, s.cached_sidecar.base_b, 1.0f};
    scan.has_auto_res = true;
  };
  // The single-bracket search (-E or --target without given gains, or a previous run's result reused with the knobs)
  // has its final E, g, b only in solve, after the half-size frame is gone: a copy of the frame (and the scan's
  // exposure) stays for its roll-off, which solve measures there as decode measures every other conversion's.
  auto keep_rolloff_frame = [&](const negicc::RgbView& frame, double scan_shutter_s, double scan_iso) {
    const int w = frame.width, h = frame.height;
    scan.rolloff_frame.resize((size_t)w * h * 3);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; ++y) {
      uint16_t* row = scan.rolloff_frame.data() + (size_t)y * w * 3;
      for (int x = 0; x < w; ++x) frame.sample(x, y, row + (size_t)x * 3);
    }
    scan.rolloff_frame_w = w;
    scan.rolloff_frame_h = h;
    scan.rolloff_shutter_s = scan_shutter_s;
    scan.rolloff_iso = scan_iso;
  };

  // The film base read from the scan for a bundle conversion (film_base_detect.h): from the whole half-size image
  // (flat field applied) and the frame detector's base patch on the mosaic, as negicc-station's Auto reads it. On the
  // zero-copy CFA path (Sony RAWs, pixel shift) that is the station's reading; the white-light decode runs the
  // detector on the mosaic before decoding it (find_frame), as the station's does; RGB files have no mosaic for the
  // detector, so the base comes from the statistics.
  // TODO: a film base from an image (a capture of unexposed film, as negicc-station's Film Base tab takes) in its place.
  auto detect_base = [&](const negicc::RgbView& half, const FrameDetection* det, int sensor_w, int sensor_h) {
    if (!s.is_bundle) return;
    const double* xt = (s.bundle.has_crosstalk && !s.white_light) ? s.bundle.crosstalk_matrix_d : nullptr;
    const bool found = negicc::detect_film_base(half, det, sensor_w, sensor_h, xt, scan.film_base);
    const negicc::FilmBaseDetection& fb = scan.film_base;
    if (found) {
      log_info("Film base (%s): R=%.1f G=%.1f B=%.1f\n", fb.source.c_str(), fb.rgb[0], fb.rgb[1], fb.rgb[2]);
    } else {
      log_info("Film base: none read from the scan; the bundle's\n");
    }
  };

  // The frame detection the film base reads with a given --roi on the white-light decode (find_frame, below).
  FrameDetection base_det;
  bool has_base_det = false;
  // The size of the mosaic the frame detector ran on (its boxes are in those pixels), for the base patch's scale.
  int det_sensor_w = 0, det_sensor_h = 0;
  // The analysis of the half-size image on the LibRaw, TIFF, JPEG and three-capture paths: for every bundle conversion,
  // since every one reads the film base from it and measures the enclosures on it (the CFA path does both inline).
  // --roi auto's frame: the detected crop, or the whole full_w x full_h image when the detection is not confident.
  auto take_detected_frame = [&](int full_w, int full_h) {
    if (scan.frame_det.conf < kLowFrameConfidence) {
      scan.roi[0] = 0;
      scan.roi[1] = 0;
      scan.roi[2] = full_w;
      scan.roi[3] = full_h;
    } else {
      for (int i = 0; i < 4; ++i) scan.roi[i] = scan.frame_det.crop[i];
    }
    scan.has_roi = true;
  };
  // The film base read from a half-size image (detect_base); gives the image as a view.
  auto read_base = [&](const LibRaw& raw) {
    const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
    negicc::RgbView half_view = negicc::RgbView::from_interleaved_rgb(
        &raw.imgdata.image[0][0], half_w, half_h, 4, (ptrdiff_t)half_w * 4);
    // The film base from the whole scan's statistics (there is no mosaic for the frame detector on this path).
    // The detector's boxes are in the mosaic's pixels: give its size, not the decode's (the half-size decode's
    // own size put the base patch off the image, so neg_process fell back to the statistics).
    // A single RAW's detection (--roi auto's, or base_det with a given frame) ran on its mosaic, white light or
    // not; before, only white light used it, so TriLED with -q >= 0 (this path, not the CFA one) lost the rebate
    // base and read it from the statistics.
    const bool raw_det = s.white_light || (o.input_mode == InputMode::kRaw && o.files.size() == 1);
    detect_base(half_view, (raw_det && o.roi_auto) ? &scan.frame_det : has_base_det ? &base_det : nullptr,
                raw_det ? det_sensor_w : 0, raw_det ? det_sensor_h : 0);
    return half_view;
  };
  // The white-light decode's balance of the camera channels on the film base: the base read from its first half-size
  // decode is put on the bundle's (film_base_balance_gains). The analysis then reads the base again, balanced.
  ChannelBalance balance;
  if (s.white_light && s.bundle.has_film_base) {
    balance = [&](const LibRaw& raw, float k[3]) {
      read_base(raw);
      if (scan.film_base.found)
        negicc::film_base_balance_gains(raw, scan.film_base.rgb, s.bundle.profile_film_base_rgb, k);
      std::copy(k, k + 3, scan.balance_gains);
      log_info("Film-base balance before the camera matrix: camera channel gains %.6f %.6f %.6f\n", k[0], k[1], k[2]);
    };
  }
  HalfSizeAnalysis analyse;
  if (s.is_bundle) {
    analyse = [&](const LibRaw& raw) -> bool {
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      scan.scan_exposure(raw, scan_shutter_s, scan_iso);
      // An RGB file or three captures with --roi auto: the detection ran on the whole image before this analysis,
      // which sees the whole image too, so the frame is placed from it here (the RAW path's find_frame does so before
      // the half-size decode). raw's rawdata sizes are the full image's (load_rgb_input).
      if (o.roi_auto && !scan.has_roi &&
          (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot)) {
        take_detected_frame(raw.imgdata.rawdata.sizes.width, raw.imgdata.rawdata.sizes.height);
      }
      // --roi halved, or the whole image.
      const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
      int* bbox = scan.auto_frame_half;
      bbox[0] = 0;
      bbox[1] = 0;
      bbox[2] = half_w;
      bbox[3] = half_h;
      if (scan.has_roi) {
        bbox[0] = scan.roi[0] / 2;
        bbox[1] = scan.roi[1] / 2;
        bbox[2] = std::max(1, (scan.roi[0] + scan.roi[2]) / 2 - bbox[0]);
        bbox[3] = std::max(1, (scan.roi[1] + scan.roi[3]) / 2 - bbox[1]);
        clamp_rect(bbox, half_w, half_h);
      }
      // Step 3 & Step 4: zero-copy RgbView over the half-size image, cropped and oriented in place.
      negicc::RgbView half_view = read_base(raw);
      negicc::RgbView frame =
          half_view.crop_and_orient(bbox[0], bbox[1], bbox[2], bbox[3], scan.rot_cw, scan.hflip, scan.vflip);
      if (s.bundle_search && s.has_cached_sidecar && s.cached_sidecar.is_auto) {
        restore_cached_auto();
      } else if (s.bundle_search) {
        // Auto's time (the sidecar's auto.time_ms): the search and the curve fit, as on the CFA path.
        const double t_auto_0 = omp_get_wtime();
        ParamSearchOptions opts;
        if (o.dino_iters_given) opts.max_steps = o.dino_iters;
        opts.film_base_rgb_d = scan.film_base_rgb();
        opts.film_base_from_scan = opts.film_base_rgb_d != nullptr;
        if (!search_conversion_params(bundle, s.engine, frame, scan_shutter_s, scan_iso, opts, scan.auto_res)) {
          log_error("The conversion parameter search failed\n");
          return false;
        }
        scan.has_auto_res = true;
        if (s.fit_curves_active) {
          // The curves on the bracket the search chose, then E re-metered on its combined profile to Auto's median
          // L* (the station's Fit curves toggle); the conversion takes those (Scan::params). Auto: the same on every
          // white-light bundle search.
          negicc::CurveFitOptions fit_opts;  // the defaults (input_curves.h); the sidecar records them
          // The fit and the re-meter render the grid as the export will: -P's encoding (srgb or srgb-g10, prepare).
          const int export_encoding = o.colorspace == "srgb" ? ENCODE_SRGB : ENCODE_SRGB_LINEAR;
          s.curves_base_idx = scan.auto_res.target_idx;  // the fit composes the curves into it in place
          s.curves_base_prof = bundle.targets[s.curves_base_idx].film_prof;
          s.curves_base_prof.rebind_pointers();
          // --roi auto that fell back to the whole image: the grid may hold rebate, bare light or holder.
          ParamSearchOptions fit_search_opts = opts;
          fit_search_opts.fallback_crop = o.roi_auto && scan.frame_det.conf < kLowFrameConfidence;
          fit_search_opts.film_base_is_rebate = scan.film_base.is_rebate();
          if (!run_white_light_fit(bundle, s.engine, frame, scan_shutter_s, scan_iso, fit_search_opts,
                                             scan.auto_res, fit_opts, export_encoding,
                                             s.tone_head.loaded() ? &s.tone_head : nullptr, s.curve_fit)) {
            log_error("Fitting the input curves failed\n");
            return false;
          }
          // The tone stage's plan from the fit's: the descriptors of the final render's preview, the curve between
          // them and the heads' predictions, and the chroma factor that keeps the mean chroma.
          s.tone_plan = TonePlanCurve{};
          if (ToneStagePlan& tp = s.curve_fit.tone; !tp.skipped && !tp.preview.empty()) {
            const size_t npx = tp.preview.size() / 3;
            s.tone_plan.render = tone_descriptors(tp.preview.data(), npx);
            s.tone_plan.curve = tone_plan(s.tone_plan.render, tp.pred_mean[ToneHead::kL2], tp.pred_mean[ToneHead::kL98],
                                          tp.pred_mean[ToneHead::kR9010], tp.preview.data(), npx, 1.0,
                                          &s.tone_plan.targets);
            s.tone_plan.chroma_after = tone_mean_chroma_after(tp.preview.data(), npx, s.tone_plan.curve);
            tp.planned = true;
            if (fit_search_opts.verbose) {
              const TonePlanCurve& t = s.tone_plan;
              log_info("Tone: L2 %.1f -> %.1f, L98 %.1f -> %.1f, L90-L10 %.1f -> %.1f; curve black %.2f white %.2f "
                       "gamma %.3f contrast %.3f chroma x%.4f\n",
                       (float)t.render.L2, (float)t.targets.L2, (float)t.render.L98, (float)t.targets.L98,
                       (float)t.render.range(), (float)t.targets.range, (float)t.curve.black, (float)t.curve.white,
                       (float)t.curve.gamma, (float)t.curve.contrast, (float)t.curve.chroma);
            }
          }
          if (!s.curve_fit.neutral_anchors) {
            log_info("Curves: %s; no scene-neutral anchors, the fit keeps Auto's colour\n",
                     s.neutral_head.empty() ? "no neutral head" : "no neutral evidence from the head");
          }
          s.input_curves = s.curve_fit.fit.curves;
          s.input_curves_source = o.fit_curves == Options::kFitCurvesOn ? "fit" : "auto";
          // The fit composed its tint trim into the bracket's tables already; the knobs rebuild them at
          // their own E, g and b (from the curves, else from the bracket's own tables).
          if (s.curve_fit.tint_trim.ran) {
            s.has_tint_trim = true;
            s.tint_trim = s.curve_fit.tint_trim.tables;
            s.tint_trim_target_idx = scan.auto_res.target_idx;
            s.tint_trim_composed = true;
            s.tint_trim_base_prof = s.curves_base_prof;
            s.tint_trim_base_prof.rebind_pointers();
          }
          if (s.curve_fit.tone.planned) {
            s.has_tone = true;
            s.tone_curve = s.tone_plan.curve;
          }
          scan.has_curve_params = true;
          scan.curve_params = s.curve_fit.params;
        }
        scan.t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
      }
      // The bracket analysis (timings_ms.analysis, not raw_decode): the enclosures, the knobs and the roll-off, or the
      // frame's copy for the single-bracket search.
      const double t_analysis_0 = omp_get_wtime();
      if (!s.bundle_search) measure_enclosures(frame, scan_shutter_s, scan_iso);
      // The single-bracket search (-E or --target without given gains) solves in solve, on the full image; its
      // roll-off is measured there, on a copy of this frame, at its final E, g, b.
      if (s.bundle_search || !s.use_dino) {
        if (!solve_knobs_and_rolloff(frame, scan_shutter_s, scan_iso)) return false;
      } else {
        keep_rolloff_frame(frame, scan_shutter_s, scan_iso);
      }
      scan.t_analysis_ms += (omp_get_wtime() - t_analysis_0) * 1000.0;
      return true;
    };
  }


  FrameDetectionParams det_params;
  det_params.format = o.film_format;
  det_params.axis_hint = o.roi_axis_hint;
  RawRegionFinder find_frame;
  // The film base's frame detection when the frame is given (--roi x y w h, or none): negicc-station's white-light
  // decode runs the detector on every scan's mosaic as unpacked, with the default parameters
  // (RawProcessor::load_raw_white_light), and reads the base from its rebate patch; neg_process does the same, with
  // --film_format when given (every format otherwise, as the station), so both read the same base. With --roi auto
  // the detection that found the frame serves (base_det: above).
  if (!o.roi_auto && s.is_bundle && o.files.size() == 1 && o.input_mode == InputMode::kRaw) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      FilmImage mosaic;
      if (film_image_from_raw(raw, mosaic)) {
        FrameDetectionParams base_params;
        if (o.film_format_given) base_params.format = o.film_format;  // else every format, as the station
        detect_film_frame(mosaic, base_params, base_det);
        has_base_det = true;
        det_sensor_w = mosaic.width;
        det_sensor_h = mosaic.height;
      }
      roi[0] = roi[1] = roi[2] = roi[3] = 0;   // only looked: the given roi stays
      return true;
    };
  }
  if (o.roi_auto && o.files.size() != 4) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        return false;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, scan.frame_det);
      scan.t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      det_sensor_w = mosaic.width;
      det_sensor_h = mosaic.height;
      take_detected_frame(mosaic.width, mosaic.height);
      for (int i = 0; i < 4; ++i) roi[i] = scan.roi[i];
      return true;
    };
  }

  // Logs --roi auto's detection, and warns when it fell back or the frame is vertical and not turned.
  auto print_frame_detection = [&]() {
    const FrameDetection& det = scan.frame_det;
    const char* axis_name = det.axis == 'h' ? "h" : (det.axis == 'v' ? "v" : "undecided");
    log_info("Film frame: axis=%s format=%s orientation=%s rotation=%.2f deg gaps=%d frames=%d conf=%.2f%s%s\n",
           axis_name, film_format_name(det.format), det.orientation.c_str(), det.rotation_deg,
           det.n_gaps, 1 + (int)det.secondaries.size(), det.conf,
           det.reason.empty() ? "" : " reason: ", det.reason.c_str());
    if (det.n_gaps > 0) {
      log_info("  Gaps (sensor px):");
      for (int g = 0; g < 2; ++g)
        if (det.gaps[g][0] >= 0) log_info(" [%d, %d]", det.gaps[g][0], det.gaps[g][1]);
      log_info("\n");
    }
    log_info("  Frame (sensor px): x=%d y=%d w=%d h=%d (frame centre %.1f, %.1f, size %.1f x %.1f, %.1f px/mm)\n",
           scan.roi[0], scan.roi[1], scan.roi[2], scan.roi[3], det.primary.rect.cx,
           det.primary.rect.cy, det.primary.rect.w, det.primary.rect.h, det.px_per_mm);
    for (const FilmFrame& f : det.secondaries)
      log_info("  Frame %+d (sensor px): visible x=%d y=%d w=%d h=%d (%.0f%% of a frame)\n", f.index, f.visible[0],
             f.visible[1], f.visible[2], f.visible[3], 100.0f * f.visible_fraction);
    if (det.conf < kLowFrameConfidence) {
      log_warn("low detection confidence (conf=%.2f%s%s); falling back to the full sensor frame\n",
               det.conf, det.reason.empty() ? "" : ", reason: ", det.reason.c_str());
    }
    if (s.use_dino && det.axis == 'v' && scan.rot_cw == 0) {
      log_warn("axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
    }
  };


  // Determine whether the 5-step zero-copy CFA pipeline applies (Step 1..5).
  bool use_cfa_pipeline = false;
  scan.t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  LibRaw *unpacked_probe = nullptr;
  float cfa_white_level = 65535.0f;
  int cfa_rx = 0, cfa_ry = 0, cfa_rw = 0, cfa_rh = 0;
  negicc::RgbView full_half_view;
  negicc::RgbView oriented_half_view;
  negicc::RgbView search_half_view;
  negicc::FlatSampleArgs flat_full{};

  if ((o.input_mode == InputMode::kRaw || o.input_mode == InputMode::kLinearDng) &&
      (o.files.size() == 4 || (s.raw_decode == RawDecode::kCameraRgb && o.quality < 0))) {
    LibRaw* probe = new LibRaw();
    log_info("Loading RAW file %s\n", o.files[0].c_str());
    int ret = open_and_unpack_raw_file(probe, o.files[0]);
    if (ret != LIBRAW_SUCCESS) {
      log_error("Cannot open/unpack %s: %s\n", o.files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    if (o.files.size() == 4) {
      if (!is_sony_a7rm4(probe)) {
        log_error("4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
                probe->imgdata.idata.make, probe->imgdata.idata.model);
        delete probe;
        return 1;
      }
      use_cfa_pipeline = true;
    } else if (is_sony_cfa_geometry(probe) && probe->imgdata.rawdata.raw_image) {
      use_cfa_pipeline = true;
    }
    if (use_cfa_pipeline) {
      proc = probe;
    } else {
      unpacked_probe = probe;
    }
  }


  if (use_cfa_pipeline) {
    if (o.roi_auto) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(*proc, mosaic)) {
        log_error("--roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        delete proc;
        return 1;
      }
      det_params.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, det_params, scan.frame_det);
      scan.t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
    }

    // Step 1: Per-channel black subtraction in-place on the unpacked CFA mosaic.
    if (!subtract_cfa_black_in_place(proc, cfa_white_level)) {
      log_error("Invalid CFA mosaic geometry in %s\n", o.files[0].c_str());
      delete proc;
      return 1;
    }
    const int src_w = proc->imgdata.sizes.width;
    const int src_h = proc->imgdata.sizes.height;
    const size_t cfa_pitch = proc->imgdata.sizes.raw_pitch / 2;
    const unsigned int filters = proc->imgdata.idata.filters;
    const uint16_t* cfa_origin = proc->imgdata.rawdata.raw_image +
                                 (size_t)proc->imgdata.sizes.top_margin * cfa_pitch +
                                 proc->imgdata.sizes.left_margin;

    // For 4x pixel shift, unpack and black-subtract frames 1..3 in Step 1 and keep all 4 CFA mosaics in memory.
    if (o.files.size() == 4) {
      for (int mi = 1; mi < 4; ++mi) {
        scan.ps_extra[mi - 1].reset(new LibRaw());
        LibRaw* fp = scan.ps_extra[mi - 1].get();
        log_info("Loading RAW file %s\n", o.files[mi].c_str());
        int fret = fp->open_file(o.files[mi].c_str());
        if (fret != LIBRAW_SUCCESS || (fret = fp->unpack()) != LIBRAW_SUCCESS) {
          log_error("Cannot open/unpack %s: %s\n", o.files[mi].c_str(), libraw_strerror(fret));
          delete proc;
          return 1;
        }
        const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
        if (!is_sony_a7rm4(fp) || fp->imgdata.sizes.width != src_w || fp->imgdata.sizes.height != src_h ||
            fpitch != cfa_pitch || fp->imgdata.idata.filters != filters ||
            fp->imgdata.sizes.top_margin != proc->imgdata.sizes.top_margin ||
            fp->imgdata.sizes.left_margin != proc->imgdata.sizes.left_margin ||
            std::fabs(fp->imgdata.other.iso_speed - proc->imgdata.other.iso_speed) > 1e-3f ||
            std::fabs(fp->imgdata.other.shutter - proc->imgdata.other.shutter) > 1e-6f) {
          log_error("4-shot pixel shift merge is only available with matching Sony A7RM4 frames\n");
          delete proc;
          return 1;
        }
        float fw_white = 0.0f;
        if (!subtract_cfa_black_in_place(fp, fw_white)) {
          log_error("Invalid CFA mosaic geometry in %s\n", o.files[mi].c_str());
          delete proc;
          return 1;
        }
      }
    }

    // Step 2: Downsizing via zero-copy RgbView over the CFA mosaic.
    const int half_w = src_w / 2;
    const int half_h = src_h / 2;
    std::shared_ptr<const negicc::FlatTables> flat_half_tables;
    negicc::FlatSampleArgs flat_half{};
    if (s.flat_field) {
      try {
        flat_half_tables = s.flat_field->tables(half_w, half_h, true);
        flat_half = flat_half_tables->host();
        scan.flat_full_tables = s.flat_field->tables(src_w, src_h, false);
        flat_full = scan.flat_full_tables->host();
      } catch (const std::exception& e) {
        log_error("%s\n", e.what());
        delete proc;
        return 1;
      }
    }
    full_half_view = negicc::RgbView::from_cfa_half(
        cfa_origin, cfa_pitch, src_w, src_h, filters, nullptr, flat_half, flat_half_tables);
    if (s.is_bundle) {
      // The frame detector on the black-subtracted mosaic of the first frame, with its default parameters, as
      // negicc-station runs it for the film base (CapturedImage::film_frame_detection).
      FrameDetection base_det;
      FilmImage mosaic;
      const float no_black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const bool has_det =
          filters > 1000 &&
          film_image_from_cfa(cfa_origin, src_w, src_h, (ptrdiff_t)cfa_pitch, 2,
                              [filters](int row, int col) { return negicc::bayer_color_at(row, col, filters); },
                              no_black, cfa_white_level, mosaic);
      FrameDetectionParams base_params;
      if (o.film_format_given) base_params.format = o.film_format;  // else every format, as the station
      if (has_det) detect_film_frame(mosaic, base_params, base_det);
      detect_base(full_half_view, has_det ? &base_det : nullptr, src_w, src_h);
    }

    // Step 3: Crop, rotate, flip & inline ROI search on the zero-copy view.
    if (o.roi_auto) {
      take_detected_frame(src_w, src_h);
      print_frame_detection();
    }

    cfa_rx = 0;
    cfa_ry = 0;
    cfa_rw = src_w;
    cfa_rh = src_h;
    if (scan.has_roi) {
      if (scan.roi[0] < 0 || scan.roi[1] < 0 || scan.roi[0] >= src_w || scan.roi[1] >= src_h ||
          scan.roi[2] <= 0 || scan.roi[3] <= 0) {
        log_error("ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
                scan.roi[0], scan.roi[1], scan.roi[2], scan.roi[3], src_w, src_h);
        delete proc;
        return 1;
      }
      cfa_rx = scan.roi[0];
      cfa_ry = scan.roi[1];
      cfa_rw = std::min(scan.roi[2], src_w - cfa_rx);
      cfa_rh = std::min(scan.roi[3], src_h - cfa_ry);
    } else if (!o.no_crop && o.files.size() == 1) {
      const auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
      const bool has_inset_crop = crop_ref.cwidth > 0 && crop_ref.cheight > 0 &&
                                  crop_ref.cleft != 0xffff && crop_ref.ctop != 0xffff;
      if (has_inset_crop && (crop_ref.cleft || crop_ref.ctop)) {
        cfa_rx = std::max(0, std::min((int)crop_ref.cleft, src_w - 1));
        cfa_ry = std::max(0, std::min((int)crop_ref.ctop, src_h - 1));
        cfa_rw = std::max(1, std::min((int)crop_ref.cwidth, src_w - cfa_rx));
        cfa_rh = std::max(1, std::min((int)crop_ref.cheight, src_h - cfa_ry));
      }
    }

    int out_half[4] = {
        cfa_rx / 2,
        cfa_ry / 2,
        std::max(1, (cfa_rx + cfa_rw) / 2 - cfa_rx / 2),
        std::max(1, (cfa_ry + cfa_rh) / 2 - cfa_ry / 2),
    };
    clamp_rect(out_half, half_w, half_h);
    for (int i = 0; i < 4; ++i) scan.auto_frame_half[i] = out_half[i];
    search_half_view = full_half_view.crop_and_orient(
        scan.auto_frame_half[0], scan.auto_frame_half[1], scan.auto_frame_half[2], scan.auto_frame_half[3],
        scan.rot_cw, scan.hflip, scan.vflip);
    oriented_half_view = search_half_view;

    // Step 4: Conversion parameter search inline on the zero-copy oriented RgbView.
    if (s.bundle_search && s.has_cached_sidecar && s.cached_sidecar.is_auto) {
      restore_cached_auto();
    } else if (s.bundle_search) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      ParamSearchOptions opts;
      if (o.dino_iters_given) opts.max_steps = o.dino_iters;
      opts.film_base_rgb_d = scan.film_base_rgb();
      opts.film_base_from_scan = opts.film_base_rgb_d != nullptr;
      if (!search_conversion_params(bundle, s.engine, search_half_view, scan_shutter_s, scan_iso, opts,
                                    scan.auto_res)) {
        log_error("The conversion parameter search failed\n");
        delete proc;
        return 1;
      }
      scan.has_auto_res = true;
      scan.t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    }
    if (s.is_bundle) {
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
      const double t_analysis_0 = omp_get_wtime();  // timings_ms.analysis, as on the LibRaw path
      if (!s.bundle_search) measure_enclosures(search_half_view, scan_shutter_s, scan_iso);
      if (s.bundle_search || !s.use_dino) {
        if (!solve_knobs_and_rolloff(search_half_view, scan_shutter_s, scan_iso)) {
          delete proc;
          return 1;
        }
      } else {
        // The single-bracket search: its roll-off is measured in solve at its final E, g, b, on a copy of the frame
        // (the mosaic this view reads is freed by render).
        keep_rolloff_frame(search_half_view, scan_shutter_s, scan_iso);
      }
      scan.t_analysis_ms += (omp_get_wtime() - t_analysis_0) * 1000.0;
    }
    scan.t_raw_decode_ms = (omp_get_wtime() - scan.t_raw_0) * 1000.0 - scan.t_auto_ms - scan.t_frame_det_ms -
                           scan.t_analysis_ms;

    // The mosaic and the frame, for render.
    scan.mosaic.frames[0] = cfa_origin;
    scan.mosaic.n_frames = o.files.size() == 4 ? 4 : 1;
    for (int i = 0; i + 1 < scan.mosaic.n_frames; ++i) {
      LibRaw* fp = scan.ps_extra[i].get();
      const size_t fpitch = fp->imgdata.sizes.raw_pitch / 2;
      scan.mosaic.frames[i + 1] = fp->imgdata.rawdata.raw_image + (size_t)fp->imgdata.sizes.top_margin * fpitch +
                                  fp->imgdata.sizes.left_margin;
    }
    scan.mosaic.pitch = cfa_pitch;
    scan.mosaic.width = src_w;
    scan.mosaic.height = src_h;
    scan.mosaic.filters = filters;
    scan.mosaic.white_level = cfa_white_level;
    scan.mosaic.flat = flat_full;
    scan.cfa_region = {cfa_rx, cfa_ry, cfa_rw, cfa_rh, scan.rot_cw, scan.hflip, scan.vflip};
  } else {
    if (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot) {
      proc = load_rgb_input(o.files, o.input_mode, s.camera_rgb_input, o.quality,
                            !o.no_crop && !scan.has_roi && !o.roi_auto,
                            (!o.roi_auto && scan.has_roi) ? scan.roi : nullptr, analyse,
                            o.roi_auto ? &scan.frame_det : nullptr, det_params, scan.input_source);
    } else {
      proc = load_raw(o.files[0], true,
                      o.quality,
                      !o.no_crop && !scan.has_roi && !o.roi_auto,
                      s.raw_decode,
                      (!o.roi_auto && scan.has_roi) ? scan.roi : nullptr,
                      analyse, balance, find_frame, s.flat_field.get(),
                      unpacked_probe);
    }
    unpacked_probe = nullptr;
    if (!proc) {
      log_error("Cannot open %s\n", o.files[0].c_str());
      return 1;
    }
    if (o.roi_auto) {
      if (o.input_mode == InputMode::kRgbFile || o.input_mode == InputMode::kThreeShot) {
        const auto& rs = proc->imgdata.rawdata.sizes;
        const auto& is = proc->imgdata.sizes;
        take_detected_frame(rs.width ? rs.width : is.width, rs.height ? rs.height : is.height);
      }
      print_frame_detection();
    }
    if (scan.rot_cw != 0 || scan.hflip || scan.vflip) {
      if (!extract_roi_and_transform(proc, 0, 0, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                     scan.rot_cw, scan.hflip, scan.vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      log_info("Transformed subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, scan.rot_cw, scan.hflip ? 1 : 0,
             scan.vflip ? 1 : 0);
    }
    scan.t_raw_decode_ms = (omp_get_wtime() - scan.t_raw_0) * 1000.0 - scan.t_auto_ms - scan.t_frame_det_ms -
                           scan.t_analysis_ms;
  }

  const int orig_w = proc->imgdata.rawdata.sizes.width ? proc->imgdata.rawdata.sizes.width : proc->imgdata.sizes.width;
  const int orig_h = proc->imgdata.rawdata.sizes.height ? proc->imgdata.rawdata.sizes.height : proc->imgdata.sizes.height;

  scan.proc.reset(proc);
  scan.cfa = use_cfa_pipeline;
  scan.orig_w = orig_w;
  scan.orig_h = orig_h;
  log_info("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  log_info("Shutter Speed: %f\n", proc->imgdata.other.shutter);
  return 0;
}

int begin_conversion(const Options& o, const Setup& s, Conversion& c) {
  const std::vector<int> unit_fb = {1, 1, 1};
  c.exposure_comp = o.exposure_comp;
  c.gain_g = o.gain_g;
  c.gain_b = o.gain_b;
  c.knee = kKnee;

  // The sidecar records the bundle's film base. The film base scale of the input matrix is 1: a bundle conversion's
  // film base (the scan's, else the bundle's) is in the station's merged matrix (solve).
  c.sidecar_profile_film_base_rgb = {1, 1, 1};
  if (s.is_bundle && s.bundle.has_film_base) {
    c.sidecar_profile_film_base_rgb = {s.bundle.profile_film_base_rgb[0], s.bundle.profile_film_base_rgb[1],
                                       s.bundle.profile_film_base_rgb[2]};
  }
  c.profile_film_base_rgb = {1, 1, 1};

  if (s.is_bundle) {
    // The bundle's film base (when none is read from the scan) is in the station's merged matrix, so in E, g and b,
    // not applied as a film base scale.
    c.sidecar_film_base_rgb = c.sidecar_profile_film_base_rgb;
    c.film_base_source = "bundle";
  }

  if (o.exposure_given) c.pin_mask |= PIN_E;
  if (o.gain_g_given) c.pin_mask |= PIN_G;
  if (o.gain_b_given) c.pin_mask |= PIN_B;

  adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff,
                           c.exposure_comp,
                           c.gain_g, c.gain_b,
                           c.profile_film_base_rgb,
                           unit_fb,
                           c.merged_matrix);
  return 0;
}

int render(const Options& o, const Setup& s, Scan& scan, Backend& backend) {
  (void)o;
  (void)s;
  if (!scan.cfa) return 0;  // LibRaw's decode, a TIFF, a JPEG or three captures: the image is in scan.proc already
  const double t_step5_debayer_0 = omp_get_wtime();
  LibRaw* proc = scan.proc.get();
  const Region& r = scan.cfa_region;
  int norm_rot = r.rot_cw;
  bool norm_hf = r.hflip, norm_vf = r.vflip;
  normalize_rotation(norm_rot, norm_hf, norm_vf);
  const bool swap_dims = (norm_rot == 90 || norm_rot == 270);
  const int dst_w = swap_dims ? r.h : r.w;
  const int dst_h = swap_dims ? r.w : r.h;
  scan.rgb3.resize((size_t)dst_w * dst_h * 3);
  log_info(scan.mosaic.n_frames == 1 ? "Debayer quality: grain-aware\n" : "Merging 4 images...\n");
  backend.render(scan.mosaic, r, dst_w, dst_h, scan.rgb3.data());

  // The mosaics are not needed any more: LibRaw's raw buffer is freed (proc keeps its metadata), and a pixel shift's
  // other frames go.
  if (proc->imgdata.rawdata.raw_alloc && !proc->imgdata.image) {
    proc->imgdata.image = (ushort (*)[4])proc->imgdata.rawdata.raw_alloc;
    proc->imgdata.rawdata.raw_alloc = nullptr;
    proc->imgdata.rawdata.raw_image = nullptr;
    proc->free_image();
  }
  proc->recycle_datastream();
  for (auto& p : scan.ps_extra) p.reset();
  scan.mosaic = MosaicInput();

  proc->imgdata.sizes.iwidth = dst_w;
  proc->imgdata.sizes.iheight = dst_h;
  proc->imgdata.idata.colors = 3;
  proc->imgdata.params.output_bps = 16;
  log_info("Processed image size: %dx%d\n", dst_w, dst_h);
  scan.t_raw_decode_ms += (omp_get_wtime() - t_step5_debayer_0) * 1000.0;
  return 0;
}

int solve(const Options& o, Setup& s, Scan& scan, Conversion& c) {
  ProfileBundle& bundle = s.bundle;
  LibRaw* proc = scan.proc.get();
  // With a bundle, E, g and b are the station's (the search's, and what -E, --gain_g and --gain_b take): the
  // conversion is the station's, the bundle's merged matrix (crosstalk, film base and the scan's exposure ratio)
  // scaled per row by E, E*g and E*b, so neg_process renders what negicc-station renders. row_scale turns them into
  // the absolute exposure and gains on the crosstalk (the sidecar's matrix.*) and back.
  float station_base[9];
  double row_scale[3] = {1.0, 1.0, 1.0};
  if (s.is_bundle) {
    double scan_shutter_s = 0.125, scan_iso = 100.0;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, scan.film_base_rgb(), scan.film_base_rgb() != nullptr,
                                  station_base, nullptr, row_scale);
  }
  auto apply_station = [&](float e, float g, float b) {
    c.station_params = std::array<float, 3>{e, g, b};
    negicc::candidate_matrix(station_base, e, g, b, c.merged_matrix);
    c.exposure_comp = (float)(row_scale[0] * (double)e);
    c.gain_g = (float)(row_scale[1] / row_scale[0] * (double)g);
    c.gain_b = (float)(row_scale[2] / row_scale[0] * (double)b);
  };
  const std::vector<int> unit_fb = {1, 1, 1};

  if (s.is_bundle && !s.bundle_search) {
    // Each bracket's exposure range, which the target selection, set_target_bounds and given_station_e read below,
    // was measured in decode on the half-size frame (measure_enclosures), before decode picked a bracket.
    if (!s.has_cached_sidecar && (s.use_dino || s.manual_target_idx < 0 || !(c.pin_mask & PIN_E)) &&
        !scan.has_enclosures) {
      log_error("Internal error: the brackets' exposure ranges were not measured on the scan\n");
      return 1;
    }
  } else if (!s.is_bundle) {
    if (!s.film_prof_arg.empty() && !load_film_profile(s.film_prof_arg, c.film_prof)) {
      log_error("Cannot load film profile %s\n", s.film_prof_arg.c_str());
    }
  }

  auto log_knobs = [&]() {
    if (!scan.has_knob_res) return;
    log_info("Knobs (brightness %+.2f, tint %+.2f, warmth %+.2f%s): grey L*=%.2f a*=%.2f b*=%.2f -> E %.4f g %.4f b %.4f\n",
             o.brightness, o.tint, o.warmth, scan.knob_res.limited ? ", limited" : "",
             scan.knob_res.grey_lab[0], scan.knob_res.grey_lab[1], scan.knob_res.grey_lab[2],
             scan.knob_res.e, scan.knob_res.g, scan.knob_res.b);
  };

  if (scan.has_auto_res) {
    const ParamSearchResult& res = scan.auto_res;
    c.bundle_target_idx = res.target_idx;
    c.bundle_target_name = res.target_name;
    c.film_prof = bundle.targets[c.bundle_target_idx].film_prof;
    c.film_prof.rebind_pointers();
    // The search's E, g and b (after the curve fit's re-meter where the fit runs), moved by the knobs when
    // given, converted as the station converts them (the sidecar's cli_equivalent reproduces it).
    const ConversionParams& p = scan.params();
    const float fe = scan.has_knob_res ? scan.knob_res.e : p.e;
    const float fg = scan.has_knob_res ? scan.knob_res.g : p.g;
    const float fb = scan.has_knob_res ? scan.knob_res.b : p.b;
    apply_station(fe, fg, fb);
    if (!(s.has_cached_sidecar && s.cached_sidecar.is_auto)) {
      log_info("Auto: %s E %.4f g %.4f b %.4f (resid %.4f, %s), %d inferences\n", res.target_name.c_str(),
             res.params.e, res.params.g, res.params.b, res.residual, to_string(res.status),
             res.inferences);
      auto ms = [](std::chrono::steady_clock::duration d) {
        return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
      };
      log_info("Timings (ms): render %lld, infer %lld, residual %lld, search %lld, total %lld\n",
             ms(res.render_time), ms(res.infer_time), ms(res.residual_time), ms(res.search_time),
             ms(res.total_time));
    }
    if (scan.has_curve_params && s.curve_fit.fallback) {
      log_info("Curve fit fell back (%s): no curves; E %.4f g %.4f b %.4f on the bracket's profile\n",
               s.curve_fit.fallback, p.e, p.g, p.b);
    } else if (scan.has_curve_params) {
      log_info("Fitted curves: E %.4f g %.4f b %.4f on the combined profile\n", p.e, p.g, p.b);
    }
    log_knobs();
    log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n\n", fe, fg, fb, c.bundle_target_idx);
  } else if (s.use_dino) {
    if (!s.is_bundle && !c.film_prof.data.has_profile) {
      log_error("The parameter search requires a film profile to evaluate intent losses.\n");
      return 1;
    }

    SolverConfig cfg;
    cfg.knee = c.knee;
    cfg.shoulder_auto = s.is_bundle;  // the roll-off, as the bracket search renders it

    const int max_iters = o.dino_iters;
    IntentGains user_gains{c.exposure_comp, c.gain_g, c.gain_b};
    if (!s.has_cached_sidecar && s.is_bundle) {
      // With a bundle the search runs on the station's merged matrix (the crosstalk with each row normalised by the
      // film base, station_base), in the station's E, g and b, as the bundle search and given values do: it starts from
      // -E (or 1) and g = b = 1, the film base rendered neutral.
      const float e0 = (c.pin_mask & PIN_E) ? o.exposure_comp : 1.0f;
      negicc::candidate_matrix(station_base, e0, 1.0f, 1.0f, c.merged_matrix);
      user_gains = {e0, 1.0f, 1.0f};
    }

    if (s.is_bundle) {
      // -E without --target: the bracket the bundle picks without the search.
      c.bundle_target_idx = s.has_cached_sidecar && s.cached_sidecar.target_idx >= 0
                                ? s.cached_sidecar.target_idx
                                : (s.manual_target_idx >= 0 ? s.manual_target_idx
                                                            : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx));
      log_info("Selected target: %s\n", bundle.targets[c.bundle_target_idx].display_name.c_str());
      const auto& tgt = bundle.targets[c.bundle_target_idx];
      c.bundle_target_name = tgt.name;
      c.film_prof = tgt.film_prof;
      c.film_prof.rebind_pointers();
      set_target_bounds(cfg, tgt);
      // The search's E is the station's: it centres on the E the bundle search meters for this bracket (Step 2, the E
      // its bootstrap starts the bracket from) and searches the bracket's range as the bundle search does, so --target
      // without -E starts where Auto would on that bracket. The bounds are ratios to the centre.
      const int ti = c.bundle_target_idx;
      if (ti < (int)scan.bracket_exposure.size() && scan.bracket_exposure[ti].valid) {
        const BracketExposure& be = scan.bracket_exposure[ti];
        cfg.has_custom_bounds = true;
        cfg.custom_e_center = be.e_metered;
        cfg.custom_e_bounds[0] = be.search_lo / be.e_metered;
        cfg.custom_e_bounds[1] = be.search_hi / be.e_metered;
        cfg.custom_e_bounds[2] = 1.0f;
      } else if (cfg.has_custom_bounds) {
        cfg.custom_e_center = (float)((double)cfg.custom_e_center / row_scale[0]);
      }
    } else {
      cfg.prof = &c.film_prof.data;
    }

    // The base E, g, b: a previous run's (<output>.json, with the knobs) or the solver's; with a bundle the station's
    // values, else the absolute ones.
    float base_e = 1.0f, base_g = 1.0f, base_b = 1.0f;
    if (s.has_cached_sidecar && !s.cached_sidecar.is_auto) {
      base_e = s.cached_sidecar.base_e;
      base_g = s.cached_sidecar.base_g;
      base_b = s.cached_sidecar.base_b;
    } else {
      // TODO: Run `solve_dinov3_intent` on `oriented_half_view` (zero-copy half-size CFA view) instead of the
      // full-resolution `cfa_export_rgb` buffer once single-target solver baselines are updated.
      const bool solve_ok = scan.cfa
          ? solve_dinov3_intent(s.engine, cfg, c.merged_matrix,
                                negicc::RgbView::from_interleaved_rgb(
                                    scan.rgb3.data(), proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight),
                                user_gains, c.pin_mask, max_iters, c.dino_res)
          : solve_dinov3_intent(s.engine, cfg, c.merged_matrix, (const uint16_t (*)[4])proc->imgdata.image,
                                proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                user_gains, c.pin_mask, max_iters, c.dino_res);
      if (!solve_ok) {
        log_error("DINOv3 intent solver failed.\n");
        return 1;
      }


      std::string pins_str = "none";
      if (c.pin_mask != 0) {
        pins_str = "";
        if (c.pin_mask & PIN_E) pins_str += "E ";
        if (c.pin_mask & PIN_G) pins_str += "g ";
        if (c.pin_mask & PIN_B) pins_str += "b";
        if (!pins_str.empty() && pins_str.back() == ' ') pins_str.pop_back();
      }
      log_info("\nDINOv3 solver: grid %dx%d (%dx%d), backend %s, E_c %.3f, bounds [%.3f, %.3f], max iters %d, pins %s\n",
             c.dino_res.grid_w, c.dino_res.grid_h, c.dino_res.target_w, c.dino_res.target_h,
             s.engine.backend_name(), c.dino_res.e_center, c.dino_res.bounds_lo, c.dino_res.bounds_hi,
             max_iters, pins_str.c_str());
      if (c.dino_res.anchor_clamped) {
        log_warn("ANCHOR_CLAMPED (median L* 50 not reached within E_c 2^[-4, 4])\n");
      }
      log_info(" it      E      g      b   resid    L_L    L_a    L_b  clip%%  P_clip  P_anch       J\n");
      for (const auto& it : c.dino_res.iterations) {
        if (it.residual < 0.0f) {
          log_info("  %c  %5.3f  %5.3f  %5.3f       -  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
                 it.tag, it.e, it.g, it.b, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
                 it.penalty_clip, it.penalty_anchor, it.objective);
        } else {
          log_info("  %c  %5.3f  %5.3f  %5.3f  %6.3f  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
                 it.tag, it.e, it.g, it.b, it.residual, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
                 it.penalty_clip, it.penalty_anchor, it.objective);
        }
      }
      log_info("Status: %s after %d inferences (best it %d, resid %.3f)\n",
             c.dino_res.status.c_str(), c.dino_res.total_inferences, c.dino_res.best_iteration, c.dino_res.best_residual);
      log_info("Solved (relative): E %.3f g %.3f b %.3f\n",
             c.dino_res.solved_relative.e, c.dino_res.solved_relative.g, c.dino_res.solved_relative.b);
      if (s.is_bundle) {
        // The station's values, which -E, --gain_g and --gain_b take with a bundle: the search's own, since it ran on
        // the station's matrix.
        c.station_params = std::array<float, 3>{c.dino_res.solved_absolute.e, c.dino_res.solved_absolute.g,
                                                c.dino_res.solved_absolute.b};
        log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n", (*c.station_params)[0],
                 (*c.station_params)[1], (*c.station_params)[2], c.bundle_target_idx);
      } else {
        log_info("Solved: -E %.9g --gain_g %.9g --gain_b %.9g\n", c.dino_res.solved_absolute.e,
                 c.dino_res.solved_absolute.g, c.dino_res.solved_absolute.b);
      }
      if (c.dino_res.solved_relative.g < 0.5f || c.dino_res.solved_relative.g > 2.0f ||
          c.dino_res.solved_relative.b < 0.5f || c.dino_res.solved_relative.b > 2.0f) {
        log_warn("GAIN_RANGE (relative g or b outside [0.5, 2.0])\n");
      }
      if (c.dino_res.best_iteration >= 0 && (size_t)c.dino_res.best_iteration + 1 < c.dino_res.iterations.size()) {
        const auto& bit = c.dino_res.iterations[c.dino_res.best_iteration + 1];
        float log2_ratio = (bit.e > 0.0f) ? std::log2(bit.e) : 0.0f;
        log_info("Penalties: clip %.3f (%.2f%% %s 2.50%%), anchor %.3f (log2(E/E_c) %+.3f)\n",
               bit.penalty_clip, bit.clip_pct, bit.clip_pct <= 2.5f ? "<=" : ">", bit.penalty_anchor, log2_ratio);
      }
      log_info("Timings (ms): views %.1f, render %.1f, infer %.1f, residual %.1f, search %.1f\n\n",
             c.dino_res.timing_views_ms, c.dino_res.timing_render_ms, c.dino_res.timing_infer_ms,
             c.dino_res.timing_residual_ms, c.dino_res.timing_search_ms);
      if (s.is_bundle) {
        base_e = (*c.station_params)[0];
        base_g = (*c.station_params)[1];
        base_b = (*c.station_params)[2];
      } else {
        base_e = c.dino_res.solved_absolute.e;
        base_g = c.dino_res.solved_absolute.g;
        base_b = c.dino_res.solved_absolute.b;
      }
    }
    float final_e = base_e, final_g = base_g, final_b = base_b;
    const double t_analysis_0 = omp_get_wtime();  // the knobs and the roll-off: timings_ms.analysis
    if (o.knobs_given) {
      scan.knob_base[0] = base_e;
      scan.knob_base[1] = base_g;
      scan.knob_base[2] = base_b;
      if (!solve_grey_knobs(c.film_prof.data, base_e, base_g, base_b, o.brightness, o.tint, o.warmth, scan.knob_res)) {
        log_error("Failed to solve grey-point knobs\n");
        return 1;
      }
      scan.has_knob_res = true;
      final_e = scan.knob_res.e;
      final_g = scan.knob_res.g;
      final_b = scan.knob_res.b;
      log_knobs();
    }
    if (s.is_bundle) {
      // Rendered as the station renders those values, so the sidecar's cli_equivalent reproduces it.
      apply_station(final_e, final_g, final_b);
      // The roll-off at those values (after the knobs), on the half-size frame decode kept, as every other bundle
      // conversion measures it: the solver's own x_hi is at its pre-knob values, and a previous run's result reused
      // with the knobs has none.
      if (scan.rolloff_frame.empty()) {
        log_error("Internal error: no half-size frame kept for the roll-off\n");
        return 1;
      }
      const negicc::RgbView rolloff_view = negicc::RgbView::from_interleaved_rgb(
          scan.rolloff_frame.data(), scan.rolloff_frame_w, scan.rolloff_frame_h);
      scan.rolloff_x_hi = measure_rolloff_x_hi(bundle, c.bundle_target_idx, rolloff_view, scan.rolloff_shutter_s,
                                               scan.rolloff_iso, final_e, final_g, final_b,
                                               ParamSearchOptions{}.crop_ratio, &scan.rolloff_q995,
                                               scan.film_base_rgb());
      scan.has_rolloff = true;
      std::vector<uint16_t>().swap(scan.rolloff_frame);
      scan.t_analysis_ms += (omp_get_wtime() - t_analysis_0) * 1000.0;
    } else {
      c.exposure_comp = final_e;
      c.gain_g = final_g;
      c.gain_b = final_b;
      adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff, c.exposure_comp, c.gain_g, c.gain_b,
                               c.profile_film_base_rgb, unit_fb, c.merged_matrix);
    }
  } else if (s.is_bundle) {
    if (s.manual_target_idx >= 0) {
      c.bundle_target_idx = s.manual_target_idx;
    } else {
      c.bundle_target_idx = select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
    }
    log_info("Selected target: %s\n", bundle.targets[c.bundle_target_idx].display_name.c_str());
    c.bundle_target_name = bundle.targets[c.bundle_target_idx].name;
    c.film_prof = bundle.targets[c.bundle_target_idx].film_prof;
    c.film_prof.rebind_pointers();
    // The given values are the station's (E defaults to the bracket's e_center, g and b to 1).
    double scan_shutter_s = 0.125, scan_iso = 100.0;
    scan.scan_exposure(*proc, scan_shutter_s, scan_iso);
    if (scan.has_knob_res) {
      apply_station(scan.knob_res.e, scan.knob_res.g, scan.knob_res.b);
      log_knobs();
    } else {
      apply_station(given_station_e(o, scan, c.bundle_target_idx),
                    c.gain_g, c.gain_b);
    }
  } else if (o.knobs_given) {
    // A single ICC profile with given gains: the knobs move the given (absolute) E, g and b through that profile, as
    // they move the solver's result above.
    if (!c.film_prof.data.has_profile) {
      log_error("--brightness, --tint and --warmth need the film profile to solve the grey point\n");
      return 1;
    }
    scan.knob_base[0] = c.exposure_comp;
    scan.knob_base[1] = c.gain_g;
    scan.knob_base[2] = c.gain_b;
    if (!solve_grey_knobs(c.film_prof.data, c.exposure_comp, c.gain_g, c.gain_b, o.brightness, o.tint, o.warmth,
                          scan.knob_res)) {
      log_error("Failed to solve grey-point knobs\n");
      return 1;
    }
    scan.has_knob_res = true;
    c.exposure_comp = scan.knob_res.e;
    c.gain_g = scan.knob_res.g;
    c.gain_b = scan.knob_res.b;
    adjust_correction_matrix(s.r_coeff, s.g_coeff, s.b_coeff, c.exposure_comp, c.gain_g, c.gain_b,
                             c.profile_film_base_rgb, unit_fb, c.merged_matrix);
    log_knobs();
  }

  if (s.is_bundle && c.bundle_target_idx >= 0) {
    c.embedded_icc = bundle.targets[c.bundle_target_idx].icc_bytes;
  }
  if (s.is_bundle && c.bundle_target_idx >= 0 && c.station_params) {
    // The highlight roll-off of the conversion, as the bracket search measures it at its final parameters (on the
    // half-size frame), so given values (cli_equivalent) reproduce it: Auto's result and given values were measured
    // in decode, the single-bracket search above at its final E, g, b.
    if (!scan.has_rolloff) {
      log_error("Internal error: the roll-off was not measured\n");
      return 1;
    }
    c.shoulder.x_hi = scan.rolloff_x_hi;
    const float q995 = scan.rolloff_q995;
    c.shoulder.has_q995 = true;
    c.shoulder.q995 = q995;
    if (c.shoulder.active()) {
      c.shoulder.source = "rolloff";
      negicc::PipelineConfig pc{};
      negicc::set_shoulder(pc, c.shoulder.x_hi);
      c.shoulder.has_knee = true;
      c.shoulder.knee = pc.shoulder_x_k;
    }
    log_info("Roll-off: q995 %.3f, x_hi %.3f (sRGB output only)\n", q995, c.shoulder.x_hi);
  }
  return 0;
}

int convert(const Options& o, const Setup& s, Scan& scan, Conversion& c, Backend& backend) {
  OutputSpec& out = c.out;
  if (!o.colorspace.empty()) {
    const std::string& colorspace_str = o.colorspace;
    if (colorspace_str == "srgb") {
      out.encoding = ENCODE_SRGB;
      out.attach_profile = "srgb";
      log_info("Using elle sRGB TRC as output profile\n");
    } else if (colorspace_str == "srgb-g10" || colorspace_str == "srgb-linear") {
      out.encoding = ENCODE_SRGB_LINEAR;
      out.attach_profile = "srgb-g10";
      log_info("Using elle sRGB Linear (Gamma = 1.0) as output profile\n");
    } else {
      std::string resolved_custom = resolve_profile_path(colorspace_str);
      out.custom = cmsOpenProfileFromFile(resolved_custom.c_str(), "r");
      if (!out.custom) {
        log_error("Cannot read output ICC profile: %s\n", colorspace_str.c_str());
        return 1;
      }
      out.xyz = cmsCreateXYZProfile();
      out.encoding = ENCODE_CUSTOM;
      out.attach_profile = resolved_custom;
      log_info("Reading output ICC profile: %s\n", resolved_custom.c_str());
    }
  } else if (s.has_profile) {
    out.encoding = ENCODE_NONE;
    if (!s.is_bundle) out.attach_profile = s.film_prof_arg;
  }

  const double t_start = omp_get_wtime();
  // The shared pipeline (color_pipeline.h) with the frozen shoulder: the stage every consumer runs. Only the sRGB
  // encoding applies it.
  const negicc::PipelineConfig export_pc = make_pipeline_config(c.film_prof.data, c.knee, c.shoulder.x_hi, out.encoding);
  backend.convert(out, export_pc, c.merged_matrix, scan.image());
  // The white-light tone stage (np_pipeline.h) on the final sRGB image: the curve fit's plan, or the recorded curve on
  // a replay. Under the knobs the recorded curve is applied after them, as recorded: a fixed monotone map on top of the
  // knobbed render (zero knobs reproduce the export; a brighter knob gives a brighter result), not re-solved against
  // the knobbed render, which would pull the shadow level back to the prediction and undo the knob.
  scan.tone_applied = false;
  if (s.has_tone && out.encoding == ENCODE_SRGB) {
    LibRaw* tp = scan.proc.get();
    const size_t npx = (size_t)tp->imgdata.sizes.iwidth * tp->imgdata.sizes.iheight;
    uint16_t* rgb = !scan.rgb3.empty() ? scan.rgb3.data() : &tp->imgdata.image[0][0];
    const int stride = !scan.rgb3.empty() ? 3 : 4;
    const ToneStagePlan& plan = s.curve_fit.tone;
    scan.tone_curve = s.tone_curve;
    scan.tone_share = 1.0;
    const double t_tone_0 = omp_get_wtime();
    if (plan.planned && !plan.preview.empty()) {
      // This run's plan: its highlight move capped where it would clip this image (tone_plan_for_image).
      scan.tone_curve = tone_plan_for_image(s.tone_plan.render, plan.pred_mean[ToneHead::kL2],
                                            plan.pred_mean[ToneHead::kL98], plan.pred_mean[ToneHead::kR9010],
                                            plan.preview.data(),
                                            plan.preview.size() / 3, rgb, tp->imgdata.sizes.iwidth,
                                            tp->imgdata.sizes.iheight, stride, s.tone_plan.curve, s.tone_plan.targets,
                                            &scan.tone_targets, &scan.tone_share);
    }
    const double t_tone_1 = omp_get_wtime();
    const ToneCurve& q = scan.tone_curve;
    scan.tone_clip = apply_tone_stage(rgb, npx, stride, q);
    scan.tone_applied = true;
    log_info("Tone stage: black %.3f white %.3f gamma %.4f contrast %.4f chroma x%.4f (highlight share %.3f); white clip "
             "%.4f%% -> %.4f%%; %.0f ms checking the clipping, %.0f ms applying\n",
             q.black, q.white, q.gamma, q.contrast, q.chroma, scan.tone_share, 100.0 * scan.tone_clip.white_before,
             100.0 * scan.tone_clip.white_after, 1000.0 * (t_tone_1 - t_tone_0), 1000.0 * (omp_get_wtime() - t_tone_1));
  }
  if (out.encoding == ENCODE_CUSTOM) {
    cmsCloseProfile(out.xyz);
    cmsCloseProfile(out.custom);
    out.xyz = out.custom = nullptr;
  }
  const double t_end = omp_get_wtime();
  scan.t_full_conversion_ms = (t_end - t_start) * 1000.0;
  log_info("Processing time: %f\n", t_end - t_start);
  return 0;
}

int write(const Options& o, const Setup& s, Scan& scan, Conversion& c) {
  LibRaw* proc = scan.proc.get();
  const std::string& attach_profile = c.out.attach_profile;
  const OutputEncoding encoding = c.out.encoding;
  const int out_w = proc->imgdata.sizes.iwidth, out_h = proc->imgdata.sizes.iheight;

  log_info("Writing TIFF '%s'\n", o.output.c_str());
  const double t_tiff_0 = omp_get_wtime();
  int ret = write_tiff(proc, attach_profile, o.output, s.is_bundle ? c.embedded_icc : std::vector<uint8_t>{},
                       scan.rgb3.empty() ? nullptr : scan.rgb3.data());
  scan.t_tiff_write_ms = (omp_get_wtime() - t_tiff_0) * 1000.0;

  if (ret != 0 || o.no_metadata) {
    unlink((o.output + ".json").c_str());
    return ret;
  }

  std::string cs_name = "camera-linear";
  if (encoding == ENCODE_SRGB) {
    cs_name = "srgb";
  } else if (encoding == ENCODE_SRGB_LINEAR) {
    cs_name = "srgb-linear";
  } else if (encoding == ENCODE_CUSTOM) {
    cs_name = attach_profile;
  } else if (encoding == ENCODE_NONE) {
    cs_name = attach_profile.empty() ? "camera-linear" : attach_profile;
  }

  std::string roi_source = "none";
  if (o.roi_auto) {
    roi_source = (scan.frame_det.conf < kLowFrameConfidence) ? "auto_fallback" : "auto";
  } else if (scan.has_roi) {
    roi_source = "cli";
  }

  // A bundle conversion takes the film base read from the scan (Scan::film_base), else the bundle's.
  const bool film_base_applied = s.is_bundle && scan.film_base.found;

  Sidecar sc;
  sc.output_path = o.output;
  sc.argc = o.argc;
  sc.argv = o.argv;
  sc.proc = proc;
  sc.files = o.files;
  sc.file_fingerprints = &s.input_fingerprints;
  sc.orig_w = scan.orig_w;
  sc.orig_h = scan.orig_h;
  sc.has_roi = scan.has_roi;
  sc.roi = scan.roi;
  sc.roi_source = roi_source;
  sc.frame_det = o.roi_auto ? &scan.frame_det : nullptr;
  sc.rot_cw = scan.rot_cw;
  sc.hflip = scan.hflip;
  sc.vflip = scan.vflip;
  sc.out_w = out_w;
  sc.out_h = out_h;
  sc.film_base_applied = film_base_applied;
  sc.film_base_rgb.assign(c.sidecar_film_base_rgb.begin(), c.sidecar_film_base_rgb.end());
  sc.profile_film_base_rgb = c.sidecar_profile_film_base_rgb;
  sc.film_base_source = film_base_applied ? scan.film_base.source : c.film_base_source;
  sc.film_base_detection = s.is_bundle ? &scan.film_base : nullptr;
  sc.balance_gains = s.white_light ? scan.balance_gains : nullptr;
  sc.profile_path = s.is_bundle ? s.bundle.path : s.film_prof_arg;
  sc.is_bundle = s.is_bundle;
  sc.bundle_target_name = c.bundle_target_name;
  sc.bundle_target_idx = c.bundle_target_idx;
  sc.input_space = s.bundle.input_space;
  sc.merged_matrix = c.merged_matrix;
  sc.exposure_comp = c.exposure_comp;
  sc.gain_g = c.gain_g;
  sc.gain_b = c.gain_b;
  sc.station_params = c.station_params ? c.station_params->data() : nullptr;
  sc.pin_mask = c.pin_mask;
  sc.knee = c.knee;
  sc.shoulder = c.shoulder;
  if (encoding != ENCODE_SRGB && sc.shoulder.active()) {
    // Only the sRGB output applies the roll-off: record it as off (keeping the measured 99.5% point).
    sc.shoulder.x_hi = 1.0f;
    sc.shoulder.source = "off";
    sc.shoulder.has_knee = false;
  }
  sc.dino = s.use_dino && !scan.has_auto_res && !s.has_cached_sidecar ? &c.dino_res : nullptr;
  sc.dino_model = s.dino_path;
  sc.dino_backend = s.has_cached_sidecar ? s.cached_sidecar.dino_backend
                                         : (s.use_dino ? std::string(s.engine.backend_used()) : o.dino_backend);
  sc.auto_res = scan.has_auto_res && !s.has_cached_sidecar ? &scan.auto_res : nullptr;
  if (s.has_cached_sidecar) {
    // The previous run's search records, its intent field and its curves' record carry over unchanged.
    if (s.cached_sidecar.is_auto) {
      sc.cached_auto_json_str = s.cached_sidecar.auto_json_str;
    } else {
      sc.cached_dino_json_str = s.cached_sidecar.dino_json_str;
      sc.cached_dino_views_ms = s.cached_sidecar.t_dino_views_ms;
      sc.cached_dino_render_ms = s.cached_sidecar.t_dino_render_ms;
      sc.cached_dino_infer_ms = s.cached_sidecar.t_dino_infer_ms;
      sc.cached_dino_search_ms = s.cached_sidecar.t_dino_search_ms;
    }
    sc.cached_intent_field_json_str = s.cached_sidecar.intent_field_json_str;
    sc.cached_heads_json_str = s.cached_sidecar.heads_json_str;
    sc.cached_input_curves_json_str = s.cached_sidecar.input_curves_json_str;
  }
  if (scan.has_knob_res) {
    sc.has_knobs = true;
    sc.brightness = o.brightness;
    sc.tint = o.tint;
    sc.warmth = o.warmth;
    for (int i = 0; i < 3; ++i) sc.knob_base[i] = scan.knob_base[i];
    sc.knob_res = scan.knob_res;
  }
  sc.auto_frame_half = scan.auto_frame_half;
  sc.colorspace = cs_name;
  sc.t_raw_decode_ms = scan.t_raw_decode_ms;
  sc.t_frame_det_ms = scan.t_frame_det_ms;
  sc.t_full_conversion_ms = scan.t_full_conversion_ms;
  sc.t_tiff_write_ms = scan.t_tiff_write_ms;
  sc.t_auto_ms = scan.t_auto_ms;
  sc.t_analysis_ms = scan.t_analysis_ms;
  for (int i = 0; i < 3; ++i) {
    sc.head_path[i] = s.head_path[i];
    sc.head_fp[i] = s.head_fp[i];
  }
  sc.tint_trim_tables = s.has_tint_trim && !s.tint_trim_other_bracket ? &s.tint_trim : nullptr;
  sc.replay_source = s.replay_source;
  sc.replay_tint_trim_other_bracket = s.tint_trim_other_bracket;
  sc.replay_tint_trim_given = s.has_tint_trim;
  sc.tone_curve = scan.tone_applied ? &scan.tone_curve : s.has_tone ? &s.tone_curve : nullptr;
  sc.tone_targets = scan.tone_applied ? &scan.tone_targets : nullptr;
  sc.tone_share = scan.tone_share;
  sc.tone_applied = scan.tone_applied;
  sc.tone_clip = &scan.tone_clip;
  if (!s.input_curves_source.empty()) {
    sc.input_curves_source = s.input_curves_source;
    sc.input_curves = &s.input_curves;
    sc.cli_input_curves = scan.has_knob_curves ? &scan.knob_curves : nullptr;
    sc.curve_fit = (s.input_curves_source == "fit" || s.input_curves_source == "auto") && !s.has_cached_sidecar
                       ? &s.curve_fit : nullptr;
    sc.tone_plan = sc.curve_fit ? &s.tone_plan : nullptr;
  }
  write_json_sidecar(sc);
  return 0;
}
