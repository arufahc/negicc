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

#include "tone_stage.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

// ---- Stage 4, the tone stage (tone_stage.h) ----
//
// One monotone L* curve applied in the conversion kernel (color_pipeline.h apply_tone), holding the mid-tones. It
// takes the tone heads' predictions of the shadow level, highlight level and tonal range and the preview they were
// made on; the conversion solves the curve after the fit or the search (tone_plan), scales the highlight and range
// moves down where the curve would clip the export (tone_plan_for_image, on a sample through the kernel) and
// converts the image once with the curve's table (tone_table).
//
// The preview's L* (sRGB D65, in double precision, Y from the luminance row of linear sRGB D65 -> XYZ): the descriptors
// the curve is solved from. Only the small preview is measured this way; no output pixel goes through Lab.
static inline double tone_srgb_decode(double c) {
    return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
static inline double tone_lab_f(double t) {
    return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0;
}
static inline double tone_lab_finv(double t) {
    return t * t * t > 216.0 / 24389.0 ? t * t * t : (116.0 * t - 16.0) / (24389.0 / 27.0);
}

// f(L) without the final clamp to 0..100.
static inline double tone_curve_L(double L, const ToneCurve& q) {
    double x = L / 100.0;
    x = x < 1e-5 ? 1e-5 : (x > 1 - 1e-5 ? 1 - 1e-5 : x);
    x = std::pow(x, q.gamma);
    const double xc = std::pow(x, q.contrast), yc = std::pow(1 - x, q.contrast);
    return q.black + (q.white - q.black) * xc / (xc + yc);
}

// The kernel's table: Y' = Y(f(L*(Y))), f clamped to 0..100, at Y = (i / kToneTableSize)^2.
std::vector<float> tone_table(const ToneCurve& q) {
    std::vector<float> lut(kToneTableSize + 1);
    for (int i = 0; i <= kToneTableSize; ++i) {
        const double t = (double)i / kToneTableSize;
        double L2 = tone_curve_L(116.0 * tone_lab_f(t * t) - 16.0, q);
        L2 = L2 < 0 ? 0 : (L2 > 100 ? 100 : L2);
        lut[i] = (float)tone_lab_finv((L2 + 16.0) / 116.0);
    }
    return lut;
}

ToneDescriptors tone_descriptors(const uint8_t* rgb, size_t n) {
    ToneDescriptors d;
    if (n == 0) return d;
    std::vector<double> L(n);
    // An 8-bit channel has 256 values: their sRGB decode is tabulated (the same doubles as decoding each pixel).
    double decode[256];
    for (int v = 0; v < 256; ++v) decode[v] = tone_srgb_decode(v / 255.0);
    for (size_t i = 0; i < n; ++i) {
        const double Y = 0.212671 * decode[rgb[3 * i]] + 0.715160 * decode[rgb[3 * i + 1]] +
                         0.072169 * decode[rgb[3 * i + 2]];
        L[i] = 116.0 * tone_lab_f(Y) - 16.0;
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
    return d;
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
// gradient pushes outward is held there). Returns the curve.
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
                    double highlight_share, ToneTargets* targets) {
    const ToneTargets t = tone_targets(render, pred_L2, pred_L98, pred_range, highlight_share);
    if (targets) *targets = t;
    return tone_solve(render, t);
}

// The share of the conversion's pixels clipped white (a channel at the maximum) and black (every channel 0) on a sparse
// sample of the unconverted image (every kToneClipSampleStep-th pixel each way), converted through the kernel as the
// export converts them: with pc as it is (no tone), or with curve q's table.
void tone_clip_sample(const negicc::PipelineConfig& pc, const float* matrix, const uint16_t* img, int w, int h,
                             int stride, const ToneCurve* q, double* white, double* black) {
    static constexpr int kToneClipSampleStep = 8;
    negicc::PipelineConfig c = pc;
    std::vector<float> lut;
    if (q) {
        lut = tone_table(*q);
        c.tone.lut = lut.data();
        c.tone.n = kToneTableSize;
    }
    long long n = 0, wc = 0, kc = 0;
#pragma omp parallel for schedule(static) reduction(+ : n, wc, kc)
    for (int y = 0; y < h; y += kToneClipSampleStep) {
        for (int x = 0; x < w; x += kToneClipSampleStep) {
            const negicc::Rgb v = negicc::convert_pixel(c, matrix, img + ((size_t)y * w + x) * stride);
            const uint16_t o[3] = {negicc::quantize_u16(v.r), negicc::quantize_u16(v.g), negicc::quantize_u16(v.b)};
            ++n;
            wc += (o[0] == 65535 || o[1] == 65535 || o[2] == 65535);
            kc += (o[0] == 0 && o[1] == 0 && o[2] == 0);
        }
    }
    *white = n ? (double)wc / (double)n : 0.0;
    if (black) *black = n ? (double)kc / (double)n : 0.0;
}

int tone_pick_clip_share(double white_before, const std::function<double(int)>& white_after_of) {
    // The designed curve, else the largest of the fixed shares within the allowance, else the least clipping of them
    // (the larger share on a tie). Fixed shares need no monotone clipping (a bisection does) and cost at most four more
    // curves.
    const int n = (int)(sizeof(kToneClipShares) / sizeof(kToneClipShares[0]));
    int best = 0;
    double best_after = white_after_of(0);
    if (best_after <= white_before + kToneClipAllowance) return 0;
    for (int k = 1; k < n; ++k) {
        const double after = white_after_of(k);
        if (after <= white_before + kToneClipAllowance) return k;
        if (after < best_after) {
            best = k;
            best_after = after;
        }
    }
    return best;
}

ToneCurve tone_plan_for_image(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                              const negicc::PipelineConfig& pc, const float* matrix,
                              const uint16_t* img, int w, int h, int stride, double white_before,
                              const ToneCurve& designed, const ToneTargets& designed_targets, ToneTargets* targets,
                              double* share) {
    constexpr int n = (int)(sizeof(kToneClipShares) / sizeof(kToneClipShares[0]));
    ToneCurve curves[n];
    ToneTargets tts[n];
    curves[0] = designed;
    tts[0] = designed_targets;
    const int k = tone_pick_clip_share(white_before, [&](int i) {
        if (i > 0) curves[i] = tone_plan(render, pred_L2, pred_L98, pred_range, kToneClipShares[i], &tts[i]);
        double after = 0.0;
        tone_clip_sample(pc, matrix, img, w, h, stride, &curves[i], &after, nullptr);
        return after;
    });
    *share = kToneClipShares[k];
    *targets = tts[k];
    return curves[k];
}
