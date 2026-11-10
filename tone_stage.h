// The white-light tone stage: one monotone lightness curve on CIE L* (D65), set per frame from the tone heads
// (tone_head.h), applied to the final 16-bit sRGB image. Colour is not re-balanced: a* and b* are only scaled, by a
// factor chosen so the frame's mean chroma stays what it was before the curve.
//
//   f(L) = black + (white - black) * h(g(L / 100)),  g(x) = x^gamma,  h(x) = x^contrast / (x^contrast + (1 - x)^contrast)
//   a*, b* scaled by 1 + (chroma - 1) * smoothstep(3, 8, C*): near-neutral colours keep their balance, hue is kept.
//
// The four curve numbers are solved so the render's own descriptors move toward the heads' predictions: since f is
// monotone, the toned render's percentiles are f of the render's, so L2' = f(L2), L50' = f(L50), L98' = f(L98) and
// (L90 - L10)' = f(L90) - f(L10). Each target is the render's value moved a share of the way to the prediction:
// all the way for the shadow level L2 (kToneShadowMove), half way for the highlight level L98 and the tonal range
// L90 - L10 (kToneHighlightMove, kToneRangeMove), and not at all for the mid-tone L50, which the curve holds. A weak
// prior keeps the curve near the identity where the targets leave it free. The descriptors are measured on the
// network-size preview of the final render, as the heads' training targets were on lab scans.
#ifndef TONE_STAGE_H
#define TONE_STAGE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// The shares of the way from the render's descriptor to the heads' prediction the curve moves.
constexpr double kToneShadowMove = 1.0;     // L2
constexpr double kToneHighlightMove = 0.5;  // L98
constexpr double kToneRangeMove = 0.5;      // L90 - L10 (the mid-tone L50 is held where it is)
// The curve's bounds (black, white, gamma, contrast) and the identity it starts from.
constexpr double kToneLo[4] = {-15.0, 80.0, 0.7, 0.7};
constexpr double kToneHi[4] = {15.0, 110.0, 1.45, 1.5};
// The prior's weight, and its scales: 5 L* for the black and white points, 0.1 in log for gamma and contrast.
constexpr double kTonePrior = 0.02;
// Above the render's L98 no target says where the curve goes, and two quite different white points can fit the four
// targets about equally (on a night frame, 89 or 110): which one won flipped with a 1 L* change of the render. This
// weight holds the curve's rise above L98 to the input's, f(100) = f(L98) + (100 - L98) (slope 1 on average there),
// but never above white, f(100) <= 100, so a lifted L98 does not push the top past white into clipping. That makes
// the solution unique, and the solve starts from several white points and keeps the lowest cost, so it does not
// depend on which basin a single start falls into.
constexpr double kToneTopSlope = 0.07;
constexpr double kToneWhiteStarts[3] = {100.0, 110.0, 85.0};
// The chroma factor that keeps the mean chroma: refined this many times on the preview, within these bounds.
constexpr int kToneChromaRefinements = 2;
constexpr double kToneChromaLo = 0.5, kToneChromaHi = 1.6;

struct ToneCurve {
    double black = 0.0, white = 100.0, gamma = 1.0, contrast = 1.0, chroma = 1.0;
};

// The render's descriptors the curve is solved from: pixel L* percentiles and the mean chroma C*ab.
struct ToneDescriptors {
    double L2 = 0, L10 = 0, L25 = 0, L50 = 0, L75 = 0, L90 = 0, L98 = 0, Cmean = 0;
    double range() const { return L90 - L10; }
};

// sRGB (D65) <-> CIELAB in double precision, with the white point and matrices of the stage's design.
inline double tone_srgb_decode(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }
inline double tone_srgb_encode(double c) {
    c = c < 0 ? 0 : (c > 1 ? 1 : c);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1 / 2.4) - 0.055;
}
inline double tone_lab_f(double t) { return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0; }
inline double tone_lab_finv(double t) {
    return t * t * t > 216.0 / 24389.0 ? t * t * t : (116.0 * t - 16.0) / (24389.0 / 27.0);
}
constexpr double kToneXn = 0.950456, kToneZn = 1.088754;

// Linear r, g, b -> L*, a*, b*.
inline void tone_linear_to_lab(double r, double g, double b, double lab[3]) {
    const double X = 0.412453 * r + 0.357580 * g + 0.180423 * b;
    const double Y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
    const double Z = 0.019334 * r + 0.119193 * g + 0.950227 * b;
    const double fx = tone_lab_f(X / kToneXn), fy = tone_lab_f(Y), fz = tone_lab_f(Z / kToneZn);
    lab[0] = 116.0 * fy - 16.0;
    lab[1] = 500.0 * (fx - fy);
    lab[2] = 200.0 * (fy - fz);
}

// sRGB-encoded r, g, b in 0..1 -> L*, a*, b*.
inline void tone_rgb_to_lab(double r, double g, double b, double lab[3]) {
    tone_linear_to_lab(tone_srgb_decode(r), tone_srgb_decode(g), tone_srgb_decode(b), lab);
}

// L*, a*, b* -> r, g, b in 0..1 (clipped).
inline void tone_lab_to_rgb(const double lab[3], double rgb[3]) {
    const double fy = (lab[0] + 16.0) / 116.0, fx = fy + lab[1] / 500.0, fz = fy - lab[2] / 200.0;
    const double X = tone_lab_finv(fx) * kToneXn, Y = tone_lab_finv(fy), Z = tone_lab_finv(fz) * kToneZn;
    rgb[0] = tone_srgb_encode(3.240479 * X - 1.537150 * Y - 0.498535 * Z);
    rgb[1] = tone_srgb_encode(-0.969256 * X + 1.875992 * Y + 0.041556 * Z);
    rgb[2] = tone_srgb_encode(0.055648 * X - 0.204043 * Y + 1.057311 * Z);
}

// f(L) without the final clamp to 0..100.
inline double tone_curve_L(double L, const ToneCurve& q) {
    double x = L / 100.0;
    x = x < 1e-5 ? 1e-5 : (x > 1 - 1e-5 ? 1 - 1e-5 : x);
    x = std::pow(x, q.gamma);
    const double xc = std::pow(x, q.contrast), yc = std::pow(1 - x, q.contrast);
    return q.black + (q.white - q.black) * xc / (xc + yc);
}

// One pixel's Lab through the curve and the chroma factor.
inline void tone_lab_apply(double lab[3], const ToneCurve& q) {
    double L2 = tone_curve_L(lab[0], q);
    lab[0] = L2 < 0 ? 0 : (L2 > 100 ? 100 : L2);
    const double C = std::hypot(lab[1], lab[2]);
    double t = (C - 3.0) / 5.0;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    const double k = 1.0 + (q.chroma - 1.0) * t * t * (3 - 2 * t);
    lab[1] *= k;
    lab[2] *= k;
}

// The descriptors of an 8-bit RGB image (n pixels, 3 bytes each); percentiles interpolate linearly between ranks.
inline ToneDescriptors tone_descriptors(const uint8_t* rgb, size_t n) {
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

// The mean chroma of an 8-bit RGB image after the stage (toned in floating point, not re-quantised).
inline double tone_mean_chroma_after(const uint8_t* rgb, size_t n, const ToneCurve& q) {
    if (n == 0) return 0.0;
    double c_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double lab[3], out[3];
        tone_rgb_to_lab(rgb[3 * i] / 255.0, rgb[3 * i + 1] / 255.0, rgb[3 * i + 2] / 255.0, lab);
        tone_lab_apply(lab, q);
        tone_lab_to_rgb(lab, out);
        tone_rgb_to_lab(out[0], out[1], out[2], lab);
        c_sum += std::hypot(lab[1], lab[2]);
    }
    return c_sum / (double)n;
}

// The targets the curve is solved for (the render's descriptors moved toward the predictions).
struct ToneTargets {
    double L2 = 0, L50 = 0, L98 = 0, range = 0;
};

inline ToneTargets tone_targets(const ToneDescriptors& cur, double pred_L2, double pred_L98, double pred_range) {
    ToneTargets t;
    t.L2 = cur.L2 + kToneShadowMove * (pred_L2 - cur.L2);
    t.L98 = cur.L98 + kToneHighlightMove * (pred_L98 - cur.L98);
    t.L50 = cur.L50;
    t.range = cur.range() + kToneRangeMove * (pred_range - cur.range());
    return t;
}

// The curve (black, white, gamma, contrast) that brings cur to t: bounded least squares on the four target residuals
// and the identity prior, by Levenberg-Marquardt with the bounds held by projection (a parameter at a bound that the
// gradient pushes outward is held there). Returns the curve with chroma 1.
inline ToneCurve tone_solve(const ToneDescriptors& cur, const ToneTargets& t) {
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

// What applying the stage to the full image did: the share of pixels clipped white (a channel at the maximum) and
// black (every channel 0), before and after.
struct ToneClip {
    double white_before = 0, white_after = 0, black_before = 0, black_after = 0;
};

// rgb: interleaved 16-bit sRGB with `stride` uint16 per pixel (3 or 4), n pixels.
inline ToneClip apply_tone_stage(uint16_t* rgb, size_t n, int stride, const ToneCurve& q) {
    // The decoded value of every 16-bit code (the same numbers tone_rgb_to_lab computes, once per code).
    static const std::vector<double> decode = [] {
        std::vector<double> d(65536);
        for (int v = 0; v < 65536; ++v) d[v] = tone_srgb_decode(v / 65535.0);
        return d;
    }();
    long long wb = 0, wa = 0, kb = 0, ka = 0;
#pragma omp parallel for schedule(static) reduction(+ : wb, wa, kb, ka)
    for (long long i = 0; i < (long long)n; ++i) {
        uint16_t* px = rgb + (size_t)i * stride;
        wb += (px[0] == 65535 || px[1] == 65535 || px[2] == 65535);
        kb += (px[0] == 0 && px[1] == 0 && px[2] == 0);
        double lab[3], out[3];
        tone_linear_to_lab(decode[px[0]], decode[px[1]], decode[px[2]], lab);
        tone_lab_apply(lab, q);
        tone_lab_to_rgb(lab, out);
        for (int c = 0; c < 3; ++c) px[c] = (uint16_t)std::lround(out[c] * 65535.0);
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

#endif  // TONE_STAGE_H
