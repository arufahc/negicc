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
// White clipping, at its two sources. A colour the curve lifts out of the sRGB gamut has its chroma scaled toward
// neutral at its L* (tone_lab_to_rgb_in_gamut) instead of its brightest channels clipped. Where the curve lifts L*
// itself to white on more than kToneClipAllowance more of the final image's pixels than the image had clipped, the
// highlight and range moves are scaled down (bisection, kToneClipSearchSteps steps) until it does not, measured on
// every kToneClipSampleStep-th pixel each way of the final image. When even no highlight move stays within the
// allowance, the clipping does not come from that move, and the moves stay as designed.
constexpr double kToneClipAllowance = 0.005;
constexpr int kToneClipSearchSteps = 10;
constexpr int kToneClipSampleStep = 4;
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
// The bisection steps of the in-gamut chroma scale (tone_lab_to_rgb_in_gamut): 2^-14 of the chroma.
constexpr int kToneGamutSteps = 14;

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

// L*, a*, b* -> r, g, b in 0..1, the chroma scaled toward neutral at the same L* (hue kept) as far as needed to keep
// every channel at or below the maximum: a colour the curve lifts past the sRGB gamut keeps its tone and hue instead
// of having its brightest channels clipped. Below the maximum it is tone_lab_to_rgb.
inline void tone_lab_to_rgb_in_gamut(const double lab[3], double rgb[3]) {
    auto linear_max = [](const double l[3]) {
        const double fy = (l[0] + 16.0) / 116.0, fx = fy + l[1] / 500.0, fz = fy - l[2] / 200.0;
        const double X = tone_lab_finv(fx) * kToneXn, Y = tone_lab_finv(fy), Z = tone_lab_finv(fz) * kToneZn;
        return std::max({3.240479 * X - 1.537150 * Y - 0.498535 * Z, -0.969256 * X + 1.875992 * Y + 0.041556 * Z,
                         0.055648 * X - 0.204043 * Y + 1.057311 * Z});
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

// highlight_share scales the highlight and range moves (1: as designed; less where they would clip, see above).
inline ToneTargets tone_targets(const ToneDescriptors& cur, double pred_L2, double pred_L98, double pred_range,
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

// The plan at a highlight share: the targets, the curve solved for them, and the chroma factor that keeps the
// preview's (8-bit RGB, n pixels) mean chroma at the render's.
inline ToneCurve tone_plan(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
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
        const float X = 0.412453f * r + 0.357580f * g + 0.180423f * b;
        const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
        const float Z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
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
        for (int c = 0; c < 3; ++c) {
            float v = lin[c] < 0.0f ? 0.0f : (lin[c] > 1.0f ? 1.0f : lin[c]);
            v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
            out[c] = (uint16_t)std::lround(v * 65535.0f);
        }
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
    static float lab_f(float t) { return t > 216.0f / 24389.0f ? std::cbrt(t) : (24389.0f / 27.0f * t + 16.0f) / 116.0f; }
    static float lab_finv(float t) {
        return t * t * t > 216.0f / 24389.0f ? t * t * t : (116.0f * t - 16.0f) / (24389.0f / 27.0f);
    }
    static void to_linear(float L, float A, float B, float lin[3]) {
        const float fy = (L + 16.0f) / 116.0f, fx = fy + A / 500.0f, fz = fy - B / 200.0f;
        const float X = lab_finv(fx) * (float)kToneXn, Y = lab_finv(fy), Z = lab_finv(fz) * (float)kToneZn;
        lin[0] = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;
        lin[1] = -0.969256f * X + 1.875992f * Y + 0.041556f * Z;
        lin[2] = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
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
inline double tone_white_clip_share(const uint16_t* rgb, int w, int h, int stride, int step, const ToneCurve* q) {
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

// The plan's curve for a final image: the designed moves, or, where they would clip more than kToneClipAllowance
// beyond the image's own clipping, the largest share of the highlight and range moves that does not (see above).
// share gets the share kept.
inline ToneCurve tone_plan_for_image(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                                     const uint8_t* preview, size_t n, const uint16_t* rgb, int w, int h, int stride,
                                     ToneTargets* targets, double* share) {
    const double before = tone_white_clip_share(rgb, w, h, stride, kToneClipSampleStep, nullptr);
    ToneTargets t1;
    const ToneCurve c1 = tone_plan(render, pred_L2, pred_L98, pred_range, preview, n, 1.0, &t1);
    *share = 1.0;
    *targets = t1;
    auto ok = [&](const ToneCurve& c) {
        return tone_white_clip_share(rgb, w, h, stride, kToneClipSampleStep, &c) <= before + kToneClipAllowance;
    };
    if (ok(c1)) return c1;
    ToneTargets t_lo;
    ToneCurve c_lo = tone_plan(render, pred_L2, pred_L98, pred_range, preview, n, 0.0, &t_lo);
    if (!ok(c_lo)) return c1;  // the clipping is not the highlight move's
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < kToneClipSearchSteps; ++it) {
        const double mid = 0.5 * (lo + hi);
        ToneTargets t;
        const ToneCurve c = tone_plan(render, pred_L2, pred_L98, pred_range, preview, n, mid, &t);
        if (ok(c)) {
            lo = mid;
            t_lo = t;
            c_lo = c;
        } else {
            hi = mid;
        }
    }
    *share = lo;
    *targets = t_lo;
    return c_lo;
}

// What applying the stage to the full image did: the share of pixels clipped white (a channel at the maximum) and
// black (every channel 0), before and after.
struct ToneClip {
    double white_before = 0, white_after = 0, black_before = 0, black_after = 0;
};

// rgb: interleaved 16-bit sRGB with `stride` uint16 per pixel (3 or 4), n pixels.
inline ToneClip apply_tone_stage(uint16_t* rgb, size_t n, int stride, const ToneCurve& q) {
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

#endif  // TONE_STAGE_H
