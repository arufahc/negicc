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

// The tone stage's curve, plan and clip check, shared by neg_process and negicc-station (part of libnegicc_dinov3.a;
// no LibRaw or option dependencies).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "color_pipeline.h"

// ---- The tone stage (tone_stage.cc) ----
// The tone stage, on white-light and TriLED bundles alike: one monotone lightness curve on CIE L* (D65), set per frame
// from the tone heads
// (ToneHead, dinov3_engine.h), applied in the conversion kernel to each pixel's display linear sRGB value
// (color_pipeline.h apply_tone): its luminance Y goes to Y' = Y(f(L*(Y))) on the white point's neutral, and its colour
// difference from that neutral is scaled by f'(Y) / f'(Y') (Lab's f), which holds a* and b* to first order. Colour is
// not re-balanced, and no pixel is converted to Lab.
//
//   f(L) = black + (white - black) * h(g(L / 100)),  g(x) = x^gamma,  h(x) = x^contrast / (x^contrast + (1 - x)^contrast)
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
// White clipping: where the curve would clip more than kToneClipAllowance more of the export's pixels white than the
// conversion without it, the highlight and range moves are scaled down to the largest of kToneClipShares that keeps
// it within the allowance, measured on every 8th pixel each way of the whole unconverted image (film borders included)
// through the kernel. When none does, the share that clips least is kept.
constexpr double kToneClipAllowance = 0.005;
constexpr double kToneClipShares[5] = {1.0, 0.75, 0.5, 0.25, 0.0};

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

// The share of the export's pixels clipped white (a channel at the maximum) and black (every channel 0): before the
// stage on the clip check's sample of the conversion without the curve, after it on the whole export.
struct ToneClip {
    double white_before = 0, white_after = 0, black_before = 0, black_after = 0;
};

// The descriptors of an 8-bit RGB image (n pixels, 3 bytes each); percentiles interpolate linearly between ranks.
ToneDescriptors tone_descriptors(const uint8_t* rgb, size_t n);

// The plan at a highlight share: the targets and the curve solved for them.
ToneCurve tone_plan(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                    double highlight_share, ToneTargets* targets);

// The kernel's table of curve q (color_pipeline.h ToneTable: kToneTableSize + 1 entries on [0, 1]).
constexpr int kToneTableSize = 8192;
std::vector<float> tone_table(const ToneCurve& q);

// Which of kToneClipShares the plan keeps, as an index: 0 (the designed moves) when the designed curve clips no more
// than kToneClipAllowance beyond white_before (the conversion's own white clipping); else the largest share that does
// not; else the share that clips least (the larger on a tie). white_after_of(k) gives the white clipping with the
// curve planned at kToneClipShares[k]; it is called for k = 0 first and for no share after the one kept.
int tone_pick_clip_share(double white_before, const std::function<double(int)>& white_after_of);

// The share of a sparse sample of the unconverted image img (every 8th pixel each way) that the conversion through pc
// and matrix clips white (a channel at the 16-bit maximum) and black (every channel 0): with pc as it is (q null), or
// with curve q's table. Either output may be null.
void tone_clip_sample(const negicc::PipelineConfig& pc, const float* matrix, const uint16_t* img, int w, int h,
                      int stride, const ToneCurve* q, double* white, double* black);

// The plan's curve for the image the export converts (img, unconverted, w x h with stride uint16 per pixel, through
// pc and matrix): the designed moves (designed, solved for designed_targets by tone_plan at share 1), or, where the
// curve would clip more than kToneClipAllowance beyond the conversion's own clipping (white_before), the largest
// share of the highlight and range moves that does not, or, when none does, the share that clips least. The clipping
// is measured on a sparse sample of the whole of img (film borders included, while the plan's preview shows only the
// inner crop: a known mismatch) converted through the kernel with each curve. share gets the share kept.
ToneCurve tone_plan_for_image(const ToneDescriptors& render, double pred_L2, double pred_L98, double pred_range,
                              const negicc::PipelineConfig& pc, const float* matrix,
                              const uint16_t* img, int w, int h, int stride, double white_before,
                              const ToneCurve& designed, const ToneTargets& designed_targets, ToneTargets* targets,
                              double* share);

// The tone stage's plan as the conversion solves it after the curve fit (from the fit's ToneStagePlan: the heads'
// predictions and the final render's preview): the preview's descriptors, the targets and the curve between them.
struct TonePlanCurve {
  ToneDescriptors render;         // the final render's preview before the stage
  ToneTargets targets;            // the descriptors the curve aims at
  ToneCurve curve;                // the curve
};
