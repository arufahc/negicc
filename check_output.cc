// Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
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

// The output conversion's self-check, built and run by `make check_output`: the PCS -> linear sRGB matrix
// (color_pipeline.h pcs_to_linear_srgb_unclamped) against the profiles neg_process embeds in its sRGB outputs (Elle
// Stone's sRGB V2, the sRGB-TRC one for -P srgb and the gamma 1.0 one for -P srgb-g10). For each profile:
//  1. the matrix equals the inverse of the profile's colorants (rXYZ, gXYZ, bXYZ, read by Little-CMS) to float precision;
//  2. the round trip: on a grid of PCS values inside the gamut, the 16-bit codes the pipeline writes (the matrix, the
//     encoding and the rounding neg_process uses), read back through the profile by Little-CMS (profile -> XYZ), give the
//     PCS written within half a code (and 0.02 for float rounding) beyond the profile's own TRC at the code written, in
//     each channel (the read-back's linear value, re-encoded, against the written one's);
//  3. the PCS white (color_pipeline.h kPcsWhite, the ICC's D50 in s15.16) equals the sum of the profile's colorants and
//     is written 65535 in all three channels; the decimal D50 Little-CMS uses (0.9642, 1, 0.8249) is written as
//     Little-CMS writes it through the profile (through the gamma 1.0 profile 65534 in blue: 0.999992).
// (Little-CMS's own XYZ -> profile transform is not the reference: on the grid it writes up to 65 codes where the exact
// value is 0, and its codes read back to 1.5e-3 of Y, where the pipeline's read back to 7e-5.)
// Prints each case; exits 1 if one fails.
#include <cmath>
#include <cstdio>

#include "color_pipeline.h"
#include "lcms2.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"

static int check_profile(const char* name, const unsigned char* icc, unsigned len, int encoding) {
  int failed = 0;
  cmsHPROFILE p = cmsOpenProfileFromMem(icc, len);
  if (!p) {
    std::printf("FAIL %s: Little-CMS cannot read the profile\n", name);
    return 1;
  }
  // 1. The matrix against the inverse of the colorants.
  const cmsCIEXYZ* col[3] = {(const cmsCIEXYZ*)cmsReadTag(p, cmsSigRedColorantTag),
                             (const cmsCIEXYZ*)cmsReadTag(p, cmsSigGreenColorantTag),
                             (const cmsCIEXYZ*)cmsReadTag(p, cmsSigBlueColorantTag)};
  double worst = 0.0;
  for (int c = 0; c < 3; ++c) {
    // The pipeline's matrix applied to a colorant gives that primary: (1, 0, 0) for red, and so on.
    const negicc::Rgb l = negicc::pcs_to_linear_srgb_unclamped({(float)col[c]->X, (float)col[c]->Y, (float)col[c]->Z});
    const double got[3] = {l.r, l.g, l.b};
    for (int r = 0; r < 3; ++r) worst = std::fmax(worst, std::fabs(got[r] - (r == c ? 1.0 : 0.0)));
  }
  // The double copy for host code (kPcsToLinearSrgb, the print's CLF), to double precision of its 9 digits.
  double worst_d = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double v[3] = {col[c]->X, col[c]->Y, col[c]->Z};
    for (int r = 0; r < 3; ++r) {
      const double got = negicc::kPcsToLinearSrgb[r * 3] * v[0] + negicc::kPcsToLinearSrgb[r * 3 + 1] * v[1] +
                         negicc::kPcsToLinearSrgb[r * 3 + 2] * v[2];
      worst_d = std::fmax(worst_d, std::fabs(got - (r == c ? 1.0 : 0.0)));
    }
  }
  worst = std::fmax(worst, worst_d);
  const bool ok1 = worst < 2e-6;
  failed += !ok1;
  std::printf("%s %s: the matrix maps each colorant to its primary, largest error %.2e (float precision: < 2e-6)\n",
              ok1 ? "ok  " : "FAIL", name, worst);
  // 2. The round trip through the profile.
  cmsHPROFILE xyz = cmsCreateXYZProfile();
  cmsHTRANSFORM back = cmsCreateTransform(p, TYPE_RGB_16, xyz, TYPE_XYZ_DBL, INTENT_RELATIVE_COLORIMETRIC,
                                          cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
  cmsHTRANSFORM fwd = cmsCreateTransform(xyz, TYPE_XYZ_DBL, p, TYPE_RGB_16, INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOCACHE);
  // The exact inverse of the colorants, in double, to read the PCS back into the profile's linear values.
  double m[9], inv[9];
  for (int r = 0; r < 3; ++r) {
    m[r * 3 + 0] = r == 0 ? col[0]->X : r == 1 ? col[0]->Y : col[0]->Z;
    m[r * 3 + 1] = r == 0 ? col[1]->X : r == 1 ? col[1]->Y : col[1]->Z;
    m[r * 3 + 2] = r == 0 ? col[2]->X : r == 1 ? col[2]->Y : col[2]->Z;
  }
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  inv[0] = (m[4] * m[8] - m[5] * m[7]) / det; inv[1] = (m[2] * m[7] - m[1] * m[8]) / det; inv[2] = (m[1] * m[5] - m[2] * m[4]) / det;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) / det; inv[4] = (m[0] * m[8] - m[2] * m[6]) / det; inv[5] = (m[2] * m[3] - m[0] * m[5]) / det;
  inv[6] = (m[3] * m[7] - m[4] * m[6]) / det; inv[7] = (m[1] * m[6] - m[0] * m[7]) / det; inv[8] = (m[0] * m[4] - m[1] * m[3]) / det;
  auto code_of = [&](double lin) {   // the continuous code of a linear value in this profile's coding
    const double c = std::fmin(1.0, std::fmax(0.0, lin));
    return 65535.0 * (encoding == negicc::ENCODE_SRGB ? (c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055) : c);
  };
  const cmsToneCurve* trc = (const cmsToneCurve*)cmsReadTag(p, cmsSigRedTRCTag);
  int n = 0;
  double max_diff = 0.0, max_excess = 0.0;
  const int steps = 17;
  for (int i = 0; i < steps; ++i)
    for (int j = 0; j < steps; ++j)
      for (int k = 0; k < steps; ++k) {
        const double rgb[3] = {i / (steps - 1.0), j / (steps - 1.0), k / (steps - 1.0)};
        cmsCIEXYZ pcs = {0, 0, 0};
        for (int c = 0; c < 3; ++c) {
          pcs.X += col[c]->X * rgb[c];
          pcs.Y += col[c]->Y * rgb[c];
          pcs.Z += col[c]->Z * rgb[c];
        }
        const negicc::Rgb e = negicc::encode_output(
            negicc::pcs_to_linear_srgb_unclamped({(float)pcs.X, (float)pcs.Y, (float)pcs.Z}), encoding);
        const cmsUInt16Number ours[3] = {negicc::quantize_u16(e.r), negicc::quantize_u16(e.g), negicc::quantize_u16(e.b)};
        cmsCIEXYZ got;
        cmsDoTransform(back, ours, &got, 1);
        for (int c = 0; c < 3; ++c) {
          const double lin_back = inv[c * 3] * got.X + inv[c * 3 + 1] * got.Y + inv[c * 3 + 2] * got.Z;
          const double diff = std::fabs(code_of(lin_back) - code_of(rgb[c]));
          // The profile's TRC at the code written, against the coding's formula: the read-back carries it.
          const double trc_here = std::fabs(code_of(cmsEvalToneCurveFloat(trc, (cmsFloat32Number)(ours[c] / 65535.0))) - ours[c]);
          max_diff = std::fmax(max_diff, diff);
          max_excess = std::fmax(max_excess, diff - trc_here);
        }
        ++n;
      }
  // Half a code of rounding, beyond the profile's own TRC at the code written (a profile stores its TRC as a sampled
  // curve, which Little-CMS interpolates), with 0.02 code for the pipeline's float arithmetic.
  const bool ok2 = max_excess <= 0.52;
  failed += !ok2;
  std::printf("%s %s: %d PCS values written and read back through the profile: largest difference %.3f code, %.3f beyond "
              "the profile's TRC at that code (at most 0.52: half a code and float rounding)\n",
              ok2 ? "ok  " : "FAIL", name, n, max_diff, max_excess);
  // 3. The PCS white: kPcsWhite is the colorants' sum, and is written at full scale.
  const double sum[3] = {col[0]->X + col[1]->X + col[2]->X, col[0]->Y + col[1]->Y + col[2]->Y, col[0]->Z + col[1]->Z + col[2]->Z};
  double sum_err = 0.0;
  for (int c = 0; c < 3; ++c) sum_err = std::fmax(sum_err, std::fabs(sum[c] - negicc::kPcsWhite[c]));
  const negicc::Rgb pw = negicc::encode_output(negicc::pcs_to_linear_srgb_unclamped(
      {(float)negicc::kPcsWhite[0], (float)negicc::kPcsWhite[1], (float)negicc::kPcsWhite[2]}), encoding);
  const int pc[3] = {negicc::quantize_u16(pw.r), negicc::quantize_u16(pw.g), negicc::quantize_u16(pw.b)};
  const bool ok4 = sum_err < 1e-9 && pc[0] == 65535 && pc[1] == 65535 && pc[2] == 65535;
  failed += !ok4;
  std::printf("%s %s: kPcsWhite equals the colorants' sum (error %.1e) and is written %d %d %d\n", ok4 ? "ok  " : "FAIL", name,
              sum_err, pc[0], pc[1], pc[2]);
  // And the decimal D50 that Little-CMS uses, written as Little-CMS writes it.
  const cmsCIEXYZ* d50 = cmsD50_XYZ();
  const negicc::Rgb w = negicc::encode_output(
      negicc::pcs_to_linear_srgb_unclamped({(float)d50->X, (float)d50->Y, (float)d50->Z}), encoding);
  const int wc[3] = {negicc::quantize_u16(w.r), negicc::quantize_u16(w.g), negicc::quantize_u16(w.b)};
  cmsUInt16Number lw[3];
  cmsDoTransform(fwd, d50, lw, 1);
  const bool ok3 = wc[0] == lw[0] && wc[1] == lw[1] && wc[2] == lw[2];
  failed += !ok3;
  std::printf("%s %s: the PCS white (D50 %.6f %.6f %.6f) is written %d %d %d; Little-CMS writes it %d %d %d\n",
              ok3 ? "ok  " : "FAIL", name, d50->X, d50->Y, d50->Z, wc[0], wc[1], wc[2], lw[0], lw[1], lw[2]);
  cmsDeleteTransform(back);
  cmsDeleteTransform(fwd);
  cmsCloseProfile(xyz);
  cmsCloseProfile(p);
  return failed;
}

int main() {
  int failed = check_profile("-P srgb (sRGB_elle_V2_srgbtrc)", sRGB_elle_V2_srgbtrc_icc, sRGB_elle_V2_srgbtrc_icc_len,
                             negicc::ENCODE_SRGB);
  failed += check_profile("-P srgb-g10 (sRGB_elle_V2_g10)", sRGB_elle_V2_g10_icc, sRGB_elle_V2_g10_icc_len,
                          negicc::ENCODE_SRGB_LINEAR);
  std::printf("%s: %d failed\n", failed ? "FAIL" : "ok", failed);
  return failed ? 1 : 0;
}
