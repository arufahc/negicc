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

#ifndef NEGICC_NEG_PIPELINE_H
#define NEGICC_NEG_PIPELINE_H

#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <vector>

struct Rgb {
  float r, g, b;
};

struct Lab {
  float L, a, b;
};

constexpr float kPcsScale = 1.99996948f;

enum OutputEncoding : int {
  ENCODE_SRGB = 0,
  ENCODE_SRGB_LINEAR = 1,
  ENCODE_CUSTOM = 2,
  ENCODE_NONE = 3
};

struct ProfileData {
  int has_profile = 0;
  const float* in_trc[3] = {nullptr, nullptr, nullptr};
  int in_trc_size[3] = {0, 0, 0};
  const float* out_trc[3] = {nullptr, nullptr, nullptr};
  int out_trc_size[3] = {0, 0, 0};
  float matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float offset[3] = {0, 0, 0};
  const float* clut = nullptr;
  int clut_dim[3] = {0, 0, 0};
};

struct ParsedFilmProfile {
  ProfileData data;
  std::vector<float> in_trc[3];
  std::vector<float> out_trc[3];
  std::vector<float> clut;

  void rebind_pointers() {
    for (int c = 0; c < 3; ++c) {
      data.in_trc[c] = in_trc[c].empty() ? nullptr : in_trc[c].data();
      data.out_trc[c] = out_trc[c].empty() ? nullptr : out_trc[c].data();
    }
    data.clut = clut.empty() ? nullptr : clut.data();
  }
};

static inline float clamp01(float v) {
  return fminf(1.0f, fmaxf(0.0f, v));
}

static inline Rgb load_u16(const uint16_t* px) {
  return { px[0] / 65535.0f, px[1] / 65535.0f, px[2] / 65535.0f };
}

static inline Rgb apply_matrix(const float* m, Rgb v) {
  return { v.r * m[0] + v.g * m[1] + v.b * m[2],
           v.r * m[3] + v.g * m[4] + v.b * m[5],
           v.r * m[6] + v.g * m[7] + v.b * m[8] };
}

static inline void candidate_matrix(const float* merged, float e, float g, float b, float* out) {
  const float sr = e;
  const float sg = 1.0f * e * g;
  const float sb = 1.0f * e * b;
  for (int i = 0; i < 3; ++i) {
    out[i]     = merged[i] * sr;
    out[3 + i] = merged[3 + i] * sg;
    out[6 + i] = merged[6 + i] * sb;
  }
}

static inline float knee_compress(float val, float knee) {
  if (val <= 0.0f) return 0.0f;
  if (knee >= 1.0f || val <= knee) return fminf(1.0f, val);
  const float delta = 1.0f - knee;
  return fminf(0.9999f, knee + delta * tanhf((val - knee) / delta));
}

static inline Rgb apply_knee(Rgb v, float knee) {
  return { knee_compress(v.r, knee), knee_compress(v.g, knee), knee_compress(v.b, knee) };
}

static inline float apply_trc(float val, const float* curve, int size) {
  if (size <= 0 || !curve) return val;
  const float scaled = val * (size - 1);
  int k = (int)scaled;
  k = k < 0 ? 0 : (k > size - 2 ? size - 2 : k);
  const float delta = scaled - k;
  return curve[k] * (1.0f - delta) + curve[k + 1] * delta;
}

// Tetrahedral 3D cLUT interpolation.
static inline Rgb interpolate_clut(float r, float g, float b, const float* clut, int dim_r, int dim_g, int dim_b) {
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

static inline Rgb apply_profile(const ProfileData& p, Rgb v) {
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

static inline Rgb pcs_to_linear_srgb(Rgb v) {
  // Bradford D50 -> D65
  const float xr = v.r * 0.9555766f + v.g * -0.0230393f + v.b * 0.0631636f;
  const float xg = v.r * -0.0282895f + v.g * 1.0099416f + v.b * 0.0210077f;
  const float xb = v.r * 0.0122982f + v.g * -0.0204830f + v.b * 1.3299098f;
  // XYZ -> linear sRGB
  return { clamp01(xr * 3.2406255f + xg * -1.5372080f + xb * -0.4986286f),
           clamp01(xr * -0.9689307f + xg * 1.8757561f + xb * 0.0415175f),
           clamp01(xr * 0.0557101f + xg * -0.2040211f + xb * 1.0569959f) };
}

static inline float srgb_encode(float l) {
  return (l <= 0.0031308f) ? (l * 12.92f) : (powf(l, 0.41666667f) * 1.055f - 0.055f);
}

static inline Rgb encode_output(Rgb v, int encoding) {
  if (encoding == ENCODE_SRGB) {
    v = { srgb_encode(v.r), srgb_encode(v.g), srgb_encode(v.b) };
  }
  return { clamp01(v.r), clamp01(v.g), clamp01(v.b) };
}

static inline uint8_t quantize_u8(float v) {
  return (uint8_t)roundf(clamp01(v) * 255.0f);
}

static inline uint16_t quantize_u16(float v) {
  return (uint16_t)roundf(clamp01(v) * 65535.0f);
}

static inline Rgb process_pixel(const uint16_t* in_px, const float* matrix, float knee,
                                bool has_gamma, float inv_gamma, const ProfileData& prof) {
  Rgb v = load_u16(in_px);
  v = apply_matrix(matrix, v);
  v = apply_knee(v, knee);
  if (has_gamma) {
    v.r = powf(clamp01(v.r), inv_gamma);
    v.g = powf(clamp01(v.g), inv_gamma);
    v.b = powf(clamp01(v.b), inv_gamma);
  }
  if (prof.has_profile) {
    v = apply_profile(prof, v);
  }
  return v;
}

static inline Rgb convert_pixel_srgb(const uint16_t* in_px, const float* matrix, float knee,
                                     bool has_gamma, float inv_gamma, const ProfileData& prof) {
  Rgb v = process_pixel(in_px, matrix, knee, has_gamma, inv_gamma, prof);
  v = pcs_to_linear_srgb(v);
  return encode_output(v, ENCODE_SRGB);
}

static inline float srgb_decode(float c) {
  return (c <= 0.04045f) ? (c * (1.0f / 12.92f)) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

static inline float lab_f(float t) {
  return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f);
}

static inline Lab srgb_to_lab(Rgb c) {
  const float r = srgb_decode(c.r), g = srgb_decode(c.g), b = srgb_decode(c.b);
  const float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
  const float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
  const float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
  const float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
  const float L = (Y > 0.008856f) ? (116.0f * fy - 16.0f) : (903.3f * Y);
  return { L, 500.0f * (fx - fy), 200.0f * (fy - fz) };
}

static inline bool channel_white_clipped(float v) {
  return v * 65535.0f >= 65399.5f;
}

#endif  // NEGICC_NEG_PIPELINE_H
