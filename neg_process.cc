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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <lcms2.h>
#include <lcms2_plugin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <netinet/in.h>

#include "argparse/argparse.hpp"
#include "dcraw/gamma_curve.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "libraw/libraw.h"
#include "libraw/tiff_head.h"

#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 14))
#error This code is for LibRaw 0.14+ only
#endif

struct Rgb {
  float r, g, b;
};

// ICC PCS encoding scale applied after the output TRCs (1 + 32767/32768).
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
};

inline float clamp01(float v) {
  return fminf(1.0f, fmaxf(0.0f, v));
}

inline Rgb load_u16(const uint16_t* px) {
  return { px[0] / 65535.0f, px[1] / 65535.0f, px[2] / 65535.0f };
}

inline Rgb apply_matrix(const float* m, Rgb v) {
  return { v.r * m[0] + v.g * m[1] + v.b * m[2],
           v.r * m[3] + v.g * m[4] + v.b * m[5],
           v.r * m[6] + v.g * m[7] + v.b * m[8] };
}

inline float knee_compress(float val, float knee) {
  if (val <= 0.0f) return 0.0f;
  if (knee >= 1.0f || val <= knee) return fminf(1.0f, val);
  const float delta = 1.0f - knee;
  return fminf(0.9999f, knee + delta * tanhf((val - knee) / delta));
}

inline Rgb apply_knee(Rgb v, float knee) {
  return { knee_compress(v.r, knee), knee_compress(v.g, knee), knee_compress(v.b, knee) };
}

inline float apply_trc(float val, const float* curve, int size) {
  if (size <= 0 || !curve) return val;
  const float scaled = val * (size - 1);
  int k = (int)scaled;
  k = k < 0 ? 0 : (k > size - 2 ? size - 2 : k);
  const float delta = scaled - k;
  return curve[k] * (1.0f - delta) + curve[k + 1] * delta;
}

// Tetrahedral 3D cLUT interpolation.
inline Rgb interpolate_clut(float r, float g, float b, const float* clut, int dim_r, int dim_g, int dim_b) {
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

inline Rgb apply_profile(const ProfileData& p, Rgb v) {
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

inline Rgb pcs_to_linear_srgb(Rgb v) {
  // Bradford D50 -> D65
  const float xr = v.r * 0.9555766f + v.g * -0.0230393f + v.b * 0.0631636f;
  const float xg = v.r * -0.0282895f + v.g * 1.0099416f + v.b * 0.0210077f;
  const float xb = v.r * 0.0122982f + v.g * -0.0204830f + v.b * 1.3299098f;
  // XYZ -> linear sRGB
  return { clamp01(xr * 3.2406255f + xg * -1.5372080f + xb * -0.4986286f),
           clamp01(xr * -0.9689307f + xg * 1.8757561f + xb * 0.0415175f),
           clamp01(xr * 0.0557101f + xg * -0.2040211f + xb * 1.0569959f) };
}

inline float srgb_encode(float l) {
  return (l <= 0.0031308f) ? (l * 12.92f) : (powf(l, 0.41666667f) * 1.055f - 0.055f);
}

inline Rgb encode_output(Rgb v, int encoding) {
  if (encoding == ENCODE_SRGB) {
    v = { srgb_encode(v.r), srgb_encode(v.g), srgb_encode(v.b) };
  }
  return { clamp01(v.r), clamp01(v.g), clamp01(v.b) };
}

inline uint16_t quantize_u16(float v) {
  return (uint16_t)roundf(clamp01(v) * 65535.0f);
}

inline Rgb process_pixel(const ushort* in_px, const float* matrix, float knee,
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

static void read_curve_set(_cmsStageToneCurvesData* data,
                           std::vector<float> (&out)[3],
                           int (&out_sizes)[3],
                           const float* (&out_ptrs)[3]) {
  for (int c = 0; c < 3; ++c) {
    if (!data || c >= (int)data->nCurves || !data->TheCurves[c]) {
      out[c].clear();
      out_sizes[c] = 0;
      out_ptrs[c] = nullptr;
      continue;
    }
    cmsToneCurve* tc = data->TheCurves[c];
    const int entries = cmsGetToneCurveEstimatedTableEntries(tc);
    const cmsUInt16Number* table = cmsGetToneCurveEstimatedTable(tc);
    if (entries > 0 && table) {
      out[c].resize(entries);
      for (int i = 0; i < entries; ++i) {
        out[c][i] = table[i] / 65535.0f;
      }
    } else {
      out[c].resize(4096);
      for (int i = 0; i < 4096; ++i) {
        float in_v = (float)i / 4095.0f;
        out[c][i] = cmsEvalToneCurveFloat(tc, in_v);
      }
    }
    out_sizes[c] = (int)out[c].size();
    out_ptrs[c] = out[c].data();
  }
}

std::string resolve_profile_path(const std::string& input_path) {
  FILE* fp = fopen(input_path.c_str(), "rb");
  if (fp) {
    fclose(fp);
    return input_path;
  }
  std::string alt = "profiles/" + input_path;
  fp = fopen(alt.c_str(), "rb");
  if (fp) {
    fclose(fp);
    return alt;
  }
  if (input_path.rfind("icc_out/", 0) == 0) {
    std::string remapped = "profiles/" + input_path.substr(8);
    fp = fopen(remapped.c_str(), "rb");
    if (fp) {
      fclose(fp);
      return remapped;
    }
  }
  size_t last_slash = input_path.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    std::string base = "profiles/" + input_path.substr(last_slash + 1);
    fp = fopen(base.c_str(), "rb");
    if (fp) {
      fclose(fp);
      return base;
    }
  }
  return input_path;
}

bool load_film_profile(const std::string& raw_path, ParsedFilmProfile& prof) {
  std::string resolved_path = resolve_profile_path(raw_path);
  cmsHPROFILE hprof = cmsOpenProfileFromFile(resolved_path.c_str(), "r");
  if (!hprof) {
    fprintf(stderr, "ERROR! Cannot open film ICC profile: %s\n", raw_path.c_str());
    return false;
  }
  cmsPipeline* pipeline = (cmsPipeline*)cmsReadTag(hprof, cmsSigAToB0Tag);
  if (!pipeline) {
    fprintf(stderr, "ERROR! Film ICC profile %s has no AToB0 tag\n", resolved_path.c_str());
    cmsCloseProfile(hprof);
    return false;
  }

  prof.data.has_profile = 1;
  const float ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  memcpy(prof.data.matrix, ident, sizeof(ident));
  prof.data.offset[0] = prof.data.offset[1] = prof.data.offset[2] = 0.0f;

  int curve_sets = 0;
  for (cmsStage* st = cmsPipelineGetPtrToFirstStage(pipeline); st; st = cmsStageNext(st)) {
    const cmsStageSignature type = cmsStageType(st);
    if (type == cmsSigCurveSetElemType) {
      auto* cd = (_cmsStageToneCurvesData*)cmsStageData(st);
      if (curve_sets == 0) {
        read_curve_set(cd, prof.in_trc, prof.data.in_trc_size, prof.data.in_trc);
      } else {
        read_curve_set(cd, prof.out_trc, prof.data.out_trc_size, prof.data.out_trc);
      }
      curve_sets++;
    } else if (type == cmsSigMatrixElemType) {
      auto* md = (_cmsStageMatrixData*)cmsStageData(st);
      for (int i = 0; i < 9; ++i) prof.data.matrix[i] = (float)md->Double[i];
      for (int i = 0; i < 3; ++i) prof.data.offset[i] = md->Offset ? (float)md->Offset[i] : 0.0f;
    } else if (type == cmsSigCLutElemType) {
      auto* cl = (_cmsStageCLutData*)cmsStageData(st);
      for (int i = 0; i < 3; ++i) prof.data.clut_dim[i] = cl->Params->nSamples[i];
      const size_t n = (size_t)prof.data.clut_dim[0] * prof.data.clut_dim[1] * prof.data.clut_dim[2] * 3;
      prof.clut.resize(n);
      if (cl->HasFloatValues) {
        for (size_t i = 0; i < n; ++i) prof.clut[i] = cl->Tab.TFloat[i];
      } else {
        for (size_t i = 0; i < n; ++i) prof.clut[i] = cl->Tab.T[i] / 65535.0f;
      }
      prof.data.clut = prof.clut.data();
    }
  }
  cmsCloseProfile(hprof);
  printf("Reading input ICC profile: %s\n", resolved_path.c_str());
  return true;
}

int read_profile(const std::string& prof_name, unsigned **prof_out, unsigned *size) {
  std::string resolved = resolve_profile_path(prof_name);
  FILE *fp = fopen(resolved.c_str(), "rb");
  if (fp) {
    if (fread(size, 4, 1, fp) != 1) {
      fclose(fp);
      return -1;
    }
    fseek(fp, 0, SEEK_SET);
    *prof_out = (unsigned *)malloc(*size = ntohl(*size));
    if (*prof_out && fread(*prof_out, 1, *size, fp) == *size) {
      fclose(fp);
      return 0;
    }
    if (*prof_out) free(*prof_out);
    fclose(fp);
  }
  fprintf(stderr, "ERROR! Cannot read ICC profile: %s\n", prof_name.c_str());
  return -1;
}

// Load a RAW file and decode it into linear values.
//
// If |debayer| is true, interpolation is performed to generate missing pixels
// from the color filter array (usually a bayer pattern). Otherwise, the linear
// RGB values will have missing pixels and will not be scaled to 16-bit. If the
// sensor produces 14-bit files, then a scale factor of 4 needs to be applied,
// this is needed only when merging pixel-shift images.
//
// |qual| chooses the debayer algorithm used. 0 is the faster and is bilinear.
// Since the RAW capture is supposed to be linear to dye densities, which are
// supposed to be independent and have different grain structures, interpolation
// of the RAW file often produces artifacts that accentuates visible grain. This
// is caused by the debayer algorithm reading pixels from red channel (more
// grain) to generate pixels for blue and green channels (less grain). qual = 0
// is preferred or use pixel shift to eliminate need for interpolation.
//
// When |crop| is false, the entire RAW file is used, disregarding aspect ratio
// and cropbox specified in the RAW metadata.
LibRaw* load_raw(const std::string& fn, bool debayer, bool half_size, int qual, bool crop) {
  int ret;
  LibRaw* proc = new LibRaw();

  printf("Loading RAW file %s\n", fn.c_str());
  if ((ret = proc->open_file(fn.c_str())) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot open %s: %s\n", fn.c_str(), libraw_strerror(ret));
    delete proc;
    return NULL;
  }
  printf("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  if (debayer)
    printf("Debayer quality: %d\n", qual);

  if ((ret = proc->unpack()) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot unpack %s: %s\n", fn.c_str(), libraw_strerror(ret));
    delete proc;
    return NULL;
  }
  if (!(proc->imgdata.idata.filters || proc->imgdata.idata.colors == 1)) {
    printf("Only Bayer-pattern RAW files supported, sorry....\n");
    delete proc;
    return NULL;
  }

  proc->imgdata.params.output_bps = 16;
  proc->imgdata.params.user_flip = 0;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.gamm[1] = 1;
  proc->imgdata.params.no_auto_bright = 1;
  proc->imgdata.params.no_auto_scale = 1;
  proc->imgdata.params.highlight = 1;
  proc->imgdata.params.output_color = 0;
  proc->imgdata.params.output_tiff = 1;
  if (!debayer) {
    proc->imgdata.params.no_interpolation = 1;
    proc->raw2image();
    proc->subtract_black();
  } else {
    proc->imgdata.params.half_size = half_size;
    proc->imgdata.params.user_qual = qual;
    proc->imgdata.params.use_auto_wb = 0;
    proc->imgdata.params.user_mul[0] = 1;
    proc->imgdata.params.user_mul[1] = 1;
    proc->imgdata.params.user_mul[2] = 1;
    proc->imgdata.params.user_mul[3] = 1;
#if LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21)
    auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
#else
    auto& crop_ref = proc->imgdata.sizes.raw_inset_crop;
#endif
    if (crop && (crop_ref.cleft || crop_ref.ctop)) {
      proc->imgdata.params.cropbox[0] = crop_ref.cleft;
      proc->imgdata.params.cropbox[1] = crop_ref.ctop;
      proc->imgdata.params.cropbox[2] = crop_ref.cwidth;
      proc->imgdata.params.cropbox[3] = crop_ref.cheight;
    }
    proc->dcraw_process();
  }
  printf("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}

static const int kPixelShiftMoves[4][2] = {
  {0, 0},
  {0, 1},
  {-1, 1},
  {-1, 0},
};

void merge_pixel_shift_frame(LibRaw* base, LibRaw* frame, int mi) {
  const int dc = kPixelShiftMoves[mi][0];
  const int dr = kPixelShiftMoves[mi][1];
  const int bw = base->imgdata.sizes.iwidth;
  const int fw = frame->imgdata.sizes.iwidth;
  const int fh = frame->imgdata.sizes.iheight;
  for (int r = 0; r < fh - 1; ++r) {
    for (int c = 1; c < fw; ++c) {
      const int col = frame->COLOR(r, c);
      ushort* dst = base->imgdata.image[(r + dr) * bw + (c + dc)];
      const ushort v = frame->imgdata.image[r * fw + c][col];
      if (col & 1)
        dst[1] = (mi < 2) ? v : (ushort)((v + dst[1]) / 2);
      else
        dst[col] = v;
    }
  }
}

// Merge 4 RAW files from pixel-shift captures using Sony camera, loading one frame at a time to save memory.
LibRaw* merge_pixel_shift_streaming(const std::vector<std::string>& files) {
  if (files.size() < 4) return nullptr;
  printf("Merging 4 images...\n");
  LibRaw* base = load_raw(files[0], false, false, 0, false);
  if (!base) return nullptr;
  merge_pixel_shift_frame(base, base, 0);
  for (int mi = 1; mi < 4; ++mi) {
    LibRaw* frame = load_raw(files[mi], false, false, 0, false);
    if (!frame) {
      base->recycle();
      delete base;
      return nullptr;
    }
    merge_pixel_shift_frame(base, frame, mi);
    frame->recycle();
    delete frame;
  }
  base->imgdata.idata.colors = 3;
  return base;
}

template <class T>
float dot_product(const std::vector<float>& v1, const std::vector<T>& v2) {
  float prod = 0;
  for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
    prod += v1[i] * v2[i];
  }
  return prod;
}

void adjust_correction_matrix(const std::vector<float>& r_coef, 
                              const std::vector<float>& g_coef,
                              const std::vector<float>& b_coef,
                              float global_exposure_comp,
                              float gain_g,
                              float gain_b,
                              const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb,
                              float* merged_matrix) {
  // First convert the linear RGB into corrected RGB values that are proportional
  // to transmittance.
  float cc_average_r = dot_product(r_coef, film_base_rgb);
  float cc_average_g = dot_product(g_coef, film_base_rgb);
  float cc_average_b = dot_product(b_coef, film_base_rgb);
  float cc_profile_r = dot_product(r_coef, profile_film_base_rgb);
  float cc_profile_g = dot_product(g_coef, profile_film_base_rgb);
  float cc_profile_b = dot_product(b_coef, profile_film_base_rgb);
  printf("Film base RGB (corrected): %f %f %f\n", cc_average_r, cc_average_g, cc_average_b);
  printf("Profile film base RGB (corrected): %f %f %f\n", cc_profile_r, cc_profile_g, cc_profile_b);

  // The ICC profiles map the transmittance values after the correction matrix is
  // applied to XYZ. Transmittance is relative the profile film base. For example
  // if the R value for profile film base is 10000, a R value of 5000 has
  // transmittance of 5000 / 10000 = 0.5. But we don't convert the corrected RGB
  // into a floating point value first, the actual denominator is not important.
  // Suppose we record the transmittance as a 16-bit fixed point number, we will
  // lose precision for denser film. So the ICC profiles are generated with
  // transmittance values scaled given a fixed value for mid-grey patch (e.g.
  // GS14).
  //
  // The film base RGB values are only useful in relative terms. Here we are
  // compensating the difference in transmittance between the profile and the
  // scanned film.
  //
  // Suppose the mid-grey patch has R = 5000 and G = 10000. With film base
  // R = 10000 and film base G = 20000, the transmittance is 0.5 in both channels.
  //
  // A different captured film has R = 5000 (film base R = 10000) will have
  // transmittance of 0.5 in the R channel. Let's say the G layer is denser
  // because of flutation in the film development. G = 8000 (film base G = 16000)
  // will also have a transmittance of 0.5. However using the ICC profile which
  // assumes film base G is 20000 in the G channel, will treat the transmittance
  // as 8000 / 20000 = 0.4 and the color balance will be off.
  //
  // To compensate for this we will scale the G channel (with R as reference)
  // by (20000 / 10000) / (16000 / (10000) = 1.25.
  //
  // Of course the density even for R is not the same in the target and the
  // captured film. After adjusting the color balance, all channels can be
  // adjusted with a single scaling factor.
  float g_scale = 1.0f;
  float b_scale = 1.0f;
  if (cc_average_g > 0.0f && cc_average_r > 0.0f && cc_profile_r > 0.0f && cc_average_b > 0.0f) {
    g_scale = (cc_profile_g / cc_profile_r) / (cc_average_g / cc_average_r);
    b_scale = (cc_profile_b / cc_profile_r) / (cc_average_b / cc_average_r);
  }
  printf("Scale channels to match film base: %f %f %f\n", 1.0, g_scale, b_scale);

  const float sr = global_exposure_comp;
  const float sg = g_scale * global_exposure_comp * gain_g;
  const float sb = b_scale * global_exposure_comp * gain_b;

  for (int i = 0; i < 3; ++i) {
    merged_matrix[i]     = r_coef[i] * sr;
    merged_matrix[3 + i] = g_coef[i] * sg;
    merged_matrix[6 + i] = b_coef[i] * sb;
  }
  printf("R coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[0], merged_matrix[1], merged_matrix[2]);
  printf("G coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[3], merged_matrix[4], merged_matrix[5]);
  printf("B coefficients: %1.5f %1.5f %1.5f\n", merged_matrix[6], merged_matrix[7], merged_matrix[8]);
}

int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output, bool half_size) {
  unsigned* output_profile = NULL;
  unsigned profile_size = 0;
  bool dynamically_allocated = false;

  if (!attach_profile.empty()) {
    // If the profile to attach is not the psuedo "srgb" profile, the profile will be attached to the TIFF.
    // This is a hack to force LibRaw write the ICC profile in the TIFF without conversion.
    printf("Attaching profile: %s\n", attach_profile.c_str());
    if (attach_profile == "srgb") {
      output_profile = reinterpret_cast<unsigned int*>(sRGB_elle_V2_srgbtrc_icc);
      profile_size = sRGB_elle_V2_srgbtrc_icc_len;
    } else if (attach_profile == "srgb-g10" || attach_profile == "srgb-linear") {
      output_profile = reinterpret_cast<unsigned int*>(sRGB_elle_V2_g10_icc);
      profile_size = sRGB_elle_V2_g10_icc_len;
    } else {
      if (read_profile(attach_profile, &output_profile, &profile_size)) {
        return -1;
      }
      dynamically_allocated = true;
    }
  }

  const unsigned height = proc->imgdata.sizes.iheight;
  const unsigned width = proc->imgdata.sizes.iwidth;
  struct tiff_hdr header;
  if (half_size) {
    tiff_head(proc, &header, profile_size, width / 2, height / 2);
  } else {
    tiff_head(proc, &header, profile_size);
  }
  auto* fp = fopen(output.c_str(), "wb+");
  if (!fp) {
    fprintf(stderr, "ERROR! Cannot open %s for writing\n", output.c_str());
    if (dynamically_allocated && output_profile) free(output_profile);
    return -1;
  }
  fwrite(&header, sizeof(header), 1, fp);
  if (profile_size) {
    fwrite(output_profile, profile_size, 1, fp);
  }
  if (dynamically_allocated && output_profile) {
    free(output_profile);
  }

  const unsigned output_width = half_size ? width / 2 : width;
  std::vector<ushort> row_buf(output_width * 3);
  for (unsigned row = 0; row < height; ++row) {
    if (half_size && row % 2 == 0) {
      // Even number rows, reset row buffer.
      for (unsigned col = 0; col < output_width; ++col) {
        row_buf[col * 3] = (proc->imgdata.image[row * width + 2 * col][0] + 
                            proc->imgdata.image[row * width + 2 * col + 1][0]) / 4;
        row_buf[col * 3 + 1] = (proc->imgdata.image[row * width + 2 * col][1] +
                                proc->imgdata.image[row * width + 2 * col + 1][1]) / 4;
        row_buf[col * 3 + 2] = (proc->imgdata.image[row * width + 2 * col][2] +
                                proc->imgdata.image[row * width + 2 * col + 1][2]) / 4;
      }
    } else if (half_size && row % 2) {
      // Odd number rows, add to row buffer.
      for (unsigned col = 0; col < output_width; ++col) {
        row_buf[col * 3] += (proc->imgdata.image[row * width + 2 * col][0] + 
                             proc->imgdata.image[row * width + 2 * col + 1][0]) / 4;
        row_buf[col * 3 + 1] += (proc->imgdata.image[row * width + 2 * col][1] +
                                 proc->imgdata.image[row * width + 2 * col + 1][1]) / 4;
        row_buf[col * 3 + 2] += (proc->imgdata.image[row * width + 2 * col][2] +
                                 proc->imgdata.image[row * width + 2 * col + 1][2]) / 4;
      }
      fwrite(row_buf.data(), 3 * sizeof(ushort), output_width, fp);
    } else {
      for (unsigned col = 0; col < width; ++col) {
        row_buf[col * 3]     = proc->imgdata.image[row * width + col][0];
        row_buf[col * 3 + 1] = proc->imgdata.image[row * width + col][1];
        row_buf[col * 3 + 2] = proc->imgdata.image[row * width + col][2];
      }
      fwrite(row_buf.data(), 3 * sizeof(ushort), width, fp);
    }
  }
  fclose(fp);
  return 0;
}

int main(int ac, char *av[]) {
  argparse::ArgumentParser parser("neg_process");
  parser.add_argument("-H", "--half_size")
    .help("Half size.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-Q", "--quarter_size")
    .help("Quarter size.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-C", "--no_crop")
    .help("No cropping according to aspect ratio in RAW file.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-q", "--quality")
    .help("De-bayer quality. Not used in pixel-shift mode.")
    .scan<'i', int>()
    .default_value(0);
  parser.add_argument("-r", "--r_coeff")
    .help("R (corrected) value is dot product of this 'r1 r2 r3' vector and 'R G B' values from linear RAW.")
    .nargs(3)
    .default_value(std::vector<float>{1, 0, 0})
    .scan<'g', float>();
  parser.add_argument("-g", "--g_coeff")
    .help("G (corrected) value is dot product of this 'g1 g2 g3' vector and 'R G B' values from linear RAW.")
    .nargs(3)
    .default_value(std::vector<float>{0, 1, 0})
    .scan<'g', float>();
  parser.add_argument("-b", "--b_coeff")
    .help("B (corrected) value is dot product of this 'b1 b2 b3' vector and 'R G B' values from linear RAW.")
    .nargs(3)
    .default_value(std::vector<float>{0, 0, 1})
    .scan<'g', float>();
  parser.add_argument("-E", "--exposure_comp")
    .help("Multiplier for RGB values.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("-G", "--post_correction_gamma")
    .help("Apply a gamma to linear RGB after correction but before ICC profile is applied.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--knee")
    .help("Soft knee highlight compression threshold in [0, 1]. Defaults to 0.95 (>=1.0 disables).")
    .scan<'g', float>()
    .default_value(0.95f);
  parser.add_argument("--gain_g")
    .help("Additional gain multiplier for the G channel. Defaults to 1.0.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--gain_b")
    .help("Additional gain multiplier for the B channel. Defaults to 1.0.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--profile_film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from profile.")
    .nargs(3)
    .default_value(std::vector<int>{1, 1, 1})
    .scan<'i', int>();
  parser.add_argument("--film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from captured image.")
    .nargs(3)
    .default_value(std::vector<int>{1, 1, 1})
    .scan<'i', int>();
  parser.add_argument("-p", "--film_profile")
    .help("ICC Profile that applies to the corrected RGB values (See -r -g and -b flags). Consider this as the input ICC profile.");
  parser.add_argument("-P", "--colorspace")
    .help("srgb, srgb-g10 or [ICC profile path]. If specified the corrected RGB will be converted using this as the output profile.");
  parser.add_argument("-o", "--output")
    .required()
    .help("Output file location.");
  parser.add_argument("raw_files").nargs(1, 4);

  try {
    parser.parse_args(ac, av);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  const auto files = parser.get<std::vector<std::string>>("raw_files");
  auto r_coeff = parser.get<std::vector<float>>("--r_coeff");
  auto g_coeff = parser.get<std::vector<float>>("--g_coeff");
  auto b_coeff = parser.get<std::vector<float>>("--b_coeff");
  const float global_exposure_comp = parser.get<float>("--exposure_comp");
  const float knee = parser.get<float>("--knee");
  const float gain_g = parser.get<float>("--gain_g");
  const float gain_b = parser.get<float>("--gain_b");
  const float gamma = parser.get<float>("--post_correction_gamma");
  const bool has_gamma = parser.is_used("--post_correction_gamma") && (gamma > 0.0f) && (fabsf(gamma - 1.0f) > 1e-4f);
  const float inv_gamma = has_gamma ? (1.0f / gamma) : 1.0f;

  LibRaw *proc = nullptr;
  if (files.size() == 4) {
    proc = merge_pixel_shift_streaming(files);
    if (!proc) {
      fprintf(stderr, "ERROR! Failed to merge pixel-shift files\n");
      return 1;
    }
  } else {
    proc = load_raw(files[0], true,
                    parser.get<bool>("--half_size") || parser.get<bool>("--quarter_size"),
                    parser.get<int>("--quality"),
                    !parser.get<bool>("--no_crop"));
    if (!proc) {
      fprintf(stderr, "Cannot open %s\n", files[0].c_str());
      return 1;
    }
  }
  printf("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  printf("Shutter Speed: %f\n", proc->imgdata.other.shutter);

  // A conversion matrix is applied the linear image from RAW file to:
  // 1. Remove crosstalk between color channels.
  // 2. Scale the color channels independently such that mid-grey values are aligned.
  // 3. Scale the color channels independently to compensate for density difference
  //    between target and the captured film.
  // 4. Global scale factor for brightness.
  // 5. Scale the color channels independently according to user input (optional).
  // 6. Scale from 14-bit to 16-bit in pixel-shift mode (optional).
  //
  // A single matrix is combined with the above steps combined. Note that scaling is
  // always done after crosstalk correction, hence the scale factor can be applied
  // separately to the R, G and B coefficients.
  auto profile_film_base_rgb = parser.get<std::vector<int>>("--profile_film_base_rgb");
  auto film_base_rgb = parser.get<std::vector<int>>("--film_base_rgb");
  // Both --profile_film_base_rgb and --film_base_rgb need to be specified or will
  // give incorrect scaling.
  if (!parser.is_used("--profile_film_base_rgb") || !parser.is_used("--film_base_rgb")) {
    profile_film_base_rgb = film_base_rgb = std::vector{1, 1, 1};
  }

  float merged_matrix[9];
  adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                           global_exposure_comp,
                           gain_g, gain_b,
                           profile_film_base_rgb,
                           film_base_rgb,
                           merged_matrix);

  ParsedFilmProfile film_prof;
  if (parser.is_used("--film_profile")) {
    const std::string prof_path = parser.get<std::string>("--film_profile");
    if (!load_film_profile(prof_path, film_prof)) {
      fprintf(stderr, "ERROR! Cannot load film profile %s\n", prof_path.c_str());
    }
  }

  OutputEncoding encoding = ENCODE_NONE;
  std::string colorspace_str;
  std::string attach_profile;
  cmsHPROFILE hXYZ = nullptr;
  cmsHPROFILE hCustomOut = nullptr;

  if (parser.is_used("--colorspace")) {
    colorspace_str = parser.get<std::string>("--colorspace");
    if (colorspace_str == "srgb") {
      encoding = ENCODE_SRGB;
      attach_profile = "srgb";
      printf("Using elle sRGB TRC as output profile\n");
    } else if (colorspace_str == "srgb-g10" || colorspace_str == "srgb-linear") {
      encoding = ENCODE_SRGB_LINEAR;
      attach_profile = "srgb-g10";
      printf("Using elle sRGB Linear (Gamma = 1.0) as output profile\n");
    } else {
      std::string resolved_custom = resolve_profile_path(colorspace_str);
      hCustomOut = cmsOpenProfileFromFile(resolved_custom.c_str(), "r");
      if (!hCustomOut) {
        fprintf(stderr, "ERROR! Cannot read output ICC profile: %s\n", colorspace_str.c_str());
        delete proc;
        return 1;
      }
      hXYZ = cmsCreateXYZProfile();
      encoding = ENCODE_CUSTOM;
      attach_profile = resolved_custom;
      printf("Reading output ICC profile: %s\n", resolved_custom.c_str());
    }
  } else if (parser.is_used("--film_profile")) {
    encoding = ENCODE_NONE;
    attach_profile = parser.get<std::string>("--film_profile");
  }

  const int width = proc->imgdata.sizes.iwidth;
  const int height = proc->imgdata.sizes.iheight;
  const double t_start = omp_get_wtime();

  if (encoding == ENCODE_SRGB || encoding == ENCODE_SRGB_LINEAR) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        ushort* px = proc->imgdata.image[j * width + i];
        Rgb v = process_pixel(px, merged_matrix, knee, has_gamma, inv_gamma, film_prof.data);
        v = pcs_to_linear_srgb(v);
        v = encode_output(v, encoding);
        px[0] = quantize_u16(v.r);
        px[1] = quantize_u16(v.g);
        px[2] = quantize_u16(v.b);
      }
    }
  } else if (encoding == ENCODE_CUSTOM) {
    #pragma omp parallel
    {
      cmsHTRANSFORM thread_xform = cmsCreateTransform(
          hXYZ, TYPE_XYZ_FLT, hCustomOut, TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
      std::vector<float> xyz_row(width * 3);

      #pragma omp for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          const ushort* px = proc->imgdata.image[j * width + i];
          Rgb v = process_pixel(px, merged_matrix, knee, has_gamma, inv_gamma, film_prof.data);
          xyz_row[i * 3 + 0] = v.r;
          xyz_row[i * 3 + 1] = v.g;
          xyz_row[i * 3 + 2] = v.b;
        }
        cmsDoTransform(thread_xform, xyz_row.data(), proc->imgdata.image + j * width, width);
      }
      cmsDeleteTransform(thread_xform);
    }
    cmsCloseProfile(hXYZ);
    cmsCloseProfile(hCustomOut);
  } else {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        ushort* px = proc->imgdata.image[j * width + i];
        Rgb v = load_u16(px);
        v = apply_matrix(merged_matrix, v);
        v = apply_knee(v, knee);
        if (has_gamma) {
          v.r = powf(clamp01(v.r), inv_gamma);
          v.g = powf(clamp01(v.g), inv_gamma);
          v.b = powf(clamp01(v.b), inv_gamma);
        }
        px[0] = quantize_u16(v.r);
        px[1] = quantize_u16(v.g);
        px[2] = quantize_u16(v.b);
      }
    }
  }

  const double t_end = omp_get_wtime();
  printf("Processing time: %f\n", t_end - t_start);

  const auto output = parser.get<std::string>("--output");
  printf("Writing TIFF '%s'\n", output.c_str());
  int ret = write_tiff(proc, attach_profile, output,
                       // Multi-shot mode always gets the full size.
                       parser.get<bool>("--quarter_size") && files.size() == 1);
  delete proc;
  return ret;
}
