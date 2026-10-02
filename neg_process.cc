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
#include <unistd.h>
#include <lcms2.h>
#include <lcms2_plugin.h>
#include <omp.h>

#include <algorithm>
#include <cctype>
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
#include "dinov3_engine.h"
#include "neg_pipeline.h"
#include "dinov3_auto_solver.h"
#include "film_frame_detector.h"
#include "profile_bundle.h"

#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 14))
#error This code is for LibRaw 0.14+ only
#endif

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
  if (!parse_film_profile_pipeline(hprof, prof)) {
    fprintf(stderr, "ERROR! Film ICC profile %s has no AToB0 tag\n", resolved_path.c_str());
    cmsCloseProfile(hprof);
    return false;
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

inline bool str_contains_ci(const std::string& str, const std::string& sub) {
  auto it = std::search(
    str.begin(), str.end(),
    sub.begin(), sub.end(),
    [](char a, char b) { return std::tolower((unsigned char)a) == std::tolower((unsigned char)b); }
  );
  return it != str.end();
}

// 4-shot pixel shift merge is only available with Sony A7RM4 camera.
bool is_sony_a7rm4(LibRaw* proc) {
  if (!proc) return false;
  return str_contains_ci(proc->imgdata.idata.make, "Sony") &&
         str_contains_ci(proc->imgdata.idata.model, "7RM4");
}

// Checks if the camera has Sony Bayer RGGB CFA geometry.
bool is_sony_cfa_geometry(LibRaw* proc) {
  if (!proc || !proc->imgdata.idata.filters) return false;
  if (!str_contains_ci(proc->imgdata.idata.make, "Sony")) return false;
  // Standard RGGB Bayer CFA layout:
  // (0, 0)=R(0), (0, 1)=G(1 or 3), (1, 0)=G(1 or 3), (1, 1)=B(2)
  return proc->COLOR(0, 0) == 0 &&
         (proc->COLOR(0, 1) == 1 || proc->COLOR(0, 1) == 3) &&
         (proc->COLOR(1, 0) == 1 || proc->COLOR(1, 0) == 3) &&
         proc->COLOR(1, 1) == 2;
}

// -----------------------------------------------------------------------------
// Grain-Aware Debayering for Film Negative Digitization
//
// Traditional demosaicing algorithms (AHD, PPG, VNG, Hamilton-Adams) are designed
// for natural photographic scenes assuming strong spectral correlation across
// color channels and continuous geometric edges. They estimate missing channel
// samples by computing directional gradients across neighboring channels.
//
// When digitizing color film negatives, this assumption completely breaks down:
// 1. Film grain consists of stochastic, physically disjoint silver halide and dye
//    clouds in three separate emulsion layers (cyan, magenta, yellow). High-frequency
//    spatial variation is dominated by independent dye grain fluctuations rather
//    than scene edges.
// 2. The orange film mask requires significant color balance correction (typically
//    ~9.5x gain on the green channel relative to red/blue).
// 3. Standard edge-directed demosaicing mistakes random dye grain clumps in the red
//    channel for scene gradients and steers green/blue interpolation across them.
//    Under the dye-crosstalk matrix and mask balance gains, this phase misalignment
//    is heavily amplified, turning microscopic grain into test-pattern-like 2-4 px
//    chromatic mottle (color blotches and labyrinth artifacts).
//
// Bilinear demosaicing avoids chromatic mottle because channels are interpolated
// independently, but it blurs high-frequency grain, making scans look soft.
//
// The grain-aware debayer solves this by:
// 1. Reconstructing Green using a soft Laplacian gradient with an elevated epsilon
//    (eps = 4096.0f, calibrated to sensor noise floor) so random grain fluctuations
//    do not trigger directional steering, while retaining true scene edges.
// 2. Reconstructing Red and Blue in color-ratio space (R/(G+k), B/(G+k)) with
//    k = 16.0f. Because film transmission follows multiplicative dye absorption,
//    interpolating local color ratios preserves organic grain clump boundaries as
//    pure scalar intensity variations across all downstream matrix transforms,
//    completely eliminating chromatic mottle while preserving full grain acutance.
// 3. Falling back smoothly to bilinear interpolation near sensor highlight saturation
//    (within 64 DN of maximum white level) where color ratios become singular.
//
// Results vs 4-shot ground truth (Portra 400, Sony A7RM4):
// - Recovers 91.6%-92.3% edge acutance (+35-40% boost over plain bilinear).
// - Cuts false-color zippering by 22.3% and sky chroma mottle by 19.7% (~47 dB PSNR).
//
// Caveat: This works best with narrowband light (e.g. RGB LED); on whitelight
// the blue noise is too high to work well.
// -----------------------------------------------------------------------------

inline float get_px_refl(const uint16_t* raw_img, int r, int c, int h, int w, int pitch, int black) {
  int r_refl = (r < 0) ? -r : (r >= h ? 2 * (h - 1) - r : r);
  int c_refl = (c < 0) ? -c : (c >= w ? 2 * (w - 1) - c : c);
  r_refl = std::max(0, std::min(h - 1, r_refl));
  c_refl = std::max(0, std::min(w - 1, c_refl));
  int val = (int)raw_img[(size_t)r_refl * pitch + c_refl] - black;
  return (val > 0) ? (float)val : 0.0f;
}

inline float reconstruct_green_soft(const uint16_t* m, int r, int c, int h, int w, int pitch, int black, float white, float eps) {
  if ((r + c) & 1) {
    return get_px_refl(m, r, c, h, w, pitch, black);
  }
  float c_center = get_px_refl(m, r, c, h, w, pitch, black);
  float g_up     = get_px_refl(m, r - 1, c, h, w, pitch, black);
  float g_down   = get_px_refl(m, r + 1, c, h, w, pitch, black);
  float g_left   = get_px_refl(m, r, c - 1, h, w, pitch, black);
  float g_right  = get_px_refl(m, r, c + 1, h, w, pitch, black);

  float c_up2    = get_px_refl(m, r - 2, c, h, w, pitch, black);
  float c_down2  = get_px_refl(m, r + 2, c, h, w, pitch, black);
  float c_left2  = get_px_refl(m, r, c - 2, h, w, pitch, black);
  float c_right2 = get_px_refl(m, r, c + 2, h, w, pitch, black);

  float dv = std::abs(g_up - g_down) + std::abs(2.0f * c_center - c_up2 - c_down2);
  float dh = std::abs(g_left - g_right) + std::abs(2.0f * c_center - c_left2 - c_right2);

  float g_v = 0.5f * (g_up + g_down) + 0.25f * (2.0f * c_center - c_up2 - c_down2);
  float g_h = 0.5f * (g_left + g_right) + 0.25f * (2.0f * c_center - c_left2 - c_right2);

  float wv = 1.0f / (eps + dv * dv);
  float wh = 1.0f / (eps + dh * dh);
  float g_est = (wv * g_v + wh * g_h) / (wv + wh);
  return std::max(0.0f, std::min(white, g_est));
}

// Caveat: this works best with narrowband light, and on whitelight the blue noise is too high to work well.
bool debayer_grain_aware(LibRaw* proc, bool crop, const int* roi = nullptr) {
  if (!proc || !proc->imgdata.rawdata.raw_image) return false;

  const int src_w = proc->imgdata.sizes.width;
  const int src_h = proc->imgdata.sizes.height;
  const int pitch = proc->imgdata.sizes.raw_pitch / 2;
  const int black = proc->imgdata.color.black;
  int max_blk = black;
  for (int c = 0; c < 4; ++c) {
    max_blk = std::max(max_blk, black + (int)proc->imgdata.color.cblack[c]);
  }
  const auto* raw_img = proc->imgdata.rawdata.raw_image;
  const float white_level = (float)(proc->imgdata.color.maximum - max_blk);

#if LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21)
  const auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
#else
  const auto& crop_ref = proc->imgdata.sizes.raw_inset_crop;
#endif

  int rx = 0, ry = 0, rw = src_w, rh = src_h;
  if (roi && roi[2] > 0 && roi[3] > 0) {
    rx = std::max(0, std::min(roi[0], src_w - 1));
    ry = std::max(0, std::min(roi[1], src_h - 1));
    rw = std::max(1, std::min(roi[2], src_w - rx));
    rh = std::max(1, std::min(roi[3], src_h - ry));
  } else if (crop && (crop_ref.cleft || crop_ref.ctop)) {
    rx = crop_ref.cleft;
    ry = crop_ref.ctop;
    rw = crop_ref.cwidth;
    rh = crop_ref.cheight;
  }

  if (proc->imgdata.image) {
    proc->free_image();
  }
  proc->imgdata.image = (ushort (*)[4]) malloc((size_t)rw * rh * sizeof(*proc->imgdata.image));
  if (!proc->imgdata.image) {
    fprintf(stderr, "ERROR! Failed to allocate memory for debayered image (%dx%d)\n", rw, rh);
    return false;
  }

  const float k = 16.0f;
  const float eps = 4096.0f;

  #pragma omp parallel for schedule(static)
  for (int y = 0; y < rh; ++y) {
    int r = ry + y;
    bool r_even = ((r & 1) == 0);

    for (int x = 0; x < rw; ++x) {
      int c = rx + x;
      bool c_even = ((c & 1) == 0);

      float G = reconstruct_green_soft(raw_img, r, c, src_h, src_w, pitch, black, white_level, eps);
      float c_center = get_px_refl(raw_img, r, c, src_h, src_w, pitch, black);
      bool near_clip = (c_center >= (white_level - 64.0f)) || (G >= (white_level - 64.0f));

      float R = 0.0f;
      float B = 0.0f;

      if (near_clip) {
        if (r_even && c_even) {
          R = c_center;
          float b00 = get_px_refl(raw_img, r - 1, c - 1, src_h, src_w, pitch, black);
          float b01 = get_px_refl(raw_img, r - 1, c + 1, src_h, src_w, pitch, black);
          float b10 = get_px_refl(raw_img, r + 1, c - 1, src_h, src_w, pitch, black);
          float b11 = get_px_refl(raw_img, r + 1, c + 1, src_h, src_w, pitch, black);
          B = 0.25f * (b00 + b01 + b10 + b11);
        } else if (!r_even && !c_even) {
          B = c_center;
          float r00 = get_px_refl(raw_img, r - 1, c - 1, src_h, src_w, pitch, black);
          float r01 = get_px_refl(raw_img, r - 1, c + 1, src_h, src_w, pitch, black);
          float r10 = get_px_refl(raw_img, r + 1, c - 1, src_h, src_w, pitch, black);
          float r11 = get_px_refl(raw_img, r + 1, c + 1, src_h, src_w, pitch, black);
          R = 0.25f * (r00 + r01 + r10 + r11);
        } else if (r_even && !c_even) {
          R = 0.5f * (get_px_refl(raw_img, r, c - 1, src_h, src_w, pitch, black) + get_px_refl(raw_img, r, c + 1, src_h, src_w, pitch, black));
          B = 0.5f * (get_px_refl(raw_img, r - 1, c, src_h, src_w, pitch, black) + get_px_refl(raw_img, r + 1, c, src_h, src_w, pitch, black));
        } else {
          R = 0.5f * (get_px_refl(raw_img, r - 1, c, src_h, src_w, pitch, black) + get_px_refl(raw_img, r + 1, c, src_h, src_w, pitch, black));
          B = 0.5f * (get_px_refl(raw_img, r, c - 1, src_h, src_w, pitch, black) + get_px_refl(raw_img, r, c + 1, src_h, src_w, pitch, black));
        }
      } else {
        if (r_even && c_even) {
          R = c_center;
        } else if (r_even && !c_even) {
          float g_l = reconstruct_green_soft(raw_img, r, c - 1, src_h, src_w, pitch, black, white_level, eps);
          float g_r = reconstruct_green_soft(raw_img, r, c + 1, src_h, src_w, pitch, black, white_level, eps);
          float rl = (get_px_refl(raw_img, r, c - 1, src_h, src_w, pitch, black) + k) / (g_l + k);
          float rr = (get_px_refl(raw_img, r, c + 1, src_h, src_w, pitch, black) + k) / (g_r + k);
          R = (G + k) * (0.5f * (rl + rr)) - k;
        } else if (!r_even && c_even) {
          float g_u = reconstruct_green_soft(raw_img, r - 1, c, src_h, src_w, pitch, black, white_level, eps);
          float g_d = reconstruct_green_soft(raw_img, r + 1, c, src_h, src_w, pitch, black, white_level, eps);
          float ru = (get_px_refl(raw_img, r - 1, c, src_h, src_w, pitch, black) + k) / (g_u + k);
          float rd = (get_px_refl(raw_img, r + 1, c, src_h, src_w, pitch, black) + k) / (g_d + k);
          R = (G + k) * (0.5f * (ru + rd)) - k;
        } else {
          float g00 = reconstruct_green_soft(raw_img, r - 1, c - 1, src_h, src_w, pitch, black, white_level, eps);
          float g01 = reconstruct_green_soft(raw_img, r - 1, c + 1, src_h, src_w, pitch, black, white_level, eps);
          float g10 = reconstruct_green_soft(raw_img, r + 1, c - 1, src_h, src_w, pitch, black, white_level, eps);
          float g11 = reconstruct_green_soft(raw_img, r + 1, c + 1, src_h, src_w, pitch, black, white_level, eps);
          float r00 = (get_px_refl(raw_img, r - 1, c - 1, src_h, src_w, pitch, black) + k) / (g00 + k);
          float r01 = (get_px_refl(raw_img, r - 1, c + 1, src_h, src_w, pitch, black) + k) / (g01 + k);
          float r10 = (get_px_refl(raw_img, r + 1, c - 1, src_h, src_w, pitch, black) + k) / (g10 + k);
          float r11 = (get_px_refl(raw_img, r + 1, c + 1, src_h, src_w, pitch, black) + k) / (g11 + k);
          R = (G + k) * (0.25f * (r00 + r01 + r10 + r11)) - k;
        }

        if (!r_even && !c_even) {
          B = c_center;
        } else if (!r_even && c_even) {
          float g_l = reconstruct_green_soft(raw_img, r, c - 1, src_h, src_w, pitch, black, white_level, eps);
          float g_r = reconstruct_green_soft(raw_img, r, c + 1, src_h, src_w, pitch, black, white_level, eps);
          float bl = (get_px_refl(raw_img, r, c - 1, src_h, src_w, pitch, black) + k) / (g_l + k);
          float br = (get_px_refl(raw_img, r, c + 1, src_h, src_w, pitch, black) + k) / (g_r + k);
          B = (G + k) * (0.5f * (bl + br)) - k;
        } else if (r_even && !c_even) {
          float g_u = reconstruct_green_soft(raw_img, r - 1, c, src_h, src_w, pitch, black, white_level, eps);
          float g_d = reconstruct_green_soft(raw_img, r + 1, c, src_h, src_w, pitch, black, white_level, eps);
          float bu = (get_px_refl(raw_img, r - 1, c, src_h, src_w, pitch, black) + k) / (g_u + k);
          float bd = (get_px_refl(raw_img, r + 1, c, src_h, src_w, pitch, black) + k) / (g_d + k);
          B = (G + k) * (0.5f * (bu + bd)) - k;
        } else {
          float g00 = reconstruct_green_soft(raw_img, r - 1, c - 1, src_h, src_w, pitch, black, white_level, eps);
          float g01 = reconstruct_green_soft(raw_img, r - 1, c + 1, src_h, src_w, pitch, black, white_level, eps);
          float g10 = reconstruct_green_soft(raw_img, r + 1, c - 1, src_h, src_w, pitch, black, white_level, eps);
          float g11 = reconstruct_green_soft(raw_img, r + 1, c + 1, src_h, src_w, pitch, black, white_level, eps);
          float b00 = (get_px_refl(raw_img, r - 1, c - 1, src_h, src_w, pitch, black) + k) / (g00 + k);
          float b01 = (get_px_refl(raw_img, r - 1, c + 1, src_h, src_w, pitch, black) + k) / (g01 + k);
          float b10 = (get_px_refl(raw_img, r + 1, c - 1, src_h, src_w, pitch, black) + k) / (g10 + k);
          float b11 = (get_px_refl(raw_img, r + 1, c + 1, src_h, src_w, pitch, black) + k) / (g11 + k);
          B = (G + k) * (0.25f * (b00 + b01 + b10 + b11)) - k;
        }
      }

      ushort* dst = proc->imgdata.image[(size_t)y * rw + x];
      dst[0] = (ushort)std::round(std::max(0.0f, std::min(65535.0f, R)));
      dst[1] = (ushort)std::round(std::max(0.0f, std::min(65535.0f, G)));
      dst[2] = (ushort)std::round(std::max(0.0f, std::min(65535.0f, B)));
      dst[3] = 0;
    }
  }

  proc->imgdata.sizes.iwidth = rw;
  proc->imgdata.sizes.iheight = rh;
  proc->imgdata.idata.colors = 3;
  return true;
}

// Load a RAW file and decode it into linear values.
//
// If |debayer| is true, interpolation is performed to generate missing pixels
// from the color filter array (usually a bayer pattern). Otherwise, the linear
// RGB values will have missing pixels and will not be scaled to 16-bit. If the
// sensor produces 14-bit files, then a scale factor of 4 needs to be applied,
// this is needed only when merging pixel-shift images.
//
// |qual| chooses the debayer algorithm used. If qual < 0 (default), auto
// selection is performed: grain-aware demosaicing is used when Sony CFA geometry
// is detected, otherwise falling back to bilinear (qual = 0). Explicit qual >= 0
// forces LibRaw's built-in demosaic algorithms (0 is bilinear).
//
// When |crop| is false, the entire RAW file is used, disregarding aspect ratio
// and cropbox specified in the RAW metadata.
LibRaw* load_raw(const std::string& fn, bool debayer, bool half_size, int qual, bool crop,
                 bool crosstalk_specified = false, const int* roi = nullptr) {
  int ret;
  LibRaw* proc = new LibRaw();

  printf("Loading RAW file %s\n", fn.c_str());
  if ((ret = proc->open_file(fn.c_str())) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot open %s: %s\n", fn.c_str(), libraw_strerror(ret));
    delete proc;
    return NULL;
  }
  printf("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);

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

  if (roi) {
    if (roi[0] < 0 || roi[1] < 0 ||
        roi[0] >= (int)proc->imgdata.sizes.width || roi[1] >= (int)proc->imgdata.sizes.height ||
        roi[2] <= 0 || roi[3] <= 0) {
      fprintf(stderr, "ERROR! ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
              roi[0], roi[1], roi[2], roi[3], proc->imgdata.sizes.width, proc->imgdata.sizes.height);
      delete proc;
      return NULL;
    }
  }

  proc->imgdata.params.output_bps = 16;
  proc->imgdata.params.user_flip = 0;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.gamm[1] = 1;
  proc->imgdata.params.no_auto_bright = 1;
  proc->imgdata.params.no_auto_scale = 1;
  proc->imgdata.params.highlight = 1;
  proc->imgdata.params.output_tiff = 1;
  if (!debayer) {
    proc->imgdata.sizes.iwidth = proc->imgdata.sizes.width;
    proc->imgdata.sizes.iheight = proc->imgdata.sizes.height;
  } else {
    // When -r -g -b is specified, assume camera linear RGB (output_color = 0) regardless of debayer.
    // Otherwise fallback to Rec.2020 (output_color = 8).
    proc->imgdata.params.output_color = crosstalk_specified ? 0 : 8;

    bool use_grain_aware = (!half_size && qual < 0 && crosstalk_specified && is_sony_cfa_geometry(proc));
    if (use_grain_aware) {
      printf("Debayer quality: grain-aware\n");
      if (!debayer_grain_aware(proc, crop, roi)) {
        fprintf(stderr, "Grain-aware debayer failed, falling back to bilinear\n");
        use_grain_aware = false;
      }
    }
    if (!use_grain_aware) {
      int effective_qual = (qual < 0) ? 0 : qual;
      printf("Debayer quality: %d\n", effective_qual);
      proc->imgdata.params.half_size = half_size;
      proc->imgdata.params.user_qual = effective_qual;
      proc->imgdata.params.use_auto_wb = 0;
      proc->imgdata.params.user_mul[0] = 1;
      proc->imgdata.params.user_mul[1] = 1;
      proc->imgdata.params.user_mul[2] = 1;
      proc->imgdata.params.user_mul[3] = 1;

      if (roi && roi[2] > 0 && roi[3] > 0) {
        int rx = std::max(0, std::min(roi[0], (int)proc->imgdata.sizes.width - 1));
        int ry = std::max(0, std::min(roi[1], (int)proc->imgdata.sizes.height - 1));
        int rw = std::max(1, std::min(roi[2], (int)proc->imgdata.sizes.width - rx));
        int rh = std::max(1, std::min(roi[3], (int)proc->imgdata.sizes.height - ry));
        proc->imgdata.params.cropbox[0] = rx;
        proc->imgdata.params.cropbox[1] = ry;
        proc->imgdata.params.cropbox[2] = rw;
        proc->imgdata.params.cropbox[3] = rh;
      } else {
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
      }
      proc->dcraw_process();
    }
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
  const int fw = frame->imgdata.sizes.width;
  const int fh = frame->imgdata.sizes.height;
  const ushort* raw_img = frame->imgdata.rawdata.raw_image;
  const int pitch = frame->imgdata.sizes.raw_pitch / 2;
  const int top = frame->imgdata.sizes.top_margin;
  const int left = frame->imgdata.sizes.left_margin;
  const int black = frame->imgdata.color.black;
  int cblk[4];
  for (int i = 0; i < 4; ++i) cblk[i] = black + (int)frame->imgdata.color.cblack[i];

  #pragma omp parallel for schedule(static)
  for (int r = 0; r < fh - 1; ++r) {
    const size_t raw_row = (size_t)(r + top) * pitch + left;
    for (int c = 1; c < fw; ++c) {
      const int col = frame->COLOR(r, c);
      int raw_val = (int)raw_img[raw_row + c] - cblk[col];
      const ushort v = (raw_val > 0) ? (ushort)raw_val : 0;
      ushort* dst = base->imgdata.image[(size_t)(r + dr) * bw + (c + dc)];
      if (col & 1)
        dst[1] = (mi < 2) ? v : (ushort)((v + dst[1]) / 2);
      else
        dst[col] = v;
    }
  }
}

// Merge 4 RAW files from Sony pixel-shift into camera linear RGB, loading one frame at a time to save memory.
LibRaw* merge_pixel_shift_streaming(const std::vector<std::string>& files) {
  if (files.size() < 4) return nullptr;
  printf("Merging 4 images...\n");
  LibRaw* base = load_raw(files[0], false, false, 0, false);
  if (!base) return nullptr;
  if (!is_sony_a7rm4(base)) {
    fprintf(stderr, "ERROR: 4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
            base->imgdata.idata.make, base->imgdata.idata.model);
    base->recycle();
    delete base;
    return nullptr;
  }

  const int bw = base->imgdata.sizes.width;
  const int bh = base->imgdata.sizes.height;
  base->imgdata.sizes.iwidth = bw;
  base->imgdata.sizes.iheight = bh;
  base->imgdata.image = (ushort (*)[4]) calloc((size_t)bw * bh, sizeof(*base->imgdata.image));
  if (!base->imgdata.image) {
    fprintf(stderr, "ERROR: Failed to allocate memory for pixel shift merge\n");
    base->recycle();
    delete base;
    return nullptr;
  }

  const ushort* raw0 = base->imgdata.rawdata.raw_image;
  const int pitch0 = base->imgdata.sizes.raw_pitch / 2;
  const int top0 = base->imgdata.sizes.top_margin;
  const int left0 = base->imgdata.sizes.left_margin;
  const int blk0 = base->imgdata.color.black;
  int cblk0[4];
  for (int i = 0; i < 4; ++i) cblk0[i] = blk0 + (int)base->imgdata.color.cblack[i];

  #pragma omp parallel for schedule(static)
  for (int r = 0; r < bh; ++r) {
    const size_t raw_row = (size_t)(r + top0) * pitch0 + left0;
    for (int c = 0; c < bw; ++c) {
      int col = base->COLOR(r, c);
      int v = (int)raw0[raw_row + c] - cblk0[col];
      base->imgdata.image[(size_t)r * bw + c][col] = (v > 0) ? (ushort)v : 0;
    }
  }

  merge_pixel_shift_frame(base, base, 0);
  for (int mi = 1; mi < 4; ++mi) {
    LibRaw* frame = load_raw(files[mi], false, false, 0, false);
    if (!frame) {
      base->recycle();
      delete base;
      return nullptr;
    }
    if (!is_sony_a7rm4(frame)) {
      fprintf(stderr, "ERROR: 4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
              frame->imgdata.idata.make, frame->imgdata.idata.model);
      frame->recycle();
      delete frame;
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

// Extract a sub-rectangle from decoded LibRaw image buffer with rotation and flip.
bool extract_roi_and_transform(LibRaw* proc, int rx, int ry, int rw, int rh,
                               int rot_cw, bool hflip, bool vflip) {
  if (!proc || !proc->imgdata.image) return false;

  const int cur_w = proc->imgdata.sizes.iwidth;
  const int cur_h = proc->imgdata.sizes.iheight;
  rx = std::max(0, std::min(rx, cur_w - 1));
  ry = std::max(0, std::min(ry, cur_h - 1));
  rw = std::max(1, std::min(rw, cur_w - rx));
  rh = std::max(1, std::min(rh, cur_h - ry));

  rot_cw = ((rot_cw % 360) + 360) % 360;

  // 180-degree rotation is equivalent to flipping both axes.
  if (rot_cw == 180) {
    hflip = !hflip;
    vflip = !vflip;
    rot_cw = 0;
  }

  const bool swap_dims = (rot_cw == 90 || rot_cw == 270);
  const int dst_w = swap_dims ? rh : rw;
  const int dst_h = swap_dims ? rw : rh;

  if (rx == 0 && ry == 0 && rw == cur_w && rh == cur_h && rot_cw == 0 && !hflip && !vflip) {
    return true;
  }

  ushort (*new_image)[4] = (ushort (*)[4]) malloc((size_t)dst_w * dst_h * sizeof(*new_image));
  if (!new_image) {
    fprintf(stderr, "ERROR! Failed to allocate memory for transformed ROI (%dx%d)\n", dst_w, dst_h);
    return false;
  }

  const ushort (*src_image)[4] = proc->imgdata.image;

  if (rot_cw == 0 && !hflip && !vflip) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < rh; ++y) {
      memcpy(&new_image[(size_t)y * dst_w],
             &src_image[(size_t)(ry + y) * cur_w + rx],
             (size_t)rw * sizeof(*new_image));
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < rh; ++y) {
      for (int x = 0; x < rw; ++x) {
        int x1 = (rot_cw == 0)  ? x : ((rot_cw == 90) ? (rh - 1 - y) : y);
        int y1 = (rot_cw == 0)  ? y : ((rot_cw == 90) ? x : (rw - 1 - x));
        int dx = hflip ? (dst_w - 1 - x1) : x1;
        int dy = vflip ? (dst_h - 1 - y1) : y1;

        const ushort* s = src_image[(size_t)(ry + y) * cur_w + (rx + x)];
        ushort* d = new_image[(size_t)dy * dst_w + dx];
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = s[3];
      }
    }
  }

  proc->free_image();
  proc->imgdata.image = new_image;
  proc->imgdata.sizes.iwidth = dst_w;
  proc->imgdata.sizes.iheight = dst_h;
  return true;
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

int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc = {}) {
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
  } else if (!embedded_icc.empty()) {
    printf("Attaching profile: [bundle embedded ICC] (%u bytes)\n", (unsigned)embedded_icc.size());
    output_profile = reinterpret_cast<unsigned int*>(const_cast<uint8_t*>(embedded_icc.data()));
    profile_size = (unsigned)embedded_icc.size();
  }

  const unsigned height = proc->imgdata.sizes.iheight;
  const unsigned width = proc->imgdata.sizes.iwidth;
  struct tiff_hdr header;
  tiff_head(proc, &header, profile_size);
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

  std::vector<ushort> row_buf(width * 3);
  for (unsigned row = 0; row < height; ++row) {
    for (unsigned col = 0; col < width; ++col) {
      row_buf[col * 3]     = proc->imgdata.image[row * width + col][0];
      row_buf[col * 3 + 1] = proc->imgdata.image[row * width + col][1];
      row_buf[col * 3 + 2] = proc->imgdata.image[row * width + col][2];
    }
    fwrite(row_buf.data(), 3 * sizeof(ushort), width, fp);
  }
  fclose(fp);
  return 0;
}

#ifndef NEGICC_GIT_VERSION
#define NEGICC_GIT_VERSION "master"
#endif

static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int)c);
          out += buf;
        } else {
          out += (char)c;
        }
        break;
    }
  }
  return out;
}

static std::string json_num(double val, const char* fmt = "%.6g") {
  if (!std::isfinite(val)) return "null";
  char buf[64];
  snprintf(buf, sizeof(buf), fmt, val);
  return buf;
}

static bool write_json_sidecar(
    const std::string& output_path,
    int ac, char* av[],
    const LibRaw* proc,
    const std::vector<std::string>& files,
    int orig_w, int orig_h,
    bool half_size,
    bool has_roi, const int* roi_coords,
    const std::string& roi_source,
    const FrameDetection* frame_det,
    int rot_cw, bool hflip, bool vflip,
    int out_w, int out_h,
    bool film_base_applied,
    const std::vector<int>& film_base_rgb,
    const std::vector<int>& profile_film_base_rgb,
    const std::string& film_base_source,
    float std_density,
    bool has_std_density,
    const std::string& film_profile_path,
    const float* merged_matrix,
    float exposure_comp, float gain_g, float gain_b,
    unsigned pin_mask,
    float knee, const std::string& knee_source,
    bool has_dino, const DinoSolveResult& dino_res,
    const std::string& dino_model_path,
    const std::string& dino_backend,
    const std::string& colorspace_name,
    double t_raw_decode_ms,
    double t_frame_det_ms,
    double t_full_conversion_ms,
    double t_tiff_write_ms,
    bool is_bundle = false,
    const std::string& bundle_target_name = "",
    int bundle_target_idx = -1) {
  const std::string json_path = output_path + ".json";
  const std::string tmp_path = json_path + ".tmp";
  FILE* fp = fopen(tmp_path.c_str(), "w");
  if (!fp) {
    fprintf(stderr, "Warning: Cannot open sidecar file %s for writing\n", tmp_path.c_str());
    return false;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"schema\": \"negicc.neg_process.sidecar/1\",\n");
  fprintf(fp, "  \"tool\": {\"name\": \"neg_process\", \"version\": \"%s\"},\n", json_escape(NEGICC_GIT_VERSION).c_str());

  // argv
  fprintf(fp, "  \"argv\": [");
  for (int i = 0; i < ac; ++i) {
    fprintf(fp, "%s\"%s\"", (i > 0 ? ", " : ""), json_escape(av[i]).c_str());
  }
  fprintf(fp, "],\n");

  // inputs
  std::string camera = "Unknown";
  if (proc) {
    std::string make = proc->imgdata.idata.make;
    std::string model = proc->imgdata.idata.model;
    if (!make.empty() && !model.empty()) {
      if (model.find(make) == 0) camera = model;
      else camera = make + " " + model;
    } else if (!model.empty()) camera = model;
    else if (!make.empty()) camera = make;
  }
  fprintf(fp, "  \"inputs\": {\"raw_files\": [");
  for (size_t i = 0; i < files.size(); ++i) {
    fprintf(fp, "%s\"%s\"", (i > 0 ? ", " : ""), json_escape(files[i]).c_str());
  }
  fprintf(fp, "], \"camera\": \"%s\", \"iso\": %s, \"shutter\": %s},\n",
          json_escape(camera).c_str(),
          proc ? json_num(proc->imgdata.other.iso_speed).c_str() : "0",
          proc ? json_num(proc->imgdata.other.shutter).c_str() : "0");

  // geometry
  fprintf(fp, "  \"geometry\": {\n");
  fprintf(fp, "    \"sensor_size\": [%d, %d], \"half_size\": %s,\n",
          orig_w, orig_h, half_size ? "true" : "false");
  fprintf(fp, "    \"roi_source\": \"%s\", ", json_escape(roi_source).c_str());
  if (has_roi) {
    fprintf(fp, "\"roi\": [%d, %d, %d, %d],\n", roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]);
  } else {
    fprintf(fp, "\"roi\": null,\n");
  }
  fprintf(fp, "    \"rot_cw\": %d, \"hflip\": %s, \"vflip\": %s,\n",
          rot_cw, hflip ? "true" : "false", vflip ? "true" : "false");
  fprintf(fp, "    \"output_size\": [%d, %d]\n", out_w, out_h);
  fprintf(fp, "  },\n");

  // frame_detection
  if (frame_det) {
    const int s_scale = (half_size && files.size() != 4) ? 2 : 1;
    fprintf(fp, "  \"frame_detection\": {\n");
    fprintf(fp, "    \"axis\": \"%c\", \"confidence\": %s, \"reason\": \"%s\",\n",
            frame_det->axis, json_num(frame_det->conf, "%.2f").c_str(), json_escape(frame_det->reason).c_str());
    fprintf(fp, "    \"gaps\": [");
    int g_cnt = 0;
    for (int g = 0; g < 2; ++g) {
      if (frame_det->gap[g][0] >= 0) {
        if (g_cnt > 0) fprintf(fp, ", ");
        fprintf(fp, "[%d, %d]", frame_det->gap[g][0] * s_scale, frame_det->gap[g][1] * s_scale);
        g_cnt++;
      }
    }
    fprintf(fp, "], \"segments\": %d, \"n_gaps\": %d, \"selected\": %d,\n",
            frame_det->n_segments, frame_det->n_gaps, frame_det->selected);
    fprintf(fp, "    \"aspect\": %s, \"format\": \"%s\"\n",
            json_num(frame_det->aspect, "%.4g").c_str(), json_escape(frame_det->format).c_str());
    fprintf(fp, "  },\n");
  } else {
    fprintf(fp, "  \"frame_detection\": null,\n");
  }

  // film_base
  fprintf(fp, "  \"film_base\": {\n");
  fprintf(fp, "    \"source\": \"%s\", \"applied\": %s,\n",
          json_escape(film_base_source).c_str(),
          film_base_applied ? "true" : "false");
  if (film_base_source == "none") {
    fprintf(fp, "    \"film_base_rgb\": null, \"std_density\": null,\n");
  } else {
    fprintf(fp, "    \"film_base_rgb\": [%d, %d, %d], \"std_density\": %s,\n",
            film_base_rgb[0], film_base_rgb[1], film_base_rgb[2],
            (film_base_source == "rebate" && has_std_density) ? json_num(std_density, "%.6g").c_str() : "null");
  }
  fprintf(fp, "    \"profile_film_base_rgb\": [%d, %d, %d]\n",
          profile_film_base_rgb[0], profile_film_base_rgb[1], profile_film_base_rgb[2]);
  fprintf(fp, "  },\n");

  // profile
  if (!film_profile_path.empty()) {
    if (is_bundle) {
      fprintf(fp, "  \"profile\": {\"path\": \"%s\", \"type\": \"bundle\", \"bundle_target\": \"%s\"},\n",
              json_escape(film_profile_path).c_str(), json_escape(bundle_target_name).c_str());
    } else {
      fprintf(fp, "  \"profile\": {\"path\": \"%s\", \"type\": \"icc\", \"bundle_target\": null},\n",
              json_escape(film_profile_path).c_str());
    }
  } else {
    fprintf(fp, "  \"profile\": null,\n");
  }

  // matrix
  fprintf(fp, "  \"matrix\": {\"merged\": [");
  for (int i = 0; i < 9; ++i) {
    fprintf(fp, "%s%s", (i > 0 ? ", " : ""), json_num(merged_matrix[i], "%.9g").c_str());
  }
  fprintf(fp, "],\n");
  fprintf(fp, "             \"exposure_comp\": %s, \"gain_g\": %s, \"gain_b\": %s,\n",
          json_num(exposure_comp, "%.9g").c_str(),
          json_num(gain_g, "%.9g").c_str(),
          json_num(gain_b, "%.9g").c_str());
  fprintf(fp, "             \"pins\": {\"e\": %s, \"g\": %s, \"b\": %s}},\n",
          (pin_mask & PIN_E) ? "true" : "false",
          (pin_mask & PIN_G) ? "true" : "false",
          (pin_mask & PIN_B) ? "true" : "false");

  // cli_equivalent
  fprintf(fp, "  \"cli_equivalent\": [\"-E\", \"%s\", \"--gain_g\", \"%s\", \"--gain_b\", \"%s\", \"--knee\", \"%s\"",
          json_num(exposure_comp, "%.9g").c_str(),
          json_num(gain_g, "%.9g").c_str(),
          json_num(gain_b, "%.9g").c_str(),
          json_num(knee, "%.9g").c_str());
  if (has_roi) {
    fprintf(fp, ", \"--roi\", \"%d\", \"%d\", \"%d\", \"%d\", \"%d\", \"%d\", \"%d\"",
            roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3],
            rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
  }
  if (film_base_applied) {
    fprintf(fp, ", \"--film_base_rgb\", \"%d\", \"%d\", \"%d\"",
            film_base_rgb[0], film_base_rgb[1], film_base_rgb[2]);
    fprintf(fp, ", \"--profile_film_base_rgb\", \"%d\", \"%d\", \"%d\"",
            profile_film_base_rgb[0], profile_film_base_rgb[1], profile_film_base_rgb[2]);
  }
  if (is_bundle && bundle_target_idx >= 0) {
    fprintf(fp, ", \"--target\", \"%d\"", bundle_target_idx);
  }
  fprintf(fp, "],\n");

  // dino
  if (has_dino) {
    fprintf(fp, "  \"dino\": {\n");
    fprintf(fp, "    \"model\": \"%s\", \"backend\": \"%s\", \"grid\": [%d, %d], \"target\": [%d, %d],\n",
            json_escape(dino_model_path).c_str(), json_escape(dino_backend).c_str(),
            dino_res.grid_w, dino_res.grid_h, dino_res.target_w, dino_res.target_h);
    fprintf(fp, "    \"e_center\": %s, \"e_bounds\": [%s, %s], \"status\": \"%s\", \"best_iteration\": %d, \"inferences\": %d,\n",
            json_num(dino_res.e_center).c_str(),
            json_num(dino_res.bounds_lo).c_str(),
            json_num(dino_res.bounds_hi).c_str(),
            json_escape(dino_res.status).c_str(), dino_res.best_iteration, dino_res.total_inferences);
    fprintf(fp, "    \"solved_relative\": {\"e\": %s, \"g\": %s, \"b\": %s},\n",
            json_num(dino_res.solved_relative.e).c_str(),
            json_num(dino_res.solved_relative.g).c_str(),
            json_num(dino_res.solved_relative.b).c_str());

    if (!dino_res.iterations.empty() && dino_res.iterations[0].tag == 'b') {
      const auto& b = dino_res.iterations[0];
      fprintf(fp, "    \"bootstrap\": {\"e\": %s, \"g\": %s, \"b\": %s, \"loss_L\": %s, \"loss_a\": %s, \"loss_b\": %s, "
                  "\"clip_pct\": %s, \"penalty_clip\": %s, \"penalty_anchor\": %s, \"objective\": %s},\n",
              json_num(b.e).c_str(), json_num(b.g).c_str(), json_num(b.b).c_str(),
              json_num(b.loss_L).c_str(), json_num(b.loss_a).c_str(), json_num(b.loss_b).c_str(),
              json_num(b.clip_pct).c_str(), json_num(b.penalty_clip).c_str(),
              json_num(b.penalty_anchor).c_str(), json_num(b.objective).c_str());
    } else {
      fprintf(fp, "    \"bootstrap\": null,\n");
    }

    fprintf(fp, "    \"iterations\": [");
    bool first_it = true;
    for (size_t i = 0; i < dino_res.iterations.size(); ++i) {
      if (dino_res.iterations[i].tag == 'b') continue;
      const auto& it = dino_res.iterations[i];
      if (!first_it) fprintf(fp, ",\n      ");
      else fprintf(fp, "\n      ");
      first_it = false;
      fprintf(fp, "{\"e\": %s, \"g\": %s, \"b\": %s, \"residual\": %s, \"loss_L\": %s, \"loss_a\": %s, \"loss_b\": %s, "
                  "\"clip_pct\": %s, \"penalty_clip\": %s, \"penalty_anchor\": %s, \"objective\": %s}",
              json_num(it.e).c_str(), json_num(it.g).c_str(), json_num(it.b).c_str(),
              json_num(it.residual).c_str(), json_num(it.loss_L).c_str(), json_num(it.loss_a).c_str(),
              json_num(it.loss_b).c_str(), json_num(it.clip_pct).c_str(),
              json_num(it.penalty_clip).c_str(), json_num(it.penalty_anchor).c_str(),
              json_num(it.objective).c_str());
    }
    if (!first_it) fprintf(fp, "\n    ");
    fprintf(fp, "],\n");

    fprintf(fp, "    \"tone_mass\": [");
    for (size_t i = 0; i < dino_res.tone_mass.size(); ++i) {
      fprintf(fp, "%s%s", (i > 0 ? ", " : ""), json_num(dino_res.tone_mass[i]).c_str());
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  },\n");
  } else {
    fprintf(fp, "  \"dino\": null,\n");
  }

  // tone
  fprintf(fp, "  \"tone\": {\"knee\": %s, \"knee_source\": \"%s\", ",
          json_num(knee, "%.9g").c_str(), json_escape(knee_source).c_str());
  if (has_dino && dino_res.dynamic_knee.active) {
    fprintf(fp, "\"q995\": %s, \"knee_fit\": %s},\n",
            json_num(dino_res.dynamic_knee.q995).c_str(),
            json_num(dino_res.dynamic_knee.knee_fit).c_str());
  } else {
    fprintf(fp, "\"q995\": null, \"knee_fit\": null},\n");
  }

  // output
  fprintf(fp, "  \"output\": {\"path\": \"%s\", \"colorspace\": \"%s\", \"bits\": 16},\n",
          json_escape(output_path).c_str(), json_escape(colorspace_name).c_str());

  // timings_ms
  fprintf(fp, "  \"timings_ms\": {\"raw_decode\": %s, \"frame_detection\": %s, ",
          json_num(t_raw_decode_ms, "%.1f").c_str(),
          frame_det ? json_num(t_frame_det_ms, "%.1f").c_str() : "null");
  if (has_dino) {
    fprintf(fp, "\"dino_views\": %s, \"dino_render\": %s, \"dino_inference\": %s, \"dino_search\": %s, ",
            json_num(dino_res.timing_views_ms, "%.1f").c_str(),
            json_num(dino_res.timing_render_ms, "%.1f").c_str(),
            json_num(dino_res.timing_infer_ms, "%.1f").c_str(),
            json_num(dino_res.timing_search_ms, "%.1f").c_str());
  } else {
    fprintf(fp, "\"dino_views\": null, \"dino_render\": null, \"dino_inference\": null, \"dino_search\": null, ");
  }
  fprintf(fp, "\"full_conversion\": %s, \"tiff_write\": %s}\n",
          json_num(t_full_conversion_ms, "%.1f").c_str(),
          json_num(t_tiff_write_ms, "%.1f").c_str());

  fprintf(fp, "}\n");
  const bool write_ok = (ferror(fp) == 0);
  const bool close_ok = (fclose(fp) == 0);
  if (!write_ok || !close_ok) {
    fprintf(stderr, "Warning: Failed to flush/close sidecar %s\n", tmp_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }

  if (rename(tmp_path.c_str(), json_path.c_str()) != 0) {
    fprintf(stderr, "Warning: Failed to rename sidecar from %s to %s\n", tmp_path.c_str(), json_path.c_str());
    unlink(tmp_path.c_str());
    return false;
  }
  return true;
}

int main(int ac, char *av[]) {
  argparse::ArgumentParser parser("neg_process");
  parser.add_argument("-H", "--half_size")
    .help("Half size.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-C", "--no_crop")
    .help("No cropping according to aspect ratio in RAW file.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-q", "--quality")
    .help("De-bayer quality (-1 for auto: grain-aware if Sony CFA, else bilinear; >=0 passes qual to LibRaw).")
    .scan<'i', int>()
    .default_value(-1);
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
    .help("Soft knee highlight compression threshold in [0, 1], or 'auto' (requires --dino). Defaults to 0.95 (>=1.0 disables).")
    .default_value(std::string("0.95"));
  parser.add_argument("--no-metadata")
    .help("Disable automatic JSON metadata sidecar generation.")
    .default_value(false)
    .implicit_value(true);
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
    .help("Linear (uncorrected) R G B values of the film base from captured image, or 'auto'.")
    .nargs(1, 3);
  parser.add_argument("-p", "--film_profile")
    .help("ICC Profile that applies to the corrected RGB values (See -r -g and -b flags). Consider this as the input ICC profile.");
  parser.add_argument("-P", "--colorspace")
    .help("srgb, srgb-g10 or [ICC profile path]. If specified the corrected RGB will be converted using this as the output profile.");
  parser.add_argument("--roi")
    .help("ROI crop in sensor coordinates: x y w h [rot_cw [hflip [vflip]]], or 'auto [h|v]'.")
    .nargs(1, 7);
  parser.add_argument("--rot", "--rot_cw")
    .help("Clockwise rotation in degrees (0, 90, 180, 270).")
    .scan<'i', int>()
    .default_value(0);
  parser.add_argument("--hflip")
    .help("Horizontal flip.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--vflip")
    .help("Vertical flip.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-o", "--output")
    .required()
    .help("Output file location.");
  parser.add_argument("--dino")
    .help("Path to DINOv3 model directory or file (containing model.onnx or model.engine) for photographic intent inference.");
  parser.add_argument("--dino_backend")
    .help("DINOv3 inference backend: auto, cpu, or gpu (defaults to auto).")
    .default_value(std::string("auto"));
  parser.add_argument("--dino_iters")
    .help("Maximum iterations for DINOv3 parameter solving (defaults to 4).")
    .scan<'i', int>()
    .default_value(4);
  parser.add_argument("--target", "--bundle_target")
    .help("Target index (0-based) or target name in JSON profile bundle.")
    .default_value(std::string(""));
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
  float global_exposure_comp = parser.get<float>("--exposure_comp");
  const std::string knee_str = parser.get<std::string>("--knee");
  const bool is_knee_auto = (knee_str == "auto");
  float knee = 0.95f;
  if (is_knee_auto) {
    if (!parser.is_used("--dino")) {
      fprintf(stderr, "ERROR! --knee auto requires --dino\n");
      return 1;
    }
  } else {
    char* end = nullptr;
    const float val = strtof(knee_str.c_str(), &end);
    if (!end || *end != '\0' || end == knee_str.c_str() || !std::isfinite(val) || val <= 0.0f) {
      fprintf(stderr, "ERROR! Invalid --knee argument '%s'\n", knee_str.c_str());
      return 1;
    }
    knee = val;
  }
  float gain_g = parser.get<float>("--gain_g");
  float gain_b = parser.get<float>("--gain_b");
  const float gamma = parser.get<float>("--post_correction_gamma");
  const bool has_gamma = parser.is_used("--post_correction_gamma") && (gamma > 0.0f) && (fabsf(gamma - 1.0f) > 1e-4f);
  const float inv_gamma = has_gamma ? (1.0f / gamma) : 1.0f;

  int rot_cw = 0;
  bool hflip = false;
  bool vflip = false;
  int roi_coords[4] = {0, 0, 0, 0};
  bool has_roi = false;
  bool is_roi_auto = false;
  char roi_axis_hint = 0;

  if (parser.is_used("--roi")) {
    const auto roi_tokens = parser.get<std::vector<std::string>>("--roi");
    if (roi_tokens.empty()) {
      fprintf(stderr, "ERROR! --roi requires arguments\n");
      return 1;
    }
    if (roi_tokens[0] == "auto") {
      is_roi_auto = true;
      if (roi_tokens.size() > 1) {
        if (roi_tokens[1] == "h" || roi_tokens[1] == "H") {
          roi_axis_hint = 'h';
        } else if (roi_tokens[1] == "v" || roi_tokens[1] == "V") {
          roi_axis_hint = 'v';
        } else {
          fprintf(stderr, "ERROR! Invalid --roi auto axis hint '%s' (must be 'h' or 'v')\n", roi_tokens[1].c_str());
          return 1;
        }
      }
      if (roi_tokens.size() > 2) {
        fprintf(stderr, "ERROR! Too many arguments for --roi auto\n");
        return 1;
      }
    } else {
      if (roi_tokens.size() < 4) {
        fprintf(stderr, "ERROR! --roi requires at least 4 arguments: x y w h [rot_cw [hflip [vflip]]]\n");
        return 1;
      }
      try {
        roi_coords[0] = std::stoi(roi_tokens[0]);
        roi_coords[1] = std::stoi(roi_tokens[1]);
        roi_coords[2] = std::stoi(roi_tokens[2]);
        roi_coords[3] = std::stoi(roi_tokens[3]);
        if (roi_tokens.size() >= 5) rot_cw = std::stoi(roi_tokens[4]);
        if (roi_tokens.size() >= 6) hflip = (std::stoi(roi_tokens[5]) != 0);
        if (roi_tokens.size() >= 7) vflip = (std::stoi(roi_tokens[6]) != 0);
        has_roi = true;
      } catch (const std::exception& e) {
        fprintf(stderr, "ERROR! Invalid integer argument for --roi\n");
        return 1;
      }
    }
  }

  if (parser.is_used("--rot")) {
    rot_cw = parser.get<int>("--rot");
  } else if (parser.is_used("--rot_cw")) {
    rot_cw = parser.get<int>("--rot_cw");
  }
  if (parser.get<bool>("--hflip")) hflip = true;
  if (parser.get<bool>("--vflip")) vflip = true;

  if (rot_cw % 90 != 0) {
    fprintf(stderr, "ERROR! Rotation must be a multiple of 90 degrees (got %d)\n", rot_cw);
    return 1;
  }

  bool is_film_base_auto = false;
  bool film_base_cli_used = false;
  std::vector<int> user_film_base_rgb = {1, 1, 1};

  if (parser.is_used("--film_base_rgb")) {
    const auto fb_tokens = parser.get<std::vector<std::string>>("--film_base_rgb");
    if (fb_tokens.size() == 1 && fb_tokens[0] == "auto") {
      is_film_base_auto = true;
    } else if (fb_tokens.size() == 3) {
      try {
        user_film_base_rgb[0] = std::stoi(fb_tokens[0]);
        user_film_base_rgb[1] = std::stoi(fb_tokens[1]);
        user_film_base_rgb[2] = std::stoi(fb_tokens[2]);
        film_base_cli_used = true;
      } catch (const std::exception& e) {
        fprintf(stderr, "ERROR! Invalid integer argument for --film_base_rgb\n");
        return 1;
      }
    } else {
      fprintf(stderr, "ERROR! --film_base_rgb requires 3 integers or 'auto'\n");
      return 1;
    }
  }

  bool is_bundle = false;
  ProfileBundle bundle;
  std::string film_prof_arg;
  if (parser.is_used("-p") || parser.is_used("--film_profile")) {
    film_prof_arg = parser.get<std::string>("--film_profile");
    std::string resolved_prof = resolve_profile_path(film_prof_arg);
    if (is_json_bundle(resolved_prof)) {
      if (!load_profile_bundle(resolved_prof, bundle)) {
        fprintf(stderr, "ERROR! Cannot load profile bundle %s\n", film_prof_arg.c_str());
        return 1;
      }
      is_bundle = true;
    }
  }

  if (is_bundle && bundle.has_crosstalk) {
    if (!parser.is_used("-r") && !parser.is_used("--r_coeff")) {
      r_coeff = {bundle.crosstalk_matrix[0], bundle.crosstalk_matrix[1], bundle.crosstalk_matrix[2]};
    }
    if (!parser.is_used("-g") && !parser.is_used("--g_coeff")) {
      g_coeff = {bundle.crosstalk_matrix[3], bundle.crosstalk_matrix[4], bundle.crosstalk_matrix[5]};
    }
    if (!parser.is_used("-b") && !parser.is_used("--b_coeff")) {
      b_coeff = {bundle.crosstalk_matrix[6], bundle.crosstalk_matrix[7], bundle.crosstalk_matrix[8]};
    }
  }

  if (is_film_base_auto && !is_roi_auto) {
    fprintf(stderr, "ERROR! --film_base_rgb auto requires --roi auto\n");
    return 1;
  }
  if (is_film_base_auto && !parser.is_used("--profile_film_base_rgb") && !(is_bundle && bundle.has_film_base)) {
    fprintf(stderr, "ERROR! --film_base_rgb auto requires --profile_film_base_rgb\n");
    return 1;
  }

  const double t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  if (files.size() == 4) {
    proc = merge_pixel_shift_streaming(files);
    if (!proc) {
      fprintf(stderr, "ERROR! Failed to merge pixel-shift files\n");
      return 1;
    }
    if (!is_roi_auto && (has_roi || rot_cw != 0 || hflip || vflip)) {
      int rx = has_roi ? roi_coords[0] : 0;
      int ry = has_roi ? roi_coords[1] : 0;
      int rw = has_roi ? roi_coords[2] : proc->imgdata.sizes.iwidth;
      int rh = has_roi ? roi_coords[3] : proc->imgdata.sizes.iheight;
      if (!extract_roi_and_transform(proc, rx, ry, rw, rh, rot_cw, hflip, vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      printf("Extracted subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
    }
  } else {
    const bool has_crosstalk = parser.is_used("-r") || parser.is_used("--r_coeff") ||
                               parser.is_used("-g") || parser.is_used("--g_coeff") ||
                               parser.is_used("-b") || parser.is_used("--b_coeff") ||
                               (is_bundle && bundle.has_crosstalk);
    proc = load_raw(files[0], true,
                    parser.get<bool>("--half_size"),
                    parser.get<int>("--quality"),
                    !parser.get<bool>("--no_crop") && !has_roi && !is_roi_auto,
                    has_crosstalk,
                    (!is_roi_auto && has_roi) ? roi_coords : nullptr);
    if (!proc) {
      fprintf(stderr, "Cannot open %s\n", files[0].c_str());
      return 1;
    }
    if (!is_roi_auto && (rot_cw != 0 || hflip || vflip)) {
      if (!extract_roi_and_transform(proc, 0, 0, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                     rot_cw, hflip, vflip)) {
        proc->free_image();
        delete proc;
        return 1;
      }
      printf("Transformed subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
    }
  }

  const double t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0;
  const int orig_w = proc->imgdata.sizes.raw_width ? proc->imgdata.sizes.raw_width : proc->imgdata.sizes.width;
  const int orig_h = proc->imgdata.sizes.raw_height ? proc->imgdata.sizes.raw_height : proc->imgdata.sizes.height;

  FrameDetection frame_det;
  double t_frame_det_ms = 0.0;
  bool rebate_applied = false;

  if (is_roi_auto) {
    const double t_det_0 = omp_get_wtime();
    const bool half_size = parser.get<bool>("--half_size");
    const int cur_w = proc->imgdata.sizes.iwidth;
    const int cur_h = proc->imgdata.sizes.iheight;
    const unsigned filters = proc->imgdata.idata.filters;

    bool det_ok = detect_film_frame(
        (const uint16_t (*)[4])proc->imgdata.image,
        cur_w, cur_h, filters, half_size,
        roi_axis_hint, frame_det);

    t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;

    if (!det_ok) {
      fprintf(stderr, "ERROR! Film frame detection failed\n");
      proc->free_image();
      delete proc;
      return 1;
    }

    const int s_scale = (half_size && files.size() != 4) ? 2 : 1;
    roi_coords[0] = frame_det.x * s_scale;
    roi_coords[1] = frame_det.y * s_scale;
    roi_coords[2] = frame_det.w * s_scale;
    roi_coords[3] = frame_det.h * s_scale;
    has_roi = true;

    printf("Film frame: axis=%c gaps=%d segments=%d selected=%d conf=%.2f\n",
           frame_det.axis, frame_det.n_gaps, frame_det.n_segments, frame_det.selected, frame_det.conf);
    if (frame_det.gap[0][0] >= 0 || frame_det.gap[1][0] >= 0) {
      printf("  Gaps (sensor px):");
      if (frame_det.gap[0][0] >= 0) printf(" [%d, %d]", frame_det.gap[0][0] * s_scale, frame_det.gap[0][1] * s_scale);
      if (frame_det.gap[1][0] >= 0) printf(" [%d, %d]", frame_det.gap[1][0] * s_scale, frame_det.gap[1][1] * s_scale);
      printf("\n");
    }
    printf("  Frame (sensor px): x=%d y=%d w=%d h=%d aspect=%.3f format=%s\n",
           roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3],
           frame_det.aspect, frame_det.format.c_str());

    if (frame_det.has_base) {
      if (is_film_base_auto) {
        rebate_applied = true;
        printf("  Rebate base: R=%d G=%d B=%d (std %.3f D) applied\n",
               frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2], frame_det.base_std_d);
      } else {
        printf("  Rebate base: R=%d G=%d B=%d (std %.3f D) sampled, unused\n",
               frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2], frame_det.base_std_d);
      }
    } else if (is_film_base_auto) {
      printf("Warning: --film_base_rgb auto requested but no validated rebate found\n");
    }

    if (frame_det.conf < 0.5f) {
      printf("Warning: low detection confidence (conf=%.2f%s%s), using envelope fallback\n",
             frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
    }

    if (parser.is_used("--dino") && frame_det.axis == 'v' && rot_cw == 0) {
      printf("Warning: axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
    }

    if (!extract_roi_and_transform(proc, frame_det.x, frame_det.y, frame_det.w, frame_det.h,
                                   rot_cw, hflip, vflip)) {
      fprintf(stderr, "ERROR! Failed to extract detected frame ROI\n");
      proc->free_image();
      delete proc;
      return 1;
    }
    printf("Extracted subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
           proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
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
  const auto orig_profile_film_base_rgb = parser.get<std::vector<int>>("--profile_film_base_rgb");
  std::vector<int> effective_profile_film_base_rgb = orig_profile_film_base_rgb;
  if (is_bundle && bundle.has_film_base && !parser.is_used("--profile_film_base_rgb")) {
    effective_profile_film_base_rgb = {bundle.profile_film_base_rgb[0], bundle.profile_film_base_rgb[1], bundle.profile_film_base_rgb[2]};
  }
  const std::vector<int> sidecar_profile_film_base_rgb = effective_profile_film_base_rgb;
  std::vector<int> effective_film_base_rgb = {1, 1, 1};
  std::vector<int> sidecar_film_base_rgb = {1, 1, 1};
  bool film_base_applied = false;
  std::string film_base_source = "none";

  if (rebate_applied) {
    effective_film_base_rgb = {frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2]};
    sidecar_film_base_rgb = effective_film_base_rgb;
    film_base_applied = true;
    film_base_source = "rebate";
  } else if (film_base_cli_used) {
    sidecar_film_base_rgb = user_film_base_rgb;
    film_base_source = "cli";
    if (parser.is_used("--profile_film_base_rgb") || (is_bundle && bundle.has_film_base)) {
      effective_film_base_rgb = user_film_base_rgb;
      film_base_applied = true;
    } else {
      effective_profile_film_base_rgb = {1, 1, 1};
      effective_film_base_rgb = {1, 1, 1};
      film_base_applied = false;
    }
  } else {
    if (frame_det.has_base) {
      sidecar_film_base_rgb = {frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2]};
    } else {
      sidecar_film_base_rgb = {1, 1, 1};
    }
    effective_profile_film_base_rgb = {1, 1, 1};
    effective_film_base_rgb = {1, 1, 1};
    film_base_applied = false;
    film_base_source = "none";
  }

  unsigned pin_mask = 0;
  if (parser.is_used("-E") || parser.is_used("--exposure_comp")) pin_mask |= PIN_E;
  if (parser.is_used("--gain_g")) pin_mask |= PIN_G;
  if (parser.is_used("--gain_b")) pin_mask |= PIN_B;

  float merged_matrix[9];
  adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                           global_exposure_comp,
                           gain_g, gain_b,
                           effective_profile_film_base_rgb,
                           effective_film_base_rgb,
                           merged_matrix);

  ParsedFilmProfile film_prof;
  std::vector<uint8_t> embedded_icc;
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  int manual_target_idx = -1;

  if (is_bundle) {
    if (parser.is_used("--target")) {
      const std::string tgt_str = parser.get<std::string>("--target");
      char* end = nullptr;
      long val = strtol(tgt_str.c_str(), &end, 10);
      if (end && *end == '\0' && val >= 0 && val < (long)bundle.targets.size()) {
        manual_target_idx = (int)val;
      } else {
        for (size_t i = 0; i < bundle.targets.size(); ++i) {
          if (bundle.targets[i].name == tgt_str || bundle.targets[i].display_name == tgt_str) {
            manual_target_idx = (int)i;
            break;
          }
        }
      }
      if (manual_target_idx < 0) {
        fprintf(stderr, "ERROR! Target '%s' not found in bundle\n", tgt_str.c_str());
        proc->free_image();
        delete proc;
        return 1;
      }
    }

    float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
    int num_feasible = 0;
    compute_bundle_enclosures(bundle,
                              (const uint16_t (*)[4])proc->imgdata.image,
                              proc->imgdata.sizes.iwidth,
                              proc->imgdata.sizes.iheight,
                              g_lo, g_hi, scene_ev, num_feasible);
  } else {
    if (parser.is_used("--target")) {
      fprintf(stderr, "ERROR! --target requires a JSON profile bundle\n");
      proc->free_image();
      delete proc;
      return 1;
    }
    if (parser.is_used("--film_profile")) {
      const std::string prof_path = parser.get<std::string>("--film_profile");
      if (!load_film_profile(prof_path, film_prof)) {
        fprintf(stderr, "ERROR! Cannot load film profile %s\n", prof_path.c_str());
      }
    }
  }

  DinoSolveResult dino_res;

  if (parser.is_used("--dino")) {
    if (!parser.is_used("--film_profile") || (!is_bundle && !film_prof.data.has_profile)) {
      fprintf(stderr, "ERROR! --dino requires --film_profile (-p) to evaluate intent losses.\n");
      proc->free_image();
      delete proc;
      return 1;
    }

    const std::string dino_path = parser.get<std::string>("--dino");
    const std::string dino_backend = parser.get<std::string>("--dino_backend");
    printf("Loading DINOv3 engine from '%s' (backend: %s)...\n", dino_path.c_str(), dino_backend.c_str());
    DinoV3Engine engine;
    if (!engine.load(dino_path, dino_backend)) {
      fprintf(stderr, "ERROR! Failed to load DINOv3 engine from '%s'\n", dino_path.c_str());
      proc->free_image();
      delete proc;
      return 1;
    }
    printf("DINOv3 engine loaded [%s] (memory: %.1f MB)\n",
           engine.backend_name(), (double)engine.memory_bytes() / (1024.0 * 1024.0));

    SolverConfig cfg;
    cfg.knee = knee;
    cfg.knee_auto = is_knee_auto;
    cfg.has_gamma = has_gamma;
    cfg.inv_gamma = inv_gamma;

    const int max_iters = parser.get<int>("--dino_iters");
    IntentGains user_gains{global_exposure_comp, gain_g, gain_b};

    if (is_bundle) {
      if (manual_target_idx >= 0) {
        bundle_target_idx = manual_target_idx;
        const auto& tgt = bundle.targets[bundle_target_idx];
        bundle_target_name = tgt.name;
        film_prof = tgt.film_prof;
        film_prof.rebind_pointers();
        set_target_bounds(cfg, tgt);

        if (!solve_dinov3_intent(engine, cfg, merged_matrix,
                                 (const uint16_t (*)[4])proc->imgdata.image,
                                 proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                                 user_gains, pin_mask, max_iters, dino_res)) {
          fprintf(stderr, "ERROR! DINOv3 intent solver failed.\n");
          proc->free_image();
          delete proc;
          return 1;
        }
      } else {
        bundle_target_idx = run_bundle_tournament_with_dino(
            bundle, engine, cfg, merged_matrix,
            (const uint16_t (*)[4])proc->imgdata.image,
            proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
            user_gains, pin_mask, max_iters,
            kDefaultMinTargetIdx, dino_res);
        if (bundle_target_idx < 0) {
          fprintf(stderr, "ERROR! Bundle tournament failed.\n");
          proc->free_image();
          delete proc;
          return 1;
        }
        bundle_target_name = bundle.targets[bundle_target_idx].name;
        film_prof = bundle.targets[bundle_target_idx].film_prof;
        film_prof.rebind_pointers();
      }
    } else {
      cfg.prof = &film_prof.data;
      if (!solve_dinov3_intent(engine, cfg, merged_matrix,
                               (const uint16_t (*)[4])proc->imgdata.image,
                               proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                               user_gains, pin_mask, max_iters, dino_res)) {
        fprintf(stderr, "ERROR! DINOv3 intent solver failed.\n");
        proc->free_image();
        delete proc;
        return 1;
      }
    }

    if (is_knee_auto && dino_res.dynamic_knee.active) {
      knee = dino_res.dynamic_knee.knee;
    }

    std::string pins_str = "none";
    if (pin_mask != 0) {
      pins_str = "";
      if (pin_mask & PIN_E) pins_str += "E ";
      if (pin_mask & PIN_G) pins_str += "g ";
      if (pin_mask & PIN_B) pins_str += "b";
      if (!pins_str.empty() && pins_str.back() == ' ') pins_str.pop_back();
    }
    printf("\nDINOv3 solver: grid %dx%d (%dx%d), backend %s, E_c %.3f, bounds [%.3f, %.3f], max iters %d, pins %s\n",
           dino_res.grid_w, dino_res.grid_h, dino_res.target_w, dino_res.target_h,
           engine.backend_name(), dino_res.e_center, dino_res.bounds_lo, dino_res.bounds_hi,
           max_iters, pins_str.c_str());
    if (dino_res.anchor_clamped) {
      printf("Warning: ANCHOR_CLAMPED (median L* 50 not reached within E_c 2^[-4, 4])\n");
    }
    printf(" it      E      g      b   resid    L_L    L_a    L_b  clip%%  P_clip  P_anch       J\n");
    for (const auto& it : dino_res.iterations) {
      if (it.residual < 0.0f) {
        printf("  %c  %5.3f  %5.3f  %5.3f       -  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
               it.tag, it.e, it.g, it.b, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
               it.penalty_clip, it.penalty_anchor, it.objective);
      } else {
        printf("  %c  %5.3f  %5.3f  %5.3f  %6.3f  %5.3f  %5.3f  %5.3f  %5.2f  %6.3f  %6.3f  %6.3f\n",
               it.tag, it.e, it.g, it.b, it.residual, it.loss_L, it.loss_a, it.loss_b, it.clip_pct,
               it.penalty_clip, it.penalty_anchor, it.objective);
      }
    }
    printf("Status: %s after %d inferences (best it %d, resid %.3f)\n",
           dino_res.status.c_str(), dino_res.total_inferences, dino_res.best_iteration, dino_res.best_residual);
    printf("Solved (relative): E %.3f g %.3f b %.3f\n",
           dino_res.solved_relative.e, dino_res.solved_relative.g, dino_res.solved_relative.b);
    printf("Solved (absolute): -E %.9g --gain_g %.9g --gain_b %.9g\n",
           dino_res.solved_absolute.e, dino_res.solved_absolute.g, dino_res.solved_absolute.b);
    if (dino_res.solved_relative.g < 0.5f || dino_res.solved_relative.g > 2.0f ||
        dino_res.solved_relative.b < 0.5f || dino_res.solved_relative.b > 2.0f) {
      printf("Warning: GAIN_RANGE (relative g or b outside [0.5, 2.0])\n");
    }
    if (dino_res.best_iteration >= 0 && (size_t)dino_res.best_iteration + 1 < dino_res.iterations.size()) {
      const auto& bit = dino_res.iterations[dino_res.best_iteration + 1];
      float log2_ratio = (bit.e > 0.0f) ? std::log2(bit.e) : 0.0f;
      printf("Penalties: clip %.3f (%.2f%% %s 2.50%%), anchor %.3f (log2(E/E_c) %+.3f)\n",
             bit.penalty_clip, bit.clip_pct, bit.clip_pct <= 2.5f ? "<=" : ">", bit.penalty_anchor, log2_ratio);
    }
    printf("Timings (ms): views %.1f, render %.1f, infer %.1f, residual %.1f, search %.1f\n\n",
           dino_res.timing_views_ms, dino_res.timing_render_ms, dino_res.timing_infer_ms,
           dino_res.timing_residual_ms, dino_res.timing_search_ms);

    global_exposure_comp = dino_res.solved_absolute.e;
    gain_g = dino_res.solved_absolute.g;
    gain_b = dino_res.solved_absolute.b;

    adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                             global_exposure_comp,
                             gain_g, gain_b,
                             effective_profile_film_base_rgb,
                             effective_film_base_rgb,
                             merged_matrix);
  } else if (is_bundle) {
    if (manual_target_idx >= 0) {
      bundle_target_idx = manual_target_idx;
    } else {
      bundle_target_idx = select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
    }
    printf("Selected target: %s\n", bundle.targets[bundle_target_idx].display_name.c_str());
    bundle_target_name = bundle.targets[bundle_target_idx].name;
    film_prof = bundle.targets[bundle_target_idx].film_prof;
    film_prof.rebind_pointers();
    if (!(pin_mask & PIN_E) && bundle.targets[bundle_target_idx].e_center > 0.0f) {
      global_exposure_comp = bundle.targets[bundle_target_idx].e_center;
      adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                               global_exposure_comp,
                               gain_g, gain_b,
                               effective_profile_film_base_rgb,
                               effective_film_base_rgb,
                               merged_matrix);
    }
  }

  if (is_bundle && bundle_target_idx >= 0) {
    embedded_icc = bundle.targets[bundle_target_idx].icc_bytes;
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
    if (!is_bundle) {
      attach_profile = parser.get<std::string>("--film_profile");
    }
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
  const double t_full_conversion_ms = (t_end - t_start) * 1000.0;
  printf("Processing time: %f\n", t_end - t_start);

  const auto output = parser.get<std::string>("--output");
  printf("Writing TIFF '%s'\n", output.c_str());
  const double t_tiff_0 = omp_get_wtime();
  int ret = write_tiff(proc, attach_profile, output, is_bundle ? embedded_icc : std::vector<uint8_t>{});
  const double t_tiff_write_ms = (omp_get_wtime() - t_tiff_0) * 1000.0;

  if (ret != 0 || parser.get<bool>("--no-metadata")) {
    unlink((output + ".json").c_str());
  } else {
    const std::string knee_source = is_knee_auto ? "auto" : (parser.is_used("--knee") ? "cli" : "default");
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
    if (is_roi_auto) {
      roi_source = (frame_det.conf >= 0.5f) ? "auto" : "auto_fallback";
    } else if (has_roi) {
      roi_source = "cli";
    }

    write_json_sidecar(output, ac, av, proc, files,
                       orig_w, orig_h,
                       parser.get<bool>("--half_size"),
                       has_roi, roi_coords, roi_source,
                       is_roi_auto ? &frame_det : nullptr,
                       rot_cw, hflip, vflip,
                       width, height,
                       film_base_applied, sidecar_film_base_rgb, sidecar_profile_film_base_rgb,
                       film_base_source,
                       frame_det.base_std_d, frame_det.has_base,
                       parser.is_used("--film_profile") ? parser.get<std::string>("--film_profile") : "",
                       merged_matrix,
                       global_exposure_comp, gain_g, gain_b,
                       pin_mask,
                       knee, knee_source,
                       parser.is_used("--dino"), dino_res,
                       parser.is_used("--dino") ? parser.get<std::string>("--dino") : "",
                       parser.is_used("--dino_backend") ? parser.get<std::string>("--dino_backend") : "auto",
                       cs_name,
                       t_raw_decode_ms, t_frame_det_ms, t_full_conversion_ms, t_tiff_write_ms,
                       is_bundle, bundle_target_name, bundle_target_idx);
  }

  proc->free_image();
  delete proc;
  return ret;
}
