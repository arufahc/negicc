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
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <netinet/in.h>

#include "argparse/argparse.hpp"
#include "dcraw/gamma_curve.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "libraw/libraw.h"
#include "tiff_head.h"
#include "dinov3_engine.h"
#include "neg_pipeline.h"
#include "cfa_pipeline.h"
#include "color_pipeline.h"
#include "flat_field.h"
#include "flat_sample.h"
#include "dinov3_auto_solver.h"
#include "embedded_data.h"
#include "film_frame_detector.h"
#include "multi_target_optimizer.h"
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

static bool is_directory(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

static bool is_regular_file(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

// Loads the negicc-station profile bundle JSON a -p argument names (as given or under profiles/), or the bundle.json
// of a split-bundle directory (profiles/<Key>). False when the argument is not a bundle (an ICC profile).
bool load_bundle_arg(const std::string& arg, ProfileBundle& bundle) {
  std::string resolved = resolve_profile_path(arg);
  if (is_directory(resolved)) resolved = (std::filesystem::path(resolved) / "bundle.json").string();
  if (!is_regular_file(resolved) || !is_json_bundle(resolved)) return false;
  load_profile_bundle(resolved, bundle);
  return true;
}

// Loads the built-in profile bundle with this key (embedded_bundle_key). False when none has that key.
bool load_embedded_bundle(const std::string& key, ProfileBundle& bundle) {
  std::string dir;
  const EmbeddedFile* f = find_embedded_bundle(key, dir);
  if (!f) return false;
  const BundleFileReader read_file = [&dir](const std::string& name, std::vector<uint8_t>& out) {
    const EmbeddedFile* e = find_embedded_file(dir + "/" + name);
    if (!e) return false;
    out.assign(e->data, e->data + e->size);
    return true;
  };
  load_profile_bundle_from_memory((const char*)f->data, f->size, std::string("[built-in] ") + f->name, read_file,
                                  bundle);
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
// roi, when given, lies inside the sensor (load_raw clamps it).
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
  if (roi) {
    rx = roi[0];
    ry = roi[1];
    rw = roi[2];
    rh = roi[3];
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

// Called with a LibRaw whose imgdata.image holds the half-size image of its RAW (read_half_size): iwidth x iheight
// pixels, row-major, LibRaw's 4 channels each (R, G, B in the first three); a point (x, y) of the RAW is (x / 2, y / 2)
// there.
using HalfSizeAnalysis = std::function<void(const LibRaw& proc)>;

// Called by load_raw with the unpacked RAW in proc (imgdata.rawdata.raw_image holds the CFA mosaic, margins
// included), before the half-size analysis and the decode. Sets roi to the region to decode (x, y, w, h in the
// visible sensor area, imgdata.sizes.width x height), as --roi would; false fails the load.
using RawRegionFinder = std::function<bool(LibRaw& proc, int roi[4])>;

// The unpacked CFA mosaic of proc as the frame detector reads it, in place (no copy): rawdata.raw_image from its
// first visible photosite (top_margin, left_margin), raw_pitch / 2 elements per row; the R, G and B lattices take one
// photosite of their colour per 2x2 (Bayer) or 6x6 (X-Trans) cell, each with its black level (black + cblack[colour])
// subtracted; maximum - black is the clipping point. A monochrome RAW is its one plane three times. False when proc
// holds no mosaic (e.g. a linear DNG) or when its margins put the visible area outside the raw buffer.
//
// When LibRaw holds the black level as a cblack[6..] pattern (cblack[4] rows x cblack[5] columns, indexed by the
// visible row and column as prepare_white_light_mosaic applies it), each lattice's black also takes the pattern's mean
// over the pattern entries its photosites land on: the rows y0 + j * step and columns x0 + i * step, modulo the
// pattern size. When the pattern period divides the CFA period (the usual case) that is the single entry at the
// lattice's phase, exactly the black of every photosite of the lattice; otherwise it is the mean over the lattice.
static bool film_image_from_raw(LibRaw& proc, FilmImage& img) {
  const ushort* raw = proc.imgdata.rawdata.raw_image;
  if (!raw) return false;
  const auto& sz = proc.imgdata.sizes;
  const ptrdiff_t stride = sz.raw_pitch / 2;
  // The visible area must lie inside the raw buffer (prepare_white_light_mosaic refuses such a RAW the same way).
  if (sz.width <= 0 || sz.height <= 0 || sz.top_margin + sz.height > sz.raw_height ||
      sz.left_margin + sz.width > stride) {
    return false;
  }
  const ushort* origin = raw + (size_t)sz.top_margin * stride + sz.left_margin;
  const auto& C = proc.imgdata.color;
  const float black = (float)C.black;
  float cblack[4];
  for (int c = 0; c < 4; ++c) cblack[c] = black + (float)C.cblack[c];
  const float white = std::max(1.0f, (float)C.maximum - black);
  const unsigned filters = proc.imgdata.idata.filters;
  if (!filters) {
    img = film_image_from_planes(origin, origin, origin, sz.width, sz.height, stride, white);
    for (FilmChannel& ch : img.ch) ch.black = cblack[0];
  } else if (!film_image_from_cfa(origin, sz.width, sz.height, stride, filters == 9 ? 6 : 2,
                                  [&proc](int row, int col) { return proc.COLOR(row, col); }, cblack, white, img)) {
    return false;
  }
  const unsigned dr = C.cblack[4], dc = C.cblack[5];
  if (dr > 0 && dc > 0 && (size_t)dr * dc <= LIBRAW_CBLACK_SIZE - 6) {
    for (FilmChannel& ch : img.ch) {
      // The pattern rows (columns) the lattice visits: y0 + j * step modulo dr (dc), for every j.
      std::vector<char> row_hit(dr, 0), col_hit(dc, 0);
      for (unsigned j = 0; j < dr; ++j) row_hit[(ch.y0 + j * img.step) % dr] = 1;
      for (unsigned i = 0; i < dc; ++i) col_hit[(ch.x0 + i * img.step) % dc] = 1;
      double sum = 0.0;
      int n = 0;
      for (unsigned r = 0; r < dr; ++r) {
        if (!row_hit[r]) continue;
        for (unsigned c = 0; c < dc; ++c) {
          if (!col_hit[c]) continue;
          sum += C.cblack[6 + r * dc + c];
          ++n;
        }
      }
      ch.black += (float)(sum / n);
    }
  }
  return true;
}

// LibRaw's half-size readout does not serve an X-Trans RAW (filters 9): its 2x2 quads hold several photosites of one
// colour, which LibRaw's OpenMP loop writes in a race, and the quads of only green photosites get no R or B. Replaces
// LibRaw's full-size readout in proc with a half-size image of each quad's mean per colour; R and B of the green-only
// quads are the mean of their left and right neighbours, as LibRaw's pre_interpolate fills them. False when out of
// memory.
static bool shrink_xtrans(LibRaw* proc) {
  const int w = proc->imgdata.sizes.iwidth, h = proc->imgdata.sizes.iheight;
  const int hw = (w + 1) / 2, hh = (h + 1) / 2;
  const ushort(*full)[4] = proc->imgdata.image;
  ushort(*half)[4] = (ushort(*)[4])calloc((size_t)hw * hh, sizeof(*half));
  if (!half) return false;
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < hh; ++y) {
    for (int x = 0; x < hw; ++x) {
      unsigned sum[3] = {0, 0, 0}, n[3] = {0, 0, 0};
      for (int r = 2 * y; r < std::min(2 * y + 2, h); ++r) {
        for (int c = 2 * x; c < std::min(2 * x + 2, w); ++c) {
          const int k = proc->COLOR(r, c);
          sum[k] += full[(size_t)r * w + c][k];
          n[k]++;
        }
      }
      for (int k = 0; k < 3; ++k) half[(size_t)y * hw + x][k] = n[k] ? (ushort)((sum[k] + n[k] / 2) / n[k]) : 0;
    }
  }
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < hh; ++y) {
    for (int x = 0; x < hw; ++x) {
      ushort* px = half[(size_t)y * hw + x];
      if ((px[0] | px[2]) || hw < 2) continue;
      const ushort* l = half[(size_t)y * hw + std::max(x - 1, 0)];
      const ushort* r = half[(size_t)y * hw + std::min(x + 1, hw - 1)];
      for (int k = 0; k < 3; k += 2) px[k] = x > 0 && x + 1 < hw ? (l[k] + r[k]) >> 1 : (x > 0 ? l : r)[k];
    }
  }
  proc->free_image();
  proc->imgdata.image = half;
  proc->imgdata.sizes.iwidth = hw;
  proc->imgdata.sizes.iheight = hh;
  return true;
}

// Reads LibRaw's half-size image of the unpacked RAW of proc (black subtracted, every 2x2 quad one pixel), before any
// processing, and hands proc holding it to analyse. proc is then left as unpack() left it. False when proc is not a
// 3-colour Bayer or X-Trans RAW (shrink_xtrans). For both, LibRaw's first three channels are R, G and B.
// py-ref: negicc_station/src/raw_processor.cpp @ 93be1ec (load_raw_mosaic)
static bool read_half_size(LibRaw* proc, const HalfSizeAnalysis& analyse) {
  const unsigned filters = proc->imgdata.idata.filters;
  const bool xtrans = filters == 9;
  if (!proc->imgdata.rawdata.raw_image || (filters <= 1000 && !xtrans) || proc->imgdata.idata.colors != 3) {
    fprintf(stderr, "ERROR! The parameter search supports only 3-colour Bayer and X-Trans RAW files\n");
    return false;
  }
  const int half_size = proc->imgdata.params.half_size;
  proc->imgdata.params.half_size = xtrans ? 0 : 1;
  const int ret = proc->raw2image_ex(1);
  proc->imgdata.params.half_size = half_size;
  if (ret != LIBRAW_SUCCESS || (xtrans && !shrink_xtrans(proc))) {
    fprintf(stderr, "ERROR! Cannot read the half-size RAW: %s\n",
            ret != LIBRAW_SUCCESS ? libraw_strerror(ret) : "out of memory");
    proc->free_image();
    proc->raw2image_start();
    return false;
  }
  analyse(*proc);

  // raw2image_ex subtracted the black level from imgdata.color and resized imgdata.sizes; put back what unpack() read.
  proc->free_image();
  proc->raw2image_start();
  return true;
}

// How load_raw decodes a RAW to RGB.
enum class RawDecode {
  kRec2020,     // LibRaw to linear Rec.2020 with unit channel multipliers (no profile bundle, no -r -g -b)
  kCameraRgb,   // sensor RGB, unit channel multipliers: a camera_raw_rgb (TriLED) bundle or -r -g -b
  kWhiteLight,  // a rec2020_linear (D65) bundle: prepare_white_light_mosaic, then decode_white_light
};

// White-light decode of a rec2020_linear profile bundle, in two steps on an unpacked proc. First, in place on the
// CFA mosaic and in sensor-linear units, the black level is subtracted and the camera's daylight white balance
// (pre_mul over its green) applied. Then (decode_white_light) LibRaw demosaics bilinearly straight to linear Rec.2020
// (output_color 8) through its own camera matrix with no rescaling, so values stay in sensor units. LibRaw clamps
// its output to [0, 65535]; going to Rec.2020 directly, rather than to sRGB and then to Rec.2020, keeps the colours
// between the sRGB and Rec.2020 gamuts that the sRGB clamp lost. This relies on the pinned LibRaw (3rd_party/libraw):
// older releases (0.20) do not implement the Rec.2020 output. There is no flat field here (neg_process applies none
// on any path).
// py-ref: negicc_station/src/raw_processor.cpp @ 813166c (load_raw_white_light; output_color 8 from branch libraw-from-negicc)
// py-ref: negicc_station/src/image_capture.cpp @ 813166c (CapturedImage::decode_white_light_locked; no sRGB matrix from libraw-from-negicc)
//
// Sets clip_fraction to the fraction of photosites at or above the white level. False, with the reason on stderr,
// when the RAW has no 3-colour CFA or no daylight white balance.
static bool prepare_white_light_mosaic(LibRaw* proc, double& clip_fraction, const negicc::FlatField* flat = nullptr) {
  auto& S = proc->imgdata.sizes;
  auto& C = proc->imgdata.color;
  uint16_t* raw = proc->imgdata.rawdata.raw_image;
  if (!raw || !proc->imgdata.idata.filters || proc->imgdata.idata.colors != 3) {
    fprintf(stderr, "ERROR! The white-light decode needs a 3-colour CFA RAW file\n");
    return false;
  }
  const int w = S.width, h = S.height, top = S.top_margin, left = S.left_margin;
  const size_t pitch = S.raw_pitch / 2;
  if (w <= 0 || h <= 0 || top + h > S.raw_height || left + w > (int)pitch) {
    fprintf(stderr, "ERROR! RAW margins exceed the raw buffer\n");
    return false;
  }
  // Colour of each photosite in a 6x6 tile (Bayer repeats every 2): 0..3 for the black level, 0..2 for the gains.
  uint8_t col4[36], col3[36];
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) {
      const int k = proc->COLOR(r, c);
      col4[r * 6 + c] = (uint8_t)(k & 3);
      col3[r * 6 + c] = (uint8_t)(k == 3 ? 1 : k);
    }
  float pre[4];
  for (int i = 0; i < 4; ++i) pre[i] = C.pre_mul[i];
  if (pre[3] <= 0) pre[3] = pre[1];
  if (!(pre[0] > 0 && pre[1] > 0 && pre[2] > 0)) {
    fprintf(stderr, "ERROR! No daylight white balance (pre_mul) for this camera; the white-light decode needs one\n");
    return false;
  }
  float wb[4];
  for (int i = 0; i < 4; ++i) wb[i] = pre[i] / pre[1];
  const unsigned dr = C.cblack[4], dc = C.cblack[5];
  const bool pattern = dr > 0 && dc > 0 && (size_t)dr * dc <= LIBRAW_CBLACK_SIZE - 6;
  const unsigned white = C.maximum;
  const unsigned clip_level = white - white / 200;

  size_t clipped = 0;
  #pragma omp parallel for reduction(+ : clipped) schedule(static)
  for (int y = 0; y < h; ++y) {
    uint16_t* p = raw + (size_t)(y + top) * pitch + left;
    for (int x = 0; x < w; ++x) {
      const int k = col4[(y % 6) * 6 + x % 6];
      int black = (int)C.black + (int)C.cblack[k];
      if (pattern) black += (int)C.cblack[6 + (y % dr) * dc + (x % dc)];
      clipped += p[x] >= clip_level ? 1 : 0;
      const float v = std::max(0, (int)p[x] - black) * wb[k];
      p[x] = (uint16_t)std::min(std::nearbyint(v), 65535.0f);
    }
  }
  clip_fraction = (double)clipped / ((double)w * h);
  if (flat) {
    try {
      flat->apply_cfa(raw + (size_t)top * pitch + left, pitch, w, h, col3);
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! %s\n", e.what());
      return false;
    }
  }

  // The mosaic is black-subtracted and white-balanced: LibRaw must neither subtract nor rescale again. user_black and
  // user_cblack make every dcraw_process (each restores imgdata.color from the unpacked RAW) take a zero black level,
  // so the mosaic can be decoded more than once (half-size for the search, then the output).
  C.black = 0;
  std::memset(C.cblack, 0, sizeof(C.cblack));
  auto& O = proc->imgdata.params;
  O.user_black = 0;
  for (int i = 0; i < 4; ++i) O.user_cblack[i] = 0;
  O.no_auto_scale = 1;
  O.use_camera_wb = 0;
  O.use_auto_wb = 0;
  for (int i = 0; i < 4; ++i) O.user_mul[i] = 1;
  O.output_color = 8;  // linear Rec.2020, through LibRaw's own camera matrix for this camera
  O.output_bps = 16;
  O.gamm[0] = 1;
  O.gamm[1] = 1;
  O.no_auto_bright = 1;
  O.highlight = 1;
  O.user_flip = 0;
  O.four_color_rgb = 0;
  return true;
}

// Demosaics the mosaic prepare_white_light_mosaic prepared to linear Rec.2020 in imgdata.image (half: LibRaw's
// half-size decode; qual: LibRaw's user_qual, negicc-station uses 0, bilinear). The cropbox, if any, is set by the
// caller. False when LibRaw cannot process it.
static bool decode_white_light(LibRaw* proc, bool half, int qual) {
  proc->imgdata.params.half_size = half ? 1 : 0;
  proc->imgdata.params.user_qual = qual;
  const int ret = proc->dcraw_process();
  if (ret != LIBRAW_SUCCESS) {
    fprintf(stderr, "ERROR! Cannot process the white-light decode: %s\n", libraw_strerror(ret));
    return false;
  }
  return true;
}

// Step 1 helper: subtract per-channel black in-place on the unpacked CFA mosaic.
static bool subtract_cfa_black_in_place(LibRaw* proc, float& out_white_level) {
  if (!proc || !proc->imgdata.rawdata.raw_image) return false;
  const auto& S = proc->imgdata.sizes;
  auto& C = proc->imgdata.color;
  const int w = S.width, h = S.height, top = S.top_margin, left = S.left_margin;
  const size_t pitch = S.raw_pitch / 2;
  if (w <= 0 || h <= 0 || top < 0 || left < 0 || top + h > S.raw_height || left + w > (int)pitch) {
    return false;
  }
  const int base_black = (int)C.black;
  const unsigned dr = C.cblack[4], dc = C.cblack[5];
  const bool pattern = dr > 0 && dc > 0 && (size_t)dr * dc <= LIBRAW_CBLACK_SIZE - 6;
  int pat_sum[4] = {0, 0, 0, 0}, pat_n[4] = {0, 0, 0, 0};
  if (pattern) {
    for (unsigned r = 0; r < dr; ++r) {
      for (unsigned c = 0; c < dc; ++c) {
        const int k = proc->COLOR(r, c) & 3;
        pat_sum[k] += (int)C.cblack[6 + r * dc + c];
        pat_n[k]++;
      }
    }
  }
  int max_blk = base_black;
  int cblk[4];
  for (int i = 0; i < 4; ++i) {
    cblk[i] = base_black + (int)C.cblack[i];
    const int avg_pat = pat_n[i] > 0 ? pat_sum[i] / pat_n[i] : 0;
    max_blk = std::max(max_blk, cblk[i] + avg_pat);
  }
  out_white_level = std::max(1.0f, (float)((int)C.maximum - max_blk));
  uint16_t* cfa = proc->imgdata.rawdata.raw_image + (size_t)top * pitch + left;
  #pragma omp parallel for schedule(static)
  for (int r = 0; r < h; ++r) {
    uint16_t* row = cfa + (size_t)r * pitch;
    const int blk_pair[2] = {cblk[proc->FC(r, 0)], cblk[proc->FC(r, 1)]};
    if (!pattern) {
      for (int c = 0; c < w; ++c) {
        const int v = (int)row[c] - blk_pair[c & 1];
        row[c] = (v > 0) ? (uint16_t)v : 0;
      }
    } else {
      const unsigned pr = (unsigned)r % dr;
      for (int c = 0; c < w; ++c) {
        const int blk = blk_pair[c & 1] + (int)C.cblack[6 + pr * dc + ((unsigned)c % dc)];
        const int v = (int)row[c] - blk;
        row[c] = (v > 0) ? (uint16_t)v : 0;
      }
    }
  }
  C.black = 0;
  std::memset(C.cblack, 0, sizeof(C.cblack));
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
//
// |decode| picks the colour path of a debayered decode (RawDecode); kWhiteLight decodes with LibRaw's bilinear
// demosaic (or qual when it is >= 0), never grain-aware.
//
// |analyse|, when set, is called with the half-size image of the RAW before it is decoded: LibRaw's half-size readout
// of the mosaic (read_half_size), or for kWhiteLight its half-size white-light decode, in the same space as the
// output.
//
// |find_region|, when set, is called on the unpacked mosaic first and gives the region to decode in place of roi
// (--roi auto: the detected frame), so that the analysis and the decode see only that region.
LibRaw* load_raw(const std::string& fn, bool debayer, bool half_size, int qual, bool crop,
                 RawDecode decode = RawDecode::kRec2020, const int* roi = nullptr,
                 const HalfSizeAnalysis& analyse = {}, const RawRegionFinder& find_region = {},
                 const negicc::FlatField* flat = nullptr) {
  const bool white_light = debayer && decode == RawDecode::kWhiteLight;
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
  int found_roi[4];
  if (find_region) {
    if (!find_region(*proc, found_roi)) {
      delete proc;
      return NULL;
    }
    roi = found_roi;
  }
  // The ROI is checked and clamped to the sensor before the analysis read, which runs the whole bracket search, so
  // that an ROI outside the sensor fails before the DINOv3 inferences.
  int roi_box[4];
  if (roi) {
    const int w = proc->imgdata.sizes.width, h = proc->imgdata.sizes.height;
    if (roi[0] < 0 || roi[1] < 0 || roi[0] >= w || roi[1] >= h || roi[2] <= 0 || roi[3] <= 0) {
      fprintf(stderr, "ERROR! ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
              roi[0], roi[1], roi[2], roi[3], w, h);
      delete proc;
      return NULL;
    }
    roi_box[0] = roi[0];
    roi_box[1] = roi[1];
    roi_box[2] = std::min(roi[2], w - roi[0]);
    roi_box[3] = std::min(roi[3], h - roi[1]);
    roi = roi_box;
  }
  if (white_light) {
    double clip_fraction = 0.0;
    if (!prepare_white_light_mosaic(proc, clip_fraction, flat)) {
      delete proc;
      return NULL;
    }
    printf("White-light decode (linear Rec.2020): %.4f%% of photosites clipped\n", clip_fraction * 100.0);
    if (analyse) {
      if (!decode_white_light(proc, true, 0)) {
        delete proc;
        return NULL;
      }
      analyse(*proc);
      proc->free_image();
      proc->raw2image_start();
      proc->imgdata.color.black = 0;
      std::memset(proc->imgdata.color.cblack, 0, sizeof(proc->imgdata.color.cblack));
    }
  } else if (analyse && !read_half_size(proc, analyse)) {
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
  proc->imgdata.params.output_tiff = 1;
  if (!debayer) {
    proc->imgdata.sizes.iwidth = proc->imgdata.sizes.width;
    proc->imgdata.sizes.iheight = proc->imgdata.sizes.height;
  } else {
    // Camera linear RGB (output_color = 0) for a crosstalk correction (a camera_raw_rgb bundle or -r -g -b);
    // otherwise Rec.2020 (output_color = 8). The white-light decode set its own (prepare_white_light_mosaic).
    const bool camera_rgb = decode == RawDecode::kCameraRgb;
    if (!white_light) proc->imgdata.params.output_color = camera_rgb ? 0 : 8;

    bool use_grain_aware = (!half_size && qual < 0 && camera_rgb && is_sony_cfa_geometry(proc));
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
      // Unit multipliers on every path: the white-light decode has applied its white balance to the mosaic.
      proc->imgdata.params.user_mul[0] = 1;
      proc->imgdata.params.user_mul[1] = 1;
      proc->imgdata.params.user_mul[2] = 1;
      proc->imgdata.params.user_mul[3] = 1;

      if (roi) {
        for (int i = 0; i < 4; ++i) proc->imgdata.params.cropbox[i] = roi[i];
      } else {
#if LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21)
        auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
#else
        auto& crop_ref = proc->imgdata.sizes.raw_inset_crop;
#endif
        // LibRaw reports an inset crop the RAW does not give (e.g. a DNG without DefaultCrop) as left and top 0xffff
        // with a zero size; dcraw_process would refuse that as a bad crop box.
        const bool has_inset_crop = crop_ref.cwidth > 0 && crop_ref.cheight > 0 && crop_ref.cleft != 0xffff &&
                                    crop_ref.ctop != 0xffff;
        if (crop && has_inset_crop && (crop_ref.cleft || crop_ref.ctop)) {
          proc->imgdata.params.cropbox[0] = crop_ref.cleft;
          proc->imgdata.params.cropbox[1] = crop_ref.ctop;
          proc->imgdata.params.cropbox[2] = crop_ref.cwidth;
          proc->imgdata.params.cropbox[3] = crop_ref.cheight;
        }
      }
      if (white_light) {
        if (!decode_white_light(proc, half_size, effective_qual)) {
          delete proc;
          return NULL;
        }
      } else {
        proc->dcraw_process();
      }
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
LibRaw* merge_pixel_shift_streaming(const std::vector<std::string>& files, const HalfSizeAnalysis& analyse = {}) {
  if (files.size() < 4) return nullptr;
  printf("Merging 4 images...\n");
  LibRaw* base = load_raw(files[0], false, false, 0, false, RawDecode::kCameraRgb, nullptr, analyse);
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
// A 180-degree rotation is both flips; leaves rot_cw 0, 90 or 270.
static void normalize_rotation(int& rot_cw, bool& hflip, bool& vflip) {
  rot_cw = ((rot_cw % 360) + 360) % 360;
  if (rot_cw == 180) {
    hflip = !hflip;
    vflip = !vflip;
    rot_cw = 0;
  }
}

// Where pixel (x, y) of a rw x rh region lands, (dx, dy), when the region is rotated clockwise by rot_cw (0, 90 or
// 270, see normalize_rotation) and then flipped. The result is dst_w x dst_h: rw x rh, or rh x rw when rotated.
static inline void oriented_pos(int x, int y, int rw, int rh, int dst_w, int dst_h, int rot_cw, bool hflip,
                                bool vflip, int& dx, int& dy) {
  const int x1 = (rot_cw == 0) ? x : ((rot_cw == 90) ? (rh - 1 - y) : y);
  const int y1 = (rot_cw == 0) ? y : ((rot_cw == 90) ? x : (rw - 1 - x));
  dx = hflip ? (dst_w - 1 - x1) : x1;
  dy = vflip ? (dst_h - 1 - y1) : y1;
}

bool extract_roi_and_transform(LibRaw* proc, int rx, int ry, int rw, int rh,
                               int rot_cw, bool hflip, bool vflip) {
  if (!proc || !proc->imgdata.image) return false;

  const int cur_w = proc->imgdata.sizes.iwidth;
  const int cur_h = proc->imgdata.sizes.iheight;
  rx = std::max(0, std::min(rx, cur_w - 1));
  ry = std::max(0, std::min(ry, cur_h - 1));
  rw = std::max(1, std::min(rw, cur_w - rx));
  rh = std::max(1, std::min(rh, cur_h - ry));

  normalize_rotation(rot_cw, hflip, vflip);
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
        int dx = 0, dy = 0;
        oriented_pos(x, y, rw, rh, dst_w, dst_h, rot_cw, hflip, vflip, dx, dy);

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

// Moves rect (x, y, w, h) inside bw x bh, shrinking it only when it is larger.
// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
static void clamp_rect(int r[4], int bw, int bh) {
  r[2] = std::min(r[2], bw);
  r[3] = std::min(r[3], bh);
  r[0] = std::min(std::max(0, r[0]), bw - r[2]);
  r[1] = std::min(std::max(0, r[1]), bh - r[3]);
}

// The frame the parameter search runs on, upright: rect (x, y, w, h) of the image proc holds, cut out and oriented
// as the output is (extract_roi_and_transform), as camera RGB from LibRaw's first three channels. It is w x h, or
// h x w when rotated by 90 or 270 degrees.
static void cut_upright_frame(const LibRaw& proc, const int rect[4], int rot_cw, bool hflip, bool vflip,
                              RgbImage& out) {
  normalize_rotation(rot_cw, hflip, vflip);
  const int rw = rect[2], rh = rect[3];
  const bool swap_dims = (rot_cw == 90 || rot_cw == 270);
  out.width = swap_dims ? rh : rw;
  out.height = swap_dims ? rw : rh;
  out.rgb.resize((size_t)out.width * out.height * 3);
  const int src_w = proc.imgdata.sizes.iwidth;
  const ushort (*src)[4] = proc.imgdata.image;
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < rh; ++y) {
    for (int x = 0; x < rw; ++x) {
      int dx = 0, dy = 0;
      oriented_pos(x, y, rw, rh, out.width, out.height, rot_cw, hflip, vflip, dx, dy);
      const ushort* s = src[(size_t)(rect[1] + y) * src_w + (rect[0] + x)];
      std::copy_n(s, 3, &out.rgb[((size_t)dy * out.width + dx) * 3]);
    }
  }
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
               const std::vector<uint8_t>& embedded_icc = {}, const uint16_t* rgb3 = nullptr) {
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

  if (rgb3) {
    fwrite(rgb3, 3 * sizeof(uint16_t), (size_t)width * height, fp);
  } else {
    std::vector<ushort> row_buf(width * 3);
    for (unsigned row = 0; row < height; ++row) {
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

// Below this frame detection confidence neg_process warns and the sidecar sets frame_detection.low_confidence. The
// crop is still the detector's frame: there is no fallback region.
static constexpr float kLowFrameConfidence = 0.5f;

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
    int bundle_target_idx = -1,
    const ParamSearchResult* auto_res = nullptr,
    const int* auto_frame_half = nullptr,
    double t_auto_ms = 0.0,
    InputSpace input_space = InputSpace::kCameraRawRgb) {
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
    // Sensor pixel coordinates throughout (the detector ran on the RAW before the decode). aspect and format describe
    // the frame as the previous detector did ("3:2", ...); film_format is the format the detection used. axis is
    // "h" or "v", or null when the detector left it undecided (a single square frame with nothing to orient it).
    auto rect_json = [](const FilmRect& r) {
      char b[160];
      snprintf(b, sizeof(b), "{\"cx\": %.1f, \"cy\": %.1f, \"w\": %.1f, \"h\": %.1f, \"angle_deg\": %.3f}",
               r.cx, r.cy, r.w, r.h, r.angle_deg);
      return std::string(b);
    };
    const FilmRect& pr = frame_det->primary.rect;
    const float aspect = std::max(pr.w, pr.h) / std::max(1.0f, std::min(pr.w, pr.h));
    const char* ratio = "-";
    switch (frame_det->format) {
      case FILM_35MM: ratio = "3:2"; break;
      case FILM_645: ratio = "4:3"; break;
      case FILM_6X6: ratio = "1:1"; break;
      case FILM_6X7: ratio = "7:6"; break;
      case FILM_6X9: ratio = "3:2"; break;
      default: break;
    }
    fprintf(fp, "  \"frame_detection\": {\n");
    const bool has_axis = frame_det->axis == 'h' || frame_det->axis == 'v';
    fprintf(fp, "    \"axis\": %s, \"confidence\": %s, \"low_confidence\": %s, \"reason\": \"%s\",\n",
            has_axis ? (frame_det->axis == 'h' ? "\"h\"" : "\"v\"") : "null", json_num(frame_det->conf, "%.2f").c_str(),
            frame_det->conf < kLowFrameConfidence ? "true" : "false", json_escape(frame_det->reason).c_str());
    fprintf(fp, "    \"gaps\": [");
    int g_cnt = 0;
    for (int g = 0; g < 2; ++g) {
      if (frame_det->gaps[g][0] >= 0) {
        if (g_cnt > 0) fprintf(fp, ", ");
        fprintf(fp, "[%d, %d]", frame_det->gaps[g][0], frame_det->gaps[g][1]);
        g_cnt++;
      }
    }
    // segments: the frames on the strip, in strip order (secondaries are sorted by index, negative before the
    // primary); selected: the primary's position among them.
    int selected = 0;
    for (const FilmFrame& f : frame_det->secondaries) selected += f.index < 0 ? 1 : 0;
    fprintf(fp, "], \"segments\": %d, \"n_gaps\": %d, \"selected\": %d,\n",
            1 + (int)frame_det->secondaries.size(), frame_det->n_gaps, selected);
    fprintf(fp, "    \"aspect\": %s, \"format\": \"%s\", \"film_format\": \"%s\", \"orientation\": \"%s\",\n",
            json_num(aspect, "%.4g").c_str(), ratio, film_format_name(frame_det->format),
            json_escape(frame_det->orientation).c_str());
    fprintf(fp, "    \"rotation_deg\": %s, \"px_per_mm\": %s, \"light_clipped\": %s,\n",
            json_num(frame_det->rotation_deg, "%.3f").c_str(), json_num(frame_det->px_per_mm, "%.2f").c_str(),
            frame_det->light_clipped ? "true" : "false");
    fprintf(fp, "    \"crop\": [%d, %d, %d, %d], \"primary\": %s,\n", frame_det->crop[0], frame_det->crop[1],
            frame_det->crop[2], frame_det->crop[3], rect_json(pr).c_str());
    fprintf(fp, "    \"secondaries\": [");
    for (size_t i = 0; i < frame_det->secondaries.size(); ++i) {
      const FilmFrame& f = frame_det->secondaries[i];
      fprintf(fp, "%s{\"index\": %d, \"rect\": %s, \"visible\": [%d, %d, %d, %d], \"visible_fraction\": %s}",
              i ? ", " : "", f.index, rect_json(f.rect).c_str(), f.visible[0], f.visible[1], f.visible[2],
              f.visible[3], json_num(f.visible_fraction, "%.3f").c_str());
    }
    fprintf(fp, "],\n");
    fprintf(fp, "    \"film_base\": %s, \"base_patch\": %s, \"sprockets\": %d\n",
            frame_det->has_film_base ? rect_json(frame_det->film_base).c_str() : "null",
            frame_det->has_base ? rect_json(frame_det->base_patch).c_str() : "null",
            (int)frame_det->sprockets.size());
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
      fprintf(fp, "  \"profile\": {\"path\": \"%s\", \"type\": \"bundle\", \"bundle_target\": \"%s\", "
              "\"input_space\": \"%s\"},\n",
              json_escape(film_profile_path).c_str(), json_escape(bundle_target_name).c_str(),
              to_string(input_space));
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

  // auto: the conversion parameter search (multi_target_optimizer.h)
  if (auto_res) {
    const ParamSearchResult& a = *auto_res;
    fprintf(fp, "  \"auto\": {\n");
    fprintf(fp, "    \"target\": \"%s\", \"target_idx\": %d, \"e\": %s, \"g\": %s, \"b\": %s,\n",
            json_escape(a.target_name).c_str(), a.target_idx, json_num(a.params.e, "%.9g").c_str(),
            json_num(a.params.g, "%.9g").c_str(), json_num(a.params.b, "%.9g").c_str());
    fprintf(fp, "    \"residual\": %s, \"status\": \"%s\", \"e_norm\": %s, \"input_matrix\": [",
            json_num(a.residual, "%.6g").c_str(), to_string(a.status), json_num(a.e_norm, "%.9g").c_str());
    for (int i = 0; i < 9; ++i) fprintf(fp, "%s%s", (i > 0 ? ", " : ""), json_num(a.input_matrix[i], "%.9g").c_str());
    fprintf(fp, "],\n");
    fprintf(fp, "    \"frame_half\": [%d, %d, %d, %d], \"g_lo\": %s, \"g_hi\": %s, \"scene_ev\": %s,\n",
            auto_frame_half[0], auto_frame_half[1], auto_frame_half[2], auto_frame_half[3],
            json_num(a.g_lo, "%.6g").c_str(), json_num(a.g_hi, "%.6g").c_str(), json_num(a.scene_ev, "%.4g").c_str());
    fprintf(fp, "    \"headroom\": {\"applied\": %s, \"ev\": %s, \"reason\": \"%s\"},\n",
            headroom_applied(a.headroom) ? "true" : "false", json_num(a.headroom_ev, "%.4g").c_str(),
            to_string(a.headroom));
    fprintf(fp, "    \"rescue\": {\"branch\": \"%s\", \"white_clip\": %s, \"changed\": %s},\n",
            a.rescue == RescueOutcome::kNotNeeded ? "N" : "R", json_num(a.rescue_white_clip, "%.6g").c_str(),
            a.rescue == RescueOutcome::kTaken ? "true" : "false");
    fprintf(fp, "    \"screening\": [");
    for (size_t i = 0; i < a.screening.size(); ++i) {
      const auto& r = a.screening[i];
      fprintf(fp, "%s{\"target_idx\": %d, \"e\": %s, \"g\": %s, \"b\": %s, \"intent_loss\": %s}", (i > 0 ? ", " : ""),
              r.target_idx, json_num(r.params.e, "%.6g").c_str(), json_num(r.params.g, "%.6g").c_str(),
              json_num(r.params.b, "%.6g").c_str(),
              json_num(r.intent_loss, "%.6g").c_str());
    }
    fprintf(fp, "],\n    \"tournament\": [");
    for (size_t i = 0; i < a.tournament.size(); ++i) {
      const auto& r = a.tournament[i];
      fprintf(fp, "%s{\"target_idx\": %d, \"e\": %s, \"g\": %s, \"b\": %s, \"residual\": %s, \"status\": \"%s\", "
              "\"iters\": %d}", (i > 0 ? ", " : ""), r.target_idx, json_num(r.params.e, "%.6g").c_str(),
              json_num(r.params.g, "%.6g").c_str(), json_num(r.params.b, "%.6g").c_str(),
              json_num(r.residual, "%.6g").c_str(), to_string(r.status), r.steps);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "    \"model\": \"%s\", \"backend\": \"%s\", \"inferences\": %d, \"time_ms\": %s\n",
            json_escape(dino_model_path).c_str(), json_escape(dino_backend).c_str(), a.inferences,
            json_num(t_auto_ms, "%.1f").c_str());
    fprintf(fp, "  },\n");
  } else {
    fprintf(fp, "  \"auto\": null,\n");
  }

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

// How neg_process is meant to be used, by capture (to move to the docs later):
//
// - A single RAW captured under the narrowband RGB LED (tri) or white (d65) light, e.g.
//     neg_process --film_stock portra400 --light_source tri --roi 1700 0 6686 6376 -o out.tif scan.ARW
//   A tri bundle (input_space camera_raw_rgb) converts the sensor RGB with its crosstalk correction; -r, -g and -b
//   override it with one measured for this capture. A d65 bundle (rec2020_linear) converts LibRaw's white-light
//   decode (the camera's daylight white balance and colour matrix, to linear Rec.2020) with no crosstalk
//   correction, as negicc-station does, so it takes no -r, -g or -b, and no pixel shift. -p overrides the bundle with a negicc-station bundle JSON
//   (e.g. to test parity with negicc-station) or a single ICC profile (for research).
// - Four RAW files of a Sony A7R IV 4-shot pixel shift capture, merged into one image, the same way.
// - TODO: a capture whose R, G and B are already separated, like a film scanner's linear DNG or TIFF, with no
//   crosstalk to correct (-r 1 0 0 -g 0 1 0 -b 0 0 1). The decoders take only Bayer RAW files for now.
// - TODO: three images, each captured under one of the red, green and blue lights:
//     neg_process --film_stock portra400 --light_source tri -o out.tif red.ARW green.ARW blue.ARW
//   Each image gives one channel, so there is no crosstalk to correct (as -r 1 0 0 -g 0 1 0 -b 0 0 1). Not
//   supported yet.
//
// The parameter search (the calibration bracket, the exposure and the gains, solved with DINOv3 intent) runs
// whenever a profile is given, unless --gain_g or --gain_b is; then the conversion takes the values as given. -E
// keeps the exposure as given and solves only the gains, on the bracket --target picks (or the bundle's default).
// TODO: the bracket search on the frame --roi auto finds; for now it needs the frame given with --roi.
//
// --roi auto [h|v] finds the frame to crop (film_frame_detector.h): 35mm, 645, 6x6, 6x7 or 6x9 film on bare light, in
// a strip holder or in a frame holder, tilted up to 5 degrees, partial neighbour frames beside it. --film_format gives
// the format (more accurate than trying each); h or v gives the strip direction. The detection reads the RAW's CFA
// mosaic in place before anything is decoded, and only the frame (and the film base patch beside it, for
// --film_base_rgb auto) is decoded, e.g.
//     neg_process --film_stock portra400 --light_source tri --target 0 --roi auto --film_format 6x7 -o out.tif scan.ARW
int main(int ac, char *av[]) {
  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--list_profiles") == 0) {
      for (const std::string& key : list_embedded_bundles()) {
        // Keys are "<Stock>-<camera>-<Light>".
        std::string stock = key.substr(0, key.find('-'));
        for (char& ch : stock) ch = (char)std::tolower((unsigned char)ch);
        const std::string light = key.substr(key.find_last_of('-') + 1);
        printf("--film_stock %s --light_source %s  (%s)\n", stock.c_str(),
               light == "TriLED" ? "tri" : light == "D65" ? "d65" : light.c_str(), key.c_str());
      }
      return 0;
    }
  }
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
    .help("R (corrected) value is dot product of this 'r1 r2 r3' vector and 'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none).")
    .nargs(3)
    .default_value(std::vector<float>{1, 0, 0})
    .scan<'g', float>();
  parser.add_argument("-g", "--g_coeff")
    .help("G (corrected) value is dot product of this 'g1 g2 g3' vector and 'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none).")
    .nargs(3)
    .default_value(std::vector<float>{0, 1, 0})
    .scan<'g', float>();
  parser.add_argument("-b", "--b_coeff")
    .help("B (corrected) value is dot product of this 'b1 b2 b3' vector and 'R G B' values from linear RAW. "
          "Defaults to the profile bundle's crosstalk correction (a rec2020_linear bundle takes none).")
    .nargs(3)
    .default_value(std::vector<float>{0, 0, 1})
    .scan<'g', float>();
  parser.add_argument("-E", "--exposure_comp")
    .help("Multiplier for RGB values. Given, the parameter search keeps it and solves only the gains, on the bracket "
          "--target picks (or the bundle's default).")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--knee")
    .help("Soft knee highlight compression threshold in [0, 1], or 'auto' (the parameter search on an ICC profile or a "
          "--target bracket). Defaults to 0.95 (>=1.0 disables).")
    .default_value(std::string("0.95"));
  parser.add_argument("--no-metadata")
    .help("Disable automatic JSON metadata sidecar generation.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--gain_g")
    .help("Additional gain multiplier for the G channel. Defaults to 1.0. Given, it turns the parameter search off.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--gain_b")
    .help("Additional gain multiplier for the B channel. Defaults to 1.0. Given, it turns the parameter search off.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--profile_film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from profile, in the space the RAW is decoded to (sensor "
          "RGB, or linear Rec.2020 for a rec2020_linear bundle).")
    .nargs(3)
    .default_value(std::vector<int>{1, 1, 1})
    .scan<'i', int>();
  parser.add_argument("--film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from captured image, in the same space as --profile_film_base_rgb, or 'auto'.")
    .nargs(1, 3);
  parser.add_argument("--film_stock")
    .help("Film stock of the built-in profile bundle and intent adapter: ektar100, portra160 or portra400. Without it "
          "(-p with an ICC profile or a bundle whose name names no stock with an adapter), the parameter search falls "
          "back to the portra400 intent adapter.");
  parser.add_argument("--light_source")
    .help("Scanning light of the built-in profile bundle: tri (narrowband red, green and blue LED; sensor RGB and its "
          "crosstalk correction) or d65 (white; LibRaw's white-light decode to linear Rec.2020).");
  parser.add_argument("-p", "--film_profile")
    .help("Profile in place of the built-in bundle: a negicc-station profile bundle JSON or a split bundle's "
          "directory (e.g. to test parity with negicc-station), or a single ICC profile that applies to the corrected "
          "RGB values (for research).");
  parser.add_argument("-P", "--colorspace")
    .help("srgb, srgb-g10 or [ICC profile path]. If specified the corrected RGB will be converted using this as the output profile.");
  parser.add_argument("--roi")
    .help("ROI crop in sensor coordinates: x y w h [rot_cw [hflip [vflip]]], or 'auto [h|v]'.")
    .nargs(1, 7);
  parser.add_argument("--film_format")
    .help("Film format for --roi auto: 35mm, 645, 6x6, 6x7, 6x9, or auto (each in turn; less accurate).")
    .default_value(std::string("auto"));
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
    .help("Intent adapter for the parameter search in place of the film stock's built-in one: adapter.onnx or a "
          "directory holding it.");
  parser.add_argument("--list_profiles")
    .help("List the film stocks and light sources of the built-in profile bundles and exit.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("--dino_backend")
    .help("DINOv3 inference backend: auto, cpu (USE_CUDA=0 builds) or gpu (USE_CUDA=1 builds); defaults to auto.")
    .default_value(std::string("auto"));
  parser.add_argument("--dino_iters")
    .help("Maximum iterations for DINOv3 parameter solving: the fixed-point steps of each bracket in the parameter "
          "search (defaults to 8), else the solver's iterations (defaults to 4).")
    .scan<'i', int>()
    .default_value(4);
  parser.add_argument("--target", "--bundle_target")
    .help("Target index (0-based) or target name in JSON profile bundle.")
    .default_value(std::string(""));
  parser.add_argument("--flat")
    .help("Scan flat-field .npz file (measured on the empty gate), applied in sensor coordinates.");
  parser.add_argument("--station_egb")
    .help("Conversion parameters E, g and b as negicc-station applies them (multi_target_optimizer.h): the bundle's "
          "merged matrix (crosstalk, film base and exposure ratio) scaled per row by E, E*g and E*b. Turns the "
          "parameter search off.")
    .nargs(3)
    .scan<'g', float>();
  parser.add_argument("raw_files").nargs(1, 4);

  try {
    parser.parse_args(ac, av);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  const auto files = parser.get<std::vector<std::string>>("raw_files");
  if (files.size() == 2 || files.size() == 3) {
    // TODO: three images, one per light (see the capture notes above main).
    fprintf(stderr, "ERROR! Give one RAW file, or four of a pixel shift capture (one image per light is not "
                    "supported yet)\n");
    return 1;
  }
  auto r_coeff = parser.get<std::vector<float>>("--r_coeff");
  auto g_coeff = parser.get<std::vector<float>>("--g_coeff");
  auto b_coeff = parser.get<std::vector<float>>("--b_coeff");
  float global_exposure_comp = parser.get<float>("--exposure_comp");
  const std::string knee_str = parser.get<std::string>("--knee");
  const bool is_knee_auto = (knee_str == "auto");
  float knee = 0.95f;
  if (!is_knee_auto) {
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

  const std::string film_format_arg = parser.get<std::string>("--film_format");
  const FilmFormat film_format = film_format_from_name(film_format_arg);
  if (film_format == FILM_FORMAT_AUTO && film_format_arg != "auto") {
    fprintf(stderr, "ERROR! Invalid --film_format '%s' (must be 35mm, 645, 6x6, 6x7, 6x9 or auto)\n",
            film_format_arg.c_str());
    return 1;
  }
  if (parser.is_used("--film_format") && !is_roi_auto) {
    fprintf(stderr, "ERROR! --film_format requires --roi auto\n");
    return 1;
  }

  bool is_film_base_auto = false;
  bool film_base_cli_used = false;
  std::vector<int> user_film_base_rgb = {1, 1, 1};
  double user_film_base_rgb_d[3] = {1.0, 1.0, 1.0};

  if (parser.is_used("--film_base_rgb")) {
    const auto fb_tokens = parser.get<std::vector<std::string>>("--film_base_rgb");
    if (fb_tokens.size() == 1 && fb_tokens[0] == "auto") {
      is_film_base_auto = true;
    } else if (fb_tokens.size() == 3) {
      try {
        for (int c = 0; c < 3; ++c) {
          size_t pos = 0;
          const double val = std::stod(fb_tokens[c], &pos);
          if (pos != fb_tokens[c].size() || !std::isfinite(val)) throw std::invalid_argument("bad float");
          user_film_base_rgb_d[c] = val;
          user_film_base_rgb[c] = (int)std::lround(val);
        }
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

  negicc::FlatFieldPtr flat_field;
  if (parser.is_used("--flat")) {
    try {
      flat_field = negicc::load_flat_field_file(parser.get<std::string>("--flat"));
    } catch (const std::exception& e) {
      fprintf(stderr, "ERROR! %s\n", e.what());
      return 1;
    }
  }

  // The profile: -p, else the built-in bundle of --film_stock and --light_source.
  bool is_bundle = false;
  ProfileBundle bundle;
  std::string film_prof_arg;
  const std::string film_stock = parser.is_used("--film_stock") ? parser.get<std::string>("--film_stock") : "";
  if (parser.is_used("--light_source") && film_stock.empty()) {
    fprintf(stderr, "ERROR! --light_source requires --film_stock\n");
    return 1;
  }
  if (parser.is_used("--film_profile")) {
    film_prof_arg = parser.get<std::string>("--film_profile");
    is_bundle = load_bundle_arg(film_prof_arg, bundle);
    if (!is_bundle && !is_regular_file(resolve_profile_path(film_prof_arg))) {
      fprintf(stderr, "ERROR! No profile bundle JSON or ICC profile '%s'\n", film_prof_arg.c_str());
      return 1;
    }
  } else if (!film_stock.empty()) {
    if (!parser.is_used("--light_source")) {
      fprintf(stderr, "ERROR! --film_stock requires --light_source (tri or d65)\n");
      return 1;
    }
    std::string error;
    const std::string key = embedded_bundle_key(film_stock, parser.get<std::string>("--light_source"), error);
    if (key.empty()) {
      fprintf(stderr, "ERROR! %s (--list_profiles lists them)\n", error.c_str());
      return 1;
    }
    if (!load_embedded_bundle(key, bundle)) {
      fprintf(stderr, "ERROR! Cannot load built-in profile bundle %s\n", key.c_str());
      return 1;
    }
    is_bundle = true;
  }
  if (is_bundle) {
    printf("Profile bundle: %s (%s, %zu brackets)\n", bundle.key.c_str(), bundle.path.c_str(),
           bundle.targets.size());
  }
  const bool has_profile = is_bundle || !film_prof_arg.empty();

  int manual_target_idx = -1;
  if (parser.is_used("--target")) {
    if (!is_bundle) {
      fprintf(stderr, "ERROR! --target requires a JSON profile bundle\n");
      return 1;
    }
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
      return 1;
    }
  }

  const bool station_egb_given = parser.is_used("--station_egb");
  if (station_egb_given && !is_bundle) {
    fprintf(stderr, "ERROR! --station_egb requires a JSON profile bundle\n");
    return 1;
  }
  float station_egb[3] = {1.0f, 1.0f, 1.0f};
  if (station_egb_given) {
    const auto v = parser.get<std::vector<float>>("--station_egb");
    station_egb[0] = v[0];
    station_egb[1] = v[1];
    station_egb[2] = v[2];
  }

  // The parameter search runs with a profile unless --gain_g, --gain_b or --station_egb is given. On a bundle it is
  // negicc-station's conversion parameter search (multi_target_optimizer.h), which picks the bracket, the exposure
  // and the gains; with --target or -E (and on a single ICC profile) it solves the gains, and the exposure unless -E
  // gives it, on one profile.
  const bool exposure_given = parser.is_used("-E") || parser.is_used("--exposure_comp") || station_egb_given;
  const bool use_dino = has_profile && !parser.is_used("--gain_g") && !parser.is_used("--gain_b") && !station_egb_given;
  const bool bundle_search = use_dino && is_bundle && manual_target_idx < 0 && !exposure_given;
  if (parser.is_used("--dino") && !use_dino) {
    fprintf(stderr, "ERROR! --dino requires the parameter search: a profile (--film_stock and --light_source, or "
                    "--film_profile) without --gain_g or --gain_b\n");
    return 1;
  }
  if (is_knee_auto && (!use_dino || bundle_search)) {
    fprintf(stderr, "ERROR! --knee auto requires the parameter search on an ICC profile or a --target bracket\n");
    return 1;
  }
  if (bundle_search) {
    const char* conflicting[] = {"--film_base_rgb", "--profile_film_base_rgb"};
    for (const char* opt : conflicting) {
      if (parser.is_used(opt)) {
        fprintf(stderr, "ERROR! %s cannot be used with the bracket search (the bundle gives the film base); pick a "
                        "bracket with --target to use it\n", opt);
        return 1;
      }
    }
  }

  // The crosstalk correction: the bundle's, with the rows -r, -g and -b give in its place (for the search too).
  const bool row_given[3] = {parser.is_used("--r_coeff"), parser.is_used("--g_coeff"), parser.is_used("--b_coeff")};
  std::vector<float>* rows[3] = {&r_coeff, &g_coeff, &b_coeff};
  // A rec2020_linear (white-light) bundle takes the white-light decode, which has no crosstalk to correct and, as in
  // negicc-station, decodes single-shot captures only.
  // py-ref: negicc_station/src/pipeline_api.py @ 813166c (prepare_image)
  const bool white_light = is_bundle && bundle.input_space == InputSpace::kRec2020Linear;
  if (white_light) {
    if (row_given[0] || row_given[1] || row_given[2]) {
      fprintf(stderr, "ERROR! -r, -g and -b cannot be used with the rec2020_linear profile bundle %s: its white-light "
                      "decode has no crosstalk to correct\n", bundle.key.c_str());
      return 1;
    }
    if (files.size() == 4) {
      fprintf(stderr, "ERROR! The rec2020_linear profile bundle %s converts single-shot captures only, not pixel "
                      "shift\n", bundle.key.c_str());
      return 1;
    }
  }
  if (is_bundle) {
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
  const bool has_crosstalk = row_given[0] || row_given[1] || row_given[2] || (is_bundle && bundle.has_crosstalk);
  const RawDecode raw_decode = white_light ? RawDecode::kWhiteLight
                               : has_crosstalk ? RawDecode::kCameraRgb
                                               : RawDecode::kRec2020;

  DinoV3Engine engine;
  std::string dino_path;
  const std::string dino_backend = parser.get<std::string>("--dino_backend");
  if (use_dino) {
    const EmbeddedFile* trt = find_embedded_backbone(true);
    const EmbeddedFile* onnx = find_embedded_backbone(false);
    const EmbeddedFile* backbone = trt ? trt : onnx;
    if (!backbone) {
      fprintf(stderr, "ERROR! This neg_process was built without the DINOv3 backbone of the parameter search (make "
                      "dinov3_model, or make dinov3_engine for USE_CUDA=1, then rebuild); give --gain_g or --gain_b "
                      "to convert without it\n");
      return 1;
    }
    dino_path = std::string("[built-in] ") + backbone->name;
    printf("Loading DINOv3 backbone %s (backend: %s)...\n", dino_path.c_str(), dino_backend.c_str());
    engine.load_memory(trt ? trt->data : nullptr, trt ? trt->size : 0, onnx ? onnx->data : nullptr,
                       onnx ? onnx->size : 0, dino_backend);
    // The file that runs: the .onnx when the engine was not loaded.
    if (trt && engine.backend_name() != std::string("gpu (TensorRT)")) {
      dino_path = std::string("[built-in] ") + onnx->name;
    }
    if (parser.is_used("--dino")) {
      std::string adapter = parser.get<std::string>("--dino");
      if (is_directory(adapter)) adapter = (std::filesystem::path(adapter) / "adapter.onnx").string();
      engine.load_adapter_file(adapter);
    } else {
      std::string adapter_dir;
      const std::string profile_name = !film_stock.empty() ? film_stock : is_bundle ? bundle.key : film_prof_arg;
      const EmbeddedFile* a = find_embedded_adapter(profile_name, adapter_dir);
      if (!a) {
        fprintf(stderr, "ERROR! No intent adapter for profile '%s'\n", profile_name.c_str());
        return 1;
      }
      engine.set_adapter(a->data, a->size, adapter_dir);
    }
    printf("DINOv3 engine loaded [%s] adapter %s (memory: %.1f MB)\n", engine.backend_name(),
           engine.adapter_name().c_str(), (double)engine.memory_bytes() / (1024.0 * 1024.0));
  }

  if (is_film_base_auto && !is_roi_auto) {
    fprintf(stderr, "ERROR! --film_base_rgb auto requires --roi auto\n");
    return 1;
  }
  if (is_film_base_auto && !parser.is_used("--profile_film_base_rgb") && !(is_bundle && bundle.has_film_base)) {
    fprintf(stderr, "ERROR! --film_base_rgb auto requires --profile_film_base_rgb\n");
    return 1;
  }

  double exif_shutter_d = 0.0, exif_iso_d = 0.0;
  const bool has_exif_exp = read_exif_exposure_d(files[0], exif_shutter_d, exif_iso_d);
  auto get_scan_exposure = [&](const LibRaw& r, double& shutter_out, double& iso_out) {
    shutter_out = (has_exif_exp && exif_shutter_d > 0.0)
                      ? exif_shutter_d
                      : (r.imgdata.other.shutter > 0 ? (double)r.imgdata.other.shutter : 0.125);
    iso_out = (has_exif_exp && exif_iso_d > 0.0)
                  ? std::round(exif_iso_d)
                  : (r.imgdata.other.iso_speed > 0 ? std::round((double)r.imgdata.other.iso_speed) : 100.0);
  };

  bool has_auto_res = false;
  ParamSearchResult auto_res;
  int auto_frame_half[4] = {0, 0, 0, 0};  // the frame (x, y, w, h) in the half-size image
  double t_auto_ms = 0.0;
  HalfSizeAnalysis analyse;
  if (bundle_search) {
    analyse = [&](const LibRaw& raw) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(raw, scan_shutter_s, scan_iso);
      // --roi halved, or the whole image.
      const int half_w = raw.imgdata.sizes.iwidth, half_h = raw.imgdata.sizes.iheight;
      int* bbox = auto_frame_half;
      bbox[0] = 0;
      bbox[1] = 0;
      bbox[2] = half_w;
      bbox[3] = half_h;
      if (has_roi) {
        bbox[0] = roi_coords[0] / 2;
        bbox[1] = roi_coords[1] / 2;
        bbox[2] = std::max(1, (roi_coords[0] + roi_coords[2]) / 2 - bbox[0]);
        bbox[3] = std::max(1, (roi_coords[1] + roi_coords[3]) / 2 - bbox[1]);
        clamp_rect(bbox, half_w, half_h);
      }
      // Step 3 & Step 4: zero-copy RgbView over the half-size image, cropped and oriented in place.
      negicc::RgbView half_view = negicc::RgbView::from_interleaved_rgb(
          &raw.imgdata.image[0][0], half_w, half_h, 4, (ptrdiff_t)half_w * 4);
      negicc::RgbView frame = half_view.crop_and_orient(bbox[0], bbox[1], bbox[2], bbox[3], rot_cw, hflip, vflip);
      ParamSearchOptions opts;
      if (parser.is_used("--dino_iters")) opts.max_steps = parser.get<int>("--dino_iters");
      if (!search_conversion_params(bundle, engine, frame, scan_shutter_s, scan_iso, opts, auto_res)) {
        throw std::runtime_error("the conversion parameter search failed");
      }
      has_auto_res = true;
      t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    };
  }

  FrameDetection frame_det;
  FrameDetectionParams det_params;
  det_params.format = film_format;
  det_params.axis_hint = roi_axis_hint;
  int decode_region[4] = {0, 0, 0, 0};  // sensor x, y, w, h that load_raw decoded for --roi auto
  double t_frame_det_ms = 0.0;
  bool rebate_applied = false;
  RawRegionFinder find_frame;
  if (is_roi_auto && files.size() != 4) {
    find_frame = [&](LibRaw& raw, int roi[4]) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        fprintf(stderr, "ERROR! --roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        return false;
      }
      det_params.crop_align = mosaic.step;
      if (!detect_film_frame(mosaic, det_params, frame_det)) {
        fprintf(stderr, "ERROR! Film frame detection failed\n");
        return false;
      }
      t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      roi_coords[0] = frame_det.crop[0];
      roi_coords[1] = frame_det.crop[1];
      roi_coords[2] = frame_det.crop[2];
      roi_coords[3] = frame_det.crop[3];
      has_roi = true;
      // The decoded region: the crop, grown for --film_base_rgb auto to take in the base patch when there is one (it
      // lies next to the frame: an inter-frame gap or the rebate), aligned to the CFA period.
      int x0 = frame_det.crop[0], y0 = frame_det.crop[1];
      int x1 = x0 + frame_det.crop[2], y1 = y0 + frame_det.crop[3];
      if (is_film_base_auto && frame_det.has_base) {
        const int* b = frame_det.base_box;
        const int a = mosaic.step;
        x0 = std::min(x0, b[0] / a * a);
        y0 = std::min(y0, b[1] / a * a);
        x1 = std::max(x1, std::min(mosaic.width, (b[0] + b[2] + a - 1) / a * a));
        y1 = std::max(y1, std::min(mosaic.height, (b[1] + b[3] + a - 1) / a * a));
      }
      roi[0] = decode_region[0] = x0;
      roi[1] = decode_region[1] = y0;
      roi[2] = decode_region[2] = x1 - x0;
      roi[3] = decode_region[3] = y1 - y0;
      return true;
    };
  }

  auto print_frame_detection = [&]() {
    const char* axis_name = frame_det.axis == 'h' ? "h" : (frame_det.axis == 'v' ? "v" : "undecided");
    printf("Film frame: axis=%s format=%s orientation=%s rotation=%.2f deg gaps=%d frames=%d conf=%.2f%s%s\n",
           axis_name, film_format_name(frame_det.format), frame_det.orientation.c_str(), frame_det.rotation_deg,
           frame_det.n_gaps, 1 + (int)frame_det.secondaries.size(), frame_det.conf,
           frame_det.reason.empty() ? "" : " reason: ", frame_det.reason.c_str());
    if (frame_det.n_gaps > 0) {
      printf("  Gaps (sensor px):");
      for (int g = 0; g < 2; ++g)
        if (frame_det.gaps[g][0] >= 0) printf(" [%d, %d]", frame_det.gaps[g][0], frame_det.gaps[g][1]);
      printf("\n");
    }
    printf("  Frame (sensor px): x=%d y=%d w=%d h=%d (frame centre %.1f, %.1f, size %.1f x %.1f, %.1f px/mm)\n",
           roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3], frame_det.primary.rect.cx,
           frame_det.primary.rect.cy, frame_det.primary.rect.w, frame_det.primary.rect.h, frame_det.px_per_mm);
    for (const FilmFrame& f : frame_det.secondaries)
      printf("  Frame %+d (sensor px): visible x=%d y=%d w=%d h=%d (%.0f%% of a frame)\n", f.index, f.visible[0],
             f.visible[1], f.visible[2], f.visible[3], 100.0f * f.visible_fraction);
  };

  const bool half_size_flag = parser.get<bool>("--half_size");
  const int qual_flag = parser.get<int>("--quality");
  const bool no_crop_flag = parser.get<bool>("--no_crop");

  // Determine whether the 5-step zero-copy CFA pipeline applies (Step 1..5).
  bool use_cfa_pipeline = false;
  negicc::InputFormat input_format = (files.size() == 4) ? negicc::InputFormat::SONY_PIXEL_SHIFT_4
                                                         : negicc::InputFormat::CFA_BAYER_OR_XTRANS;
  const double t_raw_0 = omp_get_wtime();
  LibRaw *proc = nullptr;
  float cfa_white_level = 65535.0f;
  int cfa_rx = 0, cfa_ry = 0, cfa_rw = 0, cfa_rh = 0;
  negicc::RgbView full_half_view;
  negicc::RgbView oriented_half_view;

  if (files.size() == 4 || (raw_decode == RawDecode::kCameraRgb && qual_flag < 0)) {
    LibRaw* probe = new LibRaw();
    printf("Loading RAW file %s\n", files[0].c_str());
    int ret = probe->open_file(files[0].c_str());
    if (ret != LIBRAW_SUCCESS) {
      fprintf(stderr, "Cannot open %s: %s\n", files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    printf("Image size: %dx%d\n", probe->imgdata.sizes.iwidth, probe->imgdata.sizes.iheight);
    if ((ret = probe->unpack()) != LIBRAW_SUCCESS) {
      fprintf(stderr, "Cannot unpack %s: %s\n", files[0].c_str(), libraw_strerror(ret));
      delete probe;
      return 1;
    }
    if (files.size() == 4) {
      if (!is_sony_a7rm4(probe)) {
        fprintf(stderr, "ERROR: 4-shot pixel shift merge is only available with Sony A7RM4 camera (got %s %s)\n",
                probe->imgdata.idata.make, probe->imgdata.idata.model);
        delete probe;
        return 1;
      }
      use_cfa_pipeline = true;
    } else if (is_sony_cfa_geometry(probe) && probe->imgdata.rawdata.raw_image &&
               probe->imgdata.color.cblack[4] == 0 && probe->imgdata.color.cblack[5] == 0) {
      use_cfa_pipeline = true;
    }
    if (use_cfa_pipeline) {
      proc = probe;
    } else {
      delete probe;
    }
  }

  double t_raw_decode_ms = 0.0;

  if (use_cfa_pipeline) {
    (void)input_format;
    // Step 1: Per-channel black subtraction in-place on the unpacked CFA mosaic.
    if (!subtract_cfa_black_in_place(proc, cfa_white_level)) {
      fprintf(stderr, "ERROR! Invalid CFA mosaic geometry in %s\n", files[0].c_str());
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

    // Step 2: Downsizing via zero-copy RgbView over the CFA mosaic.
    const int half_w = src_w / 2;
    const int half_h = src_h / 2;
    std::shared_ptr<const negicc::FlatTables> flat_half_tables;
    negicc::FlatSampleArgs flat_half{};
    if (flat_field) {
      try {
        flat_half_tables = flat_field->tables(half_w, half_h, true);
        flat_half = flat_half_tables->host();
      } catch (const std::exception& e) {
        fprintf(stderr, "ERROR! %s\n", e.what());
        delete proc;
        return 1;
      }
    }
    full_half_view = negicc::RgbView::from_cfa_half(
        cfa_origin, cfa_pitch, src_w, src_h, filters, nullptr, flat_half, flat_half_tables);

    // Step 3: Crop, rotate, flip & inline ROI search on the zero-copy view.
    if (is_roi_auto) {
      const double t_det_0 = omp_get_wtime();
      FilmImage mosaic = full_half_view.to_film_image(cfa_white_level, 2);
      if (mosaic.width <= 0 || mosaic.height <= 0) {
        fprintf(stderr, "ERROR! --roi auto needs a CFA RAW whose margins lie inside the raw buffer\n");
        delete proc;
        return 1;
      }
      det_params.crop_align = mosaic.step;
      if (!detect_film_frame(mosaic, det_params, frame_det)) {
        fprintf(stderr, "ERROR! Film frame detection failed\n");
        delete proc;
        return 1;
      }
      t_frame_det_ms = (omp_get_wtime() - t_det_0) * 1000.0;
      roi_coords[0] = frame_det.crop[0];
      roi_coords[1] = frame_det.crop[1];
      roi_coords[2] = frame_det.crop[2];
      roi_coords[3] = frame_det.crop[3];
      has_roi = true;
      print_frame_detection();

      if (frame_det.has_base && !is_film_base_auto) {
        const int* b = frame_det.base_box;
        if (b[0] < roi_coords[0] || b[1] < roi_coords[1] ||
            b[0] + b[2] > roi_coords[0] + roi_coords[2] ||
            b[1] + b[3] > roi_coords[1] + roi_coords[3]) {
          frame_det.has_base = false;
        }
      }
      if (frame_det.has_base && flat_field) {
        FilmRect patch = frame_det.base_patch;
        patch.cx *= 0.5f;
        patch.cy *= 0.5f;
        patch.w *= 0.5f;
        patch.h *= 0.5f;
        const FilmImage half_fi = full_half_view.to_film_image(cfa_white_level);
        float rgb[3], std_d;
        if (film_sample_patch(half_fi, patch, rgb, &std_d)) {
          for (int c = 0; c < 3; ++c) frame_det.base_rgb[c] = rgb[c];
        } else {
          frame_det.has_base = false;
        }
      }
      if (frame_det.has_base) {
        if (is_film_base_auto) {
          rebate_applied = true;
          printf("  Rebate base: R=%.0f G=%.0f B=%.0f (std %.3f D) applied\n",
                 frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2], frame_det.base_std_d);
        } else {
          printf("  Rebate base: R=%.0f G=%.0f B=%.0f (std %.3f D) sampled, unused\n",
                 frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2], frame_det.base_std_d);
        }
      } else if (is_film_base_auto) {
        printf("Warning: --film_base_rgb auto requested but no validated rebate found\n");
      }
      if (frame_det.conf < kLowFrameConfidence) {
        printf("Warning: low detection confidence (conf=%.2f%s%s); the crop is the detected frame, check it\n",
               frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
      }
      if (use_dino && frame_det.axis == 'v' && rot_cw == 0) {
        printf("Warning: axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
      }
    }

    cfa_rx = 0;
    cfa_ry = 0;
    cfa_rw = src_w;
    cfa_rh = src_h;
    if (has_roi) {
      if (roi_coords[0] < 0 || roi_coords[1] < 0 || roi_coords[0] >= src_w || roi_coords[1] >= src_h ||
          roi_coords[2] <= 0 || roi_coords[3] <= 0) {
        fprintf(stderr, "ERROR! ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
                roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3], src_w, src_h);
        delete proc;
        return 1;
      }
      cfa_rx = roi_coords[0];
      cfa_ry = roi_coords[1];
      cfa_rw = std::min(roi_coords[2], src_w - cfa_rx);
      cfa_rh = std::min(roi_coords[3], src_h - cfa_ry);
    } else if (!no_crop_flag && !station_egb_given && files.size() == 1) {
#if LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21)
      const auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
#else
      const auto& crop_ref = proc->imgdata.sizes.raw_inset_crop;
#endif
      const bool has_inset_crop = crop_ref.cwidth > 0 && crop_ref.cheight > 0 &&
                                  crop_ref.cleft != 0xffff && crop_ref.ctop != 0xffff;
      if (has_inset_crop && (crop_ref.cleft || crop_ref.ctop)) {
        cfa_rx = std::max(0, std::min((int)crop_ref.cleft, src_w - 1));
        cfa_ry = std::max(0, std::min((int)crop_ref.ctop, src_h - 1));
        cfa_rw = std::max(1, std::min((int)crop_ref.cwidth, src_w - cfa_rx));
        cfa_rh = std::max(1, std::min((int)crop_ref.cheight, src_h - cfa_ry));
      }
    }

    auto_frame_half[0] = has_roi ? cfa_rx / 2 : 0;
    auto_frame_half[1] = has_roi ? cfa_ry / 2 : 0;
    auto_frame_half[2] = has_roi ? std::max(1, (cfa_rx + cfa_rw) / 2 - auto_frame_half[0]) : half_w;
    auto_frame_half[3] = has_roi ? std::max(1, (cfa_ry + cfa_rh) / 2 - auto_frame_half[1]) : half_h;
    clamp_rect(auto_frame_half, half_w, half_h);
    const negicc::RgbView search_half_view = full_half_view.crop_and_orient(
        auto_frame_half[0], auto_frame_half[1], auto_frame_half[2], auto_frame_half[3],
        rot_cw, hflip, vflip);
    int out_half[4] = {
        cfa_rx / 2,
        cfa_ry / 2,
        std::max(1, (cfa_rx + cfa_rw) / 2 - cfa_rx / 2),
        std::max(1, (cfa_ry + cfa_rh) / 2 - cfa_ry / 2),
    };
    clamp_rect(out_half, half_w, half_h);
    oriented_half_view = full_half_view.crop_and_orient(
        out_half[0], out_half[1], out_half[2], out_half[3],
        rot_cw, hflip, vflip);

    // Step 4: Conversion parameter search inline on the zero-copy oriented RgbView.
    if (bundle_search) {
      const double t_auto_0 = omp_get_wtime();
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(*proc, scan_shutter_s, scan_iso);
      ParamSearchOptions opts;
      double fb_override_d[3] = {user_film_base_rgb_d[0], user_film_base_rgb_d[1], user_film_base_rgb_d[2]};
      if (rebate_applied) {
        fb_override_d[0] = (double)frame_det.base_rgb[0];
        fb_override_d[1] = (double)frame_det.base_rgb[1];
        fb_override_d[2] = (double)frame_det.base_rgb[2];
        opts.film_base_rgb_d = fb_override_d;
        opts.film_base_from_scan = true;
      } else if (film_base_cli_used) {
        opts.film_base_rgb_d = fb_override_d;
        opts.film_base_from_scan = false;
      }
      if (parser.is_used("--dino_iters")) opts.max_steps = parser.get<int>("--dino_iters");
      if (!search_conversion_params(bundle, engine, search_half_view, scan_shutter_s, scan_iso, opts, auto_res)) {
        fprintf(stderr, "ERROR! The conversion parameter search failed\n");
        delete proc;
        return 1;
      }
      has_auto_res = true;
      t_auto_ms = (omp_get_wtime() - t_auto_0) * 1000.0;
    }
    t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0 - t_auto_ms - t_frame_det_ms;
  } else {
    proc = load_raw(files[0], true,
                    half_size_flag,
                    qual_flag,
                    !no_crop_flag && !has_roi && !is_roi_auto,
                    raw_decode,
                    (!is_roi_auto && has_roi) ? roi_coords : nullptr,
                    analyse, find_frame, flat_field.get());
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
    t_raw_decode_ms = (omp_get_wtime() - t_raw_0) * 1000.0 - t_auto_ms - t_frame_det_ms;

    if (is_roi_auto) {
      roi_coords[0] = frame_det.crop[0];
      roi_coords[1] = frame_det.crop[1];
      roi_coords[2] = frame_det.crop[2];
      roi_coords[3] = frame_det.crop[3];
      has_roi = true;
      print_frame_detection();

      // The decoded image holds decode_region, at half its size with -H.
      const float dscale = (float)decode_region[2] / (float)std::max(1, (int)proc->imgdata.sizes.iwidth);
      if (frame_det.has_base) {
        // Sampled only where decoded: without --film_base_rgb auto the region was not grown to take in the patch.
        const int* b = frame_det.base_box;
        if (b[0] < decode_region[0] || b[1] < decode_region[1] ||
            b[0] + b[2] > decode_region[0] + decode_region[2] ||
            b[1] + b[3] > decode_region[1] + decode_region[3]) {
          frame_det.has_base = false;
        }
      }
      if (frame_det.has_base) {
        // The base in the decoded image's units (its colour space and scale): the same patch, sampled again there.
        FilmRect patch = frame_det.base_patch;
        patch.cx = (patch.cx - decode_region[0]) / dscale;
        patch.cy = (patch.cy - decode_region[1]) / dscale;
        patch.w /= dscale;
        patch.h /= dscale;
        const FilmImage decoded = film_image_from_interleaved(&proc->imgdata.image[0][0], proc->imgdata.sizes.iwidth,
                                                              proc->imgdata.sizes.iheight, 4);
        float rgb[3], std_d;
        if (film_sample_patch(decoded, patch, rgb, &std_d)) {
          for (int c = 0; c < 3; ++c) frame_det.base_rgb[c] = rgb[c];
        } else {
          frame_det.has_base = false;
        }
      }
      if (frame_det.has_base) {
        if (is_film_base_auto) {
          rebate_applied = true;
          printf("  Rebate base: R=%.0f G=%.0f B=%.0f (std %.3f D) applied\n",
                 frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2], frame_det.base_std_d);
        } else {
          printf("  Rebate base: R=%.0f G=%.0f B=%.0f (std %.3f D) sampled, unused\n",
                 frame_det.base_rgb[0], frame_det.base_rgb[1], frame_det.base_rgb[2], frame_det.base_std_d);
        }
      } else if (is_film_base_auto) {
        printf("Warning: --film_base_rgb auto requested but no validated rebate found\n");
      }
      if (frame_det.conf < kLowFrameConfidence) {
        printf("Warning: low detection confidence (conf=%.2f%s%s); the crop is the detected frame, check it\n",
               frame_det.conf, frame_det.reason.empty() ? "" : ", reason: ", frame_det.reason.c_str());
      }
      if (use_dino && frame_det.axis == 'v' && rot_cw == 0) {
        printf("Warning: axis=v, rot=0 (vertical frame without rotation may degrade DINOv3 intent accuracy)\n");
      }

      // The crop out of the decoded region (one copy, with the rotation and flips).
      const int cx = (int)std::lround((roi_coords[0] - decode_region[0]) / dscale);
      const int cy = (int)std::lround((roi_coords[1] - decode_region[1]) / dscale);
      const int cw = (int)std::lround(roi_coords[2] / dscale);
      const int ch = (int)std::lround(roi_coords[3] / dscale);
      if (!extract_roi_and_transform(proc, cx, cy, cw, ch, rot_cw, hflip, vflip)) {
        fprintf(stderr, "ERROR! Failed to extract detected frame ROI\n");
        proc->free_image();
        delete proc;
        return 1;
      }
      printf("Extracted subrect: %dx%d (rot=%d, hflip=%d, vflip=%d)\n",
             proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight, rot_cw, hflip ? 1 : 0, vflip ? 1 : 0);
    }
  }

  const int orig_w = proc->imgdata.sizes.raw_width ? proc->imgdata.sizes.raw_width : proc->imgdata.sizes.width;
  const int orig_h = proc->imgdata.sizes.raw_height ? proc->imgdata.sizes.raw_height : proc->imgdata.sizes.height;

  printf("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  printf("Shutter Speed: %f\n", proc->imgdata.other.shutter);

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
    effective_film_base_rgb = {(int)std::lround(frame_det.base_rgb[0]), (int)std::lround(frame_det.base_rgb[1]),
                               (int)std::lround(frame_det.base_rgb[2])};
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
      sidecar_film_base_rgb = {(int)std::lround(frame_det.base_rgb[0]), (int)std::lround(frame_det.base_rgb[1]),
                             (int)std::lround(frame_det.base_rgb[2])};
    } else {
      sidecar_film_base_rgb = {1, 1, 1};
    }
    effective_profile_film_base_rgb = {1, 1, 1};
    effective_film_base_rgb = {1, 1, 1};
    film_base_applied = false;
    film_base_source = "none";
  }

  if (bundle_search) {
    // The bundle's film base is folded into the solved exposure and gains (not applied as a film base scale).
    sidecar_film_base_rgb = sidecar_profile_film_base_rgb;
    film_base_source = "bundle";
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

  if (is_bundle && !bundle_search) {
    float g_lo = 0.0f, g_hi = 0.0f, scene_ev = 0.0f;
    int num_feasible = 0;
    if (use_cfa_pipeline) {
      compute_bundle_enclosures(bundle, oriented_half_view, g_lo, g_hi, scene_ev, num_feasible);
    } else {
      compute_bundle_enclosures(bundle,
                                (const uint16_t (*)[4])proc->imgdata.image,
                                proc->imgdata.sizes.iwidth,
                                proc->imgdata.sizes.iheight,
                                g_lo, g_hi, scene_ev, num_feasible);
    }
  } else if (!is_bundle) {
    if (!film_prof_arg.empty() && !load_film_profile(film_prof_arg, film_prof)) {
      fprintf(stderr, "ERROR! Cannot load film profile %s\n", film_prof_arg.c_str());
    }
  }

  DinoSolveResult dino_res;

  if (has_auto_res) {
    bundle_target_idx = auto_res.target_idx;
    bundle_target_name = auto_res.target_name;
    film_prof = bundle.targets[bundle_target_idx].film_prof;
    film_prof.rebind_pointers();
    // The solved conversion written as the bundle's crosstalk with an exposure and gains (the film base and
    // exposure ratio folded in), so the sidecar's cli_equivalent reproduces it with --target.
    global_exposure_comp = (float)(auto_res.row_scale[0] * auto_res.params.e * auto_res.e_norm);
    gain_g = (float)(auto_res.row_scale[1] / auto_res.row_scale[0] * auto_res.params.g);
    gain_b = (float)(auto_res.row_scale[2] / auto_res.row_scale[0] * auto_res.params.b);
    adjust_correction_matrix(r_coeff, g_coeff, b_coeff, global_exposure_comp, gain_g, gain_b,
                             effective_profile_film_base_rgb, effective_film_base_rgb, merged_matrix);
    printf("Auto: %s E %.4f g %.4f b %.4f (resid %.4f, %s), %d inferences\n", auto_res.target_name.c_str(),
           auto_res.params.e, auto_res.params.g, auto_res.params.b, auto_res.residual, to_string(auto_res.status),
           auto_res.inferences);
    auto ms = [](std::chrono::steady_clock::duration d) {
      return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    printf("Timings (ms): render %lld, infer %lld, residual %lld, search %lld, total %lld\n",
           ms(auto_res.render_time), ms(auto_res.infer_time), ms(auto_res.residual_time), ms(auto_res.search_time),
           ms(auto_res.total_time));
    printf("Solved (absolute): -E %.9g --gain_g %.9g --gain_b %.9g --target %d\n\n", global_exposure_comp, gain_g,
           gain_b, bundle_target_idx);
  } else if (use_dino) {
    if (!is_bundle && !film_prof.data.has_profile) {
      fprintf(stderr, "ERROR! The parameter search requires a film profile to evaluate intent losses.\n");
      proc->free_image();
      delete proc;
      return 1;
    }

    SolverConfig cfg;
    cfg.knee = knee;
    cfg.knee_auto = is_knee_auto;

    const int max_iters = parser.get<int>("--dino_iters");
    IntentGains user_gains{global_exposure_comp, gain_g, gain_b};

    if (is_bundle) {
      // -E without --target: the bracket the bundle picks without the search.
      bundle_target_idx = manual_target_idx >= 0 ? manual_target_idx
                                                 : select_bundle_target_without_dino(bundle, kDefaultMinTargetIdx);
      printf("Selected target: %s\n", bundle.targets[bundle_target_idx].display_name.c_str());
      const auto& tgt = bundle.targets[bundle_target_idx];
      bundle_target_name = tgt.name;
      film_prof = tgt.film_prof;
      film_prof.rebind_pointers();
      set_target_bounds(cfg, tgt);
    } else {
      cfg.prof = &film_prof.data;
    }

    const bool solve_ok = use_cfa_pipeline
        ? solve_dinov3_intent(engine, cfg, merged_matrix, oriented_half_view, user_gains, pin_mask, max_iters, dino_res)
        : solve_dinov3_intent(engine, cfg, merged_matrix, (const uint16_t (*)[4])proc->imgdata.image,
                              proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                              user_gains, pin_mask, max_iters, dino_res);
    if (!solve_ok) {
      fprintf(stderr, "ERROR! DINOv3 intent solver failed.\n");
      proc->free_image();
      delete proc;
      return 1;
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
    if (station_egb_given) {
      double scan_shutter_s = 0.125, scan_iso = 100.0;
      get_scan_exposure(*proc, scan_shutter_s, scan_iso);
      double fb_override_d[3] = {user_film_base_rgb_d[0], user_film_base_rgb_d[1], user_film_base_rgb_d[2]};
      const double* fb_ptr = nullptr;
      bool fb_from_scan = false;
      if (rebate_applied) {
        fb_override_d[0] = (double)frame_det.base_rgb[0];
        fb_override_d[1] = (double)frame_det.base_rgb[1];
        fb_override_d[2] = (double)frame_det.base_rgb[2];
        fb_ptr = fb_override_d;
        fb_from_scan = true;
      } else if (film_base_cli_used) {
        fb_ptr = fb_override_d;
      }
      float base_matrix[9];
      double row_scale[3];
      compute_station_merged_matrix(bundle, scan_shutter_s, scan_iso, fb_ptr, fb_from_scan,
                                    base_matrix, nullptr, row_scale);
      negicc::candidate_matrix(base_matrix, station_egb[0], station_egb[1], station_egb[2], merged_matrix);
      global_exposure_comp = (float)(row_scale[0] * (double)station_egb[0]);
      gain_g = (float)(row_scale[1] / row_scale[0] * (double)station_egb[1]);
      gain_b = (float)(row_scale[2] / row_scale[0] * (double)station_egb[2]);
    } else if (!(pin_mask & PIN_E) && bundle.targets[bundle_target_idx].e_center > 0.0f) {
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
  } else if (has_profile) {
    encoding = ENCODE_NONE;
    if (!is_bundle) {
      attach_profile = film_prof_arg;
    }
  }

  // Step 5: Full export (the ONLY step that allocates the cropped output RGB buffer on use_cfa_pipeline).
  std::vector<uint16_t> cfa_export_rgb;
  if (use_cfa_pipeline) {
    const double t_step5_debayer_0 = omp_get_wtime();
    const int src_w = proc->imgdata.sizes.width;
    const int src_h = proc->imgdata.sizes.height;
    const size_t cfa_pitch = proc->imgdata.sizes.raw_pitch / 2;
    const unsigned int filters = proc->imgdata.idata.filters;
    const uint16_t* cfa_origin = proc->imgdata.rawdata.raw_image +
                                 (size_t)proc->imgdata.sizes.top_margin * cfa_pitch +
                                 proc->imgdata.sizes.left_margin;
    int norm_rot = rot_cw;
    bool norm_hf = hflip, norm_vf = vflip;
    normalize_rotation(norm_rot, norm_hf, norm_vf);
    const bool swap_dims = (norm_rot == 90 || norm_rot == 270);
    int dst_w = 0, dst_h = 0;

    auto free_libraw_cfa = [](LibRaw* r) {
      if (!r) return;
      if (r->imgdata.rawdata.raw_alloc && !r->imgdata.image) {
        r->imgdata.image = (ushort (*)[4])r->imgdata.rawdata.raw_alloc;
        r->imgdata.rawdata.raw_alloc = nullptr;
        r->imgdata.rawdata.raw_image = nullptr;
        r->free_image();
      }
      r->recycle_datastream();
    };

    if (half_size_flag) {
      dst_w = oriented_half_view.width;
      dst_h = oriented_half_view.height;
      cfa_export_rgb.resize((size_t)dst_w * dst_h * 3);
      oriented_half_view.materialize(cfa_export_rgb.data());
      free_libraw_cfa(proc);
    } else {
      dst_w = swap_dims ? cfa_rh : cfa_rw;
      dst_h = swap_dims ? cfa_rw : cfa_rh;
      cfa_export_rgb.resize((size_t)dst_w * dst_h * 3);
      std::shared_ptr<const negicc::FlatTables> flat_full_tables;
      negicc::FlatSampleArgs flat_full{};
      if (flat_field) {
        try {
          flat_full_tables = flat_field->tables(src_w, src_h, false);
          flat_full = flat_full_tables->host();
        } catch (const std::exception& e) {
          fprintf(stderr, "ERROR! %s\n", e.what());
          delete proc;
          return 1;
        }
      }
      if (files.size() == 1) {
        printf("Debayer quality: grain-aware\n");
        negicc::GrainAwareDebayerArgs ga{};
        ga.cfa_bayer = cfa_origin;
        ga.src_w = src_w;
        ga.src_h = src_h;
        ga.rx = cfa_rx;
        ga.ry = cfa_ry;
        ga.rw = cfa_rw;
        ga.rh = cfa_rh;
        ga.rot_cw = rot_cw;
        ga.hflip = hflip;
        ga.vflip = vflip;
        ga.dst_w = dst_w;
        ga.dst_h = dst_h;
        ga.filters = filters;
        ga.white_level = cfa_white_level;
        ga.eps = 4096.0f;
        ga.rgb_out = cfa_export_rgb.data();
        ga.flat = flat_full;
        ga.flat_x0 = cfa_rx;
        ga.flat_y0 = cfa_ry;
        ga.cfa_pitch = cfa_pitch;
        negicc::cpu_grain_aware_debayer(ga);
        free_libraw_cfa(proc);
      } else {
        printf("Merging 4 images...\n");
        negicc::PixelShiftMergeArgs pm{};
        pm.src_w = src_w;
        pm.src_h = src_h;
        pm.rx = cfa_rx;
        pm.ry = cfa_ry;
        pm.rw = cfa_rw;
        pm.rh = cfa_rh;
        pm.rot_cw = rot_cw;
        pm.hflip = hflip;
        pm.vflip = vflip;
        pm.dst_w = dst_w;
        pm.dst_h = dst_h;
        pm.filters = filters;
        pm.white_level = cfa_white_level;
        pm.rgb_out = cfa_export_rgb.data();
        pm.flat = flat_full;
        pm.flat_x0 = cfa_rx;
        pm.flat_y0 = cfa_ry;
        pm.cfa_pitch = cfa_pitch;
        negicc::cpu_pixel_shift_merge_frame(pm, 0, cfa_origin);
        free_libraw_cfa(proc);
        for (int mi = 1; mi < 4; ++mi) {
          LibRaw frame_proc;
          printf("Loading RAW file %s\n", files[mi].c_str());
          int fret = frame_proc.open_file(files[mi].c_str());
          if (fret != LIBRAW_SUCCESS || (fret = frame_proc.unpack()) != LIBRAW_SUCCESS) {
            fprintf(stderr, "Cannot open/unpack %s: %s\n", files[mi].c_str(), libraw_strerror(fret));
            delete proc;
            return 1;
          }
          if (!is_sony_a7rm4(&frame_proc) || frame_proc.imgdata.sizes.width != src_w ||
              frame_proc.imgdata.sizes.height != src_h) {
            fprintf(stderr, "ERROR: 4-shot pixel shift merge is only available with matching Sony A7RM4 frames\n");
            delete proc;
            return 1;
          }
          float fw_white = 0.0f;
          if (!subtract_cfa_black_in_place(&frame_proc, fw_white)) {
            delete proc;
            return 1;
          }
          const size_t fpitch = frame_proc.imgdata.sizes.raw_pitch / 2;
          const uint16_t* fcfa = frame_proc.imgdata.rawdata.raw_image +
                                 (size_t)frame_proc.imgdata.sizes.top_margin * fpitch +
                                 frame_proc.imgdata.sizes.left_margin;
          pm.cfa_pitch = fpitch;
          negicc::cpu_pixel_shift_merge_frame(pm, mi, fcfa);
        }
      }
    }
    proc->imgdata.sizes.iwidth = dst_w;
    proc->imgdata.sizes.iheight = dst_h;
    proc->imgdata.idata.colors = 3;
    proc->imgdata.params.output_bps = 16;
    printf("Processed image size: %dx%d\n", dst_w, dst_h);
    t_raw_decode_ms += (omp_get_wtime() - t_step5_debayer_0) * 1000.0;
  }

  const int width = proc->imgdata.sizes.iwidth;
  const int height = proc->imgdata.sizes.iheight;
  const double t_start = omp_get_wtime();

  if (!cfa_export_rgb.empty()) {
    uint16_t* rgb3 = cfa_export_rgb.data();
    if (encoding == ENCODE_SRGB || encoding == ENCODE_SRGB_LINEAR) {
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          uint16_t* px = &rgb3[((size_t)j * width + i) * 3];
          Rgb v = process_pixel(px, merged_matrix, knee, film_prof.data);
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
            hXYZ, TYPE_XYZ_FLT, hCustomOut, TYPE_RGB_16, INTENT_PERCEPTUAL, 0);
        std::vector<float> xyz_row(width * 3);

        #pragma omp for schedule(static)
        for (int j = 0; j < height; ++j) {
          for (int i = 0; i < width; ++i) {
            const uint16_t* px = &rgb3[((size_t)j * width + i) * 3];
            Rgb v = process_pixel(px, merged_matrix, knee, film_prof.data);
            xyz_row[i * 3 + 0] = v.r;
            xyz_row[i * 3 + 1] = v.g;
            xyz_row[i * 3 + 2] = v.b;
          }
          cmsDoTransform(thread_xform, xyz_row.data(), &rgb3[(size_t)j * width * 3], width);
        }
        cmsDeleteTransform(thread_xform);
      }
      cmsCloseProfile(hXYZ);
      cmsCloseProfile(hCustomOut);
    } else {
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
          uint16_t* px = &rgb3[((size_t)j * width + i) * 3];
          Rgb v = load_u16(px);
          v = apply_matrix(merged_matrix, v);
          v = apply_knee(v, knee);
          px[0] = quantize_u16(v.r);
          px[1] = quantize_u16(v.g);
          px[2] = quantize_u16(v.b);
        }
      }
    }
  } else if (encoding == ENCODE_SRGB || encoding == ENCODE_SRGB_LINEAR) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < height; ++j) {
      for (int i = 0; i < width; ++i) {
        ushort* px = proc->imgdata.image[j * width + i];
        Rgb v = process_pixel(px, merged_matrix, knee, film_prof.data);
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
          Rgb v = process_pixel(px, merged_matrix, knee, film_prof.data);
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
  int ret = write_tiff(proc, attach_profile, output,
                       is_bundle ? embedded_icc : std::vector<uint8_t>{},
                       cfa_export_rgb.empty() ? nullptr : cfa_export_rgb.data());
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
      roi_source = "auto";  // also at low confidence (frame_detection.low_confidence): there is no fallback region
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
                       is_bundle ? bundle.path : film_prof_arg,
                       merged_matrix,
                       global_exposure_comp, gain_g, gain_b,
                       pin_mask,
                       knee, knee_source,
                       use_dino && !has_auto_res, dino_res,
                       dino_path, dino_backend,
                       cs_name,
                       t_raw_decode_ms, t_frame_det_ms, t_full_conversion_ms, t_tiff_write_ms,
                       is_bundle, bundle_target_name, bundle_target_idx,
                       has_auto_res ? &auto_res : nullptr, auto_frame_half, t_auto_ms, bundle.input_space);
  }

  proc->free_image();
  delete proc;
  return ret;
}
