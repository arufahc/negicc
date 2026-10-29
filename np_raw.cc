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

#include "np_raw.h"

#include <omp.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "np_geometry.h"

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


bool film_image_from_raw(LibRaw& proc, FilmImage& img) {
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
  const bool ok = analyse(*proc);

  // raw2image_ex subtracted the black level from imgdata.color and resized imgdata.sizes; put back what unpack() read.
  proc->free_image();
  proc->raw2image_start();
  return ok;
}


// White-light decode of a rec2020_linear profile bundle, in two steps on an unpacked proc. First, in place on the
// CFA mosaic and in sensor-linear units, the black level is subtracted and the camera's daylight white balance
// (pre_mul over its green) applied. Then (decode_white_light) LibRaw demosaics bilinearly straight to linear Rec.2020
// (output_color 8) through its own camera matrix with no rescaling, so values stay in sensor units. LibRaw clamps
// its output to [0, 65535]; going to Rec.2020 directly, rather than to sRGB and then to Rec.2020, keeps the colours
// between the sRGB and Rec.2020 gamuts that the sRGB clamp lost. This relies on the pinned LibRaw (3rd_party/libraw):
// older releases (0.20) do not implement the Rec.2020 output. When flat is given, it is applied in place on the CFA
// mosaic after black subtraction and white balance.
// py-ref: negicc_station/src/raw_processor.cpp @ 813166c (load_raw_white_light; output_color 8 from branch libraw-from-negicc)
// py-ref: negicc_station/src/image_capture.cpp @ 813166c (CapturedImage::decode_white_light_locked; no sRGB matrix from libraw-from-negicc)
//
// Sets clip_fraction to the fraction of photosites at or above the white level. False, with the reason on stderr,
// when the RAW has no 3-colour CFA or no daylight white balance.
static bool prepare_white_light_mosaic(LibRaw* proc, double& clip_fraction, const negicc::FlatField* flat = nullptr) {
  auto& S = proc->imgdata.sizes;
  auto& C = proc->imgdata.color;
  uint16_t* raw = proc->imgdata.rawdata.raw_image;
  // A linear (demosaiced) DNG has its three colours per pixel instead of a CFA mosaic: the same black subtraction and
  // daylight white balance apply to each colour.
  const int ncomp = proc->imgdata.rawdata.color3_image ? 3 : proc->imgdata.rawdata.color4_image ? 4 : 0;
  const bool linear = !raw && ncomp && proc->imgdata.idata.colors == 3;
  if (linear) {
    raw = ncomp == 3 ? &proc->imgdata.rawdata.color3_image[0][0] : &proc->imgdata.rawdata.color4_image[0][0];
    if (flat) {
      fprintf(stderr, "ERROR! --flat applies to a CFA RAW, not a linear DNG\n");
      return false;
    }
  } else if (!raw || !proc->imgdata.idata.filters || proc->imgdata.idata.colors != 3) {
    fprintf(stderr, "ERROR! The white-light decode needs a 3-colour CFA RAW file or a linear DNG\n");
    return false;
  }
  const int w = S.width, h = S.height, top = S.top_margin, left = S.left_margin;
  const size_t pitch = S.raw_pitch / 2 / (linear ? ncomp : 1);  // pixels per row of the raw buffer
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
  if (linear) {
    #pragma omp parallel for reduction(+ : clipped) schedule(static)
    for (int y = 0; y < h; ++y) {
      uint16_t* p = raw + ((size_t)(y + top) * pitch + left) * ncomp;
      for (int x = 0; x < w; ++x, p += ncomp) {
        bool clip = false;
        for (int c = 0; c < 3; ++c) {
          clip |= p[c] >= clip_level;
          const float v = std::max(0, (int)p[c] - (int)C.black - (int)C.cblack[c]) * wb[c];
          p[c] = (uint16_t)std::min(std::nearbyint(v), 65535.0f);
        }
        clipped += clip ? 1 : 0;
      }
    }
  }
  #pragma omp parallel for reduction(+ : clipped) schedule(static)
  for (int y = 0; y < (linear ? 0 : h); ++y) {
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
  const bool xtrans = proc->imgdata.idata.filters == 9;
  const int prev_threads = xtrans ? omp_get_max_threads() : 0;
  if (xtrans) omp_set_num_threads(1);
  const int ret = proc->dcraw_process();
  if (xtrans) omp_set_num_threads(prev_threads);
  if (ret != LIBRAW_SUCCESS) {
    fprintf(stderr, "ERROR! Cannot process the white-light decode: %s\n", libraw_strerror(ret));
    return false;
  }
  return true;
}

bool subtract_cfa_black_in_place(LibRaw* proc, float& out_white_level) {
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
  uint8_t col4[36];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      col4[r * 6 + c] = (uint8_t)(proc->COLOR(r, c) & 3);
    }
  }
  uint16_t* cfa = proc->imgdata.rawdata.raw_image + (size_t)top * pitch + left;
  #pragma omp parallel for schedule(static)
  for (int r = 0; r < h; ++r) {
    uint16_t* row = cfa + (size_t)r * pitch;
    const uint8_t* pat = col4 + (r % 6) * 6;
    if (!pattern) {
      for (int c = 0; c < w; ++c) {
        const int v = (int)row[c] - cblk[pat[c % 6]];
        row[c] = (v > 0) ? (uint16_t)v : 0;
      }
    } else {
      const unsigned pr = (unsigned)r % dr;
      for (int c = 0; c < w; ++c) {
        const int blk = cblk[pat[c % 6]] + (int)C.cblack[6 + pr * dc + ((unsigned)c % dc)];
        const int v = (int)row[c] - blk;
        row[c] = (v > 0) ? (uint16_t)v : 0;
      }
    }
  }
  C.black = 0;
  std::memset(C.cblack, 0, sizeof(C.cblack));
  return true;
}

LibRaw* load_raw(const std::string& fn, bool debayer, int qual, bool crop,
                 RawDecode decode, const int* roi,
                 const HalfSizeAnalysis& analyse, const RawRegionFinder& find_region,
                 const negicc::FlatField* flat,
                 LibRaw* unpacked_proc) {
  const bool white_light = debayer && decode == RawDecode::kWhiteLight;
  int ret;
  LibRaw* proc = unpacked_proc;
  if (!proc) {
    proc = new LibRaw();
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
  }
  // A linear (demosaiced) DNG: three samples a pixel, or four of which the first three are RGB (e.g. a film
  // scanner's RGB and infrared); the fourth is left out.
  const bool linear_dng = !proc->imgdata.idata.filters &&
                          (proc->imgdata.rawdata.color3_image || proc->imgdata.rawdata.color4_image) &&
                          (proc->imgdata.idata.colors == 3 || proc->imgdata.idata.colors == 4);
  if (linear_dng && proc->imgdata.idata.colors == 4) {
    printf("Linear DNG with 4 samples a pixel: the first three taken as RGB, the fourth (e.g. infrared) left out\n");
    proc->imgdata.idata.colors = 3;
  }
  if (!(proc->imgdata.idata.filters || proc->imgdata.idata.colors == 1 || linear_dng)) {
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
      if (!analyse(*proc)) {
        proc->free_image();
        delete proc;
        return NULL;
      }
      proc->free_image();
      proc->raw2image_start();
      proc->imgdata.color.black = 0;
      std::memset(proc->imgdata.color.cblack, 0, sizeof(proc->imgdata.color.cblack));
    }
  } else {
    if (flat) {
      float dummy_white = 0.0f;
      if (!proc->imgdata.idata.filters || proc->imgdata.idata.colors != 3 ||
          !subtract_cfa_black_in_place(proc, dummy_white)) {
        fprintf(stderr, "ERROR! --flat requires a 3-colour CFA RAW file with valid sensor margins\n");
        delete proc;
        return NULL;
      }
      uint8_t col3[36];
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
          const int k = proc->COLOR(r, c);
          col3[r * 6 + c] = (uint8_t)(k == 3 ? 1 : k);
        }
      }
      const auto& S = proc->imgdata.sizes;
      const size_t pitch = S.raw_pitch / 2;
      try {
        flat->apply_cfa(proc->imgdata.rawdata.raw_image + (size_t)S.top_margin * pitch + S.left_margin,
                        pitch, S.width, S.height, col3);
      } catch (const std::exception& e) {
        fprintf(stderr, "ERROR! %s\n", e.what());
        delete proc;
        return NULL;
      }
      proc->imgdata.params.user_black = 0;
      for (int i = 0; i < 4; ++i) proc->imgdata.params.user_cblack[i] = 0;
    }
    if (analyse && !read_half_size(proc, analyse)) {
      delete proc;
      return NULL;
    }
    if (flat) {
      proc->imgdata.color.black = 0;
      std::memset(proc->imgdata.color.cblack, 0, sizeof(proc->imgdata.color.cblack));
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
    // Camera linear RGB (output_color = 0) for a crosstalk correction (a camera_raw_rgb bundle or -r -g -b);
    // otherwise Rec.2020 (output_color = 8). The white-light decode set its own (prepare_white_light_mosaic).
    const bool camera_rgb = decode == RawDecode::kCameraRgb;
    if (!white_light) proc->imgdata.params.output_color = camera_rgb ? 0 : 8;

    int effective_qual = (qual < 0) ? 0 : qual;
    printf("Debayer quality: %d\n", effective_qual);
    proc->imgdata.params.half_size = 0;
    proc->imgdata.params.user_qual = effective_qual;
    proc->imgdata.params.use_auto_wb = 0;
    // Unit multipliers on every path: the white-light decode has applied its white balance to the mosaic.
    proc->imgdata.params.user_mul[0] = 1;
    proc->imgdata.params.user_mul[1] = 1;
    proc->imgdata.params.user_mul[2] = 1;
    proc->imgdata.params.user_mul[3] = 1;

    const int a = (proc->imgdata.idata.filters == 9) ? 6 : 2;
    const int pad = a * 2;
    const int sensor_w = proc->imgdata.rawdata.sizes.width ? proc->imgdata.rawdata.sizes.width : proc->imgdata.sizes.width;
    const int sensor_h = proc->imgdata.rawdata.sizes.height ? proc->imgdata.rawdata.sizes.height : proc->imgdata.sizes.height;

    int sub_x = 0, sub_y = 0, sub_w = 0, sub_h = 0;
    if (roi) {
      // Align LibRaw's cropbox origin down to the CFA period (6x6 for X-Trans, 2x2 for Bayer) so LibRaw's
      // raw2image does not shift the mosaic phase, and include a 2-tile halo on every ROI (--roi auto and
      // explicit --roi alike) so LibRaw's demosaic stencil has identical boundary neighbors.
      const int box_x = std::max(0, ((roi[0] - pad) / a) * a);
      const int box_y = std::max(0, ((roi[1] - pad) / a) * a);
      const int box_x1 = std::min(sensor_w, ((roi[0] + roi[2] + pad + a - 1) / a) * a);
      const int box_y1 = std::min(sensor_h, ((roi[1] + roi[3] + pad + a - 1) / a) * a);
      const int box_w = box_x1 - box_x;
      const int box_h = box_y1 - box_y;
      proc->imgdata.params.cropbox[0] = box_x;
      proc->imgdata.params.cropbox[1] = box_y;
      proc->imgdata.params.cropbox[2] = box_w;
      proc->imgdata.params.cropbox[3] = box_h;
      sub_x = roi[0] - box_x;
      sub_y = roi[1] - box_y;
      sub_w = roi[2];
      sub_h = roi[3];
    } else {
      auto& crop_ref = proc->imgdata.sizes.raw_inset_crops[0];
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
      if (!decode_white_light(proc, false, effective_qual)) {
        delete proc;
        return NULL;
      }
    } else {
      proc->dcraw_process();
    }
    if (roi &&
        (sub_x != 0 || sub_y != 0 || sub_w != proc->imgdata.sizes.iwidth || sub_h != proc->imgdata.sizes.iheight)) {
      if (!extract_roi_and_transform(proc, sub_x, sub_y, sub_w, sub_h, 0, false, false)) {
        proc->free_image();
        delete proc;
        return NULL;
      }
    }
  }
  printf("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}
