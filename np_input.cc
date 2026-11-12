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

// neg_process's inputs: image geometry, RAW decoding with LibRaw, TIFF and JPEG files, and the inputs besides one
// RAW (np_input.h).

#include "np_log.h"
#include "np_input.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <lcms2.h>
#include <tiffio.h>
#include <jpeglib.h>  // after <cstdio>: it needs FILE and size_t

// --------------------------------------------------------------------------------------------------------------------
// Geometry
// --------------------------------------------------------------------------------------------------------------------

void normalize_rotation(int& rot_cw, bool& hflip, bool& vflip) {
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
    log_error("Failed to allocate memory for transformed ROI (%dx%d)\n", dst_w, dst_h);
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

// py-ref: negicc_station/src/film_frame.py @ 93be1ec (clamp)
void clamp_rect(int r[4], int bw, int bh) {
  r[2] = std::min(r[2], bw);
  r[3] = std::min(r[3], bh);
  r[0] = std::min(std::max(0, r[0]), bw - r[2]);
  r[1] = std::min(std::max(0, r[1]), bh - r[3]);
}

void detect_frame_or_fallback(const FilmImage& mosaic, const FrameDetectionParams& params,
                                     FrameDetection& det) {
  if (detect_film_frame(mosaic, params, det)) return;
  const std::string why = det.reason;
  det = FrameDetection();
  det.ok = false;
  det.conf = 0.0f;
  det.reason = why.empty() ? "no film frame found" : why;
  log_warn("film frame detection failed (%s); falling back to the full visible frame\n",
          det.reason.c_str());
}

// --------------------------------------------------------------------------------------------------------------------
// RAW decoding
// --------------------------------------------------------------------------------------------------------------------

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

namespace {

// A DNG whose UniqueCameraModel starts with spaces (e.g. SilverFast's "  Multiple Frame Scanner FS") leaves
// LibRaw::open_file with make[0] == '\0' (tiff.cpp splits UniqueCameraModel at the first space), so LibRaw rejects it
// with LIBRAW_FILE_UNSUPPORTED. A stream wrapper that advances IFD0's UniqueCameraModel (0xc614) offset past any
// leading spaces lets LibRaw open the file in place without copying the image data.
class PatchedHeaderStream : public LibRaw_bigfile_datastream {
 public:
  explicit PatchedHeaderStream(const char* fname) : LibRaw_bigfile_datastream(fname) {
    if (!valid()) return;
    uint8_t hdr[8] = {0};
    seek(0, SEEK_SET);
    if (LibRaw_bigfile_datastream::read(hdr, 1, 8) != 8) return;
    const bool le = hdr[0] == 'I' && hdr[1] == 'I';
    const bool be = hdr[0] == 'M' && hdr[1] == 'M';
    if (!le && !be) return;
    auto u16 = [&](const uint8_t* p) -> uint16_t {
      return le ? (uint16_t)(p[0] | (p[1] << 8)) : (uint16_t)((p[0] << 8) | p[1]);
    };
    auto u32 = [&](const uint8_t* p) -> uint32_t {
      return le ? ((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24))
                : (((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3]);
    };
    const uint32_t ifd0 = u32(hdr + 4);
    if (ifd0 + 2 > size()) return;
    uint8_t nbuf[2] = {0};
    seek(ifd0, SEEK_SET);
    if (LibRaw_bigfile_datastream::read(nbuf, 1, 2) != 2) return;
    const uint16_t ntags = u16(nbuf);
    if (ifd0 + 2 + (INT64)ntags * 12 > size()) return;
    std::vector<uint8_t> entries((size_t)ntags * 12);
    if (LibRaw_bigfile_datastream::read(entries.data(), 1, entries.size()) != (int)entries.size()) return;
    bool has_dng = false;
    uint32_t ucm_cnt = 0, ucm_off = 0;
    for (uint16_t i = 0; i < ntags; ++i) {
      const uint8_t* e = entries.data() + (size_t)i * 12;
      const uint16_t tag = u16(e);
      if (tag == 0xc612) has_dng = true;
      if (tag == 0xc614 && u16(e + 2) == 2) {
        ucm_cnt = u32(e + 4);
        ucm_off = u32(e + 8);
      }
    }
    if (has_dng && ucm_cnt > 4 && ucm_cnt <= 256 && (INT64)ucm_off + ucm_cnt <= size()) {
      patch_.resize(ucm_cnt);
      seek(ucm_off, SEEK_SET);
      if (LibRaw_bigfile_datastream::read(patch_.data(), 1, ucm_cnt) == (int)ucm_cnt && patch_[0] == ' ') {
        for (size_t i = 0; i < patch_.size() && patch_[i] == ' '; ++i) patch_[i] = '_';
        patch_off_ = ucm_off;
        patched_ = true;
      }
    }
    seek(0, SEEK_SET);
  }
  bool is_patched() const { return patched_; }
  int read(void* ptr, size_t sz, size_t nmemb) override {
    const INT64 pos = tell();
    const int r = LibRaw_bigfile_datastream::read(ptr, sz, nmemb);
    if (patched_ && r > 0) {
      const INT64 end = pos + (INT64)r * (INT64)sz;
      const INT64 p0 = patch_off_, p1 = patch_off_ + (INT64)patch_.size();
      const INT64 lo = std::max(pos, p0), hi = std::min(end, p1);
      if (lo < hi) {
        std::memcpy(static_cast<uint8_t*>(ptr) + (lo - pos), patch_.data() + (lo - p0), (size_t)(hi - lo));
      }
    }
    return r;
  }

 private:
  INT64 patch_off_ = 0;
  std::vector<uint8_t> patch_;
  bool patched_ = false;
};

}  // namespace

int open_raw_file(LibRaw* proc, const std::string& fn) {
  int ret = proc->open_file(fn.c_str());
  if (ret == LIBRAW_FILE_UNSUPPORTED) {
    PatchedHeaderStream stream(fn.c_str());
    if (stream.valid() && stream.is_patched()) {
      ret = proc->open_datastream(&stream);
      if (ret == LIBRAW_SUCCESS) return ret;
    }
  }
  return ret;
}

int open_and_unpack_raw_file(LibRaw* proc, const std::string& fn) {
  int ret = proc->open_file(fn.c_str());
  if (ret == LIBRAW_FILE_UNSUPPORTED) {
    PatchedHeaderStream stream(fn.c_str());
    if (stream.valid() && stream.is_patched()) {
      ret = proc->open_datastream(&stream);
      if (ret != LIBRAW_SUCCESS) return ret;
      log_info("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
      return proc->unpack();
    }
  }
  if (ret != LIBRAW_SUCCESS) return ret;
  log_info("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc->unpack();
}


bool film_image_from_raw(LibRaw& proc, FilmImage& img) {
  const auto& sz = proc.imgdata.sizes;
  const auto& C = proc.imgdata.color;
  const float black = (float)C.black;
  float cblack[4];
  for (int c = 0; c < 4; ++c) cblack[c] = black + (float)C.cblack[c];
  const float white = std::max(1.0f, (float)C.maximum - black);
  const unsigned filters = proc.imgdata.idata.filters;
  const ushort* raw = proc.imgdata.rawdata.raw_image;
  if (!raw) {
    const int ncomp = proc.imgdata.rawdata.color4_image ? 4 : proc.imgdata.rawdata.color3_image ? 3 : 0;
    if (filters || !ncomp || (proc.imgdata.idata.colors != 3 && proc.imgdata.idata.colors != 4)) return false;
    const ushort* linear =
        ncomp == 4 ? &proc.imgdata.rawdata.color4_image[0][0] : &proc.imgdata.rawdata.color3_image[0][0];
    const ptrdiff_t pitch = sz.raw_pitch / 2 / ncomp;
    if (sz.width <= 0 || sz.height <= 0 || sz.top_margin + sz.height > sz.raw_height ||
        sz.left_margin + sz.width > pitch) {
      return false;
    }
    const ushort* origin = linear + ((size_t)sz.top_margin * pitch + sz.left_margin) * ncomp;
    img = film_image_from_interleaved(origin, sz.width, sz.height, ncomp, pitch * ncomp, white);
    for (int c = 0; c < 3; ++c) img.ch[c].black = cblack[c];
    return true;
  }
  const ptrdiff_t stride = sz.raw_pitch / 2;
  // The visible area must lie inside the raw buffer (prepare_white_light_mosaic refuses such a RAW the same way).
  if (sz.width <= 0 || sz.height <= 0 || sz.top_margin + sz.height > sz.raw_height ||
      sz.left_margin + sz.width > stride) {
    return false;
  }
  const ushort* origin = raw + (size_t)sz.top_margin * stride + sz.left_margin;
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

static ushort (*alloc_half(const LibRaw* proc, int& hw, int& hh))[4] {
  hw = (proc->imgdata.sizes.iwidth + 1) / 2;
  hh = (proc->imgdata.sizes.iheight + 1) / 2;
  return (ushort(*)[4])::calloc((size_t)hw * hh, sizeof(ushort[4]));
}

// Replaces proc's image with img (w x h, ::calloc-allocated; proc takes ownership: LibRaw 0.22 free_image
// calls ::free on an unregistered buffer).
static void replace_image(LibRaw* proc, ushort (*img)[4], int w, int h) {
  proc->free_image();
  proc->imgdata.image = img;
  proc->imgdata.sizes.iwidth = w;
  proc->imgdata.sizes.iheight = h;
}

// LibRaw's half-size readout does not serve an X-Trans RAW (filters 9): its 2x2 quads hold several photosites of one
// colour, which LibRaw's OpenMP loop writes in a race, and the quads of only green photosites get no R or B. Replaces
// LibRaw's full-size readout in proc with a half-size image of each quad's mean per colour; R and B of the green-only
// quads are the mean of their left and right neighbours, as LibRaw's pre_interpolate fills them. False when out of
// memory.
static bool shrink_xtrans(LibRaw* proc) {
  const int w = proc->imgdata.sizes.iwidth, h = proc->imgdata.sizes.iheight;
  int hw = 0, hh = 0;
  const ushort(*full)[4] = proc->imgdata.image;
  ushort(*half)[4] = alloc_half(proc, hw, hh);
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
  replace_image(proc, half, hw, hh);
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
    log_error("The parameter search supports only 3-colour Bayer and X-Trans RAW files\n");
    return false;
  }
  const int half_size = proc->imgdata.params.half_size;
  proc->imgdata.params.half_size = xtrans ? 0 : 1;
  const int ret = proc->raw2image_ex(1);
  proc->imgdata.params.half_size = half_size;
  if (ret != LIBRAW_SUCCESS || (xtrans && !shrink_xtrans(proc))) {
    log_error("Cannot read the half-size RAW: %s\n",
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
// mosaic after black subtraction and white balance. load_raw then multiplies the mosaic by the gains that balance the
// camera channels on the film base (ChannelBalance, scale_white_light_mosaic), so that a light of another colour
// than the profile's is corrected before the camera matrix mixes the channels, not after.
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
  const int ncomp = proc->imgdata.rawdata.color4_image ? 4 : proc->imgdata.rawdata.color3_image ? 3 : 0;
  const bool linear = !raw && ncomp && proc->imgdata.idata.colors == 3;
  if (linear) {
    raw = ncomp == 4 ? &proc->imgdata.rawdata.color4_image[0][0] : &proc->imgdata.rawdata.color3_image[0][0];
    if (flat) {
      log_error("--flat applies to a CFA RAW, not a linear DNG\n");
      return false;
    }
  } else if (!raw || !proc->imgdata.idata.filters || proc->imgdata.idata.colors != 3) {
    log_error("The white-light decode needs a 3-colour CFA RAW file or a linear DNG\n");
    return false;
  }
  const int w = S.width, h = S.height, top = S.top_margin, left = S.left_margin;
  const size_t pitch = S.raw_pitch / 2 / (linear ? ncomp : 1);  // pixels per row of the raw buffer
  if (w <= 0 || h <= 0 || top + h > S.raw_height || left + w > (int)pitch) {
    log_error("RAW margins exceed the raw buffer\n");
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
    log_error("No daylight white balance (pre_mul) for this camera; the white-light decode needs one\n");
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
      log_error("%s\n", e.what());
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

struct LibRawRgb : LibRaw {
  using LibRaw::calloc;
  using LibRaw::convert_to_rgb;
  using LibRaw::libraw_internal_data;

  static void to_rgb(LibRaw* proc) {
    auto& out = (proc->*(&LibRawRgb::libraw_internal_data)).output_data;
    if (!out.histogram) {
      out.histogram = (int(*)[LIBRAW_HISTOGRAM_SIZE])(proc->*(&LibRawRgb::calloc))(1, sizeof(*out.histogram) * 4);
    }
    proc->imgdata.sizes.width = proc->imgdata.sizes.iwidth;
    proc->imgdata.sizes.height = proc->imgdata.sizes.iheight;
    (proc->*(&LibRawRgb::convert_to_rgb))();
  }
};

// Demosaics the mosaic prepare_white_light_mosaic prepared to linear Rec.2020 in imgdata.image (half: half-size
// decode—LibRaw's half-size readout on Bayer, raw2image_ex + shrink_xtrans + convert_to_rgb on X-Trans, or a 2x2
// downsample on a linear DNG; qual: LibRaw's user_qual, negicc-station uses 0, bilinear). The cropbox, if any, is set
// by the caller. False when LibRaw cannot process it.
static bool decode_white_light(LibRaw* proc, bool half, int qual) {
  const bool linear = !proc->imgdata.idata.filters;
  const bool xtrans = proc->imgdata.idata.filters == 9;
  proc->imgdata.params.half_size = (half && !xtrans) ? 1 : 0;
  proc->imgdata.params.user_qual = qual;
  if (half && xtrans) {
    // LibRaw's half-size readout does not serve an X-Trans RAW (filters == 9): read the full mosaic with raw2image_ex,
    // shrink each 2x2 quad in shrink_xtrans (as read_half_size does) and convert directly to linear Rec.2020 without
    // running single-threaded xtrans_interpolate on the analysis and balance passes.
    const int ret = proc->raw2image_ex(1);
    if (ret != LIBRAW_SUCCESS || !shrink_xtrans(proc)) {
      log_error("Cannot process the white-light decode: %s\n",
                ret != LIBRAW_SUCCESS ? libraw_strerror(ret) : "out of memory");
      return false;
    }
    LibRawRgb::to_rgb(proc);
    return true;
  }
  const int prev_threads = xtrans ? omp_get_max_threads() : 0;
  if (xtrans) omp_set_num_threads(1);
  const int ret = proc->dcraw_process();
  if (xtrans) omp_set_num_threads(prev_threads);
  if (ret != LIBRAW_SUCCESS) {
    log_error("Cannot process the white-light decode: %s\n", libraw_strerror(ret));
    return false;
  }
  // LibRaw's raw2image_start leaves IO.shrink = 0 when filters == 0 (a linear DNG), so dcraw_process decodes at full
  // sensor dimensions even when half_size = 1. Downsample 2x2 here so callers receive half-resolution coordinates.
  if (half && linear) {
    const int w = proc->imgdata.sizes.iwidth, h = proc->imgdata.sizes.iheight;
    int hw = 0, hh = 0;
    const ushort(*full)[4] = proc->imgdata.image;
    ushort(*shrunk)[4] = alloc_half(proc, hw, hh);
    if (!shrunk) {
      log_error("Cannot process the white-light decode: out of memory\n");
      return false;
    }
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < hh; ++y) {
      const int y1 = std::min(2 * y + 1, h - 1);
      for (int x = 0; x < hw; ++x) {
        const int x1 = std::min(2 * x + 1, w - 1);
        const ushort* a = full[(size_t)(2 * y) * w + 2 * x];
        const ushort* b = full[(size_t)(2 * y) * w + x1];
        const ushort* c = full[(size_t)y1 * w + 2 * x];
        const ushort* d = full[(size_t)y1 * w + x1];
        for (int k = 0; k < 3; ++k)
          shrunk[(size_t)y * hw + x][k] = (ushort)((a[k] + b[k] + c[k] + d[k] + 2) >> 2);
      }
    }
    replace_image(proc, shrunk, hw, hh);
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
                 const HalfSizeAnalysis& analyse, const ChannelBalance& balance,
                 const RawRegionFinder& find_region,
                 const negicc::FlatField* flat,
                 LibRaw* unpacked_proc) {
  const bool white_light = debayer && decode == RawDecode::kWhiteLight;
  int ret;
  LibRaw* proc = unpacked_proc;
  if (!proc) {
    proc = new LibRawRgb();
    log_info("Loading RAW file %s\n", fn.c_str());
    if ((ret = open_and_unpack_raw_file(proc, fn)) != LIBRAW_SUCCESS) {
      log_error("Cannot open/unpack %s: %s\n", fn.c_str(), libraw_strerror(ret));
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
    log_info("Linear DNG with 4 samples a pixel: the first three taken as RGB, the fourth (e.g. infrared) left out\n");
    proc->imgdata.idata.colors = 3;
    proc->imgdata.rawdata.iparams.colors = 3;
  }
  if (!(proc->imgdata.idata.filters || proc->imgdata.idata.colors == 1 || linear_dng)) {
    log_info("Only Bayer-pattern RAW files supported, sorry....\n");
    delete proc;
    return NULL;
  }
  int found_roi[4] = {0, 0, 0, 0};
  if (find_region) {
    if (!find_region(*proc, found_roi)) {
      delete proc;
      return NULL;
    }
    // A finder that only looked at the mosaic (the film base's frame detection with a given --roi) leaves the region
    // empty: the given roi, or the whole image, stays.
    if (found_roi[2] > 0 && found_roi[3] > 0) roi = found_roi;
  }
  // The ROI is checked and clamped to the sensor before the analysis read, which runs the whole bracket search, so
  // that an ROI outside the sensor fails before the DINOv3 inferences.
  int roi_box[4];
  if (roi) {
    const int w = proc->imgdata.sizes.width, h = proc->imgdata.sizes.height;
    if (roi[0] < 0 || roi[1] < 0 || roi[0] >= w || roi[1] >= h || roi[2] <= 0 || roi[3] <= 0) {
      log_error("ROI [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n",
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
    log_info("White-light decode (linear Rec.2020): %.4f%% of photosites clipped\n", clip_fraction * 100.0);
    // Frees a half-size decode and puts back what unpack() read, with the black level the mosaic no longer has.
    auto free_half = [proc]() {
      proc->free_image();
      proc->raw2image_start();
      proc->imgdata.color.black = 0;
      std::memset(proc->imgdata.color.cblack, 0, sizeof(proc->imgdata.color.cblack));
    };
    if (analyse) {
      if (!decode_white_light(proc, true, 0)) {
        delete proc;
        return NULL;
      }
      // Balance the camera channels on the film base, then decode the half size again from the balanced mosaic, which
      // the analysis and the output both see.
      float k[3] = {1, 1, 1};
      if (balance) balance(*proc, k);
      if (k[0] != 1 || k[1] != 1 || k[2] != 1) {
        free_half();
        negicc::scale_white_light_mosaic(proc, k);
        if (!decode_white_light(proc, true, 0)) {
          delete proc;
          return NULL;
        }
      }
      if (!analyse(*proc)) {
        proc->free_image();
        delete proc;
        return NULL;
      }
      free_half();
    }
  } else {
    if (flat) {
      float dummy_white = 0.0f;
      if (!proc->imgdata.idata.filters || proc->imgdata.idata.colors != 3 ||
          !subtract_cfa_black_in_place(proc, dummy_white)) {
        log_error("--flat requires a 3-colour CFA RAW file with valid sensor margins\n");
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
        log_error("%s\n", e.what());
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
    log_info("Debayer quality: %d\n", effective_qual);
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
  log_info("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}

// --------------------------------------------------------------------------------------------------------------------
// TIFF and JPEG files
// --------------------------------------------------------------------------------------------------------------------

namespace negicc {

bool is_jpeg_file(const std::string& path) {
  unsigned char sig[3] = {0, 0, 0};
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  const size_t n = fread(sig, 1, 3, f);
  fclose(f);
  return n == 3 && sig[0] == 0xFF && sig[1] == 0xD8 && sig[2] == 0xFF;
}

namespace {

// libtiff's messages go to stderr by default; they are collected into the error instead.
thread_local std::string tiff_message;

void tiff_handler(const char* module, const char* fmt, va_list ap) {
  char buf[512];
  vsnprintf(buf, sizeof(buf), fmt, ap);
  tiff_message = std::string(module ? module : "libtiff") + ": " + buf;
}

void tiff_warning(const char*, const char*, va_list) {}

// Copies sample `src_sample` of `n` pixels (stride `src_stride` samples) to channel `dst_ch` of out, scaling 8-bit
// samples to 16 bits.
void put_samples(const uint8_t* src, int bits, int src_stride, int src_sample, int n, uint16_t* dst, int dst_stride,
                 int dst_ch) {
  if (bits == 8) {
    for (int i = 0; i < n; ++i) dst[(size_t)i * dst_stride + dst_ch] = (uint16_t)(src[(size_t)i * src_stride + src_sample] * 257);
  } else {
    const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
    for (int i = 0; i < n; ++i) dst[(size_t)i * dst_stride + dst_ch] = s[(size_t)i * src_stride + src_sample];
  }
}

}  // namespace

bool read_tiff(const std::string& path, RgbImage& out, std::string& err) {
  TIFFSetErrorHandler(tiff_handler);
  TIFFSetWarningHandler(tiff_warning);
  tiff_message.clear();
  std::unique_ptr<TIFF, void (*)(TIFF*)> tif(TIFFOpen(path.c_str(), "r"), TIFFClose);
  if (!tif) {
    err = "cannot open TIFF " + path + (tiff_message.empty() ? "" : " (" + tiff_message + ")");
    return false;
  }
  uint32_t w = 0, h = 0;
  uint16_t spp = 1, bps = 8, photo = PHOTOMETRIC_MINISBLACK, planar = PLANARCONFIG_CONTIG, fmt = SAMPLEFORMAT_UINT;
  uint16_t compression = COMPRESSION_NONE;
  TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h);
  TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
  TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
  TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
  TIFFGetFieldDefaulted(tif.get(), TIFFTAG_COMPRESSION, &compression);
  if (!TIFFGetField(tif.get(), TIFFTAG_PHOTOMETRIC, &photo)) photo = spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
  if (photo == PHOTOMETRIC_YCBCR && compression == COMPRESSION_JPEG) {
    TIFFSetField(tif.get(), TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);  // libjpeg converts to RGB
    photo = PHOTOMETRIC_RGB;
  }
  if (w == 0 || h == 0) {
    err = path + ": TIFF without an image size";
    return false;
  }
  if ((bps != 8 && bps != 16) || fmt != SAMPLEFORMAT_UINT) {
    err = path + ": only 8- or 16-bit unsigned-integer TIFF samples are supported (" + std::to_string(bps) +
          "-bit, sample format " + std::to_string(fmt) + ")";
    return false;
  }
  int channels = 0;
  if (photo == PHOTOMETRIC_RGB && spp >= 3) channels = 3;
  else if (photo == PHOTOMETRIC_MINISBLACK && spp >= 1) channels = 1;
  if (!channels) {
    err = path + ": only RGB or min-is-black grey TIFFs are supported (photometric " + std::to_string(photo) + ")";
    return false;
  }
  out = RgbImage();
  out.width = (int)w;
  out.height = (int)h;
  out.channels = channels;
  out.bits = bps;
  out.pixels.assign((size_t)w * h * channels, 0);
  uint32_t icc_len = 0;
  void* icc = nullptr;
  if (TIFFGetField(tif.get(), TIFFTAG_ICCPROFILE, &icc_len, &icc) && icc && icc_len) {
    out.icc.assign((const uint8_t*)icc, (const uint8_t*)icc + icc_len);
  }

  const bool contig = planar == PLANARCONFIG_CONTIG;
  const int planes = contig ? 1 : channels;
  const int stride = contig ? spp : 1;       // samples per pixel in one decoded plane
  if (TIFFIsTiled(tif.get())) {
    uint32_t tw = 0, th = 0;
    TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tw);
    TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &th);
    std::vector<uint8_t> tile(TIFFTileSize(tif.get()));
    for (int p = 0; p < planes; ++p) {
      for (uint32_t ty = 0; ty < h; ty += th) {
        for (uint32_t tx = 0; tx < w; tx += tw) {
          if (TIFFReadTile(tif.get(), tile.data(), tx, ty, 0, (uint16_t)p) < 0) {
            err = path + ": " + (tiff_message.empty() ? "cannot read a tile" : tiff_message);
            return false;
          }
          const uint32_t nx = std::min(tw, w - tx), ny = std::min(th, h - ty);
          for (uint32_t y = 0; y < ny; ++y) {
            const uint8_t* row = tile.data() + (size_t)y * tw * stride * (bps / 8);
            uint16_t* dst = out.pixels.data() + ((size_t)(ty + y) * w + tx) * channels;
            for (int c = 0; c < (contig ? channels : 1); ++c) {
              put_samples(row, bps, stride, contig ? c : 0, (int)nx, dst, channels, contig ? c : p);
            }
          }
        }
      }
    }
  } else {
    std::vector<uint8_t> line(TIFFScanlineSize(tif.get()));
    for (int p = 0; p < planes; ++p) {
      for (uint32_t y = 0; y < h; ++y) {
        if (TIFFReadScanline(tif.get(), line.data(), y, (uint16_t)p) < 0) {
          err = path + ": " + (tiff_message.empty() ? "cannot read a scanline" : tiff_message);
          return false;
        }
        uint16_t* dst = out.pixels.data() + (size_t)y * w * channels;
        for (int c = 0; c < (contig ? channels : 1); ++c) {
          put_samples(line.data(), bps, stride, contig ? c : 0, (int)w, dst, channels, contig ? c : p);
        }
      }
    }
  }
  return true;
}

namespace {

struct JpegError {
  jpeg_error_mgr mgr;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};

void jpeg_error_exit(j_common_ptr cinfo) {
  JpegError* e = reinterpret_cast<JpegError*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, e->message);
  longjmp(e->jump, 1);
}

}  // namespace

bool read_jpeg(const std::string& path, RgbImage& out, std::string& err) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    err = "cannot open JPEG " + path;
    return false;
  }
  jpeg_decompress_struct cinfo;
  JpegError jerr;
  cinfo.err = jpeg_std_error(&jerr.mgr);
  jerr.mgr.error_exit = jpeg_error_exit;
  if (setjmp(jerr.jump)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    err = path + ": " + jerr.message;
    return false;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, f);
  jpeg_save_markers(&cinfo, JPEG_APP0 + 2, 0xFFFF);  // the ICC profile's chunks
  jpeg_read_header(&cinfo, TRUE);
  out = RgbImage();
  JOCTET* icc = nullptr;
  unsigned int icc_len = 0;
  if (jpeg_read_icc_profile(&cinfo, &icc, &icc_len) && icc) {
    out.icc.assign(icc, icc + icc_len);
    free(icc);
  }
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);
  out.width = (int)cinfo.output_width;
  out.height = (int)cinfo.output_height;
  out.channels = 3;
  out.bits = 8;
  out.pixels.assign((size_t)out.width * out.height * 3, 0);
  std::vector<JSAMPLE> row((size_t)out.width * 3);
  JSAMPROW rows[1] = {row.data()};
  while (cinfo.output_scanline < cinfo.output_height) {
    const JDIMENSION y = cinfo.output_scanline;
    jpeg_read_scanlines(&cinfo, rows, 1);
    uint16_t* dst = out.pixels.data() + (size_t)y * out.width * 3;
    for (size_t i = 0; i < row.size(); ++i) dst[i] = (uint16_t)(row[i] * 257);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  return true;
}

bool to_linear_rec2020(RgbImage& img, std::string& source, std::string& err) {
  if (img.channels != 3) {
    err = "a grey image has no colour to convert to Rec.2020";
    return false;
  }
  cmsHPROFILE in = nullptr;
  if (!img.icc.empty()) {
    in = cmsOpenProfileFromMem(img.icc.data(), (cmsUInt32Number)img.icc.size());
    if (!in) {
      err = "the embedded ICC profile cannot be read";
      return false;
    }
    if (cmsGetColorSpace(in) != cmsSigRgbData) {
      cmsCloseProfile(in);
      err = "the embedded ICC profile is not an RGB profile";
      return false;
    }
    source = "icc";
  } else if (img.bits == 8) {
    in = cmsCreate_sRGBProfile();
    source = "srgb";
  } else {
    source = "linear";
    return true;
  }
  // Linear Rec.2020: ITU-R BT.2020 primaries, D65 white, a gamma of 1.
  const cmsCIExyY d65 = {0.3127, 0.3290, 1.0};
  const cmsCIExyYTRIPLE primaries = {{0.708, 0.292, 1.0}, {0.170, 0.797, 1.0}, {0.131, 0.046, 1.0}};
  cmsToneCurve* linear = cmsBuildGamma(nullptr, 1.0);
  cmsToneCurve* curves[3] = {linear, linear, linear};
  cmsHPROFILE outp = cmsCreateRGBProfile(&d65, &primaries, curves);
  cmsFreeToneCurve(linear);
  cmsHTRANSFORM xf = outp ? cmsCreateTransform(in, TYPE_RGB_16, outp, TYPE_RGB_16, INTENT_RELATIVE_COLORIMETRIC, 0)
                          : nullptr;
  cmsCloseProfile(in);
  if (outp) cmsCloseProfile(outp);
  if (!xf) {
    err = "cannot build the transform to linear Rec.2020";
    return false;
  }
  const size_t row = (size_t)img.width * 3;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < img.height; ++y) {
    uint16_t* p = img.pixels.data() + (size_t)y * row;
    cmsDoTransform(xf, p, p, (cmsUInt32Number)img.width);
  }
  cmsDeleteTransform(xf);
  return true;
}

}  // namespace negicc

// --------------------------------------------------------------------------------------------------------------------
// Inputs besides one RAW
// --------------------------------------------------------------------------------------------------------------------

bool libraw_opens(const std::string& path) {
  LibRaw probe;
  return open_raw_file(&probe, path) == LIBRAW_SUCCESS;
}

bool is_linear_dng(const std::string& path) {
  LibRaw probe;
  return open_raw_file(&probe, path) == LIBRAW_SUCCESS && probe.imgdata.idata.dng_version &&
         !probe.imgdata.idata.filters;
}

// A LibRaw holding a w x h image the caller fills (imgdata.image, 4 samples a pixel), set up as load_raw leaves a
// decoded RAW: 16-bit linear output, no gamma, no flip.
static LibRaw* new_image_proc(int w, int h) {
  LibRaw* proc = new LibRaw();
  proc->imgdata.image = (ushort(*)[4])calloc((size_t)w * h, sizeof(*proc->imgdata.image));
  if (!proc->imgdata.image) {
    delete proc;
    return nullptr;
  }
  auto& S = proc->imgdata.sizes;
  S.raw_width = S.width = S.iwidth = (ushort)w;
  S.raw_height = S.height = S.iheight = (ushort)h;
  proc->imgdata.rawdata.sizes.width = (ushort)w;
  proc->imgdata.rawdata.sizes.height = (ushort)h;
  if (S.width != w || S.height != h) {  // LibRaw's sizes are 16-bit
    delete proc;
    return nullptr;
  }
  proc->imgdata.idata.colors = 3;
  proc->imgdata.idata.filters = 0;
  proc->imgdata.params.output_bps = 16;
  proc->imgdata.params.user_flip = 0;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.gamm[1] = 1;
  proc->imgdata.params.output_tiff = 1;
  return proc;
}

LibRaw* load_rgb_input(const std::vector<std::string>& files, InputMode mode, bool camera_rgb, int qual,
                              bool crop, const int* roi,
                              const HalfSizeAnalysis& analyse, FrameDetection* frame_det,
                              const FrameDetectionParams& det_params, std::string& source) {
  int w = 0, h = 0;
  float white = 65535.0f;     // the image's clipping level, for the frame detector
  std::vector<uint16_t> half_planes[3];  // three captures: each channel's half-size image, for the analysis
  int half_w[3] = {0, 0, 0}, half_h[3] = {0, 0, 0};
  LibRaw* proc = nullptr;
  std::string err;
  if (mode == InputMode::kRgbFile) {
    negicc::RgbImage img;
    const bool jpeg = negicc::is_jpeg_file(files[0]);
    log_info("Loading %s file %s\n", jpeg ? "JPEG" : "TIFF", files[0].c_str());
    if (!(jpeg ? negicc::read_jpeg(files[0], img, err) : negicc::read_tiff(files[0], img, err))) {
      log_error("%s\n", err.c_str());
      return nullptr;
    }
    if (img.channels != 3) {
      log_error("%s is a grey image; one file must be RGB (three files give one channel each)\n",
              files[0].c_str());
      return nullptr;
    }
    // Camera RGB (a tri bundle) is linear and unmanaged: a TIFF with an ICC profile or 8-bit samples is
    // colour-managed RGB, converted to linear Rec.2020 (a d65 bundle).
    if (camera_rgb && (!img.icc.empty() || img.bits != 16)) {
      log_error("%s is colour-managed RGB (%s), not the linear camera RGB a tri bundle converts: give "
                "--light_source d65\n", files[0].c_str(), !img.icc.empty() ? "it has an ICC profile" : "8 bits");
      return nullptr;
    }
    std::string how = "camera";
    if (!camera_rgb && !negicc::to_linear_rec2020(img, how, err)) {
      log_error("%s: %s\n", files[0].c_str(), err.c_str());
      return nullptr;
    }
    source = std::string(jpeg ? "jpeg" : "tiff") + ":" + how;
    log_info("Image size: %dx%d, %d-bit, taken as %s\n", img.width, img.height, img.bits,
           how == "icc" ? "its ICC profile" : how == "srgb" ? "sRGB" : how == "camera" ? "linear camera RGB"
                                                                                       : "linear Rec.2020");
    w = img.width;
    h = img.height;
    if (!(proc = new_image_proc(w, h))) {
      log_error("%s: cannot hold a %dx%d image\n", files[0].c_str(), w, h);
      return nullptr;
    }
    for (size_t i = 0; i < (size_t)w * h; ++i)
      for (int c = 0; c < 3; ++c) proc->imgdata.image[i][c] = img.pixels[i * 3 + c];
  } else {
    double exposure[3] = {0.0, 0.0, 0.0};  // shutter x ISO of each capture, 0 when it does not record them
    std::string kinds;  // "raw" or "tiff" per capture, comma-separated
    for (int k = 0; k < 3; ++k) {
      std::vector<uint16_t> channel;
      int cw = 0, ch = 0;
      if (libraw_opens(files[k])) {
        // The analysis reads a RAW's own half-size readout, as it does for one RAW (load_raw), so the RAW given three
        // times is analysed (the search, the film base) as the RAW given once.
        HalfSizeAnalysis take_half;
        if (analyse) {
          take_half = [&half_planes, &half_w, &half_h, k](const LibRaw& r) {
            half_w[k] = r.imgdata.sizes.iwidth;
            half_h[k] = r.imgdata.sizes.iheight;
            half_planes[k].resize((size_t)half_w[k] * half_h[k]);
            for (size_t i = 0; i < half_planes[k].size(); ++i) half_planes[k][i] = r.imgdata.image[i][k];
            return true;
          };
        }
        LibRaw* raw = load_raw(files[k], true, qual, crop, RawDecode::kCameraRgb, nullptr, take_half);
        if (!raw) return delete proc, nullptr;
        cw = raw->imgdata.sizes.iwidth;
        ch = raw->imgdata.sizes.iheight;
        channel.resize((size_t)cw * ch);
        for (size_t i = 0; i < channel.size(); ++i) channel[i] = raw->imgdata.image[i][k];
        if (raw->imgdata.other.shutter > 0 && raw->imgdata.other.iso_speed > 0)
          exposure[k] = (double)raw->imgdata.other.shutter * raw->imgdata.other.iso_speed;
        white = std::min(white, std::max(1.0f, (float)raw->imgdata.color.maximum - (float)raw->imgdata.color.black));
        if (k == 0) {  // the first capture's metadata (camera, exposure) describes the result
          if (!(proc = new_image_proc(cw, ch))) {
            log_error("%s: cannot hold a %dx%d image\n", files[k].c_str(), cw, ch);
            delete raw;
            return nullptr;
          }
          // Plain fields only: idata also owns the RAW's XMP buffer.
          std::memcpy(proc->imgdata.idata.make, raw->imgdata.idata.make, sizeof(proc->imgdata.idata.make));
          std::memcpy(proc->imgdata.idata.model, raw->imgdata.idata.model, sizeof(proc->imgdata.idata.model));
          proc->imgdata.other.shutter = raw->imgdata.other.shutter;
          proc->imgdata.other.iso_speed = raw->imgdata.other.iso_speed;
          proc->imgdata.other.aperture = raw->imgdata.other.aperture;
          proc->imgdata.other.timestamp = raw->imgdata.other.timestamp;
        }
        raw->free_image();
        delete raw;
        kinds += k ? ",raw" : "raw";
      } else {
        negicc::RgbImage img;
        log_info("Loading TIFF file %s\n", files[k].c_str());
        if (negicc::is_jpeg_file(files[k]) || !negicc::read_tiff(files[k], img, err)) {
          log_error("%s: %s\n", files[k].c_str(),
                  err.empty() ? "three captures must be RAWs, DNGs or TIFFs" : err.c_str());
          return delete proc, nullptr;
        }
        cw = img.width;
        ch = img.height;
        const int src = img.channels == 3 ? k : 0;
        channel.resize((size_t)cw * ch);
        for (size_t i = 0; i < channel.size(); ++i) channel[i] = img.pixels[i * img.channels + src];
        if (analyse) {  // a TIFF's half-size image: the mean of each 2x2 block
          half_w[k] = cw / 2;
          half_h[k] = ch / 2;
          half_planes[k].resize((size_t)half_w[k] * half_h[k]);
          for (int y = 0; y < half_h[k]; ++y)
            for (int x = 0; x < half_w[k]; ++x) {
              const size_t i = (size_t)(2 * y) * cw + 2 * x;
              half_planes[k][(size_t)y * half_w[k] + x] =
                  (uint16_t)((channel[i] + channel[i + 1] + channel[i + cw] + channel[i + cw + 1] + 2) / 4);
            }
        }
        if (k == 0 && !(proc = new_image_proc(cw, ch))) {
          log_error("%s: cannot hold a %dx%d image\n", files[k].c_str(), cw, ch);
          return nullptr;
        }
        kinds += k ? ",tiff" : "tiff";
      }
      if (k == 0) {
        w = cw;
        h = ch;
      } else if (cw != w || ch != h) {
        log_error("The three captures differ in size: %s is %dx%d, %s is %dx%d\n", files[0].c_str(), w,
                h, files[k].c_str(), cw, ch);
        return delete proc, nullptr;
      }
      for (size_t i = 0; i < channel.size(); ++i) proc->imgdata.image[i][k] = channel[i];
    }
    if (exposure[0] > 0 && exposure[1] > 0 && exposure[2] > 0) {
      for (int k = 1; k < 3; ++k) {
        const double scale = exposure[0] / exposure[k];
        if (scale == 1.0) continue;
        log_info("Channel %c scaled by %.4g to the first capture's exposure\n", "RGB"[k], scale);
        for (size_t i = 0; i < (size_t)w * h; ++i)
          proc->imgdata.image[i][k] = (ushort)std::min(65535.0, std::round(proc->imgdata.image[i][k] * scale));
        for (uint16_t& v : half_planes[k]) v = (uint16_t)std::min(65535.0, std::round(v * scale));
      }
    }
    source = "three:" + kinds;
  }

  int region[4] = {0, 0, w, h};
  if (frame_det) {
    FilmImage fi = film_image_from_interleaved(&proc->imgdata.image[0][0], w, h, 4, 0, white);
    FrameDetectionParams params = det_params;
    detect_frame_or_fallback(fi, params, *frame_det);
    if (frame_det->conf >= kLowFrameConfidence) std::copy(frame_det->crop, frame_det->crop + 4, region);
  } else if (roi) {
    std::copy(roi, roi + 4, region);
  }
  // As load_raw: the analysis sees the whole image at half size (the film base's statistics read the rebate and the
  // gaps around the frame, not only the picture), and places the frame in it (--roi halved, or --roi auto's detected
  // crop); the frame is cut from the full image after it.
  if (analyse) {
    const int hw = w / 2, hh = h / 2;
    std::unique_ptr<LibRaw> half(new_image_proc(hw, hh));
    if (!half) return delete proc, nullptr;
    half->imgdata.other.shutter = proc->imgdata.other.shutter;
    half->imgdata.other.iso_speed = proc->imgdata.other.iso_speed;
    // The full image's size, for the analysis to place --roi auto's detected frame (its crop is in those pixels).
    half->imgdata.rawdata.sizes.width = (ushort)w;
    half->imgdata.rawdata.sizes.height = (ushort)h;
    // Three captures of the whole image: each channel's own half-size image (above), when all three have the size.
    bool planes = mode == InputMode::kThreeShot && region[0] == 0 && region[1] == 0 && region[2] == w && region[3] == h;
    for (int k = 0; k < 3 && planes; ++k) planes = half_w[k] == hw && half_h[k] == hh && !half_planes[k].empty();
    if (planes) {
      for (size_t i = 0; i < (size_t)hw * hh; ++i)
        for (int k = 0; k < 3; ++k) half->imgdata.image[i][k] = half_planes[k][i];
    }
#pragma omp parallel for schedule(static)
    for (int y = 0; y < (planes ? 0 : hh); ++y) {
      for (int x = 0; x < hw; ++x) {
        const ushort* a = proc->imgdata.image[(size_t)(2 * y) * w + 2 * x];
        const ushort* b = proc->imgdata.image[(size_t)(2 * y) * w + 2 * x + 1];
        const ushort* c = proc->imgdata.image[(size_t)(2 * y + 1) * w + 2 * x];
        const ushort* d = proc->imgdata.image[(size_t)(2 * y + 1) * w + 2 * x + 1];
        for (int k = 0; k < 3; ++k)
          half->imgdata.image[(size_t)y * hw + x][k] = (ushort)((a[k] + b[k] + c[k] + d[k] + 2) / 4);
      }
    }
    if (!analyse(*half)) return delete proc, nullptr;
  }
  if ((frame_det || roi) && (region[0] || region[1] || region[2] != w || region[3] != h)) {
    if (!extract_roi_and_transform(proc, region[0], region[1], region[2], region[3], 0, false, false)) {
      proc->free_image();
      delete proc;
      return nullptr;
    }
  }
  log_info("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}
