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
// RAW (the inputs part of negicc.h); then the film base read from a capture of unexposed film (the film base part of
// negicc.h).

#include "negicc.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <system_error>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <lcms2.h>
#include <tiffio.h>
#include <jpeglib.h>  // after <cstdio>: it needs FILE and size_t
#include <jerror.h>   // the message codes (JWRN_JPEG_EOF, JWRN_HIT_MARKER)

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

  // Done in the image's own buffer, so the decode never holds a second full-size image. Each step moves pixels
  // within the buffer; the pages past the region are given back to the system (the buffer itself stays where it is:
  // LibRaw keeps track of the image's pointer and frees it, so it is not reallocated; nothing reads past the region
  // again).
  ushort (*img)[4] = proc->imgdata.image;
  const size_t n = (size_t)rw * rh;
  auto give_back_tail = [&]() {
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    const uintptr_t tail = ((uintptr_t)(img + n) + page - 1) / page * page;
    const uintptr_t end = (uintptr_t)(img + (size_t)cur_w * cur_h) / page * page;
    if (end > tail) madvise((void*)tail, end - tail, MADV_DONTNEED);
  };

  // 1. The region to the start of the buffer, row by row in order: a row moves toward lower addresses and lands
  //    below the next rows' pixels (row y ends at y * rw + rw <= (ry + y + 1) * cur_w + rx), so no row overwrites
  //    one not yet moved.
  if (rx != 0 || ry != 0 || rw != cur_w) {
    if (rw == cur_w) {
      memmove(img, img + (size_t)ry * cur_w, n * sizeof(*img));
    } else {
      for (int y = 0; y < rh; ++y) memmove(img + (size_t)y * rw, img + (size_t)(ry + y) * cur_w + rx, rw * sizeof(*img));
    }
  }

  // 2. A quarter turn of a region that is not square: an in-place transposition (rh x rw to rw x rh) by row and column
  //    permutations (Catanzaro, Keller and Garland, "A decomposition for in-place matrix transposition", PPoPP 2014),
  //    each pass parallel, with a buffer per thread of one row, or of a column block one cache line wide; then the
  //    flip below. For a region of m rows of n pixels, c = gcd(m, n), a = m / c, b = n / c:
  //      a. if c > 1, column j rotates up by j / b: new[i][j] = old[(i + j / b) mod m][j];
  //      b. row i is scattered: new[i][((i + j / b) mod m + j m) mod n] = old[i][j];
  //      c. column j is gathered: new[i][j] = old[(j + i n - i / a) mod m][j];
  //    and the buffer then holds the n x m transposed image, row after row.
  //    The passes take the program's thread count. Measured on a 192-thread x86 machine, a 9000 x 6000 region of a
  //    60 MP RAW, one conversion at a time: 1.27 s on 1 thread, 0.41 s on 4, 0.22 s on 8, 0.13 s on 16 (the default)
  //    and 0.11 s on 24, so they gain up to the default and a little past it. The result is the same at 1, 4 and 16
  //    threads (each thread writes only its own rows, or its own block of columns, in each pass), and a
  //    ThreadSanitizer build found no race.
  if (swap_dims && rw != rh) {
    const int64_t m = rh, nn = rw;
    const int64_t c = std::gcd(m, nn), a = m / c, b = nn / c;
    long line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    const int64_t block = std::max<int64_t>(1, (line > 0 ? line : (long)sizeof(*img)) / (long)sizeof(*img));
    const int threads = omp_get_max_threads();
    const int64_t per = std::max(m * block + block, nn);   // a column block and its j mod m, or a row
    std::vector<uint64_t> scratch;
    try {
      scratch.resize((size_t)threads * per);
    } catch (const std::bad_alloc&) {
      log_error("Failed to allocate memory for transformed ROI (%dx%d)\n", dst_w, dst_h);
      return false;
    }
    static_assert(sizeof(*img) == sizeof(uint64_t), "a pixel is four 16-bit samples");
    auto at = [img, nn](int64_t i, int64_t j) { return img[i * nn + j]; };
    // The indices are kept by additions, not divisions per pixel. Column passes go a block of columns at a time
    // through the thread's buffer: the block's rows copied in, then each row written back from the rows it gathers.
    // a. Column j rotates up by r = j / b (< c <= m): new[i][j] = old[i + r (mod m)][j].
    if (c > 1) {
      #pragma omp parallel for schedule(dynamic)
      for (int64_t j0 = 0; j0 < nn; j0 += block) {
        uint64_t* t = scratch.data() + (size_t)omp_get_thread_num() * per;
        const int64_t w = std::min(block, nn - j0);
        for (int64_t i = 0; i < m; ++i) memcpy(t + i * w, at(i, j0), w * sizeof(uint64_t));
        for (int64_t k = 0; k < w; ++k) {
          const int64_t r = (j0 + k) / b;
          for (int64_t i = 0, src = r; i < m; ++i, src = src + 1 == m ? 0 : src + 1)
            memcpy(at(i, j0 + k), t + src * w + k, sizeof(uint64_t));
        }
      }
    }
    // b. Row i is scattered: new[i][d] = old[i][j], d = ((i + j / b) mod m + j m) mod n, as (i + q) mod m mod n
    //    (q = j / b, changing every b pixels) plus j m mod n (growing by m mod n).
    const int64_t m_mod_n = m % nn;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
      uint64_t* t = scratch.data() + (size_t)omp_get_thread_num() * per;
      memcpy(t, at(i, 0), nn * sizeof(uint64_t));
      int64_t jm = 0;
      for (int64_t q = 0, j = 0; j < nn; ++q) {
        const int64_t head = ((i + q) % m) % nn;
        for (const int64_t stop = std::min(nn, j + b); j < stop; ++j) {
          int64_t d = head + jm;
          if (d >= nn) d -= nn;
          memcpy(at(i, d), t + j, sizeof(uint64_t));
          jm += m_mod_n;
          if (jm >= nn) jm -= nn;
        }
      }
    }
    // c. Column j is gathered: new[i][j] = old[s][j], s = (j + i n - i / a) mod m = (j mod m + base(i)) mod m, where
    //    base(i) = (i n - i / a) mod m does not depend on the column (one table for the pass).
    std::vector<int64_t> base;
    try {
      base.resize((size_t)m);
    } catch (const std::bad_alloc&) {
      log_error("Failed to allocate memory for transformed ROI (%dx%d)\n", dst_w, dst_h);
      return false;
    }
    for (int64_t i = 0, v = 0, n_mod_m = nn % m; i < m; ++i) {
      base[(size_t)i] = v;
      v += n_mod_m;
      if ((i + 1) % a == 0) v -= 1;
      v = ((v % m) + m) % m;
    }
    #pragma omp parallel for schedule(dynamic)
    for (int64_t j0 = 0; j0 < nn; j0 += block) {
      uint64_t* t = scratch.data() + (size_t)omp_get_thread_num() * per;
      const int64_t w = std::min(block, nn - j0);
      int64_t* jmod = reinterpret_cast<int64_t*>(t + m * w);   // the block's j mod m, after its rows
      for (int64_t k = 0; k < w; ++k) jmod[k] = (j0 + k) % m;
      for (int64_t i = 0; i < m; ++i) memcpy(t + i * w, at(i, j0), w * sizeof(uint64_t));
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t k = 0; k < w; ++k) {
          int64_t src = jmod[k] + base[(size_t)i];
          if (src >= m) src -= m;
          memcpy(at(i, j0 + k), t + src * w + k, sizeof(uint64_t));
        }
      }
    }
    if (rot_cw == 90) hflip = !hflip;
    else vflip = !vflip;
  }

  //    A square region swaps pixels across its diagonal. Either way the transposition is followed by a flip: 90
  //    degrees clockwise flips the columns of the transposed image, 270 its rows.
  if (swap_dims && rw == rh) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int y = 0; y < rh; ++y) {
      for (int x = y + 1; x < rw; ++x) std::swap(img[(size_t)y * rw + x], img[(size_t)x * rw + y]);
    }
    if (rot_cw == 90) hflip = !hflip;
    else vflip = !vflip;
  }

  // 3. The flips, in place: rows swapped end for end, pixels within each row.
  if (vflip) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dst_h / 2; ++y) {
      std::swap_ranges(img + (size_t)y * dst_w, img + (size_t)y * dst_w + dst_w, img + (size_t)(dst_h - 1 - y) * dst_w);
    }
  }
  if (hflip) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < dst_h; ++y) std::reverse(img + (size_t)y * dst_w, img + (size_t)y * dst_w + dst_w);
  }

  give_back_tail();
  proc->imgdata.sizes.iwidth = dst_w;
  proc->imgdata.sizes.iheight = dst_h;
  return true;
}

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
    // LibRaw keeps reading the stream after it opens (unpack, unpack_thumb), so the stream lives on the heap and proc
    // owns it, as LibRaw::open_file owns its own: input_internal makes LibRaw delete it on recycle() and in its
    // destructor.
    PatchedHeaderStream* stream = new PatchedHeaderStream(fn.c_str());
    if (stream->valid() && stream->is_patched()) {
      internal_data_t& id = proc->get_internal_data_pointer()->internal_data;
      id.input_internal = 0;
      ret = proc->open_datastream(stream);
      if (ret == LIBRAW_SUCCESS) {
        id.input_internal = 1;
        return ret;
      }
    }
    delete stream;
  }
  return ret;
}

bool raw_levels_valid(const LibRaw& proc, const std::string& fn) {
  const auto& C = proc.imgdata.color;
  const unsigned sample_max = std::numeric_limits<uint16_t>::max();
  if (C.maximum == 0 || C.maximum > sample_max) {
    log_error("%s: white level %u is outside the 16-bit sample's range (1 to %u)\n", fn.c_str(), C.maximum,
              sample_max);
    return false;
  }
  // A DNG's black levels as the file gives them (BlackLevel, its deltas averaged in), each of which must round to a
  // sample's value (dng_fcblack[4] and [5] hold the pattern's size, not a level). They are tested as given, not
  // through LibRaw's integer black: LibRaw converts them to unsigned, which is undefined for a level outside that
  // type, and the result differs by machine (a negative level wraps on x86-64, where a sum of two can wrap back into
  // range, and becomes 0 on ARM64), so the integer does not show what the file held.
  const auto& D = C.dng_levels;
  const auto a_sample = [sample_max](float v) {
    const double r = std::floor((double)v + 0.5);
    return r >= 0.0 && r <= (double)sample_max;   // false for a NaN
  };
  float dng_black = D.dng_fblack;
  bool dng_ok = a_sample(dng_black);
  for (int i = 0; i < LIBRAW_CBLACK_SIZE && dng_ok; ++i) {
    if (i != 4 && i != 5 && !a_sample(D.dng_fcblack[i])) dng_ok = false, dng_black = D.dng_fcblack[i];
  }
  if (!dng_ok) {
    log_error("%s: DNG black level %.0f is outside the 16-bit sample's range (0 to %u)\n", fn.c_str(), dng_black,
              sample_max);
    return false;
  }
  // The black of a photosite is black + cblack[colour] + its cblack[6..] pattern entry (as subtract_cfa_black_in_place
  // and prepare_white_light_mosaic take it); a negative one wraps to a large unsigned value.
  unsigned cblack_max = 0, pattern_max = 0;
  for (int i = 0; i < 4; ++i) cblack_max = std::max(cblack_max, C.cblack[i]);
  const unsigned dr = C.cblack[4], dc = C.cblack[5];
  if (dr > 0 && dc > 0 && (size_t)dr * dc <= LIBRAW_CBLACK_SIZE - 6) {
    for (unsigned i = 0; i < dr * dc; ++i) pattern_max = std::max(pattern_max, C.cblack[6 + i]);
  }
  const struct { const char* name; unsigned value; } parts[3] = {
      {"black level", C.black}, {"per-channel black level", cblack_max}, {"black level pattern entry", pattern_max}};
  for (const auto& p : parts) {
    if (p.value > sample_max) {
      log_error("%s: %s %u is outside the 16-bit sample's range (0 to %u)\n", fn.c_str(), p.name, p.value,
                sample_max);
      return false;
    }
  }
  const uint64_t black_max = (uint64_t)C.black + cblack_max + pattern_max;
  if (black_max >= C.maximum) {
    log_error("%s: black level %llu is not below the white level %u\n", fn.c_str(), (unsigned long long)black_max,
              C.maximum);
    return false;
  }
  return true;
}

int open_and_unpack_raw_file(LibRaw* proc, const std::string& fn) {
  int ret = open_raw_file(proc, fn);
  if (ret != LIBRAW_SUCCESS) return ret;
  // LibRaw's dcraw_process replaces the image after demosaicing when the sensor's photosites are not a square grid:
  // turned 45 degrees (fuji_rotate: a Fuji Super CCD, or a DNG with CFALayout 2) or stretched to square pixels
  // (stretch: a pixel aspect other than 1). Either changes sizes.width and height but not iwidth and iheight, and puts
  // the picture off the sensor's pixel grid that crops, regions, the film base and flat fields are given in.
  if (proc->is_fuji_rotated()) {
    log_error("%s: the sensor's photosites are on a 45-degree grid (a Fuji Super CCD, or a DNG with CFALayout 2), "
              "which is not supported\n", fn.c_str());
    return LIBRAW_FILE_UNSUPPORTED;
  }
  if (proc->imgdata.sizes.pixel_aspect != 1) {
    log_error("%s: the sensor's pixels are not square (pixel aspect %g), which is not supported\n", fn.c_str(),
              proc->imgdata.sizes.pixel_aspect);
    return LIBRAW_FILE_UNSUPPORTED;
  }
  // An X-Trans pattern (filters 9) has three colours, 0 R, 1 G and 2 B, and COLOR() gives them as they are: LibRaw
  // clamps a RAF's to these but takes a DNG's CFAPattern as written, masked to 0..3. Every reader of the colours takes
  // them as 0..2 (a Bayer pattern's 3, its second green, comes from LibRaw's 2-bit filters and is read as such).
  if (proc->imgdata.idata.filters == 9) {
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        const int k = proc->imgdata.idata.xtrans[r][c];
        if (k < 0 || k > 2) {
          log_error("%s: the 6x6 colour filter pattern holds colour %d at row %d, column %d; its colours are 0 (red), "
                    "1 (green) and 2 (blue)\n", fn.c_str(), k, r, c);
          return LIBRAW_FILE_UNSUPPORTED;
        }
      }
    }
  }
  log_info("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  ret = proc->unpack();
  if (ret == LIBRAW_SUCCESS && !raw_levels_valid(*proc, fn)) ret = LIBRAW_DATA_ERROR;
  return ret;
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
bool flat_after_half_size(const LibRaw& r) {
  const auto& S = r.imgdata.rawdata.sizes;
  return r.imgdata.idata.filters && S.width > 0 && S.height > 0 && S.width % 2 == 0 && S.height % 2 == 0;
}

// The colour of each photosite of a 6x6 tile as the flat field takes it (0 R, 1 G, 2 B; the second green is green).
static std::array<uint8_t, 36> col3_of(LibRaw& proc) {
  std::array<uint8_t, 36> col3{};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      const int k = proc.COLOR(r, c);
      col3[r * 6 + c] = (uint8_t)(k == 3 ? 1 : k);
    }
  }
  return col3;
}

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
  if (!(pre[3] > 0) || !std::isfinite(pre[3])) pre[3] = pre[1];   // no second green of its own: the first's
  if (!(pre[0] > 0 && pre[1] > 0 && pre[2] > 0) || !std::isfinite(pre[0]) || !std::isfinite(pre[1]) ||
      !std::isfinite(pre[2]) || !std::isfinite(pre[0] / pre[1]) || !std::isfinite(pre[2] / pre[1])) {
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

// LibRaw gives the inset crop in the raw frame, margins included (its own adjust_to_raw_inset_crop makes cleft and ctop
// the margins), so the margins are taken off to place it in the visible image; a crop that starts in the margins keeps
// its part in the image. LibRaw reports an inset crop the RAW does not give (e.g. a DNG without DefaultCrop) as left
// and top 0xffff with a zero size; dcraw_process would refuse that as a bad crop box. A crop at the origin that is
// narrower or shorter than the image is a crop too.
bool inset_crop(const LibRaw& r, int box[4]) {
  const auto& S = r.imgdata.sizes;
  const auto& c = S.raw_inset_crops[0];
  if (c.cwidth == 0 || c.cheight == 0 || c.cleft == 0xffff || c.ctop == 0xffff) return false;
  int left = (int)c.cleft - S.left_margin, top = (int)c.ctop - S.top_margin, width = c.cwidth, height = c.cheight;
  if (left < 0) width += left, left = 0;
  if (top < 0) height += top, top = 0;
  if (!left && !top && width >= S.width && height >= S.height) return false;
  box[0] = left;
  box[1] = top;
  box[2] = width;
  box[3] = height;
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
  // A debayered decode's inset crop (with |crop| and no ROI) is decoded as an ROI is, so that its edge pixels are
  // interpolated from their neighbours outside it.
  int inset[4];
  const bool is_inset = !roi && debayer && crop && inset_crop(*proc, inset);
  if (is_inset) roi = inset;
  // The ROI is checked and clamped to the sensor before the analysis read, which runs the whole bracket search, so
  // that an ROI outside the sensor fails before the DINOv3 inferences.
  int roi_box[4];
  if (roi) {
    const int w = proc->imgdata.sizes.width, h = proc->imgdata.sizes.height;
    if (roi[0] < 0 || roi[1] < 0 || roi[0] >= w || roi[1] >= h || roi[2] <= 0 || roi[3] <= 0) {
      log_error("%s [%d, %d, %d, %d] is outside sensor boundaries (%dx%d)\n", is_inset ? "The inset crop" : "ROI",
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
    float flat_white = 0.0f;   // the white level less the black subtract_cfa_black_in_place took off
    if (flat) {
      if (!proc->imgdata.idata.filters || proc->imgdata.idata.colors != 3 ||
          !subtract_cfa_black_in_place(proc, flat_white)) {
        log_error("--flat requires a 3-colour CFA RAW file with valid sensor margins\n");
        delete proc;
        return NULL;
      }
      proc->imgdata.params.user_black = 0;
      for (int i = 0; i < 4; ++i) proc->imgdata.params.user_cblack[i] = 0;
    }
    // The flat divides the mosaic in place, for the output. The half-size image the analysis reads is decoded before
    // it, when its size is the flat's half size (an even mosaic), and the analysis then reads it through the flat
    // (flat_after_half_size): the film base's clip test sees the levels as recorded, as on the CFA path, not divided
    // by gains below 1 to above the clip.
    const auto& S = proc->imgdata.sizes;
    const bool half_before_flat = flat && analyse && flat_after_half_size(*proc);
    auto apply_flat = [&]() {
      const size_t pitch = S.raw_pitch / 2;
      try {
        flat->apply_cfa(proc->imgdata.rawdata.raw_image + (size_t)S.top_margin * pitch + S.left_margin,
                        pitch, S.width, S.height, col3_of(*proc).data());
      } catch (const std::exception& e) {
        log_error("%s\n", e.what());
        return false;
      }
      return true;
    };
    if (flat && !half_before_flat && !apply_flat()) {
      delete proc;
      return NULL;
    }
    // The black was subtracted above, before the flat: the half-size image's clip is the white level less it, which
    // LibRaw's maximum (restored as recorded, with no black left to take off) does not say. Given to the analysis
    // (the film base's clip test) for its duration only, so the decode is as before.
    const HalfSizeAnalysis analyse_flat = [&](const LibRaw& r) {   // r is *proc (read_half_size)
      const unsigned recorded = proc->imgdata.color.maximum;
      proc->imgdata.color.maximum = (unsigned)flat_white;
      const bool ok = analyse(r);
      proc->imgdata.color.maximum = recorded;
      return ok;
    };
    if (analyse && !read_half_size(proc, flat ? analyse_flat : analyse)) {
      delete proc;
      return NULL;
    }
    if (half_before_flat && !apply_flat()) {
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
    const int sensor_w =
        proc->imgdata.rawdata.sizes.width ? proc->imgdata.rawdata.sizes.width : proc->imgdata.sizes.width;
    const int sensor_h =
        proc->imgdata.rawdata.sizes.height ? proc->imgdata.rawdata.sizes.height : proc->imgdata.sizes.height;

    int sub_x = 0, sub_y = 0, sub_w = 0, sub_h = 0;
    if (roi) {
      // Align LibRaw's cropbox origin down to the CFA period (6x6 for X-Trans, 2x2 for Bayer) so LibRaw's
      // raw2image does not shift the mosaic phase, and include a 2-tile halo on every ROI (--roi auto, explicit --roi
      // and the inset crop alike) so LibRaw's demosaic stencil has identical boundary neighbors.
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
    }
    if (white_light) {
      if (!decode_white_light(proc, false, effective_qual)) {
        delete proc;
        return NULL;
      }
    } else if ((ret = proc->dcraw_process()) != LIBRAW_SUCCESS) {
      log_error("Cannot decode %s: %s\n", fn.c_str(), libraw_strerror(ret));
      delete proc;
      return NULL;
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
  // SOI (FFD8) and the next marker's FF (ITU-T T.81 Table B.1).
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
// samples to 16 bits: v * 257 = (v << 8) | v, 0 -> 0 and 255 -> 65535 (ICC.1:2022 A.4: (2^16 - 1) / (2^8 - 1)).
void put_samples(const uint8_t* src, int bits, int src_stride, int src_sample, int n, uint16_t* dst, int dst_stride,
                 int dst_ch) {
  if (bits == 8) {
    for (int i = 0; i < n; ++i)
      dst[(size_t)i * dst_stride + dst_ch] = (uint16_t)(src[(size_t)i * src_stride + src_sample] * 257);
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
  if (!TIFFGetField(tif.get(), TIFFTAG_PHOTOMETRIC, &photo))
    photo = spp >= 3 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK;
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
  // The image is held in a LibRaw image (new_image_proc), whose sizes are 16-bit: a larger size is refused before
  // the pixels are allocated.
  const uint32_t max_side = std::numeric_limits<decltype(libraw_image_sizes_t::width)>::max();
  if (w > max_side || h > max_side) {
    err = path + ": a " + std::to_string(w) + "x" + std::to_string(h) + " TIFF is larger than " +
          std::to_string(max_side) + " pixels on a side";
    return false;
  }
  // Nothing is sized from the header alone. Uncompressed samples are all in the file, so the bytes the header claims
  // must fit in it, and the image is then allocated at once. Compressed data can truly hold more than the file's
  // length, so the image's rows are allocated as libtiff decodes them.
  const bool stored = compression == COMPRESSION_NONE;
  std::error_code size_ec;
  const uint64_t file_bytes = std::filesystem::file_size(path, size_ec);
  if (size_ec) {
    err = "cannot read the size of " + path;
    return false;
  }
  const uint64_t image_bytes = (uint64_t)w * h * spp * (bps / 8);
  if (stored && image_bytes > file_bytes) {
    err = path + ": an uncompressed " + std::to_string(w) + "x" + std::to_string(h) + " TIFF holds " +
          std::to_string(image_bytes) + " bytes of samples, more than the file's " + std::to_string(file_bytes);
    return false;
  }
  out = RgbImage();
  out.width = (int)w;
  out.height = (int)h;
  out.channels = channels;
  out.bits = bps;
  const size_t row_samples = (size_t)w * channels;
  if (stored) out.pixels.assign(row_samples * h, 0);
  // The image holds rows 0 to rows - 1 at least (compressed data: called once those rows are known to decode).
  const auto have_rows = [&](uint32_t rows) {
    if (out.pixels.size() < row_samples * rows) out.pixels.resize(row_samples * rows, 0);
  };
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
    // A tile holds tw x th pixels of stride samples, which put_samples reads row by row. A tile may be larger than
    // the image (libtiff pads it), but not larger on a side than an image may be (max_side, above): the buffer is
    // sized from the tile, before libtiff reads a byte of it.
    if (tw > max_side || th > max_side) {
      err = path + ": a " + std::to_string(tw) + "x" + std::to_string(th) + " tile is larger than " +
            std::to_string(max_side) + " pixels on a side";
      return false;
    }
    const tmsize_t tile_size = TIFFTileSize(tif.get());
    if (tw == 0 || th == 0 || tile_size <= 0 || (uint64_t)tile_size < (uint64_t)tw * th * stride * (bps / 8)) {
      err = path + ": a " + std::to_string(tw) + "x" + std::to_string(th) + " tile of " + std::to_string(tile_size) +
            " bytes cannot be read";
      return false;
    }
    // A tile's buffer is sized before libtiff reads a byte of it. An uncompressed tile is in the file. A compressed
    // one may be no larger than the image rounded up to the tile unit, 16 pixels (TIFF 6.0, section 15: TileWidth
    // and TileLength are multiples of 16): a larger tile is padding only.
    if (stored && (uint64_t)tile_size > file_bytes) {
      err = path + ": an uncompressed " + std::to_string(tw) + "x" + std::to_string(th) + " tile holds " +
            std::to_string(tile_size) + " bytes, more than the file's " + std::to_string(file_bytes);
      return false;
    }
    if (!stored && (tw > (w + 15) / 16 * 16 || th > (h + 15) / 16 * 16)) {
      err = path + ": a " + std::to_string(tw) + "x" + std::to_string(th) + " tile is larger than the " +
            std::to_string(w) + "x" + std::to_string(h) + " image";
      return false;
    }
    // Not filled: its pages are written only as libtiff decodes into them.
    const std::unique_ptr<uint8_t[]> tile_buf(new (std::nothrow) uint8_t[(size_t)tile_size]);
    if (!tile_buf) {
      err = path + ": no memory for a " + std::to_string(tw) + "x" + std::to_string(th) + " tile of " +
            std::to_string(tile_size) + " bytes";
      return false;
    }
    uint8_t* const tile = tile_buf.get();
    for (int p = 0; p < planes; ++p) {
      for (uint32_t ty = 0; ty < h; ty += th) {
        for (uint32_t tx = 0; tx < w; tx += tw) {
          if (TIFFReadTile(tif.get(), tile, tx, ty, 0, (uint16_t)p) < 0) {
            err = path + ": " + (tiff_message.empty() ? "cannot read a tile" : tiff_message);
            return false;
          }
          have_rows(std::min(h, ty + th));
          const uint32_t nx = std::min(tw, w - tx), ny = std::min(th, h - ty);
          for (uint32_t y = 0; y < ny; ++y) {
            const uint8_t* row = tile + (size_t)y * tw * stride * (bps / 8);
            uint16_t* dst = out.pixels.data() + ((size_t)(ty + y) * w + tx) * channels;
            for (int c = 0; c < (contig ? channels : 1); ++c) {
              put_samples(row, bps, stride, contig ? c : 0, (int)nx, dst, channels, contig ? c : p);
            }
          }
        }
      }
    }
  } else {
    // A scanline holds w pixels of stride samples.
    const tmsize_t line_size = TIFFScanlineSize(tif.get());
    if (line_size <= 0 || (uint64_t)line_size < (uint64_t)w * stride * (bps / 8)) {
      err = path + ": a scanline of " + std::to_string(line_size) + " bytes cannot hold " + std::to_string(w) +
            " pixels";
      return false;
    }
    std::vector<uint8_t> line(line_size);
    for (int p = 0; p < planes; ++p) {
      for (uint32_t y = 0; y < h; ++y) {
        if (TIFFReadScanline(tif.get(), line.data(), y, (uint16_t)p) < 0) {
          err = path + ": " + (tiff_message.empty() ? "cannot read a scanline" : tiff_message);
          return false;
        }
        have_rows(y + 1);
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
  void (*emit_message)(j_common_ptr, int) = nullptr;   // libjpeg's own
};

void jpeg_error_exit(j_common_ptr cinfo) {
  JpegError* e = reinterpret_cast<JpegError*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, e->message);
  longjmp(e->jump, 1);
}

// libjpeg's warnings are damaged data it goes on from. Two of them mean the data ran out, and libjpeg then makes up
// the rest of the picture (grey): the file ended early (JWRN_JPEG_EOF) or a data segment did (JWRN_HIT_MARKER). Those
// are errors here: such a file is refused, not converted. The other warnings keep libjpeg's handling.
void jpeg_emit_message(j_common_ptr cinfo, int msg_level) {
  JpegError* e = reinterpret_cast<JpegError*>(cinfo->err);
  if (msg_level < 0 && (cinfo->err->msg_code == JWRN_JPEG_EOF || cinfo->err->msg_code == JWRN_HIT_MARKER))
    jpeg_error_exit(cinfo);
  (*e->emit_message)(cinfo, msg_level);
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
  jerr.emit_message = jerr.mgr.emit_message;
  jerr.mgr.emit_message = jpeg_emit_message;
  if (setjmp(jerr.jump)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    err = path + ": " + jerr.message;
    return false;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, f);
  // The ICC profile's chunks: APP2 segments beginning "ICC_PROFILE" (ICC.1:2001-04 Annex B.4)
  jpeg_save_markers(&cinfo, JPEG_APP0 + 2, 0xFFFF);
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
  // The picture's rows are allocated as libjpeg decodes them, not from the header's size: a file that claims more
  // than it holds ends in an error (above) before the rows it lacks are allocated. Memory that runs out ends the
  // read like such an error, with libjpeg's state and the file released.
  try {
    std::vector<JSAMPLE> row((size_t)out.width * 3);
    JSAMPROW rows[1] = {row.data()};
    while (cinfo.output_scanline < cinfo.output_height) {
      const JDIMENSION y = cinfo.output_scanline;
      jpeg_read_scanlines(&cinfo, rows, 1);
      out.pixels.resize((size_t)(y + 1) * row.size());
      uint16_t* dst = out.pixels.data() + (size_t)y * out.width * 3;
      for (size_t i = 0; i < row.size(); ++i) dst[i] = (uint16_t)(row[i] * 257);   // 8 to 16 bits, as put_samples
    }
  } catch (const std::bad_alloc&) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    err = path + ": not enough memory for its " + std::to_string(out.width) + "x" + std::to_string(out.height) +
          " pixels";
    out = RgbImage();
    return false;
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
  // Linear Rec.2020: ITU-R BT.2020 primaries, D65 white, a gamma of 1 (the chromaticities of ITU-R BT.2020-2 Table 3).
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
  int inset[4] = {0, 0, 0, 0};           // three RAW captures: the first one's inset crop, when cropped
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
        // times is analysed (the search, the film base) as the RAW given once. The RAW is read whole, as its
        // half-size readout is; its inset crop, like --roi, is cut after the analysis.
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
        LibRaw* raw = load_raw(files[k], true, qual, false, RawDecode::kCameraRgb, nullptr, take_half);
        if (!raw) return delete proc, nullptr;
        if (k == 0 && crop) inset_crop(*raw, inset);
        cw = raw->imgdata.sizes.iwidth;
        ch = raw->imgdata.sizes.iheight;
        channel.resize((size_t)cw * ch);
        for (size_t i = 0; i < channel.size(); ++i) channel[i] = raw->imgdata.image[i][k];
        // A shutter or ISO that is not a positive finite number records no exposure (an infinite one scaled the other
        // channels by infinity, to 65535).
        const double ek = (double)raw->imgdata.other.shutter * raw->imgdata.other.iso_speed;
        if (raw->imgdata.other.shutter > 0 && raw->imgdata.other.iso_speed > 0 && std::isfinite(ek)) exposure[k] = ek;
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
        // A capture is one channel of linear camera RGB, as a RAW capture is read: a TIFF with an ICC profile or
        // 8-bit samples is colour-managed, which the single TIFF converts (a d65 bundle) and three captures cannot.
        if (!img.icc.empty() || img.bits != 16) {
          log_error("%s is colour-managed (%s), not the linear camera RGB three captures are read as: give 16-bit "
                    "linear TIFFs without an ICC profile, or the RAWs\n", files[k].c_str(),
                    !img.icc.empty() ? "it has an ICC profile" : "8 bits");
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

  // The analysis reads the image at half size (below), which an image one pixel wide or high does not have.
  if (analyse && (w < 2 || h < 2)) {
    log_error("%s is %dx%d: the analysis needs an image at least 2 pixels wide and high\n", files[0].c_str(), w, h);
    return delete proc, nullptr;
  }
  int region[4] = {0, 0, w, h};
  if (frame_det) {
    FilmImage fi = film_image_from_interleaved(&proc->imgdata.image[0][0], w, h, 4, 0, white);
    FrameDetectionParams params = det_params;
    detect_frame_or_fallback(fi, params, *frame_det);
    if (frame_det->conf >= kLowFrameConfidence) std::copy(frame_det->crop, frame_det->crop + 4, region);
  } else if (roi || inset[2] > 0) {
    // As load_raw: an ROI (or inset crop) outside the image fails here, before the analysis runs the search, and one
    // past its right or bottom edge is clamped to it.
    const int* r = roi ? roi : inset;
    if (r[0] < 0 || r[1] < 0 || r[0] >= w || r[1] >= h || r[2] <= 0 || r[3] <= 0) {
      log_error("%s [%d, %d, %d, %d] is outside the image (%dx%d)\n", roi ? "ROI" : "The inset crop", r[0], r[1], r[2],
                r[3], w, h);
      return delete proc, nullptr;
    }
    region[0] = r[0];
    region[1] = r[1];
    region[2] = std::min(r[2], w - r[0]);
    region[3] = std::min(r[3], h - r[1]);
  }
  // As load_raw: the analysis sees the whole image at half size (the film base's statistics read the rebate and the
  // gaps around the frame, not only the picture), and places the frame in it (--roi halved, --roi auto's detected
  // crop, or the inset crop halved); the frame is cut from the full image after it.
  if (analyse) {
    const int hw = w / 2, hh = h / 2;
    std::unique_ptr<LibRaw> half(new_image_proc(hw, hh));
    if (!half) return delete proc, nullptr;
    half->imgdata.other.shutter = proc->imgdata.other.shutter;
    half->imgdata.other.iso_speed = proc->imgdata.other.iso_speed;
    half->imgdata.color.maximum = (unsigned)white;   // the image's clip, as a RAW's half-size image carries it
    // The full image's size, for the analysis to place --roi auto's detected frame (its crop is in those pixels), and
    // the first RAW capture's inset crop in those pixels when it is the frame, for the analysis to search it as it
    // does --roi (and as one RAW's CFA path does): a zero size when it is not.
    half->imgdata.rawdata.sizes.width = (ushort)w;
    half->imgdata.rawdata.sizes.height = (ushort)h;
    auto& frame_crop = half->imgdata.rawdata.sizes.raw_inset_crops[0];
    frame_crop = {};
    if (!roi && !frame_det && inset[2] > 0) {
      frame_crop.cleft = (ushort)region[0];
      frame_crop.ctop = (ushort)region[1];
      frame_crop.cwidth = (ushort)region[2];
      frame_crop.cheight = (ushort)region[3];
    }
    // Three captures: each channel's own half-size image (above), when all three have the image's half size. They are
    // the whole image whatever the frame (--roi, --roi auto's or the inset crop), which is cut after the analysis.
    bool planes = mode == InputMode::kThreeShot;
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
  if (region[0] || region[1] || region[2] != w || region[3] != h) {
    if (!extract_roi_and_transform(proc, region[0], region[1], region[2], region[3], 0, false, false)) {
      proc->free_image();
      delete proc;
      return nullptr;
    }
  }
  log_info("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}

// ====================================================================================================================
// Film base: code
// The film base read from a capture of unexposed film (read_film_base_capture).
// ====================================================================================================================

namespace {

double median_of(std::vector<float>& v) {
  if (v.empty()) return 0.0;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) {
    const float lo = *std::max_element(v.begin(), v.begin() + mid);
    m = 0.5 * (m + lo);
  }
  return m;
}

// A region of the half-size view: each channel's median, the spread of the green density and the clipped share.
struct RegionStats {
  double med[3] = {0.0, 0.0, 0.0};
  double spread = 0.0;
  double clipped = 0.0;
  size_t n = 0;
};

// view: the capture with the flat field, which the levels are read from. raw: the same mosaic without it, which the
// clipping is counted on (a photosite clips at the white level as recorded, whatever the flat field then makes of it).
RegionStats region_stats(const negicc::RgbView& view, const negicc::RgbView& raw, int x0, int y0, int w, int h,
                         float clip) {
  RegionStats st;
  // At most about 4096 samples a region, on a regular lattice.
  const int step = std::max(1, (int)std::ceil(std::sqrt((double)w * h / 4096.0)));
  std::vector<float> ch[3];
  size_t clipped = 0;
  for (int y = y0; y < y0 + h; y += step) {
    for (int x = x0; x < x0 + w; x += step) {
      uint16_t px[3], rpx[3];
      view.sample(x, y, px);
      raw.sample(x, y, rpx);
      if (rpx[0] >= clip || rpx[1] >= clip || rpx[2] >= clip) ++clipped;
      for (int c = 0; c < 3; ++c) ch[c].push_back((float)px[c]);
    }
  }
  st.n = ch[0].size();
  if (!st.n) return st;
  st.clipped = (double)clipped / (double)st.n;
  std::vector<float> green = ch[1];
  for (int c = 0; c < 3; ++c) st.med[c] = median_of(ch[c]);
  if (st.med[1] > 0.0) {
    std::vector<float> dev(green.size());
    for (size_t i = 0; i < green.size(); ++i) {
      const double g = std::max(1.0, (double)green[i]);
      dev[i] = (float)std::fabs(std::log10(g / st.med[1]));
    }
    st.spread = kMadToSigma * median_of(dev);   // the MAD as a normal standard deviation
  }
  return st;
}

// Whether a tile's colour is the film base's: its medians through the crosstalk matrix, G/R and B/R against the
// expected base's (step 3 in the film base part of negicc.h).
bool base_coloured(const double med[3], const double* crosstalk, const double* expected) {
  if (!expected || !(expected[0] > 0.0) || !(expected[1] > 0.0) || !(expected[2] > 0.0)) return true;
  double v[3];
  for (int r = 0; r < 3; ++r) {
    v[r] = 0.0;
    for (int c = 0; c < 3; ++c) v[r] += (crosstalk ? crosstalk[r * 3 + c] : (r == c ? 1.0 : 0.0)) * med[c];
  }
  if (!(v[0] > 0.0) || !(v[1] > 0.0) || !(v[2] > 0.0)) return false;
  const double dg = std::log10((v[1] / v[0]) / (expected[1] / expected[0]));
  const double db = std::log10((v[2] / v[0]) / (expected[2] / expected[0]));
  return std::fabs(dg) <= kBaseColourTol && std::fabs(db) <= kBaseColourTol;
}

// The tiles of the window (x0, y0, w, h) of the half-size view: true with the base and the kept tiles' record when a
// group of at least kBaseMinTiles agrees.
bool base_from_tiles(const negicc::RgbView& view, const negicc::RgbView& raw, int x0, int y0, int w, int h, float clip,
                     const double* crosstalk, const double* expected, FilmBaseCapture& out) {
  struct Tile { int x, y, w, h; int i, j; RegionStats st; };
  std::vector<Tile> tiles;
  // Square tiles, kBaseTileGrid along the longer side and never under 8 pixels, so that a rebate strip beside a
  // picture, far narrower than the frame, holds whole tiles.
  const int side = std::max(8, std::max(w, h) / kBaseTileGrid);
  const int gx = std::max(1, w / side), gy = std::max(1, h / side);
  for (int j = 0; j < gy; ++j) {
    for (int i = 0; i < gx; ++i) {
      Tile t;
      t.x = x0 + (int)((long long)w * i / gx);
      t.y = y0 + (int)((long long)h * j / gy);
      t.w = x0 + (int)((long long)w * (i + 1) / gx) - t.x;
      t.h = y0 + (int)((long long)h * (j + 1) / gy) - t.y;
      t.i = i;
      t.j = j;
      if (t.w < 4 || t.h < 4) continue;
      tiles.push_back(t);
    }
  }
  #pragma omp parallel for schedule(dynamic, 16)
  for (size_t k = 0; k < tiles.size(); ++k) {
    tiles[k].st = region_stats(view, raw, tiles[k].x, tiles[k].y, tiles[k].w, tiles[k].h, clip);
  }
  // The tiles that must agree: kBaseMinTiles, and at least one tile in kBaseMinShare of them.
  const int min_tiles = std::max(kBaseMinTiles, (int)(tiles.size() / kBaseMinShare));
  out.tiles_total = (int)tiles.size();
  out.tiles_kept = 0;
  // Unclipped tiles with light in every channel; then the smooth ones among them, by the median tile's spread.
  std::vector<const Tile*> lit;
  std::vector<float> spreads;
  for (const Tile& t : tiles) {
    if (t.st.n && t.st.clipped <= negicc::kBaseClipFraction && t.st.med[0] > 0.0 && t.st.med[1] > 0.0 &&
        t.st.med[2] > 0.0) {
      lit.push_back(&t);
      spreads.push_back((float)t.st.spread);
    }
  }
  if ((int)lit.size() < min_tiles) return false;
  const double spread_limit = std::max(kBaseTileMaxSpread, kBaseSpreadFactor * median_of(spreads));
  std::vector<const Tile*> smooth;
  for (const Tile* t : lit) {
    if (t->st.spread <= spread_limit && base_coloured(t->st.med, crosstalk, expected)) smooth.push_back(t);
  }
  if ((int)smooth.size() < min_tiles) return false;
  // The brightest group: from the brightest tile down, the first with kBaseMinTiles tiles within the margin of it.
  std::sort(smooth.begin(), smooth.end(), [](const Tile* a, const Tile* b) { return a->st.med[1] > b->st.med[1]; });
  std::vector<const Tile*> kept;
  for (size_t i = 0; i < smooth.size() && kept.empty(); ++i) {
    const double ref = smooth[i]->st.med[1];
    std::vector<const Tile*> group;
    for (const Tile* t : smooth) {
      if (std::fabs(std::log10(t->st.med[1] / ref)) <= kBaseTileMargin) group.push_back(t);
    }
    if ((int)group.size() >= min_tiles) kept = group;
  }
  if (kept.empty()) return false;
  // The group was found from its brightest tile; it is taken again around its own median, so that the base is the
  // level of that film and not of its brighter half.
  {
    std::vector<float> greens;
    for (const Tile* t : kept) greens.push_back((float)t->st.med[1]);
    const double ref = median_of(greens);
    std::vector<const Tile*> group;
    for (const Tile* t : smooth) {
      if (std::fabs(std::log10(t->st.med[1] / ref)) <= kBaseTileMargin) group.push_back(t);
    }
    if ((int)group.size() >= min_tiles) kept = group;
  }
  // Film at the base's level can lie in several places (the rebate on either side of a picture, an unexposed shadow
  // in it); the largest piece of touching tiles is one stretch of film under one part of the light, and is taken
  // alone when it is large enough.
  {
    std::vector<int> label((size_t)gx * gy, -1);
    for (const Tile* t : kept) label[(size_t)t->j * gx + t->i] = 0;
    int pieces = 0, best = 0, best_n = 0;
    std::vector<int> stack;
    for (const Tile* t : kept) {
      const int start = t->j * gx + t->i;
      if (label[start] != 0) continue;
      const int id = ++pieces;
      int n = 0;
      label[start] = id;
      stack.push_back(start);
      while (!stack.empty()) {
        const int at = stack.back();
        stack.pop_back();
        ++n;
        const int ai = at % gx, aj = at / gx;
        const int next[4][2] = {{ai - 1, aj}, {ai + 1, aj}, {ai, aj - 1}, {ai, aj + 1}};
        for (const auto& nb : next) {
          if (nb[0] < 0 || nb[0] >= gx || nb[1] < 0 || nb[1] >= gy) continue;
          const int q = nb[1] * gx + nb[0];
          if (label[q] == 0) {
            label[q] = id;
            stack.push_back(q);
          }
        }
      }
      if (n > best_n) {
        best_n = n;
        best = id;
      }
    }
    if (pieces > 1 && best_n >= min_tiles) {
      std::vector<const Tile*> piece;
      for (const Tile* t : kept) {
        if (label[(size_t)t->j * gx + t->i] == best) piece.push_back(t);
      }
      kept = piece;
    }
  }
  std::vector<float> ch[3];
  int bx0 = kept[0]->x, by0 = kept[0]->y, bx1 = bx0, by1 = by0;
  double lo = 1e30, hi = -1e30;
  for (const Tile* t : kept) {
    for (int c = 0; c < 3; ++c) ch[c].push_back((float)t->st.med[c]);
    bx0 = std::min(bx0, t->x);
    by0 = std::min(by0, t->y);
    bx1 = std::max(bx1, t->x + t->w);
    by1 = std::max(by1, t->y + t->h);
    const double d = -std::log10(t->st.med[1]);
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  for (int c = 0; c < 3; ++c) out.decode_rgb[c] = median_of(ch[c]);
  out.tiles_kept = (int)kept.size();
  out.spread_density = hi - lo;
  out.roi[0] = bx0 * 2;
  out.roi[1] = by0 * 2;
  out.roi[2] = (bx1 - bx0) * 2;
  out.roi[3] = (by1 - by0) * 2;
  return true;
}

}  // namespace

bool read_film_base_capture(const std::string& path, const int* roi, const negicc::FlatFieldPtr& flat,
                            const double* crosstalk, const double* expected_base, double scan_shutter_s,
                            double scan_iso, FilmBaseCapture& out, std::string& err) {
  out = FilmBaseCapture{};
  if (!is_regular_file(path)) {
    err = "no such file: " + path;
    return false;
  }
  std::unique_ptr<LibRaw> raw(new LibRaw());
  const int ret = open_and_unpack_raw_file(raw.get(), path);
  if (ret != LIBRAW_SUCCESS) {
    err = "cannot open or unpack " + path + ": " + libraw_strerror(ret);
    return false;
  }
  if (!is_sony_cfa_geometry(raw.get()) || !raw->imgdata.rawdata.raw_image) {
    err = path + " is not a Sony CFA RAW (the film base capture is read as the scan's mosaic is)";
    return false;
  }
  float white_level = 65535.0f;
  if (!subtract_cfa_black_in_place(raw.get(), white_level)) {
    err = "invalid CFA mosaic geometry in " + path;
    return false;
  }
  const int src_w = raw->imgdata.sizes.width, src_h = raw->imgdata.sizes.height;
  const size_t pitch = raw->imgdata.sizes.raw_pitch / 2;
  const unsigned filters = raw->imgdata.idata.filters;
  const uint16_t* origin = raw->imgdata.rawdata.raw_image + (size_t)raw->imgdata.sizes.top_margin * pitch +
                           raw->imgdata.sizes.left_margin;
  const int half_w = src_w / 2, half_h = src_h / 2;
  std::shared_ptr<const negicc::FlatTables> flat_tables;
  negicc::FlatSampleArgs flat_args{};
  if (flat) {
    try {
      flat_tables = flat->tables(half_w, half_h, true);
      flat_args = flat_tables->host();
    } catch (const std::exception& e) {
      err = std::string("the flat field does not fit ") + path + ": " + e.what();
      return false;
    }
  }
  const negicc::RgbView view =
      negicc::RgbView::from_cfa_half(origin, pitch, src_w, src_h, filters, nullptr, flat_args, flat_tables);
  // The same mosaic without the flat field, for the clipping (region_stats).
  const negicc::RgbView raw_view = negicc::RgbView::from_cfa_half(origin, pitch, src_w, src_h, filters, nullptr);
  const float clip = negicc::kSensorClip * white_level;

  if (roi) {
    if (roi[0] < 0 || roi[1] < 0 || roi[2] <= 0 || roi[3] <= 0 || (long long)roi[0] + roi[2] > src_w ||
        (long long)roi[1] + roi[3] > src_h) {
      err = "the film base region " + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," +
            std::to_string(roi[2]) + "," + std::to_string(roi[3]) + " is off the capture's sensor (" +
            std::to_string(src_w) + "x" + std::to_string(src_h) + ")";
      return false;
    }
    // The half-size pixels (2x2 photosites) the region touches. An odd sensor's last column or row is in none.
    const int hx = roi[0] / 2, hy = roi[1] / 2;
    if (hx >= half_w || hy >= half_h) {
      err = "the film base region " + std::to_string(roi[0]) + "," + std::to_string(roi[1]) + "," +
            std::to_string(roi[2]) + "," + std::to_string(roi[3]) + " lies on the capture's last column or row (" +
            std::to_string(src_w) + "x" + std::to_string(src_h) + "), which the half-size decode (2x2 photosites "
            "to a pixel) does not hold";
      return false;
    }
    const int hw = std::max(1, std::min(half_w - hx, (roi[0] + roi[2] + 1) / 2 - hx));
    const int hh = std::max(1, std::min(half_h - hy, (roi[1] + roi[3] + 1) / 2 - hy));
    const RegionStats st = region_stats(view, raw_view, hx, hy, hw, hh, clip);
    if (!st.n || st.med[0] <= 0.0 || st.med[1] <= 0.0 || st.med[2] <= 0.0) {
      err = "the film base region of " + path + " holds no light";
      return false;
    }
    if (st.clipped > negicc::kBaseClipFraction) {
      err = "the film base region of " + path + " is clipped (" + std::to_string(100.0 * st.clipped) +
            "% of its pixels): capture the film base at a shorter exposure";
      return false;
    }
    for (int c = 0; c < 3; ++c) out.decode_rgb[c] = st.med[c];
    for (int k = 0; k < 4; ++k) out.roi[k] = roi[k];
    out.roi_source = "given";
    out.spread_density = st.spread;
    if (st.spread > kBaseTileMaxSpread) {
      log_warn("the film base region of %s is not smooth (green density spread %.4f)\n", path.c_str(), st.spread);
    }
  } else if (base_from_tiles(view, raw_view, 0, 0, half_w, half_h, clip, crosstalk, expected_base, out)) {
    out.roi_source = "tiles";
  } else {
    // The thinnest film on the capture, as a TriLED scan's base is read when it shows no rebate; on the mosaic as
    // recorded (the same pixels before the flat field) it must not be clipped.
    float thin[3] = {0.0f, 0.0f, 0.0f};
    bool thin_clipped = false;
    negicc::thinnest_film_base(view, white_level, thin, &thin_clipped);
    if (!(thin[0] > 1.0f) || !(thin[1] > 1.0f) || !(thin[2] > 1.0f)) {
      err = "no usable film base on " + path + " (no smooth film)";
      return false;
    }
    if (thin_clipped) {
      err = "the film base capture " + path + " is clipped: capture the film base at a shorter exposure";
      return false;
    }
    log_warn("the film base capture %s has no smooth region; its thinnest film is taken\n", path.c_str());
    for (int c = 0; c < 3; ++c) out.decode_rgb[c] = thin[c];
    out.roi[0] = out.roi[1] = 0;
    out.roi[2] = src_w;
    out.roi[3] = src_h;
    out.roi_source = "statistics";
    out.tiles_kept = 0;
  }

  // The capture's exposure: EXIF, else LibRaw's.
  // As the scan's (Scan::scan_exposure): a capture that records none is taken at 1/8 s and ISO 100, with a warning.
  double shutter = 0.0, iso = 0.0;
  const bool has_exif = read_exif_exposure_d(path, shutter, iso);
  if (!has_exif || !(shutter > 0.0))
    shutter = raw->imgdata.other.shutter > 0 ? (double)raw->imgdata.other.shutter : 0.0;
  if (!has_exif || !(iso > 0.0)) {
    iso = raw->imgdata.other.iso_speed > 0 ? std::round((double)raw->imgdata.other.iso_speed) : 0.0;
  }
  if (!(shutter > 0.0) || !(iso > 0.0)) {
    if (!(shutter > 0.0)) shutter = kDefaultScanShutterS;
    if (!(iso > 0.0)) iso = kDefaultScanIso;
    log_warn("the film base capture %s records no exposure; taking %.4g s at ISO %.0f to scale it to the scan's\n",
             path.c_str(), shutter, iso);
  }
  out.shutter_s = shutter;
  out.iso = iso;
  out.exposure_scale = (scan_shutter_s * scan_iso) / (shutter * iso);
  for (int r = 0; r < 3; ++r) {
    double v = 0.0;
    for (int c = 0; c < 3; ++c) v += (crosstalk ? crosstalk[r * 3 + c] : (r == c ? 1.0 : 0.0)) * out.decode_rgb[c];
    out.rgb[r] = v * out.exposure_scale;
  }
  if (!(out.rgb[0] > 0.0) || !(out.rgb[1] > 0.0) || !(out.rgb[2] > 0.0)) {
    err = "the film base of " + path + " is not positive after the crosstalk correction";
    return false;
  }
  return true;
}
