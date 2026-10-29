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

#include "rgb_input.h"

#include <algorithm>
#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>

#include <lcms2.h>
#include <tiffio.h>

#include <jpeglib.h>  // after <cstdio>: it needs FILE and size_t

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
