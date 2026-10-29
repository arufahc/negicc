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

#include "np_input.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "np_geometry.h"
#include "rgb_input.h"


bool libraw_opens(const std::string& path) {
  LibRaw probe;
  return probe.open_file(path.c_str()) == LIBRAW_SUCCESS;
}

bool is_linear_dng(const std::string& path) {
  LibRaw probe;
  return probe.open_file(path.c_str()) == LIBRAW_SUCCESS && probe.imgdata.idata.dng_version &&
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
  LibRaw* proc = nullptr;
  std::string err;
  if (mode == InputMode::kRgbFile) {
    negicc::RgbImage img;
    const bool jpeg = negicc::is_jpeg_file(files[0]);
    printf("Loading %s file %s\n", jpeg ? "JPEG" : "TIFF", files[0].c_str());
    if (!(jpeg ? negicc::read_jpeg(files[0], img, err) : negicc::read_tiff(files[0], img, err))) {
      fprintf(stderr, "ERROR! %s\n", err.c_str());
      return nullptr;
    }
    if (img.channels != 3) {
      fprintf(stderr, "ERROR! %s is a grey image; one file must be RGB (three files give one channel each)\n",
              files[0].c_str());
      return nullptr;
    }
    std::string how = "camera";
    if (!camera_rgb && !negicc::to_linear_rec2020(img, how, err)) {
      fprintf(stderr, "ERROR! %s: %s\n", files[0].c_str(), err.c_str());
      return nullptr;
    }
    source = std::string(jpeg ? "jpeg" : "tiff") + ":" + how;
    printf("Image size: %dx%d, %d-bit, taken as %s\n", img.width, img.height, img.bits,
           how == "icc" ? "its ICC profile" : how == "srgb" ? "sRGB" : how == "camera" ? "linear camera RGB"
                                                                                       : "linear Rec.2020");
    w = img.width;
    h = img.height;
    if (!(proc = new_image_proc(w, h))) {
      fprintf(stderr, "ERROR! %s: cannot hold a %dx%d image\n", files[0].c_str(), w, h);
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
        LibRaw* raw = load_raw(files[k], true, qual, crop, RawDecode::kCameraRgb);
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
            fprintf(stderr, "ERROR! %s: cannot hold a %dx%d image\n", files[k].c_str(), cw, ch);
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
        printf("Loading TIFF file %s\n", files[k].c_str());
        if (negicc::is_jpeg_file(files[k]) || !negicc::read_tiff(files[k], img, err)) {
          fprintf(stderr, "ERROR! %s: %s\n", files[k].c_str(),
                  err.empty() ? "three captures must be RAWs, DNGs or TIFFs" : err.c_str());
          return delete proc, nullptr;
        }
        cw = img.width;
        ch = img.height;
        const int src = img.channels == 3 ? k : 0;
        channel.resize((size_t)cw * ch);
        for (size_t i = 0; i < channel.size(); ++i) channel[i] = img.pixels[i * img.channels + src];
        if (k == 0 && !(proc = new_image_proc(cw, ch))) {
          fprintf(stderr, "ERROR! %s: cannot hold a %dx%d image\n", files[k].c_str(), cw, ch);
          return nullptr;
        }
        kinds += k ? ",tiff" : "tiff";
      }
      if (k == 0) {
        w = cw;
        h = ch;
      } else if (cw != w || ch != h) {
        fprintf(stderr, "ERROR! The three captures differ in size: %s is %dx%d, %s is %dx%d\n", files[0].c_str(), w,
                h, files[k].c_str(), cw, ch);
        return delete proc, nullptr;
      }
      for (size_t i = 0; i < channel.size(); ++i) proc->imgdata.image[i][k] = channel[i];
    }
    if (exposure[0] > 0 && exposure[1] > 0 && exposure[2] > 0) {
      for (int k = 1; k < 3; ++k) {
        const double scale = exposure[0] / exposure[k];
        if (scale == 1.0) continue;
        printf("Channel %c scaled by %.4g to the first capture's exposure\n", "RGB"[k], scale);
        for (size_t i = 0; i < (size_t)w * h; ++i)
          proc->imgdata.image[i][k] = (ushort)std::min(65535.0, std::round(proc->imgdata.image[i][k] * scale));
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
  // As load_raw: --roi auto decodes (here: keeps) only the frame before the analysis; a given --roi is applied by the
  // analysis to its half-size image and cut from the full image after it.
  if (frame_det && (region[0] || region[1] || region[2] != w || region[3] != h)) {
    if (!extract_roi_and_transform(proc, region[0], region[1], region[2], region[3], 0, false, false)) {
      proc->free_image();
      delete proc;
      return nullptr;
    }
    w = proc->imgdata.sizes.iwidth;
    h = proc->imgdata.sizes.iheight;
  }
  if (analyse) {
    const int hw = w / 2, hh = h / 2;
    std::unique_ptr<LibRaw> half(new_image_proc(hw, hh));
    if (!half) return delete proc, nullptr;
    half->imgdata.other.shutter = proc->imgdata.other.shutter;
    half->imgdata.other.iso_speed = proc->imgdata.other.iso_speed;
#pragma omp parallel for schedule(static)
    for (int y = 0; y < hh; ++y) {
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
  if (!frame_det && roi && (region[0] || region[1] || region[2] != w || region[3] != h)) {
    if (!extract_roi_and_transform(proc, region[0], region[1], region[2], region[3], 0, false, false)) {
      proc->free_image();
      delete proc;
      return nullptr;
    }
  }
  printf("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}
