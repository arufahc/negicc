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

// neg_process's inputs: image geometry (rotation, cropping, the --roi auto fallback), RAW decoding with LibRaw
// (load_raw), TIFF and JPEG files, and the inputs besides one RAW (an RGB file, three captures one per light).

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "film_frame_detector.h"
#include "flat_field.h"
#include "libraw/libraw.h"
#include "white_light_balance.h"

// --------------------------------------------------------------------------------------------------------------------
// Geometry
// neg_process's image geometry: rotation, cropping and turning a decoded LibRaw image, rectangles, and the frame
// detection fallback of --roi auto.
// --------------------------------------------------------------------------------------------------------------------

// Below this frame detection confidence neg_process warns, sets frame_detection.low_confidence, and falls back to the
// full visible sensor frame (geometry.roi_source = "auto_fallback").
constexpr float kLowFrameConfidence = 0.5f;

// A 180-degree rotation is both flips; leaves rot_cw 0, 90 or 270.
void normalize_rotation(int& rot_cw, bool& hflip, bool& vflip);
// Extracts a sub-rectangle (rx, ry, rw, rh) of proc's decoded image (imgdata.image), turned by rot_cw and flipped,
// in place of it. False when the rectangle is outside the image or memory runs out.
bool extract_roi_and_transform(LibRaw* proc, int rx, int ry, int rw, int rh, int rot_cw, bool hflip, bool vflip);
// Moves rect (x, y, w, h) inside bw x bh, shrinking it only when it is larger.
void clamp_rect(int r[4], int bw, int bh);

// detect_film_frame, but a failure (a lens-cap or blank frame, an image too small to analyse) is a low-confidence
// result: ok false with a reason, conf 0, so --roi auto takes the full visible frame like any other low confidence.
void detect_frame_or_fallback(const FilmImage& mosaic, const FrameDetectionParams& params, FrameDetection& det);

// --------------------------------------------------------------------------------------------------------------------
// RAW decoding
// neg_process's RAW decoding with LibRaw: the camera checks, the mosaic as the frame detector reads it, the black
// subtraction of the zero-copy CFA path, and load_raw (LibRaw's decode: camera RGB, Rec.2020 or the white-light
// decode of a d65 bundle; also a linear DNG).
// --------------------------------------------------------------------------------------------------------------------

// 4-shot pixel shift merge is only available with Sony A7RM4 camera.
bool is_sony_a7rm4(LibRaw* proc);
// Checks if the camera has Sony Bayer RGGB CFA geometry.
bool is_sony_cfa_geometry(LibRaw* proc);
// The four files of a Sony 4-shot pixel shift capture, validated by the group tags the camera writes into their
// makernotes (LibRaw: PixelShiftGroupID, nShotsInPixelShiftGroup, numInPixelShiftGroup).
struct PixelShiftTags {
  unsigned group_id = 0;
  int shots_in_group = 0;  // nShotsInPixelShiftGroup
  int num_in_group = 0;    // numInPixelShiftGroup, 1 for the first shot
  bool present() const { return shots_in_group > 0 && num_in_group > 0; }
};

enum class PixelShiftGroupCheck { kNoTags, kOk, kBad };

// kNoTags: none of the four files carries group tags (the caller falls back to the file counters). kOk: one group of
// four shots, given in shot order. kBad: err says why (some files tagged and some not, different groups, a group of
// another size, shots out of order or repeated).
inline PixelShiftGroupCheck check_pixel_shift_group(const PixelShiftTags tags[4], std::string& err) {
  int n_present = 0;
  for (int i = 0; i < 4; ++i) n_present += tags[i].present() ? 1 : 0;
  if (n_present == 0) return PixelShiftGroupCheck::kNoTags;
  if (n_present != 4) {
    err = "only some of the 4 files carry pixel shift group tags";
    return PixelShiftGroupCheck::kBad;
  }
  for (int i = 0; i < 4; ++i) {
    if (tags[i].group_id != tags[0].group_id) {
      err = "the 4 files belong to different pixel shift groups";
      return PixelShiftGroupCheck::kBad;
    }
    if (tags[i].shots_in_group != 4) {
      err = "the files belong to a pixel shift group of " + std::to_string(tags[i].shots_in_group) +
            " shots, not 4";
      return PixelShiftGroupCheck::kBad;
    }
    if (tags[i].num_in_group != i + 1) {
      err = "4-shot pixel shift files are out of frame order (file " + std::to_string(i + 1) + " is shot " +
            std::to_string(tags[i].num_in_group) + " of its group)";
      return PixelShiftGroupCheck::kBad;
    }
  }
  return PixelShiftGroupCheck::kOk;
}

// Opens a RAW or DNG file into proc, repairing leading spaces in UniqueCameraModel (e.g. SilverFast DNGs) if needed.
int open_raw_file(LibRaw* proc, const std::string& fn);
// Opens and unpacks a RAW or DNG file into proc, keeping any patched header stream alive across unpack().
int open_and_unpack_raw_file(LibRaw* proc, const std::string& fn);

// Called with a LibRaw whose imgdata.image holds the half-size image of its RAW (read_half_size): iwidth x iheight
// pixels, row-major, LibRaw's 4 channels each (R, G, B in the first three); a point (x, y) of the RAW is (x / 2, y / 2)
// there. False fails the load.
using HalfSizeAnalysis = std::function<bool(const LibRaw& proc)>;

// Called by load_raw's white-light decode with its first half-size image (as HalfSizeAnalysis sees it). Leaves in k
// the gains for the camera's R, G and B channels that the mosaic takes before the camera matrix
// (film_base_balance_gains, white_light_balance.h); k comes in as 1, 1, 1, which leaves the mosaic as it is.
using ChannelBalance = std::function<void(const LibRaw& half, float k[3])>;


// Called by load_raw with the unpacked RAW in proc (imgdata.rawdata.raw_image holds the CFA mosaic, margins
// included), before the half-size analysis and the decode. Sets roi to the region to decode (x, y, w, h in the
// visible sensor area, imgdata.sizes.width x height), as --roi would; false fails the load.
using RawRegionFinder = std::function<bool(LibRaw& proc, int roi[4])>;

// How load_raw decodes a RAW to RGB.
enum class RawDecode {
  kRec2020,     // LibRaw to linear Rec.2020 with unit channel multipliers (no profile bundle, no -r -g -b)
  kCameraRgb,   // sensor RGB, unit channel multipliers: a camera_raw_rgb (TriLED) bundle or -r -g -b
  kWhiteLight,  // a rec2020_linear (D65) bundle: prepare_white_light_mosaic, then decode_white_light
};

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
bool film_image_from_raw(LibRaw& proc, FilmImage& img);

// Step 1 of the zero-copy CFA path: subtracts each channel's black in place on the unpacked CFA mosaic, and gives the
// white level above black.
bool subtract_cfa_black_in_place(LibRaw* proc, float& out_white_level);

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
// |balance|, when set along with analyse, is called for kWhiteLight with a first half-size decode. When it gives gains other than 1,
// the mosaic's camera channels are multiplied by them and the half size is decoded again for analyse; the output is
// decoded from the same balanced mosaic. Other decodes ignore it.
//
// |find_region|, when set, is called on the unpacked mosaic first and gives the region to decode in place of roi
// (--roi auto: the detected frame), so that the analysis and the decode see only that region. A finder that leaves
// the region empty (width or height 0) only looked at the mosaic, and roi stays.
LibRaw* load_raw(const std::string& fn, bool debayer, int qual, bool crop,
                 RawDecode decode = RawDecode::kRec2020, const int* roi = nullptr,
                 const HalfSizeAnalysis& analyse = {}, const ChannelBalance& balance = {},
                 const RawRegionFinder& find_region = {},
                 const negicc::FlatField* flat = nullptr,
                 LibRaw* unpacked_proc = nullptr);

// --------------------------------------------------------------------------------------------------------------------
// TIFF and JPEG files
// Non-RAW inputs of neg_process: TIFF (libtiff) and JPEG (libjpeg-turbo) files read into 16-bit samples, and their
// conversion to the linear Rec.2020 a rec2020_linear (d65) bundle converts (Little-CMS).
// --------------------------------------------------------------------------------------------------------------------

namespace negicc {

// A decoded TIFF or JPEG: width x height pixels of `channels` interleaved samples (1, grey, or 3, RGB; a TIFF's
// extra samples such as alpha are dropped), scaled to 0..65535 (8-bit samples times 257).
struct RgbImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  int bits = 0;                  // bits per sample in the file: 8 or 16
  std::vector<uint16_t> pixels;  // width * height * channels
  std::vector<uint8_t> icc;      // the file's embedded ICC profile, empty when it has none
};

// True when the file starts with a JPEG signature.
bool is_jpeg_file(const std::string& path);

// Reads an 8- or 16-bit unsigned-integer TIFF (strips or tiles, chunky or planar, any codec libtiff is built with:
// none, LZW, Deflate, PackBits, JPEG, ...), RGB or grey (min-is-black). False with err set otherwise.
bool read_tiff(const std::string& path, RgbImage& out, std::string& err);

// Reads a baseline or progressive JPEG as RGB (a grey JPEG is expanded to RGB), with its ICC profile (APP2).
bool read_jpeg(const std::string& path, RgbImage& out, std::string& err);

// Converts an RGB image in place to linear Rec.2020 (D65 white), relative colorimetric, through its embedded ICC
// profile; without one, 8-bit data is taken as sRGB and 16-bit data as already linear Rec.2020 (left as it is).
// Returns how the image was taken ("icc", "srgb" or "linear") in `source`.
bool to_linear_rec2020(RgbImage& img, std::string& source, std::string& err);

}  // namespace negicc

// --------------------------------------------------------------------------------------------------------------------
// Inputs besides one RAW
// neg_process's inputs besides one RAW: the input modes, and loading an RGB TIFF or JPEG or three captures, one per
// light, into a LibRaw image as load_raw leaves a decoded RAW.
// --------------------------------------------------------------------------------------------------------------------

// The inputs neg_process reads besides one RAW (or the four of a pixel shift).
enum class InputMode {
  kRaw,        // a CFA RAW or DNG (load_raw)
  kLinearDng,  // a linear (demosaiced) DNG, e.g. a film scanner's: load_raw too, white light (d65) only
  kRgbFile,    // an RGB TIFF or JPEG (read_tiff, read_jpeg): linear Rec.2020 (d65), or a TIFF of linear camera RGB (tri)
  kThreeShot,  // three captures, one under each of the red, green and blue lights: RAWs, DNGs or TIFFs (tri)
};

// True when LibRaw opens the file (a RAW or a DNG; the header only).
bool libraw_opens(const std::string& path);
// True when the file is a DNG with no CFA (its pixels already demosaiced; the header only).
bool is_linear_dng(const std::string& path);

// The non-RAW inputs (InputMode): an RGB TIFF or JPEG through its ICC profile to linear Rec.2020 (to_linear_rec2020), or
// with |camera_rgb| a TIFF's values as the linear camera RGB a tri bundle converts, or three captures, each giving the channel of its light: R of the first, G of the second, B of the third, in camera
// units (a RAW or DNG decoded to camera RGB as load_raw decodes it, a TIFF's samples as they are; a grey TIFF gives
// its one channel), each scaled to the first capture's exposure when all three record theirs. Returns a LibRaw holding
// the image as load_raw returns a decoded RAW, after cropping it to |roi| or, with |frame_det|, to the frame the
// detector finds on the image (as --roi auto), and running |analyse| on its half-size copy, as load_raw does.
LibRaw* load_rgb_input(const std::vector<std::string>& files, InputMode mode, bool camera_rgb, int qual,
                              bool crop, const int* roi,
                              const HalfSizeAnalysis& analyse, FrameDetection* frame_det,
                              const FrameDetectionParams& det_params, std::string& source);
