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

// Two things, one after the other:
//
//   1. negicc's library: negicc_core.h, then the parts that need a third-party header (LittleCMS's lcms2.h, LibRaw's
//      libraw/libraw.h), in this order (each needs only those before it): the white-light balance, neg_process's
//      inputs, the profile bundle, the DINOv3 intent model with the Auto search, and the conversion parameter search.
//      The library's sources (libnegicc_dinov3.a) and the companion application's own sources use only this part.
//   2. neg_process's own declarations, from the banner "neg_process's own declarations" to the end: what its program
//      sources (neg_process.cc, np_input.cc, print_emulation.cc, np_output.cc and pipeline.cc) share, in this order:
//      the print, the film base, the options, the output and the pipeline.
//
// Contents, each part under a banner with its name:
//
//   White-light balance      the film-base balance of the white-light decode
//   Inputs                   geometry, RAW decoding, TIFF and JPEG files, the inputs besides one RAW
//   Profile bundle           the film profile as the host holds it, the bundles and their brackets, the profiles, the
//                            input matrix and the shoulder
//   DINOv3                   the intent model and the Auto search that runs on it
//   Parameter search         the conversion parameter search, its renderers, the curve fit and the tone stage's plan
//   ---- neg_process's own declarations ----
//   Print                    the density output's encoding and record, the print materials, a frame's print and its
//                            CLF, and the print ICC
//   Film base                the film base read from a capture of unexposed film
//   Options                  the command line
//   Output                   the picture's TIFF, the density file (TIFF or DPX) and its print CLFs, and the sidecar
//   Pipeline                 the stages main runs in turn and the state they hand on

#ifndef NEGICC_H
#define NEGICC_H

#include <lcms2.h>

#include <array>
#include <chrono>
#include <cstring>
#include <optional>

#include "libraw/libraw.h"
#include "negicc_core.h"

// ====================================================================================================================
// White-light balance: the film-base balance of the white-light decode (LibRaw)
// ====================================================================================================================

// The film-base balance of the white-light decode: gains on the camera's own channels, applied to the mosaic before
// the camera matrix. Header-only, so that neg_process (np_input.cc, load_raw) and negicc-station (its CapturedImage
// decode) compile the same code and balance a scan the same way.

namespace negicc {

// The gains for the camera channels that put the film base |base| (linear Rec.2020, as the white-light decode of proc
// gives it) on the profile's film base |profile_base|: K = (M^-1 profile_base) / (M^-1 base) per channel, M the camera
// matrix the decode applies (rec2020_rgb x rgb_cam), divided by its largest so no channel is raised and nothing new
// clips. k is left as it is when either base has a camera channel that is not positive, or when a gain is not a finite
// positive number (a base so extreme that the arithmetic overflows or underflows), or when the camera matrix's
// determinant is 0 or not finite. (A camera matrix of rank 2 whose determinant is rounding residue still passes; the
// gains it gives then lie within a few 1e-4 of 1.)
//
// Why before the matrix: a light of another colour than the profile's gives the base another colour in the camera's
// own channels. Corrected after the matrix has mixed them, the matrix's negative terms drive dense areas below zero
// (clamped: blue shadows and skies with a warm light). And the profile's patches were read as Rec.2020 divided by
// the calibration scan's base, which is M with its rows divided by that base: dividing by another scan's base scales
// the rows by another vector, while after these gains every scan's base is the profile's. For the calibration scan
// itself K is 1.
inline void film_base_balance_gains(const LibRaw& proc, const double base[3], const int profile_base[3], float k[3]) {
  // M = rec2020_rgb x rgb_cam, as LibRaw's convert_to_rgb builds it for output_color 8.
  double m[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      m[i][j] = 0;
      for (int n = 0; n < 3; ++n) m[i][j] += LibRaw_constants::rec2020_rgb[i][n] * proc.imgdata.color.rgb_cam[n][j];
    }
  // adj = det(M) M^-1 (the adjugate). The determinant cancels in the ratio of the two bases; it only gives the sign.
  double adj[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      adj[j][i] = m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3] -
                  m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3];
  const double det = m[0][0] * adj[0][0] + m[0][1] * adj[1][0] + m[0][2] * adj[2][0];
  if (!std::isfinite(det) || det == 0.0) return;   // no inverse: k as it is
  double gain[3];
  for (int c = 0; c < 3; ++c) {
    double cam = 0, cam_profile = 0;  // the two bases in camera channel c, each times det(M)
    for (int j = 0; j < 3; ++j) {
      cam += adj[c][j] * base[j];
      cam_profile += adj[c][j] * profile_base[j];
    }
    if (!(cam / det > 0 && cam_profile / det > 0)) return;
    gain[c] = cam_profile / cam;
    if (!(std::isfinite(gain[c]) && gain[c] > 0)) return;
  }
  const double largest = std::max({gain[0], gain[1], gain[2]});
  float out[3];
  for (int c = 0; c < 3; ++c) {
    out[c] = (float)(gain[c] / largest);
    if (!(out[c] > 0)) return;
  }
  for (int c = 0; c < 3; ++c) k[c] = out[c];
}

// Multiplies the white-light mosaic (black-subtracted and white-balanced in place, as np_input.cc's
// prepare_white_light_mosaic leaves it) by k, per camera channel. A CFA mosaic holds one sample a
// photosite, of the photosite's colour; a linear DNG holds three or four a pixel, R, G and B first.
// Each product is rounded and held in the samples' range, 0 to 65535 (0 for one that is not a number), so a gain
// above 1, negative or not finite cannot overflow a sample. The mosaic is left as it is when proc holds none, or when
// a photosite of the CFA has no colour of the camera's channels (a sensor without a colour filter array).
inline void scale_white_light_mosaic(LibRaw* proc, const float k[3]) {
  const auto& S = proc->imgdata.sizes;
  const auto& R = proc->imgdata.rawdata;
  const int samples = R.raw_image ? 1 : R.color4_image ? 4 : 3;
  uint16_t* raw = R.raw_image      ? R.raw_image
                  : R.color4_image ? &R.color4_image[0][0]
                  : R.color3_image ? &R.color3_image[0][0]
                                   : nullptr;   // no mosaic
  if (!raw) return;
  const size_t pitch = S.raw_pitch / 2 / samples;  // pixels per row of the raw buffer
  float tile[36];  // the gain of each photosite of a 6x6 CFA tile (Bayer repeats every 2, X-Trans every 6)
  for (int r = 0; samples == 1 && r < 6; ++r)
    for (int c = 0; c < 6; ++c) {
      const int colour = proc->COLOR(r, c);
      if (colour < 0 || colour > 3) return;
      tile[r * 6 + c] = k[colour == 3 ? 1 : colour];  // 3 is the second green
    }
  auto scaled = [](uint16_t v, float gain) {
    const float s = std::nearbyint(v * gain);
    return s > 0.0f ? (uint16_t)std::min(s, (float)UINT16_MAX) : (uint16_t)0;
  };
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < S.height; ++y) {
    uint16_t* p = raw + ((size_t)(y + S.top_margin) * pitch + S.left_margin) * samples;
    for (int x = 0; x < S.width; ++x, p += samples) {
      if (samples == 1) {
        p[0] = scaled(p[0], tile[(y % 6) * 6 + x % 6]);
      } else {
        for (int c = 0; c < 3; ++c) p[c] = scaled(p[c], k[c]);
      }
    }
  }
}

}  // namespace negicc

// ====================================================================================================================
// Inputs: geometry, RAW decoding with LibRaw, TIFF and JPEG files, multi-file inputs
// ====================================================================================================================

// neg_process's inputs: image geometry (rotation, cropping, the --roi auto fallback), RAW decoding with LibRaw
// (load_raw), TIFF and JPEG files, and the inputs besides one RAW (an RGB file, three captures one per light).

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
// Opens and unpacks a RAW or DNG file into proc (open_raw_file, then unpack()). LIBRAW_FILE_UNSUPPORTED, with the
// reason on stderr, for a sensor whose photosites are not a square grid (45-degree or non-square pixels) and for an
// X-Trans pattern holding a colour other than 0, 1 and 2; LIBRAW_DATA_ERROR for a file whose levels fail
// raw_levels_valid.
int open_and_unpack_raw_file(LibRaw* proc, const std::string& fn);
// Whether the black and white levels of the unpacked proc are what its 16-bit samples can hold: the white level
// (maximum) within 1 to 65535, each black (black, cblack[0..3], the cblack[6..] pattern and a DNG's black levels as
// the file gives them) within 0 to 65535, and the largest black of a photosite below the white level. False, with
// the level named on stderr (fn names the file), when not.
bool raw_levels_valid(const LibRaw& proc, const std::string& fn);

// Called with a LibRaw whose imgdata.image holds the half-size image of its RAW (read_half_size): iwidth x iheight
// pixels, row-major, LibRaw's 4 channels each (R, G, B in the first three); a point (x, y) of the RAW is (x / 2, y / 2)
// there. False fails the load.
using HalfSizeAnalysis = std::function<bool(const LibRaw& proc)>;

// Called by load_raw's white-light decode with its first half-size image (as HalfSizeAnalysis sees it). Leaves in k
// the gains for the camera's R, G and B channels that the mosaic takes before the camera matrix
// (film_base_balance_gains, the white-light balance above); k comes in as 1, 1, 1, which leaves the mosaic as it is.
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
// and cropbox specified in the RAW metadata. When it is true and there is no roi, a debayered decode cuts the RAW's
// inset crop (inset_crop) as it cuts an roi.
//
// |decode| picks the colour path of a debayered decode (RawDecode); kWhiteLight decodes with LibRaw's bilinear
// demosaic (or qual when it is >= 0), never grain-aware.
//
// |analyse|, when set, is called with the half-size image of the RAW before it is decoded: LibRaw's half-size readout
// of the mosaic (read_half_size), or for kWhiteLight its half-size white-light decode, in the same space as the
// output.
//
// |balance|, when set along with analyse, is called for kWhiteLight with a first half-size decode. When it gives gains
// other than 1, the mosaic's camera channels are multiplied by them and the half size is decoded again for analyse; the
// output is decoded from the same balanced mosaic. Other decodes ignore it.
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

// The RAW's inset crop as left, top, width and height in the visible image (sizes.width x sizes.height, after the
// margins), when the RAW gives one that is not the whole image. Its right and bottom edges are not clamped to the image.
bool inset_crop(const LibRaw& r, int box[4]);

// Whether load_raw's decode through LibRaw with a flat field (not white light) decodes the half-size image its
// analysis reads before the flat divides the mosaic: when the visible mosaic is even, so the half-size image is the
// flat's half size. The analysis then reads that image through the flat (an RgbView carrying it), as the CFA path
// does; otherwise the flat divided the mosaic first.
bool flat_after_half_size(const LibRaw& r);

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
  kRgbFile,    // an RGB TIFF or JPEG (read_tiff, read_jpeg): linear Rec.2020 (d65), or a TIFF of linear camera RGB
               // (tri)
  kThreeShot,  // three captures, one under each of the red, green and blue lights: RAWs, DNGs or TIFFs (tri)
};

// True when LibRaw opens the file (a RAW or a DNG; the header only).
bool libraw_opens(const std::string& path);
// True when the file is a DNG with no CFA (its pixels already demosaiced; the header only).
bool is_linear_dng(const std::string& path);

// The non-RAW inputs (InputMode): an RGB TIFF or JPEG through its ICC profile to linear Rec.2020 (to_linear_rec2020),
// or with |camera_rgb| a TIFF's values as the linear camera RGB a tri bundle converts, or three captures, each giving
// the channel of its light: R of the first, G of the second, B of the third, in camera units (a RAW or DNG decoded to
// camera RGB as load_raw decodes it, a TIFF's samples as they are; a grey TIFF gives its one channel), each scaled to
// the first capture's exposure when all three record theirs. Returns a LibRaw holding the image as load_raw returns a
// decoded RAW: |analyse| runs on the whole image at half size (three RAW captures: their own half-size readouts), as
// load_raw does, and the image is then cropped to |roi|, with |frame_det| to the frame the detector finds on the
// image (as --roi auto), or with |crop| to the first RAW capture's inset crop.
LibRaw* load_rgb_input(const std::vector<std::string>& files, InputMode mode, bool camera_rgb, int qual,
                              bool crop, const int* roi,
                              const HalfSizeAnalysis& analyse, FrameDetection* frame_det,
                              const FrameDetectionParams& det_params, std::string& source);

// ====================================================================================================================
// Profile bundle: the film profile as the host holds it, the bundles and their brackets (LittleCMS)
// ====================================================================================================================

struct SolverConfig;  // the DINOv3 part below

// --------------------------------------------------------------------------------------------------------------------
// The film profile as the host holds it (the parsed tables and the pointers into them), the output encodings and the
// shared pipeline's configuration built from a profile. Every pixel stage lives once, in the colour pipeline part of
// negicc_core.h (namespace negicc); these are what the host needs around it.
// --------------------------------------------------------------------------------------------------------------------

using Rgb = negicc::Rgb;
using Lab = negicc::Lab;

// neg_process's output encodings: the two the shared pipeline encodes, and two that bypass its encoding stage.
enum OutputEncoding : int {
  ENCODE_SRGB = negicc::ENCODE_SRGB,                // sRGB transfer curve, with the highlight shoulder
  ENCODE_SRGB_LINEAR = negicc::ENCODE_SRGB_LINEAR,  // linear-light sRGB primaries
  ENCODE_CUSTOM = 2,                                // PCS XYZ through Little-CMS into an output ICC profile
  ENCODE_NONE = 3,                                  // the corrected camera RGB, no film profile applied
  ENCODE_DENSITY = 4                                // density above the film base as code values (print_emulation.cc)
};

// The film profile's AToB0 pipeline (negicc::ProfileData) with host defaults: no profile, identity matrix.
struct ProfileData : negicc::ProfileData {
  ProfileData()
      : negicc::ProfileData{0, {nullptr, nullptr, nullptr}, {0, 0, 0}, {nullptr, nullptr, nullptr}, {0, 0, 0}, 0,
                            {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}, nullptr, {0, 0, 0}} {}
};

// A parsed film profile: the tables ProfileData points into.
struct ParsedFilmProfile {
  ProfileData data;
  std::vector<float> in_trc[3];
  std::vector<float> out_trc[3];
  std::vector<float> clut;

  // Points data at this object's tables (after a copy or a move).
  void rebind_pointers() {
    for (int c = 0; c < 3; ++c) {
      data.in_trc[c] = in_trc[c].empty() ? nullptr : in_trc[c].data();
      data.out_trc[c] = out_trc[c].empty() ? nullptr : out_trc[c].data();
    }
    data.clut = clut.empty() ? nullptr : clut.data();
  }
};

// The shared pipeline's configuration for a profile, knee and shoulder. Built once per conversion or search
// iteration, never per pixel: set_shoulder() solves the shoulder's knee.
inline negicc::PipelineConfig make_pipeline_config(const ProfileData& prof, float knee, float shoulder_x_hi = 1.0f,
                                                   int encoding = ENCODE_SRGB) {
  negicc::PipelineConfig cfg{};
  cfg.profile = prof;
  cfg.knee = knee;
  cfg.encoding = encoding;
  negicc::set_shoulder(cfg, shoulder_x_hi);
  return cfg;
}

// --------------------------------------------------------------------------------------------------------------------
// The profile bundle: a film stock's brackets, each with its film profile.
// --------------------------------------------------------------------------------------------------------------------

struct BundleTarget {
  int index = 0;
  std::string name;
  std::string display_name;
  double iso = 100.0;
  std::string shutter = "1/8s";
  bool has_target_ev = false;
  double target_ev = 0.0;
  std::vector<float> patches_g;
  float tgt_min_g = 0.0f;
  float tgt_max_g = 0.0f;
  std::vector<uint8_t> icc_bytes;
  std::string icc_file;   // the ICC file a split bundle names for the bracket; empty: inline in the bundle
  ParsedFilmProfile film_prof;

  // For the conversion parameter search (the parameter search part below): the bracket's camera exposure
  // (shutter * ISO / 100, seconds), the tonal capacity of its grey wedge and the profcheck record.
  float exposure_s = 0.125f;
  float tonal_capacity_ev = 0.0f;
  std::string profcheck_output;
};

// The colour space a bundle's brackets were measured in, which is the space a scan is decoded to for it.
enum class InputSpace {
  // Sensor RGB (narrowband TriLED light): black subtracted, no white balance, then the bundle's crosstalk matrix.
  kCameraRawRgb,
  // Linear Rec.2020 (white D65 light): LibRaw's white-light decode (np_input.cc decode_white_light), with no
  // crosstalk matrix.
  kRec2020Linear,
};

// "camera_raw_rgb" or "rec2020_linear", as the bundles write it.
const char* to_string(InputSpace space);

// A density measure a scan's densities are written in (-P): Status M, SMPTE RP 180 printing density (the Cineon
// convention) or Academy Printing Density (SMPTE ST 2065-2, which ADX encodes).
enum DensityMeasure { kDensityStatusM = 0, kDensityRp180 = 1, kDensityApd = 2, kDensityMeasures = 3 };
// The -P value ("statusm", "rp180", "apd") and the bundle's density_matrices key of a measure.
const char* density_measure_name(DensityMeasure m);
const char* density_measure_key(DensityMeasure m);

// A bundle's matrix for one density measure: D_measure = m * D_light, both above the film base, rows R, G, B.
// m is the product of the bundle's two parts: the measure's film_to_measure (the film's dye amounts to the measure's
// density) times light_to_film (the light's density to the dye amounts), with the measure's light_density_factors.
struct DensityMatrix {
  bool present = false;
  double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double film_to_measure[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // the measure's part alone
  double light_to_film[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};     // the light's part alone
  double light_density_factors[3] = {1.0, 1.0, 1.0};         // the calibration on the light's densities, when applied
  std::string target_curves;          // what defines the measure, as the bundle names it
};

// What a print of a density file takes from the film's chart for one exposure bracket (density_matrices.brackets):
// the chart's mid grey as the light's density above the film base, and the slopes that give the film's grey scale
// equal contrast in the three channels (green 1), in Academy Printing Density.
struct DensityBracket {
  std::string name;
  double grey_light[3] = {0.0, 0.0, 0.0};
  // The bracket's own slopes, when the bundle keeps them. They are read only for a bundle without grey_balance_curves;
  // a bundle with the curve, under either light, takes its slopes from the curve (print_emulation.cc curve_slopes).
  bool has_slopes = false;
  double slopes[3] = {1.0, 1.0, 1.0};
  // The bracket's slope relative to the grey balance curve
  // (density_matrices.brackets.targets.<bracket>.grey_relative.b), fitted by the profile-building tools over its grey
  // steps: each step's residual from the curve, r_c = G_green - G_c in the curve's APD, regressed as a_c + t_c
  // (G_c - A_mid). The print takes the tilt t_c linear in its mid grey A between the two brackets whose A_mid enclose
  // A, held beyond the end ones (print_emulation.cc print_tilt_at); the offset a_c is not read. Zero without
  // grey_relative: the curve alone.
  bool has_tilt = false;
  double tilt[3] = {0.0, 0.0, 0.0};
};

// The film model's spectral grid, 380-780 nm at 1 nm: kSpectralSamples samples from kSpectralFirstNm in steps of
// kSpectralStepNm, the grid of print_emulation.cc's CIE tables.
constexpr int kSpectralSamples = 401;
constexpr double kSpectralFirstNm = 380.0, kSpectralStepNm = 1.0;

// The film stock's spectra (the bundle's film_model), on the spectral grid: the spectral density of a unit amount
// of each dye, the base, and the dye amounts along the film's neutral scale. A print material given as a file takes
// its matrix from them (print_emulation.cc load_spektrafilm_print).
struct FilmModel {
  bool present = false;
  std::string name;
  std::vector<double> dye[3], base;           // spectral densities, kSpectralSamples samples
  std::vector<double> log_exposure;           // of each sample; it selects those the print matrix is fitted to
  std::vector<double> neutral[3];             // dye amounts along log_exposure
};

// The film's grey balance curve (density_matrices.grey_balance_curves, under either light): for each light channel c,
// apd[c][i] is the APD (Academy Printing Density) of the film's grey that has light density start + step i in
// channel c, rising. From the chart's grey steps of every bracket, computed by the profiling tool.
struct GreyBalanceCurves {
  bool present = false;
  double start = 0.0, step = 0.01;
  std::vector<double> apd[3];
};

// One of a bundle's print ICCs (print_profiles): the film printed on a print material, for --print <id>. The ICC's
// own input curves are for reference_bracket; neg_process writes each bracket's from it (print_emulation.cc). The ICC
// itself is read only when a print asks for it (read_icc): a bundle holds eight, 4.6 MB, which a conversion without
// --print does not keep, and a missing one refuses only the print that needs it. The two SHA-256 (hex; empty when not
// recorded) are of the spektrafilm profile the ICC was computed from and of the ICC itself; an exported bundle (its ICC
// a file beside bundle.json) records the ICC's, a station bundle (the ICC inline) does not.
struct BundlePrint {
  std::string id, name, reference_bracket, spektrafilm_sha256, icc_sha256;
  bool icc_in_file = false;
  std::string icc_file;   // the file beside bundle.json, when icc_in_file
  std::function<bool(std::vector<uint8_t>& icc, std::string& err)> read_icc;
};

struct ProfileBundle {
  std::string path;
  std::string key;          // bundle key ("Portra400-A7R4-TriLED"): the export record's, else parsed from path
  std::string source_file;  // the station bundle a split bundle was exported from
  std::string camera_name;
  // The camera models the bundle holds for (camera_models), as a RAW records them (LibRaw's imgdata.idata.model). A
  // camera_raw_rgb bundle's crosstalk correction is not applied to a RAW that states a model not among them, compared
  // exactly (pipeline.cc decode). has_camera_models false: a bundle from before the field, which is not compared.
  std::vector<std::string> camera_models;
  bool has_camera_models = false;
  InputSpace input_space = InputSpace::kCameraRawRgb;
  // The crosstalk correction of a camera_raw_rgb bundle; identity (has_crosstalk false) for rec2020_linear.
  float crosstalk_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double crosstalk_matrix_d[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool has_crosstalk = false;
  bool has_film_base = false;
  int profile_film_base_rgb[3] = {1, 1, 1};
  float film_base_avg[3] = {0.0f, 0.0f, 0.0f};    // unrounded film base (profile_film_base_rgb rounds it)
  double film_base_avg_d[3] = {0.0, 0.0, 0.0};    // double precision for exact station parity
  std::string film_base_shutter = "1/8s";
  float film_base_iso = 100.0f;
  double normalization_target = 55000.0;
  std::vector<BundleTarget> targets;
  // The density export (-P statusm, rp180, apd): the 3x3 matrices that turn the scanning light's channel densities
  // above the film base into a standard density measure, fitted when the bundle was built from the light's spectrum
  // (density_matrices), and the light's record with that spectrum (light_source), kept as the bundle writes it for
  // the output's metadata. A bundle from before the fields has none.
  DensityMatrix density_matrix[kDensityMeasures];
  std::vector<DensityBracket> density_brackets;   // empty when the bundle has none
  GreyBalanceCurves grey_balance;                 // not present in a bundle from before it
  std::vector<BundlePrint> prints;                // the bundle's print ICCs, by print material id
  FilmModel film_model;                           // not present in a bundle from before the field
  std::string light_source_json;      // the "light_source" object, compact; empty when the bundle has none
  std::string density_fit_json;       // "density_matrices" without the matrices (the dye spectra and the fit's record)
};

// Seconds of a shutter speed as the profiles record it, a string ("1/8s", "0.4s", "2"); throws
// std::invalid_argument when it cannot be read.
float shutter_to_seconds(const std::string& shutter);
double shutter_to_seconds_d(const std::string& shutter);

// Reads (ExposureTime in seconds, ISO) in double precision from a TIFF-based RAW's EXIF IFD (matching
// auto_exposure.read_exif_exposure). Returns true when ExposureTime was read: a RATIONAL above 0. An ISO is taken when
// above 0; shutter_s and iso keep what they held for a value the file does not give.
bool read_exif_exposure_d(const std::string& filepath, double& shutter_s, double& iso);

// Whether a crosstalk correction matrix, as the floats that are applied, is usable: every element finite, and the
// determinant a normal float that is larger than the rounding error of its own computation (so exactly dependent
// rows, whose determinant comes out as rounding residue, are not taken as invertible). The correction is the inverse
// of the light's mixing (DESIGN.md 2.4), so it is invertible. One rule for a bundle's matrix and for rows given with
// -r, -g and -b.
bool crosstalk_matrix_invertible(const float m[9]);

// Computes negicc-station's merged crosstalk matrix (film_profiling.compute_merged_crosstalk) in double precision
// and rounds to float32 when out_f32 is non-null. When film_base_from_scan is true (e.g. rebate sampled from the
// scan itself), exposure_ratio is 1.0; otherwise exposure_ratio is exposure_base / exposure_scan.
void compute_station_merged_matrix(const ProfileBundle& bundle, double scan_shutter_s, double scan_iso,
                                   const double* custom_film_base_rgb, bool film_base_from_scan,
                                   float* out_f32, double* out_f64 = nullptr, double* out_row_scale_f64 = nullptr);

// Bundle and stock keys of a profile name: "profile_Portra160-A7R4-TriLED_20261003_190751.json" gives
// ("Portra160-A7R4-TriLED", "portra160"). Empty strings when the name has none.
void parse_profile_stem(const std::string& raw_name, std::string& bundle_key, std::string& stock_key);

// Check by content if file is a JSON profile bundle (first non-space character is '{')
bool is_json_bundle(const std::string& resolved_path);

// Parse AToB0 pipeline tags from an open cmsHPROFILE into ParsedFilmProfile. False, with the reason logged, when the
// profile has no AToB0 tag or one with a matrix other than the identity.
bool parse_film_profile_pipeline(cmsHPROFILE hprof, ParsedFilmProfile& prof);

// Reads a file a split bundle names (a target's "icc_profile") into out; false when it cannot.
using BundleFileReader = std::function<bool(const std::string& name, std::vector<uint8_t>& out)>;

// Load and parse a multi-bracket JSON profile bundle: an all-in-one bundle from negicc-station (ICC profiles in
// base64), or a split one (profiles/<Key>/bundle.json, its directory, with the ICC profiles in files next to it;
// negicc-station tools/export_negicc_data.py). Throws when the bundle cannot be read or has no usable bracket, when
// its input_space is neither camera_raw_rgb nor rec2020_linear (no input_space is camera_raw_rgb: bundles from
// before the field), or when a rec2020_linear bundle carries a crosstalk correction matrix.
void load_profile_bundle(const std::string& resolved_path, ProfileBundle& bundle);

// Parse a JSON profile bundle held in memory (one embedded in neg_process); name is reported as bundle.path and
// read_file supplies the ICC profiles of a split bundle. Throws as load_profile_bundle.
void load_profile_bundle_from_memory(const char* data, size_t size, const std::string& name,
                                     const BundleFileReader& read_file, ProfileBundle& bundle);

// The bracket's mid grey as green APD: its grey_light_density through the bundle's APD matrix. The loader refuses a
// bracket for which it is not finite; the grey balance curve and the print read it (print_emulation.cc).
double bracket_mid_apd(const ProfileBundle& bundle, const DensityBracket& bracket);

// --------------------------------------------------------------------------------------------------------------------
// Profiles
// neg_process's profiles: resolving a profile path, and loading an ICC film profile, or a profile bundle JSON.
// --------------------------------------------------------------------------------------------------------------------

bool is_directory(const std::string& path);
bool is_regular_file(const std::string& path);
// The path of a profile argument: as given, else under profiles/ (and a few historical locations).
std::string resolve_profile_path(const std::string& input_path);
// Loads an ICC film profile (-p with a single ICC profile, for research).
bool load_film_profile(const std::string& raw_path, ParsedFilmProfile& prof);
// Loads the negicc-station profile bundle JSON a -p argument names (as given or under profiles/), or the bundle.json
// of a split-bundle directory (profiles/<Key>). False when the argument is not a bundle (an ICC profile).
bool load_bundle_arg(const std::string& arg, ProfileBundle& bundle);
// Reads an ICC profile file into a malloc'ed buffer (*prof_out, *size): 0 on success, -1 otherwise.
int read_profile(const std::string& prof_name, unsigned** prof_out, unsigned* size);

// --------------------------------------------------------------------------------------------------------------------
// Input matrix and shoulder
// neg_process's conversion: the input matrix (crosstalk correction, film base scale, exposure and gains) and the
// highlight shoulder's record.
// --------------------------------------------------------------------------------------------------------------------

// The soft knee of the highlight compression: fixed, as the station's Auto (multi_target_optimizer.cc AUTO_KNEE).
constexpr float kKnee = 0.95f;

// The highlight roll-off (negicc_core.h's shoulder) as the sidecar records it (tone.shoulder*): measured for every
// bundle conversion (measure_rolloff_x_hi in the parameter search part below), off with an ICC profile. Only the sRGB
// output applies it.
struct ShoulderInfo {
  float x_hi = 1.0f;                 // 1 is off
  std::string source = "off";        // off | rolloff
  bool has_q995 = false;
  float q995 = 0.0f;                 // the frame's 99.5% point of the pre-shoulder max channel
  bool has_knee = false;
  float knee = 1.0f;                 // the linear knee x_k
  bool active() const { return x_hi > negicc::kShoulderOff; }
};

// The input matrix: the crosstalk rows r_coef, g_coef and b_coef, each scaled by its channel's film base ratio
// (profile_film_base_rgb / film_base_rgb) and by global_exposure_comp, gain_g and gain_b as negicc-station applies
// them, into merged_matrix (3x3, row-major). Prints the rows. Throws std::invalid_argument when a row has fewer than
// three coefficients or merged_matrix is null.
void adjust_correction_matrix(const std::vector<float>& r_coef, const std::vector<float>& g_coef,
                              const std::vector<float>& b_coef, float global_exposure_comp, float gain_g,
                              float gain_b, const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb, float* merged_matrix);

// ====================================================================================================================
// DINOv3: the intent model and the Auto search that runs on it
// ====================================================================================================================

// The DINOv3 intent model and the Auto search that runs on it, in the pipeline's order: the preprocessing of a
// preview, the inference runners and the token heads, DinoV3Engine, and the Auto solver. dinov3.cc holds the code.

// ====================================================================================================================
// Preprocessing
// Takes an 8-bit RGB preview; gives the patch-aligned network size, the resized image and the tone-bin weights of
// each patch, as the network was trained on them.
// ====================================================================================================================

// Antialiased 2-pass separable bilinear resize on 8-bit RGB with kernel support widening;
// bit-exact with Pillow's Image.Resampling.BILINEAR (fixed point, 8-bit intermediate).
// The preprocessing functions here and trained_cells_lab, self_consistency_residual and residual_from_cells throw
// std::invalid_argument for a size below 1, a src_stride short of a row (src_w * 3 bytes), or sizes whose products
// (a row in bytes, a grid's sides in pixels, the cells times 3) do not fit an int.
void antialiased_bilinear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                                     uint8_t* dst, int dst_w, int dst_h);

// cv2.resize(..., interpolation=cv2.INTER_LINEAR) on 8-bit RGB (the fixed-point SIMD path): a bilinear point sample.
// The search's residual does not use it (it reads trained_cells_lab, below); it is kept for
// negicc-station's parity test of this resize (src/test_dinov3_preprocess.cpp).
void cv_linear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int dst_w, int dst_h);

// intent_model.compute_tone_weights on a (grid_w * patch) x (grid_h * patch) RGB image: [N, 3] weights.
void compute_tone_weights(const uint8_t* img_u8, int grid_w, int grid_h, int patch_size, float* out_weights);

// Computes patch-aligned target dimensions preserving aspect ratio (intent_model.compute_aspect_preserved_shape)
void compute_aspect_preserved_shape(int w, int h, int max_size, int patch_size,
                                    int& target_w, int& target_h, int& grid_w, int& grid_h);

// ====================================================================================================================
// Engine
// The DINOv3 intent model: the inference runners (ONNX Runtime on the CPU, TensorRT on CUDA), the small heads that
// read the backbone's tokens (NeutralHead, ToneHead), and DinoV3Engine, which preprocesses a frame and runs them.
// Takes an 8-bit sRGB preview and the backbone, its per-stock adapter and the token heads; gives the intent field
// (mu, sigma per patch), the ranker features and the heads' outputs (DinoV3InferenceResult).
// ====================================================================================================================

// --------------------------------------------------------------------------------------------------------------------
// Runners
// Inference runners behind DinoV3Engine: one forward pass on preprocessed tensors.
//
// A runner holds either a fused intent model (backbone + adapter: outputs "mu" and "sigma") or the DINOv3 backbone
// alone (output "tokens", the layer 9 + 12 patch tokens [1, N, 768]). A backbone runner is paired with a
// DinoV3AdapterRunner, the per-stock adapter that maps the tokens to the intent field, so one backbone file serves
// every film stock and the adapters ship inside neg_process (the embedded data part of negicc_core.h).
//
// The adapter runs where the backbone runs: ONNX Runtime with an ONNX Runtime backbone, TensorRT with a TensorRT
// backbone, reading the backbone's tokens from device memory (DinoV3Runner::device_tokens) so they never leave the GPU.
//
// The ONNX Runtime CPU runners (dinov3.cc) are built everywhere. The TensorRT runners (dinov3.cc, after them)
// are built only with USE_CUDA=1 and run the device-specific backbone.engine and adapter.engine files
// `make dinov3_engine` builds. When USE_CUDA=0, dinov3.cc holds fallback stubs instead.
// These sources live in negicc and are consumed by negicc-station via git submodule at 3rd_party/negicc.
// --------------------------------------------------------------------------------------------------------------------

// Multi-head inference result matching Python IntentPrediction.
struct DinoV3InferenceResult {
    std::vector<float> mu;            // [1, N, 3] Target Lab color field
    std::vector<float> sigma;         // [1, N, 3] Tolerance field
    std::vector<float> pooled;        // [1, 1152] Exposure ranker CLS+mean features
    std::vector<float> pooled_tone;   // [1, 3, 768] Tone-stratified ranker features
    std::vector<float> tone_mass;     // [3] Shadows, mid-tones, highlights mass fractions
    std::vector<float> features_3459; // [3459] Contiguous concatenated: pooled + pooled_tone + tone_mass
    std::vector<float> neutral_p;     // [N] p(neutral grey) per patch, when a neutral head is set (else empty)
    std::vector<float> token_sd;      // [768] the tokens' standard deviation over the patches (N - 1 in the
                                      // denominator), with DinoV3Engine::set_token_spread(true) (else empty)
    double neutral_ms = 0.0;          // time of the neutral head, included in duration_ms
    bool has_pooled = false;
    bool has_pooled_tone = false;
    int target_w = 0;
    int target_h = 0;
    int grid_w = 0;
    int grid_h = 0;
    int num_patches = 0;
    double duration_ms = 0.0;
};

// Destination buffers for multi-head DINOv3 model outputs.
// Any pointer left as nullptr will not be copied back by the runner.
struct DinoV3RunnerOutputs {
    float* tokens = nullptr;       // [1, patches, 768] layer 9 + 12 patch tokens (backbone models)
    float* mu = nullptr;           // [1, patches, 3] CIELAB mean field
    float* sigma = nullptr;        // [1, patches, 3] CIELAB tolerance field
    float* pooled = nullptr;       // [1, 1152] Exposure ranker CLS + patch mean
    float* pooled_tone = nullptr;  // [1, 3, 768] Tone-stratified ranker features
};

class DinoV3Runner {
public:
    virtual ~DinoV3Runner() = default;
    virtual const char* name() const = 0;
    // The model takes the tone-bin weights input "pool_weights" (exports since the tone features were added).
    virtual bool has_pool_weights() const = 0;
    // The model ends at the backbone ("tokens") rather than the intent field ("mu", "sigma").
    virtual bool has_tokens() const { return false; }
    virtual bool has_pooled() const { return false; }
    virtual bool has_pooled_tone() const { return false; }
    virtual size_t memory_bytes() const { return 0; }
    // Device address of the last run's "tokens" output when the backbone runs on TensorRT (valid until the next run),
    // so a TensorRT adapter reads the tokens where they are; null for other runners.
    virtual const void* device_tokens() const { return nullptr; }
    // Device-to-host copies of the tokens made so far (tests: a GPU backbone paired with a GPU adapter makes none).
    virtual int token_downloads() const { return 0; }

    // Multi-head forward pass with selectable destination buffers.
    virtual void run(const float* pixel_values, int height, int width, const float* pool_weights, int patches,
                     const DinoV3RunnerOutputs& outputs) = 0;
};

// Per-stock adapter: tokens [1, N, 768] -> mu, sigma [1, N, 3].
class DinoV3AdapterRunner {
public:
    virtual ~DinoV3AdapterRunner() = default;
    // Tokens in host memory.
    virtual void run(const float* tokens, int patches, float* mu, float* sigma) = 0;
    // The adapter runs on the GPU and takes its tokens from device memory (run_device).
    virtual bool on_device() const { return false; }
    // Tokens in device memory (DinoV3Runner::device_tokens of a TensorRT backbone); mu and sigma land on the host.
    virtual void run_device(const void* /*device_tokens*/, int /*patches*/, float* /*mu*/, float* /*sigma*/);
    virtual size_t memory_bytes() const { return 0; }
};

// ONNX Runtime adapter from an in-memory adapter.onnx (input "tokens", outputs "mu", "sigma").
std::unique_ptr<DinoV3AdapterRunner> make_ort_adapter(const void* onnx_data, size_t onnx_size, int num_threads = 1);

// TensorRT adapter from an in-memory adapter.engine (make dinov3_engine); throws std::runtime_error on failure or in
// CPU-only builds.
std::unique_ptr<DinoV3AdapterRunner> make_trt_adapter(const void* engine_data, size_t engine_size);

// Deserialises a TensorRT engine; throws std::runtime_error on failure or in CPU-only builds.
std::unique_ptr<DinoV3Runner> make_trt_runner(const std::string& engine_path);
// The same from a serialized engine held in memory.
std::unique_ptr<DinoV3Runner> make_trt_runner(const void* engine_data, size_t engine_size);

// Creates an ONNX Runtime CPU runner. num_threads: the runtime's own threads; 0 takes the program's thread count
// (omp_get_max_threads(): see default_thread_count in negicc_core.h), so the network runs with the count every parallel
// region has.
std::unique_ptr<DinoV3Runner> make_ort_runner(const std::string& model_path, int num_threads = 0);
// The same from an .onnx held in memory.
std::unique_ptr<DinoV3Runner> make_ort_runner(const void* onnx_data, size_t onnx_size, int num_threads = 0);

// Checks if a CUDA device is available.
bool cuda_device_available();

// --------------------------------------------------------------------------------------------------------------------
// Token heads
// The small heads that read the DINOv3 tokens alongside the intent adapter, each a plain C++ implementation that
// needs no runtime of its own: the neutral-patch head (NeutralHead), which gives the white-light curve fit its neutral
// anchors, and the tone heads (ToneHead), which give the tone stage its targets on both lights.
// --------------------------------------------------------------------------------------------------------------------

// The exact GELU (Hendrycks and Gimpel, 2016; torch.nn.GELU's default), x Phi(x), in double: the heads' activation.
// 0.7071... is 1 / sqrt(2).
inline double exact_gelu(double a) { return 0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440)); }

// The neutral-patch head: per DINOv3 patch, the probability that the patch is neutral grey in a good rendering of
// the frame, from the backbone's layer 9 + 12 tokens [N, 768] (DinoV3Runner "tokens", the adapters' input).
// p = sigmoid(w2 . gelu(W1 ((t - mean) / std) + b1) + b2), W1 [256][768], exact (erf) GELU, as the PyTorch model it
// was trained as. Weights: models/neutral_head/neutral_head.bin (models/neutral_head/export_neutral_head.py writes
// it from neutral_head.pt; README.md there has the training and validation).
// A plain C++ implementation (about 0.2 GFLOP per 672-patch preview), so the head needs no runtime of its own and
// runs wherever the tokens are on the host.
class NeutralHead {
public:
    // Parses neutral_head.bin (magic "NPH1", uint32 in_dim and hidden, then float32 mean, std, w1, b1, w2, b2).
    // False, with the reason in error, when the data is not such a file.
    bool load(const void* data, size_t size, std::string& error) {
        in_dim_ = hidden_ = 0;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (!p || size < 12 || std::memcmp(p, "NPH1", 4) != 0) {
            error = "not a neutral head file (magic NPH1)";
            return false;
        }
        uint32_t d = 0, h = 0;
        std::memcpy(&d, p + 4, 4);
        std::memcpy(&h, p + 8, 4);
        const size_t floats = 2 * (size_t)d + (size_t)h * d + 2 * (size_t)h + 1;
        if (d == 0 || h == 0 || d > 65536 || h > 65536 || size != 12 + 4 * floats) {
            error = "neutral head file has the wrong size for in_dim " + std::to_string(d) + ", hidden " +
                    std::to_string(h);
            return false;
        }
        std::vector<float> v(floats);
        std::memcpy(v.data(), p + 12, 4 * floats);
        const float* f = v.data();
        mean_.assign(f, f + d);
        f += d;
        std_.assign(f, f + d);  // divided by, as in PyTorch (not multiplied by a reciprocal)
        f += d;
        // w1 transposed to [in][hidden], so the inner loop runs over contiguous hidden units (vectorises without
        // reassociating the sums, which -ffp-contract=off and the absence of -ffast-math keep in order).
        w1t_.resize((size_t)d * h);
        for (uint32_t j = 0; j < h; ++j)
            for (uint32_t k = 0; k < d; ++k) w1t_[(size_t)k * h + j] = f[(size_t)j * d + k];
        f += (size_t)h * d;
        b1_.assign(f, f + h);
        f += h;
        w2_.assign(f, f + h);
        f += h;
        b2_ = *f;
        for (const float x : v) {
            if (!std::isfinite(x)) {
                error = "neutral head file has a weight that is not finite";
                return false;
            }
        }
        for (const float s : std_) {
            if (!(s > 0.0f)) {
                error = "neutral head file has a standard deviation that is not positive";
                return false;
            }
        }
        in_dim_ = (int)d;
        hidden_ = (int)h;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }
    int in_dim() const { return in_dim_; }

    // p[i] for the n patches of tokens (n x in_dim, row-major).
    void run(const float* tokens, int n, float* p) const {
        const int d = in_dim_, h = hidden_;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::vector<float> x(d), acc(b1_);
            const float* t = tokens + (size_t)i * d;
            for (int k = 0; k < d; ++k) x[k] = (t[k] - mean_[k]) / std_[k];
            for (int k = 0; k < d; ++k) {
                const float xk = x[k];
                const float* w = w1t_.data() + (size_t)k * h;
                for (int j = 0; j < h; ++j) acc[j] += xk * w[j];
            }
            double z = b2_;
            for (int j = 0; j < h; ++j) {
                const double a = acc[j];
                z += (double)w2_[j] * exact_gelu(a);
            }
            p[i] = (float)(1.0 / (1.0 + std::exp(-z)));
        }
    }

private:
    int in_dim_ = 0, hidden_ = 0;
    std::vector<float> mean_, std_, w1t_, b1_, w2_;
    float b2_ = 0.0f;
};

// The tone heads of the tone stage (both lights): from the frozen DINOv3 features of a preview, the tone a lab scan of
// the frame would have, as 11 descriptors of its pixel L* and chroma at network size (mean and standard deviation of
// each):
//   L2 L5 L10 L50 L90 L95 L98 (L* percentiles), IQR (L75 - L25), R9010 (L90 - L10), Cmean (mean C*ab), C90.
// Input: the 1920 pooled features [CLS of layer 12, mean patch token of layer 9, mean patch token of layer 12 (the
// backbone's "pooled" output, 1152), the patch tokens' standard deviation (DinoV3InferenceResult::token_sd, 768)],
// standardised with the stored xm and xs. Model: Linear(1920, 512), GELU (exact, erf), Linear(512, 256), GELU,
// Linear(256, 22); mean = out[j] * ys[j] + ym[j], sd = exp(clamp(out[11 + j], -6, 4) / 2) * ys[j], as the PyTorch
// model it was trained as. Weights: models/tone_head/tone_head.bin (models/tone_head/export_tone_head.py writes it
// from tone_head.pt; README.md there has the training and validation). Plain C++, about 1.1 M multiply-adds per run.
class ToneHead {
public:
    enum Target { kL2, kL5, kL10, kL50, kL90, kL95, kL98, kIQR, kR9010, kCmean, kC90, kTargets };
    static constexpr int kInDim = 1920;
    static const char* target_name(int t) {
        static const char* names[kTargets] = {"L2", "L5", "L10", "L50", "L90", "L95", "L98", "IQR", "R9010",
                                              "Cmean", "C90"};
        return t >= 0 && t < kTargets ? names[t] : "";
    }

    // Parses tone_head.bin (magic "TNH1", uint32 in_dim, hidden1, hidden2, targets, then float32 xm, xs, w1, b1, w2,
    // b2, w3, b3, ym, ys). False, with the reason in error, when the data is not such a file.
    bool load(const void* data, size_t size, std::string& error) {
        in_dim_ = 0;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (!p || size < 20 || std::memcmp(p, "TNH1", 4) != 0) {
            error = "not a tone head file (magic TNH1)";
            return false;
        }
        uint32_t d = 0, h1 = 0, h2 = 0, n = 0;
        std::memcpy(&d, p + 4, 4);
        std::memcpy(&h1, p + 8, 4);
        std::memcpy(&h2, p + 12, 4);
        std::memcpy(&n, p + 16, 4);
        if (d != (uint32_t)kInDim || n != (uint32_t)kTargets || h1 == 0 || h2 == 0 || h1 > 65536 || h2 > 65536) {
            error = "tone head file is not 1920 features to 11 targets";
            return false;
        }
        const size_t floats = 2 * (size_t)d + (size_t)h1 * d + h1 + (size_t)h2 * h1 + h2 + 2 * (size_t)n * h2 +
                              2 * (size_t)n + 2 * (size_t)n;
        if (size != 20 + 4 * floats) {
            error = "tone head file has the wrong size";
            return false;
        }
        std::vector<float> v(floats);
        std::memcpy(v.data(), p + 20, 4 * floats);
        const float* f = v.data();
        auto take = [&f](std::vector<float>& dst, size_t k) {
            dst.assign(f, f + k);
            f += k;
        };
        take(xm_, d);
        take(xs_, d);
        take(w1_, (size_t)h1 * d);
        take(b1_, h1);
        take(w2_, (size_t)h2 * h1);
        take(b2_, h2);
        take(w3_, 2 * (size_t)n * h2);
        take(b3_, 2 * (size_t)n);
        take(ym_, n);
        take(ys_, n);
        for (const float x : v) {
            if (!std::isfinite(x)) {
                error = "tone head file has a weight that is not finite";
                return false;
            }
        }
        for (const float s : xs_) {
            if (!(s > 0.0f)) {
                error = "tone head file has a feature scale that is not positive";
                return false;
            }
        }
        in_dim_ = (int)d;
        h1_ = (int)h1;
        h2_ = (int)h2;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }

    // mean[kTargets] and sd[kTargets] from the 1920 features. False when the head is not loaded.
    bool run(const float* features, double* mean, double* sd) const {
        if (!loaded()) return false;
        std::vector<double> x(in_dim_), a1(h1_), a2(h2_);
        for (int k = 0; k < in_dim_; ++k) x[k] = (double)((features[k] - xm_[k]) / xs_[k]);  // float, as in training
        for (int j = 0; j < h1_; ++j) {
            const float* w = w1_.data() + (size_t)j * in_dim_;
            double s = b1_[j];
            for (int k = 0; k < in_dim_; ++k) s += (double)w[k] * x[k];
            a1[j] = exact_gelu(s);
        }
        for (int j = 0; j < h2_; ++j) {
            const float* w = w2_.data() + (size_t)j * h1_;
            double s = b2_[j];
            for (int k = 0; k < h1_; ++k) s += (double)w[k] * a1[k];
            a2[j] = exact_gelu(s);
        }
        for (int t = 0; t < kTargets; ++t) {
            double o[2];
            for (int half = 0; half < 2; ++half) {
                const int j = half * kTargets + t;
                const float* w = w3_.data() + (size_t)j * h2_;
                double s = b3_[j];
                for (int k = 0; k < h2_; ++k) s += (double)w[k] * a2[k];
                o[half] = s;
            }
            mean[t] = o[0] * ys_[t] + ym_[t];
            sd[t] = std::exp(0.5 * std::clamp(o[1], -6.0, 4.0)) * ys_[t];
        }
        return true;
    }

private:
    int in_dim_ = 0, h1_ = 0, h2_ = 0;
    std::vector<float> xm_, xs_, w1_, b1_, w2_, b2_, w3_, b3_, ym_, ys_;
};

// --------------------------------------------------------------------------------------------------------------------
// Engine
// --------------------------------------------------------------------------------------------------------------------

// The fused DINOv3 intent model (src/export_intent_engine.py): inputs pixel_values, pool_weights; outputs mu,
// sigma, pooled, pooled_tone. Preprocessing reproduces intent_model.predict / dino_geometry.
class DinoV3Engine {
public:
    DinoV3Engine();
    ~DinoV3Engine();

    DinoV3Engine(const DinoV3Engine&) = delete;
    DinoV3Engine& operator=(const DinoV3Engine&) = delete;
    DinoV3Engine(DinoV3Engine&&) noexcept;
    DinoV3Engine& operator=(DinoV3Engine&&) noexcept;

    // model_spec: a model directory, an .onnx file or an .engine file. A directory holds either the DINOv3 backbone
    // alone (backbone.onnx, backbone.engine: models/dinov3-small/export_dinov3_backbone.py) or a fused intent model
    // (model.onnx, model.engine); the backbone wins when both are present. A backbone needs an adapter
    // (set_adapter) before infer().
    // backend: "cpu" (ONNX Runtime), "gpu" (TensorRT, USE_CUDA=1 builds) or "auto" ("gpu" when this build has
    // a CUDA device and the model has a TensorRT engine, as intent_model.resolve_intent_backend).
    // num_threads: ONNX Runtime's threads for the backbone on the CPU; 0 takes the program's thread count (see
    // make_ort_runner).
    // max_size, patch_size: patch_size at least 1, max_size at least patch_size, and the largest preview's values
    // ((max_size / patch_size + 1) * patch_size pixels a side, times 3) within an int: max_size at most 26751 for a
    // patch_size of 16. A load drops what an earlier one set: the adapter and the neutral head.
    // Throws std::runtime_error (std::invalid_argument for a bad backend or sizes) when the model cannot be loaded.
    void load(const std::string& model_spec, const std::string& backend = "auto", int num_threads = 0,
              int max_size = 512, int patch_size = 16);

    // The backbone built into neg_process, held in memory: its TensorRT engine (USE_CUDA=1 builds) and its .onnx,
    // each null when not built in. backend picks between them as load() does: "auto" runs the engine when this build
    // has a CUDA device and falls back to the .onnx when the engine is missing or fails to load. Throws as load()
    // does.
    void load_memory(const void* engine_data, size_t engine_size, const void* onnx_data, size_t onnx_size,
                     const std::string& backend = "auto", int num_threads = 0, int max_size = 512,
                     int patch_size = 16);

    // Adapter for a backbone model, after load(): its adapter.onnx and its TensorRT adapter.engine (make
    // dinov3_engine), held in memory (embedded in neg_process) or read from files. The adapter runs where the
    // backbone runs: a TensorRT backbone needs the engine (there is no CPU adapter behind a GPU backbone), an ONNX
    // Runtime backbone uses the .onnx. name is reported by adapter_name(). Ignored by fused models, which carry their
    // own adapter. num_threads: ONNX Runtime's threads for the adapter on the CPU; 0 takes the program's thread
    // count (see make_ort_runner). Throws std::runtime_error when the adapter the backbone needs is missing or cannot
    // be loaded.
    void set_adapter(const void* onnx_data, size_t onnx_size, const std::string& name, int num_threads = 1,
                     const void* engine_data = nullptr, size_t engine_size = 0);
    // path: adapter.onnx or adapter.engine (or a directory holding them); the other file is taken from beside it. Only
    // the file the backbone's runtime takes is read: adapter.engine behind a TensorRT backbone (refused when it is
    // older than the adapter.onnx beside it, whose earlier weights it holds), adapter.onnx behind ONNX Runtime, neither
    // for a fused model. adapter_path() names the file read.
    void load_adapter_file(const std::string& path, int num_threads = 1);
    // The adapter file load_adapter_file read (empty after set_adapter, or for a fused model).
    const std::string& adapter_path() const { return adapter_path_; }
    // The adapter runs on the GPU (TensorRT).
    bool adapter_on_device() const { return adapter_ && adapter_->on_device(); }
    // Device-to-host copies of the backbone's tokens so far (0 when the backbone and adapter both run on the GPU).
    int token_downloads() const { return runner_ ? runner_->token_downloads() : 0; }
    const std::string& adapter_name() const { return adapter_name_; }
    // The loaded model is a backbone and needs an adapter.
    bool needs_adapter() const;

    // The neutral-patch head (NeutralHead) from a neutral_head.bin held in memory: with it, infer() also gives
    // DinoV3InferenceResult::neutral_p from the backbone's tokens. When the backbone and adapter run on the GPU the
    // tokens are copied to the host once per inference for it. Null data clears it. Throws std::runtime_error on
    // a bad file or a fused model (no tokens to read).
    void set_neutral_head(const void* data, size_t size);
    bool has_neutral_head() const { return neutral_head_ != nullptr; }
    // With it on, infer() also gives DinoV3InferenceResult::token_sd, the spread of the backbone's tokens over the
    // patches: with the pooled features, the input of the tone heads (ToneHead). Like the neutral head
    // it needs the tokens on the host (one copy per inference when the backbone and adapter run on the GPU).
    void set_token_spread(bool on) { token_spread_ = on; }

    bool is_loaded() const;
    const char* backend_name() const;
    const char* backend_used() const;
    bool has_pooled() const;
    bool has_pooled_tone() const;
    bool has_pool_weights() const;
    size_t memory_bytes() const;
    int max_size() const { return max_size_; }
    int patch_size() const { return patch_size_; }

    // Run inference on 8-bit sRGB preview image, its rows stride bytes apart (at least width * 3). Not thread-safe (a
    // TensorRT execution context runs one inference at a time): callers that share one engine between threads
    // serialize their calls, as negicc-station's binding does.
    DinoV3InferenceResult infer(const uint8_t* srgb_data, int width, int height, int stride);

private:
    std::unique_ptr<DinoV3Runner> runner_;
    std::unique_ptr<DinoV3AdapterRunner> adapter_;
    std::unique_ptr<NeutralHead> neutral_head_;
    bool token_spread_ = false;
    std::vector<uint8_t> adapter_file_;
    std::string adapter_name_;
    std::string adapter_path_;
    int max_size_ = 512;
    int patch_size_ = 16;
};

// ====================================================================================================================
// Auto solver
// Takes the frame (an RgbView in camera RGB), its profile and matrix, and the engine; gives the (E, g, b) the intent
// field asks for, with the search's log (DinoSolveResult).
// ====================================================================================================================

// Flags for pin_mask (manual argument overrides)
constexpr unsigned PIN_E = 1 << 0;
constexpr unsigned PIN_G = 1 << 1;
constexpr unsigned PIN_B = 1 << 2;

struct IntentGains {
  float e = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
};

struct IntentLoss {
  float L = 0.0f;
  float a = 0.0f;
  float b = 0.0f;
  float clip = 0.0f;  // in percent [0, 100]
};

// The input curves of a profile with the gains g and b in them, after the knee (the print, print_emulation.cc
// PrintContinuum): three tables of trc_size entries each, as ParsedFilmProfile::in_trc holds them.
using GainCurves = std::function<void(float g, float b, std::vector<float> trc[3])>;

struct SolverConfig {
  const ProfileData* prof = nullptr;
  // When set, a candidate's g and b go into prof's input curves, after the knee, and its matrix carries E alone, as
  // the print's conversion renders it; intent_cells and intent_loss (the search's optimiser) read it.
  const GainCurves* gain_curves = nullptr;
  float knee = 0.95f;
  // Highlight shoulder x_hi (negicc_core.h set_shoulder); 1.0 is off.
  float shoulder = 1.0f;
  float enc_e_min = 0.0f;
  float enc_e_max = 0.0f;
  // The values pinned parameters (pin_mask) take in optimize_target_for_intent when it is given no start; with a
  // start they stay at the start's.
  float pin_e = 1.0f, pin_g = 1.0f, pin_b = 1.0f;
};

struct DinoIterationLog {
  char tag = ' ';  // 'b' for bootstrap, '0', '1', etc.
  int step = 0;
  float e = 1.0f;  // normalized E (relative to E_c)
  float g = 1.0f;
  float b = 1.0f;
  float residual = -1.0f;  // -1 if unmeasured (bootstrap)
  float loss_L = 0.0f;
  float loss_a = 0.0f;
  float loss_b = 0.0f;
  float clip_pct = 0.0f;
  float penalty_clip = 0.0f;
  float penalty_anchor = 0.0f;
  float objective = 0.0f;
  float dp = 0.0f;
  float rel = 0.0f;
};

struct DinoSolveResult {
  std::string status = "CONVERGED";  // "CONVERGED", "DIVERGED", "MAX_ITERS", "PINNED"
  int best_iteration = -1;
  float best_residual = 0.0f;
  IntentGains solved_relative;  // relative to user baseline (E_c * e_best, g_best, b_best)
  IntentGains solved_absolute;  // E_user * E_c * e_best, g_user * g_best, b_user * b_best
  float e_center = 1.0f;
  bool anchor_clamped = false;  // E_c bisection did not straddle median L* 50
  // The default e_bounds, as ratios to e_center: 2^-1.5 and 2^1.5, 1.5 stops either way.
  float bounds_lo = 0.35355339f;
  float bounds_hi = 2.82842712f;
  std::vector<DinoIterationLog> iterations;
  std::vector<float> best_mu;
  std::vector<float> best_sigma;
  std::vector<float> tone_mass;
  int grid_w = 0;
  int grid_h = 0;
  int target_w = 0;
  int target_h = 0;
  int total_inferences = 0;
  double timing_views_ms = 0.0;
  double timing_render_ms = 0.0;
  double timing_infer_ms = 0.0;
  double timing_residual_ms = 0.0;
  double timing_search_ms = 0.0;
  double timing_total_ms = 0.0;
};

// The L*a*b* intent_loss scores for one candidate: each pixel of the camera-RGB grid rendered at cand as the loss
// renders it (displayed value, after the shoulder), n_px x 3 values into lab_out.
void intent_cells(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                  const IntentGains& cand, float* lab_out);

// The pipeline configuration and the matrix (written to m) cfg renders candidate cand with: the gains in the matrix,
// or, with cfg.gain_curves, in prof's input curves (trc holds the tables the configuration points into).
negicc::PipelineConfig candidate_config(const SolverConfig& cfg, const float* merged, const IntentGains& cand,
                                        float m[9], std::vector<float> trc[3]);

// Batched loss calculation across candidate matrices
void intent_loss(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16, int n_px,
                 const IntentGains* cand, int n_cand, const float* mu, const float* sigma, IntentLoss* out);

// The shoulder x_hi (the colour pipeline part of negicc_core.h) of a max-channel histogram of the pre-shoulder render
// (max_hist_bin layout): the x_hi that maps its 99.5% point to sRGB kShoulderAnchorCode (shoulder_x_hi_anchored, the
// roll-off); q995_out gets the point.
float shoulder_x_hi_from_hist(const uint64_t* hist, float* q995_out = nullptr);

// Coordinate descent parameter search against a frozen intent field (mu, sigma). Without e_bounds, E moves within
// [0.40, 1.80] from no start, or within [0.20, 3.0] around a start (one outside that range is moved to its nearer end).
float optimize_target_for_intent(const SolverConfig& cfg, const float* merged, const uint16_t* grid_rgb16,
                                 int gw, int gh, const float* mu, const float* sigma,
                                 const IntentGains* initial, const float* e_bounds /* lo, hi, center or null */,
                                 unsigned pin_mask, IntentGains& out);

// The preview's L*a*b* per intent-grid cell, made as the intent adapter's training targets are made
// (negicc-station training/extract.py, patch_lab_targets): the encoded preview (w x h, 3 bytes per pixel) resampled in
// linear light by area average to gw*patch x gh*patch pixels, CIELAB per pixel, averaged over each patch. gw*gh*3
// values, row-major, into lab_out.
void trained_cells_lab(const uint8_t* preview_u8, int w, int h, int gw, int gh, int patch, float* lab_out);

// Mean squared z-score of the preview's trained cells (trained_cells_lab) against the intent field predicted from it.
float self_consistency_residual(const uint8_t* preview_u8, int w, int h, int gw, int gh, int patch,
                                const float* mu, const float* sigma);

// The same residual from trained cells already computed (trained_cells_lab's gw*gh*3 values, n = gw*gh):
// self_consistency_residual is trained_cells_lab followed by this.
float residual_from_cells(const float* lab, int n, const float* mu, const float* sigma);

// Solves optimal (E, g, b) parameters against active DINOv3 intent field
bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const uint16_t (*image)[4], int w, int h,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out);
bool solve_dinov3_intent(DinoV3Engine& eng, const SolverConfig& cfg, const float* user_merged,
                         const negicc::RgbView& view,
                         IntentGains user_gains, unsigned pin_mask,
                         int max_iters, DinoSolveResult& out);

// ====================================================================================================================
// Parameter search: the conversion parameter search, its renderers, the curve fit and the tone stage's plan
// ====================================================================================================================

// The conversion parameter search, ported from negicc-station's (PerTargetConverter.convert with its defaults). It
// does not convert anything: on the half-size analysis decode of one frame it picks the calibration bracket of a
// profile bundle and the exposure E and gains g, b that neg_process then converts the full image with.
//
//   1. Enclosure: each bracket's calibrated range of E (the scene's green range inside the green range its patches
//      were calibrated over) and its usable range, the calibrated range widened by the highlight and shadow
//      tolerance; every bracket is searched over its usable range.
//   2. Bootstrap: meter each admissible bracket so the scene's log-average renders at L* 50, render every metered
//      bracket at its metered E clamped to its usable range (when none meters, the bracket whose range the scene fills
//      best, at its e_center) and average the intent fields DINOv3 predicts from those renders: the bootstrap field.
//      Its L* is the frame's one lightness target.
//   3. Screening: search (E, g, b) of every metered bracket (every admissible one when none meters) against the
//      bootstrap field.
//   4. Tournament: run the fixed-point loop on the best two screened brackets and keep the lower residual, whichever
//      way each loop ended. The loops hold L* to the frame's lightness target.
//   5. The clip fallback: move to the next bracket when the chosen one white-clips (ClipFallbackOutcome).
//   6. The highlight pullback: when a bright frame still has many highlights near white (sRGB 242 and 235), nudge E
//      (at most 0.15 stop) toward the chosen bracket's calibrated range, or, when that bracket cannot hold the scene,
//      run Steps 2-4 again with penalties on E outside the bracket's calibrated range in place of the pull toward the
//      centre, and take the result if it clips at most 0.5 percentage point more and moves to a later bracket or a
//      higher E.
//
// Every render carries the highlight roll-off (negicc_core.h shoulder_x_hi_anchored): the fixed-point loop
// measures its x_hi at each iteration's incoming parameters, and the search measures it once more at its final
// parameters, after the clip fallback and the pullback have moved E (and possibly the bracket) past the loop that
// froze it, so the written image's shoulder matches its highlights. The intent field the result carries
// (ParamSearchResult::intent_mu) is the one the chosen loop step was scored against, at intent_params: its L* is the
// frame's lightness target, its a* and b* the step's own prediction.
//
// The intent field is DINOv3's prediction, from an 8-bit sRGB render of the frame, of how each patch of the image
// should look: a mean mu and a spread sigma of CIELAB per patch of its grid. How far a render is from a field is
// the mean squared z-score over the grid,
//
//   z2(render, field) = 1 / (3 N) * sum_i sum_{k in L, a, b} ((lab_i,k - mu_i,k) / sigma_i,k)^2,
//
// with lab_i the CIELAB of the render over patch i and N the number of patches.

// The parameters the search solves. With the input matrix M the search runs on (rows R, G, B; input_matrix on
// TriLED, M_frame on white light, see ParamSearchResult::input_matrix) the conversion matrix is
//
//   diag(E, E g, E b) * M,
//
// so E is the global gain on the linear camera RGB after the crosstalk correction and the film base
// normalisation, ahead of the film profile, and g and b are the green and blue gains relative to red.
struct ConversionParams {
  float e = 1.0f, g = 1.0f, b = 1.0f;
  // The highlight shoulder x_hi (the colour pipeline part of negicc_core.h) the render, the intent field and the loss
  // use: per frame, frozen by the fixed-point loop; 1 is off.
  float shoulder = 1.0f;
};

// How the fixed-point loop of a bracket stopped.
enum class FixedPointStatus {
  // E, g, b moved by 0.02 or less in total in the last step; or, on TriLED, the residual by 1.5 % or less (in the
  // pullback's search also not rising); or, on white light, the residual, already above the best step's, did not fall
  // again (the best step is kept). The 1.5 % rule does not apply on white light.
  kConverged,
  kDiverged,   // the residual grew by more than 25 % and more than 1.5 in one step
  kMaxIters,   // neither within ParamSearchOptions::max_steps steps
};
const char* to_string(FixedPointStatus status);

// What the clip fallback did. It is tried when the chosen bracket white-clips more than 0.03 % of the inner crop, on
// the next bracket of the bundle (target_idx + 1), and taken when that one passes every check: it must at least halve
// the clipping, keep the median L* within -4.5 .. +0.5 of the first choice's (the midtones may darken a little to save
// the highlights, but the frame's rendering must not change), and keep the residual within 10 % (DINOv3 must still
// find the render as plausible).
enum class ClipFallbackOutcome {
  kNotNeeded,        // the chosen bracket does not clip
  kNotScreened,      // the next bracket was not screened
  kNotAdmissible,    // the next bracket's calibrated range is too narrow for the scene
  kClipsTooMuch,     // the next bracket clips more than half as much as the chosen one
  kShiftsLightness,  // the next bracket moves the median L* outside -4.5 .. +0.5
  kWorseResidual,    // the next bracket's residual is more than 10 % above the chosen one's
  kTaken,            // the next bracket replaces the chosen one
};
const char* to_string(ClipFallbackOutcome outcome);

// The passes over the frame the bundle search renders: the preview of the inner crop and the clip statistics (the
// roll-off's 99.5 % point, white-clip shares). The defaults are negicc::cpu_render and negicc::cpu_highlight_clip; a
// caller that renders the same pipeline elsewhere (the station's GPU) overrides them. search_conversion_params,
// search_print_continuous, run_white_light_fit (the curve fit, its roll-off and its tone plan) and
// plan_auto_tone_stage send every pass of theirs through it (reanchor_input_curves renders nothing); the passes are
// serial, called from the caller's thread one at a time.
//
// What stays and what changes between calls:
// - bind_frame is called before the passes of each measurement with the frame they crop, and again, with the same
//   frame, by each roll-off measurement (measure_rolloff_x_hi). The frame's pixels and the view given to bind_frame
//   stay valid until the search returns.
// - RenderArgs::in_view and HighlightClipArgs::in_view point to views made for the call (crops of the bound frame): a
//   renderer reads them during the call and keeps no pointer to them.
// - The bracket's 3D table (cfg's clut) and output tables are the bundle's and stay for the whole search. The three
//   input tables (cfg's in_trc) may change between any two calls, in place at the same address (the curve fit's
//   re-meter passes) or freed and replaced (when the fit composes its curves into the bracket): a renderer reads them
//   on every call, or compares their contents, and never keys a copy by their address.
// - The search owns every output buffer: out is sized before each call and must be filled before it returns. The clip
//   pass's black_out, when set, must be filled too: the shadow alternative reads it, and a renderer that left it
//   unfilled would never offer one.
//
// A renderer may refuse a frame it cannot read (a CFA view with its private flat-field mapping, say) by throwing from
// bind_frame. A failed pass throws: the exception leaves the search, the fit, the tone plan or the roll-off measurement
// to the caller (none of them catches it), so a caller can tell a render failure from every other one and run the
// search again on the CPU. The bundle is then as it was given: run_white_light_fit restores the bracket's own input
// tables when a pass throws after it has composed its curves into them. The result structures (ParamSearchResult,
// WhiteLightFitResult, ToneStagePlan) are then partly filled, and a caller must not read them.
//
// neg_process --check_renderer runs a conversion with a pass-through renderer (PassThroughRenderer) and counts any
// pass that runs outside it (PassAudit), on any thread, and fails when a pass ran outside it or none ran at all; with a
// CUDA device it also compares each pass with the GPU's (CompareRenderer). Its picture must equal a conversion whose
// passes run on the CPU, bit for bit, which nothing checks for it. Run it, and compare its picture with such a
// conversion's (cmp), on a TriLED scan and on a white-light scan with -P srgb (the curve fit and the tone plan), before
// a change to the search, the curve fit, the tone plan or this interface is merged.
struct PreviewRenderer {
  virtual ~PreviewRenderer() = default;
  virtual void bind_frame(const negicc::RgbView& frame) { (void)frame; }
  virtual void render(const negicc::RenderArgs& a) { negicc::cpu_render(a); }
  virtual void highlight_clip(const negicc::HighlightClipArgs& a) { negicc::cpu_highlight_clip(a); }
};

// The check aid (neg_process --check_renderer): the passes the bundle search's entry points run through a renderer
// and outside it. AuditScope marks those entry points; a CPU pass inside one that does not come through
// PassThroughRenderer is counted as outside (negicc::cpu_pass_hook).
struct PassAudit {
  long binds = 0, renders = 0, clips = 0, outside = 0;
};
PassAudit& pass_audit();
void audit_cpu_pass(int kind);   // the hook neg_process sets in negicc::cpu_pass_hook
struct PassThroughRenderer : PreviewRenderer {
  void bind_frame(const negicc::RgbView& frame) override;
  void render(const negicc::RenderArgs& a) override;
  void highlight_clip(const negicc::HighlightClipArgs& a) override;
};

// The passes on a CUDA device (negicc_core.h gpu_render and gpu_highlight_clip): neg_process's on a USE_CUDA=1 build
// with a device whose kernels it was built for (gpu_device_available), unless --dino_backend cpu. The constructor
// throws without one.
// - The frame: bind_frame copies the bound frame to the device, each pixel as the view samples it (a CFA view's flat
//   field included), unless device_frame gives the address at which the GPU already reads it (a caller whose frame is
//   already in pinned, mapped memory). Binding the same view again (the same pointers, size, layout and flat field)
//   keeps the copy, so a caller that rewrites a frame in place, or frees it and allocates another at the same address,
//   calls release_frame before binding it. A pass's view, the bound frame or an upright crop inside it with the same
//   layout, is read there at its offset; any other view throws.
// - release_frame: drops the bound frame and frees its device copy (neg_process calls it after the last pass of the
//   search, before the full-resolution conversion).
// - The tables: each table a pass reads (the cLUT, the input and output tables, the tone table) is compared with the
//   copy on the device on every pass and copied again when its contents differ, so a table changed in place (the
//   curve fit's) or another one at the same address is never read stale.
// - The output: a render is written to device memory and copied into the search's buffer.
// A failed pass throws negicc::GpuError, which leaves the search to its caller.
struct GpuRenderer : PreviewRenderer {
  GpuRenderer();
  ~GpuRenderer() override;
  GpuRenderer(const GpuRenderer&) = delete;
  GpuRenderer& operator=(const GpuRenderer&) = delete;
  void bind_frame(const negicc::RgbView& frame) override;
  void render(const negicc::RenderArgs& a) override;
  void highlight_clip(const negicc::HighlightClipArgs& a) override;
  void release_frame();
  // For a frame the GPU reads in place: the device address of frame.r_ptr (an interleaved RGB view without a flat
  // field, whose pitch and stride hold on the device), or null to have the frame copied.
  std::function<const uint16_t*(const negicc::RgbView& frame)> device_frame;
  // For a caller that needs the bound frame no more after one render (the full-size conversion): a 16-bit render of
  // the whole frame is written over the renderer's own copy of it, so the device holds the picture once, not twice.
  // The kernel reads each pixel before it writes it. The frame is released by that render. Such a frame, when it is
  // already 3 samples a pixel row after row with no flat field, is also uploaded as it is, with no copy of it made
  // on the host first. Set before bind_frame. Not an option of the program: only the full-size conversion sets it.
  bool render_over_frame = false;
  long binds = 0, renders = 0, clips = 0;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The CPU and the GPU passes compared (neg_process --check_renderer on a build with a device that runs its kernels):
// every pass runs on both from the same inputs and the search reads the CPU's, so the search is the CPU's and each GPU
// pass is measured against it. The CPU's side is PassThroughRenderer's, so the passes are counted as without a device.
// - A render: how many values differ and the largest difference; the largest and the mean change in L* per pixel,
//   each pixel read in the render's encoding (sRGB or linear sRGB; an 8-bit preview, what the network reads, a 16-bit
//   or a float render alike); and with
//   statistics, the largest difference of the white and black clip shares and of each histogram bin (as a share of
//   the crop's pixels), and of the six zone means of a* and b*.
// - A clip pass: the largest difference of its three exceedance shares and its white-clip share, its median L*, and
//   each bin of its max-channel histogram (as a share of the crop's pixels).
// - A value that is not finite on either side counts as differing and fails the comparison.
// within_tolerance(): every render's mean change in L* per pixel and every zone mean within kCompareMeanDL, every
// share (a clip share or a histogram bin's) within kCompareShare %, every median within one histogram bin
// (kCompareMedianL), and no value that is not finite.
struct CompareRenderer : PassThroughRenderer {
  static constexpr double kCompareMeanDL = 0.05;   // CIELAB: a render's mean change in L* per pixel; a zone mean
  static constexpr double kCompareShare = 0.01;    // %: each clip share, each histogram bin
  static constexpr double kCompareMedianL = 0.1;   // L*: the median, one bin of the clip pass's histogram
  void bind_frame(const negicc::RgbView& frame) override;
  void render(const negicc::RenderArgs& a) override;
  void highlight_clip(const negicc::HighlightClipArgs& a) override;
  bool within_tolerance() const;
  std::string summary() const;   // one line for the log
  struct Pass {
    bool render = false, stats = false;   // a render, with its statistics
    long long values = 0, differing = 0, nonfinite = 0;
    // max_share: the clip shares (a render's white and black clip); max_hist: a histogram bin, as a share of the
    // crop's pixels (%); max_zone: a render's zone means of a* and b*.
    double max_abs = 0.0, max_dl = 0.0, mean_dl = 0.0, max_share = 0.0, max_median_l = 0.0, max_hist = 0.0,
           max_zone = 0.0;
  };
  std::vector<Pass> passes;
  GpuRenderer gpu;
};

struct ParamSearchOptions {
  // The passes over the frame (PreviewRenderer); null: the CPU's, as neg_process runs.
  PreviewRenderer* renderer = nullptr;
  float crop_ratio = 0.90f;  // the frame's centred inner crop the statistics and the previews use
  int max_steps = 8;         // the most steps of each bracket's fixed-point loop (neg_process's --dino_iters)
  // A film base measured on this capture (crosstalk-corrected RGB) instead of the bundle's; null for the bundle's. Set
  // one of the two (neg_process sets film_base_rgb_d): the curve fit's roll-off reads film_base_rgb_d only.
  const float* film_base_rgb = nullptr;
  const double* film_base_rgb_d = nullptr;
  // True when the custom film base was sampled from the scan itself (e.g. rebate), so exposure_ratio is 1.0.
  bool film_base_from_scan = false;
  bool verbose = true;
  // The frame is not a detected film frame but the whole image (--roi auto fell back below its confidence, or a given
  // --roi is the whole image): it may hold rebate, bare light or holder. The curve fit and the knobs' re-anchoring of
  // the curves (reanchor_input_curves) then leave out the grid cells that are holder-dark, and (when
  // film_base_is_rebate is true) those at the film base's level (fallback_picture_cells). When the film base
  // came from the scan's statistics (the 99.5th percentile of the film itself), its level is the thinnest picture
  // highlights rather than unexposed rebate, so only holder-dark cells are left out.
  bool fallback_crop = false;
  bool film_base_is_rebate = false;
  // Whether to look for the shadow alternative (ParamSearchResult::shadow) after the highlight pullback. The caller
  // sets it where an alternative could be kept: neg_process with an sRGB export, the tone stage on, no density output
  // and no print (the choice is made by the tone stage's clip check). Not set: the step does not run.
  bool shadow_alternative = false;
  // Pins (neg_process's --target, -E, --gain_g and --gain_b on a bundle): the search and the curve fit run
  // as without them, with the bracket restricted to pin_target (when >= 0) and E, g or b (pin_mask: PIN_E, PIN_G and
  // PIN_B of the DINOv3 part above) held at pin_e, pin_g and pin_b (E in ParamSearchResult::params' units). A pinned
  // bracket leaves out the clip fallback, which only changes the bracket; a pinned E leaves out the highlight
  // pullback, which only moves E or the bracket. The curve fit (run_white_light_fit) holds the pinned g and b at the
  // search's and does not re-meter a pinned E.
  int pin_target = -1;
  unsigned pin_mask = 0;
  float pin_e = 1.0f, pin_g = 1.0f, pin_b = 1.0f;
  // neg_process's --probe_intent (a measurement aid): when probe_path is set, the search writes what the intent network
  // predicts from chosen renders to that JSON file and stops after its bootstrap, before screening
  // (ParamSearchResult::probed). Rendered: every metered bracket at its metered E with the bootstrap's gains (and at
  // the E its window clamps it to, where that differs), then every probe_params set (target index, E in the search's
  // units, g, b) at each exposure offset of probe_stops (stops), with the roll-off measured at each as the fixed-point
  // loop measures it. Each render's cells are recorded two ways: render_cells (the trained cells, as the residual reads
  // them and as the training targets are made) and optimiser_cells (what intent_loss scores at those parameters, before
  // the per-bracket correction). Each render's predicted is the network's own prediction from it, the mean of its mu
  // cells, with nothing held; the file's target_L is the mean L* of the frame's lightness target (the bootstrap field's
  // L*, which the search holds).
  std::string probe_path;
  std::vector<std::array<float, 4>> probe_params;
  std::vector<float> probe_stops = {0.0f};
};

// Step 3, one row per metered bracket: one coordinate descent of (E, g, b) against the bootstrap intent field (the
// mean of the fields predicted from the metered brackets' renders), each bracket aimed at that field minus its own
// correction (update_correction). No inference runs during the screening, so the rows compare the brackets cheaply;
// the best two go on to the tournament.
struct ScreeningResult {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  ConversionParams params;  // the best (E, g, b), E in the search's units (reference on TriLED, this frame's on white
                            // light; see ParamSearchResult::input_matrix)
  // The intent loss at params (intent_loss): z2 of the optimiser's cells, the patch-averaged grid rendered at params,
  // against the bootstrap field minus the bracket's correction, plus the search's penalties on clipping and on E far
  // from the bracket's centre exposure.
  float intent_loss = 0;
  float e_lo = 0, e_hi = 0;  // the bounds E was searched within (the search's units)
  float e_metered = 0;       // E that renders the scene's log-average at L* 50; e_center when no bracket metered
};

// Step 4, one row for each of the best two screened brackets: its fixed-point loop. Each step renders the frame at
// the current (E, g, b), runs DINOv3 on the render (one inference), replaces the predicted L* with the frame's
// lightness target, measures the residual and searches the next (E, g, b) against that field. The row keeps the step
// with the lowest residual.
struct FixedPointResult {
  int target_idx = 0;       // the bracket, an index into ProfileBundle::targets
  ConversionParams params;  // (E, g, b) of the best step, E in the search's units
  // The residual of that step: z2 of the render against the field it was scored against (L* the frame's lightness
  // target, a* and b* predicted from the same render). Low when the render already looks the way DINOv3 expects.
  float residual = 0;
  FixedPointStatus status = FixedPointStatus::kMaxIters;
  int steps = 0;  // render -> DINOv3 -> search steps run, one inference each
  // The frozen shoulder of the best step (params.shoulder) and how it was found.
  float shoulder_q995 = 0.0f;   // the measured 99.5% point before clamping (auto only)
  float shoulder_knee = 1.0f;   // the linear knee x_k (1 when off)
  bool shoulder_guard = false;  // the L* part of the intent loss is more than 2% worse with the shoulder than without
  // The intent field of the best step: L* (and its sigma) the frame's lightness target, a* and b* (and their sigma) as
  // DINOv3 predicted them from that step's preview. Its residual was measured against this field, and the next
  // parameters were searched against it. grid_h x grid_w cells, row-major, [L*, a*, b*] per cell.
  std::vector<float> mu, sigma;
  // The neutral head's p per cell from the same preview (DinoV3InferenceResult::neutral_p); empty without one.
  std::vector<float> neutral_p;
  // The tone heads' input from the same preview: the backbone's pooled features and the tokens' spread (1920 values,
  // ToneHead); empty unless the engine gives the spread (DinoV3Engine::set_token_spread).
  std::vector<float> tone_features;
  int grid_w = 0, grid_h = 0;
};

struct ParamSearchResult;

// One screened bracket in the shadow alternative's clip pass: its shares of the inner crop (percent) with a channel
// white-clipped and with a channel black-clipped, at its screened parameters.
struct ShadowBracketRow {
  int target_idx = -1;
  float white = 0.0f, black = 0.0f;
};

// The search's shadow alternative (search_conversion_params, after the highlight pullback; DESIGN.md 4.2): the
// screened bracket with the least black clipping among those that clip white no more than the result plus
// kToneClipAllowance, with its fixed point. The conversion takes it in place of the result only when the tone
// stage's clip check of both finds it no higher at white and at black, before and after the stage, and lower at one.
struct ShadowAlternative {
  bool ran = false;                    // the step ran (not with a pinned bracket or E)
  std::vector<ShadowBracketRow> rows;  // the screened brackets, in screening order
  float result_white = 0.0f, result_black = 0.0f;  // the result in its bracket's pass (percent)
  int picked = -1;                     // the bracket the step chose (-1: none within the white allowance)
  const char* none = nullptr;          // why there is no alternative, or null
  bool reused_run = false;             // the alternative's fixed point is the tournament's run of that bracket
  bool chosen_within_allowance = false;  // the chosen bracket's row is within the white allowance
  int inferences = 0;                  // the step's own DINOv3 inferences (its fixed point's)
  double ms = 0.0;                     // the step's time
  std::shared_ptr<ParamSearchResult> result;  // the alternative (null: none)
};

struct ParamSearchResult {
  int target_idx = -1;      // the chosen bracket, an index into ProfileBundle::targets
  std::string target_name;  // its name in the bundle ("Target 3")
  ConversionParams params;  // the chosen (E, g, b), E in this frame's units: diag(E, E g, E b) * M_frame
  float residual = 0.0f;    // the chosen bracket's FixedPointResult::residual
  // The chosen bracket's shoulder (params.shoulder is x_hi): measured q995, linear knee, and the guard.
  float shoulder_q995 = 0.0f;
  float shoulder_knee = 1.0f;
  bool shoulder_guard = false;
  FixedPointStatus status = FixedPointStatus::kMaxIters;  // and its FixedPointResult::status
  // The chosen step's intent field (FixedPointResult::mu, sigma, grid_w, grid_h), written to the sidecar's
  // intent_field, and that step's parameters (E in the search's units).
  std::vector<float> intent_mu, intent_sigma;
  std::vector<float> intent_neutral_p;  // FixedPointResult::neutral_p of that step (empty without a neutral head)
  std::vector<float> intent_tone_features;  // FixedPointResult::tone_features of that step (may be empty)
  int intent_grid_w = 0, intent_grid_h = 0;
  ConversionParams intent_params;

  // The bracket's input matrix and the frame's exposure. With X the bundle's crosstalk matrix, f the film base
  // RGB, n the bundle's normalisation target and rho the exposure of the film base capture over this frame's
  // (shutter x ISO / 100):
  //
  //   M_frame[r][c] = X[r][c] * n / f[r] * rho            (the merged input matrix of this frame)
  //   e_norm        = M_frame[1][1] / M_ref[1][1]         (M_ref: the same for a 1/8 s, ISO 100 scan)
  //   input_matrix  = M_frame / e_norm                    (reference units)
  //   row_scale[r]  = n / f[r] * rho / e_norm,  so  input_matrix[r][c] = row_scale[r] * X[r][c]
  //
  // On TriLED the search runs on input_matrix: its E is in reference units, so the same E means the same rendering
  // whatever the scan's exposure, and params.e is E / e_norm. On white light (no crosstalk) it runs on M_frame and
  // scales the scene's green range by e_norm instead (bracket_enclosure), so its E is params.e. row_scale lets the
  // result be written back as the bundle's crosstalk with -E, --gain_g and --gain_b (neg_process's cli_equivalent).
  float input_matrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  float e_norm = 1.0f;
  float row_scale[3] = {1.0f, 1.0f, 1.0f};

  // The scene's green range over the inner crop (its 0.5 % and 99.5 % points, 16-bit) and its span in stops.
  float g_lo = 0, g_hi = 0, scene_ev = 0;

  ClipFallbackOutcome clip_fallback = ClipFallbackOutcome::kNotNeeded;
  float clip_fallback_white_clip = 0.0f;  // the fraction of the inner crop the first choice white-clips
  bool highlight_pullback = false;        // the highlight pullback moved E or the bracket
  // The chosen bracket's calibrated range of E (its enclosure's e_min and e_max) in params.e's units; 0 when the
  // bracket has none (infeasible). The curve fit bounds E with it.
  float e_range_lo = 0.0f, e_range_hi = 0.0f;

  std::vector<ScreeningResult> screening;
  std::vector<FixedPointResult> tournament;
  // DINOv3 inferences of the whole search: the bootstraps and every fixed-point step (the clip fallback's and the
  // pullback's included).
  int inferences = 0;
  bool probed = false;  // --probe_intent: the search wrote its probe file and stopped (ParamSearchOptions::probe_path)
  // Time spent rendering previews, in DINOv3, in the (E, g, b) searches, on residuals, and in the whole search.
  std::chrono::steady_clock::duration render_time{}, infer_time{}, search_time{}, residual_time{}, total_time{};

  // The frame statistics the search computed, and what they were computed from: the frame view, the crop ratio, and
  // the network's size and patch. plan_auto_tone_stage takes them when it is called with the same ones, and computes
  // its own otherwise. The view is only compared (its pointers and geometry), never read, so the frame's pixels must
  // not change between the search and the plan.
  std::shared_ptr<const void> frame_stats;  // a FrameStats, a type of multi_target_optimizer.cc
  negicc::RgbView frame_stats_view;
  float frame_stats_crop_ratio = 0.0f;
  int frame_stats_max_size = 0, frame_stats_patch = 0;

  // The shadow alternative this result's search found (ShadowAlternative): the conversion takes it in place of this
  // result only when the tone stage's clip check says so (pipeline.cc).
  ShadowAlternative shadow;
};

// The highlight roll-off's x_hi (negicc_core.h shoulder_x_hi_anchored) of frame, a whole frame at any size and
// orientation, converted through prof (first overload: a bracket's profile, or a combined profile built from it) or
// bracket target_idx (second) at the search's E, g and b (ParamSearchResult::params, the station's): the 99.5% point of
// the pre-shoulder max channel over its inner crop, mapped to sRGB 242. The search's last step measures the same on its
// half-size frame; neg_process measures every bundle conversion's there too, so given values reproduce it. q995 gets
// the point. film_base_rgb: a film base read from this scan (the film base detector part of negicc_core.h), as
// ParamSearchOptions::film_base_rgb_d with film_base_from_scan; null for the bundle's. Throws std::invalid_argument
// when target_idx is not one of the bundle's brackets, or, without a base from this scan, when the scan's shutter and
// ISO are not positive finite numbers (as the search does).
float measure_rolloff_x_hi(const ProfileBundle& bundle, const negicc::ProfileData& prof, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995 = nullptr, const double* film_base_rgb = nullptr,
                           PreviewRenderer* renderer = nullptr);
float measure_rolloff_x_hi(const ProfileBundle& bundle, int target_idx, const negicc::RgbView& frame,
                           double scan_shutter_s, double scan_iso, float e, float g, float b, float crop_ratio,
                           float* q995 = nullptr, const double* film_base_rgb = nullptr,
                           PreviewRenderer* renderer = nullptr);

struct KnobSolveResult {
  float e = 1.0f, g = 1.0f, b = 1.0f;
  bool limited = false;                     // the asked Lab was out of reach: e, g, b reach as far toward it as can be
  float grey_lab[3] = {50.0f, 0.0f, 0.0f};  // the Lab the grey renders at with e, g, b
  // When solve_grey_knobs returns false because no grey was found (GreyPoint's solve for a neutral L* 50 on the grey's
  // profile did not come within 0.25 of it): the dE76 that solve ended at. 0 otherwise.
  float grey_miss = 0.0f;
};

// Solves the conversion gains (E, g, b) that move the grey that base=(base_e, base_g, base_b) renders as neutral L* 50
// through prof by brightness_stops (in stops of luminance), tint_a (CIELAB a*), and warmth_b (CIELAB b*), matching
// negicc-station's GreyPoint and ConversionSession._solve_knobs. grey_prof, when given, is the profile whose neutral
// L* 50 defines the grey (the bracket's profile before re-anchored curves changed it), rendered through prof. False
// when base is not positive and finite, when a knob is not finite, or when no grey is found (out.grey_miss).
bool solve_grey_knobs(const ProfileData& prof, float base_e, float base_g, float base_b, float brightness_stops,
                      float tint_a, float warmth_b, KnobSolveResult& out, const ProfileData* grey_prof = nullptr);
struct PrintKnobResult {
  double a = 0.0;                           // the print's mid grey A after the knobs
  float g = 1.0f, b = 1.0f;                 // its gains (in the print's curves)
  bool limited = false;
  float grey_lab[3] = {50.0f, 0.0f, 0.0f};  // the grey's Lab through the print at (a, g, b)
};
// The knobs on a print, solved on the print's own profile (multi_target_optimizer.cc).
// print_at(a, g, b, prof) builds the print profile at mid grey a with gains g, b in its curves; A stays within
// [a_lo, a_hi], the range the print's curves are defined on (PrintContinuum::a_lo, a_hi).
bool solve_print_knobs(const std::function<void(double a, double g, double b, ParsedFilmProfile& prof)>& print_at,
                       double a0, float g0, float b0, double a_lo, double a_hi, float brightness_stops, float tint_a,
                       float warmth_b, PrintKnobResult& out);
// The Lab through prof at gains egb of the grey that base renders as neutral L* 50 through grey_prof. False when base
// or egb is not positive and finite, or when no grey is found.
bool knob_grey_lab(const ProfileData& grey_prof, const ProfileData& prof, const float base[3], const float egb[3],
                   float lab_out[3]);
// How far, in stops of luminance, a grey rendered at L* grey_L lies from the L* brightness_stops asks for.
float knob_grey_miss_stops(float brightness_stops, float grey_L);

// Searches the conversion parameters of one frame. frame is the frame upright, already cut out of the half-size
// decode and oriented as the output (DINOv3 judges an image the way it will be seen); scan_shutter_s and scan_iso
// are the capture's exposure, positive finite numbers unless the film base is read from this scan
// (opts.film_base_from_scan): otherwise it throws std::invalid_argument.
bool search_conversion_params(const ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                              double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                              ParamSearchResult& out);

// A stop of exposure in density, log10 2: the print search's unit of A.
inline const double kStopDensity = std::log10(2.0);

// The print's search (--print with Auto on a bundle with grey_balance_curves, either light): the print searched with
// its mid grey A as a continuous variable in place of the bracket (print_emulation.cc PrintContinuum), E held at 1 in
// the search's reference units (the negative's own density above base, as the brackets were measured): A0 metered once
// (the A at which the frame's log-average cell prints at L* 50, as the scene's search meters a bracket); the lightness
// target from the mean field of the renders at A0 - boot_h, A0, A0 + boot_h (A0 alone when boot_h is 0), held; g and b
// screened at A0 on the optimiser's wide grids; then the fixed point over (A, g, b): each step renders the print at its
// A, measures the optimiser's correction, predicts a* b* (L* is the lightness target) and solves A (bracketed by steps
// of 0.1 APD downhill from the current A, then golden section to 0.001, each A with g and b by the optimiser, E
// pinned). The scene loop's thresholds stop it (A counted in stops, dA / log10 2); the lowest-residual step is kept. No
// tournament, no bracket fallback, no highlight pullback. ParamSearchResult::target_idx is the bracket whose A_mid is
// nearest A.
// The print's film base held no darker than the picture's thinnest film, for a base that is not a capture (the
// scan's rebate or statistics, or the bundle's): a base darker than the thinnest film would push thin film past the
// knee and cut the print's shadows. The factor β = max(1, the largest over the three channels of the inner crop's
// kBasePercentile point of (X v)_c / fb_c) (DESIGN.md 5.8), v a pixel of the frame, X the bundle's crosstalk matrix
// and fb the base (f in DESIGN.md) in the conversion's film base space (fb_c reads n0 at the print's E). The base the
// print takes is fb β, its colour kept; on a base already raised β is 1. Every 4th pixel each way.
float print_floor_factor(const ProfileBundle& bundle, const negicc::RgbView& frame, float crop_ratio,
                         const double fb[3]);
// The base a print takes when the film base is not a capture: found (the scan's rebate or statistics), or null for none
// found, when the bundle's base is taken, scaled once by the exposures (the scan's over its base capture's) as the
// conversion takes it. Writes the base before the floor (base_in) and after it (base_out, base_in times β) at the
// scan's exposure, in the conversion's film base space, and returns β.
float raise_base_to_print_floor(const ProfileBundle& bundle, const negicc::RgbView& frame, float crop_ratio,
                                const double* found, double scan_shutter_s, double scan_iso, double base_in[3],
                                double base_out[3]);

struct ContinuousPrint {
  // The print's input tables at A with the gains (r, g, b) in them, after the knee (null: none), as the conversion
  // renders the print (print_emulation.cc PrintContinuum::trc_at).
  std::function<void(double a, const double gains[3], std::vector<float> trc[3])> trc_at;
  double a_lo = 0.0, a_hi = 0.0;     // the range A is searched in
  std::vector<double> a_mid;         // per bundle.targets
  double boot_h = kStopDensity / 2;  // half a stop of printing exposure (the bootstrap's spread)
};
struct ContinuousPrintResult {
  double a = 0.0, a0 = 0.0;
  int steps = 0;
  bool converged = false;
  float above_knee = 0.0f;           // the share of the inner crop above the knee at the bootstrap (the search's E,
                                     // g = b = 1)
  // At the final settings, per channel (the gains are in the print's curves, after the knee, so this is at E alone):
  // the share of the inner crop past the knee (input above 0.95, 0.054 density thinner than the base) and past the cut
  // (above 1.1227, where the knee's tanh reaches its cap of 0.9999: 0.126 density thinner), where the picture keeps no
  // detail.
  float past_knee[3] = {0.0f, 0.0f, 0.0f}, past_cut[3] = {0.0f, 0.0f, 0.0f};
  float target_mean_l = 0.0f;        // the mean L* of the lightness target it held
  float above_white[3] = {0, 0, 0};   // the share of the inner crop with the channel above 1.002 (linear) at the
                                      // result: cut
  float at_white[3] = {0, 0, 0};      // the share with the channel at 0.999 or more: the paper's white and above
  bool a0_at_range_end = false;      // A0's bisection ended at an end of the range
  float last_move[3] = {0, 0, 0};    // the last step's move: |dA| in stops, |dg|, |db|
  const char* stopped_by = "steps";   // "parameters" (the move within TOL_PARAM), "residual" (its change within
                                      // TOL_REL_ERR), "steps"
};
// One print material's search: print_base is a print profile of the bundle (its table and output curves; its input
// curves are rewritten at each A, with g and b in them for every render and optimiser pass, the matrix at E alone, as
// the conversion renders it), in its continuum; the search writes out and res. Each outlives the search.
struct PrintMaterialSearch {
  const ParsedFilmProfile* print_base = nullptr;
  const ContinuousPrint* in = nullptr;
  ParamSearchResult* out = nullptr;
  ContinuousPrintResult* res = nullptr;
};
// The print's search on one frame for each material in prints, in order (a picture print has one, a density output's
// --print one per material). The materials share the bundle's copy, the frame's crop, statistics and scale, and its
// shares at E; each has its own search.
bool search_print_continuous(const ProfileBundle& bundle, const std::vector<PrintMaterialSearch>& prints,
                             DinoV3Engine& engine, const negicc::RgbView& frame, double scan_shutter_s,
                             double scan_iso, const ParamSearchOptions& opts);
// The bracket that names a print at mid grey a: the one whose own A_mid is nearest a (the first of equals, in the
// bundle's order). The search names its result's this way, and a print the knobs moved is named by its moved A.
int nearest_bracket(const std::vector<double>& a_mid, double a);

// ---- Fitted G/B input curves (the input curves part of negicc_core.h) -------------------------------------------

// Composes the G and B input tables of prof with curves, in place (its own tables, then rebound), so every copy of
// the bracket's profile (the search's renders, the grey point, the export) is the combined profile. No curves: no
// change.
void apply_input_curves(ParsedFilmProfile& prof, const negicc::InputCurves& curves);
// The same from base's (un-curved) G and B input tables into prof's: curves applied again at another anchor replace
// the earlier composition instead of composing on top of it. No curves: prof gets base's tables back.
void apply_input_curves(ParsedFilmProfile& prof, const ParsedFilmProfile& base, const negicc::InputCurves& curves);

// curves (anchored at from = {E, g, b}) re-anchored at to = {E, g, b} on frame, for the knobs, which move E, g and b
// after the curves were fitted or recorded: ref, u_med and u_lo moved by what the frame's film base and grid move
// between anchor_curves at from and at to, so with to == from the result is curves unchanged, bit for bit. frame,
// scan_shutter_s, scan_iso and opts (its film base and crop) as run_white_light_fit took them; engine gives
// the grid's size only (it need not be loaded). Where the re-anchored curves leave the shape over the bracket's input
// table (curve_shape_ok at max_dev_stops), each channel's a and k are scaled down by the largest factor that keeps it
// (scale[0], scale[1]; 1: not needed). False (reported) on a bundle with a crosstalk matrix or a bad bracket.
bool reanchor_input_curves(const ProfileBundle& bundle, const DinoV3Engine& engine, int target_idx,
                           const negicc::RgbView& frame, double scan_shutter_s, double scan_iso,
                           const ParamSearchOptions& opts, const negicc::InputCurves& curves, const float from[3],
                           const float to[3], float max_dev_stops, negicc::InputCurves& out, float scale[2]);

// The plausible range of Auto's grid median L* (the lightness the curve fit keeps): outside it the intent failed (a
// render nearly black or blown out, as when the frame holds rebate or holder), and the fit does not chase it.
constexpr float kIntentMinL = 18.0f;
constexpr float kIntentMaxL = 85.0f;

// The tone stage's plan (the tone stage part of negicc_core.h): the tone heads' predictions (on in the DINOv3 part
// above the features of one inference on the final render's preview) and that preview; the conversion solves the curve
// between the preview's descriptors and the predictions (TonePlanCurve) and applies it to the final image.
struct ToneStagePlan {
  bool planned = false;           // a curve was solved (by the conversion) and the conversion applies it
  const char* skipped = nullptr;  // why not: "off", "encoding" (not -P srgb), "features" (no pooled features)
  // Whose DINOv3 features the heads read: "final" (the final state's preview, one inference); "search" (the intent
  // preview) only if that inference failed.
  const char* features = "search";
  double pred_mean[11] = {}, pred_sd[11] = {};  // the heads' predictions (ToneHead targets, in their order)
  // The preview (8-bit RGB at DINOv3's size) the curve is solved on, kept for the conversion, which also caps the
  // highlight move on the final image where it would clip (tone_plan_for_image), solving it again at a smaller share.
  std::vector<uint8_t> preview;
};

// What run_white_light_fit did.
struct WhiteLightFitResult {
  // The joint fit of the last pass: its curves (with ref and u_med), g, b and E, its data losses, evaluations, grid
  // renders for E and steps, whether it settled, and its never-greener guard.
  negicc::JointCurveFit fit;
  ConversionParams params_before; // the search's parameters (E in this frame's units, as ParamSearchResult::params)
  ConversionParams params;        // the result: E re-metered on the combined profile, the fit's g and b (frame units)
  float target_median_L = 0.0f;   // Auto's grid median L* (the re-meter's target), on the export's render at Auto's
  float median_L = 0.0f;          // the grid median L* at params on the combined profile, on the export's render
  // The export's render: its encoding and roll-off (only the sRGB encoding applies one; 1 otherwise).
  int encoding = negicc::ENCODE_SRGB;  // OutputEncoding
  float x_hi_auto = 1.0f;         // the export's roll-off at Auto's parameters on the bracket's profile
  float x_hi_fit = 1.0f;          // the roll-off the last fit pass ran with (the first runs with x_hi_auto)
  int fit_passes = 0;             // fits run: one, and again while the result's roll-off moved (at most three)
  // The roll-off at params on the combined profile (and its q995): measured as the export measures it (only for the
  // sRGB encoding: rolloff_measured), so the export reuses it with no knobs instead of measuring it again.
  float x_hi = 1.0f, x_hi_q995 = 0.0f;
  bool rolloff_measured = false;
  int remeter_passes = 0;         // re-meter and roll-off alternations, over every re-meter (the shape rounds re-meter)
  bool remeter_converged = false; // the roll-off of the last pass equals the one its re-meter ran with
  double loss_export = 0.0;       // the intent loss at params on the combined profile, on the export's render
  // G's and B's a and k scaled by these to keep the shape at the export's anchor (1: not needed; 0: dropped), and the
  // smaller of the two.
  float shape_scale_ch[2] = {1.0f, 1.0f};
  float shape_scale = 1.0f;
  // Why the fit fell back to Auto's result with no curves, or null when it did not: "cells" (fewer than 8 grid cells in
  // range, so no fit ran), "start" (Auto's start could not keep target_L within the E bounds, so no fit ran; the
  // re-meter still runs on Auto's values), "shape" (still past the shape after three rounds of scaling), "clip" (E
  // would leave [e_lo, e_hi] to keep Auto's median L*), or "intent_lightness" (Auto's grid median L* outside
  // [kIntentMinL, kIntentMaxL]: the intent failed, and E is re-metered so the grid's median L* is the nearer end of
  // that range instead of kept at Auto's).
  const char* fallback = nullptr;
  bool intent_lightness_failed = false;
  // On "intent_lightness": the L* E was re-metered to, whether the re-meter stopped at a bound of E before reaching
  // it, and whether E moved from Auto's.
  float fallback_target_L = 0.0f;
  bool fallback_clipped = false, fallback_e_moved = false;
  // Grid cells the fit left out on a fallback crop (ParamSearchOptions::fallback_crop), of the grid's cells.
  int cells_excluded = 0, cells_total = 0;
  // The bounds of E for the fit and the re-meter: the bracket's calibrated range widened to hold Auto's E.
  float e_lo = 0.0f, e_hi = 0.0f;
  // The search's intent field was predicted again at its final parameters (the pullback or the clip fallback moved
  // them after its preview), and the fit scored against that one: its a* and b*, with L* held to the frame's
  // lightness target.
  bool intent_repredicted = false;
  // The fit's options as it ran (the sidecar records them).
  negicc::CurveFitOptions options;
  // The loss's lightness and colour parts on the export's render: Auto's result on the bracket's profile, the fit's
  // result at its own E, and the result on the combined profile; and the time the fit, re-meter and checks took.
  negicc::LossParts parts_auto, parts_fit, parts_export;  // parts_fit: after the (last) fit, at fit.e
  negicc::NeutralStats neutral_auto, neutral_export;
  // The fit had scene-neutral anchors (the neutral head's p with some evidence); without them the anchors hold Auto's
  // colour and only the intent field moves the overall a*.
  bool neutral_anchors = false;
  // Whose p the guard (fit.guard) weighted by: "search" (the search's chosen preview) or "refit" (predicted again at
  // the search's final parameters, intent_repredicted).
  const char* guard_p_source = "search";
  // The tone stage's plan.
  ToneStagePlan tone;
  double time_ms = 0.0;
};

// The tone stage's plan after a search with no curve fit (TriLED bundles, and white light with --fit_curves off): the
// heads' predictions and the preview of the search's own result, res.params on the bracket's profile at the roll-off
// the conversion measures there. Nothing of the search's result changes. False (reported) on a bad bracket; a plan
// that could not be made says why in out.skipped.
bool plan_auto_tone_stage(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                          double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                          const ParamSearchResult& res, int encoding, const ToneHead& tone_head, ToneStagePlan& out);

// The white-light fit on plain data: what run_white_light_fit does after the frame and its intent field are known,
// for callers with their own frame and search (negicc-station). Every pointer is the caller's and outlives the call.
struct WhiteLightFitData {
  const negicc::ProfileData* prof = nullptr;  // the bracket's own profile (un-curved tables)
  const float* merged = nullptr;              // the 3x3 input matrix before the gains, E in the frame's units
  const uint16_t* grid = nullptr;             // n cells of 16-bit camera-linear RGB (the frame's statistics grid)
  int n = 0, grid_w = 0;                      // cells, and the grid's width (the cells' 4 x 4 blocks)
  // The film base the conversion uses, in the grid's units; without one the brightest grid cell per channel.
  bool has_film_base = false;
  float film_base[3] = {0.0f, 0.0f, 0.0f};
  // The frame is the whole image, not a detected frame (cells at the film base's level or holder-dark are left
  // out), and whether its base was read from the rebate.
  bool fallback_crop = false, film_base_is_rebate = false;
  // The intent field (n x 3 each) and the neutral head's p per cell (n values, or empty when the head did not run),
  // of a preview at (e, g, b) on the bracket's profile.
  std::vector<float> mu, sigma, neutral_p;
  float e = 1.0f, g = 1.0f, b = 1.0f;         // the search's result
  float e_range_lo = 0.0f, e_range_hi = 0.0f; // the bracket's calibrated range of E (0: unknown)
  int pin_mask = 0;                           // PIN_E | PIN_G | PIN_B the caller holds
  bool verbose = false;
  int encoding = negicc::ENCODE_SRGB;         // the export's (the roll-off applies to sRGB only)
};
// The export's roll-off x_hi (and its 99.5% point) at (e, g, b) on the bracket's tables composed with curves
// (combined holds them composed; curves inactive: the bracket's own), measured as the caller's conversion measures it.
using WhiteLightRolloff = std::function<float(const negicc::ProfileData& combined, const negicc::InputCurves& curves,
                                              float e, float g, float b, float* q995)>;
// Steps 1 and 2 of run_white_light_fit below and its record: out.fit.curves anchored at out.params, which the caller
// composes into its profile (apply_input_curves, or compose_input_table per channel). out.params_before, out.tone and
// out.time_ms are the caller's. False (reported) when the data do not fit together.
bool fit_white_light_curves(const WhiteLightFitData& data, const negicc::CurveFitOptions& fit_opts,
                            const WhiteLightRolloff& rolloff, WhiteLightFitResult& out);

// The station's ConversionSession.set_fit_curves on the bracket a finished search chose (res), for white-light
// bundles, on the same frame as the search (the WhiteLightFit steps in multi_target_optimizer.cc):
//   0. prepare: the search's intent field (predicted again at the search's final parameters when the pullback or the
//      clip fallback moved them after its preview), the cells the fit uses (a fallback crop leaves out cells at the
//      film base's level or holder-dark), Auto's render and grid median L*, E's bounds (the bracket's calibrated range
//      widened to hold Auto's E) and the scene-neutral anchors (JointCurveModel::set_scene_neutral);
//   1. the joint fit (negicc::fit_input_curves_joint): g, b, a_G and a_B in one Levenberg-Marquardt solve on the
//      anchors' balance and the intent field's overall a* (the evidence shrinkage inside the residuals), each
//      candidate at the E that keeps Auto's grid median L*, and the never-greener guard checked on the result; its
//      curves anchored (ref, u_med) on the film base (the one opts gives, else the brightest grid cell per channel) at
//      its own E, g, b;
//   2. E re-metered on the real combined tables around the fit's E, so the grid's median L* at the fitted g and b
//      equals Auto's (no DINOv3 pass), each channel's curve scaled down where its shape fails at the re-metered
//      anchor; when the re-metered E leaves the bounds, or the shape still fails, Auto's own result with no curves
//      (WhiteLightFitResult::fallback). Steps 1 and 2 repeat (up to three fits) while the re-metered roll-off differs
//      from the one the fit ran with;
//   3. the bracket's G and B input tables composed with the curves, anchored at the result, in place
//      (apply_input_curves on bundle.targets[res.target_idx]);
//   4. record: the export's parameters, losses and neutral statistics, and with tone_head the tone stage's plan
//      (ToneStagePlan: the heads' predictions and the final render's preview) for the conversion to solve and apply
//      to the final image.
// Every step renders the grid as the export renders it (encoding: the export's OutputEncoding, ENCODE_SRGB or
// ENCODE_SRGB_LINEAR, which is what the fit models), with the roll-off measured as the export measures it.
// False (reported) on a bundle with a crosstalk matrix, a search without an intent field, or a failed inference.
bool run_white_light_fit(ProfileBundle& bundle, DinoV3Engine& engine, const negicc::RgbView& frame,
                         double scan_shutter_s, double scan_iso, const ParamSearchOptions& opts,
                         const ParamSearchResult& res, const negicc::CurveFitOptions& fit_opts,
                         int encoding, const ToneHead* tone_head, WhiteLightFitResult& out);

// ####################################################################################################################
// neg_process's own declarations
// ####################################################################################################################
//
// The library's interface ends above. From here to the end are the declarations neg_process's program sources share
// (neg_process.cc, np_input.cc, print_emulation.cc, np_output.cc and pipeline.cc): the print, the film base, the
// options, the output and the pipeline. The library's sources and the companion application's own sources use none
// of them. They are in the global namespace, so their names reach every source that includes this header.

// ====================================================================================================================
// Print: declarations
// ====================================================================================================================

// The density output (-P statusm, rp180, apd): the negative's density above the film base in a standard measure, as
// code values in a 10-bit DPX or a 16-bit TIFF.
//
// A pixel, after the film base's merged matrix (crosstalk correction and film base scale; the film base lands on
// base_level), is a transmittance relative to bare film in each of the scanning light's channels:
//
//     T = v / base_level,  D_light = -log10(max(T, kDensityMinT)),  D = M * D_light,  code = round(offset + scale * D)
//
// M is the bundle's 3x3 matrix for the measure (the profile bundle part of negicc.h). The two encodings are the Cineon
// one (10 bits: offset 95, 500 codes per unit density) and the 16-bit one (offset 1520, 8000 codes per unit density),
// so RP 180 density in a DPX is a Cineon file. Academy Printing Density is written as ADX: ADX10 and ADX16 have these
// offsets and scales (SMPTE ST 2065-3:2020 4.3.3 equation 2 and 4.3.2 equation 1), with the standard's channel factors
// on the density, code = offset + scale * k * D with k = 1.00, 0.92, 0.95 for R, G, B.
// Status M and RP 180 take no factors (k = 1).
//
// Both files carry what is needed to read and recompute them, as JSON (schema negicc.density/1): the measure, the
// encoding, the film base, the matrix applied and the light's record with its spectrum. A TIFF has it in its
// ImageDescription tag, a DPX in its user-defined data section.

constexpr double kDensityMinT = 1e-4;   // the transmittance floor before the log (4.0 density)

struct DensityEncoding {
  int bits = 16;
  int offset = 1520;
  int scale = 8000;
  double k[3] = {1.0, 1.0, 1.0};      // the channel factors on the density (ADX: 1.00, 0.92, 0.95)
  int max_code() const { return (1 << bits) - 1; }
};
// The channel factors of ADX (SMPTE ST 2065-3:2020, equations 1 and 2).
constexpr double kAdxChannelFactor[3] = {1.00, 0.92, 0.95};
inline DensityEncoding density_encoding(bool dpx, DensityMeasure measure) {
  DensityEncoding e;
  if (dpx) {
    e.bits = 10;
    e.offset = 95;
    e.scale = 500;
  }
  if (measure == kDensityApd) {
    for (int c = 0; c < 3; ++c) e.k[c] = kAdxChannelFactor[c];
  }
  return e;
}

// Pixels whose code was clamped, per channel (R, G, B): below 0 (film much thinner than the base) and above the
// largest code (film denser than the encoding reaches).
// floor: pixels whose transmittance in a light channel was at or below kDensityMinT, so that the density of that
// channel was held at its floor before the matrix (R, G, B of the light).
struct DensityClip {
  uint64_t low[3] = {0, 0, 0};
  uint64_t high[3] = {0, 0, 0};
  uint64_t floor[3] = {0, 0, 0};
  uint64_t pixels = 0;
};

// Converts the image in place (width x height, `channels` 16-bit samples a pixel, the first three R, G, B) to density
// codes: merged the film base's merged matrix (negicc_core.h apply_matrix on the sample / 65535), base_level the
// value the film base lands on (the bundle's normalization target / 65535), m the measure's matrix.
void density_encode_in_place(uint16_t* pixels, int width, int height, int channels, const float merged[9],
                             double base_level, const double m[9], const DensityEncoding& enc, DensityClip& clip);

// What the file's metadata records beyond the bundle's own (ProfileBundle::light_source_json, density_fit_json).
struct DensityRecord {
  DensityMeasure measure = kDensityStatusM;
  DensityEncoding encoding;
  std::string container;              // "dpx" or "tiff"
  std::string bundle_key;             // the profile bundle
  double film_base_rgb[3] = {0, 0, 0};
  std::string film_base_source;
  const DensityMatrix* matrix = nullptr;
  const ProfileBundle* bundle = nullptr;
  const DensityClip* clip = nullptr;  // null before the conversion
  // The balance the picture conversion's parameter search found for this frame (E, g, b as -E, --gain_g and --gain_b
  // take them, and the bracket it chose). It is not applied to the densities; the file records the colour part
  // (g and b) as offsets a grading tool can add (density_metadata_json: suggested_balance).
  bool has_balance = false;
  double balance_e = 1.0, balance_g = 1.0, balance_b = 1.0;
  std::string balance_target;
  std::string tool_version;
  // The metadata's "print" block, as JSON text: how the frame prints on each print material
  // (DensityPrints::metadata_json). Empty: none.
  std::string print_json;
  // A density run with --print, on either light: each material's own print search (pipeline.cc;
  // search_print_continuous), by material id: its mid grey A, E (the -E units), g, b and the nearest bracket. Its CLF
  // prints the frame as the picture conversion would print it on that material. Empty (a material without a print
  // search): its CLF takes the scene's search.
  struct PrintSearch {
    double a = 0.0, e = 1.0, g = 1.0, b = 1.0;
    std::string bracket;
    bool converged = false;
    int steps = 0;
    double last_move[3] = {0, 0, 0};    // |dA| in stops, |dg|, |db| of the search's last step
    double above_knee = 0.0;            // the crop's share above the knee at the bootstrap
    double past_knee[3] = {0, 0, 0}, past_cut[3] = {0, 0, 0};   // at E (the matrix at E alone), per channel (shares)
    std::string stopped_by;             // parameters, residual or steps
    double a0 = 0.0, target_l = 0.0;    // the A its bootstrap started from; the mean L* of the lightness target it held
    double above_white[3] = {0, 0, 0}, at_white[3] = {0, 0, 0};   // at the result, per channel (shares)
    // --brightness, --tint, --warmth on this print (solve_print_knobs): a, g and b above are the moved ones; these the
    // search's, the knobs as given (brightness in stops, tint a*, warmth b*), and the print's grey where they put it
    bool knobs = false, knobs_limited = false;
    double knob_base_a = 0.0, knob_base_g = 1.0, knob_base_b = 1.0;
    double knob_given[3] = {0, 0, 0};
    double knob_grey_lab[3] = {0, 0, 0};
  };
  std::map<std::string, PrintSearch> print_search;
};
// The metadata as JSON text (schema negicc.density/1); indent < 0 for one line.
std::string density_metadata_json(const DensityRecord& rec, int indent = -1);

// The colour balance the picture conversion found for the frame, as offsets on the measure's density with their mean
// taken out (zero sum), and that mean. The conversion multiplies the light's channels by E, E g and E b, so the whole
// of what g and b do to the density is offset + mean in every channel: the zero-sum part is the colour, the mean is
// level. False, and zeros, when the record has no balance.
bool density_balance_offset(const DensityRecord& rec, double offset[3], double* mean = nullptr);

// ---- The print of a density file ------------------------------------------------------------------------------------
//
// Beside a density file goes, for each print material --print names, a CLF that prints it: density -> the print
// material's printing density -> its curves and dyes -> sRGB.
//
// A print material (a paper or a print film) and what is computed with it when a density file is written: one
// matrix for the frame, and the print's curves and colour table as CLF nodes.
//
// A print material is a print profile of spektrafilm (https://github.com/andreavolpato/spektrafilm), a JSON file read
// as published: those under 3rd_party/spektrafilm/profiles/ are built into neg_process, and --print takes any other
// by its path. The profile gives the curve of each layer, the dye spectra and base, and the layers' spectral
// sensitivity; the rest is computed when it is loaded (load_spektrafilm_print), the print matrix for the film of the
// bundle in use. A profile holds nothing of a film stock, and a film's bundle holds nothing of a print material.
//
// The model, for a log exposure of each layer:
//     dye amounts   delta_c = curve_c(logE_c)                    (linear interpolation, held at the ends)
//     transmittance tau(lambda) = 10^-(sum_c delta_c dye_c(lambda) + base(lambda))
//     XYZ           = sum_lambda tau(lambda) observer(lambda)
// and a printing density pd is exposed as logE = aim - (pd - grey_density), aim the exposure that prints an 18 % grey.

struct PrintMaterial {
  std::string id, name, viewing_illuminant;
  // The printing density exposed as the print's mid grey: a stated value (DESIGN.md), which cancels inside the CLF.
  double grey_density = 0.70;
  // The 3x3 from a measure's density to the density this material reads, indexed by DensityMeasure (Status M, RP 180,
  // Academy Printing Density), for the film model named (the bundle's the material was loaded for).
  struct ToPrint {
    std::string film;
    std::array<double, 9> m;
  };
  std::vector<ToPrint> to_print[kDensityMeasures];
  // Why a measure has no matrix when the bundle's film_model gives none (not finite); empty otherwise.
  std::string no_print_matrix[kDensityMeasures];
  // The matrix for a film model; null when the material has none for the measure and the film.
  const ToPrint* to_print_for(int measure, const std::string& film) const {
    for (const ToPrint& t : to_print[measure]) {
      if (t.film == film) return &t;
    }
    return nullptr;
  }
  std::vector<double> log_exposure;           // ascending
  std::vector<double> curves[3];              // dye amount of each layer along log_exposure
  std::vector<double> dye[3], base;           // spectral densities on the wavelength grid
  std::vector<double> observer[3];            // X, Y, Z weights on the grid (viewing light x observer)
  double white[3] = {1, 1, 1};
  double aim[3] = {0, 0, 0};
  double to_srgb[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // XYZ under the viewing light -> linear sRGB
};

// A print material from its spektrafilm print profile (a JSON with info and data), for the film of one bundle.
// Computed here:
//
//   the print's own density   read with its layers' sensitivity under a 3200 K blackbody printer lamp
//   the observer              the CIE 1931 2 degree observer (the print code's CIE tables) under D50 for a paper, D55
//                             for a print film (info.support), adapted to D65 with CAT02 for sRGB
//   aim                       the exposure of each layer that prints an 18 % grey
//   the print matrix          Q L inv(M) for each measure of the bundle, computed as Q inv(F) when the measure has
//                             no light_density_factors: F its film_to_measure (dye amounts to the measure's density),
//                             L the bundle's light_to_film, M = F L diag(alpha) the measure's full matrix, and Q the
//                             3x3 from the film's dye amounts to the print's own density, least squares on fixed
//                             sample colours of the bundle's film_model (every point of its neutral scale between
//                             log exposure -1.5 and 1.0, each with 27 colour variations). P M = Q L for every
//                             measure, so the bracket's uncalibrated APD slopes apply once
//
// The profile is checked first: info.stage "printing", info.support paper or film (it picks the viewing light),
// log_exposure ascending with one row of density_curves per value, each layer's curve rising with exposure (by more
// than 0.5 over the profile, never more than 0.02 below its highest value so far) and each layer's log_sensitivity
// peaking within 6 (log10) of the most sensitive layer's. --print loads a file given to it at option time, without a
// bundle, so every rule here stops a wrong file before any frame is converted.
//
// id names the material when the profile's info.stock is not a string. False with why in err; a bundle without a
// film_model gives a material without matrices, which frame_print refuses, and one whose film_model gives a matrix
// that is not finite a material without matrices and no_print_matrix saying why.
bool load_spektrafilm_print(const std::string& json_text, const std::string& id, const ProfileBundle& bundle,
                            PrintMaterial& out, std::string& err);

// The material as CLF nodes, to follow a Matrix that gives its printing density: Matrix (density -> log exposure) /
// Range / LUT1D (the curve of each layer) / Matrix / LUT3D (dye amounts -> linear sRGB, cube points a side) /
// Exponent (the sRGB curve). The text ends before </ProcessList>; empty when a value is not a finite number.
std::string print_clf_nodes(const PrintMaterial& m, int curve_points = 4096, int cube = 45);

// How one frame prints on one print material. Everything is looked up and multiplied; nothing is fitted when a file
// is written. The bundle gives, for the exposure bracket the picture conversion's search chose (without a search, the
// middle one), what the film's chart says of that bracket (DensityBracket), and the print material gives its matrix:
//
//   grey      the frame's mid grey as the measure's density: the bracket's mid grey, denser by log10(E) in every
//             light channel for the exposure E the search found. With the balance (the gains g and b, level
//             included) it prints as the print's mid grey. A bundle with the film's grey balance curve, under
//             either light, takes the curve's grey at the bracket's mid grey APD (curve_grey), as its print ICCs do.
//   slopes    they give the film's grey scale equal contrast in the three channels: from the grey balance curve
//             (curve_slopes), else the bracket's own (read only for a bundle without the curve).
//
//     printing density = grey_density + slopes * (to_print * (D + balance - grey))
struct FramePrint {
  std::string id, name;               // the print material
  double to_print[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::string to_print_for;           // the film model the matrix is for
  std::string bracket;
  bool searched = false;              // the bracket and exposure come from the search
  double exposure = 1.0;
  double grey[3] = {0, 0, 0};
  double slopes[3] = {1, 1, 1};
  bool from_curve = false;            // grey and slopes are the film's grey balance curve's (curve_grey, curve_slopes)
  // The material's own print search (DensityRecord::print_search): the mid grey A the grey and slopes are taken at,
  // and its gains, which the CLF's balance applies in place of the picture conversion's.
  bool print_searched = false;
  double a = NAN, gain_g = 1.0, gain_b = 1.0;
  bool converged = false;
  int steps = 0;
  double last_move[3] = {0, 0, 0}, above_knee = 0.0;
  double past_knee[3] = {0, 0, 0}, past_cut[3] = {0, 0, 0};   // at E (the matrix at E alone), per channel (shares)
  std::string stopped_by;
  double a0 = 0.0, target_l = 0.0, above_white[3] = {0, 0, 0}, at_white[3] = {0, 0, 0};   // (PrintSearch)
  bool knobs = false, knobs_limited = false;   // the knobs moved a, gain_g and gain_b from these (PrintSearch)
  double knob_base_a = NAN, knob_base_g = 1.0, knob_base_b = 1.0, knob_grey_lab[3] = {0, 0, 0};
  double knob_given[3] = {0, 0, 0};            // brightness, tint, warmth as given
  double grey_density = 0.70;
};
// False, with why in err, when the bundle has nothing for the bracket or the material has no matrix for the measure.
bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err);

// A Common LUT Format (CLF, ST 2136-1) transform for one density file and one print material: the file's code values
// (0-1) to the print, as sRGB. Its first node is a Matrix for the frame, with the balance (density_balance_offset),
// the exposure and the slopes of FramePrint; nodes is the print material's own (print_clf_nodes). This is the only
// place the balance, the exposure and the slopes are applied.
std::string frame_print_clf(const DensityRecord& rec, const FramePrint& print, const std::string& nodes);

// The print materials built into neg_process (3rd_party/spektrafilm/profiles/<id>.json), in the order of their ids,
// with their matrices for the film of bundle; without a bundle, or with one that has no film_model, they have no
// matrix (enough to list them). One that cannot be read is left out, with a warning.
std::vector<PrintMaterial> builtin_print_materials(const ProfileBundle* bundle = nullptr);
// Whether a --print value names a file (a spektrafilm print profile: a path, or a name ending in .json) and not a
// built-in print material.
bool is_print_file(const std::string& name);

// The text of a spektrafilm print profile given as a file, and its stem (the name without folder and .json), the id of
// a profile without info.stock. False with "cannot be read" in err for a file that cannot be opened or read.
bool read_print_file(const std::string& path, std::string& text, std::string& stem, std::string& err);

struct DensityPrints {
  std::vector<PrintMaterial> materials;
  std::vector<FramePrint> frames;
  std::vector<std::string> paths;
  std::string metadata_json;          // the "print" block of the file's metadata; empty with no print
  std::string error;                  // a material asked for that has no CLF, and why: the run fails
};
// The prints of one density file: for each print material asked for (--print: ids of builtin_print_materials, "all",
// or the path of a spektrafilm print profile; none gives no print), how the frame prints on it and the CLF to write,
// named after the whole output as the sidecar is (<output>.<material>.clf: a CLF fits one file, its code scale,
// measure and channel factors).
DensityPrints density_prints(const DensityRecord& rec, const std::string& output,
                             const std::vector<std::string>& print_ids);

// ---- The print ICC: a frame printed as the picture conversion renders it --------------------------------------------
//
// With --print and a picture output on a profile bundle, the print's own search runs on the print profile
// (search_print_continuous) and the conversion renders that profile at the search's A: the film printed on the print
// material, shaped as the chart profiles are (ICC v2, RGB -> PCS XYZ D50, A2B0 lut16: 4096-entry input curves, a 45^3
// table, identity output curves), so the conversion and the output run it as they run them. The bundle carries one
// print ICC per print material (print_profiles, built by the profiling tool when it saved the bundle);
// its table serves every bracket, and each bracket's three input curves are computed here:
//
//     in_c(x) = Paper_c((G_c(-log10(unknee(x) / n0)) - A_mid) (1 + t_c))
//
//   G_c       the film's grey balance curve (GreyBalanceCurves): light density in channel c -> APD of the film's grey
//   A_mid     the bracket's mid grey as APD: the green APD of its gs10 (bracket_mid_apd)
//   t_c       the bracket's tilt relative to the curve (DensityBracket::tilt; 0 without grey_relative)
//   Paper_c   0.75 curve_c(aim_c - eta_c alpha_i) / top_c + 0.25 (1 - i / (kPaperAlphaPoints - 1)) on
//             alpha_i = kPaperAlphaStart + kPaperAlphaStep i (-1.5 .. 1.8), eta = P (1, 1, 1) with P the material's
//             print matrix for APD, top_c the channel's largest value; held strictly decreasing on [0, 1]
//             (paper_curve)
//   unknee    the inverse of the soft knee at kKnee the conversion applies before the profile
//
// and the table, at node a: alpha_c = Paper_c^-1(a_c), printing density 0.70 + Q L (psi alpha) with
// psi = M^-1 (1, 1, 1) and Q L = P M, the material's dyes, XYZ under its viewing light adapted to D50 (Bradford; none
// for a paper). A print material without a bundle ICC (a --print file), or whose bundle ICC no longer matches what is
// computed here, gets its table computed here (print_table).
// The print ICC's copyright tag, as spektrafilm's licence asks of a derivative (its author, its source, the licence,
// and that and by whom it was modified). The profile-building tools write the same text into the bundle's print ICCs.
// The print CLF's description (frame_print_clf in print_emulation.cc) gives the same attribution in its own words,
// with the licence at spektrafilm's main branch rather than at this pinned commit.
constexpr const char* kPrintIccCopyright =
    "Tables derived by negicc from spektrafilm's print profile (Andrea Volpato, "
    "https://github.com/andreavolpato/spektrafilm), modified: its curves resampled and its dye spectra computed into "
    "colours. CC BY-SA 4.0, https://github.com/andreavolpato/spektrafilm/blob/"
    "3bb2c2d2801ff68b92019cf1dbcbb133d60832bc/SPEKTRAFILM_LICENSE.txt";
constexpr int kPrintIccPoints = 4096;
constexpr int kPrintIccCube = 45;
// Paper_c's grid: -1.5 to 1.8 in steps of 0.005, 661 points.
constexpr double kPaperAlphaStart = -1.5, kPaperAlphaStep = 0.005;
constexpr int kPaperAlphaPoints = 661;
using PrintCurves = std::array<std::vector<double>, 3>;

// curve_grey and curve_slopes at a mid grey A (green APD) in place of a bracket's.
void curve_grey_at(const ProfileBundle& bundle, double a, double light[3]);
void curve_slopes_at(const ProfileBundle& bundle, double a, double slopes[3]);
// The print's bracket term at a mid grey A, the one rule for the picture and the CLF: each bracket has its own tilt,
// and between two brackets' A_mid the tilt is interpolated linearly in A, held beyond the end ones, so at a bracket's
// own A_mid it is that bracket's tilt; a bundle with one bracket has that bracket's tilt. a_mid and tilt per bracket,
// in any order; zeros without brackets.
void print_tilt_at(const std::vector<double>& a_mid, const std::vector<std::array<double, 3>>& tilt, double a,
                   double t[3]);
// The same from a bundle's density brackets.
void print_tilt_at(const ProfileBundle& bundle, double a, double t[3]);
// The light densities of the grey balance curve's grey with the bracket's mid grey APD: the bracket's mid grey as the
// curve has it. A print CLF of a bundle with grey_balance_curves, under either light, takes it as the frame's mid grey
// (frame_print).
void curve_grey(const ProfileBundle& bundle, const DensityBracket& bracket, double light[3]);
// The slopes a print CLF takes for the bracket from the curve (DESIGN.md 5.9): with n(A) the light densities of the
// curve's grey at green APD A, gamma_c = sum_j M_cj dn_j/dA at the mid grey (dn_j/dA = 1 / G_j', G_j' a central
// difference of one step of the curve's table), slopes = (gamma_green / gamma_red, 1, gamma_green / gamma_blue).
void curve_slopes(const ProfileBundle& bundle, const DensityBracket& bracket, double slopes[3]);
// Paper_c on kPaperAlphaStart + kPaperAlphaStep i.
PrintCurves paper_curve(const PrintMaterial& m, const double p_apd[9]);
// A bracket's input curves on x = i / (kPrintIccPoints - 1), held non-decreasing (a 1e-9 ramp). gains (r, g, b): the
// print's gains as a density offset after the knee, dL_c = -log10(unknee(x) g_c / n0) (null: none).
PrintCurves bracket_print_curves(const GreyBalanceCurves& g, const PrintCurves& paper, double a_mid, double n0,
                                 const double tilt[3], const double gains[3] = nullptr);
// The table: kPrintIccCube^3 XYZ D50, red slowest; empty when a value is not a finite number.
std::vector<double> print_table(const PrintMaterial& m, const double p_apd[9], const double m_apd[9],
                                const PrintCurves& paper);
// An ICC v2.4 input profile with tags desc, cprt, wtpt and A2B0 (the curves, the table, identity output curves),
// dated date (year, month, day, hour, minute, second), so the same inputs give the same bytes.
std::vector<uint8_t> write_print_icc(const PrintCurves& curves, const std::vector<double>& table,
                                     const std::string& description, const std::string& copyright,
                                     const std::array<int, 6>& date);
// The date a bundle's print ICCs carry: its source file's timestamp (profile_<Key>_<YYYYMMDD>_<HHMMSS>), else
// 2026-01-01 00:00:00.
std::array<int, 6> bundle_date(const ProfileBundle& bundle);
// Multiplies the X, Y and Z of an ICC's A2B0 table (a lut16's) by k[0], k[1] and k[2] in place, each code rounded and
// held at 65535: the print relative to its paper's white in the profile an output embeds. False when it has none.
bool scale_print_icc_table(std::vector<uint8_t>& icc, const double k[3]);
// Writes xyz into an ICC's wtpt tag in place (the media white: the paper's, for a print made relative to its paper).
// False when it has none.
bool set_icc_wtpt(std::vector<uint8_t>& icc, const double xyz[3]);
// The paper's white under the material's viewing light, as the print CLF's XYZ has it: the colour of the print where
// every layer holds its least dye (each curve's lowest value: the base and the layers' fog), the lightest it gets.
void print_paper_xyz(const PrintMaterial& m, double xyz[3]);
// Writes curves into an ICC's A2B0 input tables in place (a lut16 with as many entries). False when it has none.
bool set_print_icc_curves(std::vector<uint8_t>& icc, const PrintCurves& curves);

// The print at a continuous mid grey A (pipeline.cc: the print search): the input curves above with A for
// A_mid and the brackets' tilt at A (print_tilt_at: linear in A between the two neighbouring brackets, held beyond
// the end ones). At A = A_mid of a bracket the profile is that bracket's to the bit (set_curves_at writes the 16-bit
// codes an ICC holds, as code / 65535).
struct PrintContinuum {
  GreyBalanceCurves g;
  PrintCurves paper;
  double n0 = 1.0;
  std::vector<double> a_mid;                  // per bundle.targets, in its order
  std::vector<std::array<double, 3>> tilt;    // per bundle.targets
  void tilt_at(double a, double t[3]) const;  // print_tilt_at over these brackets
  // gains (r, g, b): the print's gains in its curves, after the knee (null: none).
  PrintCurves curves_at(double a, const double gains[3] = nullptr) const;
  void set_curves_at(double a, ParsedFilmProfile& prof, const double gains[3] = nullptr) const;
  // The tables set_curves_at writes into prof.in_trc.
  void trc_at(double a, const double gains[3], std::vector<float> trc[3]) const;
  double a_lo() const;                        // the range A is searched in: the curve's green APD range
  double a_hi() const;
};

// The print profile of every bracket of a bundle on one print material (a built-in id or a spektrafilm file).
struct BracketPrints {
  PrintContinuum continuum;                    // the print at a continuous A
  std::string id, name;
  std::string file, file_sha256;               // a material given as a file: its path as given and its SHA-256
  bool from_bundle = false;                    // the table is the bundle's print ICC's, else computed here
  std::vector<std::vector<uint8_t>> icc;       // per bundle.targets, in its order
};
bool bracket_print_profiles(const ProfileBundle& bundle, const std::string& print, BracketPrints& out,
                            std::string& err);

// ====================================================================================================================
// Film base: declarations
// ====================================================================================================================

// The film base read from a capture of unexposed film (--film_base <file>[:x,y,w,h]), in place of the base read from
// the scan itself (the film base detector part of negicc_core.h).
//
// The capture is decoded as the scan's half-size view is: a Sony CFA RAW, black subtracted, its 2x2 cells as RGB,
// with the flat field. The base is read over a region of it, taken through the crosstalk matrix and scaled to the
// scan's exposure, which gives it in the conversion's film base space, as a base read from the scan is.
// The region and the base on it are read on that view by film_base_from_capture (the film base detector part of
// negicc_core.h, with FilmBaseCapture and the tiles' thresholds).

// The exposure of a scan or a film base capture whose file records none (Scan::scan_exposure,
// read_film_base_capture): 1/8 s at ISO 100.
constexpr double kDefaultScanShutterS = 0.125, kDefaultScanIso = 100.0;

// Reads the film base from the capture at path. roi: the region in sensor pixels, or null to find it. flat: the scan's
// flat field (null for none). crosstalk: the 3x3 row-major crosstalk matrix (null for none). expected_base: the
// bundle's film base through the crosstalk matrix, whose colour the tiles are held to (null: no colour test).
// scan_shutter_s, scan_iso: the scan's exposure. False with err on a file that cannot be read, a capture that is not a
// Sony CFA RAW, a region off the sensor or clipped, or a capture with no usable film.
bool read_film_base_capture(const std::string& path, const int* roi, const negicc::FlatFieldPtr& flat,
                            const double* crosstalk, const double* expected_base, double scan_shutter_s,
                            double scan_iso, FilmBaseCapture& out, std::string& err);

// ====================================================================================================================
// Options: declarations
// ====================================================================================================================

// neg_process's command line: the options, parsed and checked on their own (what needs the profile is checked by
// prepare, the pipeline part of negicc.h).

struct Options {
  int argc = 0;
  char** argv = nullptr;                       // the command line, for the sidecar

  // Input
  std::vector<std::string> files;              // one RAW, DNG, TIFF or JPEG, three captures, or a pixel shift's four
  InputMode input_mode = InputMode::kRaw;
  bool tiff_input = false;                     // one TIFF: camera RGB with a tri bundle, else Rec.2020
  const char* implied_light = nullptr;         // the light the input implies ("tri", "d65"), nullptr for a RAW
  int quality = -1;                            // --quality
  bool no_crop = false;                        // --no_crop
  std::string flat;                            // --flat, empty when not given

  // Profile
  std::string film_stock, light_source, film_profile;  // --film_stock, --light_source, -p (empty when not given)
  std::string target;                          // --target, empty when not given
  bool target_given = false;

  // Conversion parameters
  std::vector<float> r_coeff, g_coeff, b_coeff;  // -r, -g, -b
  bool row_given[3] = {false, false, false};
  float exposure_comp = 1.0f;                  // -E
  bool exposure_given = false;                 // -E (the search keeps it)
  float gain_g = 1.0f, gain_b = 1.0f;          // --gain_g, --gain_b (the search holds them)
  bool gain_g_given = false, gain_b_given = false;
  float brightness = 0.0f;                     // --brightness (stops of the grey's luminance)
  float tint = 0.0f;                           // --tint (a* of the grey)
  float warmth = 0.0f;                         // --warmth (b* of the grey)
  bool knobs_given = false;                    // --brightness, --tint or --warmth given

  // Geometry
  int roi[4] = {0, 0, 0, 0};                   // --roi x y w h (sensor pixels)
  bool has_roi = false, roi_auto = false;      // --roi given as numbers / as 'auto'
  char roi_axis_hint = 0;                      // --roi auto h|v
  FilmFormat film_format = FILM_35MM;          // --film_format
  bool film_format_given = false;              // --film_format was given (else 35mm for --roi auto)
  int rot_cw = 0;                              // --rot_cw (or the --roi's fifth value)
  bool hflip = false, vflip = false;

  // Parameter search
  // --dino: the intent adapter (an adapter.onnx or its directory) and the neutral and tone heads' files that replace
  // the built-in ones; empty: the built-in one. dino_arg is --dino's value as given, which the replay line repeats.
  std::string dino_arg, dino_adapter, neutral_head_file, tone_head_file;
  // The tone stage (the tone stage part of negicc_core.h). --tone_stage auto (the default) runs it after every bundle
  // search with -P srgb, on white light and TriLED alike, whether or not the curve fit runs; on does the same and is an
  // error where it does not apply; off turns it off. curve:black,white,gamma,contrast (ToneCurve) replays a recorded
  // curve (a sidecar's cli_equivalent), with given values and no search. highlights_first and least_total run it as
  // auto does, with their own rule where no share of the moves holds both ends (tone_fallback; auto and on keep the
  // shadows first).
  enum ToneStageMode { kToneStageAuto, kToneStageOn, kToneStageOff };
  ToneStageMode tone_stage = kToneStageAuto;
  ToneClipFallback tone_fallback = kToneFallbackShadowsFirst;
  bool tone_replay_given = false;
  double tone_replay[4] = {};
  std::string dino_backend = "auto";
  int dino_iters = 4;
  bool check_renderer = false;   // --check_renderer: the bundle search's passes through a counting renderer
  // --probe_intent FILE, --probe_params "t,E,g,b;...", --probe_stops s [s ...]: a measurement aid
  // (ParamSearchOptions::probe_path). The search writes the network's predictions for chosen renders to
  // FILE and neg_process stops without converting (the print's search writes it and the conversion goes on).
  std::string probe_intent;
  std::vector<std::array<float, 4>> probe_params;
  std::vector<float> probe_stops = {0.0f};
  bool dino_iters_given = false;

  // Fitted G/B input curves (the input curves part of negicc_core.h). --fit_curves auto (the default) fits them on
  // every white-light bundle search with -P srgb or srgb-g10, given parameters or not, and nowhere else (TriLED, whose
  // curves are the measured ones; another or no -P); on does the same and is an error where it does not apply; off
  // turns the fit off. The tone stage has its own flag.
  enum FitCurvesMode { kFitCurvesAuto, kFitCurvesOn, kFitCurvesOff };
  FitCurvesMode fit_curves = kFitCurvesAuto;
  bool input_curves_given = false;             // recorded curves given to --fit_curves (replayed, not fitted)
  negicc::InputCurves input_curves;

  // The film base (--film_base): auto (the default: a base capture's values kept in <output>.json by a previous run,
  // else read from the scan), or a capture of unexposed film, with the region to read it over (x y w h in sensor
  // pixels) when given after a colon; without one the region is found on the capture (the film base part of
  // negicc.h).
  std::string film_base_file;                  // empty: auto, or scan
  bool film_base_scan = false;                 // --film_base scan: read from the scan, whatever <output>.json holds
  bool film_base_roi_given = false;
  int film_base_roi[4] = {0, 0, 0, 0};

  // Output
  std::string output;                          // -o
  std::string colorspace;                      // -P, empty when not given
  // -P statusm, rp180 or apd: the output is the negative's density above the film base in that measure, not a
  // picture. -o's extension gives the container: .dpx (10 bits, code = 95 + 500 D) or .tif / .tiff (16 bits,
  // code = 1520 + 8000 D).
  bool density = false;
  DensityMeasure density_measure = kDensityStatusM;
  bool output_dpx = false;
  // --print: the print materials (ids of the built-in ones, "all", or the path of a spektrafilm print profile): with a
  // picture output the first is the print the conversion renders; with a density output a CLF is written beside the
  // file for each.
  std::vector<std::string> prints;
  bool no_metadata = false;
};

// ====================================================================================================================
// Output: declarations
// ====================================================================================================================

// neg_process's output: the picture's 16-bit TIFF with its ICC profile, the density file (a 16-bit TIFF or a 10-bit
// DPX) with its print CLFs, and the JSON sidecar beside the output. Their code is np_output.cc.

// Writes proc's image (or rgb3, 3 samples a pixel, in its place) as a 16-bit TIFF to output with the ICC profile
// attach_profile names ("srgb", "srgb-g10" or a file) or, when that is empty, embedded_icc. 0 on success.
int write_tiff(LibRaw* proc, const std::string& attach_profile, const std::string& output,
               const std::vector<uint8_t>& embedded_icc = {}, const uint16_t* rgb3 = nullptr);

struct DensityFileInfo {
  std::string make, model;            // the camera
  std::string software;               // "neg_process <version>"
  long long timestamp = 0;            // the capture's time (seconds since the epoch); 0 or less: no date written
};

// Writes density codes (width x height, `channels` samples a pixel, the first three R, G, B) to a 16-bit RGB TIFF with
// description (the metadata JSON) in its ImageDescription tag, or to a 10-bit RGB DPX (three samples in each 32-bit
// word, big-endian; the transfer and colorimetric codes and the element's description name the measure) with it in the
// user-defined data. 0 on success.
int write_density_tiff(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                       const std::string& description, const DensityFileInfo& info);
int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, DensityMeasure measure,
                      const DensityFileInfo& info);

// Writes the print CLFs of a density file (density_prints, the print part). 0 on success; a CLF that cannot be
// written removes those written before it and returns 1.
int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints);
void remove_print_clfs(const DensityPrints& prints);

// What the sidecar records.
// An input's size and CRC-32 (file_fingerprint), kept so a run hashes each input once: the sidecar loader's check and
// the sidecar writer share it.

struct FileFingerprint {
  bool ok = false;
  uintmax_t size = 0;
  uint32_t crc = 0;
};

struct Sidecar {
  std::string output_path;               // the TIFF; the sidecar is <output_path>.json
  std::string search_renderer = "cpu";   // where the search's passes ran; the auto block records it when not the CPU
  int argc = 0;
  char** argv = nullptr;
  const LibRaw* proc = nullptr;          // the camera, ISO and shutter (nullptr: none recorded)
  std::vector<std::string> files;
  std::vector<std::string> bundle_files;  // the ICC files beside a bundle given as a file (bundle_side_files)
  // Each file's fingerprint when this run already took it (the sidecar loader); null or short: taken here.
  const std::vector<FileFingerprint>* file_fingerprints = nullptr;
  int orig_w = 0, orig_h = 0;
  bool half_size = false;
  bool has_roi = false;
  const int* roi = nullptr;              // x, y, w, h when has_roi
  std::string roi_source;
  const FrameDetection* frame_det = nullptr;  // --roi auto's detection
  std::string film_format;               // the format the base's detection looked for, given or not; or empty
  int rot_cw = 0;
  bool hflip = false, vflip = false;
  int out_w = 0, out_h = 0;
  // The film base: source "rebate" or "statistics" (read from the scan: film_base_detection), "bundle" or "none".
  bool film_base_applied = false;
  std::vector<double> film_base_rgb;
  std::vector<int> profile_film_base_rgb;
  std::string film_base_source;
  // A print on a base that is not a capture (Scan::print_floor_*): the base as found, its source and the factor.
  bool print_floor_applied = false;
  double print_floor_found_rgb[3] = {0, 0, 0};
  std::string print_floor_found_source;
  float print_floor_k = 1.0f;
  double knob_base_a = NAN;   // a print's A before the knobs (knobs.base.a)
  // The frame the floor read (the picture's): "roi" (the given --roi, or --roi auto's crop), "detected" (no --roi: the
  // frame detector's crop, at or above kBaseMinFrameConf), "inset_crop" (no --roi and no confident detection: the RAW's
  // inset crop, the search's frame) or "whole_scan" (the same with no inset crop, or a --roi auto that fell back to the
  // whole image). The detection's confidence only where the detection decided the
  // frame (a --roi auto that fell back; no --roi, with a detection that maps onto the scan), else -1.
  std::string print_floor_frame;
  float print_floor_frame_conf = -1.0f;
  const negicc::FilmBaseDetection* film_base_detection = nullptr;
  // The white-light decode's gains on the camera channels before the camera matrix; null: not a white-light decode.
  const float* balance_gains = nullptr;
  std::string profile_path;              // empty: no profile
  bool is_bundle = false;
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  InputSpace input_space = InputSpace::kCameraRawRgb;
  const float* merged_matrix = nullptr;
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;  // absolute, on the crosstalk (matrix.*)
  const float* station_params = nullptr;  // with a bundle: E, g, b as -E, --gain_g, --gain_b take them
  unsigned pin_mask = 0;
  float knee = 0.95f;
  ShoulderInfo shoulder;
  const DinoSolveResult* dino = nullptr; // the single-profile search, when it ran
  std::string dino_model, dino_backend, dino_arg;
  std::string dino_backend_arg;   // --dino_backend when given other than auto (where the search's passes run)
  bool dino_iters_given = false;
  int dino_iters = 4;
  const ParamSearchResult* auto_res = nullptr;  // the bundle search, when it ran
  const int* auto_frame_half = nullptr;
  std::string cached_auto_json_str;      // preserved "auto" JSON when reusing a sidecar
  std::string shadow_json_str;           // the search's shadow alternative and the choice (auto.shadow_alternative)
  std::string cached_dino_json_str;      // preserved "dino" JSON when reusing a sidecar
  std::string cached_intent_field_json_str;  // preserved "intent_field" JSON when reusing a sidecar
  std::string cached_input_curves_json_str;  // preserved "input_curves" JSON when reusing a sidecar
  std::string cached_print_search_json_str;  // preserved "profile.print.search" JSON when reusing a sidecar
  double cached_dino_views_ms = 0.0, cached_dino_render_ms = 0.0;
  double cached_dino_infer_ms = 0.0, cached_dino_search_ms = 0.0;
  bool has_knobs = false;
  float brightness = 0.0f, tint = 0.0f, warmth = 0.0f;
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  std::string colorspace;
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;
  double t_auto_ms = 0.0;
  double t_analysis_ms = 0.0;  // Scan::t_analysis_ms
  // Fitted G/B input curves of the conversion's bracket (the input curves part of negicc_core.h): source "fit"
  // (--fit_curves on, with curve_fit), "auto" (likewise) or "cli" (recorded curves given to --fit_curves); empty: none.
  std::string input_curves_source;
  const negicc::InputCurves* input_curves = nullptr;
  const WhiteLightFitResult* curve_fit = nullptr;
  // The models the search ran with: the intent adapter, the neutral head and the tone heads ("[built-in] <name>" or
  // the path --dino gave, with the file's size and CRC-32; empty: not loaded). A run that reused a previous sidecar
  // keeps that run's record (cached_heads_json_str).
  std::string head_path[3];
  FileFingerprint head_fp[3];
  std::string cached_heads_json_str;
  // The tone curve the conversion rendered with (curve_fit's, or replayed; null: none), whether it was applied to the
  // final image, and the clipping it measured.
  // A replay ("knobs" or "cli", else empty).
  std::string replay_source;
  const ToneCurve* tone_curve = nullptr;
  const ToneTargets* tone_targets = nullptr;  // this run's plan, as applied: its targets and highlight share
  // This run's tone stage, when it was planned (after the curve fit, or after the search where no fit runs): the
  // heads' plan and the curve the conversion solved from it.
  const ToneStagePlan* tone_stage_plan = nullptr;
  const TonePlanCurve* tone_plan = nullptr;
  double tone_share = 1.0;
  double tone_shadow_share = 1.0;
  // The rule that chose the curve where no share of the moves held both ends ("shadows_first", "highlights_first" or
  // "least_total"); null where a share held.
  const char* tone_fallback = nullptr;
  bool tone_applied = false;
  // This run's plan kept shares that move nothing (tone_moves_nothing), so no curve was applied: the record says so,
  // with the shares, the targets and the clipping, and the replay converts without the stage.
  bool tone_moved_nothing = false;
  const ToneClip* tone_clip = nullptr;
  // The curves the cli_equivalent replays when they differ from input_curves (re-anchored at the knobs' E, g and b);
  // null: input_curves.
  const negicc::InputCurves* cli_input_curves = nullptr;
  // A film base read from a capture (--film_base; film_base_source "capture", or "sidecar" when a previous run's is
  // taken): the capture's record as JSON (film_base.capture). Empty: none.
  std::string film_base_capture_json;
  // The density output's record (the print part of negicc.h, schema negicc.density/1) as JSON, also in the file.
  // Empty: none.
  std::string density_json;
  std::string used_json;                 // what the conversion used (conversion_used in pipeline.cc)
  bool density_fit_off = false;      // a white-light density run without --print, with --fit_curves off: its replay
                                     // keeps it
  // A density run's --print materials as given (built-in ids, 'all', or files): its replay gives them again, so the
  // base's floor (raised only with --print) and the CLFs are made again. Empty: no --print.
  std::vector<std::string> density_prints;
  std::string print_id, print_name;  // --print with a picture output: the print ICC that replaced the chart profiles
  std::string print_file, print_file_sha256;  // a material given as a file: its path as given and its SHA-256
  bool print_from_bundle = false;
  // The print's mid grey A (NaN without a print), the chosen bracket's A_mid, and the search's A0 (the A at which
  // the frame's log-average cell prints at L* 50).
  double print_a = NAN, print_bracket_a_mid = NAN, print_a0 = NAN;
  double print_tilt[3] = {NAN, NAN, NAN};   // the brackets' tilt at the A the print used (print_tilt_at)
  // The paper's white the print is relative to (PCS D50).
  double print_paper_xyz[3] = {negicc::kPcsWhite[0], negicc::kPcsWhite[1], negicc::kPcsWhite[2]};
  bool print_searched = false, print_converged = false;
  int print_steps = 0;
  float print_target_l = 0.0f, print_above_knee = 0.0f;
  // ContinuousPrintResult's shares: past_knee and past_cut at E (the matrix at E alone), above_white and at_white at
  // the result.
  float print_past_knee[3] = {0, 0, 0}, print_past_cut[3] = {0, 0, 0};
  float print_above_white[3] = {0, 0, 0}, print_at_white[3] = {0, 0, 0};
  bool print_a0_at_end = false;
  float print_last_move[3] = {0, 0, 0};
  std::string print_stopped_by;
};

// Writes <output_path>.json: what neg_process read and did (schema negicc.neg_process.sidecar/1).
// The build's version (git describe), as the sidecar's tool.version records it.
const char* neg_process_version();

// A file's size and CRC-32 (zlib), as the sidecar records each input (inputs.raw_file_stats). False when unreadable.
bool file_fingerprint(const std::string& path, uintmax_t& size, uint32_t& crc);
// The same for bytes in memory (a built-in model).
FileFingerprint memory_fingerprint(const void* data, size_t size);

// s with its bytes that are not UTF-8 written as U+FFFD, as the sidecar's writer writes them (a camera model or a
// path need not be UTF-8). The log shows a string as the sidecar records it, and a previous run's sidecar is compared
// with this run's strings as they would be recorded. Valid UTF-8 comes back unchanged.
std::string utf8_replaced(const std::string& s);

bool write_json_sidecar(const Sidecar& sc);

// Whether a path a previous run's sidecar records names this run's path (pipeline.cc).
bool paths_match(const std::string& recorded, const std::string& path);
// The ICC files beside a bundle given as a file (a split bundle's brackets and prints), as paths; none for a built-in
// bundle or one that holds its profiles inline (np_output.cc).
std::vector<std::string> bundle_side_files(const ProfileBundle& b);

// ====================================================================================================================
// Pipeline: declarations
// ====================================================================================================================

// neg_process's processing pipeline: the stages main runs in turn, the state they hand on, and the split between the
// scan as data (MosaicInput, RgbInput) and the compute on it (Backend: the CPU's, and a CUDA device's for the
// full-size steps).
//
//   prepare   the profile (bundle, ICC profile or bracket), the parameter search mode, the crosstalk correction, the
//             DINOv3 engine
//   decode    the input as data: the zero-copy CFA path (a Sony Bayer RAW or a pixel shift: the mosaic, black
//             subtracted in place, with a half-size view over it) or LibRaw's decode, a TIFF, a JPEG or three
//             captures (an RGB image); the frame (--roi, --roi auto); and the bundle's parameter search on the
//             half-size view, as each path has it
//   begin_conversion
//             the film base the sidecar records, the pins (-E, --gain_g, --gain_b) and the input matrix to start from
//   render    the full-resolution image of the frame: the backend debayers or merges a mosaic; an RGB image is one
//             already
//   solve     the conversion parameters the bundle search did not give: the bracket, E, g and b (an ICC profile's
//             single-profile search, or a bundle's four given values), the knee and the shoulder
//   convert   the backend converts the image in place to the output encoding
//   write     the TIFF and the sidecar

// Cached Auto / DINOv3 metadata loaded from <output>.json when solving grey-point knobs.
struct CachedSidecar {
  bool is_auto = false;                  // true: cached from "auto"; false: cached from "dino"
  int target_idx = -1;
  std::string target_name;
  float base_e = 1.0f, base_g = 1.0f, base_b = 1.0f;
  double print_a = NAN;                  // a print's mid grey A (profile.print.a); NaN: none
  std::string print_search_json_str;     // the print's search that found it (profile.print.search); empty: none
  std::string auto_json_str;             // raw "auto" JSON object from previous sidecar
  std::string dino_json_str;             // raw "dino" JSON object from previous sidecar
  std::string dino_backend;
  double t_dino_views_ms = 0.0, t_dino_render_ms = 0.0;
  double t_dino_infer_ms = 0.0, t_dino_search_ms = 0.0;
  // The previous run's fitted G/B input curves (replayed on its bracket, not refitted) and its intent field, carried
  // into the new sidecar so the knobs' output still records them.
  bool has_curves = false;
  negicc::InputCurves curves;
  std::string curves_source;
  std::string input_curves_json_str;
  std::string intent_field_json_str;
  std::string heads_json_str;
  // Its tone curve (tone_stage.curve), when it applied one.
  bool has_tone = false;
  ToneCurve tone_curve;
};

// What the profile and the options decide before the scan is read.
struct Setup {
  // The renderer the search's passes run through: the GPU's, or --check_renderer's comparison with it (null: the CPU's,
  // or --check_renderer's pass-through renderer without a device). Made only where a search runs.
  std::unique_ptr<PreviewRenderer> renderer;
  // Each input's size and CRC-32, taken once (by the sidecar loader when the knobs look for <output>.json) and
  // reused by the sidecar writer.
  std::vector<FileFingerprint> input_fingerprints;
  bool is_bundle = false;
  ProfileBundle bundle;
  std::string film_prof_arg;             // -p naming an ICC profile (not a bundle)
  bool has_profile = false;
  int manual_target_idx = -1;            // --target, -1 when not given
  bool exposure_given = false;           // -E: the exposure is not searched
  bool use_dino = false;                 // a parameter search runs
  bool bundle_search = false;            // the bundle's search (bracket, exposure and gains)
  // The bundle's search with --target, -E, --gain_g or --gain_b held (either light): Auto with those pinned.
  bool pinned_search = false;
  bool fit_curves_active = false;        // the curve fit runs (--fit_curves auto or on, where it applies)
  bool tone_active = false;              // the tone stage runs (--tone_stage auto or on, where it applies)
  bool has_cached_sidecar = false;       // <output>.json matched and supplies the base parameters
  CachedSidecar cached_sidecar;
  bool white_light = false;              // a rec2020_linear (d65) bundle
  bool density = false;                  // -P statusm, rp180 or apd: the output is density (the print declarations
                                         // above)
  // --print with a picture output on a profile bundle: the print material, and each bracket's print profile
  // (bracket_print_profiles in the print part of negicc.h, in bundle.targets' order). Auto searches the print itself
  // (search_print_continuous); the conversion renders the print at its mid grey A (print_a: the search's, or the
  // reused run's). Empty without.
  std::string print_id, print_name;
  std::string print_file, print_file_sha256;  // a material given as a file: the path as given and the file's SHA-256
  bool print_from_bundle = false;        // the print ICC's table is the bundle's (else computed by neg_process)
  std::vector<ParsedFilmProfile> print_profs;
  std::vector<std::vector<uint8_t>> print_iccs;
  PrintContinuum print_cont;          // the print at a continuous mid grey (the print part of negicc.h)
  ContinuousPrint print_in;           // the search's view of it
  double print_a = NAN;               // the print's mid grey A: the search's or a reused run's (NaN until then)
  // The lightest node of the print's table: the paper's white.
  double print_paper_xyz[3] = {negicc::kPcsWhite[0], negicc::kPcsWhite[1], negicc::kPcsWhite[2]};
  ContinuousPrintResult print_res;    // the print's search
  std::map<std::string, DensityRecord::PrintSearch> density_print_search;   // a density run's --print, per material
  bool print_searched = false;
  bool has_crosstalk = false;
  bool camera_rgb_input = false;         // a TIFF of linear camera RGB (a TIFF with a camera_raw_rgb bundle)
  RawDecode raw_decode = RawDecode::kRec2020;
  std::vector<float> r_coeff, g_coeff, b_coeff;  // the crosstalk correction's rows
  // Where each row comes from: "bundle", "given" (-r, -g, -b) or "identity" (three captures, one per light; an ICC
  // profile's row not given; a bundle's correction off); empty: no correction.
  std::string crosstalk_row_source[3];
  // Why a camera_raw_rgb bundle's correction is off (decode: the scan states a camera the bundle does not list); empty
  // when it is applied.
  std::string crosstalk_off;
  negicc::FlatFieldPtr flat_field;
  DinoV3Engine engine;
  std::string dino_path;                 // the backbone that runs ([built-in] ...)
  std::string neutral_head;              // the neutral-patch head that runs ([built-in] ...), empty when none
  // The intent adapter, neutral head and tone heads that run ([built-in] <name> or the --dino path) and their files'
  // size and CRC-32, for the sidecar.
  std::string head_path[3];
  FileFingerprint head_fp[3];
  ToneHead tone_head;                    // the tone heads (where the tone stage runs)
  // The tone curve the conversion renders with: this run's tone stage's, a previous run's (the knobs on its sidecar) or
  // given (--tone_stage curve:...); not predicted again on a replay.
  // A replay: "knobs" (the knobs on a previous run's sidecar) or "cli" (recorded values given); empty otherwise.
  std::string replay_source;
  bool has_tone = false;
  ToneCurve tone_curve;
  // Fitted G/B input curves the conversion's bracket renders through (the input curves part of negicc_core.h): source
  // "fit" (--fit_curves on: curve_fit has the fit), "auto" (--fit_curves auto, likewise), "cli" (recorded curves given
  // to --fit_curves) or empty (none).
  std::string input_curves_source;
  negicc::InputCurves input_curves;
  WhiteLightFitResult curve_fit;
  // This run's tone stage: the heads' plan (the curve fit's on its result, or plan_auto_tone_stage's on the search's
  // own result where no fit runs) and the curve solved from it (when tone_stage.planned). tone_stage_ran: a plan was
  // attempted this run (the sidecar then says what came of it).
  ToneStagePlan tone_stage;
  bool tone_stage_ran = false;
  TonePlanCurve tone_plan;
  // The bracket the curves were composed into (-1: none) and its profile before that (the un-curved input tables),
  // from which the knobs compose the curves again at the E, g and b they move to (reanchor_input_curves).
  int curves_base_idx = -1;
  ParsedFilmProfile curves_base_prof;
};

// The scan as data for a backend: a CFA mosaic (one, or the four frames of a pixel shift), black subtracted, in place.
struct MosaicInput {
  const uint16_t* frames[4] = {nullptr, nullptr, nullptr, nullptr};
  int n_frames = 0;                      // 1, or 4 for a pixel shift
  size_t pitch = 0;                      // uint16_t elements per row
  int width = 0, height = 0;             // the visible sensor area
  unsigned filters = 0;                  // LibRaw's CFA pattern
  float white_level = 65535.0f;          // the clipping level above black
  negicc::FlatSampleArgs flat;           // the flat field at full size (empty without one)
  const negicc::FlatTables* flat_tables = nullptr;   // the tables flat points into (for a copy on a device)
};

// The scan as data for a backend: an RGB image, `channels` samples a pixel (3, or LibRaw's 4).
struct RgbInput {
  uint16_t* pixels = nullptr;
  int width = 0, height = 0, channels = 3;
};

// A region of the sensor (x, y, w, h) and its orientation: the frame to render.
struct Region {
  int x = 0, y = 0, w = 0, h = 0;
  int rot_cw = 0;
  bool hflip = false, vflip = false;
};

// The output encoding and its profiles.
struct OutputSpec {
  OutputEncoding encoding = ENCODE_NONE;
  std::string attach_profile;            // "srgb", "srgb-g10", an ICC file, or the -p profile
  cmsHPROFILE xyz = nullptr;             // ENCODE_CUSTOM: the XYZ input and the output profile
  cmsHPROFILE custom = nullptr;
};
// The density output's conversion (the print part of negicc.h): the measure's matrix, the encoding and the clamped
// pixels.
struct DensityOutput {
  double base_level = 1.0;               // what the film base lands on after the film base's merged matrix
  float merged[9] = {0};                 // that matrix (crosstalk correction x film base scale), E = g = b = 1
  DensityEncoding encoding;
  DensityClip clip;
};

// The compute on the scan's data.
class Backend {
 public:
  virtual ~Backend() = default;
  // The full-resolution RGB of the region of a mosaic, turned and flipped, into rgb_out (3 samples a pixel,
  // dst_w x dst_h: w x h, or h x w when turned by 90 or 270 degrees): the grain-aware debayer of one frame, or the
  // merge of a pixel shift's four, with the flat field. The backend sizes rgb_out, at the point where it first needs
  // it. frame_done(k) frees in.frames[k]: a backend that copies a frame elsewhere calls it as soon as that frame is
  // copied and reads it no more, so that the two copies are not held longer than the copying takes; the caller calls
  // it for every frame after the render in any case.
  virtual void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h,
                      std::vector<uint16_t>& rgb_out, const std::function<void(int)>& frame_done) = 0;
  // Converts img in place to the output encoding through the shared pixel pipeline (the colour pipeline part of
  // negicc_core.h) of pc: the input matrix and knee, the film profile (tone curves and cLUT) and, for sRGB, the
  // shoulder. ENCODE_NONE stops after the knee; ENCODE_CUSTOM hands the profile's PCS XYZ to out.custom.
  virtual void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
                       const RgbInput& img) = 0;
  // Where the last render and the last convert ran: "cpu" or "gpu" (the record's "full_size").
  const char* render_ran = "cpu";
  const char* convert_ran = "cpu";
};

class CpuBackend : public Backend {
 public:
  void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, std::vector<uint16_t>& rgb_out,
              const std::function<void(int)>& frame_done) override;
  void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
               const RgbInput& img) override;
};

// The same on a CUDA device: the debayer or the merge by gpu_grain_aware_debayer and gpu_pixel_shift_merge, and the
// conversion to sRGB or linear sRGB by the search's GPU render (GpuRenderer). The mosaics, the flat field's tables and
// the picture are copied to the device and the result copied back.
//
// Against the CPU's picture: the debayer and the merge are equal in every value. The conversion computes the same
// convert_pixel, but the device rounds powf, tanhf and cbrtf differently in the last place (DESIGN.md 4.2), so a
// value of the exported picture can differ from the CPU's by one 16-bit code.
//
// Another output encoding, or an image that is not three samples a pixel, keeps the CPU's conversion (a custom
// profile's last step is Little CMS), and so does every step when no device runs this build's kernels: the device is
// asked for only where a step is about to use it.
class GpuBackend : public CpuBackend {
 public:
  void render(const MosaicInput& in, const Region& region, int dst_w, int dst_h, std::vector<uint16_t>& rgb_out,
              const std::function<void(int)>& frame_done) override;
  void convert(const OutputSpec& out, const negicc::PipelineConfig& pc, const float merged_matrix[9],
               const RgbInput& img) override;
};

// The backend of the full-size steps: the device's (which is the CPU's without a device that runs this build's
// kernels), unless --dino_backend cpu, which keeps everything on the CPU, or --check_renderer, whose picture is the
// CPU's. The same rule as the search's passes.
std::unique_ptr<Backend> make_backend(const Options& o);

struct LibRawDeleter {
  void operator()(LibRaw* p) const;
};
using LibRawPtr = std::unique_ptr<LibRaw, LibRawDeleter>;

// The search's shadow alternative (ParamSearchResult::shadow) through the stages after the search: one candidate's
// state from them (the curve fit, the tone plan and the roll-off), with its bracket's profile before (prof0) and after
// (prof) the fit composed curves into it.
struct ShadowCandidate {
  ParamSearchResult auto_res;
  bool has_curve_params = false;
  ConversionParams curve_params;
  bool has_rolloff = false;
  float rolloff_x_hi = 1.0f, rolloff_q995 = 0.0f;
  WhiteLightFitResult curve_fit;
  ToneStagePlan tone_stage;
  bool tone_stage_ran = false;
  TonePlanCurve tone_plan;
  bool has_tone = false;
  ToneCurve tone_curve;
  negicc::InputCurves input_curves;
  std::string input_curves_source;
  int curves_base_idx = -1;
  ParsedFilmProfile curves_base_prof;
  // The knobs' state (--brightness, --tint, --warmth): their solve on this candidate, and the curves re-anchored at it.
  bool has_knob_res = false;
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  bool has_knob_curves = false;
  negicc::InputCurves knob_curves;
  double knob_base_a = NAN;
  bool has_plain = false;  // and the conversion without them (Scan::has_plain)
  float plain_x_hi = 1.0f, plain_q995 = 0.0f;
  bool has_plain_prof = false;
  ParsedFilmProfile plain_prof;
  int prof_idx = -1;
  ParsedFilmProfile prof0, prof;
  ParsedFilmProfile result_prof0;  // in the alternative's: the result's bracket's profile before the stages
};

// The tone stage's clip check of the kept candidate, made in solve; convert takes it instead of checking again.
struct ToneCheck {
  bool valid = false;
  double white_before = 0, black_before = 0;
  ToneCurve curve;
  ToneTargets targets;
  double share = 1.0, shadow_share = 1.0;
  bool fell_back = false;  // no share held both ends (Options::tone_fallback chose the curve)
  // Made on the conversion without the knobs: convert takes its curve and counts the shares before the stage on the
  // knobbed image it converts, as a replay does.
  bool without_knobs = false;
};

// The choice between the search's result and its shadow alternative (solve). The shares are of the tone stage's clip
// check sample: white before, black before, white after and black after the stage.
struct ShadowChoice {
  bool checked = false;            // both went through the clip check
  bool kept_alternative = false;   // the alternative was taken
  double result[4] = {0, 0, 0, 0}, alternative[4] = {0, 0, 0, 0};
  double stages_ms = 0.0;          // the alternative's stages after the search
  double check_ms = 0.0;           // the alternative's solve and both candidates' clip checks
  const char* not_checked = nullptr;  // why an alternative was not checked
};

// The scan through the pipeline.
struct Scan {
  LibRawPtr proc;                        // the RAW (its metadata, and its image on the LibRaw path)
  LibRawPtr ps_extra[3];                 // a pixel shift's other three frames
  bool cfa = false;                      // the zero-copy CFA path
  MosaicInput mosaic;                    // on the CFA path
  Region cfa_region;                     // on the CFA path: the frame to render (sensor pixels)
  std::shared_ptr<const negicc::FlatTables> flat_full_tables;  // keeps mosaic.flat's tables alive
  std::vector<uint16_t> rgb3;            // on the CFA path: the rendered frame, 3 samples a pixel
  int orig_w = 0, orig_h = 0;            // the sensor size the sidecar records
  std::string input_source = "raw";      // what was read: "raw", "tiff:icc", "jpeg:srgb", "three:raw,raw,raw", ...
  // The camera make and model of the first input as LibRaw reads them (empty: an RGB TIFF or JPEG, or a RAW that
  // states none). camera_compared: a camera_raw_rgb bundle with a camera_models list and a stated model, compared
  // (decode); camera_listed: the list holds the model. Not compared: a white-light bundle, a bundle without the list,
  // or an input that states no model.
  std::string camera_make, camera_model;
  bool camera_compared = false, camera_listed = false;
  // The first input's inset crop (inset_crop), which a single RAW, or three captures, is cut to without --roi or
  // --no_crop.
  bool has_inset = false;
  int inset[4] = {0, 0, 0, 0};

  FrameDetection frame_det;              // --roi auto
  // The format the frame detection that placed the film base's patch looked for (FILM_FORMAT_AUTO: every format, or
  // no detection ran); cli_equivalent gives it as --film_format whenever the detection used one, given or --roi auto's,
  // so that a replay with the explicit --roi reads the same base.
  FilmFormat base_det_format = FILM_FORMAT_AUTO;
  int roi[4] = {0, 0, 0, 0};             // the frame in sensor pixels (--roi, or --roi auto's)
  bool has_roi = false;
  int rot_cw = 0;
  bool hflip = false, vflip = false;

  // The film base read from the scan (the film base detector part of negicc_core.h) for a bundle conversion, as
  // negicc-station's Auto reads it when no film base is given; not found: the bundle's.
  negicc::FilmBaseDetection film_base;
  const double* film_base_rgb() const { return film_base.found ? film_base.rgb : nullptr; }
  // The film base given in place of the scan's own (--film_base): read from a capture of unexposed film
  // (the film base part of negicc.h; film_base.source "capture"), or, with --film_base auto, the base a previous run
  // read from a capture and kept in <output>.json (source "sidecar", with that run's capture record as JSON).
  // has_given_base: film_base holds it and the scan's is not read.
  bool has_given_base = false;
  negicc::FilmBaseDetection given_base;
  // A print on a base that is not a capture (print_floor_factor): the base as found, its source ("rebate",
  // "statistics" or "bundle") and the factor it was raised by (film_base holds the raised base; 1: not raised).
  bool print_floor_applied = false;
  double print_floor_found_rgb[3] = {0, 0, 0};
  std::string print_floor_found_source;
  float print_floor_k = 1.0f;
  // The frame the floor read (the picture's): "roi" (the given --roi, or --roi auto's crop), "detected" (no --roi: the
  // frame detector's crop, at or above kBaseMinFrameConf), "inset_crop" (no --roi and no confident detection: the RAW's
  // inset crop, the search's frame) or "whole_scan" (the same with no inset crop, or a --roi auto that fell back to the
  // whole image). The detection's confidence only where the detection decided the
  // frame (a --roi auto that fell back; no --roi, with a detection that maps onto the scan), else -1.
  std::string print_floor_frame;
  float print_floor_frame_conf = -1.0f;
  double knob_base_a = NAN;   // a print's A before the knobs (recorded as knobs.base.a)
  bool has_base_capture = false;
  FilmBaseCapture base_capture;
  std::string base_capture_json;         // the capture record for the sidecar (film_base.capture)
  // The white-light decode's gains on the camera channels (ChannelBalance); 1, 1, 1: the mosaic was left as it was.
  float balance_gains[3] = {1, 1, 1};

  bool has_auto_res = false;             // the bundle search ran
  ParamSearchResult auto_res;
  int auto_frame_half[4] = {0, 0, 0, 0}; // the frame (x, y, w, h) in the half-size image the search saw
  bool frame_is_whole = false;           // that frame is the whole half-size image
  // Where the fit runs: the curve fit's re-metered E and fitted g and b on the combined profile, which the
  // conversion takes in place of the search's (auto_res keeps the search's own).
  bool has_curve_params = false;
  ConversionParams curve_params;
  const ConversionParams& params() const { return has_curve_params ? curve_params : auto_res.params; }
  // The curves re-anchored at the knobs' E, g and b, which the bracket renders through after the knobs and the
  // cli_equivalent replays (Setup::input_curves stays the record at the knobs' base, as the sidecar keeps it).
  bool has_knob_curves = false;
  negicc::InputCurves knob_curves;

  bool has_knob_res = false;             // --brightness, --tint or --warmth solved
  float knob_base[3] = {1.0f, 1.0f, 1.0f};
  KnobSolveResult knob_res;
  // A fresh run's conversion without the knobs, which its tone stage check is made on (solve): the roll-off at the
  // knobs' base and the bracket's profile before the knobs re-anchored its curves (a print: its A before the knobs,
  // knob_base_a). Set by the knobs where the tone stage runs and no record is replayed.
  bool has_plain = false;
  float plain_x_hi = 1.0f, plain_q995 = 0.0f;
  bool has_plain_prof = false;
  ParsedFilmProfile plain_prof;

  // The highlight roll-off of every bundle conversion, measured on the half-size frame as the bracket search measures
  // it: Auto's result and given values in decode, while that frame exists (render frees the mosaic; LibRaw's
  // half-size decode is analysed and gone).
  bool has_rolloff = false;
  float rolloff_x_hi = 1.0f, rolloff_q995 = 0.0f;
  // The tone stage as convert applied it to the final image (Setup::tone_stage has the plan).
  bool tone_applied = false;
  ToneClip tone_clip;
  ToneCurve tone_curve;            // the curve applied (the plan's, capped on this image; or the recorded one)
  ToneTargets tone_targets;        // with this run's plan: the targets it was solved for
  double tone_share = 1.0;         // and the share of the highlight and range moves kept
  double tone_shadow_share = 1.0;  // and of the shadow move
  bool tone_fell_back = false;     // and whether no share held both ends (Options::tone_fallback chose the curve)
  bool tone_moved_nothing = false; // and whether its kept shares moved nothing, so no curve was applied
  // The search's shadow alternative (ParamSearchResult::shadow, as the search returned it), the alternative after its
  // stages (null: none, or already chosen in solve), the choice, and the kept candidate's clip check.
  ShadowAlternative shadow_search;
  std::shared_ptr<ShadowCandidate> shadow_candidate;
  ShadowChoice shadow_choice;
  ToneCheck tone_check;

  double t_raw_0 = 0.0;                  // timings (ms) the sidecar records
  double t_raw_decode_ms = 0.0, t_frame_det_ms = 0.0, t_auto_ms = 0.0;
  // The bracket analysis on the half-size frame outside Auto's time: the knobs and the roll-off.
  double t_analysis_ms = 0.0;
  double t_full_conversion_ms = 0.0, t_tiff_write_ms = 0.0;
  // Where the full-size steps ran, "cpu" or "gpu": the debayer or the merge of a mosaic (the CPU's for any other
  // decode), and the conversion to the output encoding.
  const char* full_size_decode = "cpu";
  const char* full_size_conversion = "cpu";

  // The scan's exposure: EXIF (of exposure_file, the first input), else LibRaw's, else 1/8 s at ISO 100 (warned once).
  std::string exposure_file;
  bool has_exif_exp = false;
  double exif_shutter_d = 0.0, exif_iso_d = 0.0;
  bool warned_default_exposure = false;
  void scan_exposure(const LibRaw& r, double& shutter_out, double& iso_out);

  // The full-resolution image the conversion runs on: rgb3 on the CFA path, LibRaw's image otherwise.
  RgbInput image();
};

// The conversion parameters: what solve decides, convert applies and write records.
struct Conversion {
  float exposure_comp = 1.0f, gain_g = 1.0f, gain_b = 1.0f;
  unsigned pin_mask = 0;
  float merged_matrix[9] = {0};
  float knee = 0.95f;
  ShoulderInfo shoulder;
  ParsedFilmProfile film_prof;
  std::vector<uint8_t> embedded_icc;     // the bracket's ICC profile
  std::string bundle_target_name;
  int bundle_target_idx = -1;
  DinoSolveResult dino_res;              // the single-profile search
  // With a bundle: E, g and b on the station's scale (what -E, --gain_g and --gain_b take), for cli_equivalent.
  std::optional<std::array<float, 3>> station_params;
  // The film base the sidecar records. A bundle conversion takes the base read from the scan (Scan::film_base), else
  // the bundle's, through the station's merged matrix; the film base scale of the input matrix stays 1.
  std::vector<int> profile_film_base_rgb = {1, 1, 1};
  std::vector<int> sidecar_profile_film_base_rgb;
  std::vector<int> sidecar_film_base_rgb = {1, 1, 1};
  std::string film_base_source = "none";
  OutputSpec out;
  DensityOutput density;                 // with Setup::density
};

int prepare(const Options& o, Setup& s);
int decode(const Options& o, Setup& s, Scan& scan);
// After the last pass of the search (after decode, or where --probe_intent stops): logs the GPU's passes and frees its
// frame copy, and fails when --check_renderer counts a pass outside the renderer or its comparison with the GPU is
// outside the tolerance, before anything is written.
int finish_search_passes(const Options& o, Setup& s);
int begin_conversion(const Options& o, const Setup& s, Conversion& c);
int render(const Options& o, const Setup& s, Scan& scan, Backend& backend);
int solve(const Options& o, Setup& s, Scan& scan, Conversion& c);
int convert(const Options& o, const Setup& s, Scan& scan, Conversion& c, Backend& backend);
int write(const Options& o, const Setup& s, Scan& scan, Conversion& c);

#endif  // NEGICC_H
