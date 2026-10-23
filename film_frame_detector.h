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

// LibRaw-free film frame detector for camera-scanned (or flatbed-scanned) film, in negative colour.
//
// Finds the picture area of the frame to crop (the "primary" frame), the other frames visible on the strip
// (neighbours and partial frames, the "secondaries"), the visible film strip (picture plus rebate, the "film base"
// rectangle), the 35mm sprocket holes, the film's tilt, the strip direction and the frame orientation, and a flat
// patch of unexposed film base for --film_base_rgb auto. 35mm, 645, 6x6, 6x7 and 6x9 frames, given by the caller or
// tried in turn (FILM_FORMAT_AUTO). Film on bare light without a holder, in a strip holder or in a frame holder.
//
// Input: the image is read in place through FilmImage, three channel lattices that may point into one CFA mosaic
// (one sample per photosite) or into an RGB image (interleaved or planar). Nothing of the input is copied; the
// detector reads it in two passes: a box-averaged downsample to a work image of at most params.work_size pixels on
// the long side, and a few thousand bilinear samples along the frame edges and in the film base patch. All
// intermediate buffers are work-image sized (see film_frame_detector.cc for each one).
//
// Output coordinates are image pixels: the pixel grid FilmImage::width x FilmImage::height (the sensor for a CFA
// mosaic, the decoded image for an RGB image), x right and y down, pixel (i, j) covering [i, i + 1) x [j, j + 1).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// One colour channel of the input: a regular lattice of uint16_t samples, read in place. Sample (i, j) of the
// channel is data[i * pitch + j * stride] and lies on image pixel (x0 + i * FilmImage::step, y0 + j * step). Its
// value is (sample - black) / white, clamped below at 0.
struct FilmChannel {
  const uint16_t* data = nullptr;
  ptrdiff_t pitch = 1;   // uint16_t elements from sample (i, j) to (i + 1, j)
  ptrdiff_t stride = 0;  // uint16_t elements from sample (i, j) to (i, j + 1)
  int x0 = 0;            // image pixel of sample (0, 0), 0 <= x0, y0 < step
  int y0 = 0;
  float black = 0.0f;    // black level of this channel, subtracted from every sample
};

// The image the detector reads: R, G and B as three lattices of the same step. For an RGB image step is 1 and the
// three channels share the pixel grid; for a CFA mosaic step is the CFA period (2 Bayer, 6 X-Trans) and each channel
// takes one photosite of its colour per period x period cell (see film_image_from_cfa). Each channel has
// samples_x() x samples_y() samples.
struct FilmImage {
  int width = 0;         // image size in pixels
  int height = 0;
  int step = 1;          // image pixels between neighbouring samples of a channel
  float white = 65535.0f;  // sample value (black subtracted) of a clipped photosite: the normalising scale
  FilmChannel ch[3];     // R, G, B
  int samples_x() const;
  int samples_y() const;
};

// An interleaved RGB image of |channels| (3 or 4, R G B first) uint16_t per pixel, |row_stride| uint16_t elements
// per row (0: width * channels); e.g. LibRaw's image[][4] after dcraw_process or a pixel-shift merge.
FilmImage film_image_from_interleaved(const uint16_t* pixels, int width, int height, int channels,
                                      ptrdiff_t row_stride = 0, float white = 65535.0f);

// Three planes (or three times the same plane for a monochrome image), |row_stride| elements per row each.
FilmImage film_image_from_planes(const uint16_t* r, const uint16_t* g, const uint16_t* b, int width, int height,
                                 ptrdiff_t row_stride, float white = 65535.0f);

// A CFA mosaic: one uint16_t per photosite, |row_stride| elements per row, pixel (0, 0) at |mosaic| (for LibRaw's
// rawdata.raw_image that is the first visible photosite: raw_image + top_margin * raw_pitch / 2 + left_margin).
// |period| is the CFA repeat (2 Bayer, 6 X-Trans) and |color| gives the colour of photosite (row, col) as LibRaw's
// COLOR() does (0 R, 1 G, 2 B, 3 the second G of a Bayer quad). For each of R, G and B the first photosite of that
// colour in the period x period cell (row-major, G before G2) becomes the channel lattice, with
// black[color] as its black level. False when a colour is missing from the cell.
bool film_image_from_cfa(const uint16_t* mosaic, int width, int height, ptrdiff_t row_stride, int period,
                         const std::function<int(int row, int col)>& color, const float black[4], float white,
                         FilmImage& out);

enum FilmFormat { FILM_FORMAT_AUTO = 0, FILM_35MM, FILM_645, FILM_6X6, FILM_6X7, FILM_6X9 };
enum FilmPolarity { FILM_NEGATIVE = 0, FILM_POSITIVE = 1 };

const char* film_format_name(FilmFormat f);
// "35mm" (or "135"), "645", "6x6", "6x7", "6x9"; anything else is FILM_FORMAT_AUTO.
FilmFormat film_format_from_name(const std::string& s);

struct FrameDetectionParams {
  FilmFormat format = FILM_FORMAT_AUTO;  // set by the user for the best accuracy; AUTO tries every format
  FilmPolarity polarity = FILM_NEGATIVE;
  char axis_hint = 0;            // 'h' the strip runs along x, 'v' along y, 0 detect
  float max_rotation_deg = 5.0f; // tilt search range
  int work_size = 1024;          // long side of the work image, pixels
  bool refine = true;            // sub-pixel refinement of the primary frame's edges on the full-resolution image
  float crop_inset = 0.005f;     // FrameDetection::crop: inset per side, as a fraction of the frame size
  int crop_align = 1;            // FrameDetection::crop: x, y, x + w and y + h multiples of this (the CFA period)
};

// A rectangle turned clockwise by angle_deg about its centre: its corners are
//   (cx, cy) + R(angle) (+-w / 2, +-h / 2),  R = [[cos, -sin], [sin, cos]]
// (y down, so a positive angle turns it clockwise on screen). w is its extent along x before the turn.
struct FilmRect {
  float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f, angle_deg = 0.0f;
  void corners(float xy[8]) const;  // x0 y0 x1 y1 ..., top-left, top-right, bottom-right, bottom-left before the turn
};

struct FilmFrame {
  FilmRect rect;                 // the whole frame (a partial frame reaches past the image border)
  int visible[4] = {0, 0, 0, 0}; // x, y, w, h: bounding box of the part inside the image
  float visible_fraction = 0.0f; // visible picture area / nominal frame area
  bool complete = false;         // the whole frame is inside the image
  int index = 0;                 // position along the strip from the primary (..., -1, 0, +1, ...)
};

struct FrameDetection {
  bool ok = false;
  FilmFrame primary;                  // the frame to crop
  std::vector<FilmFrame> secondaries; // the other frames on the strip, ordered by index
  // The crop: x, y, w, h of the largest axis-aligned rectangle with the primary frame's aspect inside it (the
  // frame is tilted by rotation_deg), inset by params.crop_inset and aligned to params.crop_align.
  int crop[4] = {0, 0, 0, 0};
  bool has_film_base = false;         // the strip extends past the primary frame
  FilmRect film_base;                 // the visible strip: picture and rebate
  std::vector<FilmRect> sprockets;    // 35mm perforations found beside the primary frame
  float rotation_deg = 0.0f;          // tilt of the film (FilmRect convention)
  char axis = '?';                    // 'h' strip along x, 'v' along y, '?' a single square frame, undecided
  std::string orientation;            // "landscape", "portrait" or "square": the primary frame on the image
  FilmFormat format = FILM_FORMAT_AUTO;
  float px_per_mm = 0.0f;             // image pixels per mm of film
  // The film gaps at the primary frame's two ends (before, after it along the strip): their extent along the strip
  // axis in image pixels (x for axis 'h', y for 'v'), within [0, width or height], {-1, -1} where the end is
  // bounded otherwise or open. n_gaps counts the ends that have one.
  int n_gaps = 0;
  int gaps[2][2] = {{-1, -1}, {-1, -1}};
  // A flat patch of unexposed film next to the primary frame (an inter-frame gap or the rebate, tilted with the
  // film), the bounding box (x, y, w, h) of its part inside the image, its median per channel in sample units
  // (black subtracted, not normalised) and the spread of its green density (standard deviation of log10, as the
  // previous detector). has_base when that spread is at most 0.025.
  bool has_base = false;
  FilmRect base_patch;
  int base_box[4] = {0, 0, 0, 0};
  float base_rgb[3] = {0.0f, 0.0f, 0.0f};
  float base_std_d = 0.0f;
  bool light_clipped = false;         // bare light around the film is clipped
  float conf = 0.0f;                  // 1: the frame is complete and bounded by gaps or holder on both ends
  std::string reason;                 // comma-separated diagnostics ("" when nothing to report)
};

// Detects the film frame in |image|. False when the image is too small or holds no frame.
bool detect_film_frame(const FilmImage& image, const FrameDetectionParams& params, FrameDetection& out);

// Median per channel (sample units, black subtracted) of the samples whose photosite centre lies inside |patch|
// (image pixels), and the standard deviation of their green log10 density. False when it holds fewer than 32
// samples of a channel. Used for the base patch on the image the caller converts (e.g. the decoded crop in its own
// colour space; shift the patch's centre by the crop's origin, and halve it for a half-size decode).
bool film_sample_patch(const FilmImage& image, const FilmRect& patch, float rgb[3], float* std_d);
