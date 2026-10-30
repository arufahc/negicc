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

// neg_process's image geometry: rotation, cropping and turning a decoded LibRaw image, rectangles, and the frame
// detection fallback of --roi auto.

#pragma once

#include "film_frame_detector.h"
#include "libraw/libraw.h"

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
