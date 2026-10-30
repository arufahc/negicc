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

#include "np_options.h"
#include "np_pipeline.h"

#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 21))
#error This code requires LibRaw 0.21+ (vendored in 3rd_party/libraw)
#endif

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
// - A linear (demosaiced) DNG, e.g. a film scanner's, decoded by LibRaw as a RAW is: camera RGB and the bundle's
//   crosstalk correction for a tri bundle, the DNG's colour data to linear Rec.2020 for a d65 bundle. A fourth
//   sample per pixel (a scanner's infrared) is left out.
// - An RGB TIFF (8 or 16 bits) or JPEG, with a d65 bundle: converted to linear Rec.2020 through its embedded ICC
//   profile (rgb_input.h); without one, 8-bit data is taken as sRGB and 16-bit data as linear Rec.2020.
// - Three images, each captured under one of the red, green and blue lights (RAWs, DNGs or TIFFs, the same size):
//     neg_process --film_stock portra400 --light_source tri -o out.tif red.ARW green.ARW blue.ARW
//   R comes from the first, G from the second and B from the third (a grey TIFF gives its one channel), each scaled
//   to the first capture's exposure when all three record theirs. There is no crosstalk between the lights to
//   correct, so the correction is the identity (-r, -g and -b give another). --flat applies to RAWs only.
//
// The parameter search (the calibration bracket, the exposure and the gains, solved with DINOv3 intent) runs
// whenever a profile is given, unless --gain_g or --gain_b is; then the conversion takes the values as given. -E
// keeps the exposure as given and solves only the gains, on the bracket --target picks (or the bundle's default).
// TODO: the bracket search on the frame --roi auto finds; for now it needs the frame given with --roi.
//
// --roi auto [h|v] finds the frame to crop (film_frame_detector.h): 35mm, 645, 6x6, 6x7 or 6x9 film on bare light, in
// a strip holder or in a frame holder, tilted up to 5 degrees, partial neighbour frames beside it. --film_format gives
// the format (more accurate than trying each); h or v gives the strip direction. The detection reads the RAW's CFA
// mosaic in place before anything is decoded, and only the frame is decoded, e.g.
//     neg_process --film_stock portra400 --light_source tri --target 0 --roi auto --film_format 6x7 -o out.tif scan.ARW
int main(int ac, char* av[]) {
  Options o;
  if (const int r = parse_options(ac, av, o)) return r < 0 ? 0 : r;
  Setup s;
  if (prepare(o, s)) return 1;
  Scan scan;
  if (decode(o, s, scan)) return 1;
  Conversion c;
  if (begin_conversion(o, s, c)) return 1;
  CpuBackend backend;
  if (render(o, s, scan, backend)) return 1;
  if (solve(o, s, scan, c)) return 1;
  if (convert(o, s, scan, c, backend)) return 1;
  return write(o, s, scan, c);
}
