// Detection only: the film frame detector (as neg_process --roi auto runs it) on each RAW given, one JSON line each.
// usage: frame_detect [--film_format 35mm|645|6x6|6x7|6x9|auto] raw...
#include <cstdio>
#include <cstring>
#include <string>

#include "film_frame_detector.h"
#include "libraw/libraw.h"
#include "np_geometry.h"
#include "np_raw.h"

int main(int argc, char** argv) {
  FilmFormat fmt = FILM_35MM;
  int i = 1;
  if (argc > 2 && !std::strcmp(argv[1], "--film_format")) {
    fmt = std::strcmp(argv[2], "auto") ? film_format_from_name(argv[2]) : FILM_FORMAT_AUTO;
    i = 3;
  }
  for (; i < argc; ++i) {
    LibRaw raw;
    if (raw.open_file(argv[i]) != LIBRAW_SUCCESS || raw.unpack() != LIBRAW_SUCCESS) {
      std::printf("{\"file\": \"%s\", \"error\": \"open\"}\n", argv[i]);
      continue;
    }
    FilmImage mosaic;
    if (!film_image_from_raw(raw, mosaic)) {
      std::printf("{\"file\": \"%s\", \"error\": \"mosaic\"}\n", argv[i]);
      continue;
    }
    FrameDetectionParams p;
    p.format = fmt;
    p.crop_align = mosaic.step;
    FrameDetection d;
    detect_frame_or_fallback(mosaic, p, d);
    std::printf("{\"file\": \"%s\", \"ok\": %s, \"conf\": %.3f, \"axis\": \"%c\", \"format\": \"%s\", \"crop\": [%d, %d, %d, "
                "%d], \"n_gaps\": %d, \"rotation\": %.3f, \"px_per_mm\": %.2f, \"sensor\": [%d, %d], \"reason\": \"%s\"}\n",
                argv[i], d.ok ? "true" : "false", d.conf, d.axis ? d.axis : '-', film_format_name(d.format), d.crop[0],
                d.crop[1], d.crop[2], d.crop[3], d.n_gaps, d.rotation_deg, d.px_per_mm, mosaic.width, mosaic.height,
                d.reason.c_str());
    std::fflush(stdout);
  }
  return 0;
}
