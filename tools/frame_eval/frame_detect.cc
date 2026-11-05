// Detection only: the film frame detector on each input given, read as neg_process reads it for --roi auto, one JSON
// line each on stdout (neg_process's own progress messages go to stderr): a CFA RAW or DNG on its mosaic and a linear
// DNG on its demosaiced pixels (film_image_from_raw), a TIFF or JPEG through load_rgb_input as white light (d65).
// usage: frame_detect [--film_format 35mm|645|6x6|6x7|6x9|auto] input...
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "film_frame_detector.h"
#include "libraw/libraw.h"
#include "np_geometry.h"
#include "np_input.h"
#include "np_raw.h"
#include "rgb_input.h"

int main(int argc, char** argv) {
  // The JSON lines keep stdout; the loaders' progress messages (log_info, written to stdout) go to stderr.
  std::fflush(stdout);
  FILE* out = fdopen(dup(STDOUT_FILENO), "w");
  if (!out || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) return 1;
  FilmFormat fmt = FILM_35MM;
  int i = 1;
  if (argc > 2 && !std::strcmp(argv[1], "--film_format")) {
    fmt = std::strcmp(argv[2], "auto") ? film_format_from_name(argv[2]) : FILM_FORMAT_AUTO;
    i = 3;
  }
  for (; i < argc; ++i) {
    FrameDetectionParams p;
    p.format = fmt;
    FrameDetection d;
    int sensor_w = 0, sensor_h = 0;
    // As neg_process picks the input mode (np_options.cc): a JPEG, or a file LibRaw does not open, is an RGB file.
    if (negicc::is_jpeg_file(argv[i]) || !libraw_opens(argv[i])) {
      std::string source;
      std::unique_ptr<LibRaw> proc(load_rgb_input({argv[i]}, InputMode::kRgbFile, false, 0, false, nullptr, {}, &d, p,
                                                  source));
      if (!proc) {
        std::fprintf(out, "{\"file\": \"%s\", \"error\": \"open\"}\n", argv[i]);
        std::fflush(out);
        continue;
      }
      // load_rgb_input keeps only the detected frame; the image's own size stays in rawdata.sizes.
      sensor_w = proc->imgdata.rawdata.sizes.width;
      sensor_h = proc->imgdata.rawdata.sizes.height;
      proc->free_image();
    } else {
      LibRaw raw;
      if (open_and_unpack_raw_file(&raw, argv[i]) != LIBRAW_SUCCESS) {
        std::fprintf(out, "{\"file\": \"%s\", \"error\": \"open\"}\n", argv[i]);
        std::fflush(out);
        continue;
      }
      FilmImage mosaic;
      if (!film_image_from_raw(raw, mosaic)) {
        std::fprintf(out, "{\"file\": \"%s\", \"error\": \"mosaic\"}\n", argv[i]);
        std::fflush(out);
        continue;
      }
      p.crop_align = mosaic.step;
      detect_frame_or_fallback(mosaic, p, d);
      sensor_w = mosaic.width;
      sensor_h = mosaic.height;
    }
    std::fprintf(out,
                 "{\"file\": \"%s\", \"ok\": %s, \"conf\": %.3f, \"axis\": \"%c\", \"format\": \"%s\", \"crop\": [%d, "
                 "%d, %d, %d], \"n_gaps\": %d, \"rotation\": %.3f, \"px_per_mm\": %.2f, \"sensor\": [%d, %d], "
                 "\"reason\": \"%s\"}\n",
                 argv[i], d.ok ? "true" : "false", d.conf, d.axis ? d.axis : '-', film_format_name(d.format), d.crop[0],
                 d.crop[1], d.crop[2], d.crop[3], d.n_gaps, d.rotation_deg, d.px_per_mm, sensor_w, sensor_h,
                 d.reason.c_str());
    std::fflush(out);
  }
  return 0;
}
