#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "libraw/libraw.h"

enum MODE {
  NONE = 0,
  SHUTTER_SPEED = 1,
  ISO = 2,
  CENTER_WEIGHT_AVERAGE = 4,
};

void print_center_weight_averages(ushort (*image)[4], ushort width, ushort height) {
  uint64_t total_r = 0;
  uint64_t total_g = 0;
  uint64_t total_b = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      total_r += image[j * width + i][0];
      total_g += image[j * width + i][1];
      total_b += image[j * width + i][2];
    }
  }
  uint32_t pixels = width * height;
  ushort avg_r = total_r / pixels;
  ushort avg_g = total_g / pixels;
  ushort avg_b = total_b / pixels;
  printf("%d %d %d # Center-weight average RGB\n", avg_r, avg_g, avg_b);
  total_r = total_g = total_b = 0;
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      total_r += (image[j * width + i][0] - avg_r) * (image[j * width + i][0] - avg_r);
      total_g += (image[j * width + i][1] - avg_g) * (image[j * width + i][1] - avg_g);
      total_b += (image[j * width + i][2] - avg_b) * (image[j * width + i][2] - avg_b);
    }
  }
  printf("%f %f %f # Center-weight RGB stddev\n", sqrt(total_r / pixels), sqrt(total_g / pixels), sqrt(total_b / pixels));
}

int main(int ac, char *av[]) {
  if (ac < 2) {
    usage:
      printf("raw_info - Show RAW info using libRAW %s\n"
	     "Usage: %s [options] raw-files....\n"
	     "  -s: Shutter speed.\n"
	     "  -i: ISO.\n"
	     "  -w: Compute center-weight average.\n",
	     LibRaw::version(), av[0]);
      return 0;
  }

  char* files[16];
  int fp = 0;
  int mode = 0;
  for (int i = 1; i < ac; i++) {
    if (av[i][0] == '-') {
      switch (av[i][1]) {
      case 's': mode |= SHUTTER_SPEED; break;
      case 'i': mode |= ISO; break;
      case 'w': mode |= CENTER_WEIGHT_AVERAGE; break;
      default:
	goto usage;
	continue;
      }
    } else {
      files[fp++] = av[i];
    }
  }

  for (int i = 0; i < fp; ++i) {
    int ret;
    LibRaw* proc = new LibRaw();
    char* fn = files[i];
    if ((ret = proc->open_file(fn)) != LIBRAW_SUCCESS) {
      fprintf(stderr, "Cannot open %s: %s\n", fn, libraw_strerror(ret));
      return 1;
    }
    if (mode & SHUTTER_SPEED) {
      printf("%f # Shutter speed\n", proc->imgdata.other.shutter);
    }
    if (mode & ISO) {
      printf("%f # ISO speed\n", proc->imgdata.other.iso_speed);
    }
    if (mode & CENTER_WEIGHT_AVERAGE) {
      if ((ret = proc->unpack()) != LIBRAW_SUCCESS) {
        fprintf(stderr, "Cannot unpack %s: %s\n", fn, libraw_strerror(ret));
        return -1;
      }
      // Params needed to perform half size linear conversion.
      proc->imgdata.params.output_bps = 16;
      proc->imgdata.params.user_flip = 0;
      proc->imgdata.params.gamm[0] = 1;
      proc->imgdata.params.gamm[1] = 1;
      proc->imgdata.params.no_auto_bright = 1;
      proc->imgdata.params.highlight = 0;
      proc->imgdata.params.output_color = 0;
      proc->imgdata.params.half_size = 1;
      proc->imgdata.params.highlight = 1;
      proc->imgdata.params.use_auto_wb = 0;
      proc->imgdata.params.user_mul[0] = 1;
      proc->imgdata.params.user_mul[1] = 1;
      proc->imgdata.params.user_mul[2] = 1;
      proc->imgdata.params.user_mul[3] = 1;
      ushort crop_width = std::min(proc->imgdata.sizes.width, proc->imgdata.sizes.height) / 4;
      proc->imgdata.params.cropbox[0] = (proc->imgdata.sizes.width - crop_width) / 2;
      proc->imgdata.params.cropbox[1] = (proc->imgdata.sizes.height - crop_width) / 2;
      proc->imgdata.params.cropbox[2] = crop_width;
      proc->imgdata.params.cropbox[3] = crop_width;
      proc->dcraw_process();
      print_center_weight_averages(proc->imgdata.image, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
    }
  }
  return 0;
}
