#include <stdio.h>
#include <string.h>
#include <time.h>

#include "libraw/libraw.h"

enum MODE {
  SHUTTER_SPEED = 0,
  ISO,
};

int main(int ac, char *av[]) {
  int i, ret;
  char out_fn[1024];
  char prof_fn[1024];

  if (ac < 2) {
    usage:
      printf("raw_info - Show RAW info using libRAW %s\n"
	     "Usage: %s [options] raw-files....\n"
	     "  -s: Shutter speed.\n"
	     "  -i: ISO.\n",
	     LibRaw::version(), av[0]);
      return 0;
  }

  char* files[16];
  int fp = 0;
  MODE mode;
  for (i = 1; i < ac; i++) {
    if (av[i][0] == '-') {
      switch (av[i][1]) {
      case 's': mode = SHUTTER_SPEED; break;
      case 'i': mode = ISO; break;
      default:
	goto usage;
	continue;
      }
    } else {
      files[fp++] = av[i];
    }
  }

  for (i = 0; i < fp; ++i) {
    int ret;
    LibRaw* proc = new LibRaw();
    char* fn = files[i];
    if ((ret = proc->open_file(fn)) != LIBRAW_SUCCESS) {
      fprintf(stderr, "Cannot open %s: %s\n", fn, libraw_strerror(ret));
      return 1;
    }
    switch (mode) {
    case SHUTTER_SPEED:
      printf("%f\n", proc->imgdata.other.shutter);
      break;
    case ISO:
      printf("%f\n", proc->imgdata.other.iso_speed);
      break;
    }
  }
  return 0;
}
