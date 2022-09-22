#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <algorithm>
#include <string>

#include "libraw/libraw.h"

#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 14))
#error This code is for LibRaw 0.14+ only
#endif

void print_pixel(LibRaw* proc, int row, int col) {
  printf("Pixel data at (%d %d) color = %d: %d\n", row, col, proc->COLOR(row, col),
	 proc->imgdata.image[row * proc->imgdata.sizes.iwidth + col][proc->COLOR(row, col)]);
}

LibRaw* load_raw(char* fn, bool debayer, bool half_size, bool write_tiff) {
  int ret;
  LibRaw* proc = new LibRaw();

  printf("Processing file %s\n", fn);
  if ((ret = proc->open_file(fn)) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot open %s: %s\n", fn, libraw_strerror(ret));
    return NULL;
  }
  printf("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);

  if ((ret = proc->unpack()) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot unpack %s: %s\n", fn, libraw_strerror(ret));
    return NULL;
  }
  if (!(proc->imgdata.idata.filters || proc->imgdata.idata.colors == 1)) {
    printf("Only Bayer-pattern RAW files supported, sorry....\n");
    return NULL;
  }

  proc->imgdata.params.output_bps = 16;
  proc->imgdata.params.user_flip = 0;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.no_auto_bright = 1;
  proc->imgdata.params.highlight = 0;
  proc->imgdata.params.output_color = 0;
  if (write_tiff)
    proc->imgdata.params.output_tiff = 1;
  if (!debayer) {
    proc->imgdata.params.no_interpolation = 1;
    proc->raw2image();
    proc->subtract_black();
  } else {
    proc->imgdata.params.half_size = half_size;
    proc->imgdata.params.highlight = 1;
    proc->imgdata.params.user_qual = 12;
    proc->imgdata.params.user_mul[0] = 1;
    proc->imgdata.params.user_mul[1] = 1;
    proc->imgdata.params.user_mul[2] = 1;
    proc->imgdata.params.user_mul[3] = 1;
    proc->dcraw_process();
  }
  return proc;
}

void post_process(LibRaw* proc, float* r_coef, float* g_coef, float* b_coef) {
  for (int i = 0; i < proc->imgdata.sizes.iwidth; ++i) {
    for (int j = 0; j < proc->imgdata.sizes.iheight; ++j) {
      int r = proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][0];
      int g = proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][1];
      int b = proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][2];
      float fr = r * r_coef[0] + g * r_coef[1] + b * r_coef[2] + 0.5f;
      float fg = r * g_coef[0] + g * g_coef[1] + b * g_coef[2] + 0.5f;
      float fb = r * b_coef[0] + g * b_coef[1] + b * b_coef[2] + 0.5f;
      r = std::min(65535, (int)std::max(.0f, fr));
      g = std::min(65535, (int)std::max(.0f, fg));
      b = std::min(65535, (int)std::max(.0f, fb));
      proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][0] = r;
      proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][1] = g;
      proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][2] = b;
    }
  }
}

int main(int ac, char *av[]) {
  int i, ret;
  char out_fn[1024];
  char prof_fn[1024];
  memset(prof_fn, 0, sizeof(prof_fn));

  float r_coef[3] = {1.0f,0,0};
  float g_coef[3] = {0,1.0f,0};
  float b_coef[3] = {0,0,1.0f};

  if (ac < 2) {
    usage:
      printf("neg_process - Process Negative with LibRAW %s\n"
	     "Usage: %s [-h] raw-files....\n"
             "  More than 4 raw-files supplied will be combined assuming a Sony 4-shot pixel sequence.\n"
	     "  -h: Half size.\n"
	     "  -r: R correction coefficients.\n"
	     "  -g: R correction coefficients.\n"
	     "  -b: R correction coefficients.\n"
             "  -p: ICC Profile to attach to TIFF file.\n"
	     "  -o: Output file location.\n",
	     LibRaw::version(), av[0]);
      return 0;
  }

  char* files[16];
  int fp = 0;
  bool half_size = false;
  bool write_tiff = false;
  for (i = 1; i < ac; i++) {
    if (av[i][0] == '-') {
      switch (av[i][1]) {
      case 'h':
	half_size = true;
	break;
      case 'o':
	strncpy(out_fn, av[i+1], strlen(av[i+1]));
	++i;
	break;
      case 'r':
	sscanf(av[i+1],"%f %f %f", r_coef, r_coef+1, r_coef+2);
	++i;
	break;
      case 'g':
	sscanf(av[i+1],"%f %f %f", g_coef, g_coef+1, g_coef+2);
	++i;
	break;
      case 'b':
	sscanf(av[i+1],"%f %f %f", b_coef, b_coef+1, b_coef+2);
	++i;
	break;
      case 'p':
	strncpy(prof_fn, av[i+1], strlen(av[i+1]));
	write_tiff = true;
	++i;
	break;
      default:
	goto usage;
	continue;
      }
    } else {
      files[fp++] = av[i];
    }
  }

  LibRaw *proc[4];
  if (fp == 4) {
    for (int i = 0; i < 4; ++i) {
      proc[i] = load_raw(files[i], false, false, write_tiff);
    }
    printf("Merging 4 images...\n");

    int movements[4][2] = {
      // x, y movements
      {0, 0},
      {0, 1},
      {-1, 1},
      {-1, 0},
    };

#define P(n, r, c) proc[n]->imgdata.image[(r) * proc[n]->imgdata.sizes.iwidth + (c)]
#define FOR_PIXEL for (int r = 0; r < proc[mi]->imgdata.sizes.iheight - 1; ++r) \
      for (int c = 1; c < proc[mi]->imgdata.sizes.iwidth; ++c)

    for (int mi = 0; mi < 2; ++mi) {
      int dc = movements[mi][0];
      int dr = movements[mi][1];

      FOR_PIXEL {
	int col = proc[mi]->COLOR(r, c);
	if (col & 1)
	  P(0, r+dr, c+dc)[1] = P(mi, r, c)[col];
	else
	  P(0, r+dr, c+dc)[col] = P(mi, r, c)[col];
      }
      if (mi > 0) {
	proc[mi]->recycle();
	delete proc[mi];
      }
    }

    for (int mi = 2; mi < 4; ++mi) {
      int dc = movements[mi][0];
      int dr = movements[mi][1];

      FOR_PIXEL {
	int col = proc[mi]->COLOR(r, c);
	if (col & 1)
	  P(0, r+dr, c+dc)[1] = (P(mi, r, c)[col] + P(0, r+dr, c+dc)[1]) / 2;
	else
	  P(0, r+dr, c+dc)[col] = P(mi, r, c)[col];
      }
      proc[mi]->recycle();
      delete proc[mi];
    }
    proc[0]->imgdata.idata.colors = 3;
    for (int i = 0; i < 3; ++i) {
      int scale_factor = 4;
      r_coef[i] *= scale_factor;
      g_coef[i] *= scale_factor;
      b_coef[i] *= scale_factor;
    }
  } else {
    proc[0] = load_raw(files[0], true, half_size, write_tiff);
  }
  printf("R coefficients: %1.5f %1.5f %1.5f\n", r_coef[0], r_coef[1], r_coef[2]);
  printf("G coefficients: %1.5f %1.5f %1.5f\n", g_coef[0], g_coef[1], g_coef[2]);
  printf("B coefficients: %1.5f %1.5f %1.5f\n", b_coef[0], b_coef[1], b_coef[2]);
  post_process(proc[0], r_coef, g_coef, b_coef);

  // This is a hack to force LibRaw write the ICC profile in the TIFF.
  if (prof_fn[0]) {
    printf("Attaching profile: %s\n", prof_fn);
    FILE *fp = fopen(prof_fn, "rb");
    unsigned size;
    fread(&size, 4, 1, fp);
    fseek(fp, 0, SEEK_SET);
    proc[0]->get_internal_data_pointer()->output_data.oprof = (unsigned *)malloc(size = ntohl(size));
    fread(proc[0]->get_internal_data_pointer()->output_data.oprof, 1, size, fp);
    fclose(fp);
  }
  printf("Writing TIFF %s\n", out_fn);
  proc[0]->dcraw_ppm_tiff_writer(out_fn);
  return 0;
}
