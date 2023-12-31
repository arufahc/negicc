#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <string>

#include <netinet/in.h>
#include <lcms2.h>

#include "libraw/libraw.h"


#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 14))
#error This code is for LibRaw 0.14+ only
#endif

void print_pixel(LibRaw* proc, int row, int col) {
  printf("Pixel data at (%d %d) color = %d: %d\n", row, col, proc->COLOR(row, col),
	 proc->imgdata.image[row * proc->imgdata.sizes.iwidth + col][proc->COLOR(row, col)]);
}

LibRaw* load_raw(char* fn, bool debayer, bool half_size, int qual, bool crop) {
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
  proc->imgdata.params.output_tiff = 1;
  if (!debayer) {
    proc->imgdata.params.no_interpolation = 1;
    proc->raw2image();
    proc->subtract_black();
  } else {
    proc->imgdata.params.half_size = half_size;
    proc->imgdata.params.highlight = 1;
    proc->imgdata.params.user_qual = qual;
    proc->imgdata.params.use_auto_wb = 0;
    proc->imgdata.params.user_mul[0] = 1;
    proc->imgdata.params.user_mul[1] = 1;
    proc->imgdata.params.user_mul[2] = 1;
    proc->imgdata.params.user_mul[3] = 1;
    if (crop &&
        (proc->imgdata.sizes.raw_inset_crops[0].cleft || proc->imgdata.sizes.raw_inset_crops[0].ctop)) {
      proc->imgdata.params.cropbox[0] = proc->imgdata.sizes.raw_inset_crops[0].cleft;
      proc->imgdata.params.cropbox[1] = proc->imgdata.sizes.raw_inset_crops[0].ctop;
      proc->imgdata.params.cropbox[2] = proc->imgdata.sizes.raw_inset_crops[0].cwidth;
      proc->imgdata.params.cropbox[3] = proc->imgdata.sizes.raw_inset_crops[0].cheight;
    }
    proc->dcraw_process();
  }
  return proc;
}

void post_process(LibRaw* proc, float* r_coef, float* g_coef, float* b_coef) {
  for (int j = 0; j < proc->imgdata.sizes.iheight; ++j) {
    for (int i = 0; i < proc->imgdata.sizes.iwidth; ++i) {
      ushort r = proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][0];
      ushort g = proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][1];
      ushort b = proc->imgdata.image[j * proc->imgdata.sizes.iwidth + i][2];
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

int read_profile(const char* prof_name, unsigned **prof_out, unsigned *size) {
  FILE *fp;
  if ((fp = fopen(prof_name, "rb"))) {
    fread(size, 4, 1, fp);
    fseek(fp, 0, SEEK_SET);
    *prof_out = (unsigned *)malloc(*size = ntohl(*size));
    fread(*prof_out, 1, *size, fp);
    fclose(fp);
    return 0;
  }
  return -1;
}

int apply_profile(ushort (*image)[4], ushort width, ushort height, const char *input, const char *output) {
  cmsHPROFILE in_profile = 0, out_profile = 0;
  cmsHTRANSFORM transform;
  unsigned *prof, *oprof;
  unsigned size;

  if (!read_profile(input, &prof, &size)) {
    printf("Reading input ICC profile: %s\n", input);
    if (!(in_profile = cmsOpenProfileFromMem(prof, size))) {
      free(prof);
      prof = 0;
      return -1;
    }
  } else {
    return -1;
  }

  if (!strcmp(output, "srgb")) {
    printf("Creating standard sRGB profile\n");
    out_profile = cmsCreate_sRGBProfile();
  } else if (!read_profile(output, &oprof, &size)) {
    printf("Reading output ICC profile: %s\n", output);
    if (!(out_profile = cmsOpenProfileFromMem(oprof, size))) {
      free(oprof);
      oprof = 0;
      return -1;
    }
  } else {
    return -1;
  }
  transform = cmsCreateTransform(in_profile, TYPE_RGBA_16, out_profile,
                                  TYPE_RGBA_16, INTENT_PERCEPTUAL, 0);
  cmsDoTransform(transform, image, image, width * height);
  cmsDeleteTransform(transform);
  cmsCloseProfile(out_profile);
 quit:
  cmsCloseProfile(in_profile);
  return 0;
}

int main(int ac, char *av[]) {
  int i, ret;
  char out_fn[1024];
  char in_prof_fn[1024];
  char out_prof_fn[1024];
  memset(in_prof_fn, 0, sizeof(in_prof_fn));
  memset(out_prof_fn, 0, sizeof(out_prof_fn));

  float r_coef[4] = {1.0f,0,0,0.0};
  float g_coef[4] = {0,1.0f,0,0.0};
  float b_coef[4] = {0,0,1.0f,0.0};

  if (ac < 2) {
    usage:
    printf("neg_process - Process Negative with LibRAW %s\n"
           "Usage: %s [options] raw-files....\n"
           "  More than 4 raw-files supplied will be combined assuming a Sony 4-shot pixel sequence.\n"
           "  -h: Half size.\n"
           "  -C: No cropping based on raw aspect ratio.\n"
           "  -q: Quality.\n"
           "  -r: R (corrected) value is dot product of this 'r1 r2 r3' vector and 'R G B' values from linear RAW.\n"
           "  -g: G (corrected) value is dot product of this 'g1 g2 g3' vector and 'R G B' values from linear RAW.\n"
           "  -b: B (corrected) value is dot product of this 'b1 b2 b3' vector and 'R G B' values from linear RAW.\n"
           "  -p: ICC Profile that applies to the corrected RGB values (See -r -g and -b flags). Consider this as the input ICC profile.\n"
           "  -P: srgb or [ICC profile path]. If specified the corrected RGB will be converted using this as the output profile.\n"
           "  -o: Output file location.\n",
           LibRaw::version(), av[0]);
    return 0;
  }

  char* files[16];
  int fp = 0;
  bool half_size = false;
  bool crop = true;
  int qual = 0;
  for (i = 1; i < ac; i++) {
    if (av[i][0] == '-') {
      switch (av[i][1]) {
      case 'h':
        half_size = true;
        break;
      case 'C':
        crop = false;
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
      case 'q':
        sscanf(av[i+1], "%d", &qual);
        ++i;
        break;
      case 'p':
        strncpy(in_prof_fn, av[i+1], strlen(av[i+1]));
        ++i;
        break;
      case 'P':
        strncpy(out_prof_fn, av[i+1], strlen(av[i+1]));
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
      proc[i] = load_raw(
          files[i], false, false,
          /* Quality doesn't matter because no interpolation. */
          0,
          /* Don't crop since it might mess up pixel-shift merging. */
          false);
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
      // Operating on data without interpolation and for Sony A7RM4 sensor
      // the input is 14-bit and multiply by 4 to expand into 16-bit.
      int scale_factor = 4;
      r_coef[i] *= scale_factor;
      g_coef[i] *= scale_factor;
      b_coef[i] *= scale_factor;
    }
  } else {
    proc[0] = load_raw(files[0], true, half_size, qual, crop);
  }
  printf("ISO Speed: %f\n", proc[0]->imgdata.other.iso_speed);
  printf("Shutter %f\n", proc[0]->imgdata.other.shutter);
  printf("R coefficients: %1.5f %1.5f %1.5f\n", r_coef[0], r_coef[1], r_coef[2]);
  printf("G coefficients: %1.5f %1.5f %1.5f\n", g_coef[0], g_coef[1], g_coef[2]);
  printf("B coefficients: %1.5f %1.5f %1.5f\n", b_coef[0], b_coef[1], b_coef[2]);
  post_process(proc[0], r_coef, g_coef, b_coef);

  // By default attach the input profile to the output file only, no conversion.
  char* attach_prof_fn = in_prof_fn;
  if (in_prof_fn[0] && out_prof_fn[0]) {
    apply_profile(proc[0]->imgdata.image, proc[0]->imgdata.sizes.iwidth, proc[0]->imgdata.sizes.iheight, in_prof_fn, out_prof_fn);
    // Attach the output profile since conversion has applied.
    attach_prof_fn = out_prof_fn;
  }

  if (attach_prof_fn[0] && strcmp(attach_prof_fn, "srgb")) {
    // If the profile to attach is not the psuedo "srgb" profile, the profile will be attached to the TIFF.
    // This is a hack to force LibRaw write the ICC profile in the TIFF without conversion.
    printf("Attaching profile: %s\n", attach_prof_fn);
    unsigned size;
    if (read_profile(attach_prof_fn, &proc[0]->get_internal_data_pointer()->output_data.oprof, &size)) {
      printf("Cannot read profile.\n");
      return -1;
    }
  }
  printf("Writing TIFF '%s'\n", out_fn);
  proc[0]->dcraw_ppm_tiff_writer(out_fn);
  return 0;
}
