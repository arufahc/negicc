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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <lcms2.h>

#include <algorithm>
#include <string>
#include <vector>

#include <netinet/in.h>

#include "argparse/argparse.hpp"
#include "libraw/libraw.h"

#if !(LIBRAW_COMPILE_CHECK_VERSION_NOTLESS(0, 14))
#error This code is for LibRaw 0.14+ only
#endif

void debug_pixel(LibRaw* proc, int row, int col) {
  printf("Pixel data at (%d %d) color = %d: %d\n", row, col, proc->COLOR(row, col),
         proc->imgdata.image[row * proc->imgdata.sizes.iwidth + col][proc->COLOR(row, col)]);
}

// Load a RAW file and decode it into linear values.
//
// If |debayer| is true, interpolation is performed to generate missing pixels
// from the color filter array (usually a bayer pattern). Otherwise, the linear
// RGB values will have missing pixels and will not be scaled to 16-bit. If the
// sensor produces 14-bit files, then a scale factor of 4 needs to be applied,
// this is needed only when merging pixel-shift images.
//
// |qual| chooses the debayer algorithm used. 0 is the faster and is bilinear.
// Since the RAW capture is supposed to be linear to dye densities, which are
// supposed to be independent and have different grain structures, interpolation
// of the RAW file often produces artifacts that accentuates visible grain. This
// is caused by the debayer algorithm reading pixels from red channel (more
// grain) to generate pixels for blue and green channels (less grain). qual = 0
// is preferred or use pixel shift to eliminate need for interpolation.
//
// When |crop| is false, the entire RAW file is used, disregarding aspect ratio
// and cropbox specified in the RAW metadata.
LibRaw* load_raw(const std::string& fn, bool debayer, bool half_size, int qual, bool crop) {
  int ret;
  LibRaw* proc = new LibRaw();

  printf("Loading RAW file %s\n", fn.c_str());
  if ((ret = proc->open_file(fn.c_str())) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot open %s: %s\n", fn.c_str(), libraw_strerror(ret));
    return NULL;
  }
  printf("Image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  if (debayer)
    printf("Debayer quality: %d\n", qual);

  if ((ret = proc->unpack()) != LIBRAW_SUCCESS) {
    fprintf(stderr, "Cannot unpack %s: %s\n", fn.c_str(), libraw_strerror(ret));
    return NULL;
  }
  if (!(proc->imgdata.idata.filters || proc->imgdata.idata.colors == 1)) {
    printf("Only Bayer-pattern RAW files supported, sorry....\n");
    return NULL;
  }

  proc->imgdata.params.output_bps = 16;
  proc->imgdata.params.user_flip = 0;
  proc->imgdata.params.gamm[0] = 1;
  proc->imgdata.params.gamm[1] = 1;
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
  printf("Processed image size: %dx%d\n", proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight);
  return proc;
}

// Merge 4 RAW files from pixel-shift captures using Sony camera.
LibRaw* merge_pixel_shift_raw(LibRaw *proc[4]) {
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
  return proc[0];
}

template <class T>
float dot_product(const std::vector<float>& v1, const std::vector<T>& v2) {
  float prod = 0;
  for (int i = 0; i < v1.size() && i < v2.size(); ++i) {
    prod += v1[i] * v2[i];
  }
  return prod;
}

void scale(std::vector<float>& v, float factor) {
  for (int i = 0; i < v.size(); ++i) {
    v[i] *= factor;
  }
}

void adjust_correction_matrix(std::vector<float>& r_coef, 
                              std::vector<float>& g_coef,
                              std::vector<float>& b_coef,
                              float global_scale_factor,
                              const std::vector<int>& profile_film_base_rgb,
                              const std::vector<int>& film_base_rgb) {
  // First convert the linear RGB into corrected RGB values that are proportional
  // to transmittance.
  float cc_average_r = dot_product(r_coef, film_base_rgb);
  float cc_average_g = dot_product(g_coef, film_base_rgb);
  float cc_average_b = dot_product(b_coef, film_base_rgb);
  float cc_profile_r = dot_product(r_coef, profile_film_base_rgb);
  float cc_profile_g = dot_product(g_coef, profile_film_base_rgb);
  float cc_profile_b = dot_product(b_coef, profile_film_base_rgb);
  printf("Film base RGB (corrected): %f %f %f\n", cc_average_r, cc_average_g, cc_average_b);
  printf("Profile film base RGB (corrected): %f %f %f\n", cc_profile_r, cc_profile_g, cc_profile_b);

  // The ICC profiles map the transmittance values after the correction matrix is
  // applied to XYZ. Transmittance is relative the profile film base. For example
  // if the R value for profile film base is 10000, a R value of 5000 has
  // transmittance of 5000 / 10000 = 0.5. But we don't convert the corrected RGB
  // into a floating point value first, the actual denominator is not important.
  // Suppose we record the transmittance as a 16-bit fixed point number, we will
  // lose precision for denser film. So the ICC profiles are generated with
  // transmittance values scaled given a fixed value for mid-grey patch (e.g.
  // GS14). 
  //
  // The film base RGB values are only useful in relative terms. Here we are
  // compensating the difference in transmittance between the profile and the
  // scanned film.
  //
  // Suppose the mid-grey patch has R = 5000 and G = 10000. With film base
  // R = 10000 and film base G = 20000, the transmittance is 0.5 in both channels.
  //
  // A different captured film has R = 5000 (film base R = 10000) will have
  // transmittance of 0.5 in the R channel. Let's say the G layer is denser
  // because of flutation in the film development. G = 8000 (film base G = 16000)
  // will also have a transmittance of 0.5. However using the ICC profile which
  // assumes film base G is 20000 in the G channel, will treat the transmittance
  // as 8000 / 20000 = 0.4 and the color balance will be off.
  //
  // To compensate for this we will scale the G channel (with R as reference)
  // by (20000 / 10000) / (16000 / (10000) = 1.25.
  //
  // Of course the density even for R is not the same in the target and the
  // captured film. After adjusting the color balance, all channels can be
  // adjusted with a single scaling factor.
  float g_scale = (cc_profile_g / cc_profile_r) / (cc_average_g / cc_average_r);
  float b_scale = (cc_profile_b / cc_profile_r) / (cc_average_b / cc_average_r);
  printf("Scale channels to match film base: %f %f %f\n", 1.0, g_scale, b_scale);
  scale(r_coef, global_scale_factor);
  scale(g_coef, g_scale * global_scale_factor);
  scale(b_coef, b_scale * global_scale_factor);
}

void post_process(LibRaw* proc,
                  const std::vector<float>& r_coef,
                  const std::vector<float>& g_coef,
                  const std::vector<float>& b_coef) {
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

int read_profile(const std::string& prof_name, unsigned **prof_out, unsigned *size) {
  FILE *fp;
  if ((fp = fopen(prof_name.c_str(), "rb"))) {
    fread(size, 4, 1, fp);
    fseek(fp, 0, SEEK_SET);
    *prof_out = (unsigned *)malloc(*size = ntohl(*size));
    fread(*prof_out, 1, *size, fp);
    fclose(fp);
    return 0;
  }
  fprintf(stderr, "ERROR! Cannot read ICC profile: %s\n", prof_name.c_str());
  return -1;
}

int apply_profile(ushort (*image)[4], ushort width, ushort height,
                  const std::string& input, const std::string& output) {
  cmsHPROFILE in_profile = 0, out_profile = 0;
  cmsHTRANSFORM transform;
  unsigned *prof, *oprof;
  unsigned size;

  if (!read_profile(input, &prof, &size)) {
    printf("Reading input ICC profile: %s\n", input.c_str());
    if (!(in_profile = cmsOpenProfileFromMem(prof, size))) {
      free(prof);
      prof = 0;
      return -1;
    }
  } else {
    fprintf(stderr, "ERROR! cannot read input ICC profile\n");
    return -1;
  }

  if (output == "srgb") {
    printf("Using standard sRGB profile\n");
    out_profile = cmsCreate_sRGBProfile();
  } else if (!read_profile(output, &oprof, &size)) {
    printf("Reading output ICC profile: %s\n", output.c_str());
    if (!(out_profile = cmsOpenProfileFromMem(oprof, size))) {
      free(oprof);
      oprof = 0;
      return -1;
    }
  } else {
    fprintf(stderr, "ERROR! cannot read output ICC profile\n");
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
  argparse::ArgumentParser parser("neg_process");
  parser.add_argument("-H", "--half_size")
    .help("Half size.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-C", "--no_crop")
    .help("No cropping according to aspect ratio in RAW file.")
    .default_value(false)
    .implicit_value(true);
  parser.add_argument("-q", "--quality")
    .help("De-bayer quality. Not used in pixel-shift mode.")
    .scan<'i', int>()
    .default_value(0);
  parser.add_argument("-r", "--r_coeff")
    .help("R (corrected) value is dot product of this 'r1 r2 r3' vector and 'R G B' values from linear RAW.")
    .nargs(3)
    .default_value(std::vector<float>{1, 0, 0})
    .scan<'g', float>();
  parser.add_argument("-g", "--g_coeff")
    .help("G (corrected) value is dot product of this 'g1 g2 g3' vector and 'R G B' values from linear RAW.")
    .nargs(3)
    .default_value(std::vector<float>{0, 1, 0})
    .scan<'g', float>();
  parser.add_argument("-b", "--b_coeff")
    .help("B (corrected) value is dot product of this 'b1 b2 b3' vector and 'R G B' values from linear RAW.")
    .nargs(3)
    .default_value(std::vector<float>{0, 0, 1})
    .scan<'g', float>();
  parser.add_argument("-E", "--post_correction_scale")
    .help("Scale post-correction RGB values with this number.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("-G", "--output_gamma")
    .help("Apply a gamma to RGB values at the final output stage.")
    .scan<'g', float>()
    .default_value(1.0f);
  parser.add_argument("--profile_film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from profile.")
    .nargs(3)
    .default_value(std::vector<int>{1, 1, 1})
    .scan<'i', int>();
  parser.add_argument("--film_base_rgb")
    .help("Linear (uncorrected) R G B values of the film base from captured image.")
    .nargs(3)
    .default_value(std::vector<int>{1, 1, 1})
    .scan<'i', int>();
  parser.add_argument("-p", "--film_profile")
    .help("ICC Profile that applies to the corrected RGB values (See -r -g and -b flags). Consider this as the input ICC profile.");
  parser.add_argument("-P", "--colorspace")
    .help("srgb or [ICC profile path]. If specified the corrected RGB will be converted using this as the output profile.");
  parser.add_argument("-o", "--output")
    .required()
    .help("Output file location.");
  parser.add_argument("raw_files").nargs(1, 4);

  try {
    parser.parse_args(ac, av);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  const auto files = parser.get<std::vector<std::string>>("raw_files");
  auto r_coeff = parser.get<std::vector<float>>("--r_coeff");
  auto g_coeff = parser.get<std::vector<float>>("--g_coeff");
  auto b_coeff = parser.get<std::vector<float>>("--b_coeff");
  float global_scale_factor = parser.get<float>("--post_correction_scale");

  // TODO: Prepare a half size conversion only when --auto_crop is specified. 
  // LibRaw *half_proc = load_raw(files[0], /*debayer*/true, /*half_size*/true, /*quality*/0, /*crop*/true);

  LibRaw *proc;
  if (files.size() == 4) {
    LibRaw *four_proc[4];
    for (int i = 0; i < 4; ++i) {
      four_proc[i] = load_raw(
                         files[i], false, false,
                         /* Quality doesn't matter because no interpolation. */
                         0,
                         /* Don't crop since it might mess up pixel-shift merging. */
                         false);
    }
    proc = merge_pixel_shift_raw(four_proc);
    // Operating on data without interpolation and for Sony A7RM4 sensor
    // the input is 14-bit and multiply by 4 to expand into 16-bit.
    global_scale_factor = 4;
  } else {
    proc = load_raw(files[0], true,
                       parser.get<bool>("--half_size"),
                       parser.get<int>("--quality"),
                       !parser.get<bool>("--no_crop"));
  }
  printf("ISO Speed: %f\n", proc->imgdata.other.iso_speed);
  printf("Shutter Speed: %f\n", proc->imgdata.other.shutter);
  // A conversion matrix is applied the linear image from RAW file to:
  // 1. Remove crosstalk between color channels.
  // 2. Scale the color channels independently such that mid-grey values are aligned.
  // 3. Scale the color channels independently to compensate for density difference
  //    between target and the captured film.
  // 4. Global scale factor for brightness.
  // 5. Scale the color channels independently according to user input (optional).
  // 6. Scale from 14-bit to 16-bit in pixel-shift mode (optional).
  //
  // A single matrix is combined with the above steps combined. Note that scaling is
  // always done after crosstalk correction, hence the scale factor can be applied
  // separately to the R, G and B coefficients.
  auto profile_film_base_rgb = parser.get<std::vector<int>>("--profile_film_base_rgb");
  auto film_base_rgb = parser.get<std::vector<int>>("--film_base_rgb");
  // Both --profile_film_base_rgb and --film_base_rgb need to be specified or will
  // give incorrect scaling.
  if (!parser.is_used("--profile_film_base_rgb") || !parser.is_used("--film_base_rgb")) {
    profile_film_base_rgb = film_base_rgb = std::vector{1, 1, 1};
  }
  adjust_correction_matrix(r_coeff, g_coeff, b_coeff,
                           global_scale_factor,
                           profile_film_base_rgb,
                           film_base_rgb);
  printf("R coefficients: %1.5f %1.5f %1.5f\n", r_coeff[0], r_coeff[1], r_coeff[2]);
  printf("G coefficients: %1.5f %1.5f %1.5f\n", g_coeff[0], g_coeff[1], g_coeff[2]);
  printf("B coefficients: %1.5f %1.5f %1.5f\n", b_coeff[0], b_coeff[1], b_coeff[2]);
  post_process(proc, r_coeff, g_coeff, b_coeff);

  // By default attach the input profile to the output file only, no conversion.
  std::string attach_profile;
  if (parser.is_used("--film_profile") && parser.is_used("--colorspace")) {
    printf("Applying colorspace profile: %s\n", parser.get<std::string>("--colorspace").c_str());
    if (apply_profile(proc->imgdata.image, proc->imgdata.sizes.iwidth, proc->imgdata.sizes.iheight,
                      parser.get<std::string>("--film_profile"),
                      parser.get<std::string>("--colorspace")) < 0) {
      fprintf(stderr, "ERROR! Cannot convert using colorspace profile.\n");
    }
    // Attach the output colorspace profile since conversion has happened.
    attach_profile = parser.get<std::string>("--colorspace");
  } else if (parser.is_used("--film_profile")) {
    attach_profile = parser.get<std::string>("--film_profile");
  }

  if (!attach_profile.empty() && attach_profile != "srgb") {
    // If the profile to attach is not the psuedo "srgb" profile, the profile will be attached to the TIFF.
    // This is a hack to force LibRaw write the ICC profile in the TIFF without conversion.
    printf("Attaching profile: %s\n", attach_profile.c_str());
    unsigned size;
    if (read_profile(attach_profile, &proc->get_internal_data_pointer()->output_data.oprof, &size)) {
      return -1;
    }
  }

  if (parser.is_used("--output_gamma")) {
    proc->imgdata.params.gamm[0] = aparser.get<float>("--output_gamma");
    proc->imgdata.params.gamm[1] = 0;
  }
  const auto output = parser.get<std::string>("--output");
  printf("Writing TIFF '%s'\n", output.c_str());
  proc->dcraw_ppm_tiff_writer(output.c_str());
  return 0;
}
