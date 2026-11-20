/* -*- C++ -*-
 * File: libraw_internal.h
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sat Mar  8 , 2008
 *
 * LibRaw internal data structures (not visible outside)

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

// This is a fork of libraw/src/write/file_write.cpp.
// tiff_head is not public API and is copied here.

#include <tiff.h>
#include <time.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "libraw/libraw.h"

#define FORC(cnt) for (c = 0; c < cnt; c++)

// count and val are unsigned 32-bit, as the TIFF fields they fill: a strip of more than 2 GB keeps its byte count.
void tiff_set(struct tiff_hdr *th, ushort *ntag, ushort tag,
                      ushort type, uint32_t count, uint32_t val)
{
  struct libraw_tiff_tag *tt;
  int c;

  tt = (struct libraw_tiff_tag *)(ntag + 1) + (*ntag)++;
  tt->val.i = val;
  if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_BYTE) && count <= 4)
    FORC(4) tt->val.c[c] = val >> (c << 3);
  else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_ASCII))
  {
    count = int(strnlen((char *)th + val, count - 1)) + 1;
    if (count <= 4)
      FORC(4) tt->val.c[c] = ((char *)th)[val + c];
  }
  else if (tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT) && count <= 2)
    FORC(2) tt->val.s[c] = val >> (c << 4);
  tt->count = count;
  tt->type = type;
  tt->tag = tag;
}

#define TOFF(ptr) ((char *)(&(ptr)) - (char *)th)

// If |width_override| or |height_override| are greater than 0.
// Use their values instead of width and height from |proc|.
// This is useful when outputting an image that is different size
// than that of the processed raw.
// The image is three 16-bit samples a pixel, RGB, in one strip, as write_tiff (np_output.cc) writes it whatever the
// RAW's colours; the caller checks that the header, the profile and the strip fit a classic TIFF's 32-bit offsets.
// A field the RAW does not record usably (no capture time; an exposure time, f-number or focal length that is negative,
// not finite or too large for its numerator; an ISO speed a SHORT cannot hold) is left out rather than written wrong.
void tiff_head(const LibRaw* proc, struct tiff_hdr *th,
               size_t profile_size, size_t width_override = 0, size_t height_override = 0) {
  int c;

  // Commonly used POD.
  const unsigned colors = 3;
  auto width = width_override ? width_override : proc->imgdata.sizes.iwidth;
  auto height = height_override ? height_override : proc->imgdata.sizes.iheight;
  const unsigned output_bps = 16;

  memset(th, 0, sizeof *th);
  th->t_order = htonl(0x4d4d4949) >> 16;
  th->magic = 42;
  th->ifd = 10;
  th->rat[0] = th->rat[2] = 300;
  th->rat[1] = th->rat[3] = 1;
  FORC(6) th->rat[4 + c] = 1000000;
  // A value in millionths, the rational's numerator over the 1000000 above (the product in float, as LibRaw's
  // th->rat[4] *= shutter computes it); false when it is negative, not finite or more than th->rat's int holds.
  auto micro = [](float v, int& num) {
    const float n = 1000000.0f * v;
    if (!(n >= 0.0f && n < (float)std::numeric_limits<int>::max())) return false;
    num = (int)n;
    return true;
  };
  const bool has_shutter = micro(proc->imgdata.other.shutter, th->rat[4]);
  const bool has_aperture = micro(proc->imgdata.other.aperture, th->rat[6]);
  const bool has_focal_len = micro(proc->imgdata.other.focal_len, th->rat[8]);
  strncpy(th->t_desc, proc->imgdata.other.desc, 512);
  strncpy(th->t_make, proc->imgdata.idata.make, 64);
  strncpy(th->t_model, proc->imgdata.idata.model, 64);
  strcpy(th->soft, "NegProcess");
  // The capture's time, "YYYY:MM:DD HH:MM:SS" (TIFF 6.0 DateTime, which fills th->date); none when the RAW records no
  // time (0), when the time cannot be converted, or when its year takes more than four digits.
  bool has_date = false;
  struct tm tm_buf;
  if (proc->imgdata.other.timestamp > 0 && localtime_r(&proc->imgdata.other.timestamp, &tm_buf)) {
    const int n = snprintf(th->date, sizeof th->date, "%04d:%02d:%02d %02d:%02d:%02d", tm_buf.tm_year + 1900,
                           tm_buf.tm_mon + 1, tm_buf.tm_mday, tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    has_date = n == (int)sizeof th->date - 1;
    if (!has_date) memset(th->date, 0, sizeof th->date);
  }
  strncpy(th->t_artist, proc->imgdata.other.artist, 64);
  tiff_set(th, &th->ntag, TIFFTAG_SUBFILETYPE, 4, 1, 0);
  tiff_set(th, &th->ntag, TIFFTAG_IMAGEWIDTH, 4, 1, width);
  tiff_set(th, &th->ntag, TIFFTAG_IMAGELENGTH, 4, 1, height);
  tiff_set(th, &th->ntag, TIFFTAG_BITSPERSAMPLE, 3, colors, output_bps);
  if (colors > 2)
    th->tag[th->ntag - 1].val.i = TOFF(th->bps);
  FORC(4) th->bps[c] = output_bps;
  tiff_set(th, &th->ntag, TIFFTAG_COMPRESSION, 3, 1, 1);
  tiff_set(th, &th->ntag, TIFFTAG_PHOTOMETRIC, 3, 1, 1 + (colors > 1));
  tiff_set(th, &th->ntag, TIFFTAG_IMAGEDESCRIPTION, 2, 512, TOFF(th->t_desc));
  tiff_set(th, &th->ntag, TIFFTAG_MAKE, 2, 64, TOFF(th->t_make));
  tiff_set(th, &th->ntag, TIFFTAG_MODEL, 2, 64, TOFF(th->t_model));
  tiff_set(th, &th->ntag, TIFFTAG_STRIPOFFSETS, 4, 1, sizeof *th + profile_size);
  tiff_set(th, &th->ntag, TIFFTAG_SAMPLESPERPIXEL, 3, 1, colors);
  tiff_set(th, &th->ntag, TIFFTAG_ROWSPERSTRIP, 4, 1, height);
  tiff_set(th, &th->ntag, TIFFTAG_STRIPBYTECOUNTS, 4, 1,
           height * width * colors * output_bps / 8);
  // tiff_set(th, &th->ntag, TIFFTAG_XRESOLUTION, 3, 1, "12435867"[flip] - '0');
  tiff_set(th, &th->ntag, TIFFTAG_XRESOLUTION, 5, 1, TOFF(th->rat[0]));
  tiff_set(th, &th->ntag, TIFFTAG_YRESOLUTION, 5, 1, TOFF(th->rat[2]));
  tiff_set(th, &th->ntag, TIFFTAG_PLANARCONFIG, 3, 1, 1);
  tiff_set(th, &th->ntag, TIFFTAG_RESOLUTIONUNIT, 3, 1, 2);
  tiff_set(th, &th->ntag, TIFFTAG_SOFTWARE, 2, 32, TOFF(th->soft));
  if (has_date)
    tiff_set(th, &th->ntag, TIFFTAG_DATETIME, 2, 20, TOFF(th->date));
  tiff_set(th, &th->ntag, TIFFTAG_ARTIST, 2, 64, TOFF(th->t_artist));
  tiff_set(th, &th->ntag, TIFFTAG_EXIFIFD, 4, 1, TOFF(th->nexif));
  if (profile_size)
    tiff_set(th, &th->ntag, TIFFTAG_ICCPROFILE, 7, profile_size, sizeof *th);
  if (has_shutter)
    tiff_set(th, &th->nexif, EXIFTAG_EXPOSURETIME, 5, 1, TOFF(th->rat[4]));
  if (has_aperture)
    tiff_set(th, &th->nexif, EXIFTAG_FNUMBER, 5, 1, TOFF(th->rat[6]));
  // ISOSpeedRatings is a SHORT: a speed it cannot hold (negative, not finite, or above 65535) is left out.
  const float iso = proc->imgdata.other.iso_speed;
  if (iso >= 0.0f && iso <= (float)std::numeric_limits<ushort>::max())
    tiff_set(th, &th->nexif, EXIFTAG_ISOSPEEDRATINGS, 3, 1, (uint32_t)iso);
  if (has_focal_len)
    tiff_set(th, &th->nexif, EXIFTAG_FOCALLENGTH, 5, 1, TOFF(th->rat[8]));

  const unsigned* gpsdata = proc->imgdata.other.gpsdata;
  if (gpsdata[1])
  {
    // The GPS values first: the ASCII tags below measure their strings in th->gps when they are set.
    memcpy(th->gps, gpsdata, sizeof th->gps);
    // GPSLatitudeRef and GPSLongitudeRef: one character and its NUL, held in the tag itself. LibRaw's tiff_head passes
    // the offset of a stack array from th, which tiff_set then reads through th: undefined behaviour, and at -O3 the
    // tags carried whatever was on the stack, so the same scan wrote different bytes on every run.
    auto ref_tag = [&](ushort tag, unsigned ref) {
      struct libraw_tiff_tag* tt = (struct libraw_tiff_tag*)(&th->ngps + 1) + th->ngps;
      th->ngps++;
      tt->tag = tag;
      tt->type = 2;
      tt->count = (char)ref ? 2 : 1;
      tt->val.i = 0;
      tt->val.c[0] = (char)ref;
    };
    tiff_set(th, &th->ntag, TIFFTAG_GPSIFD, 4, 1, TOFF(th->ngps));
    tiff_set(th, &th->ngps, 0, 1, 4, 0x202);
    ref_tag(1, gpsdata[29]);
    tiff_set(th, &th->ngps, 2, 5, 3, TOFF(th->gps[0]));
    ref_tag(3, gpsdata[30]);
    tiff_set(th, &th->ngps, 4, 5, 3, TOFF(th->gps[6]));
    tiff_set(th, &th->ngps, 5, 1, 1, gpsdata[31]);
    tiff_set(th, &th->ngps, 6, 5, 1, TOFF(th->gps[18]));
    tiff_set(th, &th->ngps, 7, 5, 3, TOFF(th->gps[12]));
    tiff_set(th, &th->ngps, 18, 2, 12, TOFF(th->gps[20]));
    tiff_set(th, &th->ngps, 29, 2, 12, TOFF(th->gps[23]));
  }
}
