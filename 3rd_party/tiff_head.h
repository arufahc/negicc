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

#include "libraw/libraw.h"

#define FORC(cnt) for (c = 0; c < cnt; c++)

void tiff_set(struct tiff_hdr *th, ushort *ntag, ushort tag,
                      ushort type, int count, int val)
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
void tiff_head(const LibRaw* proc, struct tiff_hdr *th,
               size_t profile_size, size_t width_override = 0, size_t height_override = 0) {
  int c;
  struct tm *t;

  // Commonly used POD.
  auto colors = proc->imgdata.idata.colors;
  auto width = width_override ? width_override : proc->imgdata.sizes.iwidth;
  auto height = height_override ? height_override : proc->imgdata.sizes.iheight;
  auto output_bps = proc->imgdata.params.output_bps;

  memset(th, 0, sizeof *th);
  th->t_order = htonl(0x4d4d4949) >> 16;
  th->magic = 42;
  th->ifd = 10;
  th->rat[0] = th->rat[2] = 300;
  th->rat[1] = th->rat[3] = 1;
  FORC(6) th->rat[4 + c] = 1000000;
  th->rat[4] *= proc->imgdata.other.shutter;
  th->rat[6] *= proc->imgdata.other.aperture;
  th->rat[8] *= proc->imgdata.other.focal_len;
  strncpy(th->t_desc, proc->imgdata.other.desc, 512);
  strncpy(th->t_make, proc->imgdata.idata.make, 64);
  strncpy(th->t_model, proc->imgdata.idata.model, 64);
  strcpy(th->soft, "NegProcess");
  t = localtime(&proc->imgdata.other.timestamp);
  sprintf(th->date, "%04d:%02d:%02d %02d:%02d:%02d", t->tm_year + 1900,
          t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
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
  tiff_set(th, &th->ntag, TIFFTAG_DATETIME, 2, 20, TOFF(th->date));
  tiff_set(th, &th->ntag, TIFFTAG_ARTIST, 2, 64, TOFF(th->t_artist));
  tiff_set(th, &th->ntag, TIFFTAG_EXIFIFD, 4, 1, TOFF(th->nexif));
  if (profile_size)
    tiff_set(th, &th->ntag, TIFFTAG_ICCPROFILE, 7, profile_size, sizeof *th);
  tiff_set(th, &th->nexif, EXIFTAG_EXPOSURETIME, 5, 1, TOFF(th->rat[4]));
  tiff_set(th, &th->nexif, EXIFTAG_FNUMBER, 5, 1, TOFF(th->rat[6]));
  tiff_set(th, &th->nexif, EXIFTAG_ISOSPEEDRATINGS, 3, 1, proc->imgdata.other.iso_speed);
  tiff_set(th, &th->nexif, EXIFTAG_FOCALLENGTH, 5, 1, TOFF(th->rat[8]));

  const unsigned* gpsdata = proc->imgdata.other.gpsdata;
  if (gpsdata[1])
  {
    uchar latref[4] = { (uchar)(gpsdata[29]),0,0,0 },
          lonref[4] = { (uchar)(gpsdata[30]),0,0,0 };
    tiff_set(th, &th->ntag, TIFFTAG_GPSIFD, 4, 1, TOFF(th->ngps));
    tiff_set(th, &th->ngps, 0, 1, 4, 0x202);
    tiff_set(th, &th->ngps, 1, 2, 2, TOFF(latref));
    tiff_set(th, &th->ngps, 2, 5, 3, TOFF(th->gps[0]));
    tiff_set(th, &th->ngps, 3, 2, 2, TOFF(lonref));
    tiff_set(th, &th->ngps, 4, 5, 3, TOFF(th->gps[6]));
    tiff_set(th, &th->ngps, 5, 1, 1, gpsdata[31]);
    tiff_set(th, &th->ngps, 6, 5, 1, TOFF(th->gps[18]));
    tiff_set(th, &th->ngps, 7, 5, 3, TOFF(th->gps[12]));
    tiff_set(th, &th->ngps, 18, 2, 12, TOFF(th->gps[20]));
    tiff_set(th, &th->ngps, 29, 2, 12, TOFF(th->gps[23]));
    memcpy(th->gps, gpsdata, sizeof th->gps);
  }
}
