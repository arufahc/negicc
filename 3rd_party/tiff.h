#ifndef MINIMAL_TIFF_H
#define MINIMAL_TIFF_H

// Subset of libtiff tiff.h so we don't need libtiff-dev.
#define TIFFTAG_SUBFILETYPE          254
#define TIFFTAG_IMAGEWIDTH           256
#define TIFFTAG_IMAGELENGTH          257
#define TIFFTAG_BITSPERSAMPLE        258
#define TIFFTAG_COMPRESSION          259
#define TIFFTAG_PHOTOMETRIC          262
#define TIFFTAG_IMAGEDESCRIPTION     270
#define TIFFTAG_MAKE                 271
#define TIFFTAG_MODEL                272
#define TIFFTAG_STRIPOFFSETS         273
#define TIFFTAG_SAMPLESPERPIXEL      277
#define TIFFTAG_ROWSPERSTRIP         278
#define TIFFTAG_STRIPBYTECOUNTS      279
#define TIFFTAG_XRESOLUTION          282
#define TIFFTAG_YRESOLUTION          283
#define TIFFTAG_PLANARCONFIG         284
#define TIFFTAG_RESOLUTIONUNIT       296
#define TIFFTAG_SOFTWARE             305
#define TIFFTAG_DATETIME             306
#define TIFFTAG_ARTIST               315
#define TIFFTAG_EXIFIFD              34665
#define TIFFTAG_ICCPROFILE           34675
#define TIFFTAG_GPSIFD               34853

// Standard EXIF Tag Definitions
#define EXIFTAG_EXPOSURETIME         33434
#define EXIFTAG_FNUMBER              33437
#define EXIFTAG_ISOSPEEDRATINGS      34855
#define EXIFTAG_FOCALLENGTH          37386

#endif // MINIMAL_TIFF_H
