/* tiffvers.h for libtiff 4.7.2, written for negicc's Makefile from tiffvers.h.cmake.in (VERSION, RELEASE-DATE). */
#define TIFFLIB_VERSION_STR "LIBTIFF, Version 4.7.2\nCopyright (c) 1988-1996 Sam Leffler\nCopyright (c) 1991-1996 Silicon Graphics, Inc."
#define TIFFLIB_VERSION 20260627
#define TIFFLIB_MAJOR_VERSION 4
#define TIFFLIB_MINOR_VERSION 7
#define TIFFLIB_MICRO_VERSION 2
#define TIFFLIB_VERSION_STR_MAJ_MIN_MIC "4.7.2"
#define TIFFLIB_AT_LEAST(major, minor, micro) \
    (TIFFLIB_MAJOR_VERSION > (major) || \
     (TIFFLIB_MAJOR_VERSION == (major) && TIFFLIB_MINOR_VERSION > (minor)) || \
     (TIFFLIB_MAJOR_VERSION == (major) && TIFFLIB_MINOR_VERSION == (minor) && \
      TIFFLIB_MICRO_VERSION >= (micro)))
