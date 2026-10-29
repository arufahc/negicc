/* tiffconf.h for libtiff 4.7.2 (3rd_party/libtiff), written for negicc's Makefile in place of the one its CMake build
 * generates from tiffconf.h.cmake.in: 64-bit little-endian Linux (x86_64, aarch64); codecs: none, CCITT, JPEG
 * (libjpeg-turbo), LogLuv, LZW, NeXT, PackBits, PixarLog and Deflate (zlib), ThunderScan. */
#ifndef _TIFFCONF_
#define _TIFFCONF_
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>
#define TIFF_INT16_T int16_t
#define TIFF_INT32_T int32_t
#define TIFF_INT64_T int64_t
#define TIFF_INT8_T int8_t
#define TIFF_UINT16_T uint16_t
#define TIFF_UINT32_T uint32_t
#define TIFF_UINT64_T uint64_t
#define TIFF_UINT8_T uint8_t
#define TIFF_SSIZE_T int64_t
#define HAVE_IEEEFP 1
#define HOST_BIGENDIAN 0
#define CCITT_SUPPORT 1
#define JPEG_SUPPORT 1
#define LOGLUV_SUPPORT 1
#define LZW_SUPPORT 1
#define NEXT_SUPPORT 1
#define PACKBITS_SUPPORT 1
#define PIXARLOG_SUPPORT 1
#define THUNDER_SUPPORT 1
#define ZIP_SUPPORT 1
#define STRIPCHOP_DEFAULT TIFF_STRIPCHOP
#define SUBIFD_SUPPORT 1
#define DEFAULT_EXTRASAMPLE_AS_ALPHA 1
#define CHECK_JPEG_YCBCR_SUBSAMPLING 1
#define MDI_SUPPORT 1
#endif
