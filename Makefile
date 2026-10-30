# neg_process: converts film negative RAW scans with the profiles and intent adapters built into it.
#
#   make dinov3_model       # models/dinov3-small/backbone.onnx, the DINOv3 backbone of the parameter search
#   make dinov3_engine      # its TensorRT engine, for USE_CUDA=1 builds (on the machine that runs it)
#   make neg_process        # bin_out/neg_process, with the backbone built in when it was exported
#   make lib                # bin_out/libnegicc_dinov3.a and bin_out/libraw.a, linked by negicc-station
#   make libraw             # bin_out/libraw.a, LibRaw built from the 3rd_party/libraw submodule
#   make deps               # bin_out/liblcms2.a and bin_out/libz.a, from the 3rd_party/lcms2 and zlib submodules

PYTHON ?= python3

.PHONY: all lib libraw deps neg_process raw_info dinov3_model dinov3_engine clean clean_lib clean_all
all: neg_process

# ----------------------------------------------------------------------
# LibRaw, pinned by the 3rd_party/libraw submodule (see 3rd_party/README.md) and built here as a static library, so
# neg_process and negicc-station decode with the same LibRaw on every machine instead of the system's libraw-dev.
# Consumers include "libraw/libraw.h" with $(LIBRAW_INC) (-isystem, so its headers do not warn) and link bin_out/libraw.a followed by the libraries in
# bin_out/libraw.ldlibs.
# ----------------------------------------------------------------------
LIBRAW_DIR = 3rd_party/libraw
LIBRAW_INC = -isystem $(LIBRAW_DIR)
LIBRAW_LIB = bin_out/libraw.a
ifeq ($(wildcard $(LIBRAW_DIR)/libraw/libraw.h),)
    $(error LibRaw submodule missing at $(LIBRAW_DIR): run git submodule update --init --recursive)
endif
# The library sources, as LibRaw's Makefile.dist lists them (every src/ file but the *_ph.cpp placeholders).
LIBRAW_SRCS = $(filter-out %_ph.cpp,$(wildcard $(LIBRAW_DIR)/src/*.cpp $(LIBRAW_DIR)/src/*/*.cpp))
LIBRAW_OBJS = $(patsubst %.cpp,bin_out/libraw/%.o,$(notdir $(LIBRAW_SRCS)))
vpath %.cpp $(sort $(dir $(LIBRAW_SRCS)))
# Lossy (JPEG-compressed) DNG support (needs libjpeg-dev): pinned off so every machine decodes the same files the same
# way; build with LIBRAW_JPEG=1 to enable it.
LIBRAW_JPEG ?= 0
LIBRAW_DEFS = -DUSE_ZLIB $(if $(filter 1,$(LIBRAW_JPEG)),-DUSE_JPEG -DUSE_JPEG8)
# Libraries linked after libraw.a beyond the pinned zlib ($(ZLIB_LIB)): the system's libjpeg with LIBRAW_JPEG=1.
LIBRAW_LDLIBS = $(if $(filter 1,$(LIBRAW_JPEG)),-ljpeg)
# Thread-safe build (no LIBRAW_NOTHREADS), with OpenMP as distribution packages build it. -ffp-contract=off as the
# rest of negicc, so aarch64 (where GCC fuses multiply-adds by default) decodes as x86 does.
LIBRAW_CXXFLAGS = -O3 -w -fPIC -fopenmp -ffp-contract=off $(LIBRAW_INC) $(ZLIB_INC) $(LIBRAW_DEFS)

bin_out/libraw/%.o: %.cpp bin_out/.libraw_jpeg_$(LIBRAW_JPEG)
	@mkdir -p bin_out/libraw
	$(CXX) $(LIBRAW_CXXFLAGS) -c $< -o $@

bin_out/.libraw_jpeg_$(LIBRAW_JPEG):
	mkdir -p bin_out
	rm -f bin_out/.libraw_jpeg_*
	touch $@

$(LIBRAW_LIB): $(LIBRAW_OBJS)
	rm -f $@
	ar rcs $@ $(LIBRAW_OBJS)

# The libraries consumers link after libraw.a, rewritten whenever they change (LIBRAW_JPEG), not only when LibRaw is
# rebuilt, so a build tree from before the pinned zlib does not keep linking the system's.
$(shell mkdir -p bin_out; printf '%s\n' "$(LIBRAW_LDLIBS)" | cmp -s - bin_out/libraw.ldlibs || \
        printf '%s\n' "$(LIBRAW_LDLIBS)" > bin_out/libraw.ldlibs)

libraw: $(LIBRAW_LIB)

raw_info: raw_info.cc $(LIBRAW_LIB) $(ZLIB_LIB)
	mkdir -p bin_out
	$(CXX) -o bin_out/raw_info raw_info.cc -I3rd_party $(LIBRAW_INC) $(LIBRAW_LIB) $(ZLIB_LIB) $(LIBRAW_LDLIBS) -fopenmp

# ----------------------------------------------------------------------
# Little-CMS 2 and zlib, pinned by the 3rd_party/lcms2 (2.14) and 3rd_party/zlib (1.3) submodules and built here as
# static libraries like LibRaw, so neg_process and negicc-station transform colours and read and write compressed
# data with the same code on every machine instead of the system's liblcms2-dev and zlib1g-dev. Consumers include
# <lcms2.h> with $(LCMS2_INC) and <zlib.h> with $(ZLIB_INC) and link bin_out/liblcms2.a and bin_out/libz.a. Built
# at -O2 as distributions build them, with -ffp-contract=off as the rest of negicc.
# ----------------------------------------------------------------------
LCMS2_DIR = 3rd_party/lcms2
LCMS2_INC = -isystem $(LCMS2_DIR)/include
LCMS2_LIB = bin_out/liblcms2.a
ZLIB_DIR = 3rd_party/zlib
ZLIB_INC = -isystem $(ZLIB_DIR)
ZLIB_LIB = bin_out/libz.a
ifeq ($(wildcard $(LCMS2_DIR)/include/lcms2.h),)
    $(error Little-CMS submodule missing at $(LCMS2_DIR): run git submodule update --init --recursive)
endif
ifeq ($(wildcard $(ZLIB_DIR)/zlib.h),)
    $(error zlib submodule missing at $(ZLIB_DIR): run git submodule update --init --recursive)
endif
LCMS2_OBJS = $(patsubst $(LCMS2_DIR)/src/%.c,bin_out/lcms2/%.o,$(wildcard $(LCMS2_DIR)/src/*.c))
ZLIB_OBJS = $(addprefix bin_out/zlib/,adler32.o compress.o crc32.o deflate.o gzclose.o gzlib.o gzread.o gzwrite.o \
                                     infback.o inffast.o inflate.o inftrees.o trees.o uncompr.o zutil.o)
DEP_CFLAGS = -O2 -fPIC -ffp-contract=off -w

bin_out/lcms2/%.o: $(LCMS2_DIR)/src/%.c
	@mkdir -p bin_out/lcms2
	$(CC) $(DEP_CFLAGS) $(LCMS2_INC) -c $< -o $@

bin_out/zlib/%.o: $(ZLIB_DIR)/%.c
	@mkdir -p bin_out/zlib
	$(CC) $(DEP_CFLAGS) -D_LARGEFILE64_SOURCE=1 -DHAVE_HIDDEN $(ZLIB_INC) -c $< -o $@

$(LCMS2_LIB): $(LCMS2_OBJS)
	rm -f $@
	ar rcs $@ $^

$(ZLIB_LIB): $(ZLIB_OBJS)
	rm -f $@
	ar rcs $@ $^

# libjpeg-turbo 3.2.1 and libtiff 4.7.2, pinned by the 3rd_party/libjpeg-turbo and 3rd_party/libtiff submodules and
# built here without their CMake builds: the headers those generate are written out in 3rd_party/config/ (Linux,
# 64-bit little-endian; libjpeg-turbo without SIMD). neg_process reads RGB TIFF and JPEG input with them.
JPEG_DIR = 3rd_party/libjpeg-turbo
JPEG_INC = -I3rd_party/config/libjpeg-turbo -I$(JPEG_DIR)/src
JPEG_LIB = bin_out/libjpeg.a
TIFF_DIR = 3rd_party/libtiff/libtiff
TIFF_INC = -I3rd_party/config/libtiff -I$(TIFF_DIR)
TIFF_LIB = bin_out/libtiff.a
ifeq ($(wildcard $(JPEG_DIR)/src/jpeglib.h),)
    $(error libjpeg-turbo submodule missing at $(JPEG_DIR): run git submodule update --init --recursive)
endif
ifeq ($(wildcard $(TIFF_DIR)/tiffio.h),)
    $(error libtiff submodule missing at $(TIFF_DIR): run git submodule update --init --recursive)
endif
# The library's sources as its CMakeLists.txt lists them (JPEG_SOURCES, with arithmetic coding), the 8-, 12- and
# 16-bit wrappers included.
JPEG_SRCS = $(addprefix $(JPEG_DIR)/src/,jcapimin.c jchuff.c jcicc.c jcinit.c jclhuff.c jcmarker.c jcmaster.c \
              jcomapi.c jcparam.c jcphuff.c jctrans.c jdapimin.c jdatadst.c jdatasrc.c jdhuff.c jdicc.c jdinput.c \
              jdlhuff.c jdmarker.c jdmaster.c jdphuff.c jdtrans.c jerror.c jfdctflt.c jmemmgr.c jmemnobs.c \
              jpeg_nbits.c jaricom.c jcarith.c jdarith.c) \
            $(filter-out %/template.c %/rdcolmap-% %/rdpng-% %/rdppm-% %/wrgif-% %/wrpng-% %/wrppm-%,$(wildcard $(JPEG_DIR)/src/wrapper/*.c))
JPEG_OBJS = $(patsubst $(JPEG_DIR)/src/%.c,bin_out/libjpeg/%.o,$(JPEG_SRCS))
TIFF_OBJS = $(patsubst $(TIFF_DIR)/%.c,bin_out/libtiff/%.o,$(filter-out %/tif_win32.c,$(wildcard $(TIFF_DIR)/tif_*.c)))

bin_out/libjpeg/%.o: $(JPEG_DIR)/src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(DEP_CFLAGS) $(JPEG_INC) -c $< -o $@

bin_out/libtiff/%.o: $(TIFF_DIR)/%.c
	@mkdir -p bin_out/libtiff
	$(CC) $(DEP_CFLAGS) -DALLOW_TIFF_NON_EXT_ALLOC_FUNCTIONS $(TIFF_INC) $(JPEG_INC) $(ZLIB_INC) -c $< -o $@

$(JPEG_LIB): $(JPEG_OBJS)
	rm -f $@
	ar rcs $@ $^

$(TIFF_LIB): $(TIFF_OBJS)
	rm -f $@
	ar rcs $@ $^

deps: $(LCMS2_LIB) $(ZLIB_LIB) $(JPEG_LIB) $(TIFF_LIB)

# Header-only libraries: nlohmann/json 3.11.3 and argparse, pinned by the 3rd_party/json and 3rd_party/argparse
# submodules and included as "nlohmann/json.hpp" and "argparse/argparse.hpp".
JSON_INC = -isystem 3rd_party/json/single_include
ARGPARSE_INC = -isystem 3rd_party/argparse/include

# ----------------------------------------------------------------------
# DINOv3 C++ Inference Engine & Library
# ----------------------------------------------------------------------
ARCH := $(shell uname -m)

ORT_CANDIDATES := $(wildcard 3rd_party/onnxruntime/include/onnxruntime_cxx_api.h \
                             ../onnxruntime/include/onnxruntime_cxx_api.h \
                             ../negicc-station/3rd_party/onnxruntime/include/onnxruntime_cxx_api.h)
ifneq ($(strip $(ORT_CANDIDATES)),)
    ORT_DIR ?= $(patsubst %/include/onnxruntime_cxx_api.h,%,$(firstword $(ORT_CANDIDATES)))
else
    ORT_DIR ?= 3rd_party/onnxruntime
endif
ORT_INC = -isystem $(ORT_DIR)/include
ifeq ($(ARCH),aarch64)
    ORT_LIB_DIR ?= $(ORT_DIR)/lib/aarch64
else
    ORT_LIB_DIR ?= $(ORT_DIR)/lib/x86_64
endif
ifneq ($(filter /%,$(ORT_LIB_DIR)),)
    ORT_RPATH = -Wl,-rpath,'$(ORT_LIB_DIR)'
else
    ORT_RPATH = -Wl,-rpath,'$$ORIGIN/../$(ORT_LIB_DIR)'
endif
ORT_LDFLAGS = -L$(ORT_LIB_DIR) -lonnxruntime $(ORT_RPATH)

NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
    ifneq ($(wildcard /usr/local/cuda/bin/nvcc),)
        NVCC := /usr/local/cuda/bin/nvcc
    else ifneq ($(wildcard /usr/local/cuda-12.6/bin/nvcc),)
        NVCC := /usr/local/cuda-12.6/bin/nvcc
    endif
endif

USE_CUDA ?= $(if $(NVCC),1,0)
TRT_INC ?= -I/usr/include/$(ARCH)-linux-gnu
TRT_LIBS ?= -L/usr/lib/$(ARCH)-linux-gnu -lnvinfer

ifeq ($(USE_CUDA),1)
    ifeq ($(NVCC),)
        $(error USE_CUDA=1 but nvcc was not found; install CUDA or build with USE_CUDA=0)
    endif
    CUDA_INC = -I/usr/local/cuda/include -I/usr/local/cuda-12.6/include
    CUDA_LIBS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/aarch64-linux/lib -L/usr/local/cuda-12.6/targets/aarch64-linux/lib -lcudart
    DINOV3_RUNNER_SRC = dinov3_trt_runner.cc
    DINOV3_RUNNER_OBJ = bin_out/dinov3_trt_runner.o
    DINOV3_EXTRA_INC = $(CUDA_INC) $(TRT_INC)
    DINOV3_EXTRA_LIBS = $(CUDA_LIBS) $(TRT_LIBS)
else ifeq ($(USE_CUDA),0)
    DINOV3_RUNNER_SRC = dinov3_nocuda.cc
    DINOV3_RUNNER_OBJ = bin_out/dinov3_nocuda.o
    DINOV3_EXTRA_INC =
    DINOV3_EXTRA_LIBS =
else
    $(error USE_CUDA must be 0 or 1, got '$(USE_CUDA)')
endif

BUILD_STAMP = bin_out/.use_cuda_$(USE_CUDA)

$(BUILD_STAMP):
	mkdir -p bin_out
	rm -f bin_out/.use_cuda_*
	touch $@

DINOV3_CXXFLAGS = -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -fsigned-char -Wall -Wextra -I. $(LCMS2_INC) $(ZLIB_INC) \
                  $(JSON_INC) $(ORT_INC) $(DINOV3_EXTRA_INC)

# The library negicc-station links (NEGICC_LIB): the DINOv3 engine and solver, and the conversion code shared with
# neg_process (profile bundles, cropping, flat field, zero-copy CFA pipeline, the conversion parameter search).
DINOV3_OBJS = bin_out/dinov3_engine.o bin_out/dinov3_preprocess.o bin_out/dinov3_ort_runner.o bin_out/dinov3_auto_solver.o \
              bin_out/flat_field.o bin_out/cfa_pipeline.o bin_out/film_frame_detector.o bin_out/film_base_detect.o \
              bin_out/profile_bundle.o bin_out/multi_target_optimizer.o $(DINOV3_RUNNER_OBJ)

bin_out/dinov3_engine.o: dinov3_engine.cc dinov3_engine.h dinov3_runner.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_engine.cc -o $@

bin_out/dinov3_preprocess.o: dinov3_preprocess.cc dinov3_preprocess.h color_pipeline.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_preprocess.cc -o $@

bin_out/dinov3_ort_runner.o: dinov3_ort_runner.cc dinov3_ort_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_ort_runner.cc -o $@

bin_out/dinov3_auto_solver.o: dinov3_auto_solver.cc dinov3_auto_solver.h cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h film_frame_detector.h neg_pipeline.h colorspace_conversion.h dinov3_engine.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_auto_solver.cc -o $@

bin_out/flat_field.o: flat_field.cc flat_field.h flat_sample.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c flat_field.cc -o $@

bin_out/cfa_pipeline.o: cfa_pipeline.cc cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h film_frame_detector.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c cfa_pipeline.cc -o $@

bin_out/dinov3_trt_runner.o: dinov3_trt_runner.cc dinov3_trt_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_trt_runner.cc -o $@

bin_out/dinov3_nocuda.o: dinov3_nocuda.cc dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_nocuda.cc -o $@

bin_out/profile_bundle.o: profile_bundle.cc profile_bundle.h cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h film_frame_detector.h neg_pipeline.h dinov3_auto_solver.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c profile_bundle.cc -o $@

bin_out/multi_target_optimizer.o: multi_target_optimizer.cc multi_target_optimizer.h cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h film_frame_detector.h colorspace_conversion.h profile_bundle.h dinov3_auto_solver.h dinov3_engine.h neg_pipeline.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c multi_target_optimizer.cc -o $@


bin_out/libnegicc_dinov3.a: $(DINOV3_OBJS) $(BUILD_STAMP)
	mkdir -p bin_out
	rm -f $@
	ar rcs $@ $(DINOV3_OBJS)

lib: bin_out/libnegicc_dinov3.a $(LIBRAW_LIB) $(LCMS2_LIB) $(ZLIB_LIB) $(TIFF_LIB) $(JPEG_LIB)

GIT_VERSION ?= $(shell git describe --always --dirty --tags 2>/dev/null || echo "master")

bin_out/film_frame_detector.o: film_frame_detector.cc film_frame_detector.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -I. -c film_frame_detector.cc -o $@

bin_out/film_base_detect.o: film_base_detect.cc film_base_detect.h cfa_pipeline.h flat_field.h flat_sample.h film_frame_detector.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -I. -c film_base_detect.cc -o $@

# Data files built into neg_process: the profile bundles and the intent adapters, exported from negicc-station
# (tools/export_negicc_data.py), and the DINOv3 backbone: backbone.onnx (make dinov3_model), and in USE_CUDA=1 builds
# also its TensorRT engine (make dinov3_engine, on the machine that runs it), which falls back to the .onnx when it
# cannot be loaded. Without the backbone neg_process still builds, but converts only with --gain_g or --gain_b given.
BACKBONE_FILES = models/dinov3-small/backbone.onnx $(if $(filter 1,$(USE_CUDA)),models/dinov3-small/backbone.engine)
EMBED_FILES = $(sort $(wildcard profiles/*/bundle.json profiles/*/*.icc models/dinov3-small_*/adapter.onnx)) \
              $(wildcard $(BACKBONE_FILES))

EMBED_IDX = $(shell seq 1 $(words $(EMBED_FILES)))

# The symbols of the embedded data, shared by bin_out/embedded_data.S and embedded_data.cc: EMBED_TABLE, an array of
# {name, data, size} entries (EmbeddedFile), and EMBED_COUNT, the number of entries.
EMBED_TABLE = negicc_embedded_table
EMBED_COUNT = negicc_embedded_count

define newline


endef

# Embeds file $(2) as entry $(1): its bytes, a NUL byte after them (so text files parse in place) and its path,
# which is its name in the table.
define EMBED_FILE_ASM
    .section .rodata
    .balign 16
$(EMBED_TABLE)_data_$(1):
    .incbin "$(2)"
$(EMBED_TABLE)_end_$(1):
    .byte 0
$(EMBED_TABLE)_name_$(1):
    .asciz "$(2)"

endef

define EMBED_TABLE_ASM
    .section .data.rel.ro, "aw"
    .balign 8
    .globl $(EMBED_TABLE)
$(EMBED_TABLE):
$(foreach i,$(EMBED_IDX),    .quad $(EMBED_TABLE)_name_$(i), $(EMBED_TABLE)_data_$(i), $(EMBED_TABLE)_end_$(i) - $(EMBED_TABLE)_data_$(i)$(newline))
    .globl $(EMBED_COUNT)
$(EMBED_COUNT):
    .quad $(words $(EMBED_FILES))
    .section .note.GNU-stack, "", %progbits
endef

bin_out:
	mkdir -p $@

# Track the list of embedded files so removing a file from profiles/ or models/ triggers regeneration of embedded_data.S.
EMBED_MANIFEST = bin_out/.embed_manifest
$(shell mkdir -p bin_out; printf '%s\n' $(EMBED_FILES) > $(EMBED_MANIFEST).tmp; \
        if ! cmp -s $(EMBED_MANIFEST).tmp $(EMBED_MANIFEST) 2>/dev/null; then \
            mv $(EMBED_MANIFEST).tmp $(EMBED_MANIFEST); \
        else \
            rm -f $(EMBED_MANIFEST).tmp; \
        fi)

# neg_process throws on a bundle it cannot read, so the bundles are checked before they are embedded: every bracket
# needs a shutter speed shutter_to_seconds (profile_bundle.cc) can read.
bin_out/.bundles_checked: $(filter %/bundle.json,$(EMBED_FILES)) $(EMBED_MANIFEST) check_profile_bundles.py | bin_out
	$(PYTHON) check_profile_bundles.py $(filter %/bundle.json,$(EMBED_FILES))
	touch $@

bin_out/embedded_data.S: $(EMBED_FILES) $(EMBED_MANIFEST) bin_out/.bundles_checked Makefile $(BUILD_STAMP) | bin_out
	$(file > $@,$(foreach i,$(EMBED_IDX),$(call EMBED_FILE_ASM,$(i),$(word $(i),$(EMBED_FILES))))$(EMBED_TABLE_ASM))

bin_out/embedded_data_files.o: bin_out/embedded_data.S
	$(CC) -c $< -o $@

bin_out/embedded_data.o: embedded_data.cc embedded_data.h profile_bundle.h Makefile $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -DEMBED_TABLE=$(EMBED_TABLE) -DEMBED_COUNT=$(EMBED_COUNT) \
	    -c embedded_data.cc -o $@

# neg_process's own sources (np_*.cc, rgb_input.cc): not in the library negicc-station links. Compiled as
# neg_process.cc is, so moving code between them leaves its arithmetic as it was.
NP_CXXFLAGS = -O3 -std=c++17 -fopenmp -ffp-contract=off -Wall -Wextra -I. -I3rd_party $(LIBRAW_INC) $(LCMS2_INC) $(ZLIB_INC) \
              $(JSON_INC) $(ARGPARSE_INC) $(ORT_INC) $(DINOV3_EXTRA_INC)
NP_SRCS = np_options.cc np_pipeline.cc np_profile.cc np_raw.cc np_geometry.cc np_conversion.cc np_output.cc np_input.cc \
          rgb_input.cc
NP_OBJS = $(patsubst %.cc,bin_out/%.o,$(NP_SRCS))
NP_HDRS = $(wildcard np_*.h) rgb_input.h neg_pipeline.h cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h \
          dinov3_auto_solver.h film_frame_detector.h film_base_detect.h pixel_shift_group.h profile_bundle.h \
          multi_target_optimizer.h embedded_data.h 3rd_party/tiff_head.h

bin_out/np_output.o: np_output.cc $(NP_HDRS) $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(NP_CXXFLAGS) -DNEGICC_GIT_VERSION="\"$(GIT_VERSION)\"" -c $< -o $@

bin_out/rgb_input.o: rgb_input.cc $(NP_HDRS) $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(NP_CXXFLAGS) $(TIFF_INC) $(JPEG_INC) -c $< -o $@

bin_out/np_%.o: np_%.cc $(NP_HDRS) $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(NP_CXXFLAGS) -c $< -o $@

NEG_PROCESS_OBJS = bin_out/embedded_data.o bin_out/embedded_data_files.o $(NP_OBJS)

neg_process: bin_out/neg_process

bin_out/neg_process: neg_process.cc $(NEG_PROCESS_OBJS) bin_out/libnegicc_dinov3.a $(LIBRAW_LIB) $(TIFF_LIB) $(JPEG_LIB) $(LCMS2_LIB) $(ZLIB_LIB) $(NP_HDRS) $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -DNEGICC_GIT_VERSION="\"$(GIT_VERSION)\"" -o $@ neg_process.cc $(NEG_PROCESS_OBJS) bin_out/libnegicc_dinov3.a $(LIBRAW_LIB) $(TIFF_LIB) $(JPEG_LIB) $(LCMS2_LIB) $(ZLIB_LIB) -I. -I3rd_party $(LIBRAW_INC) $(LCMS2_INC) $(ZLIB_INC) $(JSON_INC) $(ARGPARSE_INC) $(ORT_INC) $(DINOV3_EXTRA_INC) $(LIBRAW_LDLIBS) -O3 -std=c++17 -fopenmp -ffp-contract=off $(ORT_LDFLAGS) $(DINOV3_EXTRA_LIBS)

# The DINOv3 backbone (shared by every film stock): exported from the pretrained weights with torch and
# transformers. DINOV3_SOURCE is a Hugging Face model id or a local model directory.
DINOV3_SOURCE ?= Tooony133/dinov3-vits16-pretrain-lvd1689m

dinov3_model:
	$(PYTHON) models/dinov3-small/export_dinov3_backbone.py --source $(DINOV3_SOURCE)

dinov3_engine:
	$(PYTHON) models/dinov3-small/export_dinov3_backbone.py --engine-only

clean: clean_lib

clean_lib:
	rm -f bin_out/*.o bin_out/*.a bin_out/*.S bin_out/.use_cuda_* bin_out/.bundles_checked $(EMBED_MANIFEST)

clean_all: clean_lib
	rm -rf bin_out/libraw bin_out/libraw.a bin_out/libraw.ldlibs bin_out/.libraw_jpeg_* bin_out/lcms2 bin_out/zlib \
	       bin_out/libjpeg bin_out/libtiff
	rm -f bin_out/raw_info bin_out/neg_process
