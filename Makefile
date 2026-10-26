# neg_process: converts film negative RAW scans with the profiles and intent adapters built into it.
#
#   make dinov3_model       # models/dinov3-small/backbone.onnx, the DINOv3 backbone of the parameter search
#   make dinov3_engine      # its TensorRT engine, for USE_CUDA=1 builds (on the machine that runs it)
#   make neg_process        # bin_out/neg_process, with the backbone built in when it was exported
#   make lib                # bin_out/libnegicc_dinov3.a and bin_out/libraw.a, linked by negicc-station
#   make libraw             # bin_out/libraw.a, LibRaw built from the 3rd_party/libraw submodule

PYTHON ?= python3

.PHONY: all lib libraw neg_process raw_info dinov3_model dinov3_engine test clean clean_lib clean_all
all: neg_process

# ----------------------------------------------------------------------
# LibRaw, pinned by the 3rd_party/libraw submodule (see 3rd_party/README.md) and built here as a static library, so
# neg_process and negicc-station decode with the same LibRaw on every machine instead of the system's libraw-dev.
# Consumers include "libraw/libraw.h" with -I$(LIBRAW_DIR) and link bin_out/libraw.a followed by the libraries in
# bin_out/libraw.ldlibs.
# ----------------------------------------------------------------------
LIBRAW_DIR = 3rd_party/libraw
LIBRAW_INC = -I$(LIBRAW_DIR)
LIBRAW_LIB = bin_out/libraw.a
ifeq ($(wildcard $(LIBRAW_DIR)/libraw/libraw.h),)
    $(error LibRaw submodule missing at $(LIBRAW_DIR): run git submodule update --init --recursive)
endif
# The library sources, as LibRaw's Makefile.dist lists them (every src/ file but the *_ph.cpp placeholders).
LIBRAW_SRCS = $(filter-out %_ph.cpp,$(wildcard $(LIBRAW_DIR)/src/*.cpp $(LIBRAW_DIR)/src/*/*.cpp))
LIBRAW_OBJS = $(patsubst %.cpp,bin_out/libraw/%.o,$(notdir $(LIBRAW_SRCS)))
vpath %.cpp $(sort $(dir $(LIBRAW_SRCS)))
# Lossy (JPEG-compressed) DNG support when libjpeg's headers are installed (libjpeg-dev); LIBRAW_JPEG=0 or 1 forces it.
LIBRAW_JPEG ?= $(shell printf '\043include <stdio.h>\n\043include <jpeglib.h>\n' | $(CXX) -E -x c++ - >/dev/null 2>&1 && echo 1 || echo 0)
LIBRAW_DEFS = -DUSE_ZLIB $(if $(filter 1,$(LIBRAW_JPEG)),-DUSE_JPEG -DUSE_JPEG8)
LIBRAW_LDLIBS = -lz $(if $(filter 1,$(LIBRAW_JPEG)),-ljpeg)
# Thread-safe build (no LIBRAW_NOTHREADS), with OpenMP as distribution packages build it. -ffp-contract=off as the
# rest of negicc, so aarch64 (where GCC fuses multiply-adds by default) decodes as x86 does.
LIBRAW_CXXFLAGS = -O3 -w -fPIC -fopenmp -ffp-contract=off $(LIBRAW_INC) $(LIBRAW_DEFS)

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
	echo "$(LIBRAW_LDLIBS)" > bin_out/libraw.ldlibs

libraw: $(LIBRAW_LIB)

raw_info: raw_info.cc $(LIBRAW_LIB)
	mkdir -p bin_out
	$(CXX) -o bin_out/raw_info raw_info.cc -I3rd_party $(LIBRAW_INC) $(LIBRAW_LIB) $(LIBRAW_LDLIBS) -fopenmp

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
ORT_INC = -I$(ORT_DIR)/include
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

DINOV3_CXXFLAGS = -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -fsigned-char -Wall -Wextra -I. $(ORT_INC) $(DINOV3_EXTRA_INC)

# The library negicc-station links (NEGICC_LIB): the DINOv3 engine and solver, and the conversion code shared with
# neg_process (profile bundles, cropping, flat field, zero-copy CFA pipeline, the conversion parameter search).
DINOV3_OBJS = bin_out/dinov3_engine.o bin_out/dinov3_preprocess.o bin_out/dinov3_ort_runner.o bin_out/dinov3_auto_solver.o \
              bin_out/flat_field.o bin_out/cfa_pipeline.o bin_out/film_frame_detector.o \
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

bin_out/dinov3_auto_solver.o: dinov3_auto_solver.cc dinov3_auto_solver.h cfa_pipeline.h color_pipeline.h neg_pipeline.h colorspace_conversion.h dinov3_engine.h dinov3_preprocess.h $(BUILD_STAMP)
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

bin_out/profile_bundle.o: profile_bundle.cc profile_bundle.h cfa_pipeline.h color_pipeline.h neg_pipeline.h dinov3_auto_solver.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c profile_bundle.cc -o $@

bin_out/multi_target_optimizer.o: multi_target_optimizer.cc multi_target_optimizer.h cfa_pipeline.h color_pipeline.h colorspace_conversion.h profile_bundle.h dinov3_auto_solver.h dinov3_engine.h neg_pipeline.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c multi_target_optimizer.cc -o $@

bin_out/libnegicc_dinov3.a: $(DINOV3_OBJS) $(BUILD_STAMP)
	mkdir -p bin_out
	rm -f $@
	ar rcs $@ $(DINOV3_OBJS)

lib: bin_out/libnegicc_dinov3.a $(LIBRAW_LIB)

GIT_VERSION ?= $(shell git describe --always --dirty --tags 2>/dev/null || echo "master")

bin_out/film_frame_detector.o: film_frame_detector.cc film_frame_detector.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -I. -c film_frame_detector.cc -o $@

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

# neg_process throws on a bundle it cannot read, so the bundles are checked before they are embedded: every bracket
# needs a shutter speed shutter_to_seconds (profile_bundle.cc) can read.
bin_out/.bundles_checked: $(filter %/bundle.json,$(EMBED_FILES)) check_profile_bundles.py | bin_out
	$(PYTHON) check_profile_bundles.py $(filter %/bundle.json,$(EMBED_FILES))
	touch $@

bin_out/embedded_data.S: $(EMBED_FILES) bin_out/.bundles_checked Makefile $(BUILD_STAMP) | bin_out
	$(file > $@,$(foreach i,$(EMBED_IDX),$(call EMBED_FILE_ASM,$(i),$(word $(i),$(EMBED_FILES))))$(EMBED_TABLE_ASM))

bin_out/embedded_data_files.o: bin_out/embedded_data.S
	$(CC) -c $< -o $@

bin_out/embedded_data.o: embedded_data.cc embedded_data.h profile_bundle.h Makefile $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -DEMBED_TABLE=$(EMBED_TABLE) -DEMBED_COUNT=$(EMBED_COUNT) \
	    -c embedded_data.cc -o $@

NEG_PROCESS_OBJS = bin_out/embedded_data.o bin_out/embedded_data_files.o

neg_process: bin_out/neg_process

bin_out/neg_process: neg_process.cc $(NEG_PROCESS_OBJS) bin_out/libnegicc_dinov3.a $(LIBRAW_LIB) neg_pipeline.h cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h dinov3_auto_solver.h film_frame_detector.h profile_bundle.h multi_target_optimizer.h embedded_data.h 3rd_party/tiff_head.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -DNEGICC_GIT_VERSION="\"$(GIT_VERSION)\"" -o $@ neg_process.cc $(NEG_PROCESS_OBJS) bin_out/libnegicc_dinov3.a $(LIBRAW_LIB) -I. -I3rd_party $(LIBRAW_INC) $(ORT_INC) $(DINOV3_EXTRA_INC) $(LIBRAW_LDLIBS) -lz -O3 -llcms2 -std=c++17 -fopenmp -ffp-contract=off $(ORT_LDFLAGS) $(DINOV3_EXTRA_LIBS)

# The DINOv3 backbone (shared by every film stock): exported from the pretrained weights with torch and
# transformers. DINOV3_SOURCE is a Hugging Face model id or a local model directory.
DINOV3_SOURCE ?= Tooony133/dinov3-vits16-pretrain-lvd1689m

dinov3_model:
	$(PYTHON) models/dinov3-small/export_dinov3_backbone.py --source $(DINOV3_SOURCE)

dinov3_engine:
	$(PYTHON) models/dinov3-small/export_dinov3_backbone.py --engine-only

TEST_CXXFLAGS = -std=c++17 -O2 -fopenmp -ffp-contract=off -fsigned-char -fsanitize=address,undefined -Wall -Wextra -I. -I3rd_party

bin_out/film_frame_detector_asan.o: film_frame_detector.cc film_frame_detector.h
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) -c film_frame_detector.cc -o $@

bin_out/cfa_pipeline_asan.o: cfa_pipeline.cc cfa_pipeline.h color_pipeline.h flat_field.h flat_sample.h film_frame_detector.h
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) -c cfa_pipeline.cc -o $@

bin_out/flat_field_asan.o: flat_field.cc flat_field.h flat_sample.h
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) -Wno-maybe-uninitialized -c flat_field.cc -o $@

bin_out/detector_test: tests/detector_test.cc bin_out/film_frame_detector_asan.o
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) tests/detector_test.cc bin_out/film_frame_detector_asan.o -o $@

bin_out/rgbview_test: tests/rgbview_test.cc bin_out/cfa_pipeline_asan.o bin_out/flat_field_asan.o bin_out/film_frame_detector_asan.o
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) tests/rgbview_test.cc bin_out/cfa_pipeline_asan.o bin_out/flat_field_asan.o bin_out/film_frame_detector_asan.o -lz -o $@

bin_out/kernels_test: tests/kernels_test.cc bin_out/cfa_pipeline_asan.o bin_out/flat_field_asan.o bin_out/film_frame_detector_asan.o
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) tests/kernels_test.cc bin_out/cfa_pipeline_asan.o bin_out/flat_field_asan.o bin_out/film_frame_detector_asan.o -lz -o $@

bin_out/dinov3_shape_test: tests/dinov3_shape_test.cc dinov3_preprocess.cc color_pipeline.h dinov3_preprocess.h
	mkdir -p bin_out
	$(CXX) $(TEST_CXXFLAGS) tests/dinov3_shape_test.cc dinov3_preprocess.cc -o $@

test: bin_out/detector_test bin_out/rgbview_test bin_out/kernels_test bin_out/dinov3_shape_test
	./bin_out/detector_test
	./bin_out/rgbview_test
	./bin_out/kernels_test
	./bin_out/dinov3_shape_test

clean: clean_lib
	rm -f bin_out/*_test

clean_lib:
	rm -f bin_out/*.o bin_out/*.a bin_out/*.S bin_out/.use_cuda_* bin_out/.bundles_checked

clean_all: clean_lib
	rm -rf bin_out/libraw bin_out/libraw.a bin_out/libraw.ldlibs bin_out/.libraw_jpeg_*
	rm -f bin_out/raw_info bin_out/neg_process
