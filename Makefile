# neg_process: converts film negative RAW scans with the profiles and intent adapters built into it.
#
#   make dinov3_model       # models/dinov3-small/backbone.onnx, the DINOv3 backbone of the parameter search
#   make dinov3_engine      # its TensorRT engine, for USE_CUDA=1 builds (on the machine that runs it)
#   make neg_process        # bin_out/neg_process, with the backbone built in when it was exported
#   make lib                # bin_out/libnegicc_dinov3.a, linked by negicc-station

PYTHON ?= python3

.PHONY: all lib neg_process raw_info dinov3_model dinov3_engine clean clean_lib clean_all
all: neg_process

raw_info: raw_info.cc
	mkdir -p bin_out
	$(CXX) -o bin_out/raw_info raw_info.cc -lraw

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
# neg_process (profile bundles, cropping, the parameter search measurements, the automatic conversion).
DINOV3_OBJS = bin_out/dinov3_engine.o bin_out/dinov3_preprocess.o bin_out/dinov3_ort_runner.o bin_out/dinov3_auto_solver.o \
              bin_out/profile_bundle.o bin_out/frame_crop.o bin_out/param_search.o bin_out/auto_convert.o \
              $(DINOV3_RUNNER_OBJ)

bin_out/dinov3_engine.o: dinov3_engine.cc dinov3_engine.h dinov3_runner.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_engine.cc -o $@

bin_out/dinov3_preprocess.o: dinov3_preprocess.cc dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_preprocess.cc -o $@

bin_out/dinov3_ort_runner.o: dinov3_ort_runner.cc dinov3_ort_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_ort_runner.cc -o $@

bin_out/dinov3_auto_solver.o: dinov3_auto_solver.cc dinov3_auto_solver.h neg_pipeline.h param_search.h frame_crop.h dinov3_engine.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_auto_solver.cc -o $@

bin_out/dinov3_trt_runner.o: dinov3_trt_runner.cc dinov3_trt_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_trt_runner.cc -o $@

bin_out/dinov3_nocuda.o: dinov3_nocuda.cc dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_nocuda.cc -o $@

bin_out/profile_bundle.o: profile_bundle.cc profile_bundle.h neg_pipeline.h dinov3_auto_solver.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c profile_bundle.cc -o $@

bin_out/frame_crop.o: frame_crop.cc frame_crop.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c frame_crop.cc -o $@

bin_out/param_search.o: param_search.cc param_search.h frame_crop.h neg_pipeline.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c param_search.cc -o $@

bin_out/auto_convert.o: auto_convert.cc auto_convert.h param_search.h frame_crop.h profile_bundle.h dinov3_auto_solver.h dinov3_engine.h dinov3_preprocess.h neg_pipeline.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -c auto_convert.cc -o $@

bin_out/libnegicc_dinov3.a: $(DINOV3_OBJS) $(BUILD_STAMP)
	mkdir -p bin_out
	rm -f $@
	ar rcs $@ $(DINOV3_OBJS)

lib: bin_out/libnegicc_dinov3.a

GIT_VERSION ?= $(shell git describe --always --dirty --tags 2>/dev/null || echo "master")

bin_out/film_frame_detector.o: film_frame_detector.cc film_frame_detector.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -I. -c film_frame_detector.cc -o $@

# Data files built into neg_process: the profile bundles and the intent adapters, exported from negicc-station
# (tools/export_negicc_data.py), and the DINOv3 backbone: its TensorRT engine in USE_CUDA=1 builds (make
# dinov3_engine, on the machine that runs it), else backbone.onnx (make dinov3_model). Without the backbone
# neg_process still builds, but converts only with --gain_g or --gain_b given.
BACKBONE_FILE = models/dinov3-small/backbone.$(if $(filter 1,$(USE_CUDA)),engine,onnx)
EMBED_FILES = $(sort $(wildcard profiles/*/bundle.json profiles/*/*.icc models/dinov3-small_*/adapter.onnx)) \
              $(wildcard $(BACKBONE_FILE))

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

bin_out/embedded_data.S: $(EMBED_FILES) Makefile $(BUILD_STAMP) | bin_out
	$(file > $@,$(foreach i,$(EMBED_IDX),$(call EMBED_FILE_ASM,$(i),$(word $(i),$(EMBED_FILES))))$(EMBED_TABLE_ASM))

bin_out/embedded_data_files.o: bin_out/embedded_data.S
	$(CC) -c $< -o $@

bin_out/embedded_data.o: embedded_data.cc embedded_data.h profile_bundle.h Makefile $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -I3rd_party -DEMBED_TABLE=$(EMBED_TABLE) -DEMBED_COUNT=$(EMBED_COUNT) \
	    -c embedded_data.cc -o $@

NEG_PROCESS_OBJS = bin_out/film_frame_detector.o bin_out/embedded_data.o bin_out/embedded_data_files.o

neg_process: bin_out/neg_process

bin_out/neg_process: neg_process.cc $(NEG_PROCESS_OBJS) bin_out/libnegicc_dinov3.a neg_pipeline.h dinov3_auto_solver.h film_frame_detector.h profile_bundle.h auto_convert.h frame_crop.h param_search.h embedded_data.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -DNEGICC_GIT_VERSION="\"$(GIT_VERSION)\"" -o $@ neg_process.cc $(NEG_PROCESS_OBJS) bin_out/libnegicc_dinov3.a -I. -I3rd_party $(ORT_INC) $(DINOV3_EXTRA_INC) -lraw -lz -O3 -llcms2 -std=c++17 -fopenmp -ffp-contract=off $(ORT_LDFLAGS) $(DINOV3_EXTRA_LIBS)

# The DINOv3 backbone (shared by every film stock): exported from the pretrained weights with torch and
# transformers. DINOV3_SOURCE is a Hugging Face model id or a local model directory.
DINOV3_SOURCE ?= Tooony133/dinov3-vits16-pretrain-lvd1689m

dinov3_model:
	$(PYTHON) models/dinov3-small/export_dinov3_backbone.py --source $(DINOV3_SOURCE)

dinov3_engine:
	$(PYTHON) models/dinov3-small/export_dinov3_backbone.py --engine-only

clean: clean_lib

clean_lib:
	rm -f bin_out/*.o bin_out/*.a bin_out/*.S bin_out/.use_cuda_*

clean_all: clean_lib
	rm -f bin_out/raw_info bin_out/neg_process
