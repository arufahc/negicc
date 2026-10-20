# neg_process converts film negative RAW scans; the profiles are built in negicc-station.

.PHONY: clean clean_lib clean_all lib neg_process raw_info
clean: clean_lib

clean_lib:
	rm -f bin_out/*.o bin_out/*.a bin_out/.use_cuda_*

clean_all: clean_lib
	rm -f bin_out/raw_info bin_out/neg_process

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

DINOV3_OBJS = bin_out/dinov3_engine.o bin_out/dinov3_preprocess.o bin_out/dinov3_ort_runner.o bin_out/dinov3_auto_solver.o $(DINOV3_RUNNER_OBJ)

bin_out/dinov3_engine.o: dinov3_engine.cc dinov3_engine.h dinov3_runner.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_engine.cc -o $@

bin_out/dinov3_preprocess.o: dinov3_preprocess.cc dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_preprocess.cc -o $@

bin_out/dinov3_ort_runner.o: dinov3_ort_runner.cc dinov3_ort_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_ort_runner.cc -o $@

bin_out/dinov3_auto_solver.o: dinov3_auto_solver.cc dinov3_auto_solver.h neg_pipeline.h dinov3_engine.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_auto_solver.cc -o $@

bin_out/dinov3_trt_runner.o: dinov3_trt_runner.cc dinov3_trt_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_trt_runner.cc -o $@

bin_out/dinov3_nocuda.o: dinov3_nocuda.cc dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_nocuda.cc -o $@

bin_out/libnegicc_dinov3.a: $(DINOV3_OBJS) $(BUILD_STAMP)
	mkdir -p bin_out
	rm -f $@
	ar rcs $@ $(DINOV3_OBJS)

lib: bin_out/libnegicc_dinov3.a

GIT_VERSION ?= $(shell git describe --always --dirty --tags 2>/dev/null || echo "master")

bin_out/film_frame_detector.o: film_frame_detector.cc film_frame_detector.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -I. -c film_frame_detector.cc -o $@

bin_out/profile_bundle.o: profile_bundle.cc profile_bundle.h neg_pipeline.h dinov3_auto_solver.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -I. -I3rd_party -c profile_bundle.cc -o $@

neg_process: bin_out/neg_process

bin_out/neg_process: neg_process.cc bin_out/film_frame_detector.o bin_out/profile_bundle.o bin_out/libnegicc_dinov3.a neg_pipeline.h dinov3_auto_solver.h film_frame_detector.h profile_bundle.h $(BUILD_STAMP)
	mkdir -p bin_out profiles
	$(CXX) -DNEGICC_GIT_VERSION="\"$(GIT_VERSION)\"" -o $@ neg_process.cc bin_out/film_frame_detector.o bin_out/profile_bundle.o bin_out/libnegicc_dinov3.a -I. -I3rd_party $(ORT_INC) $(DINOV3_EXTRA_INC) -lraw -lz -O3 -llcms2 -std=c++17 -fopenmp -ffp-contract=off $(ORT_LDFLAGS) $(DINOV3_EXTRA_LIBS)


