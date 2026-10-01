PYTHON ?= python3

# Sony A7R IV shots with triband filter.
data/portra400-0.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400-0.tif --outfile=$@

data/portra400-0.5.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400-0.5.tif --outfile=$@

data/portra400-1.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400-1.tif --outfile=$@

data/portra400+1.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400+1.tif --outfile=$@

data/portra400+2.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400+2.tif --outfile=$@

data/portra400+3.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400+3.tif --outfile=$@

data/portra400+3.5.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400+3.5.tif --outfile=$@

data/portra160-0.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra160-0.tif --outfile=$@

data/portra160-1.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra160-1.tif --outfile=$@

data/portra160+1.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra160+1.tif --outfile=$@

data/portra160+2.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra160+2.tif --outfile=$@

data/ektar100-0.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100-0.tif --outfile=$@

data/ektar100-1.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100-1.tif --outfile=$@

data/ektar100-2.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100-2.tif --outfile=$@

data/ektar100-3.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100-3.tif --outfile=$@

data/ektar100+1.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100+1.tif --outfile=$@

data/ektar100+2.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100+2.tif --outfile=$@

data/ektar100+3.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/ektar100+3.tif --outfile=$@

# Triband filter with BP470 bandpass filter
# TODO: Redo this with same light source as portra400-0.tif capture.
data/portra400-0-bp475.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400-0-bp475.tif --multi --outfile=$@

# Triband filter with BP525 bandpass filter
# TODO: Redo this with same light source as portra400-0.tif capture.
data/portra400-0-bp525.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400-0-bp525.tif --multi --outfile=$@

# Triband filter with LP610 bandpass filter
# TODO: Redo this with same light source as portra400-0.tif capture.
data/portra400-0-lp610.txt:
	$(PYTHON) read_it8.py --img=it8_imgs/portra400-0-lp610.tif --multi --outfile=$@

# Combine RGB shots with single shot into a single training file.
# This training set will allow us to compute the correction matrix.
# TODO: These need to be redone since the portra400-0.tif is done with different light source than the
# bandpass filtered captures.
#data/portra400-0-cs100a_train.txt: data/portra400-0.txt \
#	data/portra400-0-bp475.txt \
#	data/portra400-0-bp525.txt \
#	data/portra400-0-lp610.txt
#	$(PYTHON) add_ref_readings.py --r=data/portra400-0-lp610.txt --g=data/portra400-0-bp525.txt --b=data/portra400-0-bp475.txt --Yxy=data/cs100a_measurements.txt data/portra400-0.txt | tr ' ' ',' > $@

data/portra400+2-cs100a_train.txt: data/portra400+2.txt
	$(PYTHON) add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400+2.txt | tr ' ' ',' > $@

data/portra160-0-cs100a_train.txt: data/portra160-0.txt
	$(PYTHON) add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160-0.txt | tr ' ' ',' > $@

data/ektar100-0-cs100a_train.txt: data/ektar100-0.txt
	$(PYTHON) add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100-0.txt | tr ' ' ',' > $@

data/portra400-0-r190808_train.json: data/portra400-0.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400-0-r190808_train.txt: data/portra400-0.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400-0.5-r190808_train.json: data/portra400-0.5.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400-0.5-r190808_train.txt: data/portra400-0.5.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400-1-r190808_train.json: data/portra400-1.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400-1-r190808_train.txt: data/portra400-1.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400+1-r190808_train.json: data/portra400+1.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400+1-r190808_train.txt: data/portra400+1.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400+2-r190808_train.json: data/portra400+2.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400+2-r190808_train.txt: data/portra400+2.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400+3-r190808_train.json: data/portra400+3.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400+3-r190808_train.txt: data/portra400+3.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400+3.5-r190808_train.json: data/portra400+3.5.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt --json $< > $@

data/portra400+3.5-r190808_train.txt: data/portra400+3.5.txt data/R190808.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160-0-r190808_train.txt: data/portra160-0.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160-1-r190808_train.txt: data/portra160-1.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160+1-r190808_train.txt: data/portra160+1.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160+2-r190808_train.txt: data/portra160+2.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-0-r190808_train.txt: data/ektar100-0.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-1-r190808_train.txt: data/ektar100-1.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-2-r190808_train.txt: data/ektar100-2.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-3-r190808_train.txt: data/ektar100-3.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100+1-r190808_train.txt: data/ektar100+1.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100+2-r190808_train.txt: data/ektar100+2.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100+3-r190808_train.txt: data/ektar100+3.txt
	$(PYTHON) add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

# Test white chromaticies are common for all film as this is fixed during test time.
# Change these values to the one used during the test environment. The following
# values are measured under sunlight at around 5400K.
test_white_xy = --white_x=0.3353 --white_y=0.3496 

# Sony A7R IV profiles. 
.PHONY: sony_a7rm4_portra400_0
sony_a7rm4_portra400_0: data/portra400-0-cs100a_train.txt
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400" --fit_intercept=1

# Coefficients copied from the above step.
# TODO: The portra400-0 it8 has ben updated and so the bandpass filter captures
# should be updated too. But the coefficients shouldn't change because they are
# intrinsic to the sensor color filters and the triband filter combination.
sony_a7rm4_triband_crosstalk_coefs = --crosstalk_r_coefs='1 -0.08262711 -0.01249409' --crosstalk_g_coefs='-0.13898878 1 -0.32017315' --crosstalk_b_coefs='-0.00664173 -0.09860774 1'

# Identity matrix, let the cLUT handle it.
sony_a7rm4_triband_portra400_crosstalk = --bypass_crosstalk

# Uncorrected RGB values of the film base, scaled by 1 / shutter speed (from negicc-station)
sony_a7rm4_triband_ektar100_film_base_rgb = --film_base_rgb='67375 104355 106090'
sony_a7rm4_triband_portra400_film_base_rgb = --film_base_rgb='84871 46183 73060'
sony_a7rm4_triband_portra160_film_base_rgb = --film_base_rgb='61990 94994 81022'

.PHONY: sony_a7rm4_portra400_0_r190808
sony_a7rm4_portra400_0_r190808: data/portra400-0-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.125000

.PHONY: sony_a7rm4_portra400-0.5_r190808
sony_a7rm4_portra400-0.5_r190808: data/portra400-0.5-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400-0.5 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.125000

.PHONY: sony_a7rm4_portra400-1_r190808
sony_a7rm4_portra400-1_r190808: data/portra400-1-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400-1 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.125000

.PHONY: sony_a7rm4_portra400+1_r190808
sony_a7rm4_portra400+1_r190808: data/portra400+1-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400+1 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.166667

.PHONY: sony_a7rm4_portra400+2_r190808
sony_a7rm4_portra400+2_r190808: data/portra400+2-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400+2 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.200000

.PHONY: sony_a7rm4_portra400+3_r190808
sony_a7rm4_portra400+3_r190808: data/portra400+3-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400+3 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.250000

.PHONY: sony_a7rm4_portra400+3.5_r190808
sony_a7rm4_portra400+3.5_r190808: data/portra400+3.5-r190808_train.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400+3.5 R190808" $(sony_a7rm4_triband_portra400_crosstalk) $(sony_a7rm4_triband_portra400_film_base_rgb) --shutter_speed=0.400000

.PHONY: sony_a7rm4_portra400_bundle
sony_a7rm4_portra400_bundle: profiles/profile_Portra400_20260927_035659.json clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --target="Target 5" --film_name="Sony A7RM4 Portra400 R190808" --bypass_crosstalk

.PHONY: sony_a7rm4_portra160_0
sony_a7rm4_portra160_0: data/portra160-0-cs100a_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra160" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --debug

.PHONY: sony_a7rm4_portra160-1_r190808
sony_a7rm4_portra160-1_r190808: data/portra160-1-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160-1 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --shutter_speed=0.066667

.PHONY: sony_a7rm4_portra160_0_r190808
sony_a7rm4_portra160_0_r190808: data/portra160-0-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs)  --shutter_speed=0.076923

.PHONY: sony_a7rm4_portra160+1_r190808
sony_a7rm4_portra160+1_r190808: data/portra160+1-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160+1 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --shutter_speed=0.11860

.PHONY: sony_a7rm4_portra160+2_r190808
sony_a7rm4_portra160+2_r190808: data/portra160+2-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160+2 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --shutter_speed=0.166667

.PHONY: sony_a7rm4_ektar100_0
sony_a7rm4_ektar100_0: data/ektar100-0-cs100a_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Ektar100"  $(sony_a7rm4_triband_crosstalk_coefs) --debug

.PHONY: sony_a7rm4_ektar100_0_r190808
sony_a7rm4_ektar100_0_r190808: data/ektar100-0-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.2

.PHONY: sony_a7rm4_ektar100-1_r190808
sony_a7rm4_ektar100-1_r190808: data/ektar100-1-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100-1 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.1

.PHONY: sony_a7rm4_ektar100-2_r190808
sony_a7rm4_ektar100-2_r190808: data/ektar100-2-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100-2 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.1

.PHONY: sony_a7rm4_ektar100-3_r190808
sony_a7rm4_ektar100-3_r190808: data/ektar100-3-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100-3 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.06667

.PHONY: sony_a7rm4_ektar100+1_r190808
sony_a7rm4_ektar100+1_r190808: data/ektar100+1-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100+1 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.2

.PHONY: sony_a7rm4_ektar100+2_r190808
sony_a7rm4_ektar100+2_r190808: data/ektar100+2-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100+2 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.33333

.PHONY: sony_a7rm4_ektar100+3_r190808
sony_a7rm4_ektar100+3_r190808: data/ektar100+3-r190808_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100+3 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.4

.PHONY: sony_a7rm4_portra400+2
sony_a7rm4_portra400+2: data/portra400+2-cs100a_train.txt clean
	$(PYTHON) build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400 +2"  $(sony_a7rm4_triband_crosstalk_coefs) --debug

.PHONY: clean lib neg_process
clean:
	rm -f *_prof.ti3 build_prof_diag.csv build_prof.h
	rm -f bin_out/*.o bin_out/*.a bin_out/.use_cuda_*

.PHONY: sony_a7rm4_portra400_r190808_all
sony_a7rm4_portra400_r190808_all: \
	sony_a7rm4_portra400+3.5_r190808 \
	sony_a7rm4_portra400+3_r190808 \
	sony_a7rm4_portra400+2_r190808 \
	sony_a7rm4_portra400+1_r190808 \
	sony_a7rm4_portra400_0_r190808 \
	sony_a7rm4_portra400-0.5_r190808 \
	sony_a7rm4_portra400-1_r190808

.PHONY: sony_a7rm4_portra160_all
sony_a7rm4_portra160_r190808_all: \
	sony_a7rm4_portra160+2_r190808 \
	sony_a7rm4_portra160+1_r190808 \
	sony_a7rm4_portra160_0_r190808 \
	sony_a7rm4_portra160-1_r190808

.PHONY: sony_a7rm4_ektar100_all
sony_a7rm4_ektar100_r190808_all: \
	sony_a7rm4_ektar100+3_r190808 \
	sony_a7rm4_ektar100+2_r190808 \
	sony_a7rm4_ektar100+1_r190808 \
	sony_a7rm4_ektar100_0_r190808 \
	sony_a7rm4_ektar100-1_r190808 \
	sony_a7rm4_ektar100-2_r190808 \
	sony_a7rm4_ektar100-3_r190808

make_icc: make_icc.c
	mkdir -p bin_out profiles
	$(CC) -o bin_out/make_icc make_icc.c -llcms2

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
ORT_LDFLAGS = -L$(ORT_LIB_DIR) -lonnxruntime -Wl,-rpath,'$$ORIGIN/../$(ORT_LIB_DIR)'

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

DINOV3_OBJS = bin_out/dinov3_engine.o bin_out/dinov3_preprocess.o bin_out/dinov3_ort_runner.o $(DINOV3_RUNNER_OBJ)

bin_out/dinov3_engine.o: dinov3_engine.cc dinov3_engine.h dinov3_runner.h dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_engine.cc -o $@

bin_out/dinov3_preprocess.o: dinov3_preprocess.cc dinov3_preprocess.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_preprocess.cc -o $@

bin_out/dinov3_ort_runner.o: dinov3_ort_runner.cc dinov3_ort_runner.h dinov3_runner.h $(BUILD_STAMP)
	mkdir -p bin_out
	$(CXX) $(DINOV3_CXXFLAGS) -c dinov3_ort_runner.cc -o $@

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

neg_process: bin_out/neg_process

bin_out/neg_process: neg_process.cc bin_out/libnegicc_dinov3.a $(BUILD_STAMP)
	mkdir -p bin_out profiles
	$(CXX) -o $@ neg_process.cc bin_out/libnegicc_dinov3.a -I. -I3rd_party $(ORT_INC) $(DINOV3_EXTRA_INC) -lraw -lz -O3 -llcms2 -std=c++17 -fopenmp $(ORT_LDFLAGS) $(DINOV3_EXTRA_LIBS)

