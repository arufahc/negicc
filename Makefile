# Sony A7R IV shots with triband filter.
data/portra400-0.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0.tif --outfile=$@

data/portra400-1.txt:
	python3 read_it8.py --img=it8_imgs/portra400-1.tif --outfile=$@

data/portra400+1.txt:
	python3 read_it8.py --img=it8_imgs/portra400+1.tif --outfile=$@

data/portra400+2.txt:
	python3 read_it8.py --img=it8_imgs/portra400+2.tif --outfile=$@

data/portra160-0.txt:
	python3 read_it8.py --img=it8_imgs/portra160-0.tif --outfile=$@

data/portra160-1.txt:
	python3 read_it8.py --img=it8_imgs/portra160-1.tif --outfile=$@

data/portra160+1.txt:
	python3 read_it8.py --img=it8_imgs/portra160+1.tif --outfile=$@

data/portra160+2.txt:
	python3 read_it8.py --img=it8_imgs/portra160+2.tif --outfile=$@

data/ektar100-0.txt:
	python3 read_it8.py --img=it8_imgs/ektar100-0.tif --outfile=$@

data/ektar100-1.txt:
	python3 read_it8.py --img=it8_imgs/ektar100-1.tif --outfile=$@

data/ektar100-2.txt:
	python3 read_it8.py --img=it8_imgs/ektar100-2.tif --outfile=$@

data/ektar100-3.txt:
	python3 read_it8.py --img=it8_imgs/ektar100-3.tif --outfile=$@

data/ektar100+1.txt:
	python3 read_it8.py --img=it8_imgs/ektar100+1.tif --outfile=$@

data/ektar100+2.txt:
	python3 read_it8.py --img=it8_imgs/ektar100+2.tif --outfile=$@

# Triband filter with BP470 bandpass filter
# TODO: Redo this with same light source as portra400-0.tif capture.
data/portra400-0-bp475.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0-bp475.tif --multi --outfile=$@

# Triband filter with BP525 bandpass filter
# TODO: Redo this with same light source as portra400-0.tif capture.
data/portra400-0-bp525.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0-bp525.tif --multi --outfile=$@

# Triband filter with LP610 bandpass filter
# TODO: Redo this with same light source as portra400-0.tif capture.
data/portra400-0-lp610.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0-lp610.tif --multi --outfile=$@

# Combine RGB shots with single shot into a single training file.
# This training set will allow us to compute the correction matrix.
# TODO: These need to be redone since the portra400-0.tif is done with different light source than the
# bandpass filtered captures.
#data/portra400-0-cs100a_train.txt: data/portra400-0.txt \
#	data/portra400-0-bp475.txt \
#	data/portra400-0-bp525.txt \
#	data/portra400-0-lp610.txt
#	python3 add_ref_readings.py --r=data/portra400-0-lp610.txt --g=data/portra400-0-bp525.txt --b=data/portra400-0-bp475.txt --Yxy=data/cs100a_measurements.txt data/portra400-0.txt | tr ' ' ',' > $@

data/portra400+2-cs100a_train.txt: data/portra400+2.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400+2.txt | tr ' ' ',' > $@

data/portra160-0-cs100a_train.txt: data/portra160-0.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160-0.txt | tr ' ' ',' > $@

data/ektar100-0-cs100a_train.txt: data/ektar100-0.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100-0.txt | tr ' ' ',' > $@

data/portra400-0-r190808_train.txt: data/portra400-0.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400-1-r190808_train.txt: data/portra400-1.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400+1-r190808_train.txt: data/portra400+1.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra400+2-r190808_train.txt: data/portra400+2.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160-0-r190808_train.txt: data/portra160-0.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160-1-r190808_train.txt: data/portra160-1.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160+1-r190808_train.txt: data/portra160+1.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/portra160+2-r190808_train.txt: data/portra160+2.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-0-r190808_train.txt: data/ektar100-0.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-1-r190808_train.txt: data/ektar100-1.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-2-r190808_train.txt: data/ektar100-2.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100-3-r190808_train.txt: data/ektar100-3.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100+1-r190808_train.txt: data/ektar100+1.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

data/ektar100+2-r190808_train.txt: data/ektar100+2.txt
	python3 add_ref_readings.py --XYZ=data/R190808.txt $< | tr ' ' ',' > $@

# Test white chromaticies are common for all film as this is fixed during test time.
# Change these values to the one used during the test environment. The following
# values are measured under sunlight at around 5400K.
test_white_xy = --white_x=0.3353 --white_y=0.3496 

# Sony A7R IV profiles. 
.PHONY: sony_a7rm4_portra400_0
sony_a7rm4_portra400_0: data/portra400-0-cs100a_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400" --fit_intercept=1 --debug

# Coefficients copied from the above step.
# TODO: The portra400-0 it8 has ben updated and so the bandpass filter captures
# should be updated too. But the coefficients shouldn't change because they are
# intrinsic to the sensor color filters and the triband filter combination.
sony_a7rm4_triband_crosstalk_coefs = --crosstalk_r_coefs='1 -0.08262711 -0.01249409' --crosstalk_g_coefs='-0.13898878 1 -0.32017315' --crosstalk_b_coefs='-0.00664173 -0.09860774 1'

# These are linear and uncorrected RGB values of the film base, multiplied by 1 / shutter speed.
# TODO: Use raw_info to compute these into a data file.
sony_a7rm4_triband_ektar100_film_base_rgb = --film_base_rgb='384120 608800 594690'
sony_a7rm4_triband_portra400_film_base_rgb = --film_base_rgb='264403 400437 330058'
sony_a7rm4_triband_portra160_film_base_rgb = --film_base_rgb='256724 396539 338117'

.PHONY: sony_a7rm4_portra400_0_r190808
sony_a7rm4_portra400_0_r190808: data/portra400-0-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400 R190808" $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_portra400_film_base_rgb) --debug --shutter_speed=0.076923

.PHONY: sony_a7rm4_portra400-1_r190808
sony_a7rm4_portra400-1_r190808: data/portra400-1-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400-1 R190808" $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_portra400_film_base_rgb) --debug --shutter_speed=0.066667

.PHONY: sony_a7rm4_portra400+1_r190808
sony_a7rm4_portra400+1_r190808: data/portra400+1-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400+1 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_portra400_film_base_rgb) --debug --shutter_speed=0.100000

.PHONY: sony_a7rm4_portra400+2_r190808
sony_a7rm4_portra400+2_r190808: data/portra400+2-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra400+2 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) $(sony_a7rm4_triband_portra400_film_base_rgb) --debug --shutter_speed=0.150219

.PHONY: sony_a7rm4_portra160_0
sony_a7rm4_portra160_0: data/portra160-0-cs100a_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra160" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --debug

.PHONY: sony_a7rm4_portra160-1_r190808
sony_a7rm4_portra160-1_r190808: data/portra160-1-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160-1 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --debug --shutter_speed=0.066667

.PHONY: sony_a7rm4_portra160_0_r190808
sony_a7rm4_portra160_0_r190808: data/portra160-0-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --debug  --shutter_speed=0.076923

.PHONY: sony_a7rm4_portra160+1_r190808
sony_a7rm4_portra160+1_r190808: data/portra160+1-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160+1 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --debug --shutter_speed=0.1563101467

.PHONY: sony_a7rm4_portra160+2_r190808
sony_a7rm4_portra160+2_r190808: data/portra160+2-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Portra160+2 R190808" $(sony_a7rm4_triband_portra160_film_base_rgb) $(sony_a7rm4_triband_crosstalk_coefs) --debug --shutter_speed=0.166667

a.PHONY: sony_a7rm4_ektar100_0
sony_a7rm4_ektar100_0: data/ektar100-0-cs100a_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Ektar100"  $(sony_a7rm4_triband_crosstalk_coefs) --debug

.PHONY: sony_a7rm4_ektar100_0_r190808
sony_a7rm4_ektar100_0_r190808: data/ektar100-0-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) --debug $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.2

.PHONY: sony_a7rm4_ektar100-1_r190808
sony_a7rm4_ektar100-1_r190808: data/ektar100-1-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100-1 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) --debug $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.1

.PHONY: sony_a7rm4_ektar100-2_r190808
sony_a7rm4_ektar100-2_r190808: data/ektar100-2-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100-2 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) --debug $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.1 --whitest_patch_scaling=0.5

.PHONY: sony_a7rm4_ektar100-3_r190808
sony_a7rm4_ektar100-3_r190808: data/ektar100-3-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100-3 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) --debug $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.06667 --whitest_patch_scaling=0.3

.PHONY: sony_a7rm4_ektar100+1_r190808
sony_a7rm4_ektar100+1_r190808: data/ektar100+1-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100+1 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) --debug $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.2

.PHONY: sony_a7rm4_ektar100+2_r190808
sony_a7rm4_ektar100+2_r190808: data/ektar100+2-r190808_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --film_name="Sony A7RM4 Ektar100+2 R190808"  $(sony_a7rm4_triband_crosstalk_coefs) --debug $(sony_a7rm4_triband_ektar100_film_base_rgb) --shutter_speed=0.33333

.PHONY: sony_a7rm4_portra400+2
sony_a7rm4_portra400+2: data/portra400+2-cs100a_train.txt make_icc
	python3 build_prof.py ${BUILD_PROF_FLAGS} --src=$< --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400 +2"  $(sony_a7rm4_triband_crosstalk_coefs) --debug

make_icc: make_icc.c
	mkdir -p bin_out
	clang -o bin_out/make_icc make_icc.c -llcms2

raw_info: raw_info.cc
	mkdir -p bin_out
	clang++ -o bin_out/raw_info raw_info.cc -lraw

neg_process: neg_process.cc
	mkdir -p bin_out
	clang++ -o bin_out/neg_process neg_process.cc -I/usr/local/opt/curl/include -I3rd_party -L/usr/local/opt/curl/lib -lraw -lz -O3 -llcms2 -std=c++17 -DCMS_NO_REGISTER_KEYWORD
