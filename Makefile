# Sony A7R IV shots with triband filter.
data/portra400-0.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0.tif --multi --outfile=$@

data/portra400+2.txt:
	python3 read_it8.py --img=it8_imgs/portra400+2.tif --multi --outfile=$@

data/portra160-0.txt:
	python3 read_it8.py --img=it8_imgs/portra160-0.tif --multi --outfile=$@

data/ektar100-0.txt:
	python3 read_it8.py --img=it8_imgs/ektar100-0.tif --multi --outfile=$@

# Triband filter with BP470 bandpass filter
data/portra400-0-bp475.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0-bp475.tif --multi --outfile=$@

# Triband filter with BP525 bandpass filter
data/portra400-0-bp525.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0-bp525.tif --multi --outfile=$@

# Triband filter with LP610 bandpass filter
data/portra400-0-lp610.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0-lp610.tif --multi --outfile=$@

# Combine RGB shots with single shot into a single training file.
# This training set will allow us to compute the correction matrix.
data/portra400-0-cs100a_train.txt: data/portra400-0.txt \
	data/portra400-0-bp475.txt \
	data/portra400-0-bp525.txt \
	data/portra400-0-lp610.txt
	python3 add_ref_readings.py --r=data/portra400-0-lp610.txt --g=data/portra400-0-bp525.txt --b=data/portra400-0-bp475.txt --Yxy=data/cs100a_measurements.txt data/portra400-0.txt | tr ' ' ',' > $@

data/portra400+2-cs100a_train.txt: data/portra400+2.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400+2.txt | tr ' ' ',' > $@

data/portra160-0-cs100a_train.txt: data/portra160-0.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160-0.txt | tr ' ' ',' > $@

data/ektar100-0-cs100a_train.txt: data/ektar100-0.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100-0.txt | tr ' ' ',' > $@

data/portra400-0-r190808d55_train.txt: data/portra400-0.txt
	python3 add_ref_readings.py --XYZ=data/R190808D55.txt data/portra400-0.txt | tr ' ' ',' > $@

data/portra400+2-r190808d55_train.txt: data/portra400+2.txt
	python3 add_ref_readings.py --XYZ=data/R190808D55.txt data/portra400+2.txt | tr ' ' ',' > $@

data/portra160-0-r190808d55_train.txt: data/portra160-0.txt
	python3 add_ref_readings.py --XYZ=data/R190808D55.txt data/portra160-0.txt | tr ' ' ',' > $@

data/ektar100-0-r190808d55_train.txt: data/ektar100-0.txt
	python3 add_ref_readings.py --XYZ=data/R190808D55.txt data/ektar100-0.txt | tr ' ' ',' > $@

# Test white chromaticies are common for all film as this is fixed during test time.
# Change these values to the one used during the test environment. The following
# values are measured under sunlight at around 5400K.
test_white_xy = --white_x=0.3353 --white_y=0.3496 

# Sony A7R IV profiles.
.PHONY: sony_a7rm4_portra400_0
sony_a7rm4_portra400_0: build_prof.py make_icc.c data/portra400-0-cs100a_train.txt
	python3 build_prof.py --src=data/portra400-0-cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400" --fit_intercept=1 --debug --install_dir="$(INSTALL_DIR)"

# Coefficients copied from the above step.
sony_a7rm4_triband_crosstalk_coefs = --crosstalk_r_coefs='1 -0.08262711 -0.01249409' --crosstalk_g_coefs='-0.13898878 1 -0.32017315' --crosstalk_b_coefs='-0.00664173 -0.09860774 1'

.PHONY: sony_a7rm4_portra160_0
sony_a7rm4_portra160_0: build_prof.py make_icc.c data/portra160-0-cs100a_train.txt
	python3 build_prof.py --src=data/portra160-0-cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra160" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

.PHONY: sony_a7rm4_ektar100_0
sony_a7rm4_ektar100_0: build_prof.py make_icc.c data/ektar100-0-cs100a_train.txt
	python3 build_prof.py --src=data/ektar100-0-cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Ektar100" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

.PHONY: sony_a7rm4_ektar100_0_r190808d55
sony_a7rm4_ektar100_0_r190808d55: build_prof.py make_icc.c data/ektar100-0-r190808d55_train.txt
	python3 build_prof.py --src=data/ektar100-0-r190808d55_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Ektar100 R190808D55" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

.PHONY: sony_a7rm4_portra160_0_r190808d55
sony_a7rm4_portra160_0_r190808d55: build_prof.py make_icc.c data/portra160-0-r190808d55_train.txt
	python3 build_prof.py --src=data/portra160-0-r190808d55_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra160 R190808D55" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

.PHONY: sony_a7rm4_portra400_0_r190808d55
sony_a7rm4_portra400_0_r190808d55: build_prof.py make_icc.c data/portra400-0-r190808d55_train.txt
	python3 build_prof.py --src=data/portra400-0-r190808d55_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400 R190808D55" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

.PHONY: sony_a7rm4_portra400+2_r190808d55
sony_a7rm4_portra400+2_r190808d55: build_prof.py make_icc.c data/portra400+2-r190808d55_train.txt
	python3 build_prof.py --src=data/portra400+2-r190808d55_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400+2 R190808D55" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

.PHONY: sony_a7rm4_portra400+2
sony_a7rm4_portra400+2: build_prof.py make_icc.c data/portra400+2-cs100a_train.txt
	python3 build_prof.py --src=data/portra400+2-cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400 +2" --fit_intercept=1 $(sony_a7rm4_triband_crosstalk_coefs) --debug --install_dir="$(INSTALL_DIR)"

make_icc: make_icc.c
	mkdir -p bin_out
	gcc -o bin_out/make_icc make_icc.c -llcms2

neg_process: neg_process.cc
	mkdir -p bin_out
	g++ -o bin_out/neg_process neg_process.cc -I/usr/local/opt/curl/include -L/usr/local/opt/curl/lib -lraw
