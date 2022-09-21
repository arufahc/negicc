# Nikon Z7 shots of target with 30M CP80C light filtration.
data/ektar100_it8_30m_cp80c_triband.txt:
	python3 read_it8.py --img=it8_imgs/ektar100_it8_30m_cp80c_triband.tiff --outfile=$@ --a1_x=73 --gs0_x=20 --hbase=1310	

data/ektar100_it8_30m_cp80c_triband_bp470.txt:
	python3 read_it8.py --img=it8_imgs/ektar100_it8_30m_cp80c_triband_bp470.tiff --outfile=$@ --a1_x=73 --gs0_x=16 --a1_y=70 --hbase=1310 --vbase=860 --gs0_y=780

data/ektar100_it8_30m_cp80c_triband_bp525.txt:
	python3 read_it8.py --img=it8_imgs/ektar100_it8_30m_cp80c_triband_bp525.tiff --outfile=$@ --a1_x=73 --gs0_x=16 --a1_y=70 --hbase=1310 --vbase=860 --gs0_y=780

data/ektar100_it8_30m_cp80c_triband_lp610.txt:
	python3 read_it8.py --img=it8_imgs/ektar100_it8_30m_cp80c_triband_lp610.tiff --outfile=$@ --a1_x=73 --gs0_x=16 --a1_y=72 --vbase=860 --gs0_y=780

data/ektar100_it8_30m_cp80c_triband_cs100a_train.txt: data/ektar100_it8_30m_cp80c_triband.txt data/ektar100_it8_30m_cp80c_triband_bp470.txt data/ektar100_it8_30m_cp80c_triband_bp525.txt data/ektar100_it8_30m_cp80c_triband_lp610.txt
	python3 add_ref_readings.py  --r=data/ektar100_it8_30m_cp80c_triband_lp610.txt --g=data/ektar100_it8_30m_cp80c_triband_bp525.txt --b=data/ektar100_it8_30m_cp80c_triband_bp470.txt --Yxy=data/cs100a_measurements.txt data/ektar100_it8_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/ektar100_it8_u2_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/ektar100_it8_u2_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100_it8_u2_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/ektar100_it8_u1_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/ektar100_it8_u1_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100_it8_u1_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/ektar100_it8_o0_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/ektar100_it8_o0_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100_it8_o0_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/ektar100_it8_o1_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/ektar100_it8_o1_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100_it8_o1_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/ektar100_it8_o2_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/ektar100_it8_o2_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/ektar100_it8_o2_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra160_it8_u2_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra160_it8_u2_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160_it8_u2_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra160_it8_u1_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra160_it8_u1_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160_it8_u1_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra160_it8_o0_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra160_it8_o0_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160_it8_o0_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra160_it8_o1_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra160_it8_o1_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160_it8_o1_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra160_it8_o2_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra160_it8_o2_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra160_it8_o2_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra400_it8_u2_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra400_it8_u2_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400_it8_u2_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra400_it8_u1_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra400_it8_u1_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400_it8_u1_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra400_it8_o0_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra400_it8_o0_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400_it8_o0_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra400_it8_o1_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra400_it8_o1_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400_it8_o1_30m_cp80c_triband.txt | tr ' ' ',' > $@

data/portra400_it8_o2_30m_cp80c_triband_cs100a_train.txt: data/cs100a_measurements.txt data/portra400_it8_o2_30m_cp80c_triband.txt
	python3 add_ref_readings.py --Yxy=data/cs100a_measurements.txt data/portra400_it8_o2_30m_cp80c_triband.txt | tr ' ' ',' > $@

# Sony A7R IV shots with triband filter.
data/portra400-0.txt:
	python3 read_it8.py --img=it8_imgs/portra400-0.tif --multi --outfile=$@

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

# Test white chromaticies are common for all film as this is fixed during test time.
# Change these values to the one used during the test environment. The following
# values are measured under sunlight at around 5400K.
test_white_xy = --white_x=0.3353 --white_y=0.3496 

# The Nikon Z7 crosstalk correction coefficients are measured from Ektar 100 IT8 shot
# with 3 other shots scanned with bandpass filters. These values are fixed for the
# Camera + triband filter and light combination. They do not change with film scanned
# so we do not need to compute them again for each IT8 image, which also saves the
# time for scanning with additional 3 bandpass filters.
nikon_z7_triband_crosstalk_coefs = --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'

sony_a7rm4_triband_crosstalk_coefs = --crosstalk_r_coefs='1 -0.08262711 -0.01249409' --crosstalk_g_coefs='-0.13898878 1 -0.32017315' --crosstalk_b_coefs='-0.00664173 -0.09860774 1'

DEBUG = 
INSTALL_DIR = 

nikon_z7_triband_common_args = $(nikon_z7_triband_crosstalk_coefs) $(test_white_xy) $(DEBUG) $(INSTALL_DIR)/

.PHONY: nikon_z7_ektar100_u2
nikon_z7_ektar100_u2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_u2_30m_cp80c_triband_cs100a_train.txt  --film_name="Nikon Z7 Ektar100 u2" $(nikon_z7_triband_crosstalk_coefs)


.PHONY: nikon_z7_ektar100_u1
nikon_z7_ektar100_u1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_u1_30m_cp80c_triband_cs100a_train.txt --film_name="Nikon Z7 Ektar100 u1" 

.PHONY: nikon_z7_ektar100_o0
nikon_z7_ektar100_o0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_o0_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Ektar100 o0" 

.PHONY: nikon_z7_ektar100_o1
nikon_z7_ektar100_o1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_o1_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Ektar100 o1" 


.PHONY: nikon_z7_ektar100_o2
nikon_z7_ektar100_o2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_o2_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Ektar100 o2" 


.PHONY: nikon_z7_portra160_u2
nikon_z7_portra160_u2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_u2_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra160 u2" 


.PHONY: nikon_z7_portra160_u1
nikon_z7_portra160_u1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_u1_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra160 u1" 


.PHONY: nikon_z7_portra160_o0
nikon_z7_portra160_o0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_o0_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra160 o0" 

.PHONY: nikon_z7_portra160_o1
nikon_z7_portra160_o1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_o1_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra160 o1" 


.PHONY: nikon_z7_portra160_o2
nikon_z7_portra160_o2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_o2_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra160 o2" 


.PHONY: nikon_z7_portra400_u2
nikon_z7_portra400_u2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_u2_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra400 u2" 


.PHONY: nikon_z7_portra400_u1
nikon_z7_portra400_u1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_u1_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra400 u1" 


.PHONY: nikon_z7_portra400_o0
nikon_z7_portra400_o0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_o0_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra400 o0" 

.PHONY: nikon_z7_portra400_o1
nikon_z7_portra400_o1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_o1_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra400 o1" 


.PHONY: nikon_z7_portra400_o2
nikon_z7_portra400_o2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_o2_30m_cp80c_triband_cs100a_train.txt $(nikon_z7_triband_common_args) --film_name="Nikon Z7 Portra400 o2" 

# Sony A7R IV profiles.
.PHONY: sony_a7rm4_portra400_0
sony_a7rm4_portra400_0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400-0-cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --film_name="Sony A7RM4 Portra400" --fit_intercept=1 --debug

make_icc: make_icc.c
	mkdir -p bin_out
	gcc -o bin_out/make_icc make_icc.c -llcms2

.PHONY: all
all: build_prof.py make_icc.c
	mkdir -p icc_out
	python3 build_prof.py --src=data/ektar100_it8_30m_cp80c_triband_cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --fit_intercept=1 --film_name="Nikon Z7 Ektar100" $(DEBUG) $(INSTALL_DIR)/
