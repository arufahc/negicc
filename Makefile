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

make_icc: make_icc.c
	mkdir -p bin_out
	gcc -o bin_out/make_icc make_icc.c -llcms2

.PHONY: all
all: build_prof.py make_icc.c
	mkdir -p icc_out
	python3 build_prof.py --src=data/ektar100_it8_30m_cp80c_triband_cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --fit_intercept=1 --film_name="Nikon Z7 Ektar100" --debug=1
	chmod 755 ~/Library/ColorSync/Profiles/NegICC\ Profiles/*.sh
