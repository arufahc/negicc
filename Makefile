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

make_icc: make_icc.c
	mkdir -p bin_out
	gcc -o bin_out/make_icc make_icc.c -llcms2

.PHONY: ektar100_u2
ektar100_u2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_u2_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Ektar100 u2" --debug=1


.PHONY: ektar100_u1
ektar100_u1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_u1_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Ektar100 u1" --debug=1


.PHONY: ektar100_o0
ektar100_o0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_o0_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Ektar100 o0" --debug=1

.PHONY: ektar100_o1
ektar100_o1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_o1_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Ektar100 o1" --debug=1


.PHONY: ektar100_o2
ektar100_o2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/ektar100_it8_o2_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Ektar100 o2" --debug=1


.PHONY: portra160_u2
portra160_u2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_u2_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra160 u2" --debug=1


.PHONY: portra160_u1
portra160_u1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_u1_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra160 u1" --debug=1


.PHONY: portra160_o0
portra160_o0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_o0_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra160 o0" --debug=1

.PHONY: portra160_o1
portra160_o1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_o1_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra160 o1" --debug=1


.PHONY: portra160_o2
portra160_o2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra160_it8_o2_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra160 o2" --debug=1


.PHONY: portra400_u2
portra400_u2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_u2_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra400 u2" --debug=1


.PHONY: portra400_u1
portra400_u1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_u1_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra400 u1" --debug=1


.PHONY: portra400_o0
portra400_o0: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_o0_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra400 o0" --debug=1

.PHONY: portra400_o1
portra400_o1: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_o1_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra400 o1" --debug=1


.PHONY: portra400_o2
portra400_o2: build_prof.py make_icc.c
	python3 build_prof.py --src=data/portra400_it8_o2_30m_cp80c_triband_cs100a_train.txt --crosstalk_r_coefs='1 -0.082115947096161 -0.018122368837481' --crosstalk_g_coefs='-0.078439679225726 1 -0.218905850589040' --crosstalk_b_coefs='-0.004963694751139 -0.100838827684968 1'  --white_x=0.3353 --white_y=0.3496 --film_name="Nikon Z7 Portra400 o2" --debug=1

.PHONY: all
all: build_prof.py make_icc.c
	mkdir -p icc_out
	python3 build_prof.py --src=data/ektar100_it8_30m_cp80c_triband_cs100a_train.txt --white_x=0.3353 --white_y=0.3496 --fit_intercept=1 --film_name="Nikon Z7 Ektar100" --debug=1
	chmod 755 ~/Library/ColorSync/Profiles/NegICC\ Profiles/*.sh
