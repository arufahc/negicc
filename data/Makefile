ektar100_it8_30m_cp80c_triband.txt:
	python3 ../read_it8.py --img=../it8_imgs/ektar100_it8_30m_cp80c_triband.tiff --outfile=$@ --a1_x=73 --gs0_x=20 --hbase=1310	

ektar100_it8_30m_cp80c_triband_bp470.txt:
	python3 ../read_it8.py --img=../it8_imgs/ektar100_it8_30m_cp80c_triband_bp470.tiff --outfile=$@ --a1_x=73 --gs0_x=16 --a1_y=70 --hbase=1310 --vbase=860 --gs0_y=780

ektar100_it8_30m_cp80c_triband_bp525.txt:
	python3 ../read_it8.py --img=../it8_imgs/ektar100_it8_30m_cp80c_triband_bp525.tiff --outfile=$@ --a1_x=73 --gs0_x=16 --a1_y=70 --hbase=1310 --vbase=860 --gs0_y=780

ektar100_it8_30m_cp80c_triband_lp610.txt:
	python3 ../read_it8.py --img=../it8_imgs/ektar100_it8_30m_cp80c_triband_lp610.tiff --outfile=$@ --a1_x=73 --gs0_x=16 --a1_y=72 --vbase=860 --gs0_y=780

ektar100_it8_30m_cp80c_triband_cs100a_train.txt: ektar100_it8_30m_cp80c_triband.txt ektar100_it8_30m_cp80c_triband_bp470.txt ektar100_it8_30m_cp80c_triband_bp525.txt ektar100_it8_30m_cp80c_triband_lp610.txt
	python3 ../add_ref_readings.py  --r=ektar100_it8_30m_cp80c_triband_lp610.txt --g=ektar100_it8_30m_cp80c_triband_bp525.txt --b=ektar100_it8_30m_cp80c_triband_bp470.txt --Yxy cs100a_measurements.txt ektar100_it8_30m_cp80c_triband.txt | tr ' ' ',' > $@

.PHONY: all
all: ektar100_it8_30m_cp80c_triband.txt ektar100_it8_30m_cp80c_triband_bp470.txt ektar100_it8_30m_cp80c_triband_bp525.txt ektar100_it8_30m_cp80c_triband_lp610.txt ektar100_it8_30m_cp80c_triband.txt


