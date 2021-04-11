build_prof.h: build_prof.py
	python3 build_prof.py --src=../tricolor/c1_test/ektar100_it8_30c_cp80c_positive_cs100a_train.csv

make_icc: make_icc.c build_prof.h
	gcc -o make_icc make_icc.c -llcms2

build_prof.icc: build_prof.ti3
	colprof -v -ax -qh -kz -u -bn -ni -np -no build_prof

.PHONY: all
all: make_icc build_prof.icc
	mkdir -p icc_out
	./make_icc
