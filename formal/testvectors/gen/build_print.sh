#!/bin/sh
# Builds the print test-vector generator against a built checkout of negicc.
#
#   make -j lib bin_out/embedded_data.o bin_out/embedded_data_files.o bin_out/np_options.o bin_out/np_output.o \
#        bin_out/np_input.o bin_out/np_film_base.o bin_out/np_print.o USE_CUDA=0 ORT_DIR=<ONNX Runtime>
#   ORT_DIR=<ONNX Runtime> formal/testvectors/gen/build_print.sh
#   bin_out/gen_print profiles/Portra400-A7R4-TriLED formal/testvectors profiles/Portra400-A7R4-D65
#
# gen_print_white.cc compiles np_pipeline.cc in place of bin_out/np_pipeline.o, to reach its file-local
# to_paper_white. The generator calls no inference, but the library it links refers to the runtime.
set -e
cd "$(dirname "$0")/../../.."
: "${ORT_DIR:=3rd_party/onnxruntime}"
ARCH=$(uname -m)
${CXX:-g++} -std=c++17 -O3 -fopenmp -ffp-contract=off -fsigned-char -I. -I3rd_party \
  -isystem 3rd_party/libraw -isystem 3rd_party/lcms2/include -isystem 3rd_party/zlib \
  -isystem 3rd_party/json/single_include -isystem 3rd_party/argparse/include -isystem "$ORT_DIR/include" \
  formal/testvectors/gen/gen_print.cc formal/testvectors/gen/gen_print_white.cc -o bin_out/gen_print \
  bin_out/embedded_data.o bin_out/embedded_data_files.o bin_out/np_options.o bin_out/np_output.o bin_out/np_input.o \
  bin_out/np_film_base.o bin_out/np_print.o \
  bin_out/libnegicc_dinov3.a bin_out/libraw.a bin_out/libtiff.a bin_out/libjpeg.a bin_out/liblcms2.a bin_out/libz.a \
  -L"$ORT_DIR/lib/$ARCH" -lonnxruntime -Wl,-rpath,"$ORT_DIR/lib/$ARCH"
echo "built bin_out/gen_print"
