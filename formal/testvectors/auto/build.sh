#!/bin/sh
# Builds the Auto search's test-vector generator against a built checkout of negicc.
#
#   make -j lib USE_CUDA=0 ORT_DIR=<ONNX Runtime>      # in the repository root, once
#   ORT_DIR=<ONNX Runtime> formal/testvectors/auto/build.sh
#   bin_out/gen_auto profiles/Portra400-A7R4-TriLED profiles/Portra400-A7R4-D65 formal/testvectors/auto
#
# The generator compiles multi_target_optimizer.cc into itself (see gen_auto.cc), so the library's copy of it is not
# linked (and GCC's warning about a source file's unnamed namespace used from another file is off). ORT_DIR is the ONNX Runtime directory the Makefile uses: the library it links refers to the runtime, though
# the generator runs no inference.
set -e
cd "$(dirname "$0")/../../.."
: "${ORT_DIR:=3rd_party/onnxruntime}"
ARCH=$(uname -m)
${CXX:-g++} -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -fsigned-char -Wall -Wextra -Wno-subobject-linkage -I. \
  -isystem 3rd_party/lcms2/include -isystem 3rd_party/zlib -isystem 3rd_party/json/single_include \
  -isystem "$ORT_DIR/include" \
  formal/testvectors/auto/gen_auto.cc -o bin_out/gen_auto \
  bin_out/libnegicc_dinov3.a bin_out/liblcms2.a bin_out/libz.a \
  -L"$ORT_DIR/lib/$ARCH" -lonnxruntime -Wl,-rpath,"$ORT_DIR/lib/$ARCH"
echo "built bin_out/gen_auto"
