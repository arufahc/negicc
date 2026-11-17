#!/bin/sh
# Builds the test-vector generator against a built checkout of negicc.
#
#   make -j lib USE_CUDA=0 ORT_DIR=<ONNX Runtime>      # in the repository root, once
#   ORT_DIR=<ONNX Runtime> formal/testvectors/gen/build.sh
#   bin_out/gen_vectors profiles/Portra400-A7R4-TriLED formal/testvectors
#
# ORT_DIR is the ONNX Runtime directory the Makefile uses (include/ and lib/<arch>/). The generator calls no
# inference, but the library it links refers to the runtime.
set -e
cd "$(dirname "$0")/../../.."
: "${ORT_DIR:=3rd_party/onnxruntime}"
ARCH=$(uname -m)
${CXX:-g++} -std=c++17 -O3 -fPIC -fopenmp -ffp-contract=off -fsigned-char -Wall -Wextra -I. \
  -isystem 3rd_party/lcms2/include -isystem 3rd_party/zlib -isystem 3rd_party/json/single_include \
  -isystem "$ORT_DIR/include" \
  formal/testvectors/gen/gen_vectors.cc -o bin_out/gen_vectors \
  bin_out/libnegicc_dinov3.a bin_out/liblcms2.a bin_out/libz.a \
  -L"$ORT_DIR/lib/$ARCH" -lonnxruntime -Wl,-rpath,"$ORT_DIR/lib/$ARCH"
echo "built bin_out/gen_vectors"
