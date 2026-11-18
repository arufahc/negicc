#!/bin/sh
# Fetches ONNX Runtime 1.30.0's shared library (libonnxruntime.so.1.30.0, the C API) for this machine into
# lib/<arch>/, from Microsoft's onnxruntime wheel on PyPI (the library ships inside the wheel). Both the wheel and the
# library are checked against the SHA-256 below. `make neg_process` runs this when it links against this directory and
# the library is not there yet; it can also be run by hand. Needs curl and python3.
set -eu

cd "$(dirname "$0")"
arch="$(uname -m)"
case "$arch" in
  x86_64)
    url="https://files.pythonhosted.org/packages/34/35/e7f862dbacbc99fadd9b14a614e49c99bf0f35fd9927a82f096e3de33531/onnxruntime-1.30.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    wheel_sha="fa688e7891a6aa206636fe7372e27ee75fd17713289f6b4fc7b190e0a7de9328"
    lib_sha="c902c70b3003c0e99fada202f37478c515ae9bba7944c2b2abd0017bae0c82ed" ;;
  aarch64)
    url="https://files.pythonhosted.org/packages/16/bd/cbc5b8f91963689fdd622f463508c01d0aa95d3f944747b1e0b1eb2160b8/onnxruntime-1.30.0-cp312-cp312-manylinux_2_28_aarch64.whl"
    wheel_sha="6c32a000d5139a38ba9349030b0032e3331acb559d596b22738d9d2b343a2b83"
    lib_sha="8a218e1e446880131115c368ec7005969a5022af9a15a26173de648fef0b7789" ;;
  *)
    echo "fetch.sh: no ONNX Runtime wheel pinned for $arch" >&2
    exit 1 ;;
esac

lib="lib/$arch/libonnxruntime.so.1.30.0"
if [ -f "$lib" ] && echo "$lib_sha  $lib" | sha256sum -c --status; then
  echo "fetch.sh: $lib is already here"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 3 -o "$tmp/ort.whl" "$url"
echo "$wheel_sha  $tmp/ort.whl" | sha256sum -c --status || { echo "fetch.sh: the wheel's SHA-256 differs" >&2; exit 1; }
mkdir -p "lib/$arch"
python3 - "$tmp/ort.whl" "$lib" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z, open(sys.argv[2], "wb") as f:
    f.write(z.read("onnxruntime/capi/libonnxruntime.so.1.30.0"))
EOF
echo "$lib_sha  $lib" | sha256sum -c --status || { rm -f "$lib"; echo "fetch.sh: the library's SHA-256 differs" >&2; exit 1; }
chmod 755 "$lib"
ln -sf libonnxruntime.so.1.30.0 "lib/$arch/libonnxruntime.so.1"
ln -sf libonnxruntime.so.1 "lib/$arch/libonnxruntime.so"
echo "fetch.sh: $lib"
