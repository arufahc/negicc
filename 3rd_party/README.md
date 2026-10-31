# Third-party code

- `libraw/`: [LibRaw](https://github.com/LibRaw/LibRaw), a git submodule pinned to release 0.22.2 (CDDL 1.0 or
  LGPL 2.1, see `libraw/LICENSE.*`). The Makefile builds it into `bin_out/libraw.a` (`make libraw`, and as part of
  `make lib` and `make neg_process`); neg_process and negicc-station link that library and include its headers
  as `"libraw/libraw.h"` with `-I3rd_party/libraw`, so both decode with the same LibRaw on every machine. To move to
  another release: `git -C 3rd_party/libraw fetch --tags && git -C 3rd_party/libraw checkout <tag>`, commit the
  submodule, and check that the TriLED conversions stay bit-identical (negicc-station's `tools/triled_parity.py`).
- `lcms2/`: [Little-CMS](https://github.com/mm2/Little-CMS), a git submodule pinned to release 2.14 (MIT, see
  `lcms2/COPYING`), built into `bin_out/liblcms2.a` (`make deps`, and as part of `make lib` and `make neg_process`).
- `zlib/`: [zlib](https://github.com/madler/zlib), a git submodule pinned to release 1.3 (zlib licence, see
  `zlib/LICENSE`), built into `bin_out/libz.a` like Little-CMS; LibRaw is compiled against it.
- `libtiff/`: [libtiff](https://gitlab.com/libtiff/libtiff), a git submodule pinned to release 4.7.2 (libtiff
  licence, see `libtiff/LICENSE.md`), built into `bin_out/libtiff.a` with Deflate (the pinned zlib), LZW, PackBits,
  JPEG (libjpeg-turbo) and the other built-in codecs; neg_process reads TIFF input with it (`rgb_input.cc`).
- `libjpeg-turbo/`: [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo), a git submodule pinned to
  release 3.2.1 (IJG and BSD licences, see `libjpeg-turbo/LICENSE.md`), built into `bin_out/libjpeg.a` without SIMD;
  neg_process reads JPEG input with it, and libtiff JPEG-compressed TIFFs.
- `config/`: the headers the two libraries' CMake builds would generate (`jconfig.h`, `jconfigint.h`, `jversion.h`,
  `tiffconf.h`, `tif_config.h`, `tiffvers.h`), written out for 64-bit little-endian Linux, so the Makefile builds
  them without CMake. Moving either library to another release means checking these against its templates.
- `json/`: [nlohmann/json](https://github.com/nlohmann/json), a shallow git submodule pinned to release 3.11.3 (MIT,
  see `json/LICENSE.MIT`), header-only, included as `"nlohmann/json.hpp"` with `-I3rd_party/json/single_include`.
- `argparse/`: [argparse](https://github.com/p-ranav/argparse), a git submodule pinned to commit 69dabd8 (v3.0 with
  the fix of its PR 310; MIT, see `argparse/LICENSE`), header-only, included as `"argparse/argparse.hpp"` with
  `-I3rd_party/argparse/include`.

  The four were the system's liblcms2-dev 2.14 and zlib1g-dev 1.3 and copies of json.hpp and argparse.hpp; the
  submodules pin the same versions (the two headers byte for byte), so nothing changes but where they come from. To
  move one: check out another tag in the submodule, commit it, and check that the TriLED conversions stay
  bit-identical (negicc-station's `tools/triled_parity.py`).
- `tiff_head.h`: LibRaw's TIFF header writer (`src/write/file_write.cpp`), which is not public API, copied here.
- `tiff.h`: TIFF tag definitions used by `tiff_head.h`.
- `elle_icc_profiles/`, `onnxruntime/`: see each directory.
