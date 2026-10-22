# Third-party code

- `libraw/`: [LibRaw](https://github.com/LibRaw/LibRaw), a git submodule pinned to release 0.22.2 (CDDL 1.0 or
  LGPL 2.1, see `libraw/LICENSE.*`). The Makefile builds it into `bin_out/libraw.a` (`make libraw`, and as part of
  `make lib` and `make neg_process`); neg_process and negicc-station link that library and include its headers
  as `"libraw/libraw.h"` with `-I3rd_party/libraw`, so both decode with the same LibRaw on every machine. To move to
  another release: `git -C 3rd_party/libraw fetch --tags && git -C 3rd_party/libraw checkout <tag>`, commit the
  submodule, and check that the TriLED conversions stay bit-identical (negicc-station's `tools/triled_parity.py`).
- `tiff_head.h`: LibRaw's TIFF header writer (`src/write/file_write.cpp`), which is not public API, copied here.
- `tiff.h`: TIFF tag definitions used by `tiff_head.h`.
- `argparse/`, `nlohmann/`, `dcraw/`, `elle_icc_profiles/`, `onnxruntime/`: see each directory.
