# ONNX Runtime 1.30.0, prebuilt

ONNX Runtime's C and C++ API headers and, once fetched, its shared library. The default build links against them
dynamically, in place of building ONNX Runtime from source (the `onnxruntime/` submodule, `STATIC=1`).

| Path | What |
|---|---|
| `include/` | The headers of the v1.30.0 release archive (`onnxruntime-linux-x64-1.30.0.tgz`), unmodified. |
| `lib/<arch>/libonnxruntime.so.1.30.0` | The library, fetched by `fetch.sh`, not committed (`.gitignore`). `libonnxruntime.so.1` (its SONAME) and `libonnxruntime.so` link to it. |
| `fetch.sh` | Downloads Microsoft's `onnxruntime` 1.30.0 wheel from PyPI for this machine (x86_64 or aarch64), checks its SHA-256, extracts the library from it and checks the library's SHA-256. |
| `LICENSE`, `ThirdPartyNotices.txt` | ONNX Runtime's licence (MIT) and third-party notices, from the same wheel. |

`make neg_process` uses this directory when no other ONNX Runtime is found (see `ORT_CANDIDATES` in the Makefile), and
fetches the library first if it is missing. `bin_out/neg_process` then finds the library through its run path. A copy of
`neg_process` taken elsewhere needs `libonnxruntime.so.1` beside it or on the loader's path; the release archive
carries it.

To move to another release: replace `include/` with the release's headers, and the wheel URL and both SHA-256 values in
`fetch.sh` for each architecture (the wheel's from PyPI, the library's computed from the extracted file).
