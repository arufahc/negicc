# Agent Instructions for negicc

This file contains mandatory rules for any agent working in this repository.

---

## Priorities

The processor's goals, in order:

1. **Protect detail:** no clipping at either end (highlights or shadows) that the processor could have avoided.
2. **Minimise colour cast**, including crossover casts.
3. **Automatic settings:** gains, film base and the curve.
4. **Auto crop.**
5. **Minimise artifacts** (for example, the grain-aware debayer).
6. **Speed.**

Lightness, contrast and the depth of the blacks can be adjusted afterwards. Lost detail and crossover casts are hard
to recover, so the processor must get those right. A gain lower in this list never pays for a loss higher in it.
Any change that alters pictures reports its effects in this order: clipping at both ends first (before and after the
tone stage), then cast, then the settings, crop, artifacts, and speed last.

---

## Git Configuration Rules

All commits to this repository must use the repository owner's Git author and committer identity:
- **Name**: `Alpha Lam`
- **Email**: `arufa.hc@gmail.com`

**Command to run at startup:**
```bash
git config user.name "Alpha Lam" && git config user.email "arufa.hc@gmail.com"
```

### No Agent Signatures or Attribution Trailers

Do **NOT** add any agent signatures, `Co-Authored-By` trailers (e.g., `Co-Authored-By: Claude ...`), `Claude-Session`
links, `Generated with ...` footers, or agent names in commit authors, committers, commit messages, or pull request
descriptions. Individual agents such as Claude must not take credit in commit history, as this project is driven by
extensive human direction, domain engineering, and Antigravity collaboration.

### Commit Message Style Rules

Do **NOT** use conventional commit prefixes (such as `feat:`, `feat(...)`, `fix:`, `fix(...)`, `refactor:`,
`refactor(...)`, `chore:`, `chore(...)`, etc.) in commit messages, titles, or CL descriptions.
Write direct, plain English summaries (imperative mood, concise, descriptive), for example:
- "Add native C++ DINOv3 inference engine, runners, and neg_process integration"
- "Address review follow-ups for device check, mutex, input validation, and build targets"

---

## Documentation Rules

### Never Use Hardcoded Absolute Paths

Do **not** embed machine-specific absolute paths (e.g., `/home/user/Projects/...`) anywhere in documentation, markdown
files, scripts, or source comments. Use relative paths instead.

---

## Code Style Rules

- Do not introduce arbitrary namespaces (e.g. `namespace dinov3`) in application and engine sources (`neg_process.cc`,
  `pipeline.cc`, `print_emulation.cc`, `np_output.cc`, `tone_stage.cc`, etc.), which reside in the global namespace.
  The shared pipeline files consumed by
  `negicc-station` (the colour pipeline, flat field, image view, input curves and film base detector parts of
  `negicc_core.h`, the white-light balance part of `negicc.h`, and `rgb_view.cc`, `gpu_passes.cu`, `flat_field.cc`
  and the film base detector in `film_detector.cc`) use `namespace negicc`, and so do the TIFF and JPEG readers in
  the inputs part of
  `negicc.h` and in `np_input.cc` (`RgbImage`, `read_tiff`, `read_jpeg`, `to_linear_rec2020`) and `dilate` in the
  frame detector part of `negicc_core.h`; the rest of the inputs part and `np_input.cc`, and of the frame detector
  part and the frame detector in `film_detector.cc`, is in the global namespace.

---

## Parallel Regions

Rules for any OpenMP parallel region added to this repository:

- **Measure it for contention before adding it.** Time the region at several thread counts on a many-core machine (for
  example 1, 8, 32, 96 and every hardware thread of a 192-thread machine) and on the smallest target (a Jetson Orin
  Nano), one conversion at a time on an otherwise idle machine. Put the table in the pull request.
- **Use no more threads than the measurement supports.** A region that does not gain from threads stays serial. A
  region that does takes the program's thread count (`OMP_NUM_THREADS`), the same as every other region. Do not give
  a region a thread count of its own (`num_threads`): the runtime rebuilds its thread team at every change of count,
  and the regions around it get slower. Measured: with one region limited to 32 threads in a program running at 96,
  the resize next to it took 11.5 ms in place of 0.9 ms.
- **The results are identical at any thread count.** Keep sums whose order matters on one thread or in a fixed order,
  and show equal output at two thread counts at least.
- **Say it at the region.** A short comment gives what was measured, where, and what thread count the region uses.
- **One thread count for the whole program; when none is given, the smaller of 16 and the physical cores the process
  may run on.** A program calls `negicc::apply_default_thread_count()` once before its first parallel region
  (`neg_process` does). `OMP_NUM_THREADS` is honoured as it is. The 16 (`kDefaultThreadCap`, `negicc_core.h`) is the
  owner's number: a default must not take every core for a small gain. Measured on a 192-thread machine (96 cores),
  a conversion takes 12.1 s at 8 threads, 10.3 s at 16, 9.1 s at 32 and 7.7 s at 96, and uses 57, 62, 73 and 155 CPU
  seconds; OpenMP's own default, every hardware thread, is slower than 96. A Jetson Orin Nano has 6 cores, so its
  count stays 6.
- **The network on the CPU runs with the program's thread count too.** ONNX Runtime keeps its own threads; a thread
  count of 0 given to `DinoV3Engine` (the default for the backbone, and what `neg_process` gives the adapter) takes
  the program's count, so nothing in the program runs with a number of its own. Do not give a runtime a fixed count.
- **Batch runs set `OMP_NUM_THREADS` for every process,** so that processes times threads stays within the machine's
  cores.
- **Measured so far:** the quarter turn of a region that is not square (`extract_roi_and_transform`, `np_input.cc`),
  on a 9000 x 6000 region of a 60 MP RAW: 1.27 s on 1 thread, 0.41 s on 4, 0.22 s on 8, 0.13 s on 16 and 0.11 s on
  24, so it takes the program's thread count with no cap; identical output at 1, 4 and 16 threads, and no race in a
  ThreadSanitizer build at those counts.
