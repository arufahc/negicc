# Agent Instructions for negicc

This file contains mandatory rules for any agent working in this repository.

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
  `np_pipeline.cc`, `tone_stage.cc`, etc.), which reside in the global namespace. The shared pipeline files consumed by
  `negicc-station` (`rgb_view.h`, `rgb_view.cc`, `gpu_passes.cu`, `color_pipeline.h`, `flat_field.h`, `flat_field.cc`,
  `input_curves.h`, `film_base_detector.h`, `film_base_detector.cc`, and `white_light_balance.h`) use
  `namespace negicc`, and so do the TIFF and JPEG readers in `np_input.h` and `np_input.cc` (`RgbImage`, `read_tiff`,
  `read_jpeg`, `to_linear_rec2020`) and `dilate` in `film_frame_detector.h`; the rest of `np_input.*` and
  `film_frame_detector.*` is in the global namespace.

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
- **Batch runs set `OMP_NUM_THREADS` for every process,** so that processes times threads stays within the machine's
  cores. The default is every hardware thread, which the small regions run once per network call do not gain from.
