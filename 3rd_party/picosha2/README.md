# PicoSHA2

`picosha2.h` of [PicoSHA2](https://github.com/okdshin/PicoSHA2) by okdshin, a header-only SHA-256, unmodified, from
commit `161cb3fc4170fa7a3eca9e582cebd27cc4d1fe29`, with its licence file `LICENSE`.

`neg_process` takes from it the SHA-256 of a bundle's print ICC and of the spektrafilm print profile built in, which it
compares with the hashes the bundle records, and of a print material given as a file, which the sidecar records
(`profile.print.file_sha256`).

**Licence.** MIT, Copyright (c) 2017 okdshin (`LICENSE`; the header carries the same notice), not negicc's licence.
