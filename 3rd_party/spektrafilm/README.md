# spektrafilm print profiles

The print profiles of [spektrafilm](https://github.com/andreavolpato/spektrafilm) by Andrea Volpato, unmodified, from
commit `3bb2c2d2801ff68b92019cf1dbcbb133d60832bc` (`src/spektrafilm/data/profiles/`). `SHA256SUMS` has each file's
hash.

`neg_process` builds them in. For a density file written with `--print <print>` it computes from one of them a CLF
that renders the file as a print on that material (`np_print.h`, and "Print emulation" in `DESIGN.md`). `--print`
alone lists them.

| File | Print material |
|---|---|
| `profiles/kodak_portra_endura.json` | Kodak Professional Portra Endura (paper) |
| `profiles/kodak_supra_endura.json` | Kodak Professional Supra Endura (paper) |
| `profiles/kodak_ultra_endura.json` | Kodak Professional Ultra Endura (paper) |
| `profiles/kodak_endura_premier.json` | Kodak Professional Endura Premier (paper) |
| `profiles/kodak_ektacolor_edge.json` | Kodak Ektacolor Edge (paper) |
| `profiles/fujifilm_crystal_archive_typeii.json` | Fujifilm Crystal Archive Type II (paper) |
| `profiles/kodak_2383.json` | Kodak Vision 2383 (print film) |
| `profiles/kodak_2393.json` | Kodak Vision Premier 2393 (print film) |

**Licence.** The profiles are under CC BY-SA 4.0 (`SPEKTRAFILM_LICENSE.txt`), not under negicc's licence. A CLF that
`neg_process` writes with one carries that notice for its two tables, the curve per layer and the colour cube, which
derive from this data. Its matrices are computed and are not under that licence, and neither are the pictures made
through a CLF.

The profiles are models built from datasheets and have not been compared with real prints.

To add a print material, copy its spektrafilm profile into `profiles/`, add its hash to `SHA256SUMS`, and rebuild.
