# Formal specification of the conversion (Lean 4)

This directory holds the mathematics of `neg_process` written as Lean 4 definitions, so that
each formula can be read against the C++ that implements it, properties of the formulas can be
proved, and the formulas can be run and compared with the program's output. `FORMULAS.md` lists
every formula in ordinary notation with the code function and the Lean name, what is proved and
checked, and what is not formalized.

It is a specification of code that exists: nothing here changes what `neg_process` computes.

## What is in the package

| | |
|---|---|
| Lake package | `negicc` |
| Library (root module) | `Negicc` |
| Lean | `leanprover/lean4:v4.36.0` (`lean-toolchain`) |
| Mathlib | commit `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`, the commit of its tag `v4.36.0` (`lakefile.toml`) |
| Other dependencies | the exact commits Mathlib's own manifest names, in `lake-manifest.json` |

The modules, in the order the pipeline runs. The base is what the other parts build on:

| Module | Contents |
|---|---|
| `Negicc.Base.Scalar` | The class `Scalar` of numbers a formula is written over, the functions derived from it (`log2`, `log10`, `exp2`, `fabs`, `sq`, `clamp`, `clamp01`, `roundNat`, `roundEvenNat`), and its `Float` instance |
| `Negicc.Base.Vec` | 3-vectors `Vec3` and 3x3 matrices `Mat3` |
| `Negicc.Base.Table` | `lerp`, a sampled table read by linear interpolation (`tableAt`, `tableEval`), `sum` and `mean` |
| `Negicc.Base.Colour` | The sRGB curve (`srgbEncode`, `srgbDecode`), linear sRGB to XYZ and CIELAB (`linearSrgbToLab` and its parts, `labF` and its inverse `labFInv`), `density` and `transmission` |
| `Negicc.Base.Defs` | The four modules above together. No Mathlib |
| `Negicc.Base.Real` | The real numbers as scalars (`Scalar ℝ`), and the operations restated in Mathlib's terms |
| `Negicc.Base.Props` | Proved properties of the base definitions over the real numbers |
| `Negicc.Base` | All of the above |
| `Negicc.Scene.Decode` | Black level, the half-size image, the grain-aware debayer, pixel shift, three captures |
| `Negicc.Scene.Input` | Flat field, load, the input matrix (crosstalk correction, film base normalisation, `E`, `g`, `b`), the soft knee |
| `Negicc.Scene.FilmBase` | Where the film base is read from, and the statistics that read it |
| `Negicc.Scene.Profile` | A bracket's film profile: input curves, the colour table with tetrahedral interpolation, output curves |
| `Negicc.Scene.Linear` | The profile connection space to linear sRGB, the pre-shoulder value, the clipping rules |
| `Negicc.Scene.Shoulder` | The highlight roll-off: the curve, its knee, and `x_hi` from a frame's 99.5 % point |
| `Negicc.Scene.Convert` | The whole pixel: display value, the sRGB and linear encodings, CIELAB, the 16-bit outputs |
| `Negicc.Scene.Crop` | The inner crop the statistics read |
| `Negicc.Scene.Defs`, `Negicc.Scene.Props`, `Negicc.Scene` | The scene modules together (no Mathlib); proved properties; both |
| `Negicc.Tone.Apply` | The tone curve on a pixel (`apply_tone`) |
| `Negicc.Tone.Curve` | The tone curve, its table, the render's descriptors, the targets, the solve, the clip share |
| `Negicc.Tone.Clip` | The clip check on a sample of the export |
| `Negicc.Tone.Defs`, `Negicc.Tone.Props`, `Negicc.Tone` | The tone modules together (no Mathlib); proved properties; both |
| `Negicc.WhiteLight.Balance` | The film-base balance of the white-light decode |
| `Negicc.WhiteLight.Curve` | Given green and blue input curves and their composition into the profile |
| `Negicc.WhiteLight.Defs`, `Negicc.WhiteLight.Props`, `Negicc.WhiteLight` | The white-light modules together (no Mathlib); proved properties; both |
| `Negicc.Print.Cie` | The CIE 1931 observer and the illuminants D50 and D55 of `np_print.cc` |
| `Negicc.Print.Spectral` | A print material's spectral model (dye amounts to XYZ), `numpy.interp`, the resampling of a profile's columns, the observer, the printer lamp |
| `Negicc.Print.Matrix` | 3x3 inverse, transpose and sums of the print |
| `Negicc.Print.Material` | `Q`, the print matrix `P`, the CLF's colour matrix, the Bradford adaptation, `aim`, a material's load |
| `Negicc.Print.Grey` | The grey balance curve, a bracket's mid grey, the tilt at a mid grey, the slopes |
| `Negicc.Print.Curves` | `Paper_c`, the print ICC's input curves at a mid grey with gains, its 16-bit codes, its table |
| `Negicc.Print.White` | The print relative to its paper (`to_paper_white`, `scale_print_icc_table`) |
| `Negicc.Print.Density` | The density file's codes, floor and clip counts, the balance offsets |
| `Negicc.Print.Clf` | The density file's CLF: the frame's grey and slopes, its first matrix, the material's nodes |
| `Negicc.Print.Floor` | The film base held no darker than the thinnest film (`print_floor_factor`), the frame it reads |
| `Negicc.Print.Search` | The print's search (the held exposure, A0, the line search in A, the fixed point) and the knobs, around an oracle for the network |
| `Negicc.Print.Defs` | The eleven print modules together. No Mathlib |
| `Negicc.Print.Props` | Proved properties of the print emulation (DESIGN.md 5.11 among them) |

## One formula, two readings

Each formula is written once, over any type `α` with an instance `Scalar α`, and read twice:

- **Over the real numbers** (`ℝ`, Mathlib's `Real`) it is mathematics. Statements about it are
  proved there. This reading is noncomputable.
- **Over `Float`** (double precision) it is a program, which can be run and compared with
  values computed by the C++.

For example, the sRGB encoding is the single definition

```lean
def srgbEncode (l : α) : α :=
  if l ≤ 0.0031308 then l * 12.92 else pow l (1.0 / 2.4) * 1.055 - 0.055
```

and `srgbEncode (x : ℝ)` and `srgbEncode (0.5 : Float)` are its two readings.

`Scalar` lists the operations the pipeline needs and nothing else: `+`, `-`, `*`, `/`, negation,
`<` and `≤` (decidable), `max`, `min`, decimal literals, the cast of a natural number, `sqrt`,
`cbrt`, `exp`, `log`, `pow`, `tanh` and `natFloor`. A type class was chosen over writing each
formula twice because it keeps one source for both readings at no cost to either: over `ℝ` the
definitions unfold to ordinary Mathlib terms, and over `Float` they compile to native code.

Conventions:

- Literals carry a decimal point (`1.0`, `116.0`), as in the C++. They are exact decimal
  numbers. The rounding of constants and of arithmetic to single or double precision is not
  modelled, which is why a numeric check has a tolerance.
- `Scalar` carries no laws, so a theorem is proved over `ℝ` and says what the formula computes
  with exact arithmetic. It does not transfer to the `Float` reading or to the program; the numeric
  check is what ties those to the formulas.
- Where an operation leaves its domain (the logarithm of a non-positive number, a power of a
  negative base, a division by zero) the real-number functions return Mathlib's conventional
  values and the program returns an infinity or a NaN. No statement relies on either. NaN and
  infinities are not modelled.
- The definitions and the `Float` instance do not import Mathlib. Only the real-number instance
  and the proofs do. An executable that imports `Negicc.Base.Defs` therefore builds in seconds
  and links without Mathlib.
- The files are Lean modules (`module` with `public import`), like Mathlib's, so both module
  and non-module files can import them.

## Installing the toolchain and building

Lean is installed per user with `elan`, its toolchain manager. Nothing is installed system-wide.

```sh
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh     # once; installs elan under ~/.elan
cd formal
lake exe cache get      # downloads compiled Mathlib for the pinned commit (several GB under .lake/)
lake build              # builds the library; about a minute the first time, seconds afterwards
```

`elan` reads `lean-toolchain` and downloads that Lean release on first use. `lake build` must
end with `Build completed successfully` and print no warning. The library contains no `sorry`.

## The numeric check against the C++

```sh
cd formal
lake exe check
```

`check` (`Check.lean`) reads the test vectors in `testvectors/` (`scene.json`: the base, the flat
field, the pixel and the shoulder; `scene_tone.json`: the tone stage; `scene_white.json`: the
white-light input curves and film-base balance; `scene_read.json`: reading the scan, the film base,
the inner crop and the roll-off's histogram)
and the bracket profiles they name (`Target_3.icc` of `profiles/Portra400-A7R4-TriLED` and
`Target_4.icc` of `profiles/Portra400-A7R4-D65`, parsed by the check itself), evaluates the Lean
definitions over `Float` on the vectors' inputs, and compares the results with the values the C++
functions computed for the same inputs. It prints one line per group of comparisons, with the
number of comparisons, the largest difference and the tolerance. It exits with a non-zero code if a
comparison is outside its tolerance or a group has another number of comparisons than
`expectedCounts` says, so a short vectors file fails. Run it from `formal/`; a different repository
root can be given as its argument (the default is `..`).

The tolerances and their reasons are at the top of `Check.lean`:

| Tolerance | Value | For |
|---|---|---|
| `tolSingle` | 5e-6 | Values the C++ computes in single precision: linear and encoded colours, curve constants, matrices |
| `tolLab` | 1e-4 | CIELAB of a given linear colour |
| `tolLabPixel` | 1e-3 | CIELAB at the end of the whole conversion |
| `tolRoot` | 2e-5 | The roll-off `x_hi` found by bisection on a curve evaluated in single precision |
| `tolDouble` | 1e-9 | Values the C++ computes in double precision |
| `tolSolve` | 1e-7 | The tone curve solved by Levenberg-Marquardt, both sides in double precision |
| `tolPercentile32`, `tolHistBin` | 5e-4, 2 / 4096 | numpy's percentile in single precision; the 99.5 % point of the largest channel's histogram |
| codes | exact | 8- and 16-bit codes are compared exactly, except where the two values lie on the two sides of a half (one code, counted apart) |

The C++ computes a pixel in single precision and the check evaluates the specification in double
precision, so a difference of a few parts in a million is rounding, not disagreement. Two kinds
of input are kept out of the vectors or counted apart, because there single precision changes
more than the last digits:

- **A value exactly on a threshold constant.** The specification's thresholds are the decimal
  numbers of the source (`1.0001`, `0.0031308`), the program's are their single precision
  roundings, and an input equal to the rounded constant falls on different sides of the two.
- **The flat field's division.** The C++ forms the quotient in single precision, whose steps near
  50000 are 1/256, and then rounds it to a whole number. Where the exact quotient is within 0.01
  of a half, the result may be the neighbouring 16-bit code. The check counts those values in a
  group of their own, with a tolerance of one code.

### Regenerating the test vectors

The vectors are written by `testvectors/gen/gen_vectors.cc`, a C++ program that calls the
library's own functions and re-implements nothing (`gen_internal.cc` compiles two of the library's
source files into it, for their functions of internal linkage). From the repository root, with the
submodules checked out and `ORT_DIR` the ONNX Runtime directory the Makefile uses:

```sh
make -j lib USE_CUDA=0 ORT_DIR=<onnxruntime>             # the library, once
ORT_DIR=<onnxruntime> formal/testvectors/gen/build.sh   # builds bin_out/gen_vectors
bin_out/gen_vectors profiles/Portra400-A7R4-TriLED profiles/Portra400-A7R4-D65 formal/testvectors
```

The check compares the specification with the committed vectors, not with the program. Whenever the
C++ that the generator calls changes, the vectors are generated again and the check is run; a
difference is then either a change of the specification to follow or a defect.

### The last recorded run

Vectors generated from commit `b2318d0` on x86-64 Linux with GCC 13.3 and glibc 2.39
(another compiler or C library may change the last digit of a single precision result, which the
tolerances cover). With Lean `v4.36.0`, the check makes 9897 comparisons, none outside its
tolerance. `lake build` last ended with `Build completed successfully` and no warning at `165da12`;
no theorem uses the definitions changed or added since (`toneClipSample`, `tonePlanForImage`,
`toneTargetsShares`, `tonePlanShares`).

## The print emulation's check

```sh
cd formal
lake exe checkPrint
```

`checkPrint` (`CheckPrint.lean`) reads `testvectors/print.json`, the spektrafilm print profiles it
names (`3rd_party/spektrafilm/profiles/`, read and resampled by the specification itself) and the
bundles' print ICCs (`profiles/<bundle>/print_<material>.icc`), and compares the specification over
`Float` with the C++: the material as loaded, `Q` and the print matrix, `Paper_c`, the input curves
and their 16-bit tables, the table (every node of a bundle's print ICC to one 16-bit code), the
print's white, a pixel through the print profile (PCS, display value and CIELAB), the density codes, the CLF's numbers and the
floor, for the Portra 400 TriLED bundle (a paper and a print film) and its white-light bundle. Its
tolerances and their reasons are at the top of `CheckPrint.lean`; values the C++ computes in double
precision agree to `1e-9`, a CLF's printed numbers to their printed digits.

The vectors are written by `testvectors/gen/gen_print.cc` (with `gen_print_white.cc`, which
compiles `np_pipeline.cc` to reach its file-local `to_paper_white`). From the repository root:

```sh
make -j lib bin_out/embedded_data.o bin_out/embedded_data_files.o bin_out/np_options.o \
  bin_out/np_output.o bin_out/np_input.o bin_out/np_film_base.o bin_out/np_print.o USE_CUDA=0 ORT_DIR=<onnxruntime>
ORT_DIR=<onnxruntime> formal/testvectors/gen/build_print.sh
bin_out/gen_print profiles/Portra400-A7R4-TriLED formal/testvectors profiles/Portra400-A7R4-D65
```

The committed `print.json` was generated from the code of `master` at `b4a9991` on x86-64 Linux with GCC 13.3.

## Using the package from another repository

Add a requirement on this repository with `subDir = "formal"` to the other project's
`lakefile.toml`, at a fixed commit:

```toml
[[require]]
name = "negicc"
git = "https://github.com/arufahc/negicc.git"
rev = "<commit>"
subDir = "formal"
```

Use the same `lean-toolchain` as this directory, so that Mathlib's compiled cache applies. Then
`import Negicc.Base` for the definitions with the real-number instance and the lemmas, or
`import Negicc.Base.Defs` for the definitions alone.

## Writing proofs over the real numbers

A proof about a formula unfolds its definition (`simp only [srgbEncode]` or `unfold`) and then
works with ordinary real-number terms. `Negicc.Base.Real` has `simp` lemmas that restate the
class operations in Mathlib's terms (`Scalar.pow_real : pow x y = x ^ y`, and so on).

Two things to know:

- In this Mathlib revision `ring` fails on a decimal literal that denotes a whole number, such
  as `1.0`. Call `norm_num` first; it turns the literals into ordinary numerals.
- After `simp` has rewritten the condition of an `if`, rewrite the `if` itself with
  `ite_eq_left` or `ite_eq_right` and a hypothesis, or split on the condition with `by_cases`.
