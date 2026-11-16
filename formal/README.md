# Formal specification of the conversion (Lean 4)

This directory holds the mathematics of `neg_process` written as Lean 4 definitions, so that
each formula can be read against the C++ that implements it, properties of the formulas can be
proved, and the formulas can be run and compared with the program's output.

It is a specification of code that exists: nothing here changes what `neg_process` computes.

## What is in the package

| | |
|---|---|
| Lake package | `negicc` |
| Library (root module) | `Negicc` |
| Lean | `leanprover/lean4:v4.36.0` (`lean-toolchain`) |
| Mathlib | commit `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`, the commit of its tag `v4.36.0` (`lakefile.toml`) |
| Other dependencies | the exact commits Mathlib's own manifest names, in `lake-manifest.json` |

The base of the library, on which the other parts build:

| Module | Contents |
|---|---|
| `Negicc.Base.Scalar` | The class `Scalar` of numbers a formula is written over, the functions derived from it (`log2`, `log10`, `exp2`, `fabs`, `sq`, `clamp`, `clamp01`, `roundNat`), and its `Float` instance |
| `Negicc.Base.Vec` | 3-vectors `Vec3` and 3x3 matrices `Mat3` |
| `Negicc.Base.Table` | `lerp`, a sampled table read by linear interpolation (`tableAt`, `tableEval`), `sum` and `mean` |
| `Negicc.Base.Colour` | The sRGB curve (`srgbEncode`, `srgbDecode`), linear sRGB to XYZ and CIELAB (`linearSrgbToLab` and its parts), `density` and `transmission` |
| `Negicc.Base.Defs` | The four modules above together. No Mathlib |
| `Negicc.Base.Real` | The real numbers as scalars (`Scalar ℝ`), and the operations restated in Mathlib's terms |
| `Negicc.Base.Props` | Proved properties of the base definitions over the real numbers |
| `Negicc.Base` | All of the above |

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
