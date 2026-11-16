/-
The numbers the formulas are evaluated over.
-/
module

@[expose] public section

/-!
# Scalars: one formula, two readings

Every formula of this library is written once, over a type `α` with an instance of the class
`Scalar α`, and is then read in two ways:

* over the real numbers `ℝ` (instance in `Negicc.Base.Real`, which needs Mathlib), where the
  formula is a mathematical object and properties about it are proved;
* over the double precision numbers `Float` (instance below, no Mathlib), where the formula can be
  run and compared with what the C++ program computes.

`Scalar α` lists exactly the operations the pipeline needs: the four arithmetic operations and
negation, the two comparisons (decidable, so that a formula can branch on them), `max` and `min`,
decimal literals, the cast of a natural number, and the functions `sqrt`, `cbrt`, `exp`, `log`,
`pow`, `tanh` and `natFloor`. Everything else (`log2`, `log10`, `exp2`, `fabs`, `clamp`, ...) is
defined from these, once, in the namespace `Negicc.Scalar`.

## Conventions

* **Literals are written with a decimal point** (`1.0`, `116.0`, `0.0031308`), as in the C++
  source. They are the exact decimal numbers; the rounding of a constant to single or double
  precision is not modelled.
* **Outside their domain the operations are not specified.** For a logarithm of a non-positive
  number, a power of a negative base or a division by zero, Mathlib's total functions return
  conventional values and the floating point program returns an infinity or a NaN. No statement in
  this library relies on either. NaN and infinities are not modelled at all.
* The instances the class provides for `+`, `≤`, literals and so on have low priority. They are
  meant for the generic definitions only: an expression written directly over `ℝ` or `Float`
  keeps using that type's own instances.
-/

namespace Negicc

/-- The numeric operations the pipeline's formulas use. See the module documentation. -/
class Scalar (α : Type) extends
    Add α, Sub α, Mul α, Div α, Neg α, LT α, LE α, Max α, Min α, OfScientific α, NatCast α where
  /-- `x < y` can be decided: classically over `ℝ`, by comparison over `Float`. -/
  decLt : (x y : α) → Decidable (x < y)
  /-- `x ≤ y` can be decided. -/
  decLe : (x y : α) → Decidable (x ≤ y)
  /-- The square root of a non-negative number (C `sqrt`). -/
  sqrt : α → α
  /-- The real cube root, an odd function (C `cbrt`). -/
  cbrt : α → α
  /-- The exponential function (C `exp`). -/
  exp : α → α
  /-- The natural logarithm of a positive number (C `log`). -/
  log : α → α
  /-- `pow x y` is `x` to the power `y`, for `x ≥ 0` (C `pow`). -/
  pow : α → α → α
  /-- The hyperbolic tangent (C `tanh`). -/
  tanh : α → α
  /-- The largest natural number not above `x`, and `0` for a negative `x`. For `x ≥ 0` this is
  what the C++ casts `(int)x` and `(size_t)x` compute. -/
  natFloor : α → Nat

attribute [instance low] Scalar.toAdd Scalar.toSub Scalar.toMul Scalar.toDiv Scalar.toNeg
  Scalar.toLT Scalar.toLE Scalar.toMax Scalar.toMin Scalar.toOfScientific Scalar.toNatCast

instance (priority := low) instDecidableLtOfScalar {α : Type} [Scalar α] (x y : α) :
    Decidable (x < y) :=
  Scalar.decLt x y

instance (priority := low) instDecidableLeOfScalar {α : Type} [Scalar α] (x y : α) :
    Decidable (x ≤ y) :=
  Scalar.decLe x y

namespace Scalar

variable {α : Type} [Scalar α]

/-- The logarithm to base 2 (C `log2`): the number of stops between `x` and `1`. -/
def log2 (x : α) : α := log x / log 2.0

/-- The logarithm to base 10 (C `log10`). -/
def log10 (x : α) : α := log x / log 10.0

/-- `2` to the power `x` (C `exp2`): the factor that `x` stops stand for. -/
def exp2 (x : α) : α := pow 2.0 x

/-- The absolute value (C `fabs`). -/
def fabs (x : α) : α := if x < 0.0 then -x else x

/-- The square `x * x`. -/
def sq (x : α) : α := x * x

/-- `x` held between `lo` and `hi` (C++ `std::clamp(x, lo, hi)`, for `lo ≤ hi`). -/
def clamp (x lo hi : α) : α := min (max x lo) hi

/-- `x` held between `0` and `1` (`clamp01` in `color_pipeline.h`). -/
def clamp01 (x : α) : α := min 1.0 (max 0.0 x)

/-- `x ≥ 0` rounded to the nearest natural number, halves upward (C `round` and `lround` for
non-negative arguments). -/
def roundNat (x : α) : Nat := natFloor (x + 0.5)

end Scalar

/-- Double precision floating point numbers, with the C library's functions. This is the reading
the numeric checks run. -/
instance instScalarFloat : Scalar Float where
  natCast := Float.ofNat
  decLt := Float.decLt
  decLe := Float.decLe
  sqrt := Float.sqrt
  cbrt := Float.cbrt
  exp := Float.exp
  log := Float.log
  pow := Float.pow
  tanh := Float.tanh
  natFloor x := x.floor.toUInt64.toNat

end Negicc
