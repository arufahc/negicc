/-
The base definitions, without the real-number instance.
-/
module

public import Negicc.Base.Scalar
public import Negicc.Base.Vec
public import Negicc.Base.Table
public import Negicc.Base.Colour

/-!
# The base definitions

Everything in `Negicc.Base` that does not need Mathlib: the `Scalar` class with its `Float`
instance, 3-vectors and matrices, sampled tables, and the colour primitives. An executable that
runs the formulas over `Float` imports this module (and the corresponding modules of the other
parts), so that it links without Mathlib.
-/
