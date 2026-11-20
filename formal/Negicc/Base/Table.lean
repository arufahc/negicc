/-
Sampled tables, sums and means.
-/
module

public import Negicc.Base.Scalar

@[expose] public section

/-!
# Sampled tables, sums and means

A table is an `Array α` of samples of a function at equally spaced points. `tableAt` reads it at
a fractional index by linear interpolation; `tableEval` reads it as a curve over `[0, 1]`, the
way the film profile's curves are stored (`apply_trc` in `color_pipeline.h`).

`sum` and `mean` are the sum and the arithmetic mean of an array.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The point a share `t` of the way from `a` to `b`: `a (1 - t) + b t`. -/
def lerp (a b t : α) : α := a * (1.0 - t) + b * t

/-- The table read at the fractional index `s` by linear interpolation between the two samples
around it. The segment is the one that contains `s`, held to the first and the last segment: below
index `0` and above index `size - 1` the end segment's line is continued (extrapolation). The
table needs at least two samples (`tableEval` reads a table of one sample as that sample, as
`apply_trc` does); a missing sample reads as `0`. Inside the
table, for `0 ≤ s ≤ size - 1`, the value lies between the samples (`tableEval_mem`). -/
def tableAt (tab : Array α) (s : α) : α :=
  let k := min (natFloor s) (tab.size - 2)
  lerp (tab.getD k 0.0) (tab.getD (k + 1) 0.0) (s - (k : α))

/-- The table read as a curve over `[0, 1]`: sample `i` of `n` is the value at `i / (n - 1)`
(`apply_trc` in `color_pipeline.h`). An empty table is the identity, and a table of one sample is
that sample. -/
def tableEval (tab : Array α) (v : α) : α :=
  if tab.size = 0 then v
  else if tab.size = 1 then tab.getD 0 0.0
  else tableAt tab (v * ((tab.size - 1 : Nat) : α))

/-- The sum of the entries, added in order from the first. -/
def sum (xs : Array α) : α := xs.foldl (· + ·) 0.0

/-- The arithmetic mean of the entries. -/
def mean (xs : Array α) : α := sum xs / (xs.size : α)

end Negicc
