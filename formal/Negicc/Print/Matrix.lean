/-
The 3x3 matrix operations the print emulation uses.
-/
module

public import Negicc.WhiteLight.Balance

@[expose] public section

/-!
# 3x3 matrices of the print

The print emulation inverts and transposes 3x3 matrices and sums them (`invert3` and
`multiply3` in `np_print.cc`; the inverses written out in `print_table`). The product is
`Mat3.mul` and the diagonal matrix `Mat3.diag` of `Negicc.Base.Vec`; the entry `Mat3.entry` and the
determinant `Mat3.det` are those of `Negicc.WhiteLight.Balance`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

namespace Mat3

/-- The inverse, or `none` when the determinant is below `1e-12` in magnitude (`invert3`). -/
def invert3 (m : Mat3 α) : Option (Mat3 α) :=
  if fabs (det m) < 1.0e-12 then none else some (adjInv m)

/-- The transpose. -/
def transpose (m : Mat3 α) : Mat3 α := ⟨m.col0, m.col1, m.col2⟩

/-- The sum, entry by entry. -/
def add (a b : Mat3 α) : Mat3 α := ⟨a.row0 + b.row0, a.row1 + b.row1, a.row2 + b.row2⟩

/-- The zero matrix. -/
def zero : Mat3 α := ⟨Vec3.fill 0.0, Vec3.fill 0.0, Vec3.fill 0.0⟩

/-- The outer product `u vᵀ`: entry `(r, c)` is `u_r v_c`. -/
def outer (u v : Vec3 α) : Mat3 α :=
  ⟨⟨u.x * v.x, u.x * v.y, u.x * v.z⟩, ⟨u.y * v.x, u.y * v.y, u.y * v.z⟩,
   ⟨u.z * v.x, u.z * v.y, u.z * v.z⟩⟩

/-- The sum of the three entries of each row: the matrix times `(1, 1, 1)`, added in the order of
the row. -/
def rowSums (m : Mat3 α) : Vec3 α :=
  ⟨m.row0.x + m.row0.y + m.row0.z, m.row1.x + m.row1.y + m.row1.z, m.row2.x + m.row2.y + m.row2.z⟩

/-- A matrix from the C++ layout `{a0, …, a8}` (row-major); a missing entry reads as `0`. -/
def ofArray (a : Array α) : Mat3 α :=
  ⟨⟨a.getD 0 0.0, a.getD 1 0.0, a.getD 2 0.0⟩, ⟨a.getD 3 0.0, a.getD 4 0.0, a.getD 5 0.0⟩,
   ⟨a.getD 6 0.0, a.getD 7 0.0, a.getD 8 0.0⟩⟩

end Mat3

end Negicc
