/-
The 3x3 matrix operations the print emulation uses.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# 3x3 matrices of the print

The print emulation inverts and transposes 3x3 matrices and sums them (`invert3` and
`multiply3` in `np_print.cc`; the inverses written out in `print_table`). The product is
`Mat3.mul` and the diagonal matrix `Mat3.diag` of `Negicc.Base.Vec`.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

namespace Mat3

/-- The entry in row `i` and column `j` (`0`, `1`, `2`; any other index reads row or column 2). -/
def entry (m : Mat3 α) (i j : Nat) : α :=
  let r := if i = 0 then m.row0 else if i = 1 then m.row1 else m.row2
  if j = 0 then r.x else if j = 1 then r.y else r.z

/-- The determinant, expanded along the first row as `invert3` writes it. -/
def det (m : Mat3 α) : α :=
  m.row0.x * (m.row1.y * m.row2.z - m.row1.z * m.row2.y) -
    m.row0.y * (m.row1.x * m.row2.z - m.row1.z * m.row2.x) +
    m.row0.z * (m.row1.x * m.row2.y - m.row1.y * m.row2.x)

/-- The inverse as the adjugate over the determinant, entry for entry as `invert3` (and
`print_table`) write it. For a matrix whose determinant is not zero it is the inverse
(`Mat3.mul_adjInv` in `Negicc.Print.Props`). -/
def adjInv (m : Mat3 α) : Mat3 α :=
  let d := det m
  ⟨⟨(m.row1.y * m.row2.z - m.row1.z * m.row2.y) / d, (m.row0.z * m.row2.y - m.row0.y * m.row2.z) / d,
    (m.row0.y * m.row1.z - m.row0.z * m.row1.y) / d⟩,
   ⟨(m.row1.z * m.row2.x - m.row1.x * m.row2.z) / d, (m.row0.x * m.row2.z - m.row0.z * m.row2.x) / d,
    (m.row0.z * m.row1.x - m.row0.x * m.row1.z) / d⟩,
   ⟨(m.row1.x * m.row2.y - m.row1.y * m.row2.x) / d, (m.row0.y * m.row2.x - m.row0.x * m.row2.y) / d,
    (m.row0.x * m.row1.y - m.row0.y * m.row1.x) / d⟩⟩

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
