/-
3-vectors and 3x3 matrices.
-/
module

public import Negicc.Base.Scalar

@[expose] public section

/-!
# 3-vectors and 3x3 matrices

`Vec3 α` is a triple. The pipeline uses it for a camera, corrected or linear sRGB colour (then
`x`, `y`, `z` are the red, green and blue channel), for the XYZ of the profile connection space,
and for a density per channel. `Mat3 α` is a 3x3 matrix given by its three rows, the layout of
the C++ `float m[9]` arrays (row-major). Its determinant `Mat3.det` and its inverse as the adjugate
over the determinant, `Mat3.adjInv`, serve the colour matrices built from primaries
(`Negicc.Base.Colour`), the white-light balance and the print.

The C++ counterparts are `negicc::Rgb`, `apply_matrix` and `candidate_matrix` in
`color_pipeline.h`.
-/

namespace Negicc

/-- A triple of values. For a colour, `x`, `y`, `z` are the red, green and blue channel. -/
@[ext] structure Vec3 (α : Type) where
  x : α
  y : α
  z : α

namespace Vec3

variable {α β γ : Type}

/-- The triple with `f` applied to each component. -/
def map (f : α → β) (v : Vec3 α) : Vec3 β := ⟨f v.x, f v.y, f v.z⟩

/-- The triple that combines `u` and `v` component by component. -/
def zipWith (f : α → β → γ) (u : Vec3 α) (v : Vec3 β) : Vec3 γ := ⟨f u.x v.x, f u.y v.y, f u.z v.z⟩

/-- The triple with three equal components. -/
def fill (a : α) : Vec3 α := ⟨a, a, a⟩

variable [Scalar α]

/-- The sum, component by component. -/
def add (u v : Vec3 α) : Vec3 α := ⟨u.x + v.x, u.y + v.y, u.z + v.z⟩

/-- The difference, component by component. -/
def sub (u v : Vec3 α) : Vec3 α := ⟨u.x - v.x, u.y - v.y, u.z - v.z⟩

/-- The product, component by component (a gain per channel). -/
def mul (u v : Vec3 α) : Vec3 α := ⟨u.x * v.x, u.y * v.y, u.z * v.z⟩

/-- Every component multiplied by `s`. -/
def smul (s : α) (v : Vec3 α) : Vec3 α := ⟨s * v.x, s * v.y, s * v.z⟩

/-- The dot product `u.x v.x + u.y v.y + u.z v.z`. -/
def dot (u v : Vec3 α) : α := u.x * v.x + u.y * v.y + u.z * v.z

/-- The largest component. -/
def maxComp (v : Vec3 α) : α := max v.x (max v.y v.z)

instance : Add (Vec3 α) := ⟨add⟩
instance : Sub (Vec3 α) := ⟨sub⟩

end Vec3

/-- A 3x3 matrix, given by its rows. `⟨⟨a, b, c⟩, ⟨d, e, f⟩, ⟨g, h, i⟩⟩` is the C++ array
`{a, b, c, d, e, f, g, h, i}`. -/
@[ext] structure Mat3 (α : Type) where
  row0 : Vec3 α
  row1 : Vec3 α
  row2 : Vec3 α

namespace Mat3

variable {α : Type} [Scalar α]

/-- The matrix times a column vector (`apply_matrix` in `color_pipeline.h`). -/
def mulVec (m : Mat3 α) (v : Vec3 α) : Vec3 α := ⟨m.row0.dot v, m.row1.dot v, m.row2.dot v⟩

/-- The first, second and third column. -/
def col0 (m : Mat3 α) : Vec3 α := ⟨m.row0.x, m.row1.x, m.row2.x⟩
@[inherit_doc col0] def col1 (m : Mat3 α) : Vec3 α := ⟨m.row0.y, m.row1.y, m.row2.y⟩
@[inherit_doc col0] def col2 (m : Mat3 α) : Vec3 α := ⟨m.row0.z, m.row1.z, m.row2.z⟩

/-- The matrix product `a b`. -/
def mul (a b : Mat3 α) : Mat3 α :=
  ⟨⟨a.row0.dot b.col0, a.row0.dot b.col1, a.row0.dot b.col2⟩,
   ⟨a.row1.dot b.col0, a.row1.dot b.col1, a.row1.dot b.col2⟩,
   ⟨a.row2.dot b.col0, a.row2.dot b.col1, a.row2.dot b.col2⟩⟩

/-- The diagonal matrix with diagonal `d`. -/
def diag (d : Vec3 α) : Mat3 α := ⟨⟨d.x, 0.0, 0.0⟩, ⟨0.0, d.y, 0.0⟩, ⟨0.0, 0.0, d.z⟩⟩

/-- The identity matrix. -/
def identity : Mat3 α := diag (Vec3.fill 1.0)

/-- The matrix with row `i` multiplied by the `i`-th component of `s`: `diag s` times `m`. -/
def scaleRows (s : Vec3 α) (m : Mat3 α) : Mat3 α :=
  ⟨m.row0.smul s.x, m.row1.smul s.y, m.row2.smul s.z⟩

/-- The entry `(i, j)` of a matrix, indices taken modulo 3. -/
def entry (m : Mat3 α) (i j : Nat) : α :=
  let row := match i % 3 with | 0 => m.row0 | 1 => m.row1 | _ => m.row2
  match j % 3 with | 0 => row.x | 1 => row.y | _ => row.z

/-- The adjugate `det(m) m⁻¹`, entry `(j, i)` the cofactor of `(i, j)`, as `film_base_balance_gains`
forms it. -/
def adjugate (m : Mat3 α) : Mat3 α :=
  let cof := fun (i j : Nat) =>
    m.entry (i + 1) (j + 1) * m.entry (i + 2) (j + 2) - m.entry (i + 1) (j + 2) * m.entry (i + 2) (j + 1)
  ⟨⟨cof 0 0, cof 1 0, cof 2 0⟩, ⟨cof 0 1, cof 1 1, cof 2 1⟩, ⟨cof 0 2, cof 1 2, cof 2 2⟩⟩

/-- The determinant, along the first row with the adjugate's first column. -/
def det (m : Mat3 α) : α :=
  m.row0.x * m.adjugate.row0.x + m.row0.y * m.adjugate.row1.x + m.row0.z * m.adjugate.row2.x

/-- The inverse as the adjugate over the determinant, entry for entry as `invert3` (and
`print_table`) write it; the determinant is `Mat3.det` (the C++ expands it along the first row in
the same products, added in another order). For a matrix whose determinant is not zero it is the inverse
(`Mat3.mul_adjInv` in `Negicc.Base.Props`). -/
def adjInv (m : Mat3 α) : Mat3 α :=
  let d := det m
  ⟨⟨(m.row1.y * m.row2.z - m.row1.z * m.row2.y) / d, (m.row0.z * m.row2.y - m.row0.y * m.row2.z) / d,
    (m.row0.y * m.row1.z - m.row0.z * m.row1.y) / d⟩,
   ⟨(m.row1.z * m.row2.x - m.row1.x * m.row2.z) / d, (m.row0.x * m.row2.z - m.row0.z * m.row2.x) / d,
    (m.row0.z * m.row1.x - m.row0.x * m.row1.z) / d⟩,
   ⟨(m.row1.x * m.row2.y - m.row1.y * m.row2.x) / d, (m.row0.y * m.row2.x - m.row0.x * m.row2.y) / d,
    (m.row0.x * m.row1.y - m.row0.y * m.row1.x) / d⟩⟩

end Mat3

end Negicc
