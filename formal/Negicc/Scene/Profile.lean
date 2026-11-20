/-
The film profile of a bracket: input curves, colour table, output curves.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The film profile

Each exposure bracket of a profile bundle has an ICC profile built from a shot of the chart. Its
`AToB0` element takes the corrected RGB of the negative (proportional to transmission) to the
colour of the scene, as XYZ under D50 (the ICC profile connection space). How a profile is
built is not part of this specification: its tables are given data. What the conversion does with
them is, and is defined here as `applyProfile` (`apply_profile` in `color_pipeline.h`):

1. each channel through its **input curve**, a sampled table over `[0, 1]`;
2. a **matrix and offset**, clamped to `[0, 1]` (the identity in the bundles' profiles);
3. the **colour table**, a three-dimensional grid of XYZ values read by tetrahedral
   interpolation (`clutEval`, `interpolate_clut`);
4. each channel through its **output curve**, then the PCS scale `1.99996948`.

The colour table is what makes the conversion non-linear in colour: it is fitted to the chart's
patches. Its interpolation is exact at the grid's nodes and, inside a cell, a weighted mean of
four of the cell's corners with non-negative weights that add up to one, so it never leaves the
range of its nodes (`clutEval_mem`).

A profile without a colour table (an empty `clut` or a zero grid size, the C++'s null pointer or
zero `clut_dim`) skips step 3: the output curves then read the input curves' values, and the matrix
result is not used, as in `apply_profile`. The bundles' profiles all have a table.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The film profile's tables. The colour table `clut` holds `dim.x * dim.y * dim.z` nodes of three
values each, the red index varying slowest: node `(i, j, k)` starts at
`((i * dim.y + j) * dim.z + k) * 3`, the layout of the C++ `[dim_r][dim_g][dim_b][3]`. -/
structure Profile (α : Type) where
  /-- The input curve of each channel, over `[0, 1]`. -/
  inCurves : Vec3 (Array α)
  /-- The matrix between the input curves and the colour table. -/
  matrix : Mat3 α
  /-- The offset added after the matrix. -/
  offset : Vec3 α
  /-- The number of grid points of the colour table along red, green and blue. -/
  dim : Vec3 Nat
  /-- The colour table's values. -/
  clut : Array α
  /-- The output curve of each channel, over `[0, 1]`. -/
  outCurves : Vec3 (Array α)

/-- The ICC PCS encoding scale applied after the output curves (`kPcsScale`). The C++ literal
`1.99996948f` is, in single precision, `1 + 32767/32768`. -/
def pcsScale : α := 1.99996948

namespace Profile

/-- The three values of the colour table's node `(i, j, k)`; a missing entry reads as `0`. -/
def node (p : Profile α) (i j k : Nat) : Vec3 α :=
  let o := ((i * p.dim.y + j) * p.dim.z + k) * 3
  ⟨p.clut.getD o 0.0, p.clut.getD (o + 1) 0.0, p.clut.getD (o + 2) 0.0⟩

end Profile

/-- The weighted sum `v0 w0 + va w1 + vp w2 + vq w3` of four nodes, component by component. -/
def tetraBlend (v0 va vp vq : Vec3 α) (w0 w1 w2 w3 : α) : Vec3 α :=
  ⟨v0.x * w0 + va.x * w1 + vp.x * w2 + vq.x * w3,
   v0.y * w0 + va.y * w1 + vp.y * w2 + vq.y * w3,
   v0.z * w0 + va.z * w1 + vp.z * w2 + vq.z * w3⟩

/-- The profile has a colour table to interpolate: a non-empty table and a grid of at least one
node along each axis (`apply_profile`'s test before `interpolate_clut`). -/
def Profile.hasClut (p : Profile α) : Bool :=
  0 < p.dim.x && 0 < p.dim.y && 0 < p.dim.z && 0 < p.clut.size

/-- Tetrahedral interpolation of the colour table at `t` in `[0, 1]³` (`interpolate_clut`).

Precondition: `t` in `[0, 1]³` and every grid size at least 1, which `applyProfile` guarantees (it
clamps the coordinates and tests `hasClut`). On that domain the definition is the C++'s. Outside
it the two differ: here the lower corner's index is held at `0`, while `interpolate_clut` computes
the upper corner's index before it clamps the lower one, so a coordinate at or below
`-1 / (dim - 1)` makes it read before the table.

Each coordinate is scaled to the grid; `(i, j, k)` is the cell's lower corner and `(dr, dg, db)`
the position inside the cell. The cell is cut into six tetrahedra by the order of `dr`, `dg`,
`db`; all six share the corners `v000` and `v111`. The value is the weighted sum of the four
corners of the tetrahedron that holds the point: with the three positions sorted as
`d1 ≥ d2 ≥ d3`, the weights are `1 - d1`, `d1 - d2`, `d2 - d3` and `d3`, on `v000`, then the
corner one step along the axis of `d1`, then the corner one more step along the axis of `d2`, then
`v111`. -/
def clutEval (p : Profile α) (t : Vec3 α) : Vec3 α :=
  let sr := t.x * ((p.dim.x - 1 : Nat) : α)
  let sg := t.y * ((p.dim.y - 1 : Nat) : α)
  let sb := t.z * ((p.dim.z - 1 : Nat) : α)
  let i := min (natFloor sr) (p.dim.x - 1)
  let j := min (natFloor sg) (p.dim.y - 1)
  let k := min (natFloor sb) (p.dim.z - 1)
  let i1 := min (i + 1) (p.dim.x - 1)
  let j1 := min (j + 1) (p.dim.y - 1)
  let k1 := min (k + 1) (p.dim.z - 1)
  let dr := sr - (i : α)
  let dg := sg - (j : α)
  let db := sb - (k : α)
  let v000 := p.node i j k
  let v100 := p.node i1 j k
  let v010 := p.node i j1 k
  let v110 := p.node i1 j1 k
  let v001 := p.node i j k1
  let v101 := p.node i1 j k1
  let v011 := p.node i j1 k1
  let v111 := p.node i1 j1 k1
  if dg ≤ dr ∧ db ≤ dg then tetraBlend v000 v100 v110 v111 (1.0 - dr) (dr - dg) (dg - db) db
  else if db ≤ dr ∧ dg < db then tetraBlend v000 v100 v101 v111 (1.0 - dr) (dr - db) (db - dg) dg
  else if dr < db ∧ dg ≤ dr then tetraBlend v000 v001 v101 v111 (1.0 - db) (db - dr) (dr - dg) dg
  else if dr < dg ∧ db ≤ dr then tetraBlend v000 v010 v110 v111 (1.0 - dg) (dg - dr) (dr - db) db
  else if db ≤ dg ∧ dr < db then tetraBlend v000 v010 v011 v111 (1.0 - dg) (dg - db) (db - dr) dr
  else tetraBlend v000 v001 v011 v111 (1.0 - db) (db - dg) (dg - dr) dr

/-- The film profile applied to a corrected RGB value: the PCS XYZ (D50) of the scene colour
(`apply_profile`). -/
def applyProfile (p : Profile α) (v : Vec3 α) : Vec3 α :=
  let v1 := Vec3.zipWith tableEval p.inCurves v
  let t := (p.matrix.mulVec v1 + p.offset).map clamp01
  let v2 := if p.hasClut then clutEval p t else v1
  (Vec3.zipWith tableEval p.outCurves v2).map (· * pcsScale)

end Negicc
