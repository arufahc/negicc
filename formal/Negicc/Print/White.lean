/-
The print's white: the print ICC made relative to its paper.
-/
module

public import Negicc.Base.Defs

@[expose] public section

/-!
# The print's white

The print ICC's table is absolute: its lightest node is the paper's base under its viewing light.
The conversion shows the print relative to its paper, as the ICC's media-relative colorimetry
defines it (DESIGN.md 5.7): each of X, Y and Z of every node is multiplied by the PCS white's over
the paper base's, so the base becomes the display's white (`to_paper_white` in `np_pipeline.cc`).
The profile an output embeds gets the same scaling on its 16-bit codes
(`scale_print_icc_table` in `np_print.cc`).

The base is the node with the largest `Y`. After the scaling no node's `Y` passes the PCS white's
(`toPaperWhite_y_le` in `Negicc.Print.Props`); `X` and `Z` of a node other than the base can, by
up to a few parts in a thousand on the bundles' print ICCs.
-/

namespace Negicc

open Scalar

variable {α : Type} [Scalar α]

/-- The PCS white, D50 as an ICC profile stores it: `(63190, 65536, 54061) / 65536`
(`kPcsWhite` in `color_pipeline.h`). -/
def pcsWhite : Vec3 α := ⟨63190.0 / 65536.0, 1.0, 54061.0 / 65536.0⟩

/-- The largest XYZ a `lut16` holds, code `FFFFh` as a `u1Fixed15Number`: `1 + 32767 / 32768`
(`kPcsMax`). A table value `v` read from the profile stands for the XYZ `v kPcsMax`. -/
def pcsMax : α := 1.0 + 32767.0 / 32768.0

/-- The index of the node with the largest `Y` (the first such node) of a table of XYZ triples. -/
def lightestNode (clut : Array α) : Nat :=
  (List.range (clut.size / 3)).foldl
    (fun best n => if clut.getD (best * 3 + 1) 0.0 < clut.getD (n * 3 + 1) 0.0 then n else best) 0

/-- The paper base's XYZ: the lightest node, as XYZ (`to_paper_white`'s `paper_xyz`). -/
def paperBase (clut : Array α) : Vec3 α :=
  let b := lightestNode clut
  ⟨clut.getD (b * 3) 0.0 * pcsMax, clut.getD (b * 3 + 1) 0.0 * pcsMax, clut.getD (b * 3 + 2) 0.0 * pcsMax⟩

/-- The factors of the media-relative scaling: the PCS white's X, Y, Z over the base's. -/
def paperWhiteGains (base : Vec3 α) : Vec3 α :=
  ⟨pcsWhite.x / base.x, pcsWhite.y / base.y, pcsWhite.z / base.z⟩

/-- Entry `i` of a table of XYZ triples times `k.x`, `k.y` or `k.z` by its place in the triple. -/
def scaleTriples (k : Vec3 α) (clut : Array α) : Array α :=
  (Array.range clut.size).map fun (i : Nat) =>
    clut.getD i 0.0 * (if i % 3 = 0 then k.x else if i % 3 = 1 then k.y else k.z)

/-- The print's table made relative to its paper (`to_paper_white`). -/
def toPaperWhite (clut : Array α) : Array α := scaleTriples (paperWhiteGains (paperBase clut)) clut

/-- The 16-bit codes of an ICC's table multiplied by `k.x`, `k.y`, `k.z` by their place in the
triple, rounded (halves up) and held at 65535 (`scale_print_icc_table`). -/
def scaleIccCodes (k : Vec3 α) (codes : Array Nat) : Array Nat :=
  (Array.range codes.size).map fun (i : Nat) =>
    min 65535 (roundNat ((codes.getD i 0 : α) * (if i % 3 = 0 then k.x else if i % 3 = 1 then k.y else k.z)))

end Negicc
