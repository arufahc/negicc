/-
The scene conversion's definitions, without Mathlib.
-/
module

public import Negicc.Scene.Decode
public import Negicc.Scene.Input
public import Negicc.Scene.FilmBase
public import Negicc.Scene.Profile
public import Negicc.Scene.Linear
public import Negicc.Scene.Shoulder
public import Negicc.Scene.Convert
public import Negicc.Scene.Bracket

/-!
# The scene conversion's definitions

Reading the scan, the film base, the per-pixel conversion from the camera's RGB to sRGB and
CIELAB, and a bracket's calibrated range. No Mathlib: an executable imports
this module.
-/
