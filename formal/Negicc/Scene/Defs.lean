/-
The scene conversion's definitions, without Mathlib.
-/
module

public import Negicc.Scene.Input
public import Negicc.Scene.Profile
public import Negicc.Scene.Linear
public import Negicc.Scene.Shoulder
public import Negicc.Scene.Convert

/-!
# The scene conversion's definitions

The per-pixel conversion of a narrow-band scan, from the camera's RGB to sRGB and CIELAB. No
Mathlib: an executable imports this module.
-/
