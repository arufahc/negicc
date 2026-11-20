# The print emulation in Lean: coverage

DESIGN.md section 5 against `Negicc.Print`. "Checked" means compared by `lake exe checkPrint` with
values the C++ computed (`testvectors/print.json`, written by `testvectors/gen/gen_print.cc`), with
the tolerance stated in `CheckPrint.lean`. "Proved" names the theorems of `Negicc.Print.Props`.

## Code function → Lean name

| C++ (file: function) | Lean | Proved | Checked |
|---|---|---|---|
| `np_print.cc`: CIE tables | `cieXBar`, `cieYBar`, `cieZBar`, `cieD50`, `cieD55` | | through `observerWeights` |
| `np_print.cc`: `interp` | `interp` | | through every table read |
| `np_print.cc`: `resample_column` | `resampleColumn` | | dyes and base of three materials |
| `np_print.cc`: `amounts_to_xyz` | `amountsToXyz` | | through `paperXyz`, the table, the cube |
| `np_print.cc`: `exposure_to_xyz` | `exposureToXyz`, `printOfDensity` | `exposureOf_grey`, `exposureOf_shift` | through `aimSolve`, the table |
| `np_print.cc`: `print_paper_xyz` | `paperXyz` | | yes |
| `load_spektrafilm_print`: observer and white | `observerWeights`, `observerWhite` | `observerWhite_y` | yes |
| `load_spektrafilm_print`: lamp, response | `planckLamp`, `printResponse` | | through `Q` |
| `load_spektrafilm_print`: CAT02 to sRGB | `toSrgbMatrix`, `cat02`, `xyzToSrgbLindbloom` | `toSrgbMatrix_white` | yes |
| `load_spektrafilm_print`: aim (Newton) | `aimNewton`, `aimSolve` | `aimSolve_residual`, `loadMaterial_grey` | yes |
| `load_spektrafilm_print` | `loadMaterial`, `SpektrafilmProfile.sensitivity` | `loadMaterial_grey` | yes |
| `np_print.cc`: `film_to_print_density` | `filmToPrintDensity`, `FilmModel.samples`, `colourVariation` | | through `P` (3 measures, 3 materials) |
| `load_spektrafilm_print`: `print_matrix` | `printMatrix`, `materialPrintMatrix` | `printMatrix_mul_m`, `printMatrix_of_allOne` | yes, Status M calibrated (φ ≠ 1) included |
| `np_print.cc`: `invert3`, `multiply3` | `Mat3.invert3`, `Mat3.adjInv`, `Mat3.mul` | `Mat3.mul_adjInv`, `Mat3.adjInv_mul`, `Mat3.invert3_eq_some` | through `P` |
| `np_print.cc`: `on_grid`, `grey_grid` | `onGrid`, `GreyBalance.grid` | | through the curves |
| `np_print.cc`: `bracket_mid_apd` | `bracketMidApd` | | yes |
| `np_print.cc`: `curve_grey_at`, `curve_grey` | `curveGreyAt` | | yes |
| `np_print.cc`: `curve_slopes_at`, `curve_slopes` | `curveSlopesAt`, `curveGamma` | `curveSlopesAt_equal_contrast` | yes |
| `np_print.cc`: `print_tilt_at` (both) | `printTiltAt`, `tiltSegment` | `tiltSegment_left`, `_right`, `_below`, `_above` | yes |
| `np_print.cc`: `paper_curve` | `paperCurve`, `paperChannel`, `runningMin` | `paperChannel_strictAnti`, `paperChannel_mem`, `paperChannel_first` | yes |
| `np_print.cc`: `bracket_print_curves` | `bracketPrintCurves`, `unknee`, `inputLightDensity`, `rampedRunMax` | `bracketPrintCurves_strictMono`, `kneeCompress_unknee`, `inputLightDensity_base` | yes (both lights) |
| `PrintContinuum::curves_at`, `tilt_at`, `a_lo`, `a_hi` | `PrintContinuum.curvesAt`, `.aLo`, `.aHi` | | yes |
| `PrintContinuum::trc_at`, `set_curves_at` | `PrintContinuum.trcAt`, `icc16` | | yes |
| `np_print.cc`: `print_table` | `printTable`, `tableNode`, `nodeAlpha`, `pcsAdaptation` | `table_grey_axis`, `pcsAdaptation_white` | every 4th node, and every node of two bundle ICCs (one per light) to one code |
| `np_print.cc`: `write_print_icc` (table and curve codes) | `printTableCodes`, `iccCode16` | | against the bundles' files |
| `np_pipeline.cc`: `to_paper_white` | `toPaperWhite`, `paperBase`, `lightestNode` | `toPaperWhite_y_le`, `toPaperWhite_base` | yes |
| `np_print.cc`: `scale_print_icc_table` | `scaleIccCodes` | | yes |
| `color_pipeline.h`: `convert_pixel_pcs`, `convert_pixel_display`, `linear_srgb_to_lab` through a print profile | `Negicc.Scene` `convertPixelPcs`, `convertPixelDisplay`, `pixelLab` with `trcAt` and `toPaperWhite` | | yes (a print's pixel, four `(A, g, b)`, two `E`; no shoulder, no tone stage) |
| `np_print.h`: `density_encoding` | `densityEncoding` | | yes |
| `np_print.cc`: `density_encode_in_place` | `densityEncodePixel`, `densityTransmittance`, `lightDensity`, `densityCode` | `lightDensity_one`, `lightDensity_le_four`, `lightDensity_anti`, `densityCode_zero` | codes and counts exactly, 3 measures × 2 encodings |
| `np_print.cc`: `density_balance_offset` | `densityBalanceOffset` | `densityBalanceOffset_sum` | yes |
| `np_print.cc`: `print_balance` | `printBalance` | `printBalance_eq` | through the CLF matrix |
| `np_print.cc`: `frame_print` | `framePrint`, `FrameGrey` | | own search and bracket, 3 measures |
| `np_print.cc`: `frame_print_clf` (first matrix) | `frameClfMatrix`, `clfPrintingDensity`, `codeDensity` | `clfPrintingDensity_slope_form`, `clf_mid_grey`, `clf_grey_density_cancels` | yes, 12 frames |
| `np_print.cc`: `print_clf_nodes` | `clfExposureOffsets`, `clfTop`, `clfCurveRow`, `clfCubeNode` | `exposureOf_shift` | yes |
| `color_pipeline.h`: `compute_inner_crop` | `innerCrop` (`Negicc.Scene.Bracket`) | | yes, here too |
| `multi_target_optimizer.cc`: `print_floor_factor` | `printFloorFactor`, `floorChannel`, `percentilePoint` | `printFloorFactor_ge_one`, `printFloorFactor_ge_channel` | yes |
| `multi_target_optimizer.cc`: `raise_base_to_print_floor` | `baseBeforeFloor`, `raisedBase` | | yes |
| `np_pipeline.cc`: `apply_print_floor` (the frame) | `floorFrame`, `clampRect` | | no: a lambda of the pipeline |
| `search_print_material`: E held, result's E | `searchExposure`, `resultExposure` | `resultExposure_held` | no: local to the search |
| `search_print_material`: A0 | `bisectLevel`, `meterA0`, `a0AtRangeEnd` | `bisectLevel_spec`, `meterA0_mem` | no: local to the search |
| `search_print_material`: bootstrap | `bootstrapAs` | | no: local to the search |
| `search_print_material`: march, golden section | `march`, `marchLoop`, `bracketA`, `goldenLoop`, `goldenSection`, `lineSearchA` | `lineSearchA_mem` and the lemmas before it | no: local to the search |
| `search_print_material`: the fixed point | `printSearchLoop`, `printSearch`, `keepBest` | `printSearch_a_mem` | no: around the network |
| `search_print_material`: nearest bracket | `nearestBracket` | | no: local to the search |
| `multi_target_optimizer.cc`: `solve_print_knobs` | `solvePrintKnobs`, `knobNewton`, `knobSolve`, `knobLargest` | `solvePrintKnobs_a_mem` | no: its render is `GreyPoint` |

## Not formalized

- **The intent network and the optimiser** (`fr.predict`, `fr.residual`, `fr.search`,
  `update_correction`, the screening of `g` and `b`): an oracle of stated shape
  (`PrintSearchOracle`); the network's weights are opaque. For the same reason the search's control
  (`Negicc.Print.Search`) is proved, not checked.
- **The knobs' render** (`GreyPoint::reference_grey`, `grey`, `lstar_after_stops`): the conversion's
  knob solver of DESIGN.md 4.2, taken as a function by `solvePrintKnobs`.
- **The sidecar's shares** (`above_knee_pct`, `past_knee_pct`, `past_cut_pct`, `above_white_pct`,
  `at_white_pct`): counts recorded in the sidecar; they change no pixel.
- **File handling**: parsing and validating a spektrafilm profile (`check_spektrafilm_print`), the
  bundle's choice between its print ICC and a computed table (SHA-256 and curve comparison in
  `bracket_print_profiles`), writing ICC, TIFF, DPX and CLF files, the metadata JSON.
