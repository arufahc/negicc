# The Auto search and the networks: coverage

The Lean modules `Negicc.Auto.*` state DESIGN.md 3.5 (the networks: how their input is made, what
they give back, how it is used) and 4.2 (the Auto search) as definitions over `Scalar`, read over
`ℝ` for the theorems of `Negicc.Auto.Props` and over `Float` for the check `lake exe checkAuto`.

The networks enter as functions of stated shape: `IntentNet` takes the network's input (pixel
values in three planes and three tone weights per cell) and gives one CIELAB mean `μ` and one
spread `σ` per cell. Everything around them is stated: the preview, its resizing and
normalisation, the cells a render is judged on, the losses, the optimiser, the loop, the brackets'
ranges and every decision of the search.

## Code and specification

"Checked" names the group of `lake exe checkAuto` that compares the definition with the C++;
"search" means the end-to-end runs of `search_conversion_params` (12 frames, both lights, pins,
the clip fallback taken and refused, the highlight pullback's re-search), "solver" the runs of
`solve_dinov3_intent`, "heads" the heads' forward passes on weights of their files' layout drawn at
random.

| C++ (file: function) | Lean | Proved | Checked |
|---|---|---|---|
| `color_pipeline.h`: `compute_inner_crop` | `innerCrop` (of `Negicc.Scene.Bracket`) | `innerCrop_spec` | geometry |
| `color_pipeline.h`: `compute_aspect_preserved_shape` | `aspectShape`, `gridSide` | `aspectShape_grid` | geometry |
| `rgb_view.cc`: `resample_view_rgb16` | `resampleU16`, `resampleImage`, `areaTaps`, `areaPixel`, `bilinearPixel`, `weightedMean`, `toU16` (of `Negicc.Scene.Decode`) | | resample |
| `dinov3_preprocess.cc`: `antialiased_bilinear_resize_rgb` | `aaResize`, `aaTaps`, `aaSample`, `clipU8` | | resample |
| `dinov3_preprocess.cc`: `compute_tone_weights` | `toneWeights`, `toneBins` | `toneBins_sum` | resample |
| `dinov3_engine.cc`: `DinoV3Engine::infer` (resize, ImageNet normalisation, pool weights, network) | `netInput`, `pixelValues`, `normalizeU8`, `infer`, `IntentNet` | | search, solver |
| `dinov3_engine.h`: `NeutralHead::run` | `neutralP`, `NeutralWeights`, `gelu`, `dotFrom` | | heads |
| `dinov3_engine.h`: `ToneHead::run` | `toneHeads`, `ToneWeights`, `gelu`, `dotFrom` | | heads |
| `dinov3_auto_solver.cc`: `infer_checked`; `multi_target_optimizer.cc`: `SearchFrame::predict` | `fieldValid`, `SearchInput.predictChecked` | | search |
| `dinov3_auto_solver.cc`: `trained_cells_lab` | `trainedCellsLab` | | stats |
| `dinov3_auto_solver.cc`: `self_consistency_residual` | `z2`, `zsq`, `residual` | `z2_nonneg`, `z2_self`, `zsq_shift`, `zsq_nonneg`, `zsq_self` | stats |
| `multi_target_optimizer.cc`: `compute_frame_stats` | `frameStats`, `statsGrid`, `logAverage`, `cropGreenRange` (with `greenHistogram`, `sceneGreenRange` of `Negicc.Scene.Bracket`), `sceneRange` | | stats |
| `multi_target_optimizer.cc`: `u16_percentile_linear` | `u16PercentileLinear` (of `Negicc.Scene.Bracket`) | | stats |
| `dinov3_auto_solver.cc`: `intent_cells`, `candidate_config` | `intentCells`, `RenderSetup` | | cells |
| `dinov3_auto_solver.cc`: `intent_loss` | `intentLoss`, `whiteClippedChannels`, `IntentLoss.mean` | | cells |
| `SearchFrame::update_correction`, `SearchFrame::search` | `correction`, `aimField`, `SearchInput.correctionAt`, `SearchInput.search` | `zsq_aim` | search |
| `SearchFrame::set_lightness_wish`, the wish in `fixed_point` | `wishOf`, `withWish` | `withWish_L`, `withWish_mu_congr` | search |
| `mean_field` (in `search_conversion_params`) | `meanField` | | search |
| `bracket_enclosure`, `pull_centre` | `bracketEnclosure`, `pullCentre` (of `Negicc.Scene.Bracket`), `bracketRange`, `BracketRange` | `usableInverted_iff_not_admissible`, `bracketEnclosure_scale` | brackets, search |
| `filter_admissible_targets` | `filterAdmissible` | | brackets |
| `frame_scale`, `merged_matrix` | `frameScale` (with `mergedMatrix`, `exposureNorm` of `Negicc.Scene.Input`) | | scale |
| `rank`, `best_fit` (in `search_conversion_params`) | `fitRank`, `fitBefore`, `minFirst`, `bestFit` | `minFirst_spec` | search |
| `metered`, `deepest_first` | `metered`, `deepestFirst` | | search |
| `trust_bounds` | `trustBounds` | `trustBounds_mem` | search |
| `tourney_score` | `tourneyScore`, `enclosurePenalty` | `enclosurePenalty_nonneg`, `enclosurePenalty_eq_zero` | search |
| `compute_penalties` (both) | `penalty`, `clipPenalty`, `centrePenalty`, `enclosurePenalty` | `penalty_nonneg`, `clipPenalty_eq_zero`, `centrePenalty_self` | optimiser |
| `optimize_impl`, `optimize_target_for_intent` | `optimize`, `eCandidates`, `sortUnique`, `dedupStep`, `eStep`, `eMoves`, `ePass`, `eRefineStep`, `gGrid`, `bGrid`, `gSweep`, `bSweep`, `chainMoves`, `chainStep`, `chain`, `secondPass`, `firstMin` | `optimize_e_mem`, `optimize_e_pinned`, `optimize_gains_near`, `eCandidates_mem`, `ePass_mem`, `chain_bound`, `firstMin_mem` | optimiser (13 cases), search |
| `SearchFrame::fixed_point` | `fixedPoint`, `fpGo`, `bundleConv`, `diverges`, `relChange`, `gainsDist`, `SearchInput.runLoop`, `SearchInput.measureStep` | `fixedPoint_steps` | search |
| `SearchFrame::shoulder_guard` | `Run.guard` | | search |
| `winner` | `winner` | `winner_spec` | search |
| the clip fallback (`ClipFallbackOutcome`) | `clipFallbackNeeded`, `clipFallbackVerdict` | `clipFallbackVerdict_taken` | search |
| the highlight pullback | `pullbackTier`, `pullbackNudge`, `pullbackDlMin`, `pullbackBrighten`, `pullbackAccept` | `pullbackNudge_bound` | search (re-search; the nudge is not reached by the vectors) |
| `measure_highlight_clip`, `cpu_highlight_clip`, `percentile_from_hist_L`, `max_excess` | `ceilLin`, `clipStats`, `labBin`, `firstReaching`, `lHistPercentile`, `ClipStats.maxExcess` | | clip |
| `SearchFrame::measure_shoulder`, `shoulder_x_hi_from_hist` | `SearchInput.measureShoulder` (with `shoulderXHiFromHist` of `Negicc.Scene.Shoulder`) | | clip, search |
| `search_conversion_params` | `SearchInput.searchParams` and the definitions above | | search |
| `lstsq3` | `lstsq3`, `jacobiRotate`, `jacobiSweeps` | | metering |
| `GreyPoint::lab`, `lstar`, `newton`, `reference_grey` | `greyLab`, `greyInput`, `greyLstar`, `newton`, `newtonPoints`, `referenceGrey` | | metering |
| `exposure_for_lightness` | `exposureForLightness`, `meterGo`, `sceneVector` | `meterGo_mem`, `exposureForLightness_mem` | metering |
| `solve_grey_knobs`, `GreyPoint::gains`, `GreyPoint::grey`, `lstar_after_stops`, `knob_grey_miss_stops` | `solveGreyKnobs`, `greyGains`, `greyAt`, `largestReachable`, `lstarAfterStops`, `knobGreyMissStops` | | knobs |
| `solve_dinov3_intent` | `solveIntent`, `anchorExposure`, `upperMedian` (of `Negicc.WhiteLight.Fit`), `solverConv` | `fixedPoint_steps` | solver |
| `meter_bracket_exposures` | `meterBracketExposures` | | given |
| `measure_rolloff_x_hi` | `measureRolloffXHi` | | given |

## Not formalized

- **The networks' weights**: the backbone and the stocks' intent heads are a function of stated
  shape (`IntentNet`); the neutral and tone heads' forward passes are stated over their weights,
  which are data. The error function in GELU is a parameter (`Scalar` has none); the check passes
  a double precision one.
- **The tone heads' input features** (`pooled`, `token_sd`, `tone_mass` in `DinoV3Engine::infer`):
  outputs and statistics of the backbone's tokens, the network's own values.
- **Reading the bundle and the profcheck report** (`profile_bundle.cc`, `profile_highlight_ceiling`):
  file parsing; their values are the vectors' inputs. The parsed ceilings set only the per-channel
  exceedance shares, which no rule of the search reads.
- **`--probe_intent`** (a measurement aid that writes a JSON file and stops), the renderer interface
  (`PreviewRenderer`, `PassAudit`), timing and logging.
- **The gains in the print's curves** (`SolverConfig::gain_curves`): the print's search (5.6).
- **`resample_grid_rgb16`**: it has no caller; it is the area average of `resample_view_rgb16`.
- **`compute_bundle_enclosures`** (`profile_bundle.cc`), the ranges kept for given values and for a
  bracket chosen without the search, is `bundleEnclosure` of `Negicc.Scene.Bracket`.

## The check

`lake exe checkAuto` (`CheckAuto.lean`) reads `testvectors/auto/auto.json` (the components) and
`testvectors/auto/auto_search.json` (whole searches) and the bundles' ICC profiles under
`../profiles/`. Its tolerances are stated at the top of `CheckAuto.lean`; the C++ is single
precision and the specification double precision. Inputs given to the C++ as single precision
constants are passed in their single precision values (the crop ratio `0.9f`), so that a margin
like `110 · 0.05` falls on the same side of a half.

The network is replaced, in both the C++ and the check, by a stand-in of stated formula
(`StandInRunner` in `gen_auto.cc`, `standIn` in `CheckAuto.lean`): per cell, from the means `m` of
the three input planes over the cell and the cell's tone weights `w`,
`μ = (46 + 9 mean(m) + 4 (w_hi − w_sh), 1.5 + 3 (m_r − m_g), 4 + 3 (m_g − m_b))` and
`σ = (7 + 2 |m_g|, 4 + w_mid, 5)`. The field follows the render, so the loops, the tournament
and the fallbacks run as with the real network.

To regenerate the vectors, from the repository root with the submodules checked out:

```sh
make -j lib USE_CUDA=0 ORT_DIR=<onnxruntime>
ORT_DIR=<onnxruntime> formal/testvectors/auto/build.sh
OMP_NUM_THREADS=16 bin_out/gen_auto profiles/Portra400-A7R4-TriLED profiles/Portra400-A7R4-D65 formal/testvectors/auto
```
