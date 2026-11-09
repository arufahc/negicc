# White-Light DINOv3 Neutral-Anchor Curve Fitting: Progression & Experiment Trajectory

This document records the engineering progression, diagnostic traces, and empirical evaluation trajectory across the 12 white-light test uploads (`00..13`) and 20 holdout frames on the `exp-dinov3-neutral-anchors` branch (`input_curves.h`, `multi_target_optimizer.cc`).

---

## 1. End-to-End Trajectory Overview

| Stage | Commit / Tag | Core Mechanism | 12-Upload Heldout $\Delta E$ | 12-Upload `LowRC_Spread` | 12-Upload CrossSlope | 20-Holdout Heldout $\Delta E$ | Key Milestone / Findings |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **0. Uncurved Auto** | Web baseline | D65 profile + exposure/WB `(E, g, b)` only, no input curves | 6.01 | 12.89 | 9.41 | 3.78 | Strong shadow/highlight color crossover on white-light scans (`00` Tunnel, `02` Clouds, `07` Cat, `10` Square). |
| **1. Step 1–5 (`B5_2i`)** | `B5` (`b92f7e8`) | 4-band raw-chroma scene-neutral selection + `fit_input_curves_joint` `(g, b, a_G, a_B)` | 4.68 | 7.71 | 4.50 | 2.85 | Fixes shadow crossover on `00` (`a*` spread `0.60`), but raw-chroma selection pulls in colorful patches (`05` warm portrait cooled from `b*=+12.1` to `+9.0`; `06` IT8 selects 16 colored patches; `07` Cat fur `dE=13.34`). |
| **2. Phase 1 (`1a` vs `1b`)** | `562ce4c` | DINOv3 neutral anchor selection: $\operatorname{hypot}(\mu_a, \mu_b) + |\sigma_{ab}| < C_{\max}$ ($C_{\max}=6$) | — | — | — | — | Rejects colorful patches (`0` colored IT8 patches on `06`; `0` warm skin/wall cells on `05`; `0` foliage cells on `08`). Variant `1a` (raw $\mu$) selected over `1b` (median-subtracted $\mu$) to preserve non-anchor chroma (`14.49` vs `11.33`). |
| **3. Phase 2 (`2auto_2ii`)** | `562ce4c` | Adaptive `2A / 2B / 2C` log-transmittance curves on 4 luma bands + bracket re-evaluation (`2-ii`) at fixed target $L^*$ | 3.62 | 12.59 | 2.76 | 2.01 | Cuts heldout $\Delta E$ (`4.68 -> 3.62` on uploads, `2.85 -> 2.01` on holdout); fixes `06` IT8 bracket (`t=4`, `Yellow_b*_p90=+48.6`, `WhiteClip=3.50%`) and `07` Cat fur (`dE=5.16`). Revealed two failure modes on `00` and `02`. |
| **4. Step 1 Trace & Diagnosis** | Diagnostic trace | Per-band & per-region trace on `00` (Tunnel), `02` (Clouds), and `04..13` | — | — | — | — | Identified why `2auto_2ii` regressed on `00` (`a*` spread `16.54`: only `Band 3` had $\ge 5$ DINOv3 anchors while `B0..B2` had `1` cell each) and `02` (`mid_top_cloud` cyan `a*=-20.3, b*=+11.5`: `Band 2` mixed cyan sky edges into neutral cloud anchors, creating an $S$-bend vs `Band 3`). |
| **5. Steps 2–4 (`F1`, `F2`, `F1+F2`, `F1+F2+F3`)** | `1260724` | Collinearity-gated bands ($B_0 \to B_3$), single-band cross-tone extension, highlight cliff de-surging (`F2`), and common DINOv3 offset targeting (`F1` / `F3`) | **3.47** | **9.58** | **1.79** | **1.98** | Solves `00` Tunnel (`a*` spread `0.59`, `sunlit_asphalt=(+6.2,-0.8)`), `02` Clouds (`mid_top_cloud=(76.7,-1.6,-4.7)`, `cloud_gap=7.79`), `05` Portrait (`Warm_b*=+13.6`), `06` IT8 (`t=4`, `Yellow_b*_p90=+48.5`), `07` Cat (`Fur_dE=5.41`), `10` Square (`Heldout_dE=2.30`), and `12/13` Red Car (`LowRC_Spread=5.61/7.62`). |

---

## 2. Diagnostic Findings from Step 1 Trace (`00`, `02`, `05..13`)

### 2.1 `00` (`a676c08a86f54e3e`, Tunnel)
- **Anchor distribution in `2auto_2ii`:** `Band 0 (L* <= 22): 1`, `Band 1 (22 < L* <= 45): 1`, `Band 2 (45 < L* <= 70): 1`, `Band 3 (L* > 70): 22` (of which `18` sit in the clipped sky portal at $L^* \ge 92$).
- **Root cause:** With only `1` anchor cell in `B0`, `B1`, and `B2`, `2auto_2ii` had no true dark/midtone anchor constraint; fitting 4-band `2C` against single-cell bands twisted shadow/midtone slopes and produced `a*` spread `= 16.54` across the 4 tunnel check regions (`sunlit_asphalt`, `walkway_concrete`, `portal_wall`, `tunnel_ceiling`).
- **Fix in `F2` (`f_mode >= 2`):**
  1. Enforce a strict band minimum of $\ge 5$ anchors and $L^* > 6.5$.
  2. When strict DINOv3 anchors exist in only a single band (`n_strict_bands == 1`, as on `00` Tunnel and `04` Street Dancers), DINOv3 cannot constrain cross-tone slope on its own; `F2` delegates the 4-parameter `(g, b, a_G, a_B)` cross-tone solve (`level == 2`) to the $L^* < 92$ scene-neutral moment system (`rc_model.residuals`), achieving `a*` spread `= 0.59` and `sunlit_asphalt = (+6.2, -0.8)` ($\Delta E = 3.32$ vs `NegPy (+9.0, +1.0)`).

### 2.2 `02` (`36bd4b92c5504055`, Mountain & Clouds)
- **Anchor distribution in `2auto_2ii`:** `Band 0: 28`, `Band 1: 36`, `Band 2: 24`, `Band 3: 54`.
- **Root cause:** In `Band 2` ($45 < L^* \le 70$), DINOv3 selected cyan sky-transition cells (`Auto_uncurved` mean `a* = -4.9, b* = -11.4`), whereas `Band 0` (dark mountain shadow) and `Band 3` (bright cloud core) were true neutrals. Fitting unconstrained per-band `2C` segments between `Band 2` and `Band 3` created a severe highlight $S$-bend that drove `mid_top_cloud` to `(87.3, -20.3, +11.5)` (`cloud_gap = 20.51`).
- **Fix in `F2` (`f_mode >= 2`):**
  1. **$B_0 \to B_3$ Collinearity Gate:** Linearly interpolates expected $(a^*, b^*)$ from `Band 0` to `Band 3` as a function of band mean $L^*$; any midtone band (`B1` or `B2`) deviating by $> 5.5\,\Delta E_{ab}$ from the $B_0 \to B_3$ line is dropped as chromatic-contaminated.
  2. **Decomposed Regularization & Highlight Cliff De-Surging:** Regularizes mean slope by $\sqrt{\lambda}$ and adjacent segment differences by $1.5\sqrt{\lambda}$, and applies `apply_highlight_cliff_desurge` when `2B` is selected, bringing `mid_top_cloud` to **`(76.7, -1.6, -4.7)`** (matching `NegPy (75.3, -0.7, -4.6)` within `0.9 a*, 0.1 b*`) and reducing `cloud_gap` from `20.51` to **`7.79`**.

### 2.3 Common DINOv3 Offset Targeting (`F1` and `F3`)
- Across strict DINOv3 anchor bands, the mean predicted $(\mu_a, \mu_b)$ defines a scene-level common offset $o = (\text{strict\_}\mu_a, \text{strict\_}\mu_b)$:
  - **`F1` (`f_mode = 1` or `3`):** Targets $(a^*, b^*) = o$ across all bands instead of forcing $(0, 0)$, preserving intentional scene warmth on `05` (`Warm_b* = +13.6`, `Skin_b* = +11.7`).
  - **`F3` (`f_mode = 4`):** Damps $o$ by `0.35` (capped at $\|o\| \le 2.5$) when broad multi-band neutral coverage ($\ge 3$ strict bands with $\ge 64$ total anchors, or $\ge 150$ anchors) indicates a neutral daylight or target-chart scene (`02`, `06`, `09`), while keeping $1.0 \times o$ on low-key / warm scenes (`05`).

---

## 3. Complete Step 3–4 Evaluation Tables

### 3.1 Aggregate Metrics — 12 White-Light Uploads (`00..13`) & 20 Holdout Frames

| Arm | 12-Upload Heldout $\Delta E$ | 12-Upload Anchor $\Delta E$ | **12-Upload `LowRC_Spread`** | 12-Upload CrossSlope | 12-Upload BandSpread | 12-Upload IntentLoss | 12-Upload WhiteClip% | 12-Upload NonAnch $C^*$ | 20-Holdout Heldout $\Delta E$ | 20-Holdout `LowRC_Spread` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Auto_uncurved` | 6.01 | 6.01 | 12.89 | 9.41 | 8.92 | 13.27 | 0.61% | 14.10 | 3.78 | 12.57 |
| `B5_2i` | 4.68 | 4.68 | 7.71 | 4.50 | 6.45 | 12.48 | 0.46% | 11.41 | 2.85 | 7.77 |
| `2auto_2ii` | 3.62 | 3.10 | 12.59 | 2.76 | 2.73 | 13.77 | 0.34% | 14.49 | 2.01 | 10.07 |
| `F1_2ii` | 3.54 | 3.29 | 13.67 | 3.22 | 4.04 | 14.95 | 0.36% | 14.58 | 2.02 | 10.15 |
| **`F2_2ii`** | **3.46** | **3.40** | **9.66** | **1.91** | **4.85** | **12.70** | **0.32%** | **12.28** | **1.89** | **10.61** |
| **`F1F2_2ii`** | **3.50** | **3.47** | **9.55** | **1.82** | **4.88** | **12.53** | **0.30%** | **12.54** | **1.98** | **10.26** |
| **`F1F2F3_2ii`** | **3.47** | **3.46** | **9.58** | **1.79** | **4.92** | **12.58** | **0.31%** | **12.51** | **1.98** | **10.31** |

### 3.2 `00` (`a676c08a86f54e3e`, Tunnel) — 4 Check Regions $(a^*, b^*)$ & Cross-Region $a^*$ Spread

| Arm | `lvl` | `(g, b, a_G, a_B)` | `sunlit_asphalt` | `walkway_concrete` | `portal_wall` | `tunnel_ceiling` | **`a*` spread** | `asphalt dE vs NegPy` |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| `NegPy` | — | — | `(+9.0, +1.0)` | `(+8.0, +3.0)` | `(+8.0, +3.0)` | `(+6.0, +2.0)` | 3.00 | 0.00 |
| `NLP` | — | — | `(-1.0, -7.0)` | `(-4.0, +3.0)` | `(-4.0, +5.0)` | `(-7.0, +4.0)` | 6.00 | 12.81 |
| `Auto_uncurved` | — | `(0.900, 0.970, 0, 0)` | `(+4.7, -5.6)` | `(+10.2, +1.8)` | `(+9.9, +2.5)` | `(+14.8, +2.4)` | 10.13 | 7.90 |
| `B5_2i` | — | `(0.816, 0.872, -0.120, -0.139)` | `(+6.2, -0.8)` | `(+5.8, +1.0)` | `(+5.8, +2.4)` | `(+5.6, +2.7)` | 0.60 | 3.32 |
| `2auto_2ii` | 3 | `(0.938, 1.029, ...)` | `(+0.3, +0.1)` | `(+11.4, +4.3)` | `(+10.8, +5.3)` | `(+16.9, +3.8)` | 16.54 | 8.73 |
| `F1_2ii` | 3 | `(0.621, 0.762, ...)` | `(-5.4, -1.7)` | `(-15.4, +3.8)` | `(-12.8, +2.6)` | `(-14.1, +4.6)` | 9.98 | 14.68 |
| **`F2_2ii`** | **2** | **`(0.817, 0.873, -0.120, -0.139)`** | **`(+6.2, -0.8)`** | **`(+5.8, +1.0)`** | **`(+5.8, +2.4)`** | **`(+5.6, +2.7)`** | **0.59** | **3.32** |
| **`F1F2_2ii`** | **2** | **`(0.817, 0.873, -0.120, -0.139)`** | **`(+6.2, -0.8)`** | **`(+5.8, +1.0)`** | **`(+5.8, +2.4)`** | **`(+5.6, +2.7)`** | **0.59** | **3.32** |
| **`F1F2F3_2ii`** | **2** | **`(0.817, 0.873, -0.120, -0.139)`** | **`(+6.2, -0.8)`** | **`(+5.8, +1.0)`** | **`(+5.8, +2.4)`** | **`(+5.6, +2.7)`** | **0.59** | **3.32** |

### 3.3 `02` (`36bd4b92c5504055`, Mountain & Clouds) — 3 Check Regions $(L^*, a^*, b^*)$ & Cloud Gap

| Arm | `mid_top_cloud (L*, a*, b*)` | `white_core (L*, a*, b*)` | `blue_sky (L*, a*, b*)` | **`cloud_gap` (mid-top vs white core)** |
| :--- | :--- | :--- | :--- | ---: |
| `NegPy` | `(75.3, -0.7, -4.6)` | `(92.2, +1.7, +1.5)` | `(66.4, +2.0, -6.8)` | 6.63 |
| `NLP` | `(81.1, +1.0, -11.0)` | `(95.5, +1.5, -2.2)` | `(72.7, +3.2, -16.9)` | 8.82 |
| `Auto_uncurved` | `(90.6, -9.5, -3.0)` | `(94.9, -1.7, +0.7)` | `(79.3, -5.0, -14.8)` | 8.64 |
| `B5_2i` | `(73.3, -2.3, -2.9)` | `(91.0, +2.7, +2.0)` | `(66.3, -0.9, -5.9)` | 6.94 |
| `2auto_2ii` | `(87.3, -20.3, +11.5)` | `(94.7, -2.0, +2.2)` | `(72.3, -10.3, +4.4)` | 20.51 |
| `F1_2ii` | `(87.6, -20.9, +12.0)` | `(94.7, -2.0, +2.3)` | `(73.0, -11.9, +5.8)` | 21.24 |
| **`F2_2ii`** | **`(76.0, -1.8, -4.3)`** | **`(92.9, +0.9, +3.2)`** | **`(68.3, +0.1, -9.4)`** | **7.96** |
| **`F1F2_2ii`** | **`(76.7, -1.6, -4.7)`** | **`(93.3, +0.6, +2.8)`** | **`(68.6, +0.6, -10.4)`** | **7.79** |
| **`F1F2F3_2ii`** | **`(76.7, -1.6, -4.7)`** | **`(93.3, +0.6, +2.8)`** | **`(68.6, +0.6, -10.4)`** | **7.79** |

### 3.4 Per-Image Progression Across All 12 Uploads (`00..13`)

| Upload (`job_id`) | Scene | `Auto` Heldout $\Delta E$ | `B5_2i` Heldout $\Delta E$ | `2auto_2ii` Heldout $\Delta E$ | **`F1F2F3_2ii` Heldout $\Delta E`** | `2auto_2ii` `LowRC_Spread` | **`F1F2F3_2ii` `LowRC_Spread`** | Scene-Specific Highlight |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `00` (`a676c08a86f54e3e`) | Tunnel | 6.35 | 7.11 | 6.71 | **4.66** | 8.59 | **2.95** | `a*` spread `0.59`, `sunlit_asphalt = (+6.2, -0.8)` |
| `02` (`36bd4b92c5504055`) | Mountain & Clouds | 10.01 | 6.64 | 5.26 | **5.29** | 12.15 | **7.92** | `mid_top_cloud = (76.7, -1.6, -4.7)`, `cloud_gap = 7.79` |
| `04` (`95cf870d5fd64bc4`) | Street Dancers | 6.00 | 6.08 | 4.74 | **4.68** | 7.45 | **4.53** | `CrossSlope` cut from `7.82` to `1.60` |
| `05` (`6b1893bdd1754d80`) | Golden Hour Portrait | 3.70 | 5.09 | 2.81 | **2.77** | 8.40 | **8.40** | Preserves golden warmth (`Warm_b* = +13.6`, `Skin_b* = +11.7`) |
| `06` (`e1cd4112d3ee4d66`) | IT8 Target Chart | 4.55 | 3.71 | 4.92 | **4.69** | 10.74 | **9.88** | `t = 4`, `Yellow_b*_p90 = +47.8`, `WhiteClip = 3.17%` |
| `07` (`0226bc66e8eb4ac5`) | Cat in Blue Bag | 10.61 | 9.26 | 6.19 | **6.18** | 34.48 | **34.12** | `Fur_dE` reduced from `19.13` (Auto) / `13.34` (B5) to **`5.41 / 5.71`** |
| `08` (`0a1f1a60cc164359`) | Forest Foliage | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 | **0.00** | `0` neutral anchors $\to$ fallback preserves foliage ($C^* = 39.96$) |
| `09` (`d409e3e808cf45c8`) | Daylight Scene 09 | 4.58 | 4.02 | 2.01 | **2.21** | 11.45 | **11.12** | `Heldout_dE` cut from `4.58` to `2.21` |
| `10` (`0ac129dfd65b42f5`) | Square Format 10 | 6.89 | 2.87 | 2.65 | **2.30** | 2.53 | **1.16** | Selects `2B` (`lvl = 2`), `CrossSlope` `7.43 -> 3.42` |
| `11` (`9758c12cbae74d8d`) | Scene 11 | 4.56 | 4.21 | 4.00 | **4.38** | 2.78 | **5.36** | `CrossSlope` cut from `6.90` (Auto) / `4.27` (`2auto`) to **`1.63`** |
| `12` (`f9eedcf6c27b46bd`) | Red Car Auto Crop | 6.77 | 3.50 | 2.01 | **3.42** | 22.89 | **5.61** | `LowRC_Spread` cut from `22.89` to **`5.61`** |
| `13` (`231f90bc390146bb`) | Red Car Manual Crop | 8.13 | 3.69 | 2.10 | **3.87** | 27.11 | **7.62** | `LowRC_Spread` cut from `27.11` to **`7.62`** |
