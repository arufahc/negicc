# Measurements

The numbers behind the claims in the [README](README.md).

## Demosaicing

Single-shot TriLED scans on a Sony RGGB sensor, against a 4-shot pixel-shift capture of the same frame.

| Algorithm | Raw MAE ↓ | Positive PSNR (dB) ↑ | Positive ΔE₀₀ mean / p95 ↓ | R−G zipper ↓ |
| :--- | :---: | :---: | :---: | :---: |
| 4-shot pixel shift (ground truth) | 0.00 | ∞ | 0.00 / 0.00 | 142.4 |
| **Grain-aware colour-ratio** (TriLED default) | **80.81** | **40.38** | **1.69 / 6.86** | 56.5 |
| LibRaw DCB (`-q 4`) | 79.60 | 39.85 | 1.76 / 7.21 | 71.7 |
| LibRaw VNG (`-q 1`) | 80.19 | 39.96 | 1.75 / 7.12 | 58.6 |
| PPG / AHD (`-q 2` / `-q 3`) | 80.61 | 40.03 | 1.73 / 6.97 | 55.3 (grain maze) |
| Bilinear (`-q 0`) | 89.15 | 39.16 | 1.90 / 7.87 | 115.0 |

Raw MAE is in 14-bit LSB; PSNR and ΔE₀₀ are measured on the inverted positive.

## Profile accuracy

The TriLED bundles, per exposure bracket.

| Bracket | Portra 400 own (median) | held-out | Portra 160 own | held-out | Ektar 100 own | held-out |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| −3 EV | 0.79 (0.50) | 2.70 (1.91) | 1.09 (0.47) | 2.35 (1.61) | 1.11 (0.49) | 3.05 (1.87) |
| −2 EV | 0.56 (0.41) | 2.04 (1.38) | 0.82 (0.42) | 2.09 (1.55) | 0.61 (0.39) | 2.68 (1.52) |
| −1 EV | 0.53 (0.40) | 2.38 (1.40) | 0.59 (0.39) | 1.96 (1.40) | 0.51 (0.36) | 2.79 (1.55) |
| 0 EV | 0.55 (0.42) | 2.62 (1.48) | 0.56 (0.44) | 2.30 (1.37) | 0.62 (0.49) | 3.25 (1.84) |
| +1 EV | 0.54 (0.40) | 2.75 (1.46) | 0.62 (0.43) | 2.65 (1.58) | 0.63 (0.49) | 3.52 (1.75) |
| +2 EV | 0.59 (0.43) | 2.92 (1.58) | 0.67 (0.49) | 2.67 (1.62) | 0.72 (0.59) | 4.01 (2.15) |
| +3 EV | 0.69 (0.54) | 3.08 (1.77) | 0.70 (0.53) | 2.82 (1.78) | 0.83 (0.65) | 3.72 (2.26) |

Mean ΔE₀₀ (median in brackets), from ArgyllCMS `profcheck` on all 288 patches (own) and from the cross-validation
(held-out). The D65 bundles are at 0.6–1.1 own mean from −1 to +3 EV and 1.3–2.2 held-out median. No profile is
de-glared: removing the chart's veiling glare before fitting was tried, did not help overall, and the bundles ship
as measured.
