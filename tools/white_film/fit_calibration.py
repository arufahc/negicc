#!/usr/bin/env python3
"""Fit the constants of the white-light film model (white_film.cc) to the IT8 calibration readings.

Model, per camera channel c:   d_c = log2(base_c / v_c)        stops below the film base
                               x_c = 2^(p_c d_c) - 1           exponent p; -1: the base's own share
                               XYZ = M x 2^e                   one matrix; e: one exposure per bracket (0 for the middle one)
Data: profiles/<Stock>-A7R4-D65/bundle.json holds the mean reading of each of the 288 IT8 patches for 7 exposure
brackets, in linear Rec.2020, with the matrix that produced them from the camera's channels (inverted here), and
the film base; --ref gives the chart's XYZ (D50). The fit is on Portra 400; the other stocks show what carries over
when only the exposures are theirs, and when the exponents and a balance w (x_c scaled by w_c) are theirs too.

  python3 tools/white_film/fit_calibration.py --ref R190808_ref.json [--profiles profiles]

Prints P_CAL and M as white_film.cc holds them, and the colour error (CIE76) per stock.
"""
import argparse, json
import numpy as np
from scipy.optimize import least_squares

WHITE_D50 = np.array([96.42, 100.0, 82.49])
STOCKS = ["Portra400-A7R4-D65", "Portra160-A7R4-D65", "Ektar100-A7R4-D65"]


def seconds(s):
    n, _, d = s.rstrip("s").partition("/")
    return float(n) / float(d) if d else float(n)


def load(profiles, stock, ref):
    """[(T [288,3] patch / base in camera channels, XYZ [288,3], is_grey [288]) per bracket]."""
    j = json.load(open(f"{profiles}/{stock}/bundle.json"))
    to_camera = np.linalg.inv(np.array(j["profiling_input_transform"]["matrix"])).T
    fb = j["film_base"]
    base = np.array([fb[c]["avg"] for c in "rgb"]) @ to_camera / seconds(fb["shutter"])
    out = []
    for t in j["targets"]:
        ids = sorted(t["patches"])
        v = np.array([[t["patches"][i][c] for c in "rgb"] for i in ids]) @ to_camera / seconds(t["shutter"])
        out.append((v / base, np.array([[ref[i][c] for c in "XYZ"] for i in ids]), np.array([i.startswith("gs") for i in ids])))
    return out


def lab(xyz):
    t = np.maximum(xyz, 1e-9) / WHITE_D50
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def model(th, T, bracket):
    """th = p (3), log2 w_G, log2 w_B (0 in the calibration fit), M (9), e (one per bracket)."""
    x = (2.0 ** (-np.log2(T) * th[:3]) - 1.0) * np.concatenate([[1.0], 2.0 ** th[3:5]])
    return x @ th[5:14].reshape(3, 3).T * 2.0 ** th[14 + bracket]


def refine(data, start, free):
    """Least squares on the Lab residuals over the parameters in `free`, the others kept at `start`."""
    mask = np.zeros(len(start), bool); mask[free] = True
    def residual(q):
        th = start.copy(); th[mask] = q
        return np.concatenate([(lab(model(th, T, b)) - lab(xyz)).ravel() for b, (T, xyz, _) in enumerate(data)])
    th = start.copy(); th[mask] = least_squares(residual, start[mask], loss="soft_l1", f_scale=3.0, xtol=1e-12, ftol=1e-12).x
    return th


def calibrate(data):
    """Exponents and exposures from the grey scale, M linearly, then everything refined (the middle exposure stays 0)."""
    n, mid = 14 + len(data), len(data) // 2
    def grey(q):
        return np.concatenate([(np.log2(np.maximum(2.0 ** (-np.log2(T[g]) * q[:3]) - 1.0, 1e-6)) + q[3:6] + q[6 + b]
                                - np.log2(xyz[g, 1:2])).ravel() for b, (T, xyz, g) in enumerate(data)])
    q = least_squares(grey, np.concatenate([[2.5, 2.0, 1.8], [0, 0, 0], 0.9 * (mid - np.arange(len(data)))]), loss="soft_l1", f_scale=0.3).x
    start = np.zeros(n)
    start[:3], start[14:] = q[:3], q[6:] - q[6 + mid]
    start[5:14] = np.eye(3).ravel()
    X = np.concatenate([model(start, T, b) for b, (T, _, _) in enumerate(data)])
    start[5:14] = np.linalg.lstsq(X, np.concatenate([xyz for _, xyz, _ in data]), rcond=None)[0].T.ravel()
    return refine(data, start, np.r_[0:3, 5:14, [14 + b for b in range(len(data)) if b != mid]])


def error(th, data):
    d = np.concatenate([np.linalg.norm(lab(model(th, T, b)) - lab(xyz), axis=1) for b, (T, xyz, _) in enumerate(data)])
    return f"dE76 median {np.median(d):.2f}, 90% {np.quantile(d, 0.9):.2f}, max {d.max():.1f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--profiles", default="profiles")
    a = ap.parse_args()
    ref = json.load(open(a.ref))["patches"]
    data = {s: load(a.profiles, s, ref) for s in STOCKS}
    th = calibrate(data[STOCKS[0]])
    np.set_printoptions(precision=5, suppress=True)
    print("P_CAL =", th[:3]); print("M =\n", th[5:14].reshape(3, 3))
    print(STOCKS[0], "(fitted):", error(th, data[STOCKS[0]]))
    for s in STOCKS[1:]:
        print(s, "with Portra 400's model and its own exposures:", error(refine(data[s], th, np.r_[14:21]), data[s]))
        print(s, "with its own exponents and balance as well:  ", error(refine(data[s], th, np.r_[0:5, 14:21]), data[s]))
