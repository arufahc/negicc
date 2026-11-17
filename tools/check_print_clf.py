#!/usr/bin/env python3
"""The print's two paths against each other (DESIGN.md, "Direct sRGB output: the print ICC", Checks): a TriLED scan
converted twice by neg_process, as a print picture (-P srgb --print <material>) and as a density file with the print's
CLF (-P apd --print <material>); the CLF is applied here to the density file and the result compared with the picture,
CIEDE2000 per pixel at 1000 px on the long side. Passes when the mean is within --mean and the 95th percentile within
--p95. Needs numpy and opencv-python.

    python3 tools/check_print_clf.py --neg-process bin_out/neg_process --print kodak_portra_endura \\
        -- --film_stock portra400 --light_source tri --roi auto scan.ARW

The CLF reader covers the nodes neg_process writes: Matrix (3x3 or 3x4), Range, LUT1D (linear), LUT3D (tetrahedral)
and Exponent (monCurveRev)."""
import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def nums(node):
    return np.array(node.find("Array").text.split(), float)


def apply_clf(path, x):
    """x: (..., 3) in the CLF's input units; returns its output."""
    ns = ""
    root = ET.parse(path).getroot()
    if root.tag.startswith("{"):
        ns = root.tag[:root.tag.index("}") + 1]
    for node in root:
        tag = node.tag[len(ns):]
        if tag == "Matrix":
            a = node.find(ns + "Array")
            dim = [int(v) for v in a.get("dim").split()]
            v = np.array(a.text.split(), float).reshape(dim[0], dim[1])
            x = x @ v[:, :3].T + (v[:, 3] if dim[1] == 4 else 0.0)
        elif tag == "Range":
            g = lambda k: float(node.find(ns + k).text)
            lo, hi, olo, ohi = g("minInValue"), g("maxInValue"), g("minOutValue"), g("maxOutValue")
            x = np.clip((x - lo) / (hi - lo) * (ohi - olo) + olo, olo, ohi)
        elif tag == "LUT1D":
            a = node.find(ns + "Array")
            n = int(a.get("dim").split()[0])
            t = np.array(a.text.split(), float).reshape(n, 3)
            grid = np.linspace(0.0, 1.0, n)
            x = np.stack([np.interp(x[..., c], grid, t[:, c]) for c in range(3)], -1)
        elif tag == "LUT3D":
            a = node.find(ns + "Array")
            n = int(a.get("dim").split()[0])
            t = np.array(a.text.split(), float).reshape(n, n, n, 3)    # red slowest, blue fastest
            x = tetrahedral(t, np.clip(x, 0.0, 1.0))
        elif tag == "Exponent":
            p = node.find(ns + "ExponentParams")
            g, off = float(p.get("exponent")), float(p.get("offset"))
            if node.get("style") != "monCurveRev" or abs(g - 2.4) > 1e-9 or abs(off - 0.055) > 1e-9:
                raise SystemExit("unsupported Exponent (neg_process writes the sRGB encoding: monCurveRev 2.4, 0.055)")
            x = np.where(x <= 0.0031308, 12.92 * x, (1 + off) * np.power(np.maximum(x, 0.0), 1.0 / g) - off)
        elif tag in ("Description", "InputDescriptor", "OutputDescriptor", "Info"):
            continue
        else:
            raise SystemExit("unsupported CLF node " + tag)
    return x


def tetrahedral(t, x):
    n = t.shape[0]
    p = x * (n - 1)
    i = np.minimum(np.floor(p).astype(int), n - 2)
    f = p - i
    r, g, b = i[..., 0], i[..., 1], i[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c = lambda dr, dg, db: t[r + dr, g + dg, b + db]
    c000, c111 = c(0, 0, 0), c(1, 1, 1)
    out = np.zeros(x.shape)
    m1 = (fr >= fg) & (fg >= fb)
    m2 = (fr >= fb) & (fb > fg)
    m3 = (fb > fr) & (fr >= fg)
    m4 = (fg > fr) & (fr >= fb)
    m5 = (fg >= fb) & (fb > fr)
    m6 = ~(m1 | m2 | m3 | m4 | m5)
    out = np.where(m1, c000 + fr * (c(1, 0, 0) - c000) + fg * (c(1, 1, 0) - c(1, 0, 0)) + fb * (c111 - c(1, 1, 0)), out)
    out = np.where(m2, c000 + fr * (c(1, 0, 0) - c000) + fb * (c(1, 0, 1) - c(1, 0, 0)) + fg * (c111 - c(1, 0, 1)), out)
    out = np.where(m3, c000 + fb * (c(0, 0, 1) - c000) + fr * (c(1, 0, 1) - c(0, 0, 1)) + fg * (c111 - c(1, 0, 1)), out)
    out = np.where(m4, c000 + fg * (c(0, 1, 0) - c000) + fr * (c(1, 1, 0) - c(0, 1, 0)) + fb * (c111 - c(1, 1, 0)), out)
    out = np.where(m5, c000 + fg * (c(0, 1, 0) - c000) + fb * (c(0, 1, 1) - c(0, 1, 0)) + fr * (c111 - c(0, 1, 1)), out)
    out = np.where(m6, c000 + fb * (c(0, 0, 1) - c000) + fg * (c(0, 1, 1) - c(0, 0, 1)) + fr * (c111 - c(0, 1, 1)), out)
    return out


def lab(srgb):
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(a, b):
    L1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    L2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0, np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2))
    T = 1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp)) + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def small(path, side=1000):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].astype(np.float64) / 65535.0
    k = side / max(img.shape[:2])
    return cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-process", required=True)
    ap.add_argument("--print", dest="material", action="append", required=True)
    ap.add_argument("--mean", type=float, default=0.5)
    ap.add_argument("--p95", type=float, default=1.0)
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.args[1:] if a.args and a.args[0] == "--" else a.args
    ok = True
    with tempfile.TemporaryDirectory() as t:
        dens = os.path.join(t, "d.tif")
        cmd = [a.neg_process, *args[:-1], "-P", "apd", *sum((["--print", m] for m in a.material), []), "-o", dens, args[-1]]
        if subprocess.run(cmd, capture_output=True).returncode:
            raise SystemExit("the density run failed: " + " ".join(cmd))
        x = small(dens)
        for m in a.material:
            pic = os.path.join(t, f"p_{m}.tif")
            cmd = [a.neg_process, *args[:-1], "-P", "srgb", "--print", m, "-o", pic, args[-1]]
            if subprocess.run(cmd, capture_output=True).returncode:
                raise SystemExit("the picture run failed: " + " ".join(cmd))
            out = np.clip(apply_clf(f"{dens}.{m}.clf", x), 0.0, 1.0)
            de = ciede2000(lab(out), lab(small(pic)))
            mean, p95 = float(de.mean()), float(np.percentile(de, 95))
            good = mean <= a.mean and p95 <= a.p95
            ok &= good
            print(f"{m}: picture against the CLF on the density file, CIEDE2000 mean {mean:.3f}, 95th percentile "
                  f"{p95:.3f} ({'within' if good else 'OUTSIDE'} {a.mean} / {a.p95})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
