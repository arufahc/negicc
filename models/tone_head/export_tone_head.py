#!/usr/bin/env python3
"""Exports the white-light tone heads (tone_head.pt) to tone_head.bin, the flat file neg_process embeds.

tone_head.pt is a dict {kind: "pool", state, xm, xs, ym, ys, tgt, pool}: state is the state_dict of
Sequential(Linear(1920, 512), GELU, Dropout(0.2), Linear(512, 256), GELU, Linear(256, 22)); xm, xs standardise the
1920 pooled features (float32), ym, ys the 11 targets tgt. The first 11 outputs are the targets' standardised means,
the last 11 their log-variances (clamped to [-6, 4]): mean = out[:11] * ys + ym, sd = exp(out[11:] / 2) * ys.

tone_head.bin, little-endian:
  char[4]  magic "TNH1"
  uint32   in_dim (1920), hidden1 (512), hidden2 (256), targets (11)
  float32  xm[in_dim], xs[in_dim]
  float32  w1[hidden1][in_dim], b1[hidden1]          (PyTorch's [out][in] layout)
  float32  w2[hidden2][hidden1], b2[hidden2]
  float32  w3[2 * targets][hidden2], b3[2 * targets]
  float32  ym[targets], ys[targets]
The targets are, in this order: L2 L5 L10 L50 L90 L95 L98 IQR R9010 Cmean C90 (ToneHead in intent_head.h).

usage: export_tone_head.py [tone_head.pt] [tone_head.bin] [--check]
--check also runs the heads on random features with PyTorch and with numpy reading the .bin back, and prints the
largest difference of the predicted means and standard deviations.
"""
import os
import struct
import sys

import numpy as np

TARGETS = ["L2", "L5", "L10", "L50", "L90", "L95", "L98", "IQR", "R9010", "Cmean", "C90"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    here = os.path.dirname(os.path.abspath(__file__))
    src = args[0] if args else os.path.join(here, "tone_head.pt")
    dst = args[1] if len(args) > 1 else os.path.join(here, "tone_head.bin")
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    assert ck.get("kind") == "pool" and list(ck["tgt"]) == TARGETS, "not the pooled-feature tone heads"
    st = {k: v.detach().cpu().numpy().astype("<f4") for k, v in ck["state"].items()}
    w1, b1 = st["net.0.weight"], st["net.0.bias"]
    w2, b2 = st["net.3.weight"], st["net.3.bias"]
    w3, b3 = st["net.5.weight"], st["net.5.bias"]
    h1, in_dim = w1.shape
    h2 = w2.shape[0]
    n_out = len(TARGETS)
    xm, xs = (np.asarray(ck[k], dtype="<f4") for k in ("xm", "xs"))
    ym, ys = (np.asarray(ck[k], dtype="<f4") for k in ("ym", "ys"))
    assert w2.shape == (h2, h1) and w3.shape == (2 * n_out, h2) and xm.shape == (in_dim,) and ym.shape == (n_out,)
    with open(dst, "wb") as f:
        f.write(b"TNH1")
        f.write(struct.pack("<IIII", in_dim, h1, h2, n_out))
        for a in (xm, xs, w1, b1, w2, b2, w3, b3, ym, ys):
            f.write(np.ascontiguousarray(a).tobytes())
    print(f"wrote {dst}: in_dim {in_dim}, hidden {h1}, {h2}, targets {n_out}, {os.path.getsize(dst)} bytes")

    if "--check" in sys.argv:
        x = (np.random.default_rng(0).normal(0.0, 1.5, (512, in_dim)) * xs + xm).astype(np.float32)
        net = torch.nn.Sequential(torch.nn.Linear(in_dim, h1), torch.nn.GELU(), torch.nn.Dropout(0.2),
                                  torch.nn.Linear(h1, h2), torch.nn.GELU(), torch.nn.Linear(h2, 2 * n_out))
        net.load_state_dict({k[len("net."):]: v for k, v in ck["state"].items()})
        net.eval()
        with torch.no_grad():
            o = net((torch.from_numpy(x) - torch.from_numpy(xm)) / torch.from_numpy(xs)).numpy().astype(np.float64)
        m_t = o[:, :n_out] * ys + ym
        s_t = np.exp(0.5 * np.clip(o[:, n_out:], -6, 4)) * ys
        raw = open(dst, "rb").read()
        d, a1, a2, n = struct.unpack_from("<IIII", raw, 4)
        v = np.frombuffer(raw, dtype="<f4", offset=20).astype(np.float64)
        sizes = [d, d, a1 * d, a1, a2 * a1, a2, 2 * n * a2, 2 * n, n, n]
        parts, o_ = [], 0
        for k in sizes:
            parts.append(v[o_:o_ + k])
            o_ += k
        XM, XS, W1, B1, W2, B2, W3, B3, YM, YS = parts
        from math import erf, sqrt
        gelu = np.vectorize(lambda t: 0.5 * t * (1.0 + erf(t / sqrt(2.0))))
        h = gelu(((x - XM) / XS) @ W1.reshape(a1, d).T + B1)
        h = gelu(h @ W2.reshape(a2, a1).T + B2)
        out = h @ W3.reshape(2 * n, a2).T + B3
        m_n = out[:, :n] * YS + YM
        s_n = np.exp(0.5 * np.clip(out[:, n:], -6, 4)) * YS
        print("max |mean_torch - mean_bin|:", float(np.abs(m_t - m_n).max()),
              " max |sd_torch - sd_bin|:", float(np.abs(s_t - s_n).max()))


if __name__ == "__main__":
    main()
