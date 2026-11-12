#!/usr/bin/env python3
"""Exports the neutral-patch head (neutral_head.pt) to neutral_head.bin, the flat file neg_process embeds.

neutral_head.pt is a dict {features: "tok", kind: "mlp", state, mean, std}: state is the state_dict of
Sequential(Linear(768, 256), GELU, Dropout(0.1), Linear(256, 1)) and mean, std the per-dimension standardisation of
the tokens (float32 [768]). p = sigmoid(net((tokens - mean) / std)).

neutral_head.bin, little-endian:
  char[4]  magic "NPH1"
  uint32   in_dim (768), hidden (256)
  float32  mean[in_dim], std[in_dim]
  float32  w1[hidden][in_dim], b1[hidden]   (Linear(768, 256), PyTorch's [out][in] layout)
  float32  w2[hidden], b2                   (Linear(256, 1))

usage: export_neutral_head.py [neutral_head.pt] [neutral_head.bin] [--check]
--check also runs the head on random tokens with PyTorch and with numpy reading the .bin back, and prints the
largest difference.
"""
import os
import struct
import sys

import numpy as np


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    here = os.path.dirname(os.path.abspath(__file__))
    src = args[0] if args else os.path.join(here, "neutral_head.pt")
    dst = args[1] if len(args) > 1 else os.path.join(here, "neutral_head.bin")
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    assert ck.get("kind") == "mlp" and ck.get("features") == "tok", "not a token MLP neutral head"
    st = {k: v.detach().cpu().numpy().astype("<f4") for k, v in ck["state"].items()}
    w1, b1, w2, b2 = st["net.0.weight"], st["net.0.bias"], st["net.3.weight"], st["net.3.bias"]
    hidden, in_dim = w1.shape
    mean = np.asarray(ck["mean"], dtype="<f4")
    std = np.asarray(ck["std"], dtype="<f4")
    assert mean.shape == (in_dim,) and std.shape == (in_dim,) and w2.shape == (1, hidden) and b2.shape == (1,)
    with open(dst, "wb") as f:
        f.write(b"NPH1")
        f.write(struct.pack("<II", in_dim, hidden))
        for a in (mean, std, w1, b1, w2.reshape(-1), b2):
            f.write(np.ascontiguousarray(a).tobytes())
    print(f"wrote {dst}: in_dim {in_dim}, hidden {hidden}, {os.path.getsize(dst)} bytes")

    if "--check" in sys.argv:
        x = np.random.default_rng(0).normal(0.0, 2.0, (4096, in_dim)).astype(np.float32) * std + mean
        net = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(hidden, 1))
        net.load_state_dict({k[len("net."):]: v for k, v in ck["state"].items()})
        net.eval()
        with torch.no_grad():
            p_t = torch.sigmoid(net((torch.from_numpy(x) - torch.from_numpy(mean)) / torch.from_numpy(std)))[:, 0]
        raw = open(dst, "rb").read()
        d_in, d_h = struct.unpack_from("<II", raw, 4)
        v = np.frombuffer(raw, dtype="<f4", offset=12)
        o = 0
        m_, s_ = v[o:o + d_in], v[o + d_in:o + 2 * d_in]
        o += 2 * d_in
        W1 = v[o:o + d_h * d_in].reshape(d_h, d_in)
        o += d_h * d_in
        B1, W2, B2 = v[o:o + d_h], v[o + d_h:o + 2 * d_h], v[o + 2 * d_h]
        from math import erf, sqrt
        h = ((x - m_) / s_).astype(np.float64) @ W1.T.astype(np.float64) + B1
        h = 0.5 * h * (1.0 + np.vectorize(lambda t: erf(t / sqrt(2.0)))(h))
        p_n = 1.0 / (1.0 + np.exp(-(h @ W2 + B2)))
        print("max |p_torch - p_bin| on random tokens:", float(np.abs(p_t.numpy() - p_n).max()))


if __name__ == "__main__":
    main()
