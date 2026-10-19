#!/usr/bin/env python3
# Copyright 2024 Alpha Lam <arufa.hc@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Exports the frozen DINOv3 backbone neg_process runs in its parameter search to ONNX, and optionally its TensorRT engine.

    make dinov3_model                 # or: python3 export_dinov3_backbone.py [--source ID_OR_DIR] [--engine]

The backbone is shared by every film stock; the per-stock intent adapters (models/dinov3-small_<stock>/adapter.onnx,
built into neg_process) turn its tokens into the intent field. Writes to models/dinov3-small/:

    backbone.onnx    inputs "pixel_values" [1, 3, H, W] (ImageNet-normalised, H and W multiples of 16 up to 512)
                     and "pool_weights" [1, N, 3]; outputs "tokens" [1, N, 768] (patch tokens of layers 9 and 12,
                     the adapter's input), "pooled" [1, 1152] and "pooled_tone" [1, 3, 768] (exposure ranker
                     features). The same computation as negicc-station's fused intent model up to its adapter.
    backbone.json    where the weights came from and the export settings.
    LICENSE*, NOTICE* the licence files of the weights' source, copied as they are.
    backbone.engine  with --engine: the TensorRT engine (device-specific, not checked in).

Needs torch, transformers (with DINOv3) and onnx.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import types

import torch

DEFAULT_SOURCE = "Tooony133/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dinov3-small")
IMAGE_SIZE = 512
PATCH_SIZE = 16
TONE_MASS_GATE = 0.03  # mass fraction at which a tone bin's pooled features reach full weight


def _rope_forward_for_export(self, pixel_values):
    """DINOv3ViTRopePositionEmbedding.forward (eval mode) with angles.tile(2) written as a concatenation: the ONNX
    exporter emits tile with a rank-check If node that TensorRT 10.3 cannot parse."""
    from transformers.models.dinov3_vit.modeling_dinov3_vit import get_patches_center_coordinates
    _, _, height, width = pixel_values.shape
    coords = get_patches_center_coordinates(height // self.config.patch_size, width // self.config.patch_size,
                                            dtype=torch.float32, device=pixel_values.device)
    angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
    angles = angles.flatten(1, 2)
    angles = torch.cat([angles, angles], dim=-1)
    return torch.cos(angles).to(pixel_values.dtype), torch.sin(angles).to(pixel_values.dtype)


class Backbone(torch.nn.Module):
    """pixel_values, pool_weights -> (tokens, pooled, pooled_tone)."""

    def __init__(self, dino):
        super().__init__()
        self.dino = dino
        rope = getattr(dino, "rope_embeddings", None)
        if rope is not None:
            rope.forward = types.MethodType(_rope_forward_for_export, rope)
        num_layers = dino.config.num_hidden_layers
        self.l_photo = int(round(num_layers * 0.75))
        self.l_sem = num_layers
        self.k_start = 1 + getattr(dino.config, "num_register_tokens", 0)

    def forward(self, pixel_values, pool_weights):
        out = self.dino(pixel_values=pixel_values, output_hidden_states=True)
        hp, hs = out.hidden_states[self.l_photo], out.hidden_states[self.l_sem]
        h_photo, h_sem = hp[:, self.k_start:, :], hs[:, self.k_start:, :]
        tokens = torch.cat([h_photo, h_sem], dim=-1)
        # Exposure ranker features: CLS of the last layer, mean patch tokens of both layers.
        pooled = torch.cat([hs[:, 0, :], h_photo.mean(dim=1), h_sem.mean(dim=1)], dim=-1)
        # Tone-stratified means, each faded in by its mass fraction.
        unnormalized = torch.bmm(pool_weights.transpose(1, 2), tokens)
        mass = pool_weights.sum(dim=1, keepdim=True).transpose(1, 2)
        mass_frac = pool_weights.mean(dim=1, keepdim=True).transpose(1, 2)
        gate = torch.clamp(mass_frac / TONE_MASS_GATE, 0.0, 1.0)
        pooled_tone = unnormalized / torch.clamp(mass, min=1e-6) * gate
        return tokens, pooled, pooled_tone


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def find_trtexec():
    for p in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"):
        if p and os.path.exists(p):
            return p
    return None


def build_engine(onnx_path, engine_path, workspace_mb, fp16):
    trtexec = find_trtexec()
    if not trtexec:
        sys.exit("trtexec not found (install TensorRT, e.g. JetPack's nvidia-tensorrt)")
    lo = 8 * PATCH_SIZE
    opt_h = IMAGE_SIZE - 2 * PATCH_SIZE
    n = lambda h, w: (h // PATCH_SIZE) * (w // PATCH_SIZE)
    cmd = [
        trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
        f"--minShapes=pixel_values:1x3x{lo}x{lo},pool_weights:1x{n(lo, lo)}x3",
        f"--optShapes=pixel_values:1x3x{opt_h}x{IMAGE_SIZE},pool_weights:1x{n(opt_h, IMAGE_SIZE)}x3",
        f"--maxShapes=pixel_values:1x3x{IMAGE_SIZE}x{IMAGE_SIZE},pool_weights:1x{n(IMAGE_SIZE, IMAGE_SIZE)}x3",
        f"--memPoolSize=workspace:{workspace_mb}M",
        "--fp16" if fp16 else "--noTF32",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Built {engine_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="Hugging Face model id or local model directory")
    ap.add_argument("--revision", default=None, help="model revision (Hugging Face sources)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--engine", action="store_true", help="also build the TensorRT engine (strict FP32)")
    ap.add_argument("--engine-only", action="store_true", help="build the engine from an existing backbone.onnx")
    ap.add_argument("--fp16", action="store_true", help="build an FP16 engine instead of strict FP32")
    ap.add_argument("--workspace-mb", type=int, default=512, help="TensorRT builder workspace limit (MiB)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, "backbone.onnx")
    if args.engine_only:
        build_engine(onnx_path, os.path.join(args.out, "backbone.engine"), args.workspace_mb, args.fp16)
        return

    from transformers import AutoModel
    if os.path.isdir(args.source):
        src_dir = args.source
    else:
        from huggingface_hub import snapshot_download
        src_dir = snapshot_download(args.source, revision=args.revision)
    dino = AutoModel.from_pretrained(src_dir).eval()
    net = Backbone(dino).eval()

    # Non-square example so both axes export as dynamic.
    h, w = IMAGE_SIZE - 2 * PATCH_SIZE, IMAGE_SIZE
    example = torch.zeros(1, 3, h, w)
    weights = torch.zeros(1, (h // PATCH_SIZE) * (w // PATCH_SIZE), 3)
    weights[:, :, 1] = 1.0
    with torch.no_grad():
        torch.onnx.export(
            net, (example, weights), onnx_path,
            input_names=["pixel_values", "pool_weights"],
            output_names=["tokens", "pooled", "pooled_tone"],
            dynamic_axes={"pixel_values": {2: "height", 3: "width"}, "pool_weights": {1: "patches"},
                          "tokens": {1: "patches"}},
            opset_version=17,
            dynamo=False,
        )
    print(f"Exported {onnx_path}")

    licences = []
    for pattern in ("LICENSE*", "LICENCE*", "NOTICE*", "README.md"):
        for f in sorted(glob.glob(os.path.join(src_dir, pattern))):
            name = os.path.basename(f) if not f.endswith("README.md") else "SOURCE_README.md"
            shutil.copyfile(f, os.path.join(args.out, name))
            licences.append(name)
    if not any(n.upper().startswith(("LICENSE", "LICENCE")) for n in licences):
        print("WARNING: the source has no LICENSE file; add the DINOv3 licence before distributing backbone.onnx")

    meta = {
        "source": args.source if not os.path.isdir(args.source) else os.path.basename(os.path.normpath(args.source)),
        "revision": args.revision,
        "architecture": dino.config.model_type,
        "num_hidden_layers": dino.config.num_hidden_layers,
        "hidden_size": dino.config.hidden_size,
        "patch_size": PATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "layers": [net.l_photo, net.l_sem],
        "token_start_idx": net.k_start,
        "tone_mass_gate": TONE_MASS_GATE,
        "inputs": {"pixel_values": "[1, 3, H, W] ImageNet-normalised", "pool_weights": "[1, N, 3]"},
        "outputs": {"tokens": "[1, N, 768]", "pooled": "[1, 1152]", "pooled_tone": "[1, 3, 768]"},
        "onnx_sha256": sha256_file(onnx_path),
        "torch": torch.__version__,
        "licence_files": licences,
    }
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except ImportError:
        pass
    with open(os.path.join(args.out, "backbone.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.join(args.out, 'backbone.json')}")

    if args.engine:
        build_engine(onnx_path, os.path.join(args.out, "backbone.engine"), args.workspace_mb, args.fp16)


if __name__ == "__main__":
    main()
