#!/usr/bin/env python3
"""Self-contained CLI and regression tests for neg_process and check_profile_bundles.py."""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile

import check_profile_bundles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NEG_PROCESS = os.path.join(ROOT, "bin_out", "neg_process")


def write_synthetic_dng(path, width, height, pixels_u16, make="Sony", model="ILCE-7RM4",
                        cfa_dim=(2, 2), cfa_pattern=(0, 1, 1, 2), black=512, white=16383):
    """Writes a minimal valid uncompressed 16-bit CFA DNG readable by LibRaw."""
    assert len(pixels_u16) == width * height
    pixel_bytes = struct.pack(f"<{width * height}H", *pixels_u16)

    entries = []
    extra = bytearray()
    # Reserve space for header (8 bytes) + IFD count (2) + 20 entries * 12 + next IFD (4) = 254 -> align to 256
    extra_base = 256

    def add_bytes(data):
        if len(data) % 2 == 1:
            data = data + b"\x00"
        off = extra_base + len(extra)
        extra.extend(data)
        return off

    def add_tag(tag, typ, count, val_bytes):
        if len(val_bytes) <= 4:
            val_field = val_bytes.ljust(4, b"\x00")
        else:
            off = add_bytes(val_bytes)
            val_field = struct.pack("<I", off)
        entries.append((tag, typ, count, val_field))

    # 254 NewSubfileType = 0 (full resolution)
    add_tag(254, 4, 1, struct.pack("<I", 0))
    # 256 ImageWidth
    add_tag(256, 4, 1, struct.pack("<I", width))
    # 257 ImageLength
    add_tag(257, 4, 1, struct.pack("<I", height))
    # 258 BitsPerSample = 16
    add_tag(258, 3, 1, struct.pack("<H", 16))
    # 259 Compression = 1 (uncompressed)
    add_tag(259, 3, 1, struct.pack("<H", 1))
    # 262 PhotometricInterpretation = 32803 (CFA)
    add_tag(262, 3, 1, struct.pack("<H", 32803))
    # 271 Make
    make_b = make.encode("ascii") + b"\x00"
    add_tag(271, 2, len(make_b), make_b)
    # 272 Model
    model_b = model.encode("ascii") + b"\x00"
    add_tag(272, 2, len(model_b), model_b)
    # 277 SamplesPerPixel = 1
    add_tag(277, 3, 1, struct.pack("<H", 1))
    # 278 RowsPerStrip = height
    add_tag(278, 4, 1, struct.pack("<I", height))
    # 279 StripByteCounts
    add_tag(279, 4, 1, struct.pack("<I", len(pixel_bytes)))
    # 284 PlanarConfiguration = 1
    add_tag(284, 3, 1, struct.pack("<H", 1))
    # 33421 CFARepeatPatternDim
    add_tag(33421, 3, 2, struct.pack("<2H", *cfa_dim))
    # 33422 CFAPattern
    add_tag(33422, 1, len(cfa_pattern), bytes(cfa_pattern))
    # 50706 DNGVersion = 1.4.0.0
    add_tag(50706, 1, 4, bytes([1, 4, 0, 0]))
    # 50708 UniqueCameraModel
    ucm_b = f"{make} {model}".encode("ascii") + b"\x00"
    add_tag(50708, 2, len(ucm_b), ucm_b)
    # 50714 BlackLevel
    add_tag(50714, 4, 1, struct.pack("<I", black))
    # 50717 WhiteLevel
    add_tag(50717, 4, 1, struct.pack("<I", white))
    # 50721 ColorMatrix1 (9 SRATIONAL)
    cm = [10000, 10000, 0, 10000, 0, 10000,
          0, 10000, 10000, 10000, 0, 10000,
          0, 10000, 0, 10000, 10000, 10000]
    add_tag(50721, 10, 9, struct.pack("<18i", *cm))

    strip_offset = extra_base + len(extra)
    # 273 StripOffsets
    add_tag(273, 4, 1, struct.pack("<I", strip_offset))

    entries.sort(key=lambda e: e[0])
    ifd = bytearray()
    ifd.extend(struct.pack("<H", len(entries)))
    for tag, typ, count, val_field in entries:
        ifd.extend(struct.pack("<HHI", tag, typ, count) + val_field)
    ifd.extend(struct.pack("<I", 0))

    header = struct.pack("<2sHI", b"II", 42, 8)
    assert len(header) + len(ifd) <= extra_base
    pad = b"\x00" * (extra_base - (len(header) + len(ifd)))
    with open(path, "wb") as f:
        f.write(header)
        f.write(ifd)
        f.write(pad)
        f.write(extra)
        f.write(pixel_bytes)


def make_picture_only_cfa(width, height, black=512):
    """Creates a synthetic Bayer mosaic with smooth mid-tone picture values and no bare light or film base."""
    pixels = []
    for y in range(height):
        for x in range(width):
            # Values between 1800 and 5200 (nowhere near white=16383)
            v = black + 1500 + ((x * 17 + y * 31) % 3000)
            pixels.append(v)
    return pixels


def make_framed_strip_cfa(width, height, black=512, white=16383):
    """Creates a horizontal 35mm strip on clipped bare light with two inter-frame gaps and a central picture frame."""
    pixels = []
    # Strip occupies y in [height*0.18, height*0.82]
    y0 = int(height * 0.18)
    y1 = int(height * 0.82)
    # Frame occupies x in [width*0.18, width*0.82], gaps at [width*0.10, width*0.18] and [width*0.82, width*0.90]
    g0_x0, f_x0, f_x1, g1_x1 = int(width * 0.08), int(width * 0.18), int(width * 0.82), int(width * 0.92)
    for y in range(height):
        for x in range(width):
            c = (0 if (x & 1) == 0 else 1) if (y & 1) == 0 else (1 if (x & 1) == 0 else 2)
            base_c = (9000, 4800, 2600)[c]
            noise = ((x * 131 + y * 197 + c * 73) % 41) - 20
            if y < y0 or y >= y1:
                v = white
            elif f_x0 <= x < f_x1 and (y0 + 12) <= y < (y1 - 12):
                # Picture content darker than film base with texture
                mod = 0.25 + 0.45 * (0.5 + 0.5 * math.sin(0.09 * x) * math.cos(0.09 * y))
                v = black + int(base_c * mod) + noise
            elif g0_x0 <= x < g1_x1:
                # Film base (rebate / inter-frame gap)
                v = black + base_c + noise
            else:
                v = white
            pixels.append(max(0, min(65535, v)))
    return pixels


def test_check_profile_bundles_shutter():
    for good in ("1/8s", "1/8", "0.4s", "2", "  1/125s "):
        assert check_profile_bundles.shutter_to_seconds(good) > 0.0
    for bad in ("", "   ", "s", "0", "0s", "-1/8", "1/-8", "1/0", "0/8", "nan", "inf", "-inf", "abc"):
        threw = False
        try:
            check_profile_bundles.shutter_to_seconds(bad)
        except ValueError:
            threw = True
        assert threw, f"Expected ValueError for {bad!r}"

    with tempfile.TemporaryDirectory() as tmp:
        bad_bundle = os.path.join(tmp, "bad_bundle.json")
        with open(bad_bundle, "w") as f:
            json.dump({
                "input_space": "camera_raw_rgb",
                "film_base": {"shutter": "0"},
                "targets": [{"name": "Target 1", "shutter": "1/8s"}],
            }, f)
        probs = check_profile_bundles.check(bad_bundle)
        assert any("film_base: cannot read shutter" in p for p in probs), probs


def test_cli_validation_and_error_handling():
    with tempfile.TemporaryDirectory() as tmp:
        out_tif = os.path.join(tmp, "out.tif")
        dng = os.path.join(tmp, "scan.dng")
        write_synthetic_dng(dng, 160, 120, make_picture_only_cfa(160, 120))

        # 1. -H / --half_size is rejected
        res = subprocess.run([NEG_PROCESS, "-H", "-o", out_tif, dng], capture_output=True, text=True)
        assert res.returncode != 0

        # 2. --dino_iters <= 0 is rejected
        for bad_iters in ("0", "-3"):
            res = subprocess.run([NEG_PROCESS, "--dino_iters", bad_iters, "-o", out_tif, dng],
                                 capture_output=True, text=True)
            assert res.returncode != 0
            assert "--dino_iters must be positive" in res.stderr

        # 3. Negative or zero --roi is rejected before opening RAW or profile
        missing_raw = os.path.join(tmp, "does_not_exist.ARW")
        for bad_roi in (["-1", "0", "10", "10"], ["0", "-2", "10", "10"], ["0", "0", "0", "10"], ["0", "0", "10", "-5"]):
            res = subprocess.run([NEG_PROCESS, "--roi", *bad_roi, "-o", out_tif, missing_raw],
                                 capture_output=True, text=True)
            assert res.returncode != 0
            assert "Invalid --roi" in res.stderr

        # 4. Corrupt JSON bundle exits cleanly with code 1 (not uncaught exception / SIGABRT)
        corrupt_json = os.path.join(tmp, "corrupt.json")
        with open(corrupt_json, "w") as f:
            f.write("{corrupt json")
        res = subprocess.run([NEG_PROCESS, "-p", corrupt_json, "-o", out_tif, dng],
                             capture_output=True, text=True)
        assert res.returncode == 1
        assert "ERROR!" in res.stderr

        # 5. Directory named foo.icc is rejected cleanly with code 1
        dir_icc = os.path.join(tmp, "foo.icc")
        os.makedirs(dir_icc)
        res = subprocess.run([NEG_PROCESS, "-p", dir_icc, "-o", out_tif, dng],
                             capture_output=True, text=True)
        assert res.returncode == 1
        assert "ERROR!" in res.stderr

        # 6. Corrupt --dino adapter file exits cleanly with code 1
        corrupt_onnx = os.path.join(tmp, "bad_adapter.onnx")
        with open(corrupt_onnx, "wb") as f:
            f.write(b"not an onnx model")
        res = subprocess.run([NEG_PROCESS, "--film_stock", "portra400", "--light_source", "tri",
                              "--dino", corrupt_onnx, "-o", out_tif, dng],
                             capture_output=True, text=True)
        assert res.returncode == 1
        assert "ERROR!" in res.stderr


def test_pixel_shift_rejection():
    with tempfile.TemporaryDirectory() as tmp:
        out_tif = os.path.join(tmp, "out.tif")
        f0 = os.path.join(tmp, "ps_0.dng")
        f1 = os.path.join(tmp, "ps_1.dng")
        f2 = os.path.join(tmp, "ps_2.dng")
        f3 = os.path.join(tmp, "ps_3.dng")
        pix = make_picture_only_cfa(160, 120)
        for p in (f0, f1, f2, f3):
            write_synthetic_dng(p, 160, 120, pix)

        # Duplicate file path rejected
        res = subprocess.run([NEG_PROCESS, "-o", out_tif, f0, f0, f2, f3], capture_output=True, text=True)
        assert res.returncode == 1
        assert "4 distinct RAW files" in res.stderr

        # Out-of-order frame indices rejected
        res = subprocess.run([NEG_PROCESS, "-o", out_tif, f0, f2, f1, f3], capture_output=True, text=True)
        assert res.returncode == 1
        assert "out of frame order" in res.stderr

        # A file counter that wraps (DSC09999 -> DSC00001) is in order: the check passes on to the camera check
        w = [os.path.join(tmp, n) for n in ("DSC09998.dng", "DSC09999.dng", "DSC00001.dng", "DSC00002.dng")]
        for p in w:
            write_synthetic_dng(p, 160, 120, pix, make="Fujifilm", model="X-T3")
        res = subprocess.run([NEG_PROCESS, "-o", out_tif] + w, capture_output=True, text=True)
        assert res.returncode == 1
        assert "out of frame order" not in res.stderr, res.stderr
        assert "only available with Sony A7RM4" in res.stderr, res.stderr
        res = subprocess.run([NEG_PROCESS, "-o", out_tif, w[0], w[2], w[1], w[3]], capture_output=True, text=True)
        assert res.returncode == 1
        assert "out of frame order" in res.stderr

        # Mismatched geometry in frame 2 rejected
        write_synthetic_dng(f2, 164, 120, make_picture_only_cfa(164, 120))
        res = subprocess.run([NEG_PROCESS, "-o", out_tif, f0, f1, f2, f3], capture_output=True, text=True)
        assert res.returncode == 1
        assert "matching Sony A7RM4 frames" in res.stderr

        # Non-A7RM4 camera rejected
        for p in (f0, f1, f2, f3):
            write_synthetic_dng(p, 160, 120, pix, make="Fujifilm", model="X-T3")
        res = subprocess.run([NEG_PROCESS, "-o", out_tif, f0, f1, f2, f3], capture_output=True, text=True)
        assert res.returncode == 1
        assert "only available with Sony A7RM4" in res.stderr


def test_manual_target_and_gains_uses_bracket_e_center():
    """Regression test for 3.3: --target N --gain_g X --gain_b Y without -E uses the bracket's e_center."""
    with tempfile.TemporaryDirectory() as tmp:
        dng = os.path.join(tmp, "scan.dng")
        write_synthetic_dng(dng, 160, 120, make_picture_only_cfa(160, 120))
        out_tif = os.path.join(tmp, "out.tif")
        res = subprocess.run([
            NEG_PROCESS,
            "--film_stock", "portra400",
            "--light_source", "tri",
            "--target", "2",
            "--gain_g", "1.05",
            "--gain_b", "0.95",
            "-P", "srgb",
            "-o", out_tif,
            dng,
        ], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        with open(out_tif + ".json") as f:
            meta = json.load(f)
        assert meta["matrix"]["pins"] == {"e": False, "g": True, "b": True}
        # e_center is computed from the bracket's patch enclosure, not left at the default 1.0
        assert abs(meta["matrix"]["exposure_comp"] - 1.0) > 1e-3, meta["matrix"]


def test_roi_auto_picture_only_fallback_and_cli_equivalent_parity():
    """Tests 3.7 (picture-only fallback to full frame) and 3.9 (--roi auto vs cli_equivalent bit parity)."""
    with tempfile.TemporaryDirectory() as tmp:
        # 1. Picture-only frame falls back to full frame with roi_source == "auto_fallback" and low_confidence == True
        pic_dng = os.path.join(tmp, "pic_only.dng")
        write_synthetic_dng(pic_dng, 240, 160, make_picture_only_cfa(240, 160))
        pic_tif = os.path.join(tmp, "pic_only.tif")
        res = subprocess.run([
            NEG_PROCESS, "-P", "srgb", "--roi", "auto", "-o", pic_tif, pic_dng,
        ], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "falling back to the full sensor frame" in res.stdout
        with open(pic_tif + ".json") as f:
            meta = json.load(f)
        assert meta["geometry"]["roi_source"] == "auto_fallback"
        assert meta["geometry"]["roi"] == [0, 0, 240, 160]
        assert meta["frame_detection"]["low_confidence"] is True
        assert meta["frame_detection"]["confidence"] < 0.50

        # 2. Framed strip with rebate on both CFA path (default) and LibRaw path (-q 0):
        #    --roi auto --film_base_rgb auto matches cli_equivalent bit-for-bit across the ENTIRE output TIFF.
        strip_dng = os.path.join(tmp, "strip.dng")
        write_synthetic_dng(strip_dng, 360, 240, make_framed_strip_cfa(360, 240))
        for tag, extra in (("cfa", []), ("libraw", ["-q", "0"])):
            auto_tif = os.path.join(tmp, f"auto_{tag}.tif")
            eq_tif = os.path.join(tmp, f"eq_{tag}.tif")
            res = subprocess.run([
                NEG_PROCESS, *extra, "-P", "srgb",
                "--roi", "auto",
                "--film_base_rgb", "auto",
                "--profile_film_base_rgb", "9000", "4800", "2600",
                "-o", auto_tif, strip_dng,
            ], capture_output=True, text=True)
            assert res.returncode == 0, res.stderr
            with open(auto_tif + ".json") as f:
                smeta = json.load(f)
            assert smeta["geometry"]["roi_source"] == "auto"
            assert smeta["film_base"]["applied"] is True
            assert smeta["film_base"]["source"] == "rebate"
            cli_eq = smeta["cli_equivalent"]
            res_eq = subprocess.run([
                NEG_PROCESS, *extra, "-P", "srgb", *cli_eq, "-o", eq_tif, strip_dng,
            ], capture_output=True, text=True)
            assert res_eq.returncode == 0, res_eq.stderr
            with open(auto_tif, "rb") as fa, open(eq_tif, "rb") as fb:
                assert fa.read() == fb.read(), f"TIFF mismatch on {tag} path"


def test_roi_auto_detection_failure_falls_back_to_full_frame():
    """A lens-cap frame (black everywhere) and a blank strip (one uniform value) fall back like any low confidence."""
    w, h = 240, 160
    scenes = {
        "lens_cap": [512 + 2 + (x * 7 + y * 3) % 3 for y in range(h) for x in range(w)],
        "blank_strip": [9000] * (w * h),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, pixels in scenes.items():
            dng = os.path.join(tmp, name + ".dng")
            tif = os.path.join(tmp, name + ".tif")
            write_synthetic_dng(dng, w, h, pixels)
            res = subprocess.run([NEG_PROCESS, "-P", "srgb", "--roi", "auto", "-o", tif, dng],
                                 capture_output=True, text=True)
            assert res.returncode == 0, f"{name}: {res.stderr}"
            with open(tif + ".json") as f:
                meta = json.load(f)
            assert meta["geometry"]["roi_source"] == "auto_fallback", name
            assert meta["geometry"]["roi"] == [0, 0, w, h], name
            assert meta["frame_detection"]["low_confidence"] is True, name
            assert "falling back" in res.stdout + res.stderr, name


def test_makefile_header_staleness():
    """Tests 5.7: touching flat_field.h triggers rebuild of profile_bundle.o, multi_target_optimizer.o, dinov3_auto_solver.o."""
    for hdr in ("flat_field.h", "flat_sample.h", "film_frame_detector.h"):
        res = subprocess.run(["make", "-n", "-W", hdr, "lib"], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        for obj in ("bin_out/profile_bundle.o", "bin_out/multi_target_optimizer.o", "bin_out/dinov3_auto_solver.o"):
            assert obj in res.stdout, f"{obj} not rebuilt when {hdr} is modified:\n{res.stdout}"


def main():
    test_check_profile_bundles_shutter()
    test_cli_validation_and_error_handling()
    test_pixel_shift_rejection()
    test_manual_target_and_gains_uses_bracket_e_center()
    test_roi_auto_picture_only_fallback_and_cli_equivalent_parity()
    test_roi_auto_detection_failure_falls_back_to_full_frame()
    test_makefile_header_staleness()
    print("neg_process_cli_test: all assertions passed.")


if __name__ == "__main__":
    main()
