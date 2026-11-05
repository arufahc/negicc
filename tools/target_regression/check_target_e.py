#!/usr/bin/env python3
"""--target against Auto: the single-bracket search, given Auto's bracket, must land near Auto's E.

For each case, neg_process runs Auto (the bundle search, curves off) and then --target with the bracket Auto chose,
with no -E or gains, on the same frame and arguments. The two E values (station units, from each sidecar's
cli_equivalent) must agree within --tol stops (default 0.3). The review's frames are 0105, 0193 (white LED) and 0030
(TriLED), in cases.json; a case names its frame by root and path, or by root and a name its file name contains.

  python3 tools/target_regression/check_target_e.py --bin bin_out/neg_process \
      --root web=<white-LED review corpus> --root triled=<TriLED development RAWs> [--cases other.json] [--tol 0.3]

Exits 1 when a case fails or cannot run; a case whose root is not given is skipped and counted.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

RAW_EXT = ('.arw', '.dng', '.nef', '.cr2', '.cr3', '.raf', '.orf', '.rw2', '.tif', '.tiff', '.jpg', '.jpeg')


def find_input(root, case):
    if case.get('path'):
        return os.path.join(root, case['path'])
    hits = []
    for d, _, files in os.walk(root):
        for f in files:
            if case['name'] in os.path.splitext(f)[0] and f.lower().endswith(RAW_EXT):
                hits.append(os.path.join(d, f))
    if len(hits) != 1:
        raise RuntimeError(f"{case['name']}: {len(hits)} files under {root} match (give its path in the case)")
    return hits[0]


def run(binary, args, out_tif):
    cmd = [binary] + args + ['-o', out_tif]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"neg_process failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    with open(out_tif + '.json') as f:
        return json.load(f)


def cli_value(sc, flag):
    cli = sc['cli_equivalent']
    return cli[cli.index(flag) + 1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--root', action='append', default=[], help='name=dir')
    ap.add_argument('--cases', default=os.path.join(here, 'cases.json'))
    ap.add_argument('--tol', type=float, default=0.3, help='stops')
    a = ap.parse_args()
    roots = dict(r.split('=', 1) for r in a.root)
    with open(a.cases) as f:
        cases = json.load(f)['cases']
    failed = skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases:
            if case['root'] not in roots:
                print(f"{case['name']}: skipped (no --root {case['root']})")
                skipped += 1
                continue
            try:
                src = find_input(roots[case['root']], case)
                base = case.get('args', []) + ['-P', 'srgb']
                auto = run(a.bin, base + ['--fit_curves', 'off', src], os.path.join(tmp, case['name'] + '_auto.tif'))
                idx = auto['auto']['target_idx']
                e_auto = float(cli_value(auto, '-E'))
                tgt = run(a.bin, base + ['--target', str(idx), src], os.path.join(tmp, case['name'] + '_target.tif'))
                e_tgt = float(cli_value(tgt, '-E'))
                d = math.log2(e_tgt / e_auto)
                ok = abs(d) <= a.tol
                failed += not ok
                print(f"{case['name']}: bracket {idx}, Auto E {e_auto:.4f}, --target E {e_tgt:.4f}, "
                      f"{d:+.2f} stops {'ok' if ok else 'FAIL'} (centre {tgt['dino']['e_center']:.4f})")
            except Exception as e:  # noqa: BLE001 - every failure is reported and counted
                failed += 1
                print(f"{case['name']}: FAIL ({e})")
    print(f"{len(cases) - failed - skipped} passed, {failed} failed, {skipped} skipped (tolerance {a.tol} stops)")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
