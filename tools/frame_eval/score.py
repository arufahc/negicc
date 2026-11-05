#!/usr/bin/env python3
"""Scores the film frame detector against truth.json.

usage: score.py --bin bin_out/frame_detect --root triled=DIR --root web=DIR --root film_scans=DIR [--jobs N]
               [--truth truth.json] [--out result.json] [--compare other_result.json]

Each frame runs with its true format (as negicc-web runs --roi auto with --film_format); a crop passes when its IoU
with the truth is at least 0.90. Frames whose root is not given are skipped and counted. --truth names the truth file
(default: truth.json next to this script). A frame's set is its "set" field when it has one, else from its root and
folder: triled, white (web), bw, neg35, mf (film_scans by folder); the summary lists the sets the truth file holds.
--compare lists the frames whose pass changed against an earlier result.
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def set_of(v):
    if "set" in v:
        return v["set"]
    if v["root"] == "triled":
        return "triled"
    if v["root"] == "web":
        return "white"
    folder = v["rel"].split("/")[0]
    return {"35mm BW": "bw", "35mm Negative": "neg35"}.get(folder, "mf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--root", action="append", default=[], help="NAME=DIR")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--truth", default=os.path.join(HERE, "truth.json"))
    ap.add_argument("--out")
    ap.add_argument("--compare")
    a = ap.parse_args()
    roots = dict(r.split("=", 1) for r in a.root)
    T = json.load(open(a.truth))
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 1) // max(1, a.jobs))))

    def run(item):
        k, v = item
        path = os.path.join(roots[v["root"]], v["rel"])
        r = subprocess.run([a.bin, "--film_format", v["format"], path], capture_output=True, text=True, env=env)
        try:
            return k, json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            return k, {"error": r.stderr[-300:]}

    items = [(k, v) for k, v in T["frames"].items() if v["root"] in roots]
    skipped = len(T["frames"]) - len(items)
    with ThreadPoolExecutor(a.jobs) as ex:
        det = dict(ex.map(run, items))
    res = {}
    for k, v in items:
        d = det[k]
        s = iou(d["crop"], v["crop"]) if "crop" in d else 0.0
        res[k] = {"set": set_of(v), "iou": round(s, 4), "pass": s >= 0.90, "crop": d.get("crop"), "truth": v["crop"],
                  "conf": d.get("conf"), "reason": d.get("reason")}
    summary = {}
    sets = list(dict.fromkeys(set_of(v) for v in T["frames"].values()))  # in the truth file's order
    for sn in sets:
        r = [x for x in res.values() if x["set"] == sn]
        summary[sn] = {"frames": len(r), "pass": sum(x["pass"] for x in r)}
    summary["all"] = {"frames": len(res), "pass": sum(x["pass"] for x in res.values()), "skipped": skipped}
    print(" ".join(f"{k} {v['pass']}/{v['frames']}" for k, v in summary.items()), f"(skipped {skipped})")
    for k, x in sorted(res.items(), key=lambda kv: (kv[1]["set"], kv[0])):
        if not x["pass"]:
            print(f"  fail {x['set']:6s} {k}  IoU {x['iou']:.2f}  crop {x['crop']}  truth {x['truth']}")
    if a.compare:
        old = json.load(open(a.compare))["frames"]
        for k in sorted(res):
            if k in old and old[k]["pass"] != res[k]["pass"]:
                print(f"  {'fixed' if res[k]['pass'] else 'BROKE'} {k}  IoU {old[k]['iou']:.2f} -> {res[k]['iou']:.2f}")
    if a.out:
        json.dump({"summary": summary, "frames": res}, open(a.out, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
