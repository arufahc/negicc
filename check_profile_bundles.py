#!/usr/bin/env python3
# Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
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

"""Checks the profile bundles the build embeds into neg_process, which throws on a bundle it cannot read.

Every bracket needs a shutter speed in a form profile_bundle.cc's shutter_to_seconds reads: a fraction ("1/8s"),
or decimal seconds ("0.4s", "2"), with or without the trailing "s". The bundle's input_space, when given, must be
camera_raw_rgb or rec2020_linear, and a rec2020_linear bundle must carry no crosstalk correction matrix (the rules of
profile_bundle.cc's parse_profile_bundle). The exports come from negicc-station (tools/export_negicc_data.py), so a
bad one fails the build here rather than a conversion later.

Usage: check_profile_bundles.py <bundle.json> ...
"""

import json
import math
import re
import sys

# What profile_bundle.cc's shutter_to_seconds_d reads: ASCII decimals, with an optional exponent. Python's float() also
# takes "1_000" and any Unicode digits, which neg_process rejects, so they are refused here too.
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)


def shutter_to_seconds(shutter):
    """Seconds of a shutter speed string; raises ValueError when it cannot be read."""
    s = str(shutter).strip()
    while s.endswith("s"):
        s = s[:-1]
    num, slash, den = s.partition("/")
    for part in ((num, den) if slash else (s,)):
        if not _NUMBER.fullmatch(part):
            raise ValueError("not a number")
    if slash:
        den_f = float(den)
        if not math.isfinite(den_f) or den_f <= 0.0:
            raise ValueError("not a positive denominator")
        value = float(num) / den_f
    else:
        value = float(s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("not a positive number of seconds")
    return value


def check(path):
    """The bundle's problems, one message per bracket that has one."""
    with open(path) as f:
        bundle = json.load(f)
    problems = []
    space = bundle.get("input_space", "camera_raw_rgb")
    if space not in ("camera_raw_rgb", "rec2020_linear"):
        problems.append("unknown input_space %r (camera_raw_rgb or rec2020_linear)" % (space,))
    elif space == "rec2020_linear":
        crosstalk = bundle.get("crosstalk_profile")
        nested = isinstance(crosstalk, dict) and crosstalk.get("crosstalk_correction_matrix") is not None
        if nested or bundle.get("crosstalk_correction_matrix") is not None:
            problems.append("a rec2020_linear bundle carries a crosstalk correction matrix (re-export it from "
                            "negicc-station)")
    fb = bundle.get("film_base")
    if isinstance(fb, dict) and "shutter" in fb:
        try:
            shutter_to_seconds(fb["shutter"])
        except (TypeError, ValueError) as e:
            problems.append("film_base: cannot read shutter %r (%s)" % (fb["shutter"], e))
    targets = bundle.get("targets")
    if not isinstance(targets, list) or not targets:
        return problems + ["no targets array"]
    for i, target in enumerate(targets):
        name = target.get("name", "target %d" % (i + 1)) if isinstance(target, dict) else "target %d" % (i + 1)
        if not isinstance(target, dict) or "shutter" not in target:
            problems.append("%s: no shutter" % name)
            continue
        try:
            shutter_to_seconds(target["shutter"])
        except (TypeError, ValueError) as e:
            problems.append("%s: cannot read shutter %r (%s)" % (name, target["shutter"], e))
    return problems


def main(paths):
    failed = False
    for path in paths:
        try:
            problems = check(path)
        except (OSError, ValueError) as e:
            problems = ["cannot read: %s" % e]
        for problem in problems:
            print("ERROR! %s: %s" % (path, problem), file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
