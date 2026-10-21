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
or decimal seconds ("0.4s", "2"), with or without the trailing "s". The exports come from negicc-station
(tools/export_negicc_data.py), so a bad one fails the build here rather than a conversion later.

Usage: check_profile_bundles.py <bundle.json> ...
"""

import json
import math
import sys


def shutter_to_seconds(shutter):
    """Seconds of a shutter speed string; raises ValueError when it cannot be read."""
    s = str(shutter).strip()
    while s.endswith("s"):
        s = s[:-1]
    num, slash, den = s.partition("/")
    value = float(num) / float(den) if slash else float(num)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("not a positive number of seconds")
    return value


def check(path):
    """The bundle's problems, one message per bracket that has one."""
    with open(path) as f:
        bundle = json.load(f)
    targets = bundle.get("targets")
    if not isinstance(targets, list) or not targets:
        return ["no targets array"]
    problems = []
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
