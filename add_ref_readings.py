# Copyright 2021 Alpha Lam <arufa.hc@gmail.com>
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

import argparse
import json
import re

try:
    import colour
except ImportError:
    colour = None

_RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")
_RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")

def read_txt_readings(file):
    f = open(file, "r")
    fields = f.readline().strip('\n\r').split(' ')
    rows = {}
    while True:
        l = f.readline()
        if not l:
            break
        vals = l.strip('\n\r').split(' ')
        rows[vals[0].lower()] = {}
        for i in range(1, len(fields)):
            rows[vals[0].lower()][fields[i]] = float(vals[i])
    f.close()
    return rows

def _rgb_from_json_patches(patches):
    return {k.lower(): {c: float(v.get(c, 0)) for c in ('r', 'g', 'b')}
            for k, v in patches.items() if isinstance(v, dict)}

def read_readings(file):
    if not file.endswith('.json'):
        return read_txt_readings(file)
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'patches' in data:
        return _rgb_from_json_patches(data['patches'])
    if data.get('targets'):
        return _rgb_from_json_patches(data['targets'][0].get('patches', {}))
    return _rgb_from_json_patches(data)

def read_xyz_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    patches = data.get('patches', data)
    return {k.lower(): {c: float(v.get(c, 0)) for c in ('X', 'Y', 'Z')}
            for k, v in patches.items() if isinstance(v, dict)}

def read_xyz_readings(file):
    if file.endswith('.json'):
        return read_xyz_json(file)
    if is_it8(file):
        return read_it8_readings(file)
    return read_txt_readings(file)

def is_it8(file):
    with open(file, "r") as f:
        return f.readline().startswith("IT8")

def read_it8_readings(file):
    f = open(file, "r")
    rows = {}
    fields = []
    name_map = {}
    name_map['SAMPLE_ID'] = 'patch'
    name_map['XYZ_X'] = 'X'
    name_map['XYZ_Y'] = 'Y'
    name_map['XYZ_Z'] = 'Z'
    while True:
        l = f.readline()
        if not l:
            break
        l = l.strip('\n\r ')
        if l == "BEGIN_DATA":
            break
        if l.startswith('SAMPLE_ID'):
            vals = _RE_COMBINE_WHITESPACE.sub(" ", l.strip('\n\r')).split(" ")
            for i in range(0, 4):
                fields.append(name_map[vals[i]])
    while True:
        l = f.readline()
        if not l:
            break
        if l.strip('\n\r ').startswith("END_DATA"):
            break
        vals = _RE_COMBINE_WHITESPACE.sub(" ", l.strip('\n\r')).split(" ")
        rows[vals[0].lower()] = {}
        for i in range(1, len(fields)):
            rows[vals[0].lower()][fields[i]] = float(vals[i])
    f.close()
    return rows

def build_empty(src, col):
    d = {}
    for key in src.keys():
        d[key] = {col: 0}
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Source readings file (.txt or .json).")
    parser.add_argument("--r", help="R channel reference file.")
    parser.add_argument("--g", help="G channel reference file.")
    parser.add_argument("--b", help="B channel reference file.")
    parser.add_argument("--Yxy", help="Yxy reference file.")
    parser.add_argument("--XYZ", help="XYZ reference file (.txt IT8 or .json).")
    parser.add_argument("--json", action="store_true", help="Output in JSON format.")
    args = parser.parse_args()

    src = read_readings(args.src)
    if args.r:
        r = read_readings(args.r)
    else:
        r = build_empty(src, 'r')
    if args.g:
        g = read_readings(args.g)
    else:
        g = build_empty(src, 'g')
    if args.b:
        b = read_readings(args.b)
    else:
        b = build_empty(src, 'b')

    if args.Yxy:
        Yxy = read_txt_readings(args.Yxy)
    elif args.XYZ:
        XYZ = read_xyz_readings(args.XYZ)

    if args.Yxy and colour is None:
        raise RuntimeError("colour module required for Yxy conversion")

    def ref_xyz(patch):
        if args.Yxy:
            return tuple(float(v) for v in colour.xyY_to_XYZ(
                [Yxy[patch]['x'], Yxy[patch]['y'], Yxy[patch]['Y']]))
        if args.XYZ:
            return XYZ[patch]['X'], XYZ[patch]['Y'], XYZ[patch]['Z']
        return None

    fields = ['r', 'g', 'b', 'refR', 'refG', 'refB']
    if args.Yxy or args.XYZ:
        fields += ['refX', 'refY', 'refZ']
    rows = {}
    for patch, vals in src.items():
        row = [vals['r'], vals['g'], vals['b'], r[patch]['r'], g[patch]['g'], b[patch]['b']]
        xyz = ref_xyz(patch)
        if xyz is not None:
            row += list(xyz)
        rows[patch] = row

    if args.json:
        print(json.dumps({'patches': {patch: dict(zip(fields, row)) for patch, row in rows.items()}}, indent=2))
    else:
        print('patch', *fields)
        for patch, row in rows.items():
            print(patch, *row)
