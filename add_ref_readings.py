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
import colour

def read_txt_readings(file):
    f = open(file, "r")
    fields = f.readline().strip('\n\r').split(' ')
    rows = {}
    while True:
        l = f.readline()
        if not l:
            break
        vals = l.strip('\n\r').split(' ')
        rows[vals[0]] = {}
        for i in range(1, len(fields)):
            rows[vals[0]][fields[i]] = float(vals[i])
    f.close()
    return rows

def build_empty(src, col):
    d = {}
    for key in src.keys():
        d[key] = {col: 0}
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Source readings file.")
    parser.add_argument("--r", help="R channel reference file.")
    parser.add_argument("--g", help="G channel reference file.")
    parser.add_argument("--b", help="B channel reference file.")
    parser.add_argument("--Yxy", help="Yxy reference file.")
    parser.add_argument("--XYZ", help="XYZ reference file.")
    args = parser.parse_args()

    src = read_txt_readings(args.src)
    if args.r:
        r =  read_txt_readings(args.r)
    else:
        r = build_empty(src, 'r')
    if args.g:
        g =  read_txt_readings(args.g)
    else:
        g = build_empty(src, 'g')
    if args.b:
        b =  read_txt_readings(args.b)
    else:
        b = build_empty(src, 'b')
    if args.Yxy:
        Yxy = read_txt_readings(args.Yxy)
    elif args.XYZ:
        XYZ = read_txt_readings(args.XYZ)

    if not args.Yxy and not args.XYZ:
        print('patch', 'r', 'g', 'b', 'refR', 'refG', 'refB')
        for patch, vals in src.items():
            print(patch, vals['r'], vals['g'], vals['b'], r[patch]['r'],  g[patch]['g'],  b[patch]['b'])
    else:
        print('patch', 'r', 'g', 'b', 'refR', 'refG', 'refB', 'refX', 'refY', 'refZ')

        for patch, vals in src.items():
            if args.Yxy:
                X, Y, Z = colour.xyY_to_XYZ([Yxy[patch]['x'], Yxy[patch]['y'], Yxy[patch]['Y']])
            elif args.XYZ:
                X, Y, Z = XYZ[patch]['X'], XYZ[patch]['Y'], XYZ[patch]['Z']
            print(patch, vals['r'], vals['g'], vals['b'], r[patch]['r'],  g[patch]['g'],  b[patch]['b'], X, Y, Z)
