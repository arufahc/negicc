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
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--emulsion", '-e',
    choices=[
        'ektar100',
        'ektar100-1',
        'ektar100+1',
        'ektar100+2',
        'portra400',
        'portra400-1',
        'portra400+1',
        'portra400+2',
        'portra160',
        'portra160-1',
        'portra160+1',
        'portra160+2',
    ],
    help="Emulsion of the scanned film or the name of the generated profile.")
parser.add_argument(
    '--raw_file', '-f',
    help="Name of the input raw file.")
parser.add_argument(
    '--measurement', '-m',
    choices=['', 'R190808'],
    default='R190808',
    help="Measurement used")
parser.add_argument('--target-mode', '-T', action='store_true')
args = parser.parse_args()

if args.target_mode:
    subprocess.run([os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
                    '-h',
                    '-o', Path(args.raw_file).stem + ('.target.tif'),
                    args.raw_file], check=True)
    exit(0)
subprocess.run([os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
                '-r', '1 -0.08262711 -0.01249409',
                '-g', '-0.13898878 1 -0.32017315',
                '-b', '-0.00664173 -0.09860774 1',
                '-p', '%s/icc_out/Sony A7RM4 %s %s cLUT.icc' % (os.path.dirname(__file__),
                                                                args.emulsion.capitalize(),
                                                                args.measurement),
                '-o', Path(args.raw_file).stem + ('.%s.tif' % args.emulsion),
                args.raw_file], check=True)
