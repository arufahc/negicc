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
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--emulsion", '-e',
    choices=[
        'ektar100',
        'portra160',
        'portra400',
    ],
    help="Emulsion of the scanned film. Profile will be selected automatically.")
parser.add_argument(
    "--profile", '-p',
    choices=[
        'ektar100',
        'ektar100-1',
        'ektar100+1',
        'ektar100+2',
        'portra160',
        'portra160-1',
        'portra160+1',
        'portra160+2',
        'portra400',
        'portra400-1',
        'portra400+1',
        'portra400+2',
    ],
    help="Profile of the scanned film or the name of the generated profile.")
parser.add_argument(
    '--raw_file', '-f',
    help="Name of the input raw file.")
parser.add_argument(
    '--multi_shot', '-M',
    action='store_true',
    help="Sony 4-shots multishot mode. Assume 4 consecutive shots from [raw_file]")
parser.add_argument(
    '--measurement', '-m',
    choices=['', 'R190808'],
    default='R190808',
    help="Measurement used")
parser.add_argument(
    '--quality', '-q',
    default=0,
    type=int,
    help="Quality. 0 = linear, 3 = AHD, 11 = DHT, 12 = mod AHD")
parser.add_argument('--target_mode', '-T', action='store_true')
args = parser.parse_args()

if args.target_mode:
    subprocess.run([os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
                    '-h',
                    '-o', Path(args.raw_file).stem + ('.target.tif'),
                    args.raw_file], check=True)
    exit(0)

def read_profile_info(name):
    matrix = []
    shutter_speed = ''
    with open('%s/icc_out/Sony A7RM4 %s %s Info.txt' % (os.path.dirname(__file__),
                                                        name.capitalize(),
                                                        args.measurement)) as f:
        for i in range(0, 3):
            matrix.append(f.readline().strip('\r\n'))
        shutter_speed = f.readline().split(' ')[0]
    return {'name': name, 'matrix': matrix, 'shutter_speed': float(shutter_speed)}

profile = {}

# If profile is not specified use emulsion and shutter speed to select profile automatically.
if not args.profile:
    if not args.emulsion:
        print('At least --emulsion needs to be specified!')
        exit(1)
    profiles = []
    profiles.append(read_profile_info(args.emulsion))
    profiles.append(read_profile_info(args.emulsion + '-1'))
    profiles.append(read_profile_info(args.emulsion + '+1'))
    profiles.append(read_profile_info(args.emulsion + '+2'))
    raw_shutter_speed = subprocess.check_output([os.path.join(os.path.dirname(__file__), 'bin_out', 'raw_info'),
                                                 '-s', args.raw_file]).decode(sys.stdout.encoding).strip()
    raw_shutter_speed = float(raw_shutter_speed.strip('\r\n'))
    print("Raw shutter speed: %f" % raw_shutter_speed)

    max_exp_diff = 100
    for p in profiles:
        exp_diff = abs(math.log2(p['shutter_speed'] / raw_shutter_speed))
        if exp_diff < max_exp_diff:
            max_exp_diff = exp_diff
            profile = p
    print("Choose profile %s with shutter speed: %f" % (profile['name'], profile['shutter_speed']))
else:
    profile = read_profile_info(args.profile)

neg_process_args = [
    os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
    '-r', profile['matrix'][0],
    '-g', profile['matrix'][1],
    '-b', profile['matrix'][2],
    '-q', str(args.quality),
    '-p', '%s/icc_out/Sony A7RM4 %s %s cLUT.icc' % (os.path.dirname(__file__),
                                                    profile['name'].capitalize(),
                                                    args.measurement),
    '-o', Path(args.raw_file).stem + ('.%s.tif' % profile['name']),
    args.raw_file]

if args.multi_shot:
    file_num = int(Path(args.raw_file).stem[-4:])
    for i in range(1,4):
        neg_process_args.append(Path(args.raw_file).stem[0:-4] + str(file_num + i) + Path(args.raw_file).suffix)

subprocess.run(neg_process_args, check=True)
