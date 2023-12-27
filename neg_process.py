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
import time
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
        'ektar100-2',
        'ektar100-3',
        'ektar100-4',
        'ektar100-5',
        'ektar100+1',
        'ektar100+2',
        'ektar100+3',
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
    "--output_profile", '-P',
    choices=[
        'srgb',
        '<ICC profile path>',
    ],
    help="Output color profile. If not spciefied input profile is attached. If specified then convert using this profile.")
parser.add_argument(
    '--raw_file', '-f',
    help="Name of the input raw file.")
parser.add_argument(
    '--multi_shot', '-M',
    action='store_true',
    help="Sony 4-shots multishot mode. Assume 4 consecutive shots from [raw_file]")
parser.add_argument(
    '--scan_mode', '-s',
    action='store_true',
    help="Scanning mode will wait for a new file and process automatically.")
parser.add_argument(
    '--debug', '-d',
    action='store_true',
    help="Debug mode and print neg_process arguments.")
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
parser.add_argument(
    '--exposure_comp', '-c',
    type=float,
    help="Multiplier to compensate for over-/under-exposure of the negative.")
parser.add_argument('--target_mode', '-T', action='store_true')
args = parser.parse_args()

if args.target_mode:
    subprocess.run([os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
                    '-h',
                    '-o', Path(args.raw_file).stem + ('.target.tif'),
                    args.raw_file], check=True)
    exit(0)

if not args.profile:
    if not args.emulsion:
        print('At least --emulsion needs to be specified!')
        exit(1)

def read_profile_info(name):
    profile_info_txt = '%s/icc_out/Sony A7RM4 %s %s Info.txt' % (os.path.dirname(__file__),
                                                                 name.capitalize(),
                                                                 args.measurement)
    if not os.path.exists(profile_info_txt):
        return None
    matrix = []
    shutter_speed = ''
    with open(profile_info_txt) as f:
        for i in range(0, 3):
            coeffs = [float(x) for x in f.readline().strip('\r\n').split(' ')]
            matrix.append(coeffs)
        shutter_speed = f.readline().split(' ')[0]
    return {'name': name, 'matrix': matrix, 'shutter_speed': float(shutter_speed)}

def compute_exposure_comp(profile, raw_shutter_speed):
    '''Assuming shutter speed is the only variable between the profile and RAW capture,
    compute the exposure compensation that should be applied to the converted RGB matrix
    to match the exposure of the profile.'''
    return profile['shutter_speed'] / raw_shutter_speed

def get_profile_and_exposure_comp(raw_file):
    ''' Assuming shutter speed is the only variable between the profile and RAW capture,
    pick the profile with shutter speed that is cloest that of the RAW file.
    Scale the color correction matrix to accomodate the exposure compensation.'''
    raw_shutter_speed = subprocess.check_output([os.path.join(os.path.dirname(__file__), 'bin_out', 'raw_info'),
                                                 '-s', raw_file]).decode(sys.stdout.encoding).strip()
    if args.profile:
        return (read_profile_info(args.profile),
                compute_exposure_comp(args.profile, raw_shutter_speed))

    # If profile is not specified use emulsion and shutter speed to select profile automatically.
    profile = {}
    profiles = []
    # Append profiles that are exposed over and under.
    for exp_diff in ['', '-1', '-2', '-3', '-4', '-5', '+1', '+2', '+3']:
        exp_diff_profile = read_profile_info(args.emulsion + exp_diff)
        if exp_diff_profile:
            profiles.append(exp_diff_profile)
    raw_shutter_speed = float(raw_shutter_speed.strip('\r\n'))
    print("Raw shutter speed: %f" % raw_shutter_speed)
    max_exp_diff = 100
    for p in profiles:
        exp_diff = abs(math.log2(p['shutter_speed'] / raw_shutter_speed))
        if exp_diff < max_exp_diff:
            max_exp_diff = exp_diff
            profile = p
    print("Chosen profile %s with shutter speed: %f" % (profile['name'], profile['shutter_speed']))
    return (profile, compute_exposure_comp(profile, raw_shutter_speed))

def run_neg_process(raw_file):
    profile, exposure_comp = get_profile_and_exposure_comp(raw_file)
    if args.exposure_comp:
        exposure_comp = args.exposure_comp
    print("Exposure compensation applied: %f" % exposure_comp)
    neg_process_args = [
        os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
        '-r', ' '.join([str(x * exposure_comp) for x in profile['matrix'][0]]),
        '-g', ' '.join([str(x * exposure_comp) for x in profile['matrix'][1]]),
        '-b', ' '.join([str(x * exposure_comp) for x in profile['matrix'][2]]),
        '-q', str(args.quality),
        '-p', '%s/icc_out/Sony A7RM4 %s %s cLUT.icc' % (os.path.dirname(__file__),
                                                        profile['name'].capitalize(),
                                                        args.measurement),
        '-o', Path(raw_file).stem + ('.%s.tif' % profile['name'])]
    if args.output_profile:
        neg_process_args += ['-P', args.output_profile]

    neg_process_args.append(raw_file)
    if args.multi_shot:
        file_num = int(Path(raw_file).stem[-4:])
        for i in range(1,4):
            neg_process_args.append(Path(raw_file).stem[0:-4] + str(file_num + i) + Path(raw_file).suffix)
    if args.debug:
        print(' '.join([("'" + x + "'" if ' ' in x else x) for x in neg_process_args]))
    subprocess.run(neg_process_args, check=True)

# Single process mode.
if not args.scan_mode:
    run_neg_process(args.raw_file)
    exit(0)

seen_files = set(os.listdir(os.getcwd()))

while True:
    for f in os.listdir(os.getcwd()):
        if not os.path.isfile(f):
            continue
        if f in seen_files:
            continue
        # Only supports .ARW right now.
        if not f.endswith('.ARW'):
            continue
        if int(time.time()) - os.path.getmtime(f) < 3:
            continue
        print('Found new file %s.' % f)
        run_neg_process(f)
        seen_files.add(f)
    time.sleep(1)
