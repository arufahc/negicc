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
import cv2
import math
import numpy as np
import colour
import os
import shutil
import subprocess
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.patches as patches
import page_slider
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
        'ektar100-2', # Looks worse than ektar100-1 profile.
        'ektar100-3', # Poor quality.
        'ektar100-4', # Poor quality.
        'ektar100-5', # Poor quality.
        'ektar100+1',
        'ektar100+2',
        'ektar100+3',
        'portra160',
        'portra160-1', # Poor quality. Might be better to use portra160 and then brighten image.
        'portra160+1',
        'portra160+2',
        'portra400',
        'portra400-1', # Poor quality. Might be better to use portra400 and then brighten image.
        'portra400+1',
        'portra400+2',
    ],
    help="Profile of the scanned film or the name of the generated profile.")
parser.add_argument(
    "--profile_type", '-t',
    choices=['cLUT', 'Matrix'],
    default='cLUT',
    help="Profile type used to attach (no -P specified) or to convert (with -P specified).")
parser.add_argument("--colorspace", '-P', help="Output color profile."
    " If not spciefied input profile is attached."
    " If specified then convert using specified profile, default is 'srgb'")
parser.add_argument(
    '--raw_file', '-f',
    help="Name of the input raw file.")
parser.add_argument(
    '--film_base_raw_file', '-F',
    help="Raw file for the capture of the film base."
    " The channel balance is computed from the film base to compute compensations"
    " that should be applied to match that of the target. This method is to"
    " account for variations of the film base density.")
parser.add_argument(
    '--film_base_rgb', '-B', nargs=3,
    help="Uncorrected RGB values of the film base."
    " The channel balance is computed from the film base to compute compensations"
    " that should be applied to match that of the target. This method is to"
    " account for variations of the film base density.")
# TODO: Fix multi_shot mode for the intermediate runs of neg_process.
parser.add_argument(
    '--multi_shot', '-M',
    action='store_true',
    help="Sony 4-shots multishot mode. Assume 4 consecutive shots from [raw_file].")
parser.add_argument(
    '--no_crop', '-C',
    action='store_true',
    default=False,
    help="No cropping based on aspect ratio from metadata in [raw_file].")
parser.add_argument(
    '--interactive_mode', '-i',
    action='store_true',
    help="Interactive mode to select profile and parameters.")
parser.add_argument(
    '--debug', '-d',
    action='store_true',
    help="Debug mode and print neg_process arguments.")
parser.add_argument(
    '--measurement', '-m',
    choices=['', 'R190808'],
    default='R190808',
    help="Measurement used.")
parser.add_argument(
    '--quality', '-q',
    default=0,
    type=int,
    help="Quality. 0 = linear, 3 = AHD, 11 = DHT, 12 = mod AHD.")
parser.add_argument(
    '--color_comp', '-c',
    help="Multipliers for corrected RGB."
    " This overrides the --film_base_raw_file option."
    " The purpose of this flag is to manually adjust channel"
    " balance before ICC profile is applied.")
parser.add_argument(
    '--exposure_comp', '-E',
    type=float,
    help="Single multiplier for RGB values.")
parser.add_argument(
    '--post_correction_gamma', '-G',
    type=float,
    default=1.0,
    help="Gamma to apply to linear RGB after correction and before ICC is applied."
    " This is applied to all RGB values and should not affect color balance, the effect is"
    " to reduce contrast and to recover blown highlight.")
parser.add_argument(
    '--half_size', '-H',
    action='store_true',
    help="Generate half size image.")
parser.add_argument(
    '--quarter_size', '-Q',
    action='store_true',
    help="Generate quarter size image.")

parser.add_argument('--target', '-T', action='store_true')
args = parser.parse_args()

if args.target:
    subprocess.run([os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
                    '--half_size',
                    '--no_crop',
                    '-o', Path(args.raw_file).stem + ('.target.tif'),
                    args.raw_file], check=True)
    exit(0)

if not args.profile:
    if not args.emulsion:
        print('At least --emulsion needs to be specified!')
        exit(1)

plt.rcParams.update({'font.size': 7})
plt.rcParams["figure.figsize"] = (15,10)
plt.rcParams["image.interpolation"] = 'none'


def read_profile_info(name):
    profile_info_txt = '%s/icc_out/Sony A7RM4 %s %s Info.txt' % (os.path.dirname(__file__),
                                                                 name.capitalize(),
                                                                 args.measurement)
    if not os.path.exists(profile_info_txt):
        return None
    matrix = []
    shutter_speed = ''
    film_base_rgb = []
    with open(profile_info_txt) as f:
        for i in range(0, 3):
            coeffs = f.readline().strip('\r\n').split(' ')[0:3]
            matrix.append(list(map(float, coeffs)))
        shutter_speed = f.readline().strip('\r\n').split(' ')[0]
        film_base_rgb = f.readline().strip('\r\n').split(' ')[0:3]
        f.readline() # Min
        f.readline() # Max
        mean_rgb = f.readline().strip('\r\n').split(' ')[0:3] # Mean
        mid_grey_rgb = f.readline().strip('\r\n').split(' ')[0:3] # Mid-grey
    return {
        'exp': int(name[-2:]) if name[-2] in ['+', '-'] else 0,
        'emulsion': name[:-2] if name[-2] in ['+', '-'] else name,
        'name': name,
        'matrix': matrix,
        'shutter_speed': float(shutter_speed),
        'film_base_rgb': list(map(int, film_base_rgb)),
        'mean_rgb': list(map(float, mean_rgb)),
        'mid_grey_rgb': list(map(float, mid_grey_rgb)),
        }


def compute_relative_transmittance(correction_mat, rgb, film_base_rgb):
    return np.matmul(correction_mat, rgb) / np.matmul(correction_mat, film_base_rgb)


def get_profile_and_scale_factors(raw_file, film_base_rgb):
    '''Assuming shutter speed is the only variable between the profile and RAW capture,
    pick the profile that is most suitable for the RAW capture. Also return the scale
    factors that should applied to the profiles such that the exposure is uniform.
    The scale factor dictionary is useful for users to select between profiles with
    consistent exposure.'''
    raw_shutter_speed = subprocess.check_output([os.path.join(os.path.dirname(__file__), 'bin_out', 'raw_info'),
                                                 '-s', raw_file]).decode(sys.stdout.encoding)
    raw_shutter_speed = float(raw_shutter_speed.split(' ')[0])
    if args.profile:
        profile = read_profile_info(args.profile)
        return profile

    # If profile is not specified use emulsion and shutter speed to select profile automatically.
    profile = {}
    profiles = []
    # Append profiles that are exposed over and under.
    # Profiles made too under-exposed have poor quality and are excluded.
    for exp_diff in ['', '-3', '-2', '-1', '+1', '+2', '+3']:
        exp_diff_profile = read_profile_info(args.emulsion + exp_diff)
        if exp_diff_profile:
            exp_diff_profile['exp_diff'] = exp_diff
            profiles.append(exp_diff_profile)

    # Use the first profile to compute the relative transmittance. Doesn't matter
    # which profile is choosen because transmittance is relative to the film base
    # so scaling applied to the corrected RGB values will cancelled out when
    # divided by the corrected film base RGB values.
    run_neg_process(args.raw_file, profiles[0], 1, 1, film_base_rgb,
                    # No ICC profile is applied
                    None, 4, False, 'temp.tif')
    neg_img = cv2.imread('temp.tif', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    h, w, _ = neg_img.shape
    CROP_FACTOR = 7
    crop_img = cv2.cvtColor(neg_img[int(h/CROP_FACTOR):int(h-h/CROP_FACTOR),
                                    int(w/CROP_FACTOR):int(w-w/CROP_FACTOR)], cv2.COLOR_BGR2RGB)
    correction_mat = np.array([profiles[0]['matrix'][0],
                               profiles[0]['matrix'][1],
                               profiles[0]['matrix'][2]])
    corrected_film_base_rgb = np.matmul(correction_mat, film_base_rgb)
    transmittance_img = crop_img / raw_shutter_speed / corrected_film_base_rgb
    mean_transmittance = np.mean(transmittance_img, axis = (0,1))

    # mean_rgb in the profile is normalized to 1s so multiply the shutter speed
    # to get original RGB values.
    profile_mean_rgb = np.array(profiles[0]['mean_rgb']) * profiles[0]['shutter_speed']

    if args.debug:
        print("Raw shutter speed: %f" % raw_shutter_speed)
        print('Mean film RGB: %f %f %f' % tuple(np.mean(crop_img, axis = (0,1))))
        print('Mean film relative transmittance: %f %f %f' % tuple(mean_transmittance))
    max_profile_distance = 10000
    p_to_scale = {}
    transmittance_vector = []
    exp_diff_vector = []
    for p in profiles:
        correction_mat = np.array([p['matrix'][0], p['matrix'][1], p['matrix'][2]])
        profile_mid_grey_transmittance = compute_relative_transmittance(
            correction_mat, p['mid_grey_rgb'], p['film_base_rgb'])
        profile_mean_transmittance = compute_relative_transmittance(
            correction_mat, p['mean_rgb'], p['film_base_rgb'])
        profile_distance = np.max(np.absolute(np.log10(mean_transmittance) - np.log10(profile_mid_grey_transmittance)))
        transmittance_vector.append(profile_mid_grey_transmittance)
        exp_diff_vector.append(int(p['exp_diff'] if p['exp_diff'] else 0))
        if args.debug:
            print('[%s] Evaluating profile' % p['name'])
            print('  Shutter speed: %f' % p['shutter_speed'])
            print('  Mean transmittance: %f %f %f' % tuple(profile_mean_transmittance))
            print('  Mid-grey transmittance: %f %f %f' % tuple(profile_mid_grey_transmittance))
            print("  Mid-grey distance to mean transmittance: %f" % profile_distance)
        if profile_distance < max_profile_distance and p['exp_diff'] not in ['-2', '-3']:
            max_profile_distance = profile_distance
            profile = p

    transmittance_map = cv2.resize(transmittance_img, (10, 10), interpolation = cv2.INTER_LINEAR)
    exp_map = []
    for row in range(0, transmittance_map.shape[0]):
        exp_row = []
        for col in range(0, transmittance_map.shape[1]):
            # Compute the distance for each profile mid-grey transmittance to that of the pixel.
            distance_vector = np.linalg.norm(np.repeat([transmittance_map[row, col]], [len(transmittance_vector)], axis=0) - transmittance_vector, axis=1)
            exp_row.append(exp_diff_vector[np.argmin(distance_vector)])
        exp_map.append(exp_row)

    print("Chosen profile %s with shutter speed: %f" % (profile['name'], profile['shutter_speed']))
    # Rely on the camera AE to properly expose the captured image. This is not very 
    # reliable because the shutter speed reported are not very accurate in AE mode.
    common_scale_factor = np.ones(3) * profile['shutter_speed'] / raw_shutter_speed
    print('Common scale factor: %f %f %f' % tuple(common_scale_factor))
    for p in profiles:
        # Assume the matrices are scaled by a simple factor between profiles.
        # Ideally the scale factor should be calculated using the matrix for the profile
        # but doing so is costly so assume the scale factor computed from base profile
        # is good enough.
        p_to_scale[p['name']] = np.mean(common_scale_factor) * (profile['matrix'][0][0] / p['matrix'][0][0])
        if args.debug:
            print('[%s] Scaling profile with factor %f' % (p['name'], p_to_scale[p['name']]))
    return profile, p_to_scale, exp_map


def compute_film_base_rgb(film_base_raw_file):
    '''Returns the film base RGB by computing average from the center of |film_base_raw_file|,
    multiplied by the 1/shutter_speed.'''
    raw_info_txt = Path(film_base_raw_file).stem + '.raw_info.txt'
    if os.path.exists(raw_info_txt):
        with open(raw_info_txt) as f:
            output = f.read()
    else:
        output = subprocess.check_output([os.path.join(os.path.dirname(__file__), 'bin_out', 'raw_info'),
                                          '-w', '-s', film_base_raw_file]).decode(sys.stdout.encoding)
        f = open(raw_info_txt, 'w+')
        f.write(output)
        f.close()
    output = output.split('\n')
    center_rgb = next((x for x in output if 'average RGB' in x), '1 1 1').split(' ')[0:3]
    shutter_speed = next((x for x in output if 'Shutter' in x), '1').split(' ')[0]
    return [int(float(x) / float(shutter_speed)) for x in center_rgb]


def run_neg_process(raw_file, profile, exposure_comp, post_correction_gamma, film_base_rgb, colorspace, scale_down_factor, no_crop, out_file_override=None):
    neg_process_args = [
        os.path.join(os.path.dirname(__file__), 'bin_out', 'neg_process'),
        '--exposure_comp', str(exposure_comp),
        '-q', str(args.quality)]
    if profile:
        neg_process_args += ['-r'] + list(map(str, profile['matrix'][0]))
        neg_process_args += ['-g'] + list(map(str, profile['matrix'][1]))
        neg_process_args += ['-b'] + list(map(str, profile['matrix'][2]))
        neg_process_args += ['--profile_film_base_rgb'] + list(map(str, profile['film_base_rgb']))
        neg_process_args += ['--film_base_rgb'] + list(map(str, film_base_rgb))
        neg_process_args += [
            '-p', '%s/icc_out/Sony A7RM4 %s %s %s.icc' % (os.path.dirname(__file__),
                                                          profile['name'].capitalize(),
                                                          args.measurement,
                                                          args.profile_type)]
        out_file = Path(raw_file).stem + ('.%s.%s.tif' % (profile['name'], args.profile_type.lower()))
    if exposure_comp is not None and exposure_comp != 1.0:
        out_file = Path(raw_file).stem + ('.%s.%s.E=%.2f.tif' % (
            profile['name'], args.profile_type.lower(), exposure_comp))
    if post_correction_gamma != 1.0:
        neg_process_args += ['--post_correction_gamma', str(post_correction_gamma)]
        out_file = Path(raw_file).stem + ('.%s.%s.E=%.2f.G=%.2f.tif' % (
            profile['name'], args.profile_type.lower(), exposure_comp, post_correction_gamma))
    if colorspace:
        neg_process_args += ['-P', colorspace]
    if scale_down_factor == 2:
        neg_process_args.append('--half_size')
    elif scale_down_factor == 4:
        neg_process_args.append('--quarter_size')
    if no_crop:
        neg_process_args.append('--no_crop')
    neg_process_args += ['-o', out_file_override or out_file]
    neg_process_args.append(raw_file)
    if args.multi_shot:
        file_num = int(Path(raw_file).stem[-4:])
        for i in range(1,4):
            neg_process_args.append(Path(raw_file).stem[0:-4] + str(file_num + i) + Path(raw_file).suffix)
    if args.debug:
        print(' '.join([("'" + x + "'" if ' ' in x else x) for x in neg_process_args]))
        subprocess.run(neg_process_args, check=True)
    else:
        subprocess.check_output(neg_process_args)
    return out_file_override or out_file


class FilmBaseSelector:
    def _line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

    def show_selector(self, path):
        film_base_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        film_base_img = cv2.cvtColor(film_base_img, cv2.COLOR_BGR2RGB)
        thumb_img = cv2.normalize(film_base_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        fig, ax = plt.subplots(1)
        plot = ax.imshow(thumb_img)
        fig.tight_layout()
        selector = widgets.RectangleSelector(ax, self._line_select_callback,
                                             useblit=True,
                                             button=[1, 3],  # don't use middle button
                                             minspanx=5, minspany=5,
                                             spancoords='pixels',
                                             interactive=True)
        img_h, img_w, _ = thumb_img.shape
        patch_w = min(img_h, img_w) / 4
        ax.add_patch(patches.Rectangle(
                ((img_w - patch_w) / 2, (img_h - patch_w) / 2),
                patch_w, patch_w, linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()
        xmin, xmax, ymin, ymax = selector.extents
        if xmax > xmin and ymax > ymin + 1:
            selected_img = film_base_img[int(ymin):int(ymax), int(xmin):int(xmax)]
            return [int(x) for x in np.mean([selected_img], axis=(0, 1, 2))]
        return None


selected_film_base_rgb = None
if args.film_base_raw_file and args.interactive_mode:
    film_base_tif = run_neg_process(args.film_base_raw_file, None, 1.0, 1.0, None, None, 4, True, 'film_base.tif')
    selected_film_base_rgb = FilmBaseSelector().show_selector(film_base_tif)
    if selected_film_base_rgb:
        raw_shutter_speed = subprocess.check_output([os.path.join(os.path.dirname(__file__), 'bin_out', 'raw_info'),
                                                     '-s', args.film_base_raw_file]).decode(sys.stdout.encoding)
        # TODO: Optimize this read from .raw_info.txt file.
        raw_shutter_speed = float(raw_shutter_speed.split(' ')[0])
        if args.debug:
            print('Selected film base shutter speed: %f' % raw_shutter_speed)
        selected_film_base_rgb = [int(x / raw_shutter_speed) for x in selected_film_base_rgb]
        print("Selected Film Base RGB: %d %d %d (normalized to 1s shutter speed)" % tuple(selected_film_base_rgb))

if selected_film_base_rgb is None:
    if args.film_base_raw_file:
        selected_film_base_rgb = compute_film_base_rgb(args.film_base_raw_file)
        print('Computed film base RGB %d %d %d (normalized to 1s shutter speed)' % tuple(selected_film_base_rgb))
    else:
        selected_film_base_rgb = list(map(int, args.film_base_rgb))
        print('Entered film base RGB %d %d %d (normalized to 1s shutter speed)' % tuple(selected_film_base_rgb))

profile, p_to_scale, exp_map = get_profile_and_scale_factors(args.raw_file, selected_film_base_rgb)

# Current params.
exp_comp = args.exposure_comp if args.exposure_comp else p_to_scale[profile['name']]
gamma = 1
profile_exp = profile['exp']
last_out_path = None

if not args.interactive_mode:
    out_file = run_neg_process(args.raw_file, profile, exp_comp, args.post_correction_gamma, selected_film_base_rgb, args.colorspace, 2 if args.half_size else (4 if args.quarter_size else 1), args.no_crop)
    print('Done %s' % out_file)
    exit(0)

fig, ax_img = plt.subplots()
fig.tight_layout()

ax_gamma = fig.add_axes([0.825, 0.30, 0.15, 0.018])
ax_exp_comp = fig.add_axes([0.825, 0.35, 0.15, 0.018])
ax_profile = fig.add_axes([0.825, 0.40, 0.15, 0.018])
ax_trans_map = fig.add_axes([0.82, 0.45, 0.15, 0.20])
ax_color_bar = fig.add_axes([0.965, 0.45, 0.01, 0.20])
ax_lab_hist = fig.add_axes([0.825, 0.68, 0.15, 0.10])
ax_l_hist = fig.add_axes([0.825, 0.82, 0.15, 0.10])

norm = matplotlib.colors.BoundaryNorm(np.linspace(-3, 4, 8), plt.cm.bone.N)
im_trans_map = ax_trans_map.imshow(exp_map, cmap='bone', norm=norm)
fig.colorbar(im_trans_map, cax=ax_color_bar, orientation='vertical', cmap='bone', norm=norm, ticks=np.arange(-3, 4))
def im_trans_map_onclick(event):
    if event.xdata != None and event.ydata != None and event.inaxes == ax_trans_map:
        val = event.inaxes.get_images()[0].get_cursor_data(event)
        slider_profile.set_val(val)
im_trans_map.figure.canvas.mpl_connect('button_press_event', im_trans_map_onclick)

slider_exp_comp = widgets.Slider(ax_exp_comp, 'Exp. Comp.', 0, 3, valinit=exp_comp)
slider_gamma = widgets.Slider(ax_gamma, 'Gamma', 0, 3, valinit=gamma)

slider_profile = page_slider.PageSlider(ax_profile, 'Profile Exp', min_page=-3, max_page=3, activecolor="orange", valinit=profile_exp)

fig.subplots_adjust(right=0.80)

def reprocess_and_show_image():
    global profile
    global last_out_path
    global exp_comp
    start = time.time()
    # cv2 reads the image without caring the embedded ICC profile.
    # Meanwhile imshow() will display the image assuming they have sRGB curves, at least in OSX.
    # For this reason apply a srgb output profile for correct brightness of the image displayed.
    new_profile_name = profile['emulsion'] + ('' if profile_exp == 0 else '%+1d' % profile_exp)
    new_profile = read_profile_info(new_profile_name)
    if new_profile and new_profile['name'] != profile['name']:
        profile = new_profile
        exp_comp = p_to_scale[profile['name']]
        slider_exp_comp.set_val(exp_comp)
        return
    last_out_path = run_neg_process(args.raw_file, profile, exp_comp, gamma,
                                    selected_film_base_rgb, 'srgb', 4, False, 'temp.tif')
    end_neg_process = time.time()
    out_img = cv2.imread(last_out_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    # The image can be shown as-is because it's in sRGB colorspace.
    ax_img.imshow(out_img, resample=False, filternorm=False)
    end_neg_process = time.time()

    # TODO: Exclude border.
    out_img = cv2.resize(out_img, (0, 0), fx = 0.1, fy = 0.1, interpolation = cv2.INTER_LINEAR)

    D50 = colour.CCS_ILLUMINANTS['cie_2_1931']['D50']
    xyz_img = colour.sRGB_to_XYZ(out_img, illuminant=D50)
    lab_img = colour.XYZ_to_Lab(xyz_img, illuminant=D50)
    L, a, b = cv2.split(lab_img)
    L = L.flatten()
    a = a.flatten()
    b = b.flatten()
    # 2D La*b* histogram.
    #hist = cv2.calcHist([lab_img], [1, 2], None, [256, 256], [0, 256, 0, 256])
    #hist = np.power(hist / hist.max(), 0.3)

    # Hack to modify in Lab colorspace.
    #D65 = colour.CCS_ILLUMINANTS['cie_2_1931']['D65']
    #xyz = colour.Lab_to_XYZ(cv2.merge([L, a, b + 30]), illuminant=D65)
    #ax_img.imshow(colour.XYZ_to_sRGB(xyz, illuminant=D65))
    ax_lab_hist.cla()
    ax_lab_hist.cla()
    ax_lab_hist.set_facecolor("grey")
    ax_lab_hist.set_xlim([-128, 128])
    a = 50*a/L
    b = 50*b/L
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    ax_lab_hist.hist(a, 50, color=('magenta' if a_mean >= 0 else 'green'))
    ax_lab_hist.hist(b, 50, color=('yellow' if b_mean >= 0 else 'blue'))
    ax_lab_hist.axvline(a_mean, color=('magenta' if a_mean >= 0 else 'green'), linestyle='dashed', linewidth=1)
    ax_lab_hist.axvline(b_mean, color=('yellow' if b_mean >= 0 else 'blue'), linestyle='dashed', linewidth=1)
    ax_lab_hist.text(a_mean + 3, 0.9, str(round(a_mean, 2)), transform=ax_lab_hist.get_xaxis_transform(), color='magenta')
    ax_lab_hist.text(b_mean + 3, 0.8, str(round(b_mean, 2)), transform=ax_lab_hist.get_xaxis_transform(), color='yellow')
    ax_lab_hist.tick_params(axis="y", labelsize=5)

    l_mean = L.mean()
    ax_l_hist.cla()
    ax_l_hist.axvline(l_mean, color='black', linestyle='dashed', linewidth=1)
    ax_l_hist.text(l_mean + 3, 0.9, str(round(l_mean, 2)), transform=ax_l_hist.get_xaxis_transform(), color='red')
    ax_l_hist.hist(L, 50, (0, 100), color='grey')
    end = time.time()
    print('Process time %f Show time %f' % (end_neg_process - start, end - start))

def update_exp_comp(val):
    global exp_comp
    exp_comp = val
    reprocess_and_show_image()
    pass

def update_gamma(val):
    global gamma
    gamma = val
    reprocess_and_show_image()
    pass
def update_profile(val):
    global profile_exp
    exp = int(val)
    if exp != profile['exp']:
        profile_exp = exp
        reprocess_and_show_image()

slider_exp_comp.on_changed(update_exp_comp)
slider_gamma.on_changed(update_gamma)
slider_profile.on_changed(update_profile)

reprocess_and_show_image()
plt.show()

print('Gamma %f' % gamma)
print('Exposure comp %f' % exp_comp)
print('Profile used %s' % profile['name'])

os.remove(last_out_path)
out_file = run_neg_process(args.raw_file, profile, exp_comp, gamma,
                selected_film_base_rgb, args.colorspace, 2 if args.half_size else (4 if args.quarter_size else 1), args.no_crop,
                Path(args.raw_file).stem + '.pos.tif')
print('Done %s' % out_file)

# TODO: Write parameters for debugging.
