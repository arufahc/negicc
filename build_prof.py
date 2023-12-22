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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import sys
from pathlib import Path
from scipy import interpolate
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# The purpose of this script is to compute the crosstalk correction matrix and
# tone curves for the RGB channels. The matrix and curves can then be used to
# build ICC profiles. Additional the training data will be corrected using the
# said matrix and tone curves to generate a TI3 file which then can be used to
# generate a cLUT profile by ArgyllCMS.
#
# The matrix, tone curves and cLUT can then be combined to profile a single
# ICC profile by make_icc tool.
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src", help="Source readings file.")
parser.add_argument(
    "--white_x",
    help="Source media white x. This is used for chromatic adaptation if the reference XYZ are not D50 adapted.")
parser.add_argument(
    "--white_y",
    help="Source media white y. This is used for chromatic adaptation if the reference XYZ are not D50 adapted.")
parser.add_argument(
    "--whitest_patch_scaling",
    help="The RGB value that should be assigned to the whitest (densest) patch. "
    "Setting this value allows extrapolation of brighter objects that whitest patch on the target.",
    default=0.6,
    type=float)
parser.add_argument(
    "--darkest_patch_scaling",
    help="The RGB value that should be assigned to the darkest (lightest) GS patch. "
    "Setting this value will scale the correct matrix according to the darkest GS patch.",
    default=40000,
    type=float)
parser.add_argument(
    "--fit_intercept",
    help="Whether to allow intercept in linear interpolation for crosstalk correction.",
    default=True)
parser.add_argument(
    "--crosstalk_r_coefs",
    help="R coefficients for the crosstalk correction matrix.")
parser.add_argument(
    "--crosstalk_g_coefs",
    help="G coefficients for the crosstalk correction matrix.")
parser.add_argument(
    "--crosstalk_b_coefs",
    help="B coefficients for the crosstalk correction matrix.")
parser.add_argument(
    "--debug",
    help="Print debug messages.",
    action="store_true")
parser.add_argument(
    "--film_name",
    help="Name of the film.",
    default="Generic")
parser.add_argument(
    "--install_dir",
    nargs='?',
    const='',
    help="Location to place the ICC profile and script.")
parser.add_argument(
    "--prescale_coef",
    help="Set a greyscale patch to pre-scale the coefficients.",
    default='gs14')

args = parser.parse_args()

# DataFrame that will keep all the source data and mutations.
df = pd.read_csv(args.src, index_col="patch")

if args.debug:
    print("Peak of a few data rows in source file:", args.src)
    print(df.head())
    print(df.tail())


def estimate_crosstalk_correction_coefficients():
    """
    Computes the crosstalk correction coefficients from global df variable. Uses the 'r', 'g'
    and 'b' series and correct then to match 'refR', 'refG', 'refB' series using linear
    interpolation.

    Returns:
      Crosstalk correction coefficients for RGB channels in a tuple.

     ALGORITHM

     We assume there is a linear relationship between the R, G, B values with the reference
     values. This assumption is generally true because the camera R value comes from the
     signals captured in the G and B spectrum, and so on for G and B vaules.
     This is particularly true if the images are captured via a narrow band filter from a
     negative IT8 or from a positive IT8 target.
     If the images are from a negative IT8 target with wide spectrum light the estimation
     might be far off because the optical density (hence strong signal) is much lower in the
     crosstalk spectrum between dyes. It is recommended to use a narrow triband filter to
     capture an image with 3 channels. Alternatively one can also use narrow band LEDs to
     illuminate the film for these shots.
    """
    if args.crosstalk_r_coefs and args.crosstalk_g_coefs and args.crosstalk_b_coefs:
        print("Using supplied crosstalk coefficients.")
        print(args.crosstalk_r_coefs)
        print(args.crosstalk_g_coefs)
        print(args.crosstalk_b_coefs)
        return (np.array([float(x) for x in args.crosstalk_r_coefs.split()]),
                np.array([float(x) for x in args.crosstalk_g_coefs.split()]),
                np.array([float(x) for x in args.crosstalk_b_coefs.split()]))

    def estimate_coef(channel):
        # Intercept must be true because the black frame is already subtracted and we don't
        # want to add constant to the ICC matrix.
        linear_regression = LinearRegression(
            fit_intercept=args.fit_intercept, copy_X=True)
        x = df[['r', 'g', 'b']]
        y = df[channel]
        reg = linear_regression.fit(x, y)
        if args.debug:
            print("R2 score for estimating %s: " % channel, reg.score(x, y))
            print("Intercept for debugging:", reg.intercept_)
        return (reg.coef_, reg.intercept_)

    r_coef, r_intercept = estimate_coef('refR')
    g_coef, g_intercept = estimate_coef('refG')
    b_coef, b_intercept = estimate_coef('refB')

    # Normalize the crosstalk correction matrix by the primary signal. The relative weight
    # between channels is not important as we will apply curve individually to each channel
    # to have the gray scale patches align. The constants don't matter also, assuming it is
    # small.
    # TODO: Print a warning if the constant is larger than a threshold.
    return (r_coef / r_coef[0], g_coef / g_coef[1], b_coef / b_coef[2])


def debug_inversion_curves(refR, refG, refB, luminance):
    # Note: These curve fitting is for mathematically evaluation only.
    # We use this mathematically model convert from electronic signal to linear RGB values
    # that is proportional to luminance:
    #
    # linear_val = a * (b / signal) ^ c
    #
    # Where
    # a is a constant scale factor
    # b is a reference signal that is strong enough
    # c is the gamma
    def func(x, a, b, c):
        return a * ((b / x) ** c)
    r_opt, _ = curve_fit(func, refR, luminance, p0=[0.1, refR['gs23'], 0.5])
    g_opt, _ = curve_fit(func, refG, luminance, p0=[0.1, refG['gs23'], 0.5])
    b_opt, _ = curve_fit(func, refB, luminance, p0=[0.1, refB['gs23'], 0.5])
    with np.printoptions(precision=5):
        print("Best fitted parameters for modeling electronic signal to RGB values:")
        print("linearR = %f * (%f / r) ^ %f" % tuple(r_opt.tolist()))
        print("linearG = %f * (%f / g) ^ %f" % tuple(g_opt.tolist()))
        print("linearB = %f * (%f / b) ^ %f" % tuple(b_opt.tolist()))


def estimate_trc_curves(corrected_gs_rgb, luminance):
    """
    Estimate the TRC curves from the RGB values to the linear luminance values.

    Args:
      rgb: Three vectors of RGB values.
      luminance: Linear luminance values measured from an instrument.

    Returns:
      A tuple of TRC curves for the RGB channels such that the incoming RGB values are
      linearalized according to the luminance vector.

    ALGORITHM
     We assume that gs patches in the training data are supposed to be well balanced.
     Then we build tone curves using interpolation to align the patches such that the
     crosstalk corrected RGB values from these patches are perfectly aligned. We have
     to pay extra attention such that the extrapolated values outside of the known values
     are monotonic, hence the Pchip interpolator is selected.

     The limitation of this method is the balanced gradation is only guranteed for the
     luminanced range of the training data, which is about 7 stops (i.e. 2^7:1 = 128:1
     contrast) limited by the reflective IT8 target. It might be possible to combine
     data from exposures taken at 1 stop apart, but this assumes the camera had very
     accurate shutter speed between stops. More research is needed to cover 10 stops
     of gradation.
    """

    # Build interpolators from the training data.
    train_gs_luminance = np.append(np.insert(luminance.tolist(), 0, 1), 0)
    corrected_gs_r, corrected_gs_g, corrected_gs_b = corrected_gs_rgb
    corrected_gs_r = np.append(np.insert(corrected_gs_r, 0, 0), 65535)
    corrected_gs_g = np.append(np.insert(corrected_gs_g, 0, 0), 65535)
    corrected_gs_b = np.append(np.insert(corrected_gs_b, 0, 0), 65535)

    # Some film will reach Dmin and hence the gs values are not strictly increasing.
    # Fake it so we can still interpolate.
    def fix_strictly_increasing(d):
        for i in range(0, len(d) - 1):
            if d[i+1] <= d[i]:
                d[i+1] = d[i] + 1
    fix_strictly_increasing(corrected_gs_r)
    fix_strictly_increasing(corrected_gs_g)
    fix_strictly_increasing(corrected_gs_b)

    interp_r = interpolate.PchipInterpolator(
        corrected_gs_r, train_gs_luminance)
    interp_g = interpolate.PchipInterpolator(
        corrected_gs_g, train_gs_luminance)
    interp_b = interpolate.PchipInterpolator(
        corrected_gs_b, train_gs_luminance)

    if args.debug:
        with np.printoptions(precision=8):
            print("GS RGB values after 3x1D LUT.")
            print(np.round(np.array([interp_r(corrected_gs_r), interp_g(
                corrected_gs_g), interp_b(corrected_gs_b)]).transpose(), 8))

    xs = np.linspace(0, 65536, 4096)  # V2 Profile allow up to 4096 points.
    r_curve = interp_r(xs)
    g_curve = interp_g(xs)
    b_curve = interp_b(xs)

    if args.debug:
        # Plot a graph to show the interpolated tone curves.
        plt.plot(corrected_gs_r, train_gs_luminance, 'o', xs, r_curve, '-')
        plt.plot(corrected_gs_g, train_gs_luminance, 'o', xs, g_curve, '-')
        plt.plot(corrected_gs_b, train_gs_luminance, 'o', xs, b_curve, '-')
        plt.show()
    return r_curve, g_curve, b_curve


def write_ti3(filename, positive_rgb=True):
    f = open(filename, 'w+')
    stdout_backup = sys.stdout
    sys.stdout = f

    # Print TI3 file for cLUT profiling using ArgyllCMS.
    print("""
CTI3

DESCRIPTOR "Argyll Calibration Target chart information 3"
ORIGINATOR "Argyll target"
CREATED "Mon Feb  8 00:47:09 2021"
DEVICE_CLASS "INPUT"
COLOR_REP "XYZ_RGB"

NUMBER_OF_FIELDS 7
BEGIN_DATA_FORMAT
SAMPLE_ID XYZ_X XYZ_Y XYZ_Z RGB_R RGB_G RGB_B
END_DATA_FORMAT

NUMBER_OF_SETS 288
BEGIN_DATA
""")
    for index, row in df.iterrows():
        print(
            index,
            row['norm_refX'],
            row['norm_refY'],
            row['norm_refZ'],
            row['pos_r'] if positive_rgb else row['corrected_r'],
            row['pos_g'] if positive_rgb else row['corrected_g'],
            row['pos_b'] if positive_rgb else row['corrected_b'])
    print("END_DATA")
    f.close()
    sys.stdout = stdout_backup


def write_build_prof_header(
        crosstalk_correction_mat,
        test_white_XYZ,
        r_curve,
        g_curve,
        b_curve):
    f = open('build_prof.h', 'w+')
    stdout_backup = sys.stdout
    sys.stdout = f

    # Print matrix.
    print("double crosstalk_correction_mat[] = {")
    mat = crosstalk_correction_mat.transpose().flatten()
    for i in range(0, len(mat)):
        print(" %.15f," % mat[i], end='')
        if (i + 1) % 3 == 0:
            print()
    print("};")

    # Print tone curves.
    def print_curve(name, curve):
        print("float %s[CURVE_POINTS] = {" % name)
        for i in range(0, len(curve)):
            if i == len(curve) - 1:
                print(" %.7f" % curve[i], end='')
            else:
                print(" %.7f," % curve[i], end='')
            if (i + 1) % 16 == 0:
                print()
        print("};\n")

    print("const int CURVE_POINTS = %d;" % len(r_curve))
    print_curve('b_curve', b_curve)
    print_curve('g_curve', g_curve)
    print_curve('r_curve', r_curve)
    f.close()
    sys.stdout = stdout_backup


def write_neg_invert_sh(file_name, crosstalk_correction_mat, clut_profile):
    f = open(file_name, 'w+')
    stdout_backup = sys.stdout
    sys.stdout = f
    with open('neg_invert.sh.template', 'r') as tf:
        print(tf.read() % (tuple(crosstalk_correction_mat.transpose().flatten()) + (clut_profile,)))
    f.close()
    sys.stdout = stdout_backup
    os.chmod(file_name, 0o755)

def write_color_correction_matrix(file_name, crosstalk_correction_mat):
    f = open(file_name, 'w+')
    stdout_backup = sys.stdout
    sys.stdout = f
    flat_cc_mat = crosstalk_correction_mat.transpose().flatten()
    print(' '.join([x.astype(str) for x in flat_cc_mat[0:3]]))
    print(' '.join([x.astype(str) for x in flat_cc_mat[3:6]]))
    print(' '.join([x.astype(str) for x in flat_cc_mat[6:9]]))
    f.close()
    sys.stdout = stdout_backup

def print_neg_process_cmd(file_name, crosstalk_correction_mat, clut_profile):
    print('Use the following command to convert a negative image: ')
    print('%s -r \'%.15f %.15f %.15f\' -g \'%.15f %.15f %.15f\' -b \'%.15f %.15f %.15f\' -p \'%s\'' % 
          ((file_name,) + (tuple(crosstalk_correction_mat.transpose().flatten()) + (clut_profile,))))


def run_chromatic_adaptation_on_ref_XYZ():
    """
    Perform chromatic adaptation of the measured tristimulus values
    into D50. Film balanced at 5500K so we should be exposing the test chart with
    5500K light and also measure the test chart with same color temperature.

    Results are written to global df and scaled to max value of 100.

    Returns:
      XYZ values of testing media white.
    """
    # This is the standard value from ICC specification.
    D50_XYZ = np.array([0.9642, 1, 0.8249])

    unadapted_XYZ = np.array(df[['refX', 'refY', 'refZ']])
    if args.white_x and args.white_y:
        test_white_XYZ = np.array(colour.xyY_to_XYZ(
            [args.white_x, args.white_y, 1]))
        adapted_XYZ = colour.adaptation.chromatic_adaptation(
            unadapted_XYZ, test_white_XYZ, D50_XYZ, method='Von Kries', transform='Bradford')
    else:
        test_white_XYZ = D50_XYZ
        adapted_XYZ = unadapted_XYZ

    if args.debug:
        print('Refernce unadapted XYZ values:')
        print(unadapted_XYZ)
        print('D50 adapted XYZ values using Bradford CAT:')
        print(adapted_XYZ)

    # colprof expects max value to be 100.0.
    norm_XYZ = adapted_XYZ.transpose() / adapted_XYZ.max().max() * 100
    df['norm_refX'], df['norm_refY'], df['norm_refZ'] = norm_XYZ
    return test_white_XYZ


print('Step 3: Build 3D LUT from positive RGB to measured XYZ.')


def compute_positive_rgb_values(
        crosstalk_correction_mat,
        r_curve,
        g_curve,
        b_curve):
    """
    Apply crosstalk correction and TRC curves to produce RGB values for the patches.

    Positive patches are written to pos_r, pos_g and pos_b serieses with max value of 100.
    """
    # Contains all rgb values of the patches of shape 3 x NUM_OF_PATCHES.
    rgb = np.array([df['r'].tolist(), df['g'].tolist(), df['b'].tolist()])

    # Crosstalk correct then clip to [0, 65535].
    corrected_rgb = np.clip(
        np.matmul(
            rgb.transpose(),
            crosstalk_correction_mat),
        0,
        65535)
    corrected_df = (corrected_rgb / 65535 * 100).transpose()
    df['corrected_r'] = pd.Series(corrected_df[0], index=df.index)
    df['corrected_g'] = pd.Series(corrected_df[1], index=df.index)
    df['corrected_b'] = pd.Series(corrected_df[2], index=df.index)

    # Apply curves to convert them positive signals.
    lut = colour.LUT3x1D(np.array([r_curve, g_curve, b_curve]).transpose())
    positive_rgb = lut.apply(corrected_rgb / 65535)  # LUT takes range [0, 1].

    # Add back the positive RGB values and the corresponding normalized XYZ values
    # into |df|. These values need to be normalized to max of 100 which is what
    # ArgyllCMS likes.
    positive_df = (positive_rgb * 100).transpose()
    df['pos_r'] = pd.Series(positive_df[0], index=df.index)
    df['pos_g'] = pd.Series(positive_df[1], index=df.index)
    df['pos_b'] = pd.Series(positive_df[2], index=df.index)


def run_colprof_clut(ti3_name):
    """
    Run colprof to produce a profile.
    """
    # This set of arguments forces colprof to generate profile with linear input and output curves.
    # Since we have already applied a curve to the input to get GS patches to be linear the input
    # curves would have been generated by ArgyllCMS only make small adjustments and otherwise does
    # not improve accuracy much so save the complexity.
    subprocess.run(['colprof', '-v',
                    '-ax',  # Generate XYZ cLUT.
                    '-qu',  # Gives 45 grid points instead of 33 of -qh.
                            # -qu crashes Capture 1 with ICC V4 mpet transform.
                    '-kz',  # TODO: Need to experiment with this option.
                    '-u',  # TODO: Need to experiment with this option.
                    '-bn',  # No need of B2A profiles.
                    # Forces linear input and output curves.
                    '-ni', '-np', '-no',
                    ti3_name], check=True)
    os.rename(ti3_name + '.icc', 'icc_out/%s_clut.icc' % ti3_name)
    return 'icc_out/%s_clut.icc' % ti3_name


def run_colprof_matrix(ti3_name):
    """
    Run colprof to produce a profile.

    This uses a simple -am option instead of -as which estimates the shaper curves.
    Using -as has marginal improvements with accuracy (avg error 7.46 vs 6.97 for Ektar100).
    The improvements are so small compared to the absolute error so don't bother adjusting
    the negative curves with the estimated shaper curves.
    """
    subprocess.run(['colprof', '-v',
                    '-am',  # Generate XYZ Matrix.
                    '-qh',
                    '-kz',  # TODO: Need to experiment with this option.
                    '-u',  # TODO: Need to experiment with this option.
                    '-bn',  # No need of B2A profiles.
                    # Forces linear input and output curves.
                    '-ni', '-np', '-no',
                    ti3_name], check=True)
    os.rename(ti3_name + '.icc', 'icc_out/%s_matrix.icc' % ti3_name)
    return 'icc_out/%s_matrix.icc' % ti3_name


def run_make_icc(
        crosstalk_correction_mat,
        test_white_XYZ,
        r_curve,
        g_curve,
        b_curve,
        film_name,
        clut_icc,
        mat_icc):
    write_build_prof_header(
        crosstalk_correction_mat,
        test_white_XYZ,
        r_curve,
        g_curve,
        b_curve)
    subprocess.run(['make', 'make_icc'], check=True)
    subprocess.run(['bin_out/make_icc', film_name, clut_icc, mat_icc], check=True)


def run_prof_check(clut_icc):
    subprocess.run(['profcheck', '-v3', 'check_prof.ti3', clut_icc], check=True)


def main():
    # Prepare the refernce XYZ values to be used for running colprof.
    test_white_XYZ = run_chromatic_adaptation_on_ref_XYZ()

    print("Step 1: Estimate the cross-talk correction matrix.")
    # TODO: The coefficients cannot be positive other than the primary signal,
    # i.e. the diagonal.
    r_coef, g_coef, b_coef = estimate_crosstalk_correction_coefficients()
    print('R Coefficients: ', r_coef)
    print('G Coefficients: ', g_coef)
    print('B Coefficients: ', b_coef)

    # Crosstalk correct matrix is always applied on the right hand side and is
    # thus transposed.
    crosstalk_correction_mat = np.array([r_coef, g_coef, b_coef]).transpose()

    if args.prescale_coef:
        gs_cell = args.prescale_coef
        prod = crosstalk_correction_mat.transpose().dot(
            np.array([df['r'][gs_cell], df['g'][gs_cell], df['b'][gs_cell]]))
        crosstalk_correction_mat = np.array([r_coef, g_coef * prod[0] / prod[1], b_coef * prod[0] / prod[2]]).transpose()
        print("Scaled crosstalk correction matrix to white balance patch: %s." % args.prescale_coef)
        print(crosstalk_correction_mat.transpose())

    print("Step 2: Estimate the TRC from cross-talk corrected RGB values.")
    gs = df.loc[['gs' + str(x) for x in range(0, 24)]]
    gs_rgb = np.array([gs['r'].tolist(), gs['g'].tolist(), gs['b'].tolist()])
    corrected_gs_rgb = np.matmul(
        gs_rgb.transpose(),
        crosstalk_correction_mat).transpose()
    print("Max GS element after color correction: %f" % corrected_gs_rgb.max())
    crosstalk_correction_mat *= args.darkest_patch_scaling / corrected_gs_rgb.max()
    corrected_gs_rgb = np.matmul(
        gs_rgb.transpose(),
        crosstalk_correction_mat).transpose()

    if args.debug:
        print("GS RGB values before correction.")
        print(gs_rgb.transpose())
        print("GS RGB values after correction.")
        print(corrected_gs_rgb.transpose())

    luminance = gs['refY'] / \
        (gs['refY'].max() / args.whitest_patch_scaling)
    if args.debug:
        debug_inversion_curves(gs['refR'], gs['refG'], gs['refB'], luminance)
    r_curve, g_curve, b_curve = estimate_trc_curves(
        corrected_gs_rgb, luminance)

    # Compute the pos_r, pos_g and pos_b series using the estimated
    # corrections.
    compute_positive_rgb_values(
        crosstalk_correction_mat,
        r_curve,
        g_curve,
        b_curve)

    print("Step 3: Run colprof to produce a profile.")
    df.to_csv('build_prof_diag.csv')
    write_ti3('build_prof.ti3')
    clut_prof = run_colprof_clut('build_prof')
    matrix_prof = run_colprof_matrix('build_prof')

    print('Step 4: Building ICC profiles using make_icc.')
    run_make_icc(
        crosstalk_correction_mat,
        test_white_XYZ,
        r_curve,
        g_curve,
        b_curve,
        args.film_name,
        clut_prof,
        matrix_prof)
    out_clut_prof = 'icc_out/%s cLUT.icc' % args.film_name
    out_matrix_prof = 'icc_out/%s Matrix.icc' % args.film_name

    write_ti3('check_prof.ti3', positive_rgb=False)
    write_color_correction_matrix(
        'icc_out/%s CC Matrix.txt' % args.film_name,
        crosstalk_correction_mat)
    run_prof_check(out_clut_prof)

    if args.install_dir:
        install_dir = os.path.realpath(args.install_dir)
        os.makedirs(install_dir, exist_ok=True)
        shutil.copy(out_clut_prof, install_dir)
        shutil.copy('bin_out/neg_process', install_dir)
        print_neg_process_cmd(os.path.join(install_dir, 'neg_process'),
                              crosstalk_correction_mat,
                              os.path.join(install_dir, out_clut_prof.split('/')[-1]))


if __name__ == "__main__":
    main()
