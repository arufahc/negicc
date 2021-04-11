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
import sys
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
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src", help="Source readings file.")
parser.add_argument("--fit_intercept", help="Whether to allow intercept in linear interpolation for crosstalk correction.", default=False)
args = parser.parse_args()

df = pd.read_csv(args.src, index_col="patch")

print("Peak of a few data rows in source file:", args.src)
print(df.head())
print(df.tail())

print("Step 1: Estimate the cross-talk correction matrix.")

# ALGORITHM
#
# We assume there is a linear relationship between the R, G, B values with the reference
# values. This assumption is generally true because the camera R value comes from the
# signals captured in the G and B spectrum, and so on for G and B vaules.
# This is particularly true if the images are captured via a narrow band filter from a
# negative IT8 or from a positive IT8 target.
# If the images are from a negative IT8 target with narrowband light the estimation
# might be far off because the optical density (hence strong signal) is much lower in the
# crosstalk spectrum between dyes. It is recommended to use a narrow triband filter to
# capture an image with 3 channels.
def estimate_coef(channel):
    # Intercept must be true because the black frame is already subtracted and we don't
    # want to add constant to the ICC matrix.
    linear_regression = LinearRegression(fit_intercept=args.fit_intercept, copy_X=True)
    x = df[['r', 'g', 'b']]
    y = df[channel]
    reg = linear_regression.fit(x, y)
    print("R2 score for estimating %s: " % channel, reg.score(x, y))
    print("Intercept for debugging:", reg.intercept_)
    return reg.coef_

r_coef = estimate_coef('refR')
g_coef = estimate_coef('refG')
b_coef = estimate_coef('refB')

# Normalize the crosstalk correction matrix by the primary signal. The relative weight
# between channels is not important as we will apply curve individually to each channel
# to have the gray scale patches align.
r_coef /= r_coef[0]
g_coef /= g_coef[1]
b_coef /= b_coef[2]
print('R Coefficients: ', r_coef)
print('G Coefficients: ', g_coef)
print('B Coefficients: ', b_coef)

print("Step 2: Estimate the TRC from cross-talk corrected RGB values to linear RGB values portional to luminance.")

# ALGORITHM
#
# We assume that gs patches in the training data are supposed to be well balanced.
# Then we build tone curves using interpolation to align the patches such that the
# crosstalk corrected RGB values from these patches are perfectly aligned. We have
# to pay extra attention such that the extrapolated values outside of the known values
# are monotonic, hence the Pchip interpolator is selected.
#
# The limitation of this method is the balanced gradation is only guranteed for the
# luminanced range of the training data, which is about 7 stops (i.e. 2^7:1 = 128:1
# contrast) limited by the reflective IT8 target. It might be possible to combine
# data from exposures taken at 1 stop apart, but this assumes the camera had very
# accurate shutter speed between stops. More research is needed to cover 10 stops
# of gradation.

gs = df.loc[['gs' + str(x) for x in range(0,24)]]
gs_rgb = np.array([gs['r'].tolist(), gs['g'].tolist(), gs['b'].tolist()])
print("GS RGB values before correction.")
print(gs_rgb.transpose())

corrected_gs_rgb = np.matmul(gs_rgb.transpose(), [r_coef, g_coef, b_coef]).transpose()
crosstalk_correction_mat = [r_coef, g_coef, b_coef]
print("GS RGB values after correction.")
print(corrected_gs_rgb.transpose())

luminance = gs['refY']
luminance /= luminance.max() / 0.8

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
r_opt, _ = curve_fit(func, gs['refR'], luminance, p0=[0.1, gs['refR']['gs23'], 0.5])
g_opt, _ = curve_fit(func, gs['refG'], luminance, p0=[0.1, gs['refG']['gs23'], 0.5])
b_opt, _ = curve_fit(func, gs['refB'], luminance, p0=[0.1, gs['refB']['gs23'], 0.5])
with np.printoptions(precision=5):
    print("Best fitted parameters for modeling electronic signal to RGB values:")
    print("linearR = %f * (%f / r) ^ %f" % tuple(r_opt.tolist()))
    print("linearG = %f * (%f / g) ^ %f" % tuple(g_opt.tolist()))
    print("linearB = %f * (%f / b) ^ %f" % tuple(b_opt.tolist()))

# Build interpolators from the trained 
train_gs_luminance = np.append(np.insert(luminance.tolist(), 0, 1), 0)
train_gs_r, train_gs_g, train_gs_b = corrected_gs_rgb
train_gs_r = np.append(np.insert(train_gs_r, 0, 0), 65535)
train_gs_g = np.append(np.insert(train_gs_g, 0, 0), 65535)
train_gs_b = np.append(np.insert(train_gs_b, 0, 0), 65535)
interp_r = interpolate.PchipInterpolator(train_gs_r, train_gs_luminance)
interp_g = interpolate.PchipInterpolator(train_gs_g, train_gs_luminance)
interp_b = interpolate.PchipInterpolator(train_gs_b, train_gs_luminance)

with np.printoptions(precision=8):
    print("GS RGB values after 3x1D LUT.")
    print(np.round(np.array([interp_r(train_gs_r), interp_g(train_gs_g), interp_b(train_gs_b)]).transpose(), 8))

# Plot a graph to show the interpolated tone curves.
xs = np.linspace(0, 65536, 4096)
r_curve = interp_r(xs)
g_curve = interp_g(xs)
b_curve = interp_b(xs)
plt.plot(train_gs_r, train_gs_luminance, 'o', xs, r_curve, '-')
plt.plot(train_gs_g, train_gs_luminance, 'o', xs, g_curve, '-')
plt.plot(train_gs_b, train_gs_luminance, 'o', xs, b_curve, '-')
plt.show()

print('Step 3: Build 3D LUT from positive RGB to measured XYZ.')

# ALGORITHM
#
# Apply in order crosstalk correction and then curve correction to all patches
# of the training data. This converts then to positive images. Then each positive
# RGB values can be mapped to known XYZ values using a cLUT.
#
# The building of cLUT is done by ArgyllCMS.

# Contains all rgb values of the patches of shape 3 x NUM_OF_PATCHES.
rgb = np.array([df['r'].tolist(), df['g'].tolist(), df['b'].tolist()])

# Crosstalk correct then clip to [0, 65535].
corrected_rgb = np.clip(np.matmul(rgb.transpose(), crosstalk_correction_mat), 0, 65535)

# Apply curves to convert them positive signals.
lut = colour.LUT3x1D(np.array([r_curve, g_curve, b_curve]).transpose())
positive_rgb = lut.apply(corrected_rgb / 65535) # LUT takes range [0, 1].

# Add back the positive RGB values and the corresponding normalized XYZ values
# into |df|. These values need to be normalized to max of 100 which is what
# ArgyllCMS likes.
positive_df = (positive_rgb * 100).transpose()
df['pos_r'] = pd.Series(positive_df[0], index=df.index)
df['pos_g'] = pd.Series(positive_df[1], index=df.index)
df['pos_b'] = pd.Series(positive_df[2], index=df.index)
max_XYZ = df[['refX', 'refY', 'refZ']].max().max()
df['norm_refX'] = df['refX'] / max_XYZ * 100
df['norm_refY'] = df['refY'] / max_XYZ * 100
df['norm_refZ'] = df['refZ'] / max_XYZ * 100
df.to_csv('build_prof_diag.csv')

print('Step 4: Writing build_prof.ti3 to be used by ArgyllCMS for building cLUT.')
print('Example command for using colprof to generate cLUT profile:')
print('    colprof -v -ax -qh -kz -u -bn -ni -np -no build_prof')

f = open('build_prof.ti3', 'w+')
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
    print(index, row['norm_refX'], row['norm_refY'], row['norm_refZ'], row['pos_r'], row['pos_g'], row['pos_b'])
print("END_DATA")
f.close()
sys.stdout = stdout_backup

print('Step 5: Writing build_prof.h to be used by make_icc to create final profile.')

f = open('neg_correct.sh', 'w+')
sys.stdout = f
print("""
dcraw -v -4 -o 0 -h -T -W "$1"
convert "${1/.NEF/.tiff}" -set colorspace RGB -color-matrix '%f %f %f %f %f %f %f %f %f' "${1/.NEF/_corrected.tiff}"
convert "${1/.NEF/_corrected.tiff}" -set profile icc_out/std_negative_clut.icc "${1/.NEF/_corrected_clut_icc.tiff}" 
convert "${1/.NEF/_corrected.tiff}" -set profile icc_out/std_negative.icc "${1/.NEF/_corrected_mat_icc.tiff}" 
""" % tuple(r_coef.tolist() + g_coef.tolist() + b_coef.tolist()))
f.close()

f = open('build_prof.h', 'w+')
sys.stdout = f

# Print matrix.
print("double crosstalk_correction_mat[] = {")
mat = np.array([r_coef.tolist(), g_coef.tolist(), b_coef.tolist()]).transpose().flatten()
for i in range(0, len(mat)):
    print(" %.15f," % mat[i], end='')
    if (i+1) % 3 == 0:
        print()
print("};");

# Print tone curves.
def print_curve(name, curve):
    print("float %s[%d] = {" % (name, len(curve)))
    for i in range(0, len(curve)):
        if i == len(curve) - 1:
            print(" %.7f" % curve[i], end='')
        else:
            print(" %.7f," % curve[i], end='')
        if (i+1) % 16 == 0:
            print()
    print("};\n")
print_curve('b_curve', interp_b(xs))
print_curve('g_curve', interp_g(xs))
print_curve('r_curve', interp_r(xs))
f.close()
