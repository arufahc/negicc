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
import matplotlib.pyplot as plt
import sys
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--vstep", help="Vertical step for moving to next row.", type=int, default=53)
parser.add_argument("--hstep", help="Horizontal step for moving to next column.", type=int, default=54)
parser.add_argument("--vbase", help="Vertical height base", type=int, default=870)
parser.add_argument("--hbase", help="Horizontal width base", type=int, default=1300)
parser.add_argument("--box_size", help="Box size", type=int, default=18)
parser.add_argument("--a1_x", help="X coordinate of A1 patch", type=int, default=77)
parser.add_argument("--a1_y", help="Y coordinate of A1 patch", type=int, default=79)
parser.add_argument("--gs0_x", help="X coordinate of GS0 patch", type=int, default=23)
parser.add_argument("--gs0_y", help="Y coordinate of GS1 patch", type=int, default=800)
parser.add_argument("--img", help="Source image", type=str)
parser.add_argument("--multi", help="Read multi-page TIFF", action='store_true')
parser.add_argument("--outfile", help="Output file", type=str)
args = parser.parse_args()

VSTEP = args.vstep
HSTEP = args.hstep
VBASE = args.vbase
HBASE = args.hbase
BOX_DIMENSION = (args.box_size/HBASE, args.box_size/VBASE)

boxes = {
    "a1": ((args.a1_x/HBASE, args.a1_y/VBASE), BOX_DIMENSION)
}

def add_horizontal_boxes(row, start=2, end=23):
    for j in range(start, end):
        left = boxes[row + str(j-1)]
        boxes[row + str(j)] = (left[0][0] + HSTEP /
                               HBASE, left[0][1]), BOX_DIMENSION


add_horizontal_boxes('a')
for i in range(1, 12):
    row = chr(ord('a') + i)
    last_row = chr(ord('a') + (i - 1))
    boxes[row + '1'] = (np.add(boxes[last_row + '1'][0],
                        (0, VSTEP/VBASE)), boxes[last_row + '1'][1])
    add_horizontal_boxes(row)

boxes["gs0"] = (args.gs0_x/HBASE, args.gs0_y/VBASE), BOX_DIMENSION
add_horizontal_boxes('gs', 1, 24)

if args.multi:
    ret, imgs = cv2.imreadmulti(args.img, [], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if not ret:
        sys.exit("Cannot read image file.")
        
    img = cv2.merge((imgs[2], imgs[1], imgs[0]))
else:
    img = cv2.imread(args.img, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape
print("Read image %d x %d" % (width, height))

def to_int(coord):
    return int(coord[0] * width), int(coord[1] * height)

def hist_median(hist, pixels):
    sum = 0
    for i in range(0, 65536):
        sum += hist[i]
        if sum >= pixels / 2:
            return i
    return None

def img_median(img):
    pixels = img.shape[0] * img.shape[1]
    hist_r = cv2.calcHist([img], [0] , None, [65536], (0, 65535))
    hist_g = cv2.calcHist([img], [1] , None, [65536], (0, 65535))
    hist_b = cv2.calcHist([img], [2] , None, [65536], (0, 65535))
    return hist_median(hist_r, pixels), hist_median(hist_g, pixels), hist_median(hist_b, pixels)

# This image is just for diagnosis and verifying.
img_scaled = cv2.normalize(img, dst=None, alpha=0,
                           beta=255, norm_type=cv2.NORM_MINMAX)

outf = None
if args.outfile:
    outf = open(args.outfile, "w+")
    sys.stdout = outf

print('patch', 'r', 'g', 'b')
for patch, b in boxes.items():
    top_left = to_int(b[0])
    bottom_right = to_int(np.add(b[0], b[1]))
    cv2.rectangle(img_scaled, top_left, bottom_right, (255, 0, 0), 1)

    patch_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    r, g, b = img_median(patch_img)
    print(patch, r, g, b)
        
if outf:
    outf.close()
plt.imshow(img_scaled)
plt.show()
