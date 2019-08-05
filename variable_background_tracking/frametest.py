"""
These are the tests described in Alli's notes, 7/30 and the output to which is
in the Blob Detection Images folder on the Google Drive.

Its purpose is to take in one frame of the outpy_blur2x2 mask, and do
a certain number of morphological operators on it. The output images are written to
the out folder
"""
import cv2
import numpy as np
import math
import cmath
import skimage
from skimage.feature import blob_log
import time
import multiprocessing
import sys

from collections import deque
from registration import register

from foreground_extraction import *
from config import *
from blob_detection import *

#same sized kernels
erosion_kernel = np.ones((3,3), np.uint8)
dilation_kernel = np.ones((3,3), np.uint8)
gaussian_kernel = np.ones((3,3), np.uint8)
open_kernel =  np.ones((3,3),np.uint8)
close_kernel =  np.ones((3,3),np.uint8)

testimg = cv2.imread("../data/2x2blur_15sec.png")

gray = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
foreground_mask = gray
"""
#Test 1:
open_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN,open_kernel)
close_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE,close_kernel)
erosion_mask = cv2.erode(foreground_mask, erosion_kernel, iterations = 1)
dilation_mask = cv2.dilate(foreground_mask, dilation_kernel, iterations = 1)

cv2.imwrite('../out/open_mask.png',open_mask)
cv2.imwrite('../out/close_mask.png',close_mask)
cv2.imwrite('../out/erosion_mask.png',erosion_mask)
cv2.imwrite('../out/dilation_mask.png',dilation_mask)
"""

#Test 2

for i in range(4):
    close_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE,close_kernel, iterations = i)
    dilation_mask = cv2.dilate(foreground_mask, dilation_kernel, iterations = i)
    cv2.imwrite("../out/closemask {0}.png".format(i),close_mask)
    cv2.imwrite("../out/dilationmask {0}.png".format(i),dilation_mask)
