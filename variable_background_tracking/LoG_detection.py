import cv2
import numpy as np
import math
from numpy import dot, ma  # masked arrays
import time
import sys


"""
def LoG(noiseless_mask):
    ''' Uses cv2's Laplacian on a removed noise mask
    '''
    #apply laplacian
    dst = cv2.Laplacian(noiseless_mask, ddepth, ksize=kernel_size)
    #convert to uint8
    abs_dst = cv2.convertScaleAbs(dst)
"""

def create_logkernel(size, std):
    """
    Copied from https://github.com/dghy/GUI_Blob_Tracker

    Creates log kernel to be convolved with using the LoG equation:
    LoG(x,y) =
(1/(pi*sigma^4)) * (1 - (x^2+y^2)/(sigma^2)) * (e ^ (- (x^2 + y^2) / 2sigma^2)

    The size of the kernel, which determines the accuracy (i think?), is set
    The standard deviation, which controls how big the blobs detected are, is set

    """
    x = np.linspace(-size, size, 2 * size + 1)
    y = np.linspace(-size, size, 2 * size + 1)
    x, y = np.meshgrid(x, y)
    arg = -(x ** 2 + y ** 2) / (2 * std ** 2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h / h.sum() if h.sum() != 0 else h
    h1 = h * (x ** 2 + y ** 2 - 2 * std ** 2) / (std ** 4)
    return h1 - h1.mean()

def do_LoG(frame, LoGsize=20, std = None):
    """
    Also comes from Blob Tracker
    """
    #Setting the standard deviation of the LoG
    if std == None:
        std = int(LoGsize * 0.8)
    LoG_kernel = create_logkernel(LoGsize,
                                 std)
    #Cross correlating the filter with the image
    frame = cv2.filter2D(frame, cv2.CV_32F, LoG_kernel)
    frame *= 255
    # Tracker says that this removes floats near 0, not sure if this is needed, since I convert to int
    frame[frame < 0] = 0

    return frame
