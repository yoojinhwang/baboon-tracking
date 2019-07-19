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

#adding image package to import images
from matplotlib.image import imread
import matplotlib.pyplot as plt

#functions
def prep(mask_image):
    """
    Optimizing the image before the LoG function
    """
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    moving_foreground = gray
    moving_foreground = remove_noise(moving_foreground)
    return moving_foreground

def process(init_image):
    mask = prep(init_image)
    frame, blob, origin = detect_blobs_LoG(mask, init_image)
    plt.imshow(frame)
    plt.show()
    plt.imshow(blob)
    plt.show()
#import pictures from data folder
#and run LoG
image1 = imread("../data/image-001.jpeg");
process(image1)
image2 = imread("../data/image-002.jpeg");
process(image2)
image3 = imread("../data/image-003.jpeg");
process(image3)
image4 = imread("../data/image-004.jpeg");
process(image4)
image5 = imread("../data/image-005.jpeg");
process(image5)
