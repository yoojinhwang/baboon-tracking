import cv2
import numpy as np
import math
import cmath
import skimage
from skimage.feature import blob_log, blob_dog
import time
import multiprocessing
from skimage import img_as_ubyte

from collections import deque
from LoG_detection import *

from config import *

import pickle
def remove_noise(foreground_mask):
    '''
    Uses OpenCV morphological transformations to make blobs more defined
    Returns a foreground mask with more defined blobs
    '''
    erosion_kernel = np.ones(EROSION_KERNEL, np.uint8)
    dilation_kernel = np.ones(DILATION_KERNEL, np.uint8)
    gaussian_kernel = np.ones(GAUSSIAN_KERNEL, np.uint8)
    open_kernel =  np.ones((2,2),np.uint8)
    close_kernel =  np.ones((3,3),np.uint8)

    #foreground_mask = cv2.erode(foreground_mask, erosion_kernel, iterations = EROSION_ITERATIONS)
    #foreground_mask = cv2.dilate(foreground_mask, dilation_kernel, iterations = 5)
    # Adding a Gaussian Blur
    #foreground_mask = cv2.GaussianBlur(foreground_mask, (3,3), 0)
    #foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN,open_kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE,close_kernel, iterations = 3)
    return foreground_mask

def detect_blobs(foreground_mask, rgb_frame, orig_frame=None):

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(BLOB_DET_PARAMS)

    # DETECT BLOBS

    #invert image (blob detection only works with white background)
    foreground_mask = cv2.bitwise_not(foreground_mask)

    # apply blur
    foreground_mask = cv2.blur(foreground_mask, BLUR_KERNEL)

    # detect
    keypoints = detector.detect(foreground_mask)

    print("keypoints: ", keypoints)

    # draw detected blobs
    frame_with_blobs = cv2.drawKeypoints(rgb_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 #creating another image with the keypoints drawn onto the image that it is finding keypoints on
    mask_with_blobs = cv2.drawKeypoints(foreground_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    orig_with_blobs = None
    if(orig_frame is not None):
        orig_with_blobs = cv2.drawKeypoints(orig_frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return frame_with_blobs, mask_with_blobs, orig_with_blobs


def detect_blobs_LoG(foreground_mask, rgb_frame, orig_frame=None):
    """
    Using the skimage LoG function
    Very slow and unuseable in real time
    Possible can serve as a point of reference for frame by frame analysis to other LoG algorithms
    """
    keypoints = []
    start = time.clock()
    blobs_log = blob_log_mod(foreground_mask)
    print("blob_log time:   "+str(time.clock() - start))
    #get the radius of the circle
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
    #create keypoints from the list
    start2 = time.clock()
    for blob in blobs_log:
        y = blob[0]
        x = blob[1]
        r = blob[2]
        keypoints.append(cv2.KeyPoint(x=x, y=y,_size=r))
    print("forloop time:   "+str(time.clock() - start2))
    print("keypoints: ", keypoints)

    start2 = time.clock()
    # draw detected blobs
    frame_with_blobs = cv2.drawKeypoints(rgb_frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 #creating another image with the keypoints drawn onto the image that it is finding keypoints on
    mask_with_blobs = cv2.drawKeypoints(foreground_mask, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("drawkeypoints time:   "+str(time.clock() - start2))

    orig_with_blobs = None
    if(orig_frame is not None):
        orig_with_blobs = cv2.drawKeypoints(orig_frame, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return frame_with_blobs, mask_with_blobs, orig_with_blobs

def detect_blobs_LoG2(foreground_mask, rgb_frame, orig_frame=None, LoGsize=20, pickle=False,frame_number=None):
    """
    LoG algorithm based on GUI_Blob_Tracker github repo
    """
    foreground_withblob = do_LoG(foreground_mask, LoGsize)
    #convert back into int
    foreground_withblob2  = cv2.normalize(foreground_withblob, None, 0,255,cv2.NORM_MINMAX)
    foreground_withblob2 = foreground_withblob2.astype(np.uint8)
    # Blob Detection
    detector = cv2.SimpleBlobDetector_create(BLOB_DET_PARAMS)

    keypoints = detector.detect(foreground_withblob2)

    im_with_keypoints = cv2.drawKeypoints(foreground_withblob2, keypoints,
                                          np.array([]), (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    mask_with_blobs = cv2.drawKeypoints(rgb_frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    orig_with_blobs = None
    if(orig_frame is not None):
        orig_with_blobs = cv2.drawKeypoints(orig_frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #if you decide to save txt files of keypoint information
    """
    if(pickle==True):
        index = []
        #get all the needed information from keypoints
        for point in keypoints:
            temp = ["baboon", str(point.response), str(point.pt[0] - (point.size/2)), str(point.pt[1] + (point.size/2)) , str(point.size), str(point.size)]
            index.append(temp)
        #set the name of the file that you will be saving to. It is set to one frame per file
        pickleTXT = "hypotheses_txt/frame"+str(frame_number)+".txt"
        #write info to file
        with open(pickleTXT, "a+") as txt_file:
            for line in index:
                txt_file.write(" ".join(line) + "\n")
    """
    if(pickle==True):
        index = []
        #get all the needed information from keypoints
        for point in keypoints:
            temp = [str((point.pt[0]- (point.size/2), point.pt[1]+ (point.size/2))) ,  str((point.pt[0]+(point.size/2), point.pt[1]-(point.size/2)))]
            index.append(temp)
        #set the name of the file that you will be saving to. It is set to one frame per file
        pickleTXT = "/home/alli/Desktop/StuffToPutinBaboonGoogleDrive/hypotheses_drawing/frame"+str(frame_number)+".txt"
        #write info to file
        with open(pickleTXT, "a+") as txt_file:
            for line in index:
                txt_file.write(" ".join(line) + "\n")


    return im_with_keypoints, mask_with_blobs, orig_with_blobs
