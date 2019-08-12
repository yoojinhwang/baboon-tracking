import cv2
import numpy as np
import math
import cmath
import skimage
import time
import multiprocessing
import sys

from collections import deque
from registration import register

from foreground_extraction import *
from blob_detection import *
from LoG_detection import *
from config import *

def main():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(INPUT_MASK)
    cap_orig = cv2.VideoCapture(INPUT_VIDEO)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        sys.exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(OUTPUT_MASK_BLOB_DETECTION, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    if (cap_orig.isOpened() == False):
        print("Can't open input.mp4")
        sys.exit()

    # Process original video if provided
    if (cap_orig.isOpened() and int(cap_orig.get(3) == frame_width and cap_orig.get(4) == frame_height)):
        using_orig = True

    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    frame_num = 1 #used for setting txt output file name in detect_blobs_LoG2
    start = time.clock()
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame

        ret, frame = cap.read()
        orig_ret, orig_frame = cap_orig.read()

        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            moving_foreground = gray

            moving_foreground = remove_noise(moving_foreground)
            #frame_with_detected_blobs, mask_blobs, origin = detect_blobs_LoG(moving_foreground, frame)
            moving_foreground, frame_with_detected_blobs, orig_blobs = detect_blobs_LoG2(moving_foreground, frame, orig_frame=orig_frame, pickle=True, frame_number=frame_num)
            #moving_foreground = do_LoG(moving_foreground)
            # Display the resulting frame

            cv2.imshow('moving_foreground', cv2.resize(moving_foreground, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
            #if(using_orig and orig_blobs is not None):
            #    cv2.imshow('orig', cv2.resize(orig_blobs, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
            #    orig_blobs = cv2.resize(orig_blobs, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                #out.write(cv2.cvtColor(orig_blobs, cv2.COLOR_RGB2BGR))
            cv2.imshow('detected_blobs', cv2.resize(frame_with_detected_blobs, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))

            #display the frame that keypoints are being found from as well as keypoints detected
            #cv2.imshow('detected_blobs_mask', cv2.resize(mask_blobs, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))

            #making sure that the frame is the right size
            frame_with_detected_blobs = cv2.resize(frame_with_detected_blobs, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            #out.write(cv2.cvtColor(frame_with_detected_blobs, cv2.COLOR_RGB2BGR))

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            frame_num += 1 #increasing frame number for txt file name in detect_blobs_LoG2

        # Break the loop
        else:
            break

    # When everything done, release the video capture object

    cap.release()
    out.release()
    print("finished writing")

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
