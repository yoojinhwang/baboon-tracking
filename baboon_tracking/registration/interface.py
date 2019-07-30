import cv2
import numpy as np
import math
import cmath
import multiprocessing

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

class Registration_Strategy():
    def __init__(self, config):
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15
        self.config = config

    def _shift_frame(self, frames):#, frame, previous_frame):
        '''
        Takes in transformation matrix; does homography transformation to register/align two frames
        '''
        frame = frames[0]
        previous_frame = frames[1]

        M = self.register(previous_frame, frame)
        return (cv2.warpPerspective(previous_frame, M, (previous_frame.shape[1], previous_frame.shape[0])).astype(np.uint8), M)

    def shift_all_frames(self, target_frame, frames, pool=None):
        '''
        Shifts all frames to target frame, returns list of shifted frames
        '''
        if(pool is not None):
            return pool.map(self._shift_frame, [((target_frame, f)) for f in frames])
        else:
            return [self._shift_frame(target_frame, f) for f in frames]