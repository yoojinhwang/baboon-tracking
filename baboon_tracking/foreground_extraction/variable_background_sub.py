import cv2
import numpy as np
import math

from .ForegroundExtraction import ForegroundExtraction

class VariableBackgroundSub_ForegroundExtraction(ForegroundExtraction):
    '''
    This is the strategy that we've been implementing, using hist of dissimilarity,
    freq of commonality, weights, etc
    '''

    def _quantize_frame(self, frame):
        '''
        Normalize pixel values from 0-255 to values from 0-10
        Returns quantized frame
        '''
        return np.floor(frame.get_frame().astype(np.float32) * 50.0 / 255.0).astype(np.uint8).astype(np.int32)

    def _intersect_frames(self, frames, q_frames):
        '''
        Intersect two consecutive frames to find common background between those two frames
        Returns the single frame produced by intersection
        '''
        print('intersect')

        mask = np.abs(q_frames[0] - q_frames[1]) <= 1
        combined = frames[0].get_frame().copy()
        combined[mask] = 0

        return combined

    def _union_frames(self, frames):
        '''
        Union all frame intersections to produce acting background for all frames
        Returns the single union frame produced by unioning all frames in input
        '''
        print('union')

        union = np.zeros(frames[0].shape).astype(np.uint8)

        f_copy = frames.copy()
        f_copy.reverse()

        for f in f_copy:
            union[union == 0] = f[union == 0]

        return union

    def _get_history_of_dissimilarity(self, frames, q_frames):
        '''
        Calculate history of dissimilarity according to figure 10 of paper
        Returns frame representing history of dissimilarity
        '''
        print('dissimilarity')

        dissimilarity = np.zeros(frames[0].get_frame().shape).astype(np.uint32)

        for i in range(len(frames)):
            if i == 0:
                continue

            mask = (np.abs(q_frames[i] - q_frames[i - 1]) > 1).astype(np.uint32)
            dissimilarity = dissimilarity + np.multiply(np.abs(frames[i].get_frame().astype(np.int32) - frames[i - 1].get_frame().astype(np.int32)), mask)

        return (dissimilarity / len(frames)).astype(np.uint8)

    def _get_weights(self, q_frames):
        '''
        Calculate weights based on frequency of commonality between frames according
        to figure 12 of paper
        Returns frame representing frequency of commonality
        '''
        print('weights')

        weights = np.zeros(q_frames[0].shape).astype(np.uint8)

        for i, _ in enumerate(q_frames):
            if i == 0:
                continue

            mask = (np.abs(q_frames[i] - q_frames[i - 1]) <= 1).astype(np.uint8)
            weights = weights + mask

        return weights

    def _zero_weights(self, frame, weights):
        '''
        Gets foreground of frame by zeroing out all pixels with large weights, i.e. pixels in which frequency of commonality
        is really high, meaning that it hasn't changed much or at all in the history frames, according to figure 13 of paper
        Returns frame representing the foreground
        '''
        print('zero')

        f = frame.copy()
        f[weights >= self.history_frames - 1] = 0

        return f

    def _get_moving_foreground(self, weights, foreground, dissimilarity):
        '''
        Calculates moving foreground according to figure 14 of paper
        Each of W and D (weights and dissimilarity) is assigned to high, medium, and low

        Medium commonality AND low commonality but low dissimiliarity are considered moving foreground
        Otherwise, it is either a still or flickering background

        Return frame representing moving foreground
        '''

        history_frame_count_third = math.floor(float(self.history_frames - 1) / 3)
        third_gray = 255.0 / 3.0

        weights_low = (weights <= history_frame_count_third).astype(np.uint8)
        weights_medium = np.logical_and(history_frame_count_third < weights, weights < self.history_frames - 1).astype(np.uint8) * 2

        weight_levels = weights_low + weights_medium

        foreground_low = (foreground <= math.floor(third_gray)).astype(np.uint8)
        foreground_medium = ((math.floor(third_gray) < foreground) + (foreground < math.floor(2 * third_gray))).astype(np.uint8) * 2
        foreground_high = (foreground >= math.floor(2 * third_gray)).astype(np.uint8) * 3

        foreground_levels = foreground_low + foreground_medium + foreground_high

        dissimilarity_low = (dissimilarity <= math.floor(third_gray)).astype(np.uint8)
        dissimilarity_medium = ((math.floor(third_gray) < dissimilarity) + (dissimilarity < math.floor(2 * third_gray))).astype(np.uint8) * 2
        dissimilarity_high = (dissimilarity >= math.floor(2 * third_gray)).astype(np.uint8) * 3

        dissimilarity_levels = dissimilarity_low + dissimilarity_medium + dissimilarity_high

        moving_foreground = np.logical_and(weight_levels == 2, np.greater_equal(foreground_levels, dissimilarity_levels)).astype(np.uint8)
        moving_foreground = moving_foreground + np.logical_and(weight_levels == 1, np.logical_and(dissimilarity_levels == 1, np.greater(foreground_levels, dissimilarity_levels))).astype(np.uint8)

        return moving_foreground * 255

    def _intersect_all_frames(self, grouped_shifted_history_frames, grouped_quantized_frames):
        '''
        Takes in two lists of frames, performs intersect on each pair and returns array of intersects
        '''
        return [self._intersect_frames(z[0], z[1]) for z in zip(grouped_shifted_history_frames, grouped_quantized_frames)]


    def generate_mask(self, frame, shifted_history_frames, Ms):
        '''
        Takes in list of registered grayscale history frames
        Quantizes them, then generates mask using weights and history of dissimilarity

        frame - current frame
        shifted_history_frames - list of registered history frames
        Ms - transformation matrices aligning each registered history frame
        '''

        print('quantize frames')

        quantized_frames = [self._quantize_frame(f) for f in shifted_history_frames]

        print('quantized frames finished')

        print('warp perspective')

        masks = [cv2.warpPerspective(np.ones(frame.get_frame().shape), M, (frame.get_frame().shape[1], frame.get_frame().shape[0])).astype(np.uint8) for M in Ms]

        print('finished warp perspective')

        frame_group_index = range(len(shifted_history_frames) - 1)
        frame_group_index = [(r, r + 1) for r in frame_group_index]

        print(frame_group_index)

        # pairs each two frames to do intersection
        grouped_shifted_history_frames = [(shifted_history_frames[g[0]], shifted_history_frames[g[1]]) for g in frame_group_index]
        grouped_quantized_frames = [(quantized_frames[g[0]], quantized_frames[g[1]]) for g in frame_group_index]

        intersects = self._intersect_all_frames(grouped_shifted_history_frames, grouped_quantized_frames)
        union = self._union_frames(intersects)

        history_of_dissimilarity = self._get_history_of_dissimilarity(shifted_history_frames, quantized_frames)

        weights = self._get_weights(quantized_frames)

        frame_new = self._zero_weights(frame.get_frame(), weights)
        union_new = self._zero_weights(union, weights)

        #foreground = np.absolute(frame_new.astype(np.int32) - union_new.astype(np.int32)).astype(np.uint8)
        foreground =  cv2.absdiff(frame_new, union_new)

        #cv2.imshow('foreground', cv2.resize(foreground * 25, (1600, 900)))

        moving_foreground = self._get_moving_foreground(weights, foreground, history_of_dissimilarity)

        # This cleans up the edges after performing image registration.
        for mask in masks:
            moving_foreground = np.multiply(moving_foreground, mask)

        return moving_foreground
