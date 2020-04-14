# Module with all the global variables.

from advanced_lane_detection.process_image import lane
import time

# Global lists in which we store camera calibration and distortion coefficients.
ret = []
mtx = []
dist = []
rvecs = []
tvecs = []

# Create two instances of the lane class for the left and right lane.
left_lane = lane.Line()
right_lane = lane.Line()

# Variables to measure elapsed time.
toc = time.time()
tic = time.time()
