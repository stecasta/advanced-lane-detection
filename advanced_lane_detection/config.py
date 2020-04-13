# Module with all the global variables.

from advanced_lane_detection.process_image import lane

# Global lists in which we store output of camera calibration.
imgpoints = []
objpoints = []

# Create two istances of the lane class for the left and right lane.
left_lane = lane.Line()
right_lane = lane.Line()