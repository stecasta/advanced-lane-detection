# Module with all the global variables.

from advanced_lane_detection.process_image import lane

imgpoints = []
objpoints = []

# Create two istances of the lane class for the left and right lane
lane_sx = lane.Line()
lane_dx = lane.Line()