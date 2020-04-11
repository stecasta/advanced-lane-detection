import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # # x values of the last n fits of the line
        # self.recent_xfitted = []
        # # average x values of the fitted line over the last n iterations
        # self.bestx = None
        # polynomial coefficients of the last n fits
        self.fits = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radii of curvature of the line in some units of last n frames
        self.curverads = []
        # distances in meters of vehicle center from the line of last n frames
        self.line_offsets = []
        # Sanity check on lane width: lane width is approximately 3.7m
        self.sanity_offset = False
        # Sanity check on curverad: the two lines have similar curverad
        self.sanity_curverad = False
        # Sanity check on slope: the two lanes should have similar slopes
        # # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        # # x values for detected line pixels
        # self.allx = None
        # # y values for detected line pixels
        # self.ally = None