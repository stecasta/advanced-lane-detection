import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from advanced_lane_detection.utils import utils

images = glob.glob('../camera_cal/calibration*.jpg') # images used for camera calibration

nx = 9 # number of inside corners in x
ny = 6 # number of inside corners in y

# Calibrate camera.
imgpoints, objpoints = utils.calibrate_cam(images, nx, ny)

# Read the image.
image = mpimg.imread("../test_images/test3.jpg")

# Undistort image.
undistorted = utils.cal_undistort(image, objpoints, imgpoints)

# Create thresholded binary image.
segmented = utils.segment(undistorted)

# Warp image.
warped_img = utils.warp(segmented)

# Segment the two lanes
lanes_img = utils.fit_polynomial(warped_img)

# Unwarp image.
unwarped_img = utils.unwarp(lanes_img)

# Merge the two images.
output_img = utils.weighted_img(image, unwarped_img, α=0.8, β=1., γ=0.)


# Plot.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Source')
ax1.imshow(image)
ax2.set_title('Lanes')
ax2.imshow(output_img)
plt.waitforbuttonpress()