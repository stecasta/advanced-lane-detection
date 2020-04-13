import numpy as np
import glob
from moviepy.editor import VideoFileClip
from advanced_lane_detection.process_image import process_image
from advanced_lane_detection.utils import utils
from advanced_lane_detection import config
from advanced_lane_detection.process_image import lane
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Calibrate camera.
images = glob.glob('../camera_cal/calibration*.jpg')  # images used for camera calibration
nx = 9  # number of inside corners in x
ny = 6  # number of inside corners in y
config.imgpoints, config.objpoints = utils.calibrate_cam(images, nx, ny)

# # Test on image
# # Read the image.
# image = mpimg.imread("../test_images/test6.jpg")
# output_img = process_image.process_image(image)

# # Plot.
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.set_title('Source')
# ax1.imshow(image)
# ax2.set_title('Lanes')
# ax2.imshow(output_img)
# plt.waitforbuttonpress()


# Process video.
white_output = '../project_video_output5.mp4'
## You can uncomment the following line for a subclip of the first 5 seconds
# clip1 = VideoFileClip("../project_video.mp4").subclip(20,25)
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(process_image.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)