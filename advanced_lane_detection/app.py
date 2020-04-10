
import glob
from moviepy.editor import VideoFileClip
from advanced_lane_detection.process_image import process_image
from advanced_lane_detection.utils import utils

# TODO make a class with calibration data

images = glob.glob('../camera_cal/calibration*.jpg')  # images used for camera calibration

# Calibrate camera.
nx = 9  # number of inside corners in x
ny = 6  # number of inside corners in y
imgpoints, objpoints = utils.calibrate_cam(images, nx, ny)

white_output = '../project_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("../project_video.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)