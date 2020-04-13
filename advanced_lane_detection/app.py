import glob
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from advanced_lane_detection.process_image import process_image
from advanced_lane_detection.utils import utils
from advanced_lane_detection import config

def run():

    # Calibrate camera.
    images = glob.glob('camera_cal/calibration*.jpg')  # images used for camera calibration
    nx = 9  # number of inside corners in x
    ny = 6  # number of inside corners in y
    config.imgpoints, config.objpoints = utils.calibrate_cam(images, nx, ny)

    # # Test on image
    # image = mpimg.imread("../test_images/test2.jpg")
    # output_img = process_image.process_image(image)
    # plt.imshow(output_img)
    # plt.waitforbuttonpress()

    # Process video.
    white_output = 'project_video_output1.mp4'
    ## You can uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("project_video.mp4").subclip(0,5)
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image.process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    return