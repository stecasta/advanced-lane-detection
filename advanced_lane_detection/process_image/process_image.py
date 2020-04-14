import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from advanced_lane_detection.utils import utils
from advanced_lane_detection import config
import time


def process_image(image):

    # Measure elapsed time for processing one frame
    # config.toc = time.time()
    # print(config.toc - config.tic)
    # config.tic = time.time()

    # Undistort image.
    undistorted = cv2.undistort(image, config.mtx, config.dist, None, config.mtx)

    # Create thresholded binary image.
    s_thresh=(170, 255) # saturation channel threshold
    sx_thresh=(20, 100) # sobelx threshold
    segmented = utils.segment(undistorted, s_thresh, sx_thresh)
    binary_img = np.zeros_like(segmented[:,:,0])
    binary_img[(segmented[:,:,1] > 0) | (segmented[:,:,2] > 0)] = 1

    # Warp image.
    warped_img = utils.warp(binary_img)

    # Code for plotting perspective transform src and dest points
    # cv2.line(image, (203, 720), (1127, 720), (255, 0, 0), 4)
    # cv2.line(image, (585, 460), (695, 460), (255, 0, 0), 4)
    # cv2.line(image, (203, 720), (585, 460), (255, 0, 0), 4)
    # cv2.line(image, (1127, 720), (695, 460), (255, 0, 0), 4)
    #
    # cv2.line(warped_img, (320, 0), (960, 0), (255, 0, 0), 4)
    # cv2.line(warped_img, (320, 720), (950, 720), (255, 0, 0), 4)
    # cv2.line(warped_img, (320, 720), (320, 0), (255, 0, 0), 4)
    # cv2.line(warped_img, (960, 720), (960, 0), (255, 0, 0), 4)

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    # ax1.set_title('Original image')
    # ax1.imshow(image)
    # ax2.set_title('Undistorted image')
    # ax2.imshow(undistorted)
    # plt.waitforbuttonpress()

    # Find lanes.
    lanes_img, ploty, left_fit_real, right_fit_real = utils.fit_polynomial(warped_img)

    # Unwarp image.
    unwarped_img = utils.unwarp(lanes_img)

    # Merge the two images.
    output_img = utils.weighted_img(image, unwarped_img, α=0.3, β=1, γ=0.)

    # Calculate the radius of curvature in meters for both lane lines.
    config.left_lane.curverad, config.right_lane.curverad = utils.measure_curvature(ploty, left_fit_real, right_fit_real)
    curverad = int((config.left_lane.curverad + config.right_lane.curverad) / 2)

    # Calculate position of the vehicle in the lane.
    offset = utils.measure_pos_in_lane(binary_img)
    if (offset > 0):
        pos = "right"
    else:
        pos = "left"

    # Write info on image
    text1 = "The radius of curvature is {} (m).".format(curverad)
    utils.write_on_image(output_img, text1, (80, 50))
    text2 = "Vehicle is %.2f (m) %s of center." % (np.absolute(offset), pos)
    utils.write_on_image(output_img, text2, (80, 120))

    return output_img