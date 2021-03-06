# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from advanced_lane_detection import config


def calibrate_cam(images, nx, ny):

    # Arrays to store object points and images points for all the images.
    objpoints = []
    imgpoints = []

    # Prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x,y coord.

    for fname in images:
        # Read each image.
        img = mpimg.imread(fname)

        # Convert image to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners are found, add object and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    return imgpoints, objpoints

def segment(img, s_thresh, sx_thresh):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

def warp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[203, 720],
         [1127, 720],
         [585, 460],
         [695, 460]])

    dst = np.float32(
        [[320, 720],
         [960, 720],
         [320, 0],
         [960, 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

def unwarp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[203, 720],
         [1127, 720],
         [585, 460],
         [695, 460]])

    dst = np.float32(
        [[320, 720],
         [960, 720],
         [320, 0],
         [960, 0]])

    Minv = cv2.getPerspectiveTransform(dst, src)

    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)

    return unwarped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if (sum(good_left_inds > minpix)):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if (sum(good_right_inds > minpix)):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_fit = config.left_lane.current_fit
    right_fit = config.right_lane.current_fit

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):

    # Check for how many frames in a row the sanity check didn't pass
    if (config.left_lane.sanity_slope):
        config.left_lane.sanity_counter += 1
    else:
        config.left_lane.sanity_counter = 0

    # Find our lane pixels
    # For the first frame we use the slower algorithm "find_lane_pixels" . If a polynomial is found, for the next
    # frames we use the faster "search_around_poly". In the case the sanity check doesn't pass for more than 60
    # frames, we go back to the first algorithm.
    if (config.left_lane.detected and config.left_lane.sanity_counter < 60):
        leftx, lefty, rightx, righty, out_img = search_around_poly(binary_warped)
    else:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    config.left_lane.current_fit = np.polyfit(lefty, leftx, 2)
    config.right_lane.current_fit = np.polyfit(righty, rightx, 2)

    # Sanity check: The two polynomials should have similar slope. Slope of parabola is 2ax + b. I check at x = 0.
    if (np.absolute(config.left_lane.current_fit[1] - config.right_lane.current_fit[1]) < 0.2 ):
        config.left_lane.sanity_slope = True
    else:
        config.left_lane.sanity_slope = False

    # Append to list of recent fits if sanity check is passed or it's the first frame
    if (config.left_lane.sanity_slope or config.left_lane.first_frame):
        config.left_lane.fits.append(config.left_lane.current_fit)
        config.right_lane.fits.append(config.right_lane.current_fit)

    config.left_lane.first_frame = False

    # Average of last 10 computed poly (if less than 10 frames are computed average all elements)
    config.left_lane.best_fit = sum(config.left_lane.fits[len(config.left_lane.fits) - np.min([10,len(config.left_lane.fits)]):]) / np.min([10,len(config.left_lane.fits)])
    config.right_lane.best_fit = sum(config.right_lane.fits[len(config.right_lane.fits) - np.min([10,len(config.right_lane.fits)]):]) / np.min([10,len(config.right_lane.fits)])

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = config.left_lane.best_fit[0] * ploty ** 2 + config.left_lane.best_fit[1] * ploty + config.left_lane.best_fit[2]
        right_fitx = config.right_lane.best_fit[0] * ploty ** 2 + config.right_lane.best_fit[1] * ploty + config.right_lane.best_fit[2]
        config.left_lane.detected = True
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        config.left_lane.detected = False
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Prepare points to be used to defined the lane region.
    draw_points_l = (np.asarray([left_fitx, ploty]).T).astype(np.int32)  # needs to be int32 and transposed
    draw_points_r = (np.asarray([right_fitx, ploty]).T).astype(np.int32)  # needs to be int32 and transposed
    draw_points_r = np.flipud(draw_points_r)
    points = np.concatenate((draw_points_l, draw_points_r))

    # Draw the left and right polynomials on the lane lines.
    # cv2.polylines(out_img, [draw_points_l], False, (255, 255, 0), 4)  # args: image, points, closed, color
    # cv2.polylines(out_img, [draw_points_r], False, (255, 255, 0), 4)  # args: image, points, closed, color

    # Fill lane region in green.
    cv2.fillPoly(out_img, [points], (0,255,0))

    return out_img, ploty, left_fit_cr, right_fit_cr

def weighted_img(img, initial_img, α, β, γ):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def measure_curvature(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature).
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Sanity check
    if np.absolute(right_curverad - left_curverad):
        config.left_lane.sanity_curverad = True
    else:
        config.left_lane.sanity_curverad = False

    # Append to list of computed curverads if sanity check is passed or it's the first frame
    if config.left_lane.sanity_curverad or config.left_lane.first_frame:
        config.left_lane.curverads.append(left_curverad)
        config.right_lane.curverads.append(right_curverad)

    # Average of last 30 computed curverads (if less than 30 frames are computed average all elements)
    left_curverad = sum(config.left_lane.curverads[len(config.left_lane.curverads) - np.min([30,len(config.left_lane.curverads)]):]) / np.min([30,len(config.left_lane.curverads)])
    right_curverad = sum(config.right_lane.curverads[len(config.right_lane.curverads) - np.min([30,len(config.right_lane.curverads)]):]) / np.min([30,len(config.right_lane.curverads)])

    return left_curverad, right_curverad

def measure_pos_in_lane(img):
    # Lane is 3.7 meters wide.
    # Camera is mounted at the center of the vehicle.
    # Find center of the lane and compute offset with center of the camera.
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Retrieve poly
    if (config.left_lane.detected):
        left_fit = config.left_lane.best_fit
        right_fit = config.right_lane.best_fit
    else:
        left_fit = config.left_lane.current_fit
        right_fit = config.right_lane.current_fit

    # Find distance from left and right lane for a given y value
    left_dist = left_fit[0] * img.shape[0] ** 2 + left_fit[1] * img.shape[0] + left_fit[2]
    right_dist = right_fit[0] * img.shape[0] ** 2 + right_fit[1] * img.shape[0] + right_fit[2]

    pos = (left_dist + right_dist) / 2
    offset_m = ((img.shape[1] / 2) - pos) * xm_per_pix

    # Sanity check
    computed_lane_width = (right_dist - left_dist) * xm_per_pix
    if 3.2 < computed_lane_width < 4.2:
        config.left_lane.sanity_offset = True
    else:
        config.left_lane.sanity_offset = False

    # Append to list of computed offsets if sanity check is passed and it's not the first frame
    if config.left_lane.sanity_offset or config.left_lane.first_frame:
        config.left_lane.line_offsets.append(offset_m)

    # Average of last 30 computed offsets (if less than 30 frames are computed average all elements)
    offset_m_avg = sum(config.left_lane.line_offsets[len(config.right_lane.line_offsets) - np.min([30,len(config.left_lane.line_offsets)]):]) / np.min([30,len(config.left_lane.line_offsets)])

    return offset_m_avg

def write_on_image(img, text, bottomLeftCornerOfText):
    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return
