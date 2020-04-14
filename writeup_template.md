## Advanced Lane Finding Project

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calib.png "Undistorted"
[image2]: ./output_images/undist.png "Binary Example"
[image3]: ./output_images/binary.png "thresholded image"
[image4]: ./output_images/warped-unwarped.png "Warp Example"
[image5]: ./output_images/fitpoly.png "Fit Visual"
[image6]: ./output_images/final.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

In this file are addressed all the rubric points.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 12 through 15 of the file called "app.py", where the calibration images
 are loaded and the function `calibrate_cam()` is called and the number of nodes `nx` and `ny` are specified. The
  mentioned function is in the file called "utils.py" from line 9 to line 36.
  
The first step is to convert the image to grayscale, then the corners are found by using the function `cv2
.findChessboardCorners()`. If the process is successful the points are appended to `imgpts` and `objpts`.
 
 The object points represent the (x, y, z) coordinates of the chessboard corners in the real world. Since the
  chessboard is put on a wall, we can assume that all the points lie in the same (x,y) plane (z=0). This means that
   the object points don't change from one calibration image to the other. Image points, on the other hand, represent
    the pixel position of the points in each image and after each successfull detection we append them to `imgpts`.

`objpts` and `imgpts` are then used as an input to the function `cv2.calibrateCamera()` which returns camera
 calibration and distortion coefficients. These coefficients are initialized and stored in a file called "config.py" so
  that the calibration process is done only once.  Finally, images can be undistorted by using `cv2.undistort()` function. An
  example is shown below. 


![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Just like in the example with the chessboard, now that we have our camera coefficients we can call the function `cv2
.undistort()` and obtain the undistorted image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. In particular I used the S channel
 and a sobelx filter. The algorithm is
 defined in
 function `segment()` from line 39 to 60 of file "utils.py". The thresholds used (defined in lines 16-17 of
  "process_image.py"
 ) were 
 ```
    s_thresh=(170, 255) # saturation channel threshold
    sx_thresh=(20, 100) # sobelx threshold
```
  Here's an example of my output
 for this step. Blue pixels are segmented by the
  saturation filter and green pixels are segmented by the sobel x filter.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I defined a function called `warp()` in "utils.py" file from line 62 to 81. This function takes an image as an imput
 and returns a warped version of it. I hardcoded the source and destination points in the following way:
 
 ```python
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
```
I found the point by showing the image un the screen and identifying in it 4 points that would form a rectangle in
 birds-eye view. The destination points were chosen arbitrary.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test
 image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to find the lane line pixels I defined two functions, `find_lane_pixels()` and `search_around_poly()`, in
 "utils
.py" through lines 104-186 and187-228 respectively. Both the functions take the binary image as an input. 

The function `find_lane_pixels()` performs an histogram to it and searches with a sliding window two peaks, which
 correspond to
 lane pixels. This
  algorithm is fairly reliable although rather slow.
   
   With `search_around_poly()` we exploit the fact that from one frame
   to another the lane won't move much. Therefore we keep track of the previously fitted poly (more on this later) by
    storing its coefficients in "config.py" by making two istances of the lane class (defined in "lane.py"). We then
     search only in a small regioun around the polynomials. If we are not able to find a good fit for many frames we
      then go back to the first algorithm.
      
  Finally, we define a function `fit_polynomial()` in "utils.py" (229-305) which calls one of the two functions
   described above based on the reasoning mentioned. In order to find the coefficients we apply the function `np
   .polyfit` to the lanes pixels. 
   
   Moreover, a low pass filter is applied to make an average of the polynomial over
    the last 10 frames. An example is showed in the figure below.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
