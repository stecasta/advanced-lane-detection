# **Advanced Lane Finding**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

Overview
---
This project implements a pipeline to detect lane lines and approximate the radius of curvature of the lane and the position of the vehicle with respect to the lane center using Python and OpenCV.

Content
---
This repository contains:

* The package `advanced_lane_detection` with the source code.
* The folder `camera_cal` with the chessboard images used for camera calibration.
* The folders `test_images` and `output_images` with the images used for testing the pipeline and the output of each step of the pipeline respectively.
* The video "project_video_output" which shows and example of the code working on a highway drive.
* A writeup in which each step of the pipeline is described in details.

The module `advanced_lane_detection` contains:
* The main script "app.py" that load a video and process it.
* The config file "config.py" that stores camera calibration data and lanes information.
* The subpackage `process_image`.
    * The module "lane.py" where the lane class is defined.
    * The module "process_image.py" that applies the pipeline to each frame.
* The subpackage `utils`
    * The module "utils.py" that contains all the functions used to process the video frames.

Dependencies
---
<!--TODO makefile.py + requirements.txt
init:
    pip install -r requirements.txt

test:
    py.test tests

.PHONY: init test

-->

* Numpy
* Matplotlib
* OpenCV
* Glob
* Moviepy

Usage
---
<!-- Dependencies-->
**Step 1:** Download the repository. `> git clone https://github.com/stecasta/advanced-lane-detection.git`

**Step 2:** Navigate to the package repository. `> cd advanced-lane-detection`

**Step 3 (optional):** Change the line `clip1 = VideoFileClip("../project_video.mp4")
` in "app.py" to load your own video. 

**Step 4:** Run the script. The output video will be saved as "project_video_output1.mp4" `python -m
 advanced_lane_detection`. (If you have both python 2 and 3 installed run `python3 -m advanced_lane_detection`).


License
---

This project is licensed under the terms of the [MIT license](https://opensource.org/licenses/MIT).
