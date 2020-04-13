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

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

