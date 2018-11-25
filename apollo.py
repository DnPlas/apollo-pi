#!/usr/bin/env python3

# ---- Imports ----
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
# ---- Local imports ----
import binarization
import perspective_karla
import hough_lines
import calibration_utils
import roi

# ---- Default values ----
# Webcam paths
webcam_path = '/dev/video1'

# Calibration utils
testdir = '/home/dnplas/dplascen-dev/apollo/test_images/camera_calibration/'
#testdir = '/home/pi/apollo-pi/test_images/camera_calibration/'
fmt='.png'
n,m = 7,7
# Yellow lines threshold
yellow_min = np.array([0, 70, 70])
yellow_max = np.array([50, 255, 255])
kernel = 5

def camera_calibration():
    ret, mtx, dist, rvecs, tvecs = calibration_utils.calibrate_camera(n,m,testdir,fmt)
    return mtx, dist

def main(mtx, dist):
    video_capture = cv2.VideoCapture(webcam_path)
    while(True):
        ret, frame = video_capture.read()
        frame = cv2.blur(frame, (7,7))
        undist_img = calibration_utils.undistort(frame, mtx, dist)
        #undist_img = roi.crop_roi(undist_img)
        hl_yellow = binarization.highlight_yellow_lines(undist_img, yellow_min, yellow_max)
        hl_white = binarization.highlight_white_lines(undist_img)
        hl_white_yellow = binarization.white_yellow(undist_img, yellow_min, yellow_max)
        warped_masked = perspective_karla.birdeye(hl_white_yellow)
        warped_unmasked = perspective_karla.birdeye(undist_img)
        left_f, right_f, angle, flag_ctl = perspective_karla.perspective_cal(undist_img, warped_masked, warped_unmasked, yellow_min, yellow_max)
        cv2.imshow('undist-img',undist_img)
        cv2.imshow('frame-clean',frame)
        cv2.imshow('frame-yellow',hl_yellow)
        cv2.imshow('frame-white',hl_white)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return left_f, right_f, angle, flag_ctl

if __name__ == '__main__':
    mtx, dist = camera_calibration()
    main(mtx, dist)

