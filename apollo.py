#!/usr/bin/env python3

# ---- Imports ----
import cv2
import numpy as np
import glob
import time
# ---- Local imports ----
import binarization
import perspective_karla
import hough_lines
import calibration_utils
import roi
import hough_lines
import convexhull
import shading
# ---- Default values ----
# Webcam paths
webcam_path = '/dev/video0'

# Calibration utils
testdir = '/home/pi/apollo-dev/apollo-pi/test_images/camera_calibration/'
fmt='.png'
n,m = 7,7
# Yellow lines threshold
yellow_min = np.array([0, 70, 70])
yellow_max = np.array([50, 255, 255])
kernel = 5
kernel2 = np.ones((1,20), np.uint8)

def camera_calibration():
    ret, mtx, dist, rvecs, tvecs = calibration_utils.calibrate_camera(n,m,testdir,fmt)
    return mtx, dist

def main(mtx, dist):
    video_capture = cv2.VideoCapture(webcam_path)
    while(True):
        ret, frame = video_capture.read()
        frame = cv2.blur(frame, (7,7))
        undist_img = calibration_utils.undistort(frame, mtx, dist)
        hl_yellow = binarization.highlight_yellow_lines(undist_img, yellow_min, yellow_max)
        hl_white = binarization.highlight_white_lines(undist_img)
        hl_white_yellow = binarization.white_yellow(undist_img, yellow_min, yellow_max)
        edges = cv2.Canny(undist_img,50, 200)
        warped_masked = perspective_karla.birdeye(edges)
        warped_unmasked = perspective_karla.birdeye(undist_img)
        angle, flag_ctl, l, r = perspective_karla.perspective_cal(warped_masked)
        #cv2.imshow('frame-white-yellow',hl_white_yellow)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    return angle, flag_ctl, l, r 

if __name__ == '__main__':
    mtx, dist = camera_calibration()
    main(mtx, dist)
