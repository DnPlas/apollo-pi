import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import binarization

dir_pairs = {'left': False, 'right': True}

def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

def birdeye(image):
#    edges=cv2.Canny(image, 50, 150, apertureSize = 3)
    edges = image
    
    h, w = image.shape[:2]
   
    src = np.float32([[490,269],    # BLUE w, h-191
                      [0, 269],    # GREEN 
                      [100,144],   # RED
                      [440, 144]])  # CYAN
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr
   # f, a = plt.subplots(1,2)
   # f.set_facecolor('white')
   # a[0].set_title('B')
   # a[0].imshow(image, cmap='gray')
   # for point in src:
   #     a[0].plot(*point, '.')
   # for point in dst:
   #     a[1].plot(*point, '.')
   # for axis in a:
   #     axis.set_axis_off()
   # plt.show()
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(edges, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped

def perspective_cal(image):
    warped = image
    histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((warped, warped, warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:np.int(histogram.shape[0]/2)])+23
    rightx_base = np.argmax(histogram[np.int(histogram.shape[0]/2):]) + midpoint
    h, w = image.shape[:2]
    if int(leftx_base) <= 23:
        angle = 20
        flag_ctl = 'go right'
        return angle, flag_ctl, leftx_base, rightx_base
    elif int(rightx_base) == 247:
        angle = -20
        flag_ctl = 'go left'
        return angle, flag_ctl, leftx_base, rightx_base
    else:
        lane_midpoint = (leftx_base+rightx_base)/2
        distance = midpoint-lane_midpoint
        angle = np.arctan2(float(distance), 288)
        if angle < 0:
            flag_ctl = 'go right'
        else:
            flag_ctl = 'go_left'
        return angle, flag_ctl, leftx_base, rightx_base
