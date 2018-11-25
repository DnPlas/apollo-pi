import cv2
import numpy as np

def crop_roi(frame):
    h,w = frame.shape[:2]
    w_2 = int(w/2)
    pts = np.array([[0,(h-50)], # bottom left corner
                    [w,(h-50)], # bottom right corner
                    [(w_2+150), 50],
                    [(w_2-150), 50]])
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = frame[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst

if __name__ == '__main__':
    frame = cv2.imread('/home/dnplas/dplascen-dev/apollo/test_images/camera_calibration/camera_distance.png')
    crop_roi(frame)
