# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import cv2
import numpy as np
 

nCorners = (5,7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((nCorners[0]*nCorners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:nCorners[0],0:nCorners[1]].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


cap = cv2.VideoCapture('video/cali_cam1.MP4')

 
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
    
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        find, corners = cv2.findChessboardCorners(gray, nCorners, None)
        
        if find == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, nCorners, corners, find)
            
            cv2.imshow('Frame',frame)
        else:
            cv2.imshow('Frame',frame)
     
        if cv2.waitKey(25) & 0xFF == ord('q'):
           break

    else: 
        break
    


cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

