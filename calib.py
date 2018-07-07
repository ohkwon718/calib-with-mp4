# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import cv2
import numpy as np
 

nCorners = (5,7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((nCorners[0]*nCorners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:nCorners[0],0:nCorners[1]].T.reshape(-1,2)
# print objp

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
frames = []

# cap = cv2.VideoCapture('video/cali_cam1.MP4')
cap = cv2.VideoCapture('video/cam1.MP4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

nFrame4calib = 3
nFrameTotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if nFrameTotal <= nFrame4calib:
    idxs = np.arange(nFrameTotal)
else:
    idxs = np.linspace(0,nFrameTotal-1,nFrame4calib).astype(int)

print idxs.size

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if i < idxs[0]:
            i = i + 1
            continue
        print "frame ",i
        

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        find, corners = cv2.findChessboardCorners(gray, nCorners, None)
        
        if find == True:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, nCorners, corners, find)
            
            frames.append(frame)    
            cv2.imshow('Frame',frame)
        else:
            cv2.imshow('Frame',frame)
     
        if cv2.waitKey(25) & 0xFF == ord('q'):
           break
        i = i + 1
        
        if idxs.size == 1:
            break
        idxs = idxs[1:]
        # if i == 20:
        #     break
    else: 
        break
    
cap.release()
cv2.destroyAllWindows()


cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


for frame, rvec, tvec, corners in zip(frames, rvecs, tvecs, imgpoints):
    rot, _ = cv2.Rodrigues(rvec)
    x = np.dot(rot,objp.T) + tvec
    x = x / x[2,:]
    x = np.dot(mtx,x)[:2,:]
    print np.max(np.abs(x - corners[:,0,:].T)) # maximum diff btw projected and original corner

    for p in x.T:
        cv2.circle(frame, tuple(p.astype(int)), 1, (0,0,255), 3)
    

    cv2.imshow('Frame',frame)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
       break




