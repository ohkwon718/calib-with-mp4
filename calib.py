# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import os
import sys
import cv2
import numpy as np
from collections import deque
import pickle
import dill

print len(sys.argv), sys.argv

# fullpath_out = sys.argv[1]
# list_fullpath_in = [sys.argv[i] for i in range(2,len(sys.argv))]

fullpath_out = 'out.txt'
list_fullpath_in = ['video/190502_02-004.mp4']

file_out = os.path.basename(fullpath_out)
filename_out = os.path.splitext(file_out)[0]

nFrame4calib = 30
bSave = True
# bSave = False
bRandomSample = True
# bRandomSample = False

nCorners = (5,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((nCorners[0]*nCorners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:nCorners[0],0:nCorners[1]].T.reshape(-1,2) * 0.0288

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
frames = []

valid_frames = []
valid_grays = []
valid_corners = []

nFrame = 0
for filename in list_fullpath_in:
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    nFrame += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(nFrame)

n = 0
for filename in list_fullpath_in:
    print(filename)
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            find, corners = cv2.findChessboardCorners(gray, nCorners, None)
            n += 1
            print n
            if find:
                valid_frames.append(frame)
                valid_grays.append(gray)
                valid_corners.append(corners)
                print "find ",len(valid_frames)
        else: 
            break
        
    cap.release()

nFrame = len(valid_frames)
print "Detected %d frames" % nFrame

if nFrame <= nFrame4calib:
    idxs = range(nFrame)
else:
    if bRandomSample:
        idxs = np.sort(np.random.choice(nFrame, nFrame4calib, replace=False)).tolist()
    else:
        idxs = np.linspace(0,nFrame-1,nFrame4calib).astype(int).tolist()

print len(idxs)
for idx in idxs:
    print "frame ",idx
    frame = valid_frames[idx]
    gray = valid_grays[idx]
    corners = valid_corners[idx]

    objpoints.append(objp)
    corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners)
    cv2.drawChessboardCorners(frame, nCorners, corners, find)
    frames.append(frame) 
print

cv2.destroyAllWindows()

cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print mtx
print dist

print len(idxs)
print len(rvecs)
print len(tvecs)

np.savetxt(fullpath_out, np.append(mtx, dist))


######################### test ##########

objpoints_test = [] # 3d point in real world space
imgpoints_test = [] # 2d points in image plane.
frames_test = []
idxs_test = range(0,len(valid_frames),5)

print len(idxs_test)
for idx in idxs_test:
    print "frame ",idx
    frame = valid_frames[idx]
    gray = valid_grays[idx]
    corners = valid_corners[idx]

    objpoints_test.append(objp)
    corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints_test.append(corners)
    cv2.drawChessboardCorners(frame, nCorners, corners, find)
    frames_test.append(frame) 

cv2.destroyAllWindows()

cali_ret, _, _, rvecs_test, tvecs_test = cv2.calibrateCamera(objpoints_test, imgpoints_test, gray.shape[::-1],mtx, dist)

tot_error = 0
for i in range(len(frames_test)):
    frame, rvec, tvec, corners, objp = frames_test[i], rvecs_test[i], tvecs_test[i], imgpoints_test[i], objpoints_test[i]

    imgp, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    error = cv2.norm(imgpoints_test[i],imgp, cv2.NORM_L2)/len(imgp)
    tot_error += error

    for p in imgp:
        cv2.circle(frame, tuple(p[0].astype(int)), 1, (0,255,255), 2)

    if bSave:
        cv2.imwrite(os.path.join('log', filename_out + '_' + str(i).zfill(3) + '.png'),frame)
        if cv2.waitKey(150) & 0xFF == ord('q'):
           break


print "mean error: ", tot_error/len(objpoints)