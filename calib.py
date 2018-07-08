# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import cv2
import numpy as np

bShow = True
delay = 100

nCorners = (5,7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((nCorners[0]*nCorners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:nCorners[0],0:nCorners[1]].T.reshape(-1,2) * 0.0288

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
frames = []

# cap = cv2.VideoCapture('video/cali_cam1.MP4')
cap = cv2.VideoCapture('video/cam1.MP4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

nFrame4calib = 30
nFrameTotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        
        if bShow:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
               break
        i = i + 1
        if idxs.size == 1:
            break
        idxs = idxs[1:]
        
    else: 
        break
    
cap.release()
cv2.destroyAllWindows()

cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


tot_error = 0
tot_error_undist = 0
mean_error = 0

for frame, rvec, tvec, corners, objp in zip(frames, rvecs, tvecs, imgpoints, objpoints):
    imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    # imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, np.zeros(5))
    
    for p in imgpoints2[:,0,:]:
        cv2.circle(frame, tuple(p.astype(int)), 1, (0,0,255), 3)

    tot_error += cv2.norm(corners,imgpoints2, cv2.NORM_L2)/len(imgpoints2)

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
     
    imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, newcameramtx, dist)
    # imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, newcameramtx, np.zeros(5))

    for p in imgpoints2[:,0,:]:
        cv2.circle(dst, tuple(p.astype(int)), 1, (0,0,255), 3)

    tot_error_undist += cv2.norm(corners,imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    
    if bShow:
        cv2.imshow('Frame',frame)
        cv2.imshow('Dst',dst)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
           break

print "total error: ", tot_error/len(objpoints)
print "total error: ", tot_error_undist/len(objpoints)

def MyProjection():
    rot, _ = cv2.Rodrigues(rvec)
    x = np.dot(rot,objp.T) + tvec
    x = x / x[2,:]
    # c = (1 + )
    # * x[0,:] 
    diffX = (x[0,:] - w/2)
    diffY = (x[1,:] - h/2)
    r = np.sqrt(diffX**2 + diffY**2)
    # c = 

    x = np.dot(mtx,x)[:2,:]
