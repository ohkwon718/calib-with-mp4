# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import cv2
import numpy as np

# filename = 'video/cali_cam1.MP4'
# filename = 'video/cam1.MP4'
lsFilename = ['video/cam1.MP4', 'video/cam1-1.MP4', 'video/cam1-2.MP4',
    'video/cam1-3.MP4', 'video/cam1-3.MP4']


nFrame4calib = 30
bShow = True
# bShow = False
delay = 100

nCorners = (5,7)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((nCorners[0]*nCorners[1],3), np.float32)
objp[:,:2] = np.mgrid[0:nCorners[0],0:nCorners[1]].T.reshape(-1,2) * 0.0288

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
frames = []

nFrame = 0
for filename in lsFilename:
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    nFrame += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

if nFrame <= nFrame4calib:
    idxs = np.arange(nFrame)
else:
    idxs = np.linspace(0,nFrame-1,nFrame4calib).astype(int)

print idxs.size


idxTotal = 0
for filename in lsFilename:
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if idxTotal < idxs[0]:
                idxTotal += 1
                continue
            print "frame ",idxTotal

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
            idxTotal += 1
            if idxs.size == 1:
                break
            idxs = idxs[1:]
            
        else: 
            break
        
cap.release()
cv2.destroyAllWindows()

cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print mtx
print dist

tot_error = 0
tot_error_undist = 0
mean_error = 0


def MyProjection(objp, rvec, tvec, mtx, dst):
    rot, _ = cv2.Rodrigues(rvec)
    imgp = np.dot(rot,objp.T) + tvec
    imgp = imgp / imgp[2,:]

    x = imgp[0,:]
    y = imgp[1,:]
    
    r = np.sqrt(x**2 + y**2)

    k1, k2, p1, p2, k3 = dst[0]
    c = (1 + k1*r**2 + k2*r**4 + k3*r**6)/(1)
    x_new = c*x + 2*p1*x*y + p2*(r**2 + 2*x**2)
    y_new = c*y + p1*(r**2 + 2*y**2) + 2*p2*x*y

    imgp[0,:]= x_new
    imgp[1,:]= y_new
    

    imgp = np.dot(mtx,imgp)[:2,:].T.astype(objp.dtype)
    return imgp


cnt_undist = 0
for frame, rvec, tvec, corners, objp in zip(frames, rvecs, tvecs, imgpoints, objpoints):
    
    # dist = np.zeros(5)
    imgp1 = MyProjection(objp, rvec, tvec, mtx, dist)
    # imgpoints1, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    # print cv2.norm(imgpoints1.reshape(imgp1.shape), imgp1, cv2.NORM_L2)
    
    for p in imgp1:
        cv2.circle(frame, tuple(p.astype(int)), 1, (0,0,255), 3)

    tot_error += cv2.norm(corners,imgp1.reshape(corners.shape), cv2.NORM_L2)/len(imgp1)

    if bShow:
        cv2.imshow('Frame',frame)
        # cv2.imshow('Dst',dst)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
           break

print "total error: ", tot_error/len(objpoints)




