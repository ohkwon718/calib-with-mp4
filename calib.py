# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import cv2
import numpy as np
from collections import deque
import pickle
import dill

# # lsFilename = ['video/180801_1-001.mp4']
# # lsFilename = ['video/180801_2-002.mp4']
# # lsFilename = ['video/180801_3-003.mp4']
# # lsFilename = ['video/180801_4-004.mp4']
# # lsFilename = ['video/180801_5-005.mp4']
# # lsFilename = ['video/180817_1-001.mp4']
# # lsFilename = ['video/180817_2-002.mp4']
# # lsFilename = ['video/180817_3-003.mp4']
# # lsFilename = ['video/180817_4-004.mp4']
# # lsFilename = ['video/180817_5-005.mp4']
# # lsFilename = ['video/180817_6-006.mp4']
# # lsFilename = ['video/180817_7-007.mp4']
# # lsFilename = ['video/180817_8-008.mp4']
# lsFilename = ['video/180817_9-009.mp4']


# nFrame4calib = 10000
# bShow = True
# # bShow = False
# bRandomSample = True
# # bRandomSample = False
# # delay = 150
# delay = 0

# nCorners = (5,7)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# objp = np.zeros((nCorners[0]*nCorners[1],3), np.float32)
# objp[:,:2] = np.mgrid[0:nCorners[0],0:nCorners[1]].T.reshape(-1,2) * 0.0288

# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# frames = []

# valid_frames = []
# valid_grays = []
# valid_corners = []


# nFrame = 0
# for filename in lsFilename:
#     cap = cv2.VideoCapture(filename)
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")

#     nFrame += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print nFrame

# n = 0
# for filename in lsFilename:
#     print filename
#     cap = cv2.VideoCapture(filename)
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")
#         continue

#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:
#             gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#             find, corners = cv2.findChessboardCorners(gray, nCorners, None)
#             n += 1
#             print n
#             if find:
#                 valid_frames.append(frame)
#                 valid_grays.append(gray)
#                 valid_corners.append(corners)
#                 print "find ",len(valid_frames)
#         else: 
#             break
#         # if n > 200:
#         #     break
#     cap.release()

# nFrame = len(valid_frames)
# print "Detected %d frames" % nFrame

# if nFrame <= nFrame4calib:
#     idxs = range(nFrame)
# else:
#     if bRandomSample:
#         idxs = np.sort(np.random.choice(nFrame, nFrame4calib, replace=False)).tolist()
#     else:
#         idxs = np.linspace(0,nFrame-1,nFrame4calib).astype(int).tolist()

# filename = 'globalsave.pkl'
# dill.dump_session(filename)

filename = 'globalsave.pkl'
dill.load_session(filename)
delay = 10

# idxs = range(0,len(valid_frames),3)
# idxs = range(0,len(valid_frames))
# idxs = range(2,len(valid_frames))
# idxs = range(2,23) + [46,47,48,49] + [63,65,66,67] + [81,109,110,111]
# idxs = range(2,23,3)
# idxs = range(2,len(valid_frames)-10)
# idxs = range(30,len(valid_frames)-10)
# idxs = range(2,29) + [32] + range(43,52) + [63,65,66,67] + [81,109,110,111]
# idxs = range(10,26)
idxs = range(2,len(valid_frames)-2)

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
    if bShow:
        cv2.imshow('Frame',frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print mtx
print dist

print len(idxs)
print len(rvecs)
print len(tvecs)

np.savetxt('out.txt', np.append(mtx, dist))


######################### test ##########
objpoints_test = [] # 3d point in real world space
imgpoints_test = [] # 2d points in image plane.
frames_test = []
idxs_test = range(0,len(valid_frames))
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
    # if bShow:
    #     cv2.imshow('Frame',frame)
    #     if cv2.waitKey(delay) & 0xFF == ord('q'):
    #         break

cv2.destroyAllWindows()


cali_ret, _, _, rvecs_test, tvecs_test = cv2.calibrateCamera(objpoints_test, imgpoints_test, gray.shape[::-1],mtx, dist)



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

    k1, k2, p1, p2, k3 = dst
    c = (1 + k1*r**2 + k2*r**4 + k3*r**6)/(1)
    x_new = c*x + 2*p1*x*y + p2*(r**2 + 2*x**2)
    y_new = c*y + p1*(r**2 + 2*y**2) + 2*p2*x*y

    imgp[0,:]= x_new
    imgp[1,:]= y_new
    

    imgp = np.dot(mtx,imgp)[:2,:].T.astype(objp.dtype)
    return imgp


# for frame, rvec, tvec, corners, objp in zip(frames_test, rvecs_test, tvecs_test, imgpoints_test, objpoints_test):
for i in range(len(frames_test)):
    frame, rvec, tvec, corners, objp = frames_test[i], rvecs_test[i], tvecs_test[i], imgpoints_test[i], objpoints_test[i]
    print i
    dist = [np.zeros(5)]
    imgp1 = MyProjection(objp, rvec, tvec, mtx, dist[0])
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






