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

nFrame4calib = 30
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
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, nCorners, corners, find)
            
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
rot, _ = cv2.Rodrigues(rvecs[0])
# print cali_ret
# print mtx
# print [tvec.T for tvec in tvecs]
print len(tvecs)
# print rvecs
# print "Rodrigues",rot
# print tvecs
# print dist
# x = np.dot(rot,objp.T) + tvecs[0]
# print tvecs[0].shape


# nFrame = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         nFrame = nFrame + 1
#         if nFrame % fRate4calib != 0:
#             continue
#     	print "frame ",nFrame
    	
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         find, corners = cv2.findChessboardCorners(gray, nCorners, None)
        
#         if find == True:
#             # objpoints = []
#             objpoints.append(objp)
#             cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#             # imgpoints = []
#             imgpoints.append(corners)
#             cv2.drawChessboardCorners(frame, nCorners, corners, find)
            
#             cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#             rot, _ = cv2.Rodrigues(rvecs[0])
#             # print cali_ret
#             # print mtx
#             print [tvec.T for tvec in tvecs]
#             # print rvecs
#             # print "Rodrigues",rot
#             # print tvecs
#             # print dist
#             # x = np.dot(rot,objp.T) + tvecs[0]
#             # print tvecs[0].shape

#             cv2.imshow('Frame',frame)
#         else:
#             cv2.imshow('Frame',frame)
     
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#            break
#         if nFrame == 20:
#             break

#     else: 
#         break
    
# cap.release()
# cv2.destroyAllWindows()



# print len(objpoints)
# print objpoints[0].shape
# print objpoints[0]

# img = cv2.imread('left12.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)


# # undistort
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)

# mean_error = 0
# for i in xrange(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     tot_error += error

# print "total error: ", mean_error/len(objpoints)



