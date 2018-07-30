# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

import cv2
import numpy as np
from collections import deque

# lsFilename = ['video/cam2-1.MP4', 'video/cam2-2.MP4',
#     'video/cam2-3.MP4', 'video/cam2-4.MP4', 'video/cam2-5.MP4']

lsFilename = ['video/cam3-1.MP4', 'video/cam3-2.MP4',
    'video/cam3-3.MP4', 'video/cam3-4.MP4', 'video/cam2-5.MP4']


nFrame4calib = 10
bShow = True
# bShow = False
bRandomSample = True
# bRandomSample = False
delay = 150

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
        continue

    nFrame += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


if nFrame <= nFrame4calib:
    idxs = range(nFrame)
else:
    if bRandomSample:
        idxs = np.sort(np.random.choice(nFrame, nFrame4calib, replace=False)).tolist()
    else:
        idxs = np.linspace(0,nFrame-1,nFrame4calib).astype(int).tolist()

print len(idxs)

qFrame = deque(maxlen = 20)
idxCurr = 0
idxTar = idxs.pop(0)

nToBe = 0
for filename in lsFilename:
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print w,h
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if idxCurr == idxTar:
                nToBe += 1
                if len(idxs) > 0:
                    idxTar = idxs[0]
                    idxs.pop(0)
                else:
                    idxTar = -1
            idxCurr += 1
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            qFrame.append(gray)
            
            while nToBe > 0 and qFrame:
                find = False
                while find == False and qFrame:
                    find, corners = cv2.findChessboardCorners(qFrame.pop(), nCorners, None)

                if find == True:

                    nToBe -= 1   
                    objpoints.append(objp)
                    corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(frame, nCorners, corners, find)
                    frames.append(frame) 
                    print "frame ",idxCurr

                    if bShow:
                        cv2.imshow('Frame',frame)
                        if cv2.waitKey(delay) & 0xFF == ord('q'):
                           break
        else: 
            break
    cap.release()

cv2.destroyAllWindows()

cali_ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print mtx
print dist

np.savetxt('out.txt', np.append(mtx, dist))


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


for frame, rvec, tvec, corners, objp in zip(frames, rvecs, tvecs, imgpoints, objpoints):
    
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



# # # import sys
# # # import os
# # # import copy
# # # import subprocess
# # # import matplotlib.pyplot as plt
# # # import matplotlib.patches as patches
# # # import numpy as np
# # # import scipy
# # # import wave
# # # import pyaudio
# # # import struct

# # # from PyQt4 import QtGui
# # # from PyQt4.QtCore import QTimer, QEvent, Qt

# # # from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# # # from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
# # # from matplotlib.figure import Figure

# # # import random
# # # import cv2



# # # class Window(QtGui.QDialog):
# # #     def __init__(self, parent=None):
# # #         super(Window, self).__init__(parent)
# # #         self.setWindowTitle("Calib-with-mp4")
# # #         w = 1280; h = 720
# # #         self.resize(w, h)
# # #         self.setAcceptDrops(True)

# # #         self.menuBar = QtGui.QMenuBar(self)     
# # #         self.menuBar.setNativeMenuBar(False)
# # #         menuFile = self.menuBar.addMenu('File')

# # #         actOpen = QtGui.QAction('Open', self)
# # #         actOpen.setShortcut("Ctrl+O")
# # #         actOpen.triggered.connect(self.openFiles)
# # #         menuFile.addAction(actOpen)

# # #         actExit = QtGui.QAction('Exit', self)
# # #         actExit.setShortcut("Ctrl+Q")
# # #         actExit.triggered.connect(exit)
# # #         menuFile.addAction(actExit)

# # #         self.figure = Figure()
# # #         self.canvas = FigureCanvas(self.figure)

# # #         self.btnSync = QtGui.QPushButton('Sync')
# # #         # self.btnSync.clicked.connect(self.sync)
# # #         self.cbBlank = QtGui.QCheckBox("Insert Blank")

# # #         layoutControl = QtGui.QGridLayout()
# # #         layoutControl.addWidget(self.btnSync,0,0,1,1)
# # #         layoutControl.addWidget(self.cbBlank,4,0,1,1)
        
# # #         self.cbHarris = QtGui.QCheckBox("Harris")
# # #         self.cbHarris.stateChanged.connect(lambda:self.evCheckBox(self.cbHarris))

# # #         self.edt = QtGui.QPlainTextEdit()
# # #         self.edt.setDisabled(True)
# # #         self.edt.setMaximumBlockCount(10)
                    
# # #         self.listFile = QtGui.QListWidget()
# # #         self.listFile.installEventFilter(self)
# # #         self.listFile.setFixedWidth(100)

# # #         layout = QtGui.QGridLayout()
# # #         layout.addWidget(self.menuBar,0,0,1,3)
# # #         layout.addWidget(self.canvas,1,0,1,3)
# # #         layout.addLayout(layoutControl,2,0,1,1)
# # #         layout.addWidget(self.listFile,2,1,1,1)
# # #         layout.addWidget(self.edt,2,2,1,1)

# # #         self.setLayout(layout)
# # #         self.lsMp4 = []
# # #         self.dictWav = {}
# # #         self.bClick = False
# # #         self.lsSplitPosition = []
# # #         self.ax = self.figure.add_subplot(111)



# # #     def eventFilter(self, obj, event):
# # #         if event.type() == QEvent.KeyPress and obj == self.listFile:
# # #             if event.key() == Qt.Key_Delete:
# # #                 listItems=self.listFile.selectedItems()
# # #                 if not listItems: return        
# # #                 for item in listItems:
# # #                     self.listFile.takeItem(self.listFile.row(item))
# # #                     for mp4 in self.lsMp4:
# # #                         if mp4['name'] == item.text():
# # #                             self.lsMp4.remove(mp4)
# # #                             break
# # #                 self.plot()         
# # #             return super(Window, self).eventFilter(obj, event)
# # #         else:
# # #             return super(Window, self).eventFilter(obj, event)

# # #     def dragEnterEvent(self, event):
# # #         if event.mimeData().hasUrls():
# # #             event.accept()
# # #         else:
# # #             event.ignore()

# # #     def dropEvent(self, event):
# # #         lsUrl = [unicode(u.toLocalFile()) for u in event.mimeData().urls()]
# # #         for url in lsUrl:
# # #             mp4 = self.loadMp4(url)
# # #             if mp4:
# # #                 self.lsMp4.append(mp4)
# # #                 item = QtGui.QListWidgetItem(mp4['name'])
# # #                 self.listFile .addItem(item)
# # #         self.keyPlot = 'wav'    
# # #         self.plot()


# # #     def openFiles(self):
# # #         dlg = QtGui.QFileDialog()
# # #         dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
# # #         dlg.setDirectory(os.getcwd())
# # #         dlg.setFilter("Text files (*.mp4)")
        

# # #         if dlg.exec_():
# # #             lsUrl = dlg.selectedFiles()
# # #             for url in lsUrl:
# # #                 pass
# # #                 # mp4 = self.loadMp4(str(url))
# # #                 # if mp4:
# # #                 #     self.lsMp4.append(mp4)
# # #                 #     item = QtGui.QListWidgetItem(mp4['name'])
# # #                 #     self.listFile .addItem(item)
# # #                 # self.keyPlot = 'wav'    
# # #                 # self.plot()

# # #     def loadMp4(self, url):
# # #         pass
# # #         # if url in [mp4['mp4-file'] for mp4 in self.lsMp4]:
# # #         #     return
# # #         # strBase = os.path.basename(url)
# # #         # strFilename, strExtension = os.path.splitext(strBase)
# # #         # if strExtension.lower() != ".mp4":
# # #         #     return
# # #         # strFileWav = os.path.join("wav", strFilename + ".wav")
# # #         # command = "ffmpeg -n -i " + url + " -ac 1 -vn "+ strFileWav
# # #         # # subprocess.call(command, shell=True)
# # #         # if os.path.isfile(strFileWav):
# # #         #     wavfile = wave.open(strFileWav,'r')
# # #         #     numCh = wavfile.getnchannels()
# # #         #     wav = np.fromstring( wavfile.readframes(-1) , 'Int16' ).reshape(-1, numCh).mean(1)
# # #         #     fr = float(wavfile.getframerate())
# # #         #     sigWav = MySignal(x=wav, f = fr)
# # #         #     mp4 = {'mp4-file':url, 'wav-file':strFileWav, 'wav':sigWav, 'name':strFilename}
# # #         #     return mp4
# # #         # return

# # #     def plot(self):
# # #         pass
# # #         # key = self.keyPlot
# # #         # if key == None:
# # #         #     return      
# # #         # self.ax.clear()

# # #         # lsLegend = []
# # #         # for mp4 in self.lsMp4:
# # #         #     step = 100
# # #         #     legend, = self.ax.plot(mp4[key].getTimeAxis()[::step], mp4[key].x[::step], label=mp4['name'])
# # #         #     lsLegend.append(legend)
        
# # #         # self.ax.legend(handles=lsLegend)
# # #         # self.ax.set_xlabel('t(sec)')
# # #         # self.canvas.draw()

# # #     def sync(self):
# # #         pass

    
# # #     def generate(self):
# # #         self.generateSegmentedVideos()



# # #     def generateSegmentedVideos(self):
# # #         tEndMax = max([mp4['time-end'] for mp4 in self.lsMp4])
# # #         for mp4 in self.lsMp4:
# # #             cap = cv2.VideoCapture(mp4['mp4-file'])
# # #             fps = cap.get(cv2.CAP_PROP_FPS)
# # #             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
# # #             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
# # #             capSize = (w, h)
# # #             fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# # #             strFilename, strExtension = os.path.splitext(os.path.basename(mp4['mp4-file']))

# # #             nFrame = 0
            
# # #             for j in range(len(self.lsSplitPosition)):
# # #                 t = self.lsSplitPosition[j]
# # #                 nFrameEnd = int(fps * t) - int(fps * mp4['time-shift'])
                
# # #                 strNum = '-%03d' % j
# # #                 strFilenameOutput = os.path.join("result", strFilename + strNum + ".mp4")
# # #                 print strFilenameOutput,
# # #                 out = cv2.VideoWriter(strFilenameOutput, fourcc, fps, capSize)

# # #                 if j == 0 and self.cbBlank.isChecked():
# # #                     print 'black %d frames'%int(fps * mp4['time-shift']),
# # #                     for _ in range(int(fps * mp4['time-shift'])):
# # #                         out.write(np.zeros((h,w,3), np.uint8))

# # #                 while(cap.isOpened() and nFrame < nFrameEnd):
# # #                     ret, frame = cap.read()
# # #                     if ret == True:
# # #                         nFrame = nFrame + 1
# # #                         out.write(frame)
# # #                     else:
# # #                         break

# # #                 out.release()
                
# # #                 test = cv2.VideoCapture(strFilenameOutput)
# # #                 print test.get(cv2.CAP_PROP_FRAME_COUNT)
            
# # #             j = len(self.lsSplitPosition)
# # #             strNum = '-%03d' % j
# # #             strFilenameOutput = os.path.join("result", strFilename + strNum + ".mp4")
# # #             print strFilenameOutput,
# # #             out = cv2.VideoWriter(strFilenameOutput, fourcc, fps, capSize)
            
# # #             while(cap.isOpened()):
# # #                 ret, frame = cap.read()
# # #                 if ret == True:
# # #                     nFrame = nFrame + 1
# # #                     out.write(frame)
# # #                 else:
# # #                     break

# # #             if self.cbBlank.isChecked():
# # #                 print 'black %d frames'%int(fps * mp4['time-shift']),
# # #                 for _ in range( int(np.ceil(fps * tEndMax)) - ( nFrame + int(fps * mp4['time-shift']) ) ):
# # #                     out.write(np.zeros((h,w,3), np.uint8))
# # #             out.release()
# # #             test = cv2.VideoCapture(strFilenameOutput)
# # #             print test.get(cv2.CAP_PROP_FRAME_COUNT)
# # #             cap.release()

# # #         sys.stdout.write('\a')
# # #         sys.stdout.flush()
                




# # # if __name__ == '__main__':
# # #     app = QtGui.QApplication(sys.argv)

# # #     main = Window()
# # #     main.show()

# # #     sys.exit(app.exec_())


