import Tkinter as tk
import ttk
import cv2
import tkFileDialog
import codecs
import os
import tkMessageBox
import thread
import time
import sys
import re
import numpy as np
from cv2 import cv as cv
import ETalgorithm as ET

def median(mylist):
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

#Save video:

#capture = cv2.VideoCapture('test.avi')
#
#ret, frame = capture.read()
#if ret:
#    width = np.size(frame,1)
#    height = np.size(frame,0)
#
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
#video_writer = cv2.VideoWriter('output.avi', -1, 40, (width, height))
#Temp hardcoded values
calNum = 40
calNumCount = 0
calTotal = 9
calTotalCount = 0

flag = True

#while (capture.isOpened()):
    #wait for button:
#    if (flag == False):
#        flag = True

#    if (flag == True):
#        ret, frame = capture.read()
#        if ret:
#            video_writer.write(frame)
#            #cv2.imshow('Video Stream', frame)
#            #cv2.waitKey(0)
#            calNumCount += 1
#
#        if calNumCount >= calNum:
#            flag = False
#            calNumCount = 0
#            calTotalCount += 1
#        #Temp fix, mangler stadig 1 frame :S
#        if (calTotalCount == 8 and calNumCount == 39):
#            calNumCount += 1
#
#    if (calTotalCount >= calTotal):
#        break

#capture.release()
#video_writer.release()
#cv2.destroyAllWindows()

capture = cv2.VideoCapture('output.avi')

retVectors = []
#Get positions FIX IT NAOW - or not
posList = [[9, 9], [992, 9], [9, 592], [992, 592], [500, 9], [500, 592], [9, 300], [992, 300], [500, 300]]

posValues = []
idx = 0
posCount = 0
posNo = []
runningVJ = True
eyeinfo = None
result = []
runVJ = True
last_eyes = [0,0]
new_eyes = [0,0]

while (capture.isOpened()):
    ret, frame = capture.read()
    if ret is True:
        result, last_eyes = ET.Track(frame, runVJ, new_eyes)
        new_eyes = last_eyes
        retVectors.append(result)
        posValues.append(posList[idx])
        posNo.append(idx)
        posCount += 1
        if posCount >= calNum:
            idx += 1
            posCount = 0
    else:
        break

#Get vectors

#Load images, do eyetracking, first left then right
print(retVectors)
nFramesPerCalPoint = np.bincount(posNo)



tempmedLeftX = []
tempmedLeftY = []
tempmedRightX = []
tempmedRightY = []
medLeftX = []
medLeftY = []
medRightX = []
medRightY = []
positions = []
distLX = []
distLY = []
distRX = []
distRY = []
distLRX = []
distLRY = []

sortedDistX = []
sortedDistY = []

for i in range(0,idx+1):
    del positions[:]
    del tempmedLeftX[:]
    del tempmedLeftY[:]
    del tempmedRightX[:]
    del tempmedRightY[:]
    del distLX[:]
    del distLY[:]
    del distRX[:]
    del distRY[:]
    del distLRX[:]
    del distLRY[:]
    for k in range(0,len(posValues)):
        if (posValues[k] == posList[i]):
            if (retVectors[k] != None):
                positions.append(retVectors[k])
    for k in range(1,len(positions)):
        if (positions[k][0][0] != [0.0]):
            tempmedLeftX.append(positions[k][0][0])
    medLeftX.append(np.median(np.array(tempmedLeftX)))
    
    for k in range(1,len(positions)):
        if (positions[k][0][1] != [0.0]):
            tempmedLeftY.append(positions[k][0][1])
    medLeftY.append(np.median(np.array(tempmedLeftY)))

    for k in range(1,len(positions)):
        if (positions[k][1][0] != [0.0]):
            tempmedRightX.append(positions[k][1][0])
    medRightX.append(np.median(np.array(tempmedRightX)))

    for k in range(1,len(positions)):
        if (positions[k][1][1] != [0.0]):
            tempmedRightY.append(positions[k][1][1])
    medRightY.append(np.median(np.array(tempmedRightY)))


    print(medLeftX)
    print(medLeftY)
    print(medRightX)
    print(medRightY)
    for k in range(1,len(positions)/2):
        if(positions[k][0][0] != [0.0]):
            distLX.append(np.linalg.norm(np.array(positions[k][0][0]-medLeftX[i])))
        if(positions[k][0][1] != [0.0]):
            distLY.append(np.linalg.norm(np.array(positions[k][0][1]-medLeftY[i])))
        if(positions[k][1][0] != [0.0]):
            distRX.append(np.linalg.norm(np.array(positions[k][1][0]-medRightX[i])))
        if(positions[k][1][1] != [0.0]):
            distRY.append(np.linalg.norm(np.array(positions[k][1][1]-medRightY[i])))
        distLRX.append(distLX[k-1] + distRX[k-1])
        distLRY.append(distLY[k-1] + distRY[k-1])

    distLRX.sort()
    distLRY.sort()
    sortedDistX.append(distLRX)
    sortedDistY.append(distLRY)

    A_left = []
    A_right = []

    for k in range(0,len(sortedDistX)):
        A_left.append([positions[k][0][0] * positions[k][0][0], positions[k][0][1] * positions[k][0][1], positions[k][0][0] * positions[k][0][1], positions[k][0][0], positions[k][0][1], 1])
        A_right.append([positions[k][1][0] * positions[k][1][0], positions[k][1][1] * positions[k][1][1], positions[k][1][0] * positions[k][1][1], positions[k][1][0], positions[k][1][1], 1])
        
        print(len(sortedDistX[k]))
        print(len(sortedDistY[k]))

b1 = []
b2 = []

print(A_left)
print(A_right)

ua, sa, va = np.linalg.svd(A_left)

print(np.shape(ua))
print(np.shape(posList))


b1 = np.dot(np.transpose(ua), np.transpose(posList)[:][0])
b1 = b1[0:6]

print(b1)
print(np.size(va))
print(np.size(sa))
calLeftX = np.dot(sa,(np.divide(b1,np.diag(va))))
print(calLeftX)
               
#print(sortedDistX)
#print(sortedDistY)




