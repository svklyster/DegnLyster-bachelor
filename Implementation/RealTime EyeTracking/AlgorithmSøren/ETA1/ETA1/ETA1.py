import numpy as np
import cv2
import math
from matplotlib import pyplot as plt



#def findReflections(): #, startingThreshold = None, endingThreshold = None):
#if startingThreshold is None:
image = cv2.imread('test.png',0) 
imgMin, imgMax, minPos, maxPos = cv2.minMaxLoc(image)
startingThreshold = max(imgMax-50,1)
#if endingThreshold is None:
endingThreshold = max(startingThreshold-100,1)

score = [0,0,0,0]

imgHis = cv2.calcHist(image, [0], None, [256], [0,256]) #image, channel = 0 for grayscale, No mask, binsize = 256, range = 0-256
loopRange = []
for i in range(int(startingThreshold), int(endingThreshold), -1):
    if imgHis[i + 1] is not 0:
        loopRange.append(i)
    else: 
        loopRange.append(0)
if len(loopRange) is 0:
    loopRange = startingThreshold
imgThres = []
for iThreshold in loopRange:
    imgThres.append( image > iThreshold )
    imgT = np.array(imgThres)
    imgT = 1*imgT
    cc = cv2.findContours(imgT,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(cc) < 4:
        continue
    else:
        break
    areas = []
    for i in range(len(cc)):
        areas.append(cv2.contourArea(cc[i]))
    cornealReflectionArea = max(areas)
    score[0] = cornealReflectionArea/(sum(areas) - cornealReflectionArea)
    if score[1] is not score[0]:
        score[3] = score[2]
        score[2] = score[1]
        score[1] = score[0]
    if score[3] > score[2] and score[2] > score[1]:
        break





#while(1):
#    cv2.imshow("Input", image)
#    cv2.imshow("Histogram", imgHis)
#    cv2.waitKey(10)