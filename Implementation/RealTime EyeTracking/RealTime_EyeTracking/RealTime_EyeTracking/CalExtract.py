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

def calData(result, left):
    global retVectorsL, retVectorsR, posList, posValuesL, posValuesR, posNoL, posNoR, idx
    if left:
        retVectorsL.append(result)
        posValuesL.append(posList[idx])
        posNoL.append(idx)
            
    else:
        retVectorsR.append(result)
        posValuesR.append(posList[idx])
        posNoR.append(idx)
            
    #print('appended:')
    #print(result)

def lastData(e_info, eyes_info, runVJinfo):
    global e_center, last_eyes, runVJ
    e_center = e_info
    last_eyes = eyes_info
    runVJ = True

areas = []

def runCalib(sessionData):
    global capture, video_writer, retVectorsL, retVectorsR, e_center, last_eyes, runVJ, posList, posValuesL, posValuesR, posNoL, posNoR, idx, calNum, calNumCount, calTotal, calTotalCount, done
    #Save video:

    capture = cv2.VideoCapture(sessionData.camnr)
    
    ret, frame = capture.read()
    if ret:
        width = np.size(frame,1)
        height = np.size(frame,0)
    
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video_writer = cv2.VideoWriter('output.avi', -1, 40, (width, height))
    #Temp hardcoded values
    calNum = 40
    calNumCount = 0
    calTotal = 9
    calTotalCount = 0
    done = False

    flag = True

def snapFrames():

    global capture, areas, video_writer, retVectorsL, retVectorsR, e_center, last_eyes, runVJ, posList, posValuesL, posValuesR, posNoL, posNoR, idx, calNum, calNumCount, calTotal, calTotalCount, done

    while (capture.isOpened()):
        ret, frame = capture.read()
        if ret:
            video_writer.write(frame)
            #cv2.imshow('Video Stream', frame)
            #cv2.waitKey(0)
            calNumCount += 1
    
        if calNumCount >= calNum:

            calNumCount = 0
            calTotalCount += 1
            break
        #Temp fix, mangler stadig 1 frame :S
        #if (calTotalCount == 8 and calNumCount == 39):
        #    calNumCount += 1
    
    if (calTotalCount >= calTotal):
        done = True

    if done:
        capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

        capture = cv2.VideoCapture('output.avi')

        retVectorsL = []
        retVectorsR = []
        #Get positions FIX IT NAOW - or not
        #posList = [[9, 9], [992, 9], [9, 592], [992, 592], [500, 9], [500, 592], [9, 300], [992, 300], [500, 300]]
        posList = areas
        posValuesL = []
        posValuesR = []
        idx = 0
        posCount = 0
        posNoL = []
        posNoR = []
        runningVJ = True
        e_center = None
        result = []
        runVJ = True
        last_eyes = [0,0]
        new_eyes = [0,0]

        while (capture.isOpened()):
            ret, frame = capture.read()
            if ret is True:
                ET.Track(frame, e_center, last_eyes, None, runVJ)
                #posValues.append(posList[idx])
                #posNo.append(idx)
                posCount += 1
                if posCount >= calNum:
                    idx += 1
                    posCount = 0
            else:
                break

        #Get vectors

        #Load images, do eyetracking, first left then right
        print('left results:')
        print(len(retVectorsL))
        print('right results:')
        print(len(retVectorsR))
        #nFramesPerCalPoint = np.bincount(posNo)

        tempmedLeftX = []
        tempmedLeftY = []
        tempmedRightX = []
        tempmedRightY = []
        medLeftX = []
        medLeftY = []
        medRightX = []
        medRightY = []
        positionsL = []
        positionsR = []
        distLX = []
        distLY = []
        distRX = []
        distRY = []
        distLRX = []
        distLRY = []

        sortedDistX = []
        sortedDistY = []
        calVectL = []
        calVectR = []

        for i in range(0,idx+1):
            del positionsL[:]
            del positionsR[:]
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
            for k in range(0,len(posValuesL)):
                if (posValuesL[k] == posList[i]):
                    #print(k)
                    positionsL.append(retVectorsL[k])
                    calVectL.append(retVectorsL[k])
            for k in range(0,len(posValuesR)):
                if (posValuesR[k] == posList[i]):
                    #print(k)
                    positionsR.append(retVectorsR[k])
                    calVectR.append(retVectorsR[k])
            for k in range(1,len(positionsL)):
                tempmedLeftX.append(positionsL[k][0])
            medLeftX.append(np.median(np.array(tempmedLeftX)))
        
            for k in range(1,len(positionsL)):
                tempmedLeftY.append(positionsL[k][1])
            medLeftY.append(np.median(np.array(tempmedLeftY)))

            for k in range(1,len(positionsR)):
                tempmedRightX.append(positionsR[k][0])
            medRightX.append(np.median(np.array(tempmedRightX)))

            for k in range(1,len(positionsR)):
                tempmedRightY.append(positionsR[k][1])
            medRightY.append(np.median(np.array(tempmedRightY)))


            print(medLeftX)
            print(medLeftY)
            print(medRightX)
            print(medRightY)
            for k in range(1,len(positionsL)/2):
                distLX.append(np.linalg.norm(np.array(positionsL[k][0]-medLeftX[i])))
                distLY.append(np.linalg.norm(np.array(positionsL[k][1]-medLeftY[i])))
                distRX.append(np.linalg.norm(np.array(positionsR[k][0]-medRightX[i])))
                distRY.append(np.linalg.norm(np.array(positionsR[k][1]-medRightY[i])))
                distLRX.append(distLX[k-1] + distRX[k-1])
                distLRY.append(distLY[k-1] + distRY[k-1])

            distLRX.sort()
            distLRY.sort()
            sortedDistX.append(distLRX)
            sortedDistY.append(distLRY)

        A_left = np.zeros((len(calVectL),6))
        A_right = np.zeros((len(calVectR),6))

        for k in range(0,len(calVectL)):
            A_left[k][:] = [calVectL[k][0] * calVectL[k][0], calVectL[k][1] * calVectL[k][1], calVectL[k][0] * calVectL[k][1], calVectL[k][0], calVectL[k][1], 1]
        for k in range(0,len(calVectR)):
            A_right[k][:] = [calVectR[k][0] * calVectR[k][0], calVectR[k][1] * calVectR[k][1], calVectR[k][0] * calVectR[k][1], calVectR[k][0], calVectR[k][1], 1]

        b1 = []
        b2 = []
        print(A_left)
        print(A_right)
        print(np.shape(A_left))
        ua, sa, va = np.linalg.svd(A_left)
        va = va.T
        print(np.shape(ua))
        print(np.shape(posValuesL))


        b1 = np.dot(np.transpose(ua), np.transpose(posValuesL)[:][0])
        b1 = b1[0:6]

        print(b1)
        print(np.size(va))
        print(np.size(sa))
        print(va)
        print(sa)
        calLeftX = np.dot(va,(np.divide(b1,sa)))
        print(calLeftX)

        b2 = np.dot(np.transpose(ua), np.transpose(posValuesL)[:][1])
        b2 = b2[0:6]
        calLeftY = np.dot(va,(np.divide(b2,sa)))

        resultX = []
        resultY = []

        for k in range(0,len(A_left)):
            resultX.append(np.dot(A_left[k], calLeftX))
            resultY.append(np.dot(A_left[k], calLeftY))
    
    
        #result = [A_left[:][0] * calLeftX, A_left[:][1] * calLeftY] 
        print(resultX)
        print(resultY)
        errorX = []
        errorY = []
        for k in range(0,len(resultX)):
            errorX.append(np.subtract(posValuesL[k][0], resultX[k]))
            errorY.append(np.subtract(posValuesL[k][1], resultY[k]))
        
        print(errorX)
        print(errorY)
    

             
    #print(sortedDistX)
    #print(sortedDistY)




