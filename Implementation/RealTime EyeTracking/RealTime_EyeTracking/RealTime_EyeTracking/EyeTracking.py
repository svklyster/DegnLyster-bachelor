import VideoCapture as vc
import tkMessageBox
import sys
import time
import datetime
import threading
import LogHandler as lh
import Calibration as cb


running = False
capture = None
last_center = None
logHandler = None
calData = None
eyesFoundCount = [0,0]
vjCount = 0
last_eyes = None
saveRaw = False

# CALIBRATION STUFF #####

calNum = 40
calNumCount = 0
calTotal = 9
calTotalCount = 0

retVectors = []
#Get positions FIX IT NAOW - or not
posList = [[9, 9], [992, 9], [9, 592], [992, 592], [500, 9], [500, 592], [9, 300], [992, 300], [500, 300]]
posValues = []
idx = 0
posCount = 0
posNo = []
result = []

########################


class EyeTrackingThread(threading.Thread):
    def __init__(self, threadID, name, framerate):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.framerate = framerate
    def run(self):
        global last_center, last_eyes, running, vjCount, calData
        if vjCount is 0:
            runVJ = True
            vjCount = 10
        else:
            runVJ = False
            vjCount -= 1

        capture.updateVideo(last_center, last_eyes, calData, runVJ)
        time.sleep(1/self.framerate)
        if running is True:
            self.run()
        
        

def StartVideoCapture(sessionData, log_handler):
    global running, capture, logHandler
    logHandler = log_handler
    try:
                                    ####### TEST#####
        capture = vc.VideoCapture(sessionData.livecam, sessionData.camnr, sessionData.videopath, True, saveRaw)
        #capture = vc.VideoCapture(False, sessionData.camnr, "C:/1min60fps.avi" , True)
        #trackingThread = threading.Thread(None, EyeTrackingThread, capture.framerate, {})
        #trackingThread = thread.start_new_thread(EyeTrackingThread,(vc.framerate))
        trackingThread = EyeTrackingThread(1, "ETthread1", capture.framerate)
        trackingThread.start()
        running = True
        #loghandler
        return running
    except:
        e = sys.exc_info()[0]
        tkMessageBox.showerror("Exception", "Error: %s" % e)
        running = False
        return running

def StopVideoCapture():
    global running, capture
    if capture is not None:
        running = capture.StopTracking()
        #del capture
    else:
        tkMessageBox.showerror("Error", "No running video capture")
    return running

def GetPoint(sessionData):
    global running, capture
    if running is True:
        x,y,trigger = capture.CaptureFrame()
        print x
        print y 
        print trigger
    else:
        StartVideoCapture(sessionData)
        GetPoint(sessionData)

def PackWithTimestamp(e_center, gaze_vector, trigger, calibration):
    global logHandler
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    if calibration is False:
        logHandler.LogData("%s, %s, %s, %s, %s" %(t,e_center,gaze_vector[0], gaze_vector[1],trigger))
    else:
        retVectors.append(gaze_vector)
        posValues.append(posList[idx])
        posNo.append(idx)
        posCount += 1
        if posCount >= calNum:
            idx += 1
            posCount = 0


def ReturnError(error):
    global logHandler
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    logHandler.LogData("%s, %s" %(t,error))

def LastRunInfo(eCenter, eyes):
    global last_center, last_eyes
    last_center = eCenter
    last_eyes = eyes

def EyesFound(e_found):
    global eyesFoundCount
    if e_found[0] > 0:
        eyesFoundCount[0] = 0
    else: 
        eyesFoundCount[0] += 1
    if e_found[1] > 0:
        eyesFoundCount[1] = 0
    else:
        eyesFoundCount[1] += 1

