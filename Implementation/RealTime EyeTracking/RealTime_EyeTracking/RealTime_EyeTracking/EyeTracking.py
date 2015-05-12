import VideoCapture as vc
import tkMessageBox
import sys
import time
import datetime
import threading
#import LogHandler as lh


running = False
capture = None
last_center = None
logHandler = None
calData = None

class EyeTrackingThread(threading.Thread):
    def __init__(self, threadID, name, framerate):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.framerate = framerate
    def run(self):
        global last_center, running
        capture.updateVideo(last_center, calData)
        time.sleep(1/self.framerate)
        if running is True:
            self.run()
        
        

def StartVideoCapture(sessionData, log_handler):
    global running, capture, logHandler
    logHandler = log_handler
    try:
                                    ####### TEST#####
        capture = vc.VideoCapture(sessionData.livecam, sessionData.camnr, sessionData.videopath, True)
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

def PackWithTimestamp(e_center, gaze_vector, trigger):
    global logHandler
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    logHandler.LogData("%s, %s, %s, %s, %s" %(t,e_center,gaze_vector[0], gaze_vector[1],trigger))

def ReturnError(error):
    global logHandler
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    logHandler.LogData("%s, %s" %(t,error))

def LastCenter(eCenter):
    global last_center
    last_center = eCenter