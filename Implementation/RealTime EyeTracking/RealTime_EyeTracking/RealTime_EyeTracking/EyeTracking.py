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

class EyeTrackingThread(threading.Thread):
    def __init__(self, threadID, name, framerate):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.framerate = framerate
    def run(self):
        global last_center
        capture.updateVideo(last_center)
        time.sleep(1/self.framerate)
        self.run()
        
        

def StartVideoCapture(sessionData, log_handler):
    global running, capture, logHandler
    logHandler = log_handler
    try:
                                    ####### TEST#####
        #capture = vc.VideoCapture(sessionData.livecam, sessionData.camnr, sessionData.videopath, True)
        capture = vc.VideoCapture(False, sessionData.camnr, "C:/1min60fps.avi" , True)
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
        capture.StopTracking()
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

def PackWithTimestamp((crx,cry), e_center, trigger):
    global logHandler
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    logHandler.LogData("%s - %s, %s, %s, %s - Trigger=%s" %(t,crx[0],cry[0],e_center[0].real, e_center[1].real,trigger))

def ReturnError(error):
    global logHandler
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    logHandler.LogData("%s - %s" %(t,error))

def LastCenter(eCenter):
    global last_center
    last_center = eCenter