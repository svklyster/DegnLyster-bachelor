import VideoCapture as vc
import tkMessageBox
import sys
import time
import datetime
import threading


running = False
capture = None

class EyeTrackingThread(threading.Thread):
    def __init__(self, threadID, name, framerate):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.framerate = framerate
    def run(self):
        capture.updateVideo()
        time.sleep(1/self.framerate)
        self.run()
        
        

def StartVideoCapture(sessionData):
    global running, capture
    try:
                                    ####### TEST#####
        #capture = vc.VideoCapture(sessionData.livecam, sessionData.camnr, sessionData.videopath, True)
        capture = vc.VideoCapture(False, sessionData.camnr, "C:/1min60fps.avi" , True)
        #trackingThread = threading.Thread(None, EyeTrackingThread, capture.framerate, {})
        #trackingThread = thread.start_new_thread(EyeTrackingThread,(vc.framerate))
        trackingThread = EyeTrackingThread(1, "ETthread1", capture.framerate)
        trackingThread.start()
        running = True
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
        #mil = int(round(time.time()*1000))
        #t = time.strftime("%H:%M:%S")+":%d" %mil
        t = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print("%s - %d, %d - Trigger=%s" %(t,(crx,cry),e_center,trigger))

def ReturnError():
    t = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("%s - %s" %(t,"Error"))
