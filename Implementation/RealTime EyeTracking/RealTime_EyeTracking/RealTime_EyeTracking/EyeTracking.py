import VideoCapture as vc
import tkMessageBox
import sys
import time
import datetime

running = False
capture = None

def StartVideoCapture(sessionData):
    global running, capture
    try:
        capture = vc.VideoCapture(sessionData.livecam, sessionData.camnr, sessionData.videopath, True)
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

def PackWithTimestamp(x,y,trigger):
        #mil = int(round(time.time()*1000))
        #t = time.strftime("%H:%M:%S")+":%d" %mil
        t = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print("%s - %d, %d - Trigger=%s" %(t,x,y,trigger))

        
