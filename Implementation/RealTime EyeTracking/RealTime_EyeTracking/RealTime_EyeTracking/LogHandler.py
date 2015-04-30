import EyeTracking as et
import tkMessageBox
import time
import os

trackingRunning = False

class LogData: 

    def __init__(self, pathname, calibData, filename = None):
        self.pathname = pathname
        if filename is None:
            self.filename = time.strftime("%y-%m-%d--%H:%M", time.localtime()) #year-month-day--hours:minutes"#
        else: 
            self.filename = filename
        self.filepath = os.path.abspath(self.pathname + "/" + self.filename+".log")
        self.calibData = calibData
        print self.filename


def StartNewTracking(sessionData):
    global trackingRunning
    if trackingRunning is False:
        trackingRunning = et.StartVideoCapture(sessionData)
    else:
        tkMessageBox.showerror("Exception", "EyeTracking already running!")
    return trackingRunning

def StopTracking():
    global trackingRunning
    trackRunning = et.StopVideoCapture()
    return trackingRunning