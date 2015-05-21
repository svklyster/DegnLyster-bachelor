import EyeTracking as et
import tkMessageBox
import time
import os
import codecs

class LogHandler: 

    def __init__(self, sessionData):
        self.sessionData = sessionData
        self.pathname = sessionData.pathname
        if sessionData.logfilename is None or '\n':
            self.filename = time.strftime("%y%m%d-%H%M", time.localtime()) #year-month-day--hours:minutes"#
        else: 
            self.filename = sessionData.logfilename
        self.filepath = os.path.abspath(self.pathname + "/" + self.filename+".log")
        self.calibData = sessionData.calfile
        self.trackingRunning = False

        #os.chdir(self.pathname)

    def StartNewTracking(self):
        if str(self.sessionData.calfile).strip() == 'None':
            tkMessageBox.showerror("Exception", "EyeTracking not calibrated!")
            return False
        if self.trackingRunning is False:
            self.trackingRunning = et.StartVideoCapture(self.sessionData, self)
        else:
            tkMessageBox.showerror("Exception", "EyeTracking already running!")
        return self.trackingRunning

    def StopTracking(self):

        self.trackingRunning = et.StopVideoCapture()
        return self.trackingRunning

    def LogData(self, log_string):

        file = codecs.open(self.filepath, "a+", "utf-8")
        file.write(log_string + os.linesep)
    