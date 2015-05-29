import codecs
import _tkinter as tk
import os

paraNames = ["SESSIONPATH","USINGCAM","CAMNR",
                 "VIDEOPATH","NOTES","RESOLUTION",
                 "CALTYPE","LOGFILENAME","CALFILENAME",
                 "LOADEDCALDATA","RECORDVIDEO", "RAWDATAPATH",
                 "VARIABLENAMES", "VARIABLEVALUES"]

class SessionData:

    def __init__(self, pathname):
        self.pathname = pathname.encode("latin-1").rstrip()
        self.filepath = os.path.abspath(self.pathname + "/" + "session.pref")
        self.livecam = False
        self.camnr = 0
        self.videopath = None
        self.notes = None
        self.resolution = None
        self.caltype = None
        self.logfilename = None
        self.calfilename = None
        self.calfile = None
        self.recordvideo = False
        self.rawdatapath = None
        self.variablenames = [None, None, None, None, None, None, None, None, None, None]
        self.variablevalues = [None, None, None, None, None, None, None, None, None, None]
        

    
    def CreateSessionFile(self):
        file = codecs.open(self.filepath, "w", "utf-8")
        try:
            file.write("")
            return "fileCreated"
        except: 
            e = sys.exc_info()[0]
            return e

    def UpdateSessionFile(self):
        file = codecs.open(self.filepath, "w", "utf-8")
        sessionStr = "SESSIONPATH " + self.pathname + os.linesep
        sessionStr += "USINGCAM " + str(self.livecam) + os.linesep
        sessionStr += "CAMNR " + str(self.camnr) + os.linesep
        if self.videopath is None:
            sessionStr += "VIDEOPATH None" + os.linesep
        else:
            sessionStr +=  "VIDEOPATH " + self.videopath + os.linesep 
        sessionStr += "NOTES " + self.notes + os.linesep 
        sessionStr += "RESOLUTION " + self.resolution + os.linesep 
        sessionStr += "CALTYPE " + self.caltype + os.linesep 
        sessionStr += "LOGFILENAME " + self.logfilename + os.linesep 
        sessionStr += "CALFILENAME " + self.calfilename + os.linesep 
        sessionStr += "LOADEDCALDATA " + str(self.calfile) + os.linesep
        sessionStr += "RECORDVIDEO " + str(self.recordvideo) + os.linesep
        sessionStr += "RAWDATAPATH " + str(self.rawdatapath) + os.linesep 
        sessionStr += "VARIABLENAMES " + str(self.variablenames) + os.linesep 
        sessionStr += "VARIABLEVALUES " + str(self.variablevalues) + os.linesep 
        try:
            file.write(sessionStr)    
            return "fileUpdated"
            #tkMessageBox.showinfo("Succes", "Session created")
        except: 
            e = sys.exc_info()[0]
            return e

    def SavePreferences(self):
        file = codecs.open(self.filepath, "w", "utf-8")
        sessionStr = "SESSIONPATH " + self.pathname + os.linesep
        sessionStr += "USINGCAM " + str(self.livecam) + os.linesep
        sessionStr += "CAMNR " + str(self.camnr) + os.linesep
        if self.videopath is None:
            sessionStr += "VIDEOPATH None" + os.linesep
        else:
            sessionStr +=  "VIDEOPATH " + str(self.videopath) + os.linesep
        sessionStr += "NOTES " + str(self.notes) + os.linesep 
        sessionStr += "RESOLUTION " + str(self.resolution) + os.linesep 
        sessionStr += "CALTYPE " + str(self.caltype) + os.linesep 
        sessionStr += "LOGFILENAME " + str(self.logfilename) + os.linesep 
        sessionStr += "CALFILENAME " + str(self.calfilename) + os.linesep 
        sessionStr += "LOADEDCALDATA " + str(self.calfile) + os.linesep
        sessionStr += "RECORDVIDEO " + str(self.recordvideo) + os.linesep
        sessionStr += "RAWDATAPATH " + str(self.rawdatapath) + os.linesep 
        sessionStr += "VARIABLENAMES " + str(self.variablenames) + os.linesep 
        sessionStr += "VARIABLEVALUES " + str(self.variablevalues) + os.linesep 
        #sessionStr += "VARIABLEVALUES " + "["
        #for i in range(10):
        #    sessionStr += self.variablevalues[i] + ","
        
       
        try:
            file.write(sessionStr)
            return "fileCreated"
            #tkMessageBox.showinfo("Succes", "Session created")
        except: 
            e = sys.exc_info()[0]
            return e

def LoadPreferences(filepath):
    file = codecs.open(filepath, "r", "utf-8")
    filestr = file.read()
    if all(names in filestr for names in paraNames):
        return "fileVerified"
    else:
        return "File not valid"

    
def LoadPreferencesFromFile(filepath):
    file = codecs.open(filepath, "r", "utf-8")
    tempPathname = file.readline()[12: ]
    newPrefData = SessionData(tempPathname)
    newPrefData.filepath = filepath
    if file.readline()[9: ] is "True":
        newPrefData.livecam = True
    else:
        newPrefData.livecam = False
    newPrefData.camnr = file.readline()[6: ]
    tempVidpath = file.readline()[10: ] 
    if tempVidpath is "None":
        newPrefData.videopath = None
    else:
        newPrefData.videopath = tempVidpath
    newPrefData.notes = file.readline()[6: ]
    newPrefData.resolution = file.readline()[11: ]
    newPrefData.caltype = file.readline()[8: ]
    newPrefData.logfilename = file.readline()[12: ]
    newPrefData.calfilename = file.readline()[12: ]
    newPrefData.calfile = file.readline()[14: ]
    newPrefData.recordvideo = file.readline()[12: ]
    newPrefData.rawdatapath = file.readline()[12: ]
    newPrefData.variablenames = file.readline()[14: ]
    newPrefData.variablevalues = file.readline()[15: ]
    return newPrefData