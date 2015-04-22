import codecs
import Tkinter as tk
import os

paraNames = ["SESSIONPATH","USINGCAM","CAMNR",
                 "VIDEOPATH","NOTES"]

class SessionData:

    def __init__(self, pathname):
        self.pathname = pathname
        self.filepath = os.path.abspath(self.pathname + "/" + "session")
        self.livecam = True
        self.camnr = 0
        self.videopath = None
        self.notes = None
    
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
        sessionStr = "SESSIONPATH " + self.pathname + '\n'
        sessionStr += "USINGCAM " + str(self.livecam) + '\n'
        sessionStr += "CAMNR " + str(self.camnr) + '\n'
        if self.videopath is None:
            sessionStr += "VIDEOPATH None" + '\n'
        else:
            sessionStr +=  "VIDEOPATH " + self.videopath + '\n'
        sessionStr += "NOTES " + self.notes + '\n'
        try:
            file.write(sessionStr)
            return "fileUpdated"
            #tkMessageBox.showinfo("Succes", "Session created")
        except: 
            e = sys.exc_info()[0]
            return e

    def SavePreferences(self):
        file = codecs.open(self.filepath, "w", "utf-8")
        sessionStr = "SESSIONPATH " + self.pathname + '\n'
        sessionStr += "USINGCAM " + str(self.livecam) + '\n'
        sessionStr += "CAMNR " + str(self.camnr) + '\n'
        if self.videopath is None:
            sessionStr += "VIDEOPATH None" + '\n'
        else:
            sessionStr +=  "VIDEOPATH " + self.videopath + '\n'
        sessionStr += "NOTES " + self.notes + '\n'
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

    return newPrefData