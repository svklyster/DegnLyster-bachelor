import Tkinter as tk
import ttk
import cv2
import tkFileDialog
import codecs
import os

class SessionData:
    def __init__(self,pathname):
        self.pathname = pathname
        self.livecam = True
        self.camnr = 0
        self.videopath = None
        self.notes = None

    def SaveToPath(self):
        completeName = os.path.abspath(self.pathname + "/session")
        file = codecs.open(completeName, "w", "utf-8")
        sessionStr = "SESSIONPATH " + self.pathname + '\n'
        sessionStr += "USINGCAM " + str(self.livecam) + '\n'
        sessionStr += "CAMNR " + str(self.camnr) + '\n'
        if self.videopath is None:
            sessionStr += "VIDEOPATH None" + '\n'
        else:
            sessionStr +=  "VIDEOPATH " + self.videopath + '\n'
        sessionStr += "NOTES " + self.notes + '\n'
        file.write(sessionStr)
        print file



root = tk.Tk()
root.title("RealTime EyeTracking")

videoRunning = False
cap = None

def CountCameras():
    maxTested = 10
    for i in range(0,maxTested):
        tempCam = cv2.VideoCapture(i)
        if tempCam.isOpened() is False:
            return i
    return maxTested

def updateVideo():
    global cap
    ret, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2.imshow('Video source', frame)
    root.after(10, updateVideo)

def OpenVideo(videopath):
    global cap
    abspath = os.path.normpath(videopath).encode('utf-8')
    cap = cv2.VideoCapture(abspath)
    global running
    running = True
    updateVideo()

def OpenCam(nr):
    global cap
    cap = cv2.VideoCapture(nr)
    global running
    running = True
    updateVideo()

def UpdateWithSessionData(sessionData):
    if sessionData.livecam is True:
        cbVideo.select()
        vVideo2.set("Camera index")
        #lVideo2.text = "Camera index"
        eVideo.delete(0,tk.END)
        eVideo.insert(0, str(sessionData.camnr))
    else:
        cbVideo.deselect()
        vVideo2.set("Video source")
        #lVideo2.text = "Video source"
        eVideo.delete(0,tk.END)
        eVideo.insert(0, sessionData.videopath)
    
    notes.insert("1.0", sessionData.notes)

def CreateSession():
    "Opening session wizard"
    #Function call for note window
    def CreateNoteWindow():
        wNotes = tk.Toplevel()
        wNotes.title("Create new session")
        lNotes = tk.Label(wNotes, text = "Session notes").pack(side=tk.TOP)
        tNotes = tk.Text(wNotes, width=50, height=30)
        tNotes.pack(side=tk.TOP)
        def CloseSessionWizard():
            notes = tNotes.get("1.0", tk.END)
            sessionData.notes = notes
            #print sessionData.printPacked()
            sessionData.SaveToPath()
            if sessionData.livecam is True:
                OpenCam(sessionData.camnr)
            else:
                OpenVideo(sessionData.videopath)
            wNotes.destroy()
            UpdateWithSessionData(sessionData)
            return
        bNotes = tk.Button(wNotes, text = "Done", command = CloseSessionWizard).pack(side = tk.BOTTOM)
        wNotes.focus_set()
        wNotes.grab_set()
    #Creating new window
    pathname = tkFileDialog.askdirectory()
    sessionData = SessionData(pathname)
    print pathname
    wSession = tk.Toplevel()
    wSession.title("Create new session")
    fLeft = tk.Frame(wSession)
    fRight = tk.Frame(wSession)
    lSession = tk.Label(fLeft, text = "Choose video input").pack()
    nSession = ttk.Notebook(fRight)
    #Camera tab
    fCamera = tk.Frame(nSession)
    lCamera = tk.Label(fCamera, text = "Choose camera").pack(side=tk.TOP)
    cameraCount = CountCameras()
    oList = []
    for i in range(0, cameraCount):
        oList.append(i)
    var1= tk.IntVar(fCamera)
    var1.set(0)
    oCamera = tk.OptionMenu(fCamera, var1, *oList).pack(side=tk.TOP)
    def ChooseCamera(var1):
        print "camera number"
        sessionData.livecam = True
        sessionData.camnr = var1
        wSession.destroy()
        CreateNoteWindow()
        return
    bCamera = tk.Button(fCamera, text = "Use camera", command = lambda: ChooseCamera(var1.get())).pack(side=tk.BOTTOM)
    nSession.add(fCamera, text = "Camera")
    #Video tab
    fVideo = tk.Frame(nSession)
    lVideo = tk.Label(fVideo, text = "Choose video file").pack(side=tk.TOP)
    def GetVideoPath():
        sessionData.videopath = tkFileDialog.askopenfilename()
        wSession.focus_set()
        wSession.grab_set()
        #eVideo.delete(0, tk.END)
        #eVideo.insert(0, videopath)
        return
    bVideoFind = tk.Button(fVideo, text = "Browse", command = GetVideoPath).pack(side=tk.TOP)
    def ChooseVideo():
        print "camera path"
        print sessionData.videopath
        sessionData.livecam = False
        wSession.destroy()
        CreateNoteWindow()
        return
    bVideoConfirm = tk.Button(fVideo, text = "Confirm", command = ChooseVideo).pack(side=tk.BOTTOM)
    nSession.add(fVideo, text = "Video")

    nSession.pack()
    fLeft.pack(side=tk.LEFT)
    fRight.pack(side=tk.RIGHT)

    wSession.focus_set()
    wSession.grab_set()

    print "new session \n"
    return

def StartCalibration():
    "Starting calibration routine"
    #Code here...
    print "starting calibration \n"
    return

def StartEyeTracking():
    "Starting eyetracking"
    #Code here...
    print "starting eyetracking \n"
    return

def StopEyeTracking():
    "Stopping eyetracking"
    #Code here...
    print "stopping eyetracking \n"
    return

def SavePreferences():
    "Saving preferences"
    #Code here...
    print "saving preferences \n"
    return

def LoadPreferences():
    "Loading preferences"
    #Code here...
    print "loading preferences \n"
    return


#Frames 
leftf = tk.Frame(root) 
preff = tk.Frame(root) 
midf = tk.Frame(root)

###Setup
#Buttons
bSession = tk.Button(leftf, text = "Create session", command = CreateSession)
bCalib = tk.Button(leftf, text = "Start calibration", command = StartCalibration)
bStart = tk.Button(leftf, text = "Start eyetracking", command = StartEyeTracking)
bStop = tk.Button(leftf, text = "Stop eyetracking", command = StopEyeTracking)

###MainDisplay
#Canvas
cMain = tk.Canvas(midf,height=300, width=400)
testImage = tk.PhotoImage(file = "wink4.gif")
image = cMain.create_image(100 ,100,image=testImage)


###Preferences box
#Buttons
fPrefBut = tk.Frame(preff)
bSave = tk.Button(fPrefBut, text = "Save preferences", command = SavePreferences)
bLoad = tk.Button(fPrefBut, text = "Load preferences", command = LoadPreferences)

#Notebook
#Notes
notePref = ttk.Notebook(preff)
nFrameNotes = tk.Frame(notePref)
NoteL1 = tk.Label(nFrameNotes, text ="Notes")
notes = tk.Text(NoteL1, width = 20, height = 10)
NoteL2 = tk.Label(nFrameNotes, text ="etc")
screenWidth = tk.Entry(NoteL2)
#Video
nFrameVideo = tk.Frame(notePref)
lVideo1 = tk.Label(nFrameVideo, text ="Using camera")
#lVideo1.pack(side=tk.LEFT)
lVideo1.grid(row=0, column=0)
cbVideo = tk.Checkbutton(nFrameVideo)
#cbVideo.pack(side=tk.RIGHT)
cbVideo.grid(row=0, column=1)
vVideo2 = tk.StringVar()
vVideo2.set("Source")
lVideo2 = tk.Label(nFrameVideo, textvariable = vVideo2)
#lVideo2.pack(side=tk.LEFT)
lVideo2.grid(row=1, column=0)
eVideo = tk.Entry(nFrameVideo)
#eVideo.pack(side=tk.RIGHT)
eVideo.grid(row=1, column=1)

###Packing tkinter objects
bSession.pack()
bCalib.pack()
bStart.pack()
bStop.pack()

#Canvas
cMain.pack()

#Preferences
bSave.pack(side=tk.LEFT)
bLoad.pack(side=tk.LEFT)
fPrefBut.pack(side=tk.BOTTOM)
    #Notebook
notes.pack(side=tk.RIGHT)
screenWidth.pack(side=tk.RIGHT)
NoteL1.pack(side=tk.TOP)
NoteL2.pack(side=tk.TOP)

notePref.add(nFrameNotes, text = "Notes")
notePref.add(nFrameVideo, text = "Video")
notePref.pack(side=tk.TOP)

leftf.pack(side=tk.LEFT)
preff.pack(side=tk.RIGHT)
midf.pack()

root.mainloop()


