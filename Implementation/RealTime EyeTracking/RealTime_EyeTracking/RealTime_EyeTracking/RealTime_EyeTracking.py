import Tkinter as tk
import ttk
import cv2
import tkFileDialog
import codecs
import os
import tkMessageBox
import SessionHandler as sh
import VideoCapture as vc
import LogHandler as lh
import EyeTracking as et
import Calibration as cal
import thread
import time
import sys

reload(sys)  
sys.setdefaultencoding('latin-1')

root = tk.Tk()
root.title("RealTime EyeTracking")

trackingRunning = False

#widget bindings (tab, focus, etc...)
def focus_next_window(event):
    event.widget.tk_focusNext().focus()
    return("break")

root.bind_class("<Tab>", focus_next_window)

def UpdateWithSessionData(sessionData):
    if sessionData.livecam is True:
        eVideo1.delete(0,tk.END)
        eVideo1.insert(0, "True")
        vVideo2.set("Camera index")
        eVideo2.delete(0,tk.END)
        eVideo2.insert(0, str(sessionData.camnr))
    else:
        eVideo1.delete(0,tk.END)
        eVideo1.insert(0, "False")
        vVideo2.set("Video source")
        eVideo2.delete(0,tk.END)
        eVideo2.insert(0, sessionData.videopath)
    
    tNotes.delete("1.0", tk.END)
    tNotes.insert("1.0", sessionData.notes)
    tResolution.delete("1.0", tk.END)
    tResolution.insert("1.0", sessionData.resolution)
    tCaltype.delete("1.0", tk.END)
    tCaltype.insert("1.0", sessionData.caltype)
    tLogfn.delete("1.0", tk.END)
    tLogfn.insert("1.0", sessionData.logfilename)
    tCalfn.delete("1.0", tk.END)
    tCalfn.insert("1.0", sessionData.calfilename)
    tNotePathname.delete("1.0", tk.END)
    tNotePathname.insert("1.0", sessionData.pathname)
    tCalfile.delete("1.0", tk.END)
    tCalfile.insert("1.0", str(sessionData.calfile))
    sessionData.variablenames = sessionData.variablenames.split(',')
    tV1name.delete("1.0", tk.END)
    tV1name.insert("1.0", str(sessionData.variablenames[0]).rstrip("'\n\r"))
    tV2name.delete("1.0", tk.END)
    tV2name.insert("1.0", str(sessionData.variablenames[1]).rstrip("'\n\r"))
    tV3name.delete("1.0", tk.END)
    tV3name.insert("1.0", str(sessionData.variablenames[2]).rstrip("'\n\r"))
    tV4name.delete("1.0", tk.END)
    tV4name.insert("1.0", str(sessionData.variablenames[3]).rstrip("'\n\r"))
    tV5name.delete("1.0", tk.END)
    tV5name.insert("1.0", str(sessionData.variablenames[4]).rstrip("'\n\r"))
    tV6name.delete("1.0", tk.END)
    tV6name.insert("1.0", str(sessionData.variablenames[5]).rstrip("'\n\r"))
    tV7name.delete("1.0", tk.END)
    tV7name.insert("1.0", str(sessionData.variablenames[6]).rstrip("'\n\r"))
    tV8name.delete("1.0", tk.END)
    tV8name.insert("1.0", str(sessionData.variablenames[7]).rstrip("'\n\r"))
    tV9name.delete("1.0", tk.END)
    tV9name.insert("1.0", str(sessionData.variablenames[8]).rstrip("'\n\r"))
    tV10name.delete("1.0", tk.END)
    tV10name.insert("1.0", str(sessionData.variablenames[9]).rstrip("'\n\r"))
    sessionData.variablevalues = sessionData.variablevalues.split(',')
    tV1value.delete("1.0", tk.END)
    tV1value.insert("1.0", str(sessionData.variablevalues[0]).rstrip("'\n\r"))
    tV2value.delete("1.0", tk.END)                 
    tV2value.insert("1.0", str(sessionData.variablevalues[1]).rstrip("'\n\r"))
    tV3value.delete("1.0", tk.END)                 
    tV3value.insert("1.0", str(sessionData.variablevalues[2]).rstrip("'\n\r"))
    tV4value.delete("1.0", tk.END)                 
    tV4value.insert("1.0", str(sessionData.variablevalues[3]).rstrip("'\n\r"))
    tV5value.delete("1.0", tk.END)                 
    tV5value.insert("1.0", str(sessionData.variablevalues[4]).rstrip("'\n\r"))
    tV6value.delete("1.0", tk.END)                 
    tV6value.insert("1.0", str(sessionData.variablevalues[5]).rstrip("'\n\r"))
    tV7value.delete("1.0", tk.END)                 
    tV7value.insert("1.0", str(sessionData.variablevalues[6]).rstrip("'\n\r"))
    tV8value.delete("1.0", tk.END)                 
    tV8value.insert("1.0", str(sessionData.variablevalues[7]).rstrip("'\n\r"))
    tV9value.delete("1.0", tk.END)                 
    tV9value.insert("1.0", str(sessionData.variablevalues[8]).rstrip("'\n\r"))
    tV10value.delete("1.0", tk.END)
    tV10value.insert("1.0", str(sessionData.variablevalues[9]).rstrip("'\n\r"))
def CreateSession():
    "Opening session wizard"
    #Function call for note window
    def CreateNoteWindow():
        wNotes = tk.Toplevel()
        wNotes.title("Create new session")
        lNotes = tk.Label(wNotes, text = "Session notes").pack(side=tk.TOP)
        Notes = tk.Text(wNotes, width=50, height=10)
        Notes.pack(side=tk.TOP)
        lResolution = tk.Label(wNotes, text = "Screen 2 resolution").pack(side=tk.TOP)
        Resolution = tk.Text(wNotes, width=50, height = 1)
        Resolution.pack(side=tk.TOP)
        lCaltype = tk.Label(wNotes, text = "Calibration type").pack(side=tk.TOP)
        Caltype = tk.Text(wNotes, width=50, height=1)
        Caltype.pack(side=tk.TOP)
        lOptional = tk.Label(wNotes, text = "Optional:").pack(side=tk.TOP)
        lLogfilename = tk.Label(wNotes, text = "Log filename").pack(side=tk.TOP)
        Logfilename = tk.Text(wNotes, width=50, height = 1)
        Logfilename.pack(side=tk.TOP)
        lCalfilename = tk.Label(wNotes, text = "Calibration log filename").pack(side=tk.TOP)
        Calfilename = tk.Text(wNotes, width=50, height = 1)
        Calfilename.pack(side=tk.TOP)

        def CloseSessionWizard():
            notes = Notes.get("1.0", tk.END)
            sessionData.notes = notes
            sessionData.resolution = Resolution.get("1.0", tk.END)
            sessionData.caltype = Caltype.get("1.0", tk.END)
            sessionData.logfilename = Logfilename.get("1.0", tk.END)
            sessionData.calfilename = Calfilename.get("1.0", tk.END)
            result = sessionData.UpdateSessionFile()
            if result is "fileUpdated":
                 tkMessageBox.showinfo("Succes", "Session created")
            else:
                tkMessageBox.showerror("Exception", "Error: %s" % result)
            bCalib.config(state = tk.NORMAL)
            bStart.config(state = tk.NORMAL)
            wNotes.destroy()
            UpdateWithSessionData(sessionData)
            return
        bNotes = tk.Button(wNotes, text = "Done", command = CloseSessionWizard).pack(side = tk.BOTTOM)
        wNotes.focus_set()
        wNotes.grab_set()
    #Creating new window
    pathname = tkFileDialog.askdirectory()
    sessionData = sh.SessionData(pathname)
    result = sessionData.CreateSessionFile()
    if result is not "fileCreated":
        tkMessageBox.showerror("Exception", "Error: %s" % result)
    wSession = tk.Toplevel()
    wSession.title("Create new session")
    fLeft = tk.Frame(wSession)
    fRight = tk.Frame(wSession)
    lSession = tk.Label(fLeft, text = "Choose video input").pack(padx = 20, pady = 20)
    nSession = ttk.Notebook(fRight)
    #Camera tab
    fCamera = tk.Frame(nSession)
    lCamera = tk.Label(fCamera, text = "Choose camera").pack(side=tk.TOP, padx = 20, pady = 10)
    cameraCount = vc.GetCameraInputs()
    oList = []
    for i in range(0, cameraCount):
        oList.append(i)
    var1= tk.IntVar(fCamera)
    var1.set(0)
    oCamera = tk.OptionMenu(fCamera, var1, *oList)
    oCamera.config(width = 8)
    oCamera.pack(side=tk.TOP, padx = 20, pady = 10)
    def ChooseCamera(var1):
        print "camera number"
        sessionData.livecam = True
        sessionData.camnr = var1
        wSession.destroy()
        CreateNoteWindow()
        return
    bCamera = tk.Button(fCamera, text = "Use camera", command = lambda: ChooseCamera(var1.get())).pack(side=tk.BOTTOM, padx = 20, pady = 10)
    nSession.add(fCamera, text = "Camera")
    #Video tab
    fVideo = tk.Frame(nSession)
    lVideo = tk.Label(fVideo, text = "Choose video file").pack(side=tk.TOP, padx = 20, pady = 10)
    def GetVideoPath():
        sessionData.videopath = tkFileDialog.askopenfilename()
        #wSession.focus_set()
        #wSession.grab_set()
        return
    bVideoFind = tk.Button(fVideo, text = "Browse", command = GetVideoPath).pack(side=tk.TOP, padx = 20, pady = 10)
    def ChooseVideo():
        print "camera path"
        print sessionData.videopath
        sessionData.livecam = False
        wSession.destroy()
        CreateNoteWindow()
        return
    bVideoConfirm = tk.Button(fVideo, text = "Confirm", command = ChooseVideo).pack(side=tk.BOTTOM, padx = 20, pady = 10)
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
   
    if tkMessageBox.askokcancel("Calibration", "Start new calibration?") is True:
        #LoadCalibrationData()
        cal.calScreen()
        sessionData = UpdateSessionWithPreferences()
        cal.runCal(sessionData)   
        #9 punkter
        for i in range(9):
            tkMessageBox.showinfo("Calibration", "Next calibration point")
            cal.snapFrames()     
    else:
        return
    bStart.config(state = tk.NORMAL)
    return

def StartEyeTracking():

    "Starting eyetracking"
    try:
        sessionData = UpdateSessionWithPreferences()
        global trackingRunning, logHandler
        logHandler = lh.LogHandler(sessionData)
        trackingRunning = logHandler.StartNewTracking()
        if trackingRunning is True:
            bStart.config(state = tk.DISABLED)
            bStop.config(state = tk.NORMAL)
            print "starting eyetracking \n"
        return
    except:
        e = sys.exc_info()[0]
        tkMessageBox.showerror("Exception", "Error: %s" % e)
    
def StopEyeTracking():
    "Stopping eyetracking"
    global trackingRunning, logHandler
    trackingRunning = logHandler.StopTracking()
    #if trackingRunning is False:
    bStart.config(state = tk.NORMAL)
    bStop.config(state = tk.DISABLED)
    fLeftEye.configure(bg = '#0000ff')
    fRightEye.configure(bg = '#0000ff')
    cv2.destroyAllWindows()
    return

def SavePreferences():
    "Saving preferences"
    #Save preferences from preferences pane
    newPrefData = sh.SessionData(tNotePathname.get("1.0", 'end-1c'))
    newPrefData.livecam = eVideo1.get()
    if newPrefData.livecam is True:
        newPrefData.camnr = eVideo2.get()
        newPrefData.videopath = None
    else:
        newPrefData.camnr = None
        newPrefData.videopath = eVideo2.get()
    newPrefData.notes = tNotes.get("1.0", tk.END).rstrip('\n')
    newPrefData.resolution = tResolution.get("1.0", tk.END).rstrip('\n')
    newPrefData.caltype = tCaltype.get("1.0", tk.END).rstrip('\n')
    newPrefData.logfilename = tLogfn.get("1.0", tk.END).rstrip('\n')
    newPrefData.calfilename = tCalfn.get("1.0", tk.END).rstrip('\n')
    newPrefData.variablenames[0] = str(tV1name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[1] = str(tV2name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[2] = str(tV3name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[3] = str(tV4name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[4] = str(tV5name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[5] = str(tV6name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[6] = str(tV7name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[7] = str(tV8name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[8] = str(tV9name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablenames[9] = str(tV10name.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[0] = str(tV1value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[1] = str(tV2value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[2] = str(tV3value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[3] = str(tV4value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[4] = str(tV5value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[5] = str(tV6value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[6] = str(tV7value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[7] = str(tV8value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[8] = str(tV9value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.variablevalues[9] = str(tV10value.get("1.0", tk.END)).rstrip('\n')
    newPrefData.filepath = os.path.abspath(tkFileDialog.asksaveasfilename(defaultextension = "pref"))
    result = newPrefData.SavePreferences()
    if result is "fileCreated":
        tkMessageBox.showinfo("Succes", "Preference file saved")
    else:
        tkMessageBox.showerror("Exception", "Error: %s" % result)
    return

def LoadPreferences():
    "Loading preferences"
    loadPrefFilepath = os.path.abspath(tkFileDialog.askopenfilename(filetypes = [('preference files','.pref')]))
    fileVerified = sh.LoadPreferences(loadPrefFilepath)
    if fileVerified is "fileVerified":
       UpdateWithSessionData(sh.LoadPreferencesFromFile(loadPrefFilepath))
       tkMessageBox.showinfo("Succes", "Preference file loaded")
    else:
       tkMessageBox.showerror("Exception", "Error: %s" % fileVerified)
    print "loading preferences \n"
    bStart.config(state = tk.NORMAL)
    bCalib.config(state = tk.NORMAL)
    return

def UpdateSessionWithPreferences():
    sessionData = sh.SessionData(tNotePathname.get("1.0", 'end-1c'))
    print(str(eVideo1.get()))
    if eVideo1.get() is "True":
        sessionData.livecam = True
        sessionData.camnr = eVideo2.get()
        sessionData.videopath = None
    else:
        sessionData.livecam = False
        sessionData.videopath = eVideo2.get()

    sessionData.notes = tNotes.get("1.0", tk.END).rstrip('\n')
    sessionData.resolution = tResolution.get("1.0", tk.END).rstrip('\n')
    sessionData.caltype = tCaltype.get("1.0", tk.END).rstrip('\n')
    sessionData.logfilename = tLogfn.get("1.0", tk.END).rstrip('\n')
    sessionData.calfilename = tCalfn.get("1.0", tk.END).rstrip('\n')
    sessionData.calfile = tCalfile.get("1.0", tk.END).rstrip('\n')
    sessionData.variablenames[0] = str(tV1name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[1] = str(tV2name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[2] = str(tV3name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[3] = str(tV4name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[4] = str(tV5name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[5] = str(tV6name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[6] = str(tV7name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[7] = str(tV8name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[8] = str(tV9name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablenames[9] = str(tV10name.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[0] = str(tV1value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[1] = str(tV2value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[2] = str(tV3value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[3] = str(tV4value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[4] = str(tV5value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[5] = str(tV6value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[6] = str(tV7value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[7] = str(tV8value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[8] = str(tV9value.get("1.0", tk.END)).rstrip('\n')
    sessionData.variablevalues[9] = str(tV10value.get("1.0", tk.END)).rstrip('\n')
    return sessionData
    
def LoadCalibrationData():
    calData = cal.Calibration(tNotePathname.get("1.0", 'end-1c'),"numbers")
    #StartEyeTracking()
    et.GetPoint(UpdateSessionWithPreferences())

def CreateCalibrationLog():
    return

def LoadCalData():
        tCalfile.delete("1.0", tk.END)
        tCalfile.insert("1.0", tkFileDialog.askopenfilename())
        return
#Frames 
leftf = tk.Frame(root) 
preff = tk.Frame(root) 
midf = tk.Frame(root)

###Setup
#Buttons
bSession = tk.Button(leftf, text = "Create session", command = CreateSession)
bCalib = tk.Button(leftf, text = "Start calibration", command = StartCalibration, state=tk.DISABLED)
bStart = tk.Button(leftf, text = "Start eyetracking", command = StartEyeTracking, state=tk.DISABLED)     
bStop = tk.Button(leftf, text = "Stop eyetracking", command = StopEyeTracking, state=tk.DISABLED)

###MainDisplay
#Canvas
cMain = tk.Canvas(midf,height=300, width=400)
testImage = tk.PhotoImage(file = "wink4.gif")
image = cMain.create_image(100 ,100,image=testImage)

#TrackingStatus
fEyes = tk.Frame(midf, height = 50, width=200) 
fRightEye = tk.Frame(fEyes, height=30, width=80, bd=1, relief=tk.SUNKEN, bg = '#0000ff')
fLeftEye = tk.Frame(fEyes, height=30, width=80, bd=1, relief=tk.SUNKEN, bg = '#0000ff')
fRightEye.grid(row=0,column=1,padx = 10, pady = 10)
fLeftEye.grid(row=0,column=0,padx = 10, pady = 10)
fEyes.pack(side=tk.BOTTOM)

###Preferences box
#Buttons
fPrefBut = tk.Frame(preff)
bSave = tk.Button(fPrefBut, text = "Save preferences", command = SavePreferences)
bLoad = tk.Button(fPrefBut, text = "Load preferences", command = LoadPreferences)

#Notebook
#Notes
notePref = ttk.Notebook(preff)
nFrameNotes = tk.Frame(notePref)
lNotes = tk.Label(nFrameNotes, text ="Notes")
lNotes.pack(side=tk.TOP)
tNotes = tk.Text(nFrameNotes, width = 40, height = 10)
tNotes.pack(side=tk.TOP)
lResolution = tk.Label(nFrameNotes, text ="Screen 2 resolution")
lResolution.pack(side=tk.TOP)
tResolution = tk.Text(nFrameNotes, width = 40, height = 1)
tResolution.pack(side=tk.TOP)

lLogfn = tk.Label(nFrameNotes, text ="Log filename")
lLogfn.pack(side=tk.TOP)
tLogfn = tk.Text(nFrameNotes, width = 40, height = 1)
tLogfn.pack(side=tk.TOP)

lPathname = tk.Label(nFrameNotes, text = "Session path")
lPathname.pack(side=tk.TOP)
tNotePathname = tk.Text(nFrameNotes, width = 40, height = 1)
tNotePathname.pack(side=tk.TOP)

#Video
nFrameVideo = tk.Frame(notePref)

lVideo1 = tk.Label(nFrameVideo,  text = "Using camera source?")
lVideo1.pack(side=tk.TOP)
eVideo1 = tk.Entry(nFrameVideo, width = 5)
eVideo1.pack(side = tk.TOP)
vVideo2 = tk.StringVar()
vVideo2.set("Source")
lVideo2 = tk.Label(nFrameVideo, textvariable = vVideo2)
lVideo2.pack(side=tk.TOP)
#lVideo2.grid(row=1, column=0)
eVideo2 = tk.Entry(nFrameVideo, width = 40)
eVideo2.pack(side=tk.TOP)
#eVideo.grid(row=1, column=1)
lRecord = tk.Label(nFrameVideo, text = "Recording video?")
lRecord.pack(side=tk.TOP)
varRec = tk.BooleanVar(nFrameVideo)
cbRecord = tk.Checkbutton(nFrameVideo, variable = varRec, onvalue = True, offvalue = False)
cbRecord.pack(side=tk.TOP)
lRawPath = tk.Label(nFrameVideo, text = "Filepath for recorded data")
lRawPath.pack(side=tk.TOP)
eRawPath = tk.Entry(nFrameVideo, width = 40)
eRawPath.pack(side=tk.TOP)
###Packing tkinter objects
bSession.pack(padx=5, pady=5, fill=tk.X)
bCalib.pack(padx=5, pady=5, fill=tk.X)
bStart.pack(padx=5, pady=5, fill=tk.X)
bStop.pack(padx=5, pady=5, fill=tk.X)

#Calibration 
nFrameCal = tk.Frame(notePref)
lCaltype = tk.Label(nFrameCal, text ="Calibration type")
lCaltype.pack(side=tk.TOP)
tCaltype = tk.Text(nFrameCal, width = 40, height = 1)
tCaltype.pack(side=tk.TOP)
lCalfile = tk.Label(nFrameCal, text = "Load calibration file")
lCalfile.pack(side=tk.TOP)
fCalfile = tk.Frame(nFrameCal, width = 40)
tCalfile = tk.Text(fCalfile, width = 28, height = 1)
tCalfile.pack(side=tk.LEFT)
bCalfile = tk.Button(fCalfile, width = 10, text = "Choose", command=LoadCalData)
bCalfile.pack(side=tk.RIGHT, padx = 9)
fCalfile.pack(side=tk.TOP)
lCalfn = tk.Label(nFrameCal, text ="New calibration log filename")
lCalfn.pack(side=tk.TOP)
tCalfn = tk.Text(nFrameCal, width = 40, height = 1)
tCalfn.pack(side=tk.TOP)

nAlgorithm = tk.Frame(notePref)

lVname = tk.Label(nAlgorithm, text = "Name")
lVvalue = tk.Label(nAlgorithm, text = "Value")
lVname.grid(row = 0, column = 1)
lVvalue.grid(row = 0, column = 2)
lV1 = tk.Label(nAlgorithm, text = "V1")
tV1name = tk.Text(nAlgorithm, width = 20, height = 1)
tV1value = tk.Text(nAlgorithm, width = 20, height = 1)
lV1.grid(row = 1, column = 0)
tV1name.grid(row = 1, column = 1)
tV1value.grid(row = 1, column = 2)
lV2 = tk.Label(nAlgorithm, text = "V2")
tV2name = tk.Text(nAlgorithm, width = 20, height = 1)
tV2value = tk.Text(nAlgorithm, width = 20, height = 1)
lV2.grid(row = 2, column = 0)
tV2name.grid(row = 2, column = 1)
tV2value.grid(row = 2, column = 2)
lV3 = tk.Label(nAlgorithm, text = "V3")
tV3name = tk.Text(nAlgorithm, width = 20, height = 1)
tV3value = tk.Text(nAlgorithm, width = 20, height = 1)
lV3.grid(row = 3, column = 0)
tV3name.grid(row = 3, column = 1)
tV3value.grid(row = 3, column = 2)
lV4 = tk.Label(nAlgorithm, text = "V4")
tV4name = tk.Text(nAlgorithm, width = 20, height = 1)
tV4value = tk.Text(nAlgorithm, width = 20, height = 1)
lV4.grid(row = 4, column = 0)
tV4name.grid(row = 4, column = 1)
tV4value.grid(row = 4, column = 2)
lV5 = tk.Label(nAlgorithm, text = "V5")
tV5name = tk.Text(nAlgorithm, width = 20, height = 1)
tV5value = tk.Text(nAlgorithm, width = 20, height = 1)
lV5.grid(row = 5, column = 0)
tV5name.grid(row = 5, column = 1)
tV5value.grid(row = 5, column = 2)
lV6 = tk.Label(nAlgorithm, text = "V6")
tV6name = tk.Text(nAlgorithm, width = 20, height = 1)
tV6value = tk.Text(nAlgorithm, width = 20, height = 1)
lV6.grid(row = 6, column = 0)
tV6name.grid(row = 6, column = 1)
tV6value.grid(row = 6, column = 2)
lV7 = tk.Label(nAlgorithm, text = "V7")
tV7name = tk.Text(nAlgorithm, width = 20, height = 1)
tV7value = tk.Text(nAlgorithm, width = 20, height = 1)
lV7.grid(row = 7, column = 0)
tV7name.grid(row = 7, column = 1)
tV7value.grid(row = 7, column = 2)
lV8 = tk.Label(nAlgorithm, text = "V8")
tV8name = tk.Text(nAlgorithm, width = 20, height = 1)
tV8value = tk.Text(nAlgorithm, width = 20, height = 1)
lV8.grid(row = 8, column = 0)
tV8name.grid(row = 8, column = 1)
tV8value.grid(row = 8, column = 2)
lV9 = tk.Label(nAlgorithm, text = "V9")
tV9name = tk.Text(nAlgorithm, width = 20, height = 1)
tV9value = tk.Text(nAlgorithm, width = 20, height = 1)
lV9.grid(row = 9, column = 0)
tV9name.grid(row = 9, column = 1)
tV9value.grid(row = 9, column = 2)
lV10 = tk.Label(nAlgorithm, text = "V10")
tV10name = tk.Text(nAlgorithm, width = 20, height = 1)
tV10value = tk.Text(nAlgorithm, width = 20, height = 1)
lV10.grid(row = 10, column = 0)
tV10name.grid(row = 10, column = 1)
tV10value.grid(row = 10, column = 2)
#Canvas
cMain.pack(side = tk.TOP, pady = 10, padx = 10)



#Preferences
bSave.pack(padx=5, pady=5, side=tk.LEFT)
bLoad.pack(padx=5, pady=5, side=tk.LEFT)
fPrefBut.pack(side=tk.BOTTOM)
#   - Notebook
notePref.add(nFrameNotes, text = "Notes")
notePref.add(nFrameVideo, text = "Video")
notePref.add(nFrameCal, text = "Calibration")
notePref.add(nAlgorithm, text = "Algorithm")
notePref.pack(side=tk.TOP)

leftf.pack(side=tk.LEFT)
preff.pack(side=tk.RIGHT)
midf.pack()

def checkEyesFound():
    global trackingRunning
    if trackingRunning is True:
        if et.eyesFoundCount[0] > 10:
            fRightEye.configure(bg = '#ff0000')
        else:
            fRightEye.configure(bg = '#00ff00')
        if et.eyesFoundCount[1] > 10:
            fLeftEye.configure(bg = '#ff0000')
        else:
            fLeftEye.configure(bg = '#00ff00')
    else:
        fLeftEye.configure(bg = '#0000ff')
        fRightEye.configure(bg = '#0000ff')
    root.after(100, checkEyesFound)


root.after(100, checkEyesFound)
root.mainloop()


