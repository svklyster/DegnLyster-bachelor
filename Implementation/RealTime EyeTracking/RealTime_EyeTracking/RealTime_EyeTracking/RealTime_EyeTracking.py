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
        varVid = True
        vVideo2.set("Camera index")
        eVideo.delete(0,tk.END)
        eVideo.insert(0, str(sessionData.camnr))
    else:
        varVid = False
        vVideo2.set("Video source")
        eVideo.delete(0,tk.END)
        eVideo.insert(0, sessionData.videopath)
    
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
    tCalfile.delete("1.0", tk.END)
    tCalfile.insert("1.0", sessionData.calfile)
    tNotePathname.delete("1.0", tk.END)
    tNotePathname.insert("1.0", sessionData.pathname)

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
    lSession = tk.Label(fLeft, text = "Choose video input").pack()
    nSession = ttk.Notebook(fRight)
    #Camera tab
    fCamera = tk.Frame(nSession)
    lCamera = tk.Label(fCamera, text = "Choose camera").pack(side=tk.TOP)
    cameraCount = vc.GetCameraInputs()
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
   
    if tkMessageBox.askokcancel("Calibration", "Start new calibration?") is True:
        LoadCalibrationData() 
    else:
        return
    bStart.config(state = tk.NORMAL)
    print "starting calibration \n"
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
    print "stopping eyetracking \n"
    return

def SavePreferences():
    "Saving preferences"
    #Save preferences from preferences pane
    newPrefData = sh.SessionData(tNotePathname.get("1.0", 'end-1c'))
    newPrefData.livecam = varVid.get()
    if newPrefData.livecam is True:
        newPrefData.camnr = eVideo.get()
        newPrefData.videopath = None
    else:
        newPrefData.camnr = None
        newPrefData.videopath = eVideo.get()
    newPrefData.notes = tNotes.get("1.0", tk.END)
    newPrefData.resolution = tResolution.get("1.0", tk.END)
    newPrefData.caltype = tCaltype.get("1.0", tk.END)
    newPrefData.logfilename = tLogfn.get("1.0", tk.END)
    newPrefData.calfilename = tCalfn.get("1.0", tk.END)
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
    sessionData.livecam = varVid.get()
    if sessionData.livecam is True or 1:
        sessionData.livecam = True
        sessionData.camnr = eVideo.get()
        sessionData.videopath = None
    else:
        sessionData.livecam = False
        sessionData.camnr = None 
        sessionData.videopath = eVideo.get()
    sessionData.notes = tNotes.get("1.0", tk.END)
    sessionData.resolution = tResolution.get("1.0", tk.END)
    sessionData.caltype = tCaltype.get("1.0", tk.END)
    sessionData.logfilename = tLogfn.get("1.0", tk.END)
    sessionData.calfilename = tCalfn.get("1.0", tk.END)
    sessionData.calfile = tCalfile.get("1.0", tk.END)
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
#lVideo1 = tk.Label(nFrameVideo, text ="Using camera")
#lVideo1.grid(row=0, column=0)
#   - Checkbutton with variable
varVid = tk.BooleanVar(nFrameVideo)
lVideo1 = tk.Label(nFrameVideo,  text = "Using camera source?")
lVideo1.pack(side=tk.TOP)
cbVideo = tk.Checkbutton(nFrameVideo, variable = varVid, onvalue = True, offvalue = False)
cbVideo.pack(side = tk.TOP)
vVideo2 = tk.StringVar()
vVideo2.set("Source")
lVideo2 = tk.Label(nFrameVideo, textvariable = vVideo2)
lVideo2.pack(side=tk.TOP)
#lVideo2.grid(row=1, column=0)
eVideo = tk.Entry(nFrameVideo, width = 40)
eVideo.pack(side=tk.TOP)
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
notePref.pack(side=tk.TOP)

leftf.pack(side=tk.LEFT)
preff.pack(side=tk.RIGHT)
midf.pack()

root.mainloop()


