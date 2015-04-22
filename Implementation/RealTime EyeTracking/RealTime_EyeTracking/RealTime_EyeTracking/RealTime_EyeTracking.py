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

root = tk.Tk()
root.title("RealTime EyeTracking")

videoRunning = False
cap = None



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
    
    notes.delete("1.0", tk.END)
    notes.insert("1.0", sessionData.notes)
    tNotePathname.delete("1.0", tk.END)
    tNotePathname.insert("1.0", sessionData.pathname)

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
            result = sessionData.UpdateSessionFile()
            if result is "fileUpdated":
                 tkMessageBox.showinfo("Succes", "Session created")
            else:
                tkMessageBox.showerror("Exception", "Error: %s" % result)
            #if sessionData.livecam is True:
            #    OpenCam(sessionData.camnr)
            #else:
            #    OpenVideo(sessionData.videopath)
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
    #Save preferences from preferences pane
    newPrefData = sh.SessionData(tNotePathname.get("1.0", 'end-1c'))
    newPrefData.livecam = varVid.get()
    if newPrefData.livecam is True:
        newPrefData.camnr = eVideo.get()
        newPrefData.videopath = None
    else:
        newPrefData.camnr = None
        newPrefData.videopath = eVideo.get()
    newPrefData.notes = notes.get("1.0", tk.END)
    newPrefData.filepath = os.path.abspath(tkFileDialog.asksaveasfilename())
    result = newPrefData.SavePreferences()
    if result is "fileCreated":
        tkMessageBox.showinfo("Succes", "Preference file saved")
    else:
        tkMessageBox.showerror("Exception", "Error: %s" % result)
    return

def LoadPreferences():
    "Loading preferences"
    loadPrefFilepath = os.path.abspath(tkFileDialog.askopenfilename())
    fileVerified = sh.LoadPreferences(loadPrefFilepath)
    if fileVerified is "fileVerified":
       UpdateWithSessionData(sh.LoadPreferencesFromFile(loadPrefFilepath))
       tkMessageBox.showinfo("Succes", "Preference file loaded")
    else:
       tkMessageBox.showerror("Exception", "Error: %s" % fileVerified)
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
NoteL1.grid(row=0, column=0)
notes = tk.Text(NoteL1, width = 40, height = 10)
notes.grid(row=0, column=1)
NoteL2 = tk.Label(nFrameNotes, text ="etc")
NoteL2.grid(row=1, column=0)
tNotePathname = tk.Text(NoteL2, width = 40, height = 1)
tNotePathname.grid(row=1, column=1)

#Video
nFrameVideo = tk.Frame(notePref)
#lVideo1 = tk.Label(nFrameVideo, text ="Using camera")
#lVideo1.grid(row=0, column=0)
#   - Checkbutton with variable
varVid = tk.BooleanVar(nFrameVideo)
cbVideo = tk.Checkbutton(nFrameVideo, text = "Using camera source", variable = varVid, onvalue = True, offvalue = False)
cbVideo.grid(row=0, column=1)
vVideo2 = tk.StringVar()
vVideo2.set("Source")
lVideo2 = tk.Label(nFrameVideo, textvariable = vVideo2)
#lVideo2.pack(side=tk.LEFT)
lVideo2.grid(row=1, column=0)
eVideo = tk.Entry(nFrameVideo, width = 40)
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
#   - Notebook
notePref.add(nFrameNotes, text = "Notes")
notePref.add(nFrameVideo, text = "Video")
notePref.pack(side=tk.TOP)

leftf.pack(side=tk.LEFT)
preff.pack(side=tk.RIGHT)
midf.pack()

root.mainloop()


