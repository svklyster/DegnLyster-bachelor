import cv2

def GetCameraInputs():
    maxTested = 5 #Assuming no more than 5 camera sources
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