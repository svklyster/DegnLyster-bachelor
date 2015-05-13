import cv2
import os
import ETalgorithm as ET


class VideoCapture: 

    def __init__(self, livecam, camnr, videopath, calData=None, tracking=None): 
        if livecam is True:
            self.cap = cv2.VideoCapture(int(camnr))
        else:
            self.cap = cv2.VideoCapture(os.path.normpath(videopath).encode('utf-8'))
        try:
            self.framerate = cv2.GetCaptureProperty(self.cap, CV_CAP_PROP_FPS)
        except:
            self.framerate = 60
        self.calData = calData
        #self.running = True
        #self.updateVideoThread = thread.start_new_thread(self.updateVideo, (self.framerate, ))
        #if tracking is not True:
        #    self.tracking = False
        #else:
        #    self.tracking = True
        #self.sourceWindow = None

    #def __del__(self):
    #    self.StopCapture()
    #    print("Video capture destroyed")
            
    #def updateVideo(self, framerate):
    #    while(self.running is True):
    #        ret, frame = self.cap.read()
    #        if ret is True:
    #            if self.tracking is True:
    #                ET.Track(frame)
    #            #cv2.imshow('Video source', frame)
    #        else:
    #            #self.StopCapture()
    #            print("Capture returning:" + ret)
    #        cv2.waitKey(1)
    #        time.sleep(1/framerate)
    #        self.updateVideo(framerate)
        
    #    cv2.destroyWindow('Video source')
    #    self.cap.release()

    def updateVideo(self, last_center, calData):
        ret, frame = self.cap.read()
        if ret is True:
            ET.Track(frame, last_center, calData)
        else:
            print("Capture returning:" + str(ret))


    def StopTracking(self):
        #global cap, running, sourceWindow
        
        self.running = False
        self.cap.release()
        return self.running
        

    def CaptureFrame(self, e_center, calData):
        ret, frame = self.cap.read()
        if ret is True:
            return ET.track(frame)
        else:
            print("Capture returning:" + ret)
        ###TEST###
        StopTracking()
        ##########

    

def GetCameraInputs():
        #maxTested = 5 #Assuming no more than 5 camera sources
        #for i in range(0,maxTested):
        #    tempCam = cv2.VideoCapture(i)
        #    if tempCam.isOpened() is False:
        #        return i
        #return maxTested
        return 0

