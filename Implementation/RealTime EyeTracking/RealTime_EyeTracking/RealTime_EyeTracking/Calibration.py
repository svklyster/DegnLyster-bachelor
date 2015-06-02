import Tkinter as tk
import ttk
import cv2
import tkFileDialog
import codecs
import os
import tkMessageBox
import thread
import time
import sys
import re
import CalExtract as calExt

class Calibration:

    def __init__(self, pathname, calibType, filename = None):
        self.calibType = calibType
        if calibType is "numbers": 
            self.numberOfPoints = 9
            self.coordinates = []
        else:
            tkMessageBox.showerror("Exception", "No calibration type " + calibType + " found!")
        self.pathname = pathname
        if filename is None:
            self.filename = "calibration"
        else: 
            self.filename = filename
            self.filepath = os.path.abspath(self.pathname + "/" + "filename"+".clog")

    def AddPoint(x,y):
        self.coordinates.append(x,y)



    


def motion(event):
    global xinterval, yinterval, widget, lineindexX, lineindexY
    x, y = event[0], event[1]
    a = [int(x), int(y)]
    if(a[0]%xinterval <= 0.6*xinterval and a[0]%xinterval >= 0.4*xinterval and a[1]%yinterval <= 0.6*yinterval and a[1]%yinterval >= 0.4*yinterval):
        widget.itemconfig(lineindexX[(a[0]/xinterval+(a[1]/yinterval*3))], fill="green")
        widget.itemconfig(lineindexY[((a[1]/yinterval*3)+a[0]/xinterval)], fill="green")
    else:
        for i in range(len(lineindexX)):
            widget.itemconfig(lineindexX[i], fill="red")
            widget.itemconfig(lineindexY[i], fill="red")

def parsegeometry(geometry):
    m = re.match("(\d+)x(\d+)([-+]\d+)([-+]\d+)", geometry)
    if not m:
        raise ValueError("failed to parse geometry string")
    return map(int, m.groups())

def callback():
    return

def calScreen():
    global xinterval, yinterval, widget, lineindexX, lineindexY
    l = False
    p = True

    root = tk.Toplevel()
    root.title("RealTime EyeTracking")

    #root.bind('<Motion>', motion)

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.overrideredirect(1)

    widget = tk.Canvas(root, width=w, height=h)
    widget.configure(background='black')
    widget.pack()
    widget.pack(side = "left")
    #widget.pack(expand = 1)
    widget.create_text(20, 30, anchor=tk.W, font="Purisa", text="Thisistest")

    widget.create_line(10,10,350,10)
    widget.create_line(0,100,200,0, fill = "red", dash = (4,4))

    #root.attributes('-fullscreen', True)
    #root.state('zoomed')
    #root.configure(background='black')

    xinterval = w/3
    yinterval = h/3
    areas = []


    lineindexX = []
    lineindexY = []

    for y in range(0,3):
        for x in range(0,3):
            lineindexX.append(widget.create_line(0.4*xinterval + x*xinterval, 0.5*yinterval + y*yinterval, 0.6*xinterval + x*xinterval, 0.5*yinterval + y*yinterval, fill = "red"))
            lineindexY.append(widget.create_line(0.5*xinterval + x*xinterval, 0.4*yinterval + y*yinterval, 0.5*xinterval + x*xinterval, 0.6*yinterval + y*yinterval, fill = "red"))
            areas.append([0.5*xinterval + x*xinterval,0.5*yinterval + y*yinterval])
            #areas[1].append(0.5*yinterval + y*yinterval)

    calExt.areas = areas
    
    
    #label = tk.Label(root, text="Message")
    #label.pack()
    if (l):
        w = -w
    if (p):
        w = 0
    #button = tk.Button(root, text="Quit", command=callback)
    #button.pack()
    newgeo = '+' + str(w) + '+0'
    root.geometry(newGeometry=newgeo)
    root.update()

    #print(parsegeometry(root.geometry()))

    #root.mainloop()

def runCal(sessionData):

    calExt.runCalib(sessionData)

def snapFrames():

    calExt.snapFrames()
    
