import os
import tkMessageBox 

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

    

