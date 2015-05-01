import numpy as np
import cv2
import EyeTracking as et

x = 0
y = 0

def Track(frame):
    global x, y
    x += 1
    y += 1
    x = x%100
    y = y%100
    trigger = False
    et.PackWithTimestamp( x,y,trigger )
