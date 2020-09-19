from gesturedetection import GestureDetector
from motiondetection import MotionDetection
import numpy as np
import imutils
import cv2

camera = cv2.VideoCapture(0)

(top, right, bot, left) = np.int32(("70,350, 285, 590").split(","))
gd = GestureDetector()
md = MotionDetector()

numFrames = 0
gesture = None
values = []

while True:
    (grabbed, frame) = camera.read()
    
    frame = imutils.resize(frame, width = 600)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    (frameH, frameW) = frame.shape[:2]
    
    roi = frame[top:bot, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    
    if numFrames < 32:
        md.update(gray)
        cv2.imshow("im1", gray)
    else :
        skin = md.detect(gray)
        
        if skin is not None:
            (thresh, c) = skin
            cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
            fingers = gd.detect(thresh, c)
            
            if gesture is None:
                gesture = [1, fingers]
            else :
                if gesture[1] == fingers:
                    gesture[0] =+ 1
                    if gesture[0] >= 25:
                        if len(values) == 2:
                            values = []
                        values.append[fingers]
                        gesture = None
                        