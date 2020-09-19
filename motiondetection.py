import cv2
import imutils

class MotionDetector:
    def __init__(self, accumWeight = 0.5):
        self.accumWeight = accumWeight
        self.bg = None
        
    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)
        
    def detect(self, image, tVal = 25):
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, tval, 255, cv2.THRESH_BINARY)[1]
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cts) == 0:
            return None
        return (thresh, max(cnts, key = cv2.contourArea))