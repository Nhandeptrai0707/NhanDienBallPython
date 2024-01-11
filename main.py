import cvzone
import cv2
from cvzone.ColorModule import ColorFinder
import numpy as np
import time
import autopy

wScr, hScr = autopy.screen.size()
cap = cv2.VideoCapture(0)
wCam, hCam = [640, 480]
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 30, 'smin': 111, 'vmin': 54, 'hmax': 60, 'smax': 255, 'vmax': 255}

while True:
    success, img = cap.read()
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContour, contours = cvzone.findContours(img, mask, minArea=50)
    if contours:
        data = contours[0]['center'][0], \
            contours[0]['center'][1], \
            int(contours[0]['area'])
        area = data[2]
        x2 = np.interp(data[0], (0, wCam), (0, wScr))
        y2 = np.interp(data[1], (0, hCam), (0, hScr))
        #autopy.mouse.move(x2, y2)
        print(area)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgContour, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    imageStack = cvzone.stackImages([img, imgColor, mask, imgContour], 2, 0.5)
    cv2.imshow("ngu",imageStack)
    cv2.waitKey(1)
