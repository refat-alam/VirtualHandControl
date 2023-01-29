import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# window property
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 160
volPer = 0

try:
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[4], lmList[8])
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 225), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 225), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (225, 0, 225), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (225, 0, 225), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)

            vol = np.interp(length, [20, 200], [minVol, maxVol])
            # print(int(length), vol)
            volBar = np.interp(length, [20, 200], [160, 500])
            # volPer = np.interp(length, [20, 200], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

            if length < 40:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (500, 20), (160, 50), (125, 255, 0), 3)
        cv2.rectangle(img, (int(volBar), 20), (160, 50), (125, 255, 0), cv2.FILLED)
        # cv2.putText(img, f'{int(volPer)}%', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, (125, 255, 0), 3)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.imshow("img", img)
        cv2.waitKey(1)
        pass
except KeyboardInterrupt:
    # Handle the keyboard interrupt
    print("Keyboard interrupt received. Exiting gracefully.")
