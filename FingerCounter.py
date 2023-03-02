import cv2
import time
import os
import HandTrackingModule as htm


wCam = 640
hCam = 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "images"
fingerList = os.listdir(folderPath)
print(fingerList)
overlayList = []
for imPath in fingerList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # for Thumb Finger
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)

        fingerCount = fingers.count(1)
        print(fingerCount)

        h,w,c = overlayList[fingerCount-1].shape

        img[0:h, 0:w] = overlayList[fingerCount-1]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(200,40), cv2.FONT_HERSHEY_PLAIN,3,(0,225,0),3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
    