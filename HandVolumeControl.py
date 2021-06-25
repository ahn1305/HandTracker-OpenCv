import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

##########################
wCam,hCam = 640,480
##########################

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector = htm.handDetector(detectionCon=0.7)


while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)


    if len(lmList) != 0:
        #print(lmList[4],lmList[8])
        pass

        # Draw on the thumb and the index
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]

        # Get the center of the line
        cx,cy = (x1+x2) //2, (y1+y2) //2

        cv2.circle(img,(x1,y1),12,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),12,(255,0,255),cv2.FILLED)

        # connect the two points
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)

        # Draw the cirlce in center
        cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)

        # getting length of the line
        length = math.hypot(x2-x1,y2-y1)
        print(length)

        # if length is less than 50 change the center color to green
        if length<50:
            cv2.circle(img,(cx,cy),12,(0,255,0),cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS: {int(fps)}',(40,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    cv2.imshow("img",img)

    if cv2.waitKey(10) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()