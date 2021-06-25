import cv2
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0) # set camera id

detector = htm.handDetector()

while True: # Run forever
    success,img = cap.read() # read the image
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    # Fps calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) # image, stringtodisplay,place, font, color,scale

    cv2.imshow("Image",img) # show

    if cv2.waitKey(10) == ord('q'): # if q is pressed
        break # break the loop

cap.release()  # release the screen
cv2.destroyAllWindows() # destroy all the windows
