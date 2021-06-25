import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0) # set camera id

# Hand detection basic setup
mpHands = mp.solutions.hands
hand = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True: # Run forever
    success,img = cap.read() # read the image
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hand.process(imageRGB) #calling the hands object, there is a method called process that process the frame and it gives the result
    # print(results.multi_hand_landmarks) # check for multiple hands 

    
    if results.multi_hand_landmarks: # Check if we have multiple hands

        for handLms in results.multi_hand_landmarks: #Check for each hand
            # Get information within the hands
            for id,lm in enumerate(handLms.landmark): # landmark we are getting=lm id relates the exact index number of the finger landmark
                # use x and y to find the location for the landmark on the hand, since values are decimal the location should be pixels
                # print(id,lm)
                h , w, c = img.shape # gives width and height
                cx,cy = int(lm.x*w),int(lm.y*h) # get the pixel
                print(id,cx,cy) # print id, cx,cy

                if id == 4:
                    cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS) # Draw the points and connections for individual hands
            


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