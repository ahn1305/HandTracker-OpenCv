import cv2
import mediapipe as mp
import time

# from mediapipe.python.solutions import hands

class handDetector():
    def __init__(self,mode=False,maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode # Create an object, it has its own var
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Hand detection basic setup
        self.mpHands = mp.solutions.hands
        self.hand = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self,img,draw=True):
        imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hand.process(imageRGB) #calling the hands object, there is a method called process that process the frame and it gives the result
        # print(results.multi_hand_landmarks) # check for multiple hands 

        
        if self.results.multi_hand_landmarks: # Check if we have multiple hands

            for handLms in self.results.multi_hand_landmarks: #Check for each hand
                # Get information within the hands
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS) # Draw the points and connections for individual hands
        
        return img

    def findPosition(self,img,handNo=0,draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark): # landmark we are getting=lm id relates the exact index number of the finger landmark
            # use x and y to find the location for the landmark on the hand, since values are decimal the location should be pixels
            # print(id,lm)
                h , w, c = img.shape # gives width and height
                cx,cy = int(lm.x*w),int(lm.y*h) # get the pixel
                #print(id,cx,cy) # print id, cx,cy
                lmList.append([id,cx,cy])

                if draw:
                    cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)

        return lmList
            


def main():

    

    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0) # set camera id

    detector = handDetector(detectionCon=0.8)

    while True: # Run forever
        success,img = cap.read() # read the image
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
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



if __name__ == "__main__":
    main()