import cv2
import mediapipe
import time

ctime=0
ptime=0

medhands=mediapipe.solutions.hands
hands=medhands.Hands(max_num_hands=1,min_detection_confidence=0.7)
draw=mediapipe.solutions.drawing_utils

def thumb_down(lmlist):
    if lmlist[12][1] > lmlist[20][1] and lmlist[4][1] < lmlist[3][1]: return True
    elif lmlist[12][1] < lmlist[20][1] and lmlist[4][1] > lmlist[3][1]: return True
    else: return False

def front_thumb_up(lmlist):
    if lmlist[12][1] > lmlist[20][1] and lmlist[4][1] > lmlist[3][1]: return True
    else: return False

def back_thumb_up(lmlist):
    if lmlist[12][1] < lmlist[20][1] and lmlist[4][1] < lmlist[3][1]: return True
    else: return False

def index_up(lmlist):
    if lmlist[8][2] < lmlist[6][2]: return True
    else: return False

def middle_down(lmlist):
    if lmlist[12][2] > lmlist[10][2]: return True
    else: return False

def ring_down(lmlist):
    if lmlist[16][2] > lmlist[14][2]: return True
    else: return False

def pinky_down(lmlist):
    if lmlist[20][2] > lmlist[18][2]: return True
    else: return False

def gesture_recognition(img):
    img = cv2.flip(img,1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    res = hands.process(imgrgb)
    
    lmlist=[]
    tipids=[4,8,12,16,20] #list of all landmarks of the tips of fingers
    
    # cv2.rectangle(img,(20,350),(90,440),(0,255,204),cv2.FILLED)
    # cv2.rectangle(img,(20,350),(90,440),(0,0,0),5)
    msg = "None"
    if res.multi_hand_landmarks:
        msg = str(len(res.multi_hand_landmarks)) + ':'
        for handlms in res.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                h,w,c= img.shape
                cx,cy=int(lm.x * w) , int(lm.y * h)
                lmlist.append([id,cx,cy])

            if len(lmlist) != 0 and len(lmlist)==21:
                if(index_up(lmlist) and middle_down(lmlist) and ring_down(lmlist) and pinky_down(lmlist)):
                    if(front_thumb_up(lmlist)):
                        msg += " Front Seven"
                    elif(back_thumb_up(lmlist)):
                        msg += " Back Seven"
                    else:
                        msg += " Unknown"
                elif(index_up(lmlist) and not middle_down(lmlist) and ring_down(lmlist) and pinky_down(lmlist) and thumb_down(lmlist)):
                    msg = " YA"
                else:
                    msg += " Unknown"
            #change color of points and lines
            draw.draw_landmarks(img,handlms,medhands.HAND_CONNECTIONS,draw.DrawingSpec(color=(0,255,204),thickness=2,circle_radius=2),draw.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=3))
    return img, msg
    
    #fps counter
    # ctime = time.time()
    # fps=1/(ctime-ptime)
    # ptime=ctime
    
    #fps display
    # cv2.putText(img,f'FPS:{str(int(fps))}',(0,12),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
          
    # cv2.imshow("hand gestures",img)
    
    #press q to quit
#     if cv2.waitKey(1) == ord('q'):
#         break
    
# cv2.destroyAllWindows()
