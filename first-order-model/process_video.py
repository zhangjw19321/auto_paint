import cv2
import numpy as np
cap = cv2.VideoCapture("b.mp4")
videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (640,480))
def convert(frame):
    random = np.random.randint(3,4)
    kernel = np.ones((random,random),np.uint8) 
    frame = cv2.resize(frame,(640,480))
    # videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (640,480))
    lower_green = np.array([35, 43, 35])
    upper_green = np.array([90, 255, 255])
    # frame = cv2.imread("251.png")
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    random = np.random.randint(3,5)
    kernel = np.ones((random,random),np.uint8) 
    closing = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    return closing
    # cv2.imwrite("ff1.png",closing)
i = 0
while i < 600:
    i += 1
    ret,frame = cap.read()
    close1 = convert(frame)
    cv2.imshow("ff",close1)
    cv2.waitKey(30)
    videoWriter.write(close1)

cv2.destroyAllWindows()
cap.release()
videoWriter.release()
