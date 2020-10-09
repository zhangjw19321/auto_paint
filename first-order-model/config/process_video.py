import cv2
cap = cv2.VideoCapture("b.mp4")

videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (640,480))
i = 1
while True:
    i = i + 1
    ret,frame = cap.read()
    if i % 50 == 1:
        cv2.imwrite(str(i) + ".png",frame)
    # img_ = cv2.GaussianBlur(img,(9,9),0)