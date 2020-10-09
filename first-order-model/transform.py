import numpy as np
import cv2

im = cv2.imread("51.png")
randomlimit = 0.1
perspective = np.eye(3,dtype=np.float32) + np.random.uniform(-randomlimit,randomlimit,(3,3))
perspective[2][2] = 1.0
im = cv2.warpPerspective(im, perspective, (im.shape[1], im.shape[0]), borderValue=(255,255,255))
cv2.imwrite("tt.png",im)
