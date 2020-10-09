import cv2
import numpy as np 
from matplotlib import pyplot as plt

def get_contour(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_bin = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img_bin = cv2.dilate(img_bin, kernel)
    cv2.imshow("ff",img_bin)
    cv2.waitKey(3000)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_id = -1
    max_area = -1
    for idx,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_id = idx
    x,y,w,h = cv2.boundingRect(contours[max_id])
    return x,y,w,h

def extract_contour(img):
    rect = (get_contour(img))
    print("bounding rect is; ",rect)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask ==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()
if __name__ == "__main__":
    img = cv2.imread('data/styles/style3.png')
    extract_contour(img)