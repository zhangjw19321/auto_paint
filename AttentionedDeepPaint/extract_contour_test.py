import cv2  
import numpy as np 
def find_contour():
    img = cv2.imread('data/styles/style1.png')  
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, 
                                cv2.BORDER_CONSTANT, (255, 255, 255))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(img,contours,-1,(0,0,255),1)  
    
    cv2.imwrite("img.png", img)  
    # cv2.waitKey(3000) 
def find_contour_canny(): 
    # Let's load a simple image with 3 black squares 
    image = cv2.imread('data/styles/style2.png') 
    cv2.waitKey(0) 
    
    # Grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Find Canny edges 
    edged = cv2.Canny(gray, 30, 200) 
    cv2.waitKey(0) 
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    cv2.imshow('Canny Edges After Contouring', edged) 
    cv2.waitKey(3000) 
    
    print("Number of Contours found = " + str(len(contours))) 
    
    # Draw all contours 
    # -1 signifies drawing all contours 
    cv2.drawContours(image, contours, -1, (0, 0, 0), -1) 
    
    cv2.imshow('Contours', image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
def test3():
    image = cv2.imread("data/styles/style2.png", 1)

    # red color boundaries [B, G, R]
    lower = [1, 0, 20]
    upper = [60, 40, 220]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    ret,thresh = cv2.threshold(mask, 40, 255, 0)
    if (int(cv2.__version__[0]) > 3):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, -1)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

    # show the images
    cv2.imshow("Result", np.hstack([image, output]))

    cv2.waitKey(0)

def test4():
    import cv2

    imgfile = "data/styles/style2.png"
    img = cv2.imread(imgfile)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find Contour
    contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 需要搞一个list给cv2.drawContours()才行！！！！！
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if(area < (h/10*w/10)):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv2.drawContours(img, c_min, -1, (255,0,0), thickness=-1)
            continue
        #
        c_max.append(cnt)
        
    cv2.drawContours(img, c_max, -1, (0, 255, 0), thickness=-1)

    cv2.imwrite("mask.png", img)
    cv2.imshow('mask',img)
    cv2.waitKey(0)
if __name__ == "__main__":
    find_contour_canny()
    # test4()