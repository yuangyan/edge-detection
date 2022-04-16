import cv2 as cv
import numpy as np
name = 'flower2.jpg'
img0 = cv.imread(name, 0)
img = img0
img = cv.GaussianBlur(img0, (7, 7), 0)

x = cv.Sobel(img,cv.CV_16S,1,0)
y = cv.Sobel(img,cv.CV_16S,0,1)
  
absX = cv.convertScaleAbs(x)   # 转回uint8
absY = cv.convertScaleAbs(y)
  
Sobel = cv.addWeighted(absX,0.5,absY,0.5,0)

ret, thres = cv.threshold(Sobel,20,255,cv.THRESH_BINARY)

cv.imwrite(name[:-4] + ' Laplace edge.jpg', thres)
cv.imwrite(name[:-4] + ' Laplace edge raw.jpg', Sobel)