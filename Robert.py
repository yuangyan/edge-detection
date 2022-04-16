import cv2 as cv
import numpy as np
name = 'scen.jpg'
img0 = cv.imread(name, 0)
img = img0
img = cv.GaussianBlur(img0, (9, 9), 0)
# flower2: 5, 5
# scen: 3, 5
# pipe organ:
# window: 7

#Roberts
kernelx = np.array([[-1,0],[0,1]], dtype=int)
kernely = np.array([[0,-1],[1,0]], dtype=int)
x = cv.filter2D(img, cv.CV_16S, kernelx)
y = cv.filter2D(img, cv.CV_16S, kernely)

absX = cv.convertScaleAbs(x)    
absY = cv.convertScaleAbs(y)  
Roberts = cv.addWeighted(absY,0.5,absY,0.5,0)

ret, thres = cv.threshold(Roberts,5,255,cv.THRESH_BINARY)

cv.imwrite(name[:-4] + ' Robert edge.jpg', thres)
cv.imwrite(name[:-4] + ' Robert edge raw.jpg', Roberts)
