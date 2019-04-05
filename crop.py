import numpy as np
import cv2

img = cv2.imread("ruidoso.jpg")
pts = np.array([[260,350],[350,450],[710,410],[450,320]])

## (1) Crop the bounding rect
rect = cv2.boundingRect(pts)
x,y,w,h = rect
croped = img[y:y+h, x:x+w].copy()

## (2) make mask
pts = pts - pts.min(axis=0)
mask = np.zeros(croped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)



#cv2.imwrite("croped.png", croped)
#cv2.imwrite("mask.png", mask)
cv2.imwrite("dst.png", dst)
#cv2.imwrite("dst2.png", dst2)



#What will be our dates that we show?
