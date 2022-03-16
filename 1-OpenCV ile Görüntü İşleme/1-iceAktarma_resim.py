# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:55:06 2022

@author: acer
"""

import cv2

## içe aktarma ve 

img = cv2.imread("messi.jpg",0)

## görsellerştirme

cv2.imshow("ilk resim", img)

k = cv2.waitKey(0) &0xFF

if(k == 27): # esc
    cv2.destroyAllWindows()
elif k == ord('s'): # s tuşu
    cv2.imwrite("messi_gray.png", img)
    cv2.destroyAllWindows()