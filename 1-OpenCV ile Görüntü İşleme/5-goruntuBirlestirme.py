# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:37:32 2022

@author: acer
"""

import cv2
import numpy as np

#resmi içe aktarma
img = cv2.imread("lenna.png")
cv2.imshow("Original", img)

#horizontal olarak ( yatayda) birleştirme
hor = np.hstack((img, img))
cv2.imshow("Horizontal", hor)

#dikeyde birleştirme
ver = np.vstack((img, img))
cv2.imshow("Vertical", ver)

k = cv2.waitKey(0) &0xFF

if(k == 27): # esc
    cv2.destroyAllWindows()