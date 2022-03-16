# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:38:03 2022

@author: acer
"""

import cv2
import numpy as np

#resmi içe aktarma
img = cv2.imread("kart.png")
cv2.imshow("Original", img)

width = 400
height = 500

#çevrilecek olana nesnenin köşelerini point etme (sol üst, sol alt, sağ üst, sağ alt)
pts1 = np.float32([[230,1],[1,472],[548,150],[338,617]])

#çevrildikten sonra köşelerin geleceği noktaları point etme
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

#transform matrisi oluşturulur
matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

#döndürme işlemi (resim, rotasyon matrisi, çıktının boyutu)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("Rotate edilen resim", imgOutput)

k = cv2.waitKey(0) &0xFF

if(k == 27): # esc
    cv2.destroyAllWindows()