# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:59:39 2022

@author: acer
"""

import cv2
import matplotlib.pyplot as plt

# resmi içe aktar
img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR den gray scale'e dönüştürür
plt.figure()
plt.imshow(img, cmap = "gray") #color map olarak siyah-beyaz(gri) verilir.
plt.axis("off") #eksenler kaldırılır
plt.show()


# eşikleme
#60 ile 255 arasındaki genliğe sahip pikselleri 255 (beyaz) yapar
#(resim, eşik değeri, max değer, eşikleme tipi)
_, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img, cmap = "gray")
plt.axis("off")
plt.show()

# uyarlamalı eşik değeri : sınırlar daha nettir. resmin belli bir alanına odaklanarak o alana özgü bir eşik değeri belirler ve komşu pikselleri belirler.
#(resim, max değer, thresholding metodu, thresh tipi, )
thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,8)
plt.figure()
plt.imshow(thresh_img2, cmap = "gray")
plt.axis("off")
plt.show()