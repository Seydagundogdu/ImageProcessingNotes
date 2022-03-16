# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:39:38 2022

@author: acer
"""

"""
KONTUR TESPİTİ
Aynı renk ve yoğunluğa sahip tüm kesintisiz noktaları (sınırla birlikte) birleştirmeyi amaçlayan yöntemdir.
Konturlar şekil analizi ve nesne algılama ve tanıma için kullanılır.

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar
img = cv2.imread("contour.jpg", 0) #siyah-beyaz
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

# farklı sürüm için 
# image, contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

for i in range(len(contours)):
    
    # external
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours, i, 255, -1) #nesnenin dış kısmı
    else: # internal
        cv2.drawContours(internal_contour,contours, i, 255, -1) #nesnenin iç kısmı

plt.figure(), plt.imshow(external_contour, cmap = "gray"),plt.axis("off")
plt.figure(), plt.imshow(internal_contour, cmap = "gray"),plt.axis("off")



































