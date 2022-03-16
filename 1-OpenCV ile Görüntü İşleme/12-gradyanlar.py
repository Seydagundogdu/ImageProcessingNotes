# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:30:04 2022

@author: acer
"""

"""
GRADYANLAR

Görüntü gradyanı, görüntüdeki yoğunluk ya da renkteki yönlü bir değişikliktir.
Kenar algılamada kullanılır.
"""

import cv2
import matplotlib.pyplot as plt

#resmi içe aktar
img = cv2.imread("sudoku.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Original Image")

# X gradyan (dik kenarların tespiti)
sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.imshow(sobelx, cmap = "gray"), plt.axis("off"), plt.title("Sobel X")

# Y gradyan (yatay kenarların tespiti)
sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.imshow(sobely, cmap = "gray"), plt.axis("off"), plt.title("Sobel Y")

#Laplacian Gradient (Hem x hem y eksenlerinin aynı anda detect edilmesi)
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure(), plt.imshow(laplacian, cmap = "gray"), plt.axis("off"), plt.title("Laplacian")

































