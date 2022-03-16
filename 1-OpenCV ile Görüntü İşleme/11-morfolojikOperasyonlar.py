# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:43:52 2022

@author: acer
"""

"""
MORFOLOJİK OPERASYONLAR

Erozyon Yöntemi: Ön plandaki nesnenin sınırlanırını aşındırır.(Beyazlıkları azaltır.)

Genişleme: Erozyonun tam tersidir. Görüntüdeki beyaz bölgeyi artırır.

Açma: Erozyon + genişlemedir. Gürültünün giderilmesinde faydalıdır.

Kapatma: Açmanın tam tersidir. Genişleme + erozyondur. Ön plandaki 
nesnedeki gürültüleri gidermek için kullanılır.

Morfolojik Gradyan: Bir görüntünün genişlemesi ve erozyon arasındaki farktır.

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#resmi içe aktar
img = cv2.imread("datai_team.jpg",0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off"), plt.title("Original Image")

"""erozyon"""
#sınırları belirtecek kutucuk
kernel = np.ones((5,5), dtype = np.uint8)
result = cv2.erode(img, kernel, iterations = 1 ) #kaç kez erozyon uygulayacağı. ne kadar çok uygulanırsa yazı o kadar küçülür
plt.figure(), plt.imshow(result, cmap = "gray"), plt.axis("off"), plt.title("Erozyon")

"""dilation (genişleme)"""
result2 = cv2.dilate(img, kernel, iterations = 1 )
plt.figure(), plt.imshow(result2, cmap = "gray"), plt.axis("off"), plt.title("Genişleme")

"""açılma"""

#açılma beyaz gürültüyü engellediği için öncelikle beyaz gürültü oluşturalım
#White Noise
whiteNoise = np.random.randint(0,2, size = img.shape[:2])
whiteNoise = whiteNoise * 255
plt.figure(), plt.imshow(whiteNoise, cmap = "gray"), plt.axis("off"), plt.title("White Noise")

white_noisy_img = whiteNoise + img #toplama sırasında boyutlar uyuşmadığı için resmi siyah beyaz olarak içe aktardık img = cv2.imread("datai_team.jpg",0)
plt.figure(), plt.imshow(white_noisy_img, cmap = "gray"), plt.axis("off"), plt.title("White Noisy Image")

#açılma (opening) yöntemi : resimdeki beyazlıkları azaltır ve sonra genişletir
opening = cv2.morphologyEx(white_noisy_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
plt.figure(), plt.imshow(opening, cmap = "gray"), plt.axis("off"), plt.title("Açılma")

"""Kapatma"""
#black noise
blackNoise = np.random.randint(0,2, size = img.shape[:2])
blackNoise = whiteNoise * -255
plt.figure(), plt.imshow(blackNoise, cmap = "gray"), plt.axis("off"), plt.title("Black Noise")

black_noisy_img = blackNoise + img
black_noisy_img[black_noisy_img <= -245] = 0
plt.figure(), plt.imshow(black_noisy_img, cmap = "gray"), plt.axis("off"), plt.title("Black Noisy Image")

#kapatma yöntemi
closing = cv2.morphologyEx(black_noisy_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
plt.figure(), plt.imshow(closing, cmap = "gray"), plt.axis("off"), plt.title("Kapama")

"""gradient"""
#edge detection yani kenar tespiti işlemleri bu yöntemle yapılır
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.figure(), plt.imshow(gradient, cmap = "gray"), plt.axis("off"), plt.title("Gradyan")

















































































