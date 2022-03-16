# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:10:34 2022

@author: acer
"""

"""
KÖŞE ALGILAMA (CORNER DETECTION)

Köşeler, iki kenarın kesişimi olduğu için bu iki kenarın yönlerinin değiştiği noktayı temsil eder.
Köşeler resimdeki renk geçişindeki bir varyasyonu temsil ettiğinden, bu "varyasyonu" arayacağız. Görüntü
yoğunluğundaki varyasyonu arayacağız.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar
img = cv2.imread("sudoku.jpg", 0) #siyah-beyaz
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

#harris corner detection
#blocksize : komşuluk boyutu
#ksize : kutucuk boyutu
#k : free parametre
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

#köşeleri daha da netleştirmek için genişletme işlemi uygulanır
#dst'leri yani tespit ettiği noktaları genişletme
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1 #dst dst'nin maximumunun %20'sinden büyükse 1'e eşitle
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")


#shi tomasi detection
#resmi içe aktar
img = cv2.imread("sudoku.jpg", 0) #siyah-beyaz
img = np.float32(img)
#(resim, kaç köşe tespit edilecek, quaility level, min distance)
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)
corners =np.int64(corners)

for i in corners:
    x,y = i.ravel() #çok boyutlu bir diziyi düzleştirme
    cv2.circle(img, (x,y), 3, (125,125,125), cv2.FILLED) #yarıçapı 3 olan ii dolu bir daire

print(img.shape)
plt.imshow(img), plt.axis("off")





















