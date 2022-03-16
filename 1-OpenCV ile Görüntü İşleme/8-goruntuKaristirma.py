# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:59:37 2022

@author: acer
"""

import cv2
import matplotlib.pyplot as plt

#karıştırma
img1 = cv2.imread("img1.jpg")
#python resimleri default olarak BGR olarak yükler bunu RGB'ye çevirmek için
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#resimlerin aynı boyuttta olup olmadığı kontrol edilir (2 tane matrisin toplanabilmesi için aynı boyutlarda olması gerekir)
print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1, (600,600))
print(img1.shape)

img2 = cv2.resize(img2, (600,600))
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#karıştırma resim = alpha*img1 + beta*img2
#alpha ve beta katsayıları hangi remin daha baskın olacağını bekirler
#katsayı 1'e yaklaştıkça resim görünürlüğü artar
blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0)
plt.figure()
plt.imshow(blended)

k = cv2.waitKey(0) &0xFF

if(k == 27): # esc
    cv2.destroyAllWindows()