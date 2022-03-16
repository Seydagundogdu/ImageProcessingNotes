# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:22:21 2022

@author: acer
"""

"""
NESNE TESPİTİ

Amaç görüntü üzerinde bulunan nesnenin koordinatlarının genişlik ve yükseklik değerlerinin bulunmasıdır. Bu aşamada sınıflandırma söz konusu değildir.
"""

"""
KENAR ALGILAMA (edge detection) : Görüntü parlaklığının keskin bir şekilde değiştiği noktaları tanımlamayı amaçlayan bir yöntemdir.

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar
img = cv2.imread("london.jpg", 0) #siyah-beyaz
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")

edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255) #0-255 arası değer verildiği için herhangi bir threshold işlemi yapılmaz ve tüm kenarlar belirginleştirilir
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")
#Fakat bu şekilde nehrin içindeki girinti ve çıkkıntılar da kenar olarak algılanır. Bunun için thresholdları ayarlayarak ve bluring işlemi yaparak sadece nesnelerin kenarlarının algılanması sağlanır

#medyan deerine göre thrashold
#resmin medyanı
med_val = np.median(img)
print(med_val) #140

#alt ve üst threshold belirleme
low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high) #0-255 arası değer verildiği için herhangi bir threshold işlemi yapılmaz ve tüm kenarlar belirginleştirilir
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

#hala su üzerindeki çizgiler görünür olduğu için blur işlemi uygulanır
#bluring
blurred_img = cv2.blur(img, ksize = (5,5)) #ksize arttıkça bulanıklık derecesi artar
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off")

#tekrardan blur resmin medyanına göre threshold yapılır
med_val = np.median(blurred_img)
print(med_val) #138

low = int(max(0, (1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = blurred_img, threshold1 = low, threshold2 = high) #0-255 arası değer verildiği için herhangi bir threshold işlemi yapılmaz ve tüm kenarlar belirginleştirilir
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

#ksize 3,3 iken sudaki kenarlar hala belirgindir. Ksize'i 5,5 yaparsak sudaki kenarlar nerdeyse tamamen gitmişken çevredeki nesnelerin kenarları hala belirgin kalabilmektedir.






































