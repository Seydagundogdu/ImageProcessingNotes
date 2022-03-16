# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:27:00 2022

@author: acer
"""
"""
HAVZA ALGORİTMASI
Havza segmentasyon için, yani bir görüntüdeki farklı nesneleri ayırmak için kullanılan klasik bir algoritmadır.
Hehangi bir gri tonlamalı görüntü, yüksek yoğunluğun zirveleri ve tepeleri, düşük yoğunluğun vadileri ifade ettiği
bir topografik yüzey olarak görülebilir.
İzole edilmiş her vadiyi (yerel minimum) farklı renkli suyla (etiketler) doldurmaya başlarsınız.
Su yükseldikçe, yakınlardaki zirvelere (gradyanlara) bağlı olarak, farklı vadiler gelen su, belli ki farklı renklerle birleşmeye
başlayacaktır.
Bundan kaçınmak için suyun birleştiği yerelere bariyerler inşa edersiniz. Tüm zirveler su altında kalana kadar su doldurma ve bariyerler inşa etme işine devaö ediyorsunuz.
Daha sonra oluşturduğunuz engeller size segmentasyon sonucunu verir.

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# içe aktar
coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")

# lpf: blurring : paraların girinti çıkıntılarının azaltılması ve sadece çevresinin belirgin kalması için
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")

# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")

# binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# kontur
# _, contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours,i,(0,255,0),10)
plt.figure(),plt.imshow(coin),plt.axis("off")

# watershed

# içe aktar
coin = cv2.imread("coins.jpg")
plt.figure(), plt.imshow(coin), plt.axis("off")

# lpf: blurring
coin_blur = cv2.medianBlur(coin, 13)
plt.figure(), plt.imshow(coin_blur), plt.axis("off")

# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
plt.figure(), plt.imshow(coin_gray, cmap="gray"), plt.axis("off")

# binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)
plt.figure(), plt.imshow(coin_thresh, cmap="gray"), plt.axis("off")

# açılma (erozyon+genişleme) : paraları küçültüp birbirlerinden ayırmak için
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")

# nesneler arası distance bulalım
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.figure(), plt.imshow(dist_transform, cmap="gray"), plt.axis("off")

# resmi küçült
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform),255,0)
plt.figure(), plt.imshow(sure_foreground, cmap="gray"), plt.axis("off")

# arka plan için resmi büyült (genişlet)
sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background,sure_foreground) #arka planla ön planın farkı alınır
plt.figure(), plt.imshow(unknown, cmap="gray"), plt.axis("off")

# bağlantıları sağlama
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0 #beyaz yerleri siyah yap
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")
# nesneler tam anlamıyla birbirinden ayrıldı ve adacıklar ortaya çıktı, bundan sonrası için havza algoritması ile segmente edilebilir
# havza
marker = cv2.watershed(coin,marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")


# kontur
# _, contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours,i,(255,0,0),2)
plt.figure(),plt.imshow(coin),plt.axis("off")













        