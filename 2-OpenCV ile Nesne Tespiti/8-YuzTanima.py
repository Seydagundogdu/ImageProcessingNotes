# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:34:43 2022

@author: acer
"""
"""
YÜZ TANIMA

Pozitif görüntü : Yüz tanımada kullanılan ve yüz içeren görüntü
Negatif görüntü : Yüz tanımada kullanılan ve yüz içermeyen görüntü
NOT : Bu bölümde eğitilmiş bir nesne tespit algoritması kullanılacaktır.

Benzer özellik işlevi, birçok pozitif ve negatif görüntü ile eğitilir.
Daha sonra diğer görüntüdeki nesneleri tespit etmek iin kullanılır.
Bunun için Haar özellikleri kullanılmaktadır.
Her özellik, siyah dikdörtgenin altındaki piksellerin toplamından beyaz dikdörtgenin altındaki
piksellerin toplamının çıkarılmasıyla elde edilen tek bir değerdir.

"""

import cv2
import matplotlib.pyplot as plt

# içe aktar 
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

# sınıflandırıcı
# yüz olup olmamasını sınıflandırır
# burun ağız göz vs. detect eden .xml kaynağı yüklenir.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # kaynak : https://github.com/opencv/opencv/tree/master/data/haarcascades

#resmin üzerinde piksel piksel ilerleyerek (diktörtgeni de büyüterek) yüz olan kısımları rectangle ile işaretleme
face_rect = face_cascade.detectMultiScale(einstein) # resmin içerisinden bir detection yap ve bunu rectangle içerisine at
# rectangle koordinatları
for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

# barcelona oyuncuları
# içe aktar 
barce = cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")

# face_rect = face_cascade.detectMultiScale(barce)
# Bu şekilde yapıldığında algoritma yüz olmayan kısımlarda da yüz varmış gibi algılar bunun önüne geçmek için detectMultiScale'in parametreleriyle oynanmalıdır.
# detectMultiScale : bir kutucuğun yanında kaç kutucuk varsa bunu yüz olarak alılaması gerektiğini belirler. (komşuluk değeri)
# detectMultiScale arttıkça yanlış sınıflaandırmalar da azalır
face_rect = face_cascade.detectMultiScale(barce, minNeighbors = 10)


for (x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")

# video üzerinden yüz algılama
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
            
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,150,0),3)
        cv2.imshow("face detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()