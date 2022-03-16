# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:47:57 2022

@author: acer
"""

"""
RENK VE NESNE TESPİTİ
Belirli renklerde bulunan nesnelerin tespitinin nasıl yapılacağını kontur bulma yöntemi ile öğreneceğiz.
Konturlar basitçe, aynı renk ve yoğunluğa sahip tüm sürekli noktaları birleştiren bir eğri olarak açıklanabilir.
Konturlar, şekil analizi ve nesne algılama ve tanıma için kullanışlı bir araçtır.

HSV : RGB gibi renklerin farklı bir formatıdır. (hue(öz,ton), saturation(doygunluk), value(parlaklık))
"""

import cv2
import numpy as np
from collections import deque #tespit edilen objenin merkezini depolamada

# nesne merkezini depolayacak veri tipi
buffer_size = 16
pts = deque(maxlen = buffer_size) #merkez pointleri. max uzunluk

# mavi renk aralığı HSV
#ışık faktörü de hesaba katılarak sınırlar biraz geniş seçilmeli
blueLower = (84,  98,  0) #tuple veri tipi
blueUpper = (179, 255, 255)

# capture: kameradan görüntü alma
cap = cv2.VideoCapture(0)
cap.set(3,960) #cam genişlik
cap.set(4,480) #cam yükseklik

#camden okuma
while True:
    
    success, imgOriginal = cap.read()
    
    if success: #frameler okunabiliyorsa
        
        # blur: detayı azaltıp noise'ı elemine etme
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) #(görüntü, pencere boyutu, standart sapma)
        
        # hsv: blurlanan görüntü hsv formatını çevrilir
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #cv2.imshow("HSV Image",hsv)
    
        
        # mavi için maske oluştur
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        #cv2.imshow("mask Image",mask)
        # maskenin etrafında kalan gürültüleri sil
        mask = cv2.erode(mask, None, iterations = 2) #erozyon
        mask = cv2.dilate(mask, None, iterations = 2) #genişletme
        #cv2.imshow("Mask + erozyon ve genisleme",mask)

        # farklı sürüm için
        # (_, contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # kontur
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None #nesnenin merkezi

        if len(contours) > 0: #eğer mavi bir nesne ya da nesneler bulunduysa
            
            # en buyuk konturu al. Birden fazla mavi nesne varsa en büyük olanı alacaktır. aynı anda birden çok nesne tespiti yapmaz
            c = max(contours, key = cv2.contourArea) #bir nesnenin içinde küçük mavicikler olabilir bunları bulmak yerine tüm nesneyi alsın (en büyük mavi nesneyi)
            
            # dikdörtgene çevir (nesneyi çevrelemek için)
            rect = cv2.minAreaRect(c) #konturu kapsayacak minimum alana sahip bir dikdörtgen
            
            #rectangle bir tuple'dır. genişlik ve yükseklik özellikleri tanımlanmalıdır
            ((x,y), (width,height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation)) #round: yuvarlama fonk.
            print(s)
            
            # kutucuk : rect ile elde edilen koordinatları kullanarak bir kutucuk oluşturur
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # moment: görüntünün merkzini bulmamıza yarayan yapı
            M = cv2.moments(c) #momente en büyük kontur yollanır
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) # tuple(x,y)
            
            # konturu çizdir: sarı
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2) #(ekrana verilecek görüntü, çizilecek kutu, kutu rengi, kutu kalınlığı)
            
            # merkere bir tane nokta çizelim: pembe
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1) #(ekrana verilecek görüntü, çizilecek daire, yarıçap, renk, kalınlık (-1 = içi dolu))
            
            
            # bilgileri ekrana yazdır
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2) #(görüntü, string, koordinatlar(ekran boyutuna göre), font, renk, kalınlık)
            cv2.imshow("Orijinal Tespit",imgOriginal)

        # deque .. takip algoritması : önceki 16 merkez noktalarını da tutan bir takip algoritması
        pts.appendleft(center) #noktalar deque'nun içerisine eklenir
        
        for i in range(1, len(pts)): #noktacılar deque'nun içerisinde dönülür
            
            if pts[i-1] is None or pts[i] is None: continue #eğer noktamın bir önceki indeksi boşsa ya da o anki indeks boşsa devam et
        
            cv2.line(imgOriginal, pts[i-1], pts[i],(0,255,0),3) # (çizilecek resim, çizginin ilk pointi, son pointi, rengi, kalınlığı)
            
        cv2.imshow("Orijinal Tespit",imgOriginal)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
































