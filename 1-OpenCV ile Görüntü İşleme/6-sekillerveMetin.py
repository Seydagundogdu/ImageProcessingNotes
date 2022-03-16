# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:52:44 2022

@author: acer
"""

import cv2
import numpy as np

#resim oluştur
img = np.zeros((512, 512, 3),np.uint8) #sıfırlardan oluşan siyah bir resim. 0 = siyah, 1 = beyaz
print(img.shape)

cv2.imshow("Siyah",img)

# çizgi ekleme
# (resim, başlangıç noktası, bitiş noktası, renk(BGR), kalınlık) 
#cv2.line(img, (100,100), (200,200), (0,255,0), 3) --> çapraz bir şekilde boydan boya çizer (x,y)
cv2.line(img, (100,100), (100,300), (0,255,0), 3)
cv2.imshow("Cizgi", img)

#dikdörtgen
#(resim, başlangıç, bitiş, renk)
cv2.rectangle(img, (0,0), (256,256), ( 255,0,0)) 
#cv2.rectangle(img, (0,0), (256,256), ( 255,0,0), cv2.FILLED) #içi dolu
cv2.imshow("Dikdortgen", img)

#çember ve daire
#(resim, merkez, yarıçap, renk)
cv2.circle(img, (300,300), 45, (0,0,255))
#cv2.circle(img, (300,300), 45, (0,0,255), cv2.FILLED)
cv2.imshow("Cember", img)

#metin
#resim, metin içeriği, başlangıç noktası, font, kalınlık, renk)
cv2.putText(img, "Resim", (350,350),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
cv2.imshow("Metin", img)



k = cv2.waitKey(0) &0xFF

if(k == 27): # esc
    cv2.destroyAllWindows()