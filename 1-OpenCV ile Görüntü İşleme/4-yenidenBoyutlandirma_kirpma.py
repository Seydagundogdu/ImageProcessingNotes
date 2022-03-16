# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:00:15 2022

@author: acer
"""

#boyutu küçük resimlerle çalışmak memory ve performans açısından daha iyi olabilir
#bazı modeller belli bir boyuttaki resimleri kabul eder bu yüzden boyutlandırma yapmak gerekebilir

import cv2

img = cv2.imread("lenna.png")

#(512,512) : siyah-beyaz resim boyutu
#(512,512,3) : rgb (renkli) resim boyutu

print("Resim Boyutu : ", img.shape) #resmin boyutunu verir
cv2.imshow("Img Original", img)

#resized
imgResized = cv2.resize(img, (800,800))
print("Resized Img Shape :",imgResized.shape)
cv2.imshow("Img Resized", imgResized)

#crop
imCropped = img[:200, :300] #[height, width]
cv2.imshow("Img Cropped", imCropped)

k = cv2.waitKey(0) &0xFF

if(k == 27): # esc
    cv2.destroyAllWindows()

