# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:02:06 2022

@author: acer
"""
import cv2

cap = cv2.VideoCapture(0) #bilgisayarın ana kamerasına bağlanır

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

#video kaydet

writer = cv2.VideoWriter("video_kaydı.mp4", cv2.VideoWriter_fourcc(*"DIVX"),20,(width, height)) 
#1. parametre : kaydedilecek videonun adı
#2. parametre : çerçeveleri sıkıştırmak için kullanılan 4 karakterli codec kodu
#3. parametre : videonun akkış hızı, her saniyede görünen resim sayısı

while True:
    
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    
    #save
    writer.write(frame) #writer bir frame deposu olarak düşünülebilir
    if cv2.waitKey(1) &0xFF == ord("q"): break

cap.release()
writer.release()
cv2.destroyAllWindows()
    
