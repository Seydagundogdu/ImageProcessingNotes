# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:13:01 2022

@author: acer
"""

import cv2
import time

video_name = "MOT17-04-DPM.mp4"

#video içe aktarma

cap = cv2.VideoCapture(video_name)

print("Genişlik: ", cap.get(3))
print("Yükseklik: ", cap.get(4))

if cap.isOpened() == False:
    print("Video açılmadı!")
    
    
#video denilen şey aslında art arda gelen binlerce resimden oluşur, bu resimleri tek tek gösterebilmek için while döngüsü kullanılır
while True: 
    
    ret, frame = cap.read()

    if ret == True:
        time.sleep(0.01) #uyarı : kullanmazsak video çok hızlı akar
        cv2.imshow("Video",frame)
    else: break

    if cv2.waitKey(1) &0xFF == ord("q"):
        break

cap.release() #stop cature
cv2.destroyAllWindows()