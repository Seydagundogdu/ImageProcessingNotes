# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:38:29 2022

@author: acer
"""

import cv2
import os

#♦bir klasörün içindeki resimleri okuma
files = os.listdir() #dosyaları return eder
print(files)
img_path_list = []
for f in files:
    if f.endswith(".jpg"): 
        img_path_list.append(f)
print(img_path_list)

for j in img_path_list:
    print(j)
    image = cv2.imread(j)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    # scale factor : resim üzerinde ne kadar zoom yapacağını belirler
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 2)
    
    #rectangle boyutlarını ayarlama
    for (i, (x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,255),2)
        cv2.putText(image, "Kedi {}".format(i+1), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
        
    cv2.imshow(j, image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue #q'ya basarak btün resimler gezilir
