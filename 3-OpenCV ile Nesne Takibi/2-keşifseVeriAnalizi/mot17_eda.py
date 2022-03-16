# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:12:02 2022

@author: acer
"""
"""
GT.TXT DOSYASINDA HER BİR FRAME'E AİT KUTU BİLGİLERİ VARDIR
Bu kısımda 29 numaralı id'ye ait gt.txt dosyası oluşturulacaktır.
"""

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

"""
https://motchallenge.net/vis/MOT17-13-SDP/det/
https://arxiv.org/pdf/1603.00831.pdf
"""

# Frame'lerin özelliklerini belirten kolonların isimleri (https://arxiv.org/pdf/1603.00831.pdf)
col_list = ["frame_number","identity_number","left","top","width","height","score","class","visibility"]

# data içerisine gt.txt'de veriler alınır ve kolon isimleri verilir
data = pd.read_csv("gt.txt", names = col_list)

plt.figure()
sns.countplot(data["class"]) # verimizin iinde kaç farklı class olduğunu gösterir

car = data[data["class"] == 3] # makalede araçların class numarasının 3 olduğu veriliyor

video_path = "MOT17-13-SDP.mp4"

cap = cv2.VideoCapture(video_path)

id1 = 29 # 29 id verilmiş olan araç
numberOfImage = np.max(data["frame_number"])
fps = 25
bound_box_list = []

for i in range(numberOfImage-1):
    
    ret, frame = cap.read()
    
    if ret:
        
        frame = cv2.resize(frame, dsize=(960,540))
        
        filter_id1 = np.logical_and(car["frame_number"] == i+1, car["identity_number"]==id1)
        
        if len(car[filter_id1]) != 0:
            
            x = int(car[filter_id1].left.values[0]/2)
            y = int(car[filter_id1].top.values[0]/2)
            w = int(car[filter_id1].width.values[0]/2)
            h = int(car[filter_id1].height.values[0]/2)
            
            
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame, (int(x+w/2),int(y+h/2)), 2, (0,0,255),-1) # (int(x+w/2),int(y+h/2) kutucuğun merkesini bulmak için
            
            # frame, x, y, genis, yuksek, center_x, center_y
            bound_box_list.append([i, x,y,w,h,int(x+w/2),int(y+h/2)])
            
        cv2.putText(frame, "Frame num:"+str(i+1), (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2) # (10,30) yazının konumu
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
    else: break

cap.release()
cv2.destroyAllWindows()        
  

df = pd.DataFrame(bound_box_list, columns = ["frame_no", "x", "y", "w", "h", "center_x", "center_y"])
      
df.to_csv("gt_new.txt",index=False) # index'i yazdırmak istemiyoruz





























