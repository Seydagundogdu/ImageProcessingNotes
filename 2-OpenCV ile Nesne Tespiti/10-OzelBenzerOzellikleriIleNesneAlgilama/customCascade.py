# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:45:42 2022

@author: acer
"""

"""
VERİ TOPLAMA
1) veri seti:
    n,p
2) cascade programı indir (https://amin-ahmadi.com/cascade-trainer-gui/))
3) cascade oluştur
4) cascade kullanarak tespit algoritması yaz
"""
# negatif ve pozitif veriler için iki ayrı klasör oluşturulur yani bu kod iki kez çalıştırılmalıdır
import cv2
import os

# resim depo klasörü
path = "images"

# resim boyutu
imgWidth = 180
imgHeight = 120

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640) # genişlik
cap.set(4, 480) # boyut
cap.set(10, 180) #  aydınlık

# resim klasörünü oluşturma
global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)): # images0 diye bir şey varsa
        countFolder += 1 # countfolder'ı 1 arttır
    os.makedirs(path+str(countFolder)) # yoksa oluştur

saveDataFunc()

# cam'ı açma
count = 0
countSave = 0

while True:
    
    success, img = cap.read() 
    
    if success: # eğer kamera başarılı bir şekilde bir şeyler okuyabiliyorsa
        
        img = cv2.resize(img, (imgWidth, imgHeight))
        
        if count % 5 == 0: # 5 tanede 1 resmi depola
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png",img) # klasöre yaz
            countSave += 1 # her bir resim save edildiğinde 1 art
            print(countSave)
        count += 1 # her iterasyonda 1 art
        
        cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()






















