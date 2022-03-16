# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:35:49 2022

@author: acer
"""

from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss # ekran üzerinden kayıt alma

# paintte belirlediğimiz pikseller
mon = {"top":391, "left":475, "width":250, "height":100} 
sct = mss()

# trex_tranin'de modelimizi eğittiğimiz boyutlar
width = 125
height = 50

# model yükle
model = model_from_json(open("model.json","r").read()) # kaydettiğimiz modeli çağırıyoruz
model.load_weights("trex_weight.h5") # modelin içine geçen ders oluşturduğumuz weight'leri de ekliyoruz

# down = 0, right = 1, up = 2
labels = ["Down", "Right", "Up"]

framerate_time = time.time()
counter = 0
i = 0 # frameleri sayar
delay = 0.4 # bir komuttan sonra yeni bir komut vermek için 0.4 saniye bekle
key_down_pressed = False

while True:
    
    img = sct.grab(mon) #♦ ekrandan belirlediğimiz piksellere göre resim al
    im = Image.frombytes("RGB", img.size, img.rgb) # resmi dönüştür
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255
    
    X =np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1) # modeli keras'ın kabul edeceği şekle dönüştür
    r = model.predict(X) # modelimizi kullanarak bir tahmin işlemi gerçekleştir ve bunu r isimli bir result'a eşitle
    
    # r : toplamı 1 olan 3 sayı
    result = np.argmax(r) # r nin içindeki en büyük sayının indexini döndürür buna göre hangi tuşa basılacağına karar verilir
    
    
    if result == 0: # down = 0
        
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True # basıldı
        
    elif result == 2:    # up = 2
        
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        if i < 1500: 
            time.sleep(0.3)
        elif 1500 < i and i < 5000: # 1500. frameden sonra oyun hızlanıyor
            time.sleep(0.2)
        else:
            time.sleep(0.17)
            
        keyboard.press(keyboard.KEY_DOWN) # aşağıya bas
        keyboard.release(keyboard.KEY_DOWN) # tuşu bırak
    
    counter += 1
    
    if (time.time() - framerate_time) > 1:
        
        counter = 0
        framerate_time = time.time()
        if i <= 1500:
            delay -= 0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0
            
        print("---------------------")
        print("Down: {} \nRight:{} \nUp: {} \n".format(r[0][0],r[0][1],r[0][2]))
        i += 1
        




























