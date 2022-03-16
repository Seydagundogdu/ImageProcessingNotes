# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:37:14 2022

@author: acer
"""

import keyboard #veri toplamak için (pip install keyboard)
import uuid #ekrandan kayıt almak
import time #süre tutmak
from PIL import Image
from mss import mss #pip install mss

"""
Veri ttoplanacak site :
http://www.trex-game.skipser.com/
"""

mon = {"top":391, "left":475, "width":250, "height":100} #paintte belirledik
sct = mss() # region of interest frame'ini elde etmemizi sağlayan library

i = 0

def record_screen(record_id, key):
    global i # hem içerde hem dışarda kullanabilmek için global yaptık
    
    i += 1
    print("{}: {}".format(key, i)) # key: klavyede bastığımız tuş. i: kaç kez klavyeye bastığımız
    img = sct.grab(mon) # ekranı mon değişkeni doğrultusunda al
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))
    
is_exit = False # fonksiyondan çıkmayı sağlar (esc)

def exit():
    global is_exit
    is_exit = True
    
keyboard.add_hotkey("esc", exit) # esc bastığında exit fonksiyonunu çağırır ve veri toplama işlemi sona erer

record_id = uuid.uuid4()

while True:
    
    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP): # zıpla
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN): # eğil
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"): # zıplamaması gereken durumlarda algoritmanın ne yapacağını bilmesi açısından etkisiz bir tuşa basılır
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError: continue # runtime error durumunda devam et
            























