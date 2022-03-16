# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:09:49 2022

@author: acer
"""
"""
ŞABLON EŞLEŞTİRME İLE NESNE TESPİTİ
Bir şablon görüntünün konumunu daha büyük bir görüntüde aramak ve bulmak için bir yöntemdir.
Şablon görüntüsünü giriş görüntüsünün üzerine kaydırır ve şablon görüntüsünün altındaki 
giriş görüntüsünün şablonu ve yamayı karşılaştırır.
"""
import cv2
import matplotlib.pyplot as plt

# template matching: sablon esleme

img = cv2.imread("cat.jpg", 0) #arama yapılacak resim  
print(img.shape)
template = cv2.imread("cat_face.jpg", 0) #şablon
print(template.shape)
h, w = template.shape #yükseklik, genişlik

#OpenCV'nin bize sağladığı bu 6 metodun ana hedefi korelasyona bakmak.
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods: #yukarıdaki metot stringini for döngüsüne alırız
    
    #yukarıda string olarak aldığımız metotları fonksiyona çevirmeliyiz
    method = eval(meth) # 'cv2.TM_CCOEFF' -> cv2.TM_CCOEFF
    
    res = cv2.matchTemplate(img, template, method) #(resim, şablon resim, uygulanack kor. metodu)
    print(res.shape) #orijinal resimle aynı olmak zorunda
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #metotların çıktıları farklı olduğu için bazılarında min_loc sol üst kçşeye denk gelirken bazılarında sağ üst köşeye denk gelir
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: #metot TM_SQDIFF ya da TM_SQDIFF_NORMED ise
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2) #(resim, kutucuk başlangıç noktası, bitiş noktası, renk, kalınlık)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray") #subplot(satır sayısı,sütun sayısı,kaçıncı sütun) 
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()