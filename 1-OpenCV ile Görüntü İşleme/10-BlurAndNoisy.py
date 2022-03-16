# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:08:24 2022

@author: acer
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np #gürültü eklemek için

import warnings
warnings.filterwarnings("ignore") #uyarıları kaldırmak için

#BLURING (detayı azaltır ve gürültüyü engeller)
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("orijinal"),plt.show()

"""
Görüntü bulanıklığı, görüntünün düşük eçişli bir filtre uygulanması ile elde edilir.
Gürültüyü gidermek için kullanışlıdır.Aslında görüntüden yüksek frekanslı içeriği (parazit, kenarlar) kaldırır.
OpenCV'de üç ana tür bulanıklaştırma tekniği vardır.

1-Ortalama Bulanıklaştırma
Bir görüntünün normalleştirilmiş bir kutu filtresiyle sarılmasıdır
Çekirdek alanı altındaki tüm piksellerin ortalamasını alır ve bu ortalamayı merkezi öğe ile yer değiştirir.
"""
#(resim, kutu boyutu 8kernel size)
dst2 = cv2.blur(img, ksize = (3,3)) #dst opencv'de çıktılara verilen addır. Girdilere ise src denir.
plt.figure(),plt.imshow(dst2),plt.axis("off"),plt.title("Ortalama Blur")

"""
2-Gauss Bulanıklaştırma (Gaussian Blur)
Kutu filtresi yerine Gauss çekirdeği kullanılır
Pozitif ve tek olması gereken çekirdeğin genişliği ve yüksekliği belirtilir
SigmaX ve SigmaY, X ve Y yönlerindeki standart sapmayı belirtmeliyiz
"""
#(resim, kutu boyutu, X yönündeki sigma, Y yönündeki sigma)
gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb), plt.axis("off"),plt.title("Gaussian Blur")

"""
3-Medyan Bulanıklaştırma
Çekirdek alanı altındaki tüm piksellerin medyanını alır ve merkesi öğe bu medyan değerle değiştirilir
Tuz ve biber gürültüsüne karşı oldukça etkilidir

"""
gb = cv2.medianBlur(img, ksize = 3)
plt.figure(), plt.imshow(gb), plt.axis("off"),plt.title("Median Blur")

#NOISE

def gaussianNoise(image):
    row, col, ch = image.shape #(543, 543, 3)
    mean = 0 #ortalama
    var = 0.05 #varyans
    sigma = var**0.5 #standart sapma varyansın kareköküdür.
    
    gauss = np.random.normal(mean, sigma, (row, col, ch)) #gaussian, normal dağlımdır
    gauss = gauss.reshape(row, col, ch) #noise'nin boyutunu teyit ediyoruz
    noisy = image + gauss #orijinal resim ile gürültüyü toplarız
    
    return noisy

#içe aktarma ve normalize etme
#orijinal resimdeki görüntü matrisini normalize ederiz (0la 1 arasına taşırız)
#Çünkü oluşturduğumuz gürültü 0 ortalamalı bir gürültü

img2 = cv2.imread("NYC.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255 #255'e bölerek normalize edilir
plt.figure(),plt.imshow(img2),plt.axis("off"),plt.title("Normalize"),plt.show()

gaussianNoisyImage = gaussianNoise(img2)
plt.figure(),plt.imshow(gaussianNoisyImage),plt.axis("off"),plt.title("Gauss Noisy"),plt.show()

#Gaussian Blur ile gürültü azaltma
gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb2), plt.axis("off"),plt.title("With Gaussian Blur")


#Salt and Pepper Noise
 def saltPepperNoise(image):
    
    row, col, ch = image.shape
    s_vs_p = 0.5
    
    amount = 0.004
    
    noisy = np.copy(image)
    
    # salt beyaz
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1
    
    # pepper siyah
    num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0
    
    return noisy
       
spImage = saltPepperNoise(img)   
plt.figure(), plt.imshow(spImage), plt.axis("off"), plt.title("SP Image")

mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize = 3)
plt.figure(), plt.imshow(mb2), plt.axis("off"), plt.title("with Medyan Blur")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    