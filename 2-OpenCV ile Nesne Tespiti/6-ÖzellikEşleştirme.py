# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:24:45 2022

@author: acer
"""
"""
ÖZELLİK EŞLEME
Görüntü işlemede nokta özelliği eşlemesi, karmaşık bir sahnede belirtilen bir hedefi
tespit etmek için etkin bir yöntemdir.
Bu yöntem, birden çok nesne yerine tek nesneleri algılar.
Örneğini bu yöntemi kullanarak, kişi dağınık bir görüntü üzerinde belirli bir kişiyi
tanıyabilir, ancak başka herhangi bir kişiyi tanıyamaz.

Brute-Force eşleştiricisi, bir görüntüdeki bir özelliğin tanımlayıcısını başka bir görüntünün
diğer tüm özellikleriyle eşleştirir ve mesafeye göre eşleşmeyi döndürür.
Tüm özelliklerle eşleşmeyi kontrol ettiği için yavaştır.

Ölçek değişmez özellik dönüşümü, anahtar noktaları ilk olarak bir dizi referans görüntüden çıkarılır ve saklanır.
Yeni görüntüdeki her bir özelliği bu saklanan veri ile ayrı ayrı karşılaştırarak ve öznitelik vektörlerinin Öklid
mesfaesine dayalı olarak aday eşleştirme özelliklerini bularak yeni bir görüntüde bir nesne tanınır.

"""

import cv2
import matplotlib.pyplot as plt

# ana görüntüyü içe aktar
chos = cv2.imread("chocolates.jpg", 0)
plt.figure(), plt.imshow(chos, cmap = "gray"),plt.axis("off")

# aranacak olan görüntü
cho = cv2.imread("nestle.jpg", 0)
plt.figure(), plt.imshow(cho, cmap = "gray"),plt.axis("off")

# orb tanımlayıcısı
# köşe-kenar gibi nesneye ait özellikler
orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None) #herhangi bir maskeleme olmadığı için none
kp2, des2 = orb.detectAndCompute(chos, None)

# bf matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# noktaları eşleştir
matches = bf.match(des1, des2)

# mesafeye göre sırala
#(sort edilecek şey, sort fonksiyonu (distance'a göre sort etme))
matches = sorted(matches, key = lambda x: x.distance)

# eşleşen resimleri görselleştirelim
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"),plt.title("orb")

# sadece orb tanımlayıcısı nesnesi eşleştirmeye yetmedi, bir de sift tanımlayıcısı kullanalım
# sift tanımlayıcısında size'lar farklı olsa da eşleşme yapılabirlir. Orb'ye göre daha iyidir.
# sift
sift = cv2.xfeatures2d.SIFT_create()

# sift ve orb bize featureları çıkaran yöntemlerdir, eşleşme brute-force ile yapılır.
# bf
bf = cv2.BFMatcher()

# anahtar nokta tespiti sift ile
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

# k = match sayısı
matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []

for match1, match2 in matches:
    
    if match1.distance < 0.75*match2.distance:
        guzel_eslesme.append([match1])
    
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho,kp1,chos,kp2,guzel_eslesme,None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")