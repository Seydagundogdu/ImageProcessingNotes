# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:32:12 2022

@author: acer
"""

import glob # resim ve klasörlere erişim
import os  # resim ve klasörlere erişim
import numpy as np
from keras.models import Sequential # derin öğrenme algoritması tasarımı ve eğitimi
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # Dense: katmanlar. Dropout: seyreltme. Flatten: düzleştirme. Conv2D: evrişim ağı. Maxpooling: piksel ekleme.
from PIL import Image # Python Image Library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # verimizdeki rakamları 0,1,2,3 diye labellar ekler. etiketlenmiş verimiz kerasta eğitilebilir hale getirilir
from sklearn.model_selection import train_test_split
import seaborn as sns # görselleştirme

import warnings
warnings.filterwarnings("ignore")

imgs = glob.glob("./img_nihai/*.png")

#ArithmeticError(
   # )width = 125
#height = 50

width = 125
height = 50

X = []
Y = []

for img in imgs:
    
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255 # 0-1
    X.append(im) # resimler
    Y.append(label) # resimlere ait olan labellar
    
X = np.array(X) # listeyi array'e çevirdik (test split için)
X = X.reshape(X.shape[0], width, height, 1) # (resim sayısı, genişliği, yüksekliği, channel değeri(1:siyah-beyaz))


# sns.countplot(Y) # kaç tane label olduğuna dair grafik (down, right, up)

def onehot_labels(values): # labelları kears'ın kabul edeceği hale getirir
    label_encoder = LabelEncoder() # etiketleri integer sayılara çevirir
    integer_encoded = label_encoder.fit_transform(values) # önce fit ediyor (öğreniyor), sonra dönüştürüyor (0,1,2)
    onehot_encoder = OneHotEncoder(sparse = False) # integer sayıları binary sayılara çevirir
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1) #☻integer_encoded.shape : (169,) olduğu için (169,1) yapılır
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
    # integer_encoded -> onehot_encoded
    # 0 -> 100
    # 1 -> 010
    # 2 -> 001

Y = onehot_labels(Y) # oluşturduğun encoder fonksiyonunu çağırdım

train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)  # (veriler, labellar, test veri seti oranı, rastgele bölünme)  

# cnn model
model = Sequential() # layerlarımızın üzerine ekleyeceğimiz temel yapı  
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1))) # convolutional layer (filtre sayısı, filtre boyutu, aktivasyon fonk., girdi boyutu)
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu")) # convolutional layer. ilk layer'ın çıktısı buranın girdisidir o yüzdeninput belirtmeye gerek yoktur.
model.add(MaxPooling2D(pool_size = (2,2))) # piksel ekleme(işlemi gerçekleştirecek olan filtrenin boyutu)
model.add(Dropout(0.25)) # seyreltme (yüzde kaçının kaybolacağı)
model.add(Flatten()) # düzleştirme (vektör)
# SINIFLANDIRMA KISMI
model.add(Dense(128, activation = "relu")) # gizli layer (nöron sayısı, aktivasyon fonk.)
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax")) # çıktı layer'ı (ikiden fazla çıktı varsa softmax kullanılır)

# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights yuklendi")    

# loss : en son hataları hesaplamayı sağlayan fonk.
# loss çoksa back propagation yapılarak model iyileştirilir (geriye doğru türev alma)
model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"]) # metrics : başarım metriği

model.fit(train_X, train_y, epochs = 35, batch_size = 64) #(eğitim verisi, eğitim label'ı, eğitim sayısı, verilerin kaçarlı grup halinde iterasyona sokulacağı)

score_train = model.evaluate(train_X, train_y) # score'in 0. indexi kaybı 1. indexi accuracy (eğitim doğruluğu)'i return eder
print("Eğitim doğruluğu: %",score_train[1]*100)    
    
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %",score_test[1]*100)      
    
# weight'leri kaydetmek için
open("model_new.json","w").write(model.to_json())
model.save_weights("trex_weight_new.h5")   
    
    

    
"""
eğitim sırasında memory %99'a ulaşırsa bilgisayarı yeniden başlat.
eğitim doğruluğu %100 test doğruluğu ise bundan düşükse ezberleme (overfitting) söz konusudur.
"""

    
    
    
    
    
    
    
    
    
    
    
    
    