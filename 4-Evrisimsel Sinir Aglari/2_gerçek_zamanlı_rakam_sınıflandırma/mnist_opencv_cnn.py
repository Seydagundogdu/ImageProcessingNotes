# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:45:05 2022

@author: acer
"""

import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle # modeli kaydetmek için

# os.chdir('C:\\Users\\acer\\ImageProcessing\\4-Evrisimsel Sinir Aglari\\2_gerçek_zamanlı_rakam_sınıflandırma') # dizin hatası için
# print(os.getcwd()) # mevcut çalışma dizinini verir

path = "myData" #labellarımızın ve verilerimizin bulunduğu dosya (10 adet label)

myList = os.listdir(path) # veri klasörüne girdik
noOfClasses = len(myList)

print("Label(sınıf) sayısı: ",noOfClasses)


images = [] # resimler
classNo = [] # label

for i in range(noOfClasses):
    myImageList = os.listdir(path + "\\"+str(i)) 
    for j in myImageList: 
        img = cv2.imread(path + "\\" + str(i) + "\\" + j) # klasörlerin içine girdik
        img = cv2.resize(img, (32,32)) #cnn'e göre
        images.append(img) # resimleri images listesine attık
        classNo.append(i) # labelları classNo listesine 
        
print(len(images))
print(len(classNo)) # classNo ve images eşit olmalı

# listeleri array'e çeviririz
images = np.array(images)
classNo = np.array(classNo)

# arraylerin boyutu
# print(images.shape) # matris
# print(classNo.shape) # vektör

# veriyi ayırma (önce train ve test olarak sonra train ve validation olarak)
# training yaparken validation işlemi yapmamız gerekecek
# en son model hazır olduğunda test veri seti ile modeli test ederiz
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.5, random_state = 42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

# print(images.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(x_validation.shape)

 # vis (görselleştirme) : dağılım için
 # fig, axes = plt.subplots(3,1,figsize=(7,7))
 # fig.subplots_adjust(hspace = 0.5)
 # sns.countplot(y_train, ax = axes[0])
 # axes[0].set_title("y_train")

 # sns.countplot(y_test, ax = axes[1])
 # axes[1].set_title("y_test")

 # sns.countplot(y_validation, ax = axes[2])
 # axes[2].set_title("y_validation")
# DAĞILIM DENGELİ

# preprocess
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # siyah-beyaza çevirdik
    img = cv2.equalizeHist(img) # histogramı 0-255 arasında genişlettik
    img = img /255 # normalize ettik
    
    return img


# idx = 311
# img = preProcess(x_train[idx])
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocess ",img)
    
# preprocess işlemini tüm veriye uygulama
x_train = np.array(list(map(preProcess, x_train))) # map(fonksiyon, uygulanacak parametre) : map, fonksiyonu ilgili parametreye uygulayan fonksiyondur
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1,32,32,1)
print(x_train.shape)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

# data generate (validation verisinde kullanılacak)
dataGen = ImageDataGenerator(width_shift_range = 0.1, # genişlikte 0.1 oranında kaydır
                             height_shift_range = 0.1, # yükseklikte 0.1 oranında kaydır
                             zoom_range = 0.1, # 0.1 oranında zoom yap
                             rotation_range = 10) 

dataGen.fit(x_train)

# verileri (label) kategorik hale getirmek (oneHotEncoder'daki gibi)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# Layers
model = Sequential()
model.add(Conv2D(input_shape = (32,32,1), filters = 8, kernel_size = (5,5), activation = "relu", padding = "same")) # padding = "same" : 1 sıra piksel ekler
model.add(MaxPooling2D(pool_size = (2,2))) # piksel ekleme

model.add(Conv2D( filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2)) # overfitting'i engellemek için
model.add(Flatten()) # düzleştirme
model.add(Dense(units=256, activation = "relu" )) # hidden layer
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses, activation = "softmax" )) # output layer

model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])

batch_size = 250

# çıktıları görselleştirme
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 15,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1)

# modeli depolama
pickle_out = open("model_trained_new.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# %% degerlendirme
hist.history.keys() # dict_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])

# eğitim loss'u ve validation loss'u grafiği
plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

# eğitim accuracy ve validation accuracy grafiği
plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()

# scores
score = model.evaluate(x_test, y_test, verbose = 1) # score[loss, accuracy]
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

# tüm classların sonucu
y_pred = model.predict(x_validation) # x validation'u kullanarak bir prediction yaparız
y_pred_class = np.argmax(y_pred, axis = 1) # bu prediction sonucu karşımıza çıkan olasılıksal değerleri maximize eden şeyi bulup onun indexini çekeriz
# ve bu bizim tahminimizdir diyoruz
Y_true = np.argmax(y_validation, axis = 1) # gerçek y değerine bakarız : y predict classların gerçekte ne olduğudur
cm = confusion_matrix(Y_true, y_pred_class) # karşılaştırma
f, ax = plt.subplots(figsize=(8,8))

# ( confusion_matrix, heatmap üzerinde değerler yazsın, çizgi kalınlığı, colormap, çizgi rengi, virgülden sonra 1 basamak olsun, eksen olarak eksenin üzerine yerleştir )
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("predicted") # x ekseni : tahminler
plt.ylabel("true") # y ekseni : gerçek değerler
plt.title("cm")
plt.show()
# GRAFİKTE DOĞRU TAHMİNLER YEŞİLLE GÖSTERİLMİŞTİR





































