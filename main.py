import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import tensorflow as tf

data= []
labels= []
classes= 3
current_path = os.getcwd()
epochs = 4
formes =["cercle", "rectangle", "triangle"]



for i in range(classes):
    forme=""
    if i==0 :
        forme="cercle"
    elif i==1:
        forme="rectangle"
    else:
        forme= "triangle"

    path = os.path.join(current_path,'data',forme)
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path+'/'+a)
            image = image.resize((128,128))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("erreur dans la lecture de l'image")

data = np.array(data)
labels = np.array(labels)


print(data.shape, labels.shape)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state= 42)

Y_train = to_categorical(Y_train, 3)
Y_test = to_categorical(Y_test, 3)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train,Y_train, batch_size=32, epochs=epochs, validation_data=(X_test,Y_test))



test_image = os.path.join(current_path, "data", "triangle", "image_7.png")

try :
    test_image = Image.open(test_image)
    test_image = test_image.resize((128,128))
    test_image = np.array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)
    predicted_class = np.argmax(prediction)

    print("forme reel: triangle")
    print(f"forme predite: {formes[predicted_class]}")
except Exception as e:
    print(f"{e}")
    
