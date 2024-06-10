#

import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

plt.imshow(X_train[0])

num_pixels = X_train.shape[1] * X_train.shape[2]
#flattening the images
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]


#creating a function to create a classification model

def classification_model():
    model = keras.models.Sequential()
    
    model.add(keras.layers.Dense(50, activation="relu", input_shape=(num_pixels,)))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


my_model = classification_model()
my_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

scores = my_model.evaluate(X_test, y_test, verbose = 0)

accuracy = scores[1]
error = 1 - scores[1]

#to save the computing power let's save the model after training
my_model.save('classification_model.keras')

#two formats for saving
pretrained_model = keras.models.load_model("classification_model.keras")
pretrained_model = keras.models.load_model("classification_model.h5")