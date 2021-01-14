import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt



X = pickle.load(open("./X.pickle","rb"))
Y = pickle.load(open("./Y.pickle","rb"))

X=np.array(X/255.0)
Y=np.array(Y)

"""
SIMPLE Neural Network
ann = Sequential([
        Flatten(input_shape=(200,200,3)),
        Dense(3000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(4, activation='sigmoid'),
    ])

ann.compile(optimizer='SGD', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X,Y,epochs=5)
"""


#Convolutional Neural Network
cnn = Sequential([

    #cnn
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(200,200,3)),
    MaxPooling2D((2,2)),

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D((2,2)),

    #dense
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

cnn.fit(X,Y,epochs=10)

cnn.save('./Plant_CNN_Model')
