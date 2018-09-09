import numpy as np
import librosa
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import pickle

f = upload()
file = open("../Desktop/datafeed","rb")
data = pickle.load(file)
labels = pickle.load(file)
f.close()


model = Sequential()
model.add(Conv2D(32, (4, 4), padding='same', activation='relu', input_shape=(600,60,1),data_format="channels_last"))
model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.summary()

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.summary()

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()


model.compile(optimizer= tf. train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])



model.fit(data, labels, epochs=25, batch_size=32)
