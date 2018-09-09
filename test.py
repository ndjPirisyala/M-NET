import librosa
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models import Model
from keras.models import load_model


wave , sr = librosa.load("C:\\Users\\malindu\\Desktop\\IRMAS-Sample\\Testing\\02. School Boy-9.wav",sr=12000,mono=True)
l=list()
l.append(np.array(wave[:36000]).reshape(600,60,1))
l.append(np.array(wave[36000:72000]).reshape(600,60,1))
l.append(np.array(wave[72000:108000]).reshape(600,60,1))
l.append(np.array(wave[108000:144000]).reshape(600,60,1))
l.append(np.array(wave[144000:180000]).reshape(600,60,1))


model = load_model('datamodel.h5')
model.compile(optimizer= tf. train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

pred = model.predict(np.array(l))
print(pred)


"""
model = load_model('datamodel.h5')
model.compile(optimizer= tf. train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

pred = model.predict_classes(np.array([np.array(list(range(36000))).reshape((600,60,1))]	))
print(pred)

count=0
feed=list()
if(len(wave)>36000):
  feed.append(np.array(wave[count*36000:(count+1)*36000]).reshape((600,60,1)))

model = load_model('model.h5')
model.compile(optimizer= tf. train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

pred = model.predict_classes(feed)
print(pred)
"""
