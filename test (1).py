import librosa
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from math import ceil
from keras.models import Model
from keras.models import load_model


wave , sr = librosa.load("C:\\Users\\malindu\\Desktop\\IRMAS-Sample\\Testing\\05 - Sonata in A minor, Op. post. 143 D.784 - I. Allegro giusto-13.wav",sr=12000,mono=True)
#wave , sr = librosa.load("C:\\Users\\malindu\\Desktop\\IRMAS-Sample\\Training\\sax\\118__[sax][nod][jaz_blu]1702__3.wav",sr=12000,mono=True)

feed=list()
size=len(wave)
print(size)
plt.plot(wave)
plt.show()

if size>36000:
	cycles = ceil(size/36000)
	count=0
	while cycles>count:
		arr=np.array(wave[count*36000:(count+1)*36000])
		if len(arr)==36000:
			feed.append(arr.reshape((600,60,1)))
		count+=1
		
	
else:
	feed.append(np.array(wave).reshape((600,60,1)))

model = load_model('datamodel.h5')
model.compile(optimizer= tf. train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

pred = model.predict(np.array(feed))
print(pred)

