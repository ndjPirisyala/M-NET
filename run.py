import numpy as np
import librosa
import pickle
import os


f=list()
l=list()

def mapping(files):
    wave , sr = librosa.load(files,sr=12000,mono=True)
    arr=np.array(wave).reshape((600,60,1))
    f.append(arr)



s=[
"../Downloads/IRMAS-TrainingData/cla",
"../Downloads/IRMAS-TrainingData/pia",
"../Downloads/IRMAS-TrainingData/gac",
"../Downloads/IRMAS-TrainingData/tru",
"../Downloads/IRMAS-TrainingData/cel",
"../Downloads/IRMAS-TrainingData/sax",
"../Downloads/IRMAS-TrainingData/org",
"../Downloads/IRMAS-TrainingData/vio",
"../Downloads/IRMAS-TrainingData/flu",
"../Downloads/IRMAS-TrainingData/gel",
"../Downloads/IRMAS-TrainingData/voi"
]

for i in s:
    for files in os.listdir(i):
        if i[-3:]=="pia":
            l.append([1,0])
        if i[-3:]!="pia":
            l.append([0,1])
        if files.endswith(".wav"):
            print(i+"/"+files)
            mapping(i+"/"+files)



print(len(f))
print(len(f[0][0]))
print(s[0][-3:])
print(len(l))


fileObject ="../Desktop/datafeed"
getit = open(fileObject,"wb")


pickle.dump(np.array(f),getit)
pickle.dump(np.array(l),getit)

getit.close()
