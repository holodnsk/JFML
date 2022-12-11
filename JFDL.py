from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import pandas as pd
import zipfile


def getwinHeadTarget():
 df_full = pd.read_csv("targethead.csv", sep=',')
 return df_full.astype(float)

def getwinMiddleTarget():
 df_full = pd.read_csv("targetmiddle.csv", sep=',')
 return df_full.astype(float)

def getwinTailTarget():
 df_full = pd.read_csv("targettail.csv", sep=',')
 return df_full.astype(float)

def getFeaturesTail():
 df_full = pd.read_csv("featureTail.csv", sep=',',dtype="bool")
 return df_full.astype(bool)

def getFeaturesMiddle():
 df_full = pd.read_csv("featureMiddle.csv", sep=',',dtype="bool")
 return df_full.astype(bool)

def getFeaturesHead():
 df_full = pd.read_csv("featureHead.csv", sep=',',dtype="bool")
 return df_full.astype(bool)


def modelTrain(X,Y):
 model = Sequential()
 model.add(Dense(165, input_dim=165, activation='relu'))
 model.add(Dense(428, activation='relu'))
 model.add(Dense(28, activation='relu'))
 model.add(Dense(1, activation='tanh'))  # !!!!! логичней tanh так так есть положительные и отрицательные значения
 opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
 model.compile(loss='mse', optimizer=opt, metrics=['mae'])
 model.fit(X, Y, batch_size=512, epochs=20, validation_split=0.2, verbose=2)
 return model

# X = getFeaturesHead()
# Y = getwinHeadTarget()
# print(X.shape)
# print(Y.shape)
#
# model = modelTrain(X,Y)
# model.save('modelHeadGen1.h5')
#
# X = getFeaturesMiddle()
# Y = getwinMiddleTarget()
# print(X.shape)
# print(Y.shape)
#
# model = modelTrain(X,Y)
# model.save('modelMiddleGen1.h5')

X = getFeaturesTail()
Y = getwinTailTarget()
print(X.shape)
print(Y.shape)

model = modelTrain(X,Y)
model.save('modelTailGen1.h5')
