from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras
import tensorflow as tf
import pandas as pd
from datetime import datetime
from keras.callbacks import CSVLogger
from keras import backend as K


csv_logger = CSVLogger(datetime.now().strftime("%H%M%S")+'log.csv', append=False, separator=';')

# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
nrows = 10000000
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
 df_full = pd.read_csv("featureTail.csv", sep=',',  dtype = 'bool') #
 return df_full.astype(bool)

def getFeaturesMiddle():
 df_full = pd.read_csv("featureMiddle.csv", sep=',',  dtype = 'bool')
 return df_full.astype(bool)

def getFeaturesHead():

 df_full = pd.read_csv("featureHead.csv", sep=',', dtype = 'bool')
 return df_full.astype(bool)


def modelTrain(X,Y):
 model = Sequential()
 model.add(Dense(165, input_dim=165, activation='relu'))
 model.add(Dense(428, activation='relu'))
 model.add(Dense(1028, activation='relu'))
 # model.add(Dense(4028, activation='relu'))
 # model.add(Dense(1028, activation='relu'))
 model.add(Dense(428, activation='relu'))
 model.add(Dense(100, activation='relu'))
 # model.add(Dense(428, activation='relu'))
 model.add(Dense(28, activation='relu'))
 model.add(Dense(1, activation='tanh'))
 opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
 model.compile(loss='mse', optimizer=opt, metrics=['mae'])
 model.fit(X, Y, batch_size=2000, epochs=1, validation_split=0.2, callbacks=[csv_logger])
 return model


model = Sequential()
model.add(Dense(165, input_dim=165, activation='relu'))
model.add(Dense(428, activation='relu'))
# model.add(Dense(1028, activation='relu'))
# model.add(Dense(4028, activation='relu'))
# model.add(Dense(1028, activation='relu'))
# model.add(Dense(428, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dense(428, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(1, activation='tanh'))
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss='mse', optimizer=opt, metrics=['mae'])

X = getFeaturesHead()
Y = getwinHeadTarget()

for n in range(0,2):
 xstart = 0
 xend = 10000000
 delta = 13000000
 print(X.shape)
 print(Y.shape)
 x_part = X[xstart:xend]
 y_part = Y[xstart:xend]
 print(str(x_part.shape[0]))
 while x_part.shape[0]>=2000:
  print(str(x_part.shape[0]))
  print(str(xstart),str(xend))
  print(x_part.shape)
  print(x_part.shape)
  model.fit(x_part, y_part, batch_size=2000, epochs=1, validation_split=0.2, callbacks=[csv_logger])
  xstart=xstart+delta
  xend=xend+delta
  x_part = X[xstart:xend]
  y_part = Y[xstart:xend]


model.save('modelHeadGen1.h5')

# X = getFeaturesMiddle()
# Y = getwinMiddleTarget()
# print(X.shape)
# print(Y.shape)
#
# model = modelTrain(X,Y)
# model.save('modelMiddleGen1.h5')
#
# X = getFeaturesTail()
# Y = getwinTailTarget()
# print(X.shape)
# print(Y.shape)
#
# model = modelTrain(X,Y)
# model.save('modelTailGen1.h5')
