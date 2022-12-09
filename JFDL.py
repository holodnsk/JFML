from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import pandas as pd
import zipfile


def getwinHeadTarget():
 df_full = pd.read_csv("targethead.csv", sep=',')
 return df_full

def getwinMiddleTarget():
 df_full = pd.read_csv("winMiddle.csv", sep=',')
 return df_full

def getwinTailTarget():
 df_full = pd.read_csv("winTail.csv", sep=',')
 return df_full

def getFeatures():
 df_full = pd.read_csv("feature.csv", sep=',')
 return df_full


features = getFeatures()
targetHead = getwinHeadTarget()
# targetMiddle = getwinMiddleTarget()
# targetTail = getwinTailTarget()

X = features.astype(int)  # Присваиваем им тип данных - int
YHead = targetHead.astype(float)
# YMiddle = targetMiddle.astype(float)
# YTail = targetTail.astype(float)

print(X.shape)  # Выводим размерность X
print(YHead.shape)  # Выводим размерность Y
# print(YMiddle.shape)  # Выводим размерность Y
# print(YTail.shape)  # Выводим размерность Y

model = Sequential()
model.add(Dense(228, input_dim=165, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss='mse', optimizer=opt, metrics=['acc'])
model.fit(X, YHead, batch_size=512, epochs=320, validation_split=0.2, verbose=2)

# def modelTrain(Y):
#  model = Sequential()
#  model.add(Dense(228, input_dim=228, activation='relu'))
#  model.add(Dense(30, activation='relu'))
#  model.add(Dropout(0.5))
#  model.add(Dense(2, activation='softmax'))
#  opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
#  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
#  model.fit(X, Y, batch_size=32, epochs=20, validation_split=0.2, verbose=2)
#  return model
#
# model = modelTrain(YHead)
# model.save('modelHead.h5')
# model = modelTrain(YMiddle)
# model.save('modelMiddle.h5')
# model = modelTrain(YTail)
# model.save('modelTail.h5')
