from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import pandas as pd
import zipfile

def getwinHeadTarget():
 zf = zipfile.ZipFile('winHead.zip')

 for filename in sorted(zf.namelist()):
  df_full = pd.read_csv(zf.open(filename), sep=',')
  return df_full

def getwinMiddleTarget():
 zf = zipfile.ZipFile('winMiddle.zip')
 for filename in sorted(zf.namelist()):
  df_full = pd.read_csv(zf.open(filename), sep=',')
  return df_full

def getwinTailTarget():
 zf = zipfile.ZipFile('winTail.zip')

 for filename in sorted(zf.namelist()):
  df_full = pd.read_csv(zf.open(filename), sep=',')
  return df_full

def getwinFeature():
 zf = zipfile.ZipFile('features.zip')

 for filename in sorted(zf.namelist()):
  df_full = pd.read_csv(zf.open(filename), sep=',')
  return df_full

features = getwinFeature()
targetHead = getwinHeadTarget()
print(targetHead.shape)  # Размерность данных
targetMiddle = getwinMiddleTarget()
print(targetMiddle.shape)  # Размерность данных
targetTail = getwinTailTarget()
print(targetTail.shape)  # Размерность данных

fdataset = features.values  # Берем только значения массива(без индексов)
X = fdataset.astype(int)  # НЕ Присваиваем им тип данных - float(с плавающей точкой) данным с 0 по 60 колонки

tHeadDataset = targetHead.values
YHead = tHeadDataset.astype(int)  # Присваеваем значению Y данные из столбца с индексом 60

tMiddleDataset = targetMiddle.values
YMiddle = tMiddleDataset.astype(int)  # Присваеваем значению Y данные из столбца с индексом 60

tTailDataset = targetTail.values
YTail = tTailDataset.astype(int)  # Присваеваем значению Y данные из столбца с индексом 60

print(X.shape)  # Выводим размерность X
print(YHead.shape)  # Выводим размерность Y
print(YMiddle.shape)  # Выводим размерность Y
print(YTail.shape)  # Выводим размерность Y

model = Sequential()
model.add(Dense(228, input_dim=228, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
opt = tf.keras.optimizers.RMSprop(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

model.fit(X, YHead, batch_size=64, epochs=40, validation_split=0.2, verbose=True)
model.save('modelHead.h5')

model = Sequential()
model.add(Dense(228, input_dim=228, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
opt = tf.keras.optimizers.RMSprop(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

model.fit(X, YMiddle, batch_size=64, epochs=40, validation_split=0.2, verbose=True)
model.save('modelMiddle.h5')

model = Sequential()
model.add(Dense(228, input_dim=228, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
opt = tf.keras.optimizers.RMSprop(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

model.fit(X, YTail, batch_size=64, epochs=40, validation_split=0.2, verbose=True)
model.save('modelTail.h5')