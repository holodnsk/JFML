from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras
import tensorflow as tf
import pandas as pd
from datetime import datetime
from keras.callbacks import CSVLogger
from keras import backend as K
from sklearn.model_selection import train_test_split

csv_logger = CSVLogger(datetime.now().strftime("%H%M%S")+'log.csv', append=True, separator=';')

nrows = 10000000
def getwinHeadTargetChance():
 df_full = pd.read_csv("featuresHeadTargetChance.csv", sep=',')
 return df_full.astype(float)

def getwinMiddleTargetChance():
 df_full = pd.read_csv("featuresMiddleTargetChance.csv", sep=',')
 return df_full.astype(float)

def getwinTailTargetChance():
 df_full = pd.read_csv("featuresTailTargetChance.csv", sep=',')
 return df_full.astype(float)

def getwinHeadTargetME():
 df_full = pd.read_csv("featuresHeadTargetME.csv", sep=',')
 return df_full.astype(float)

def getwinMiddleTargetME():
 df_full = pd.read_csv("featuresMiddleTargetME.csv", sep=',')
 return df_full.astype(float)

def getwinTailTargetME():
 df_full = pd.read_csv("featuresTailTargetME.csv", sep=',')
 return df_full.astype(float)

def getFeaturesTail():
 df_full = pd.read_csv("featuresTail.csv", sep=',',  dtype = 'bool') #
 return df_full.astype(bool)

def getFeaturesMiddle():
 df_full = pd.read_csv("featuresMiddle.csv", sep=',',  dtype = 'bool')
 return df_full.astype(bool)

def getFeaturesHead():
 df_full = pd.read_csv("featuresHead.csv", sep=',', dtype = 'bool')
 return df_full.astype(bool)

def createModel(features, target, name):
 model = Sequential()
 model.add(Dense(165, input_dim=156, activation='relu'))
 model.add(Dense(428, activation='relu'))
 model.add(Dense(1028, activation='relu'))
 model.add(Dense(428, activation='relu'))
 model.add(Dense(100, activation='relu'))
 model.add(Dense(28, activation='relu'))
 model.add(Dense(1, activation='tanh'))
 opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
 model.compile(loss='mse', optimizer=opt, metrics=['mae'])
 print(features.shape)
 print(target.shape)
 x_train, x_validation, y_train, y_validation = train_test_split(features, target, test_size=0.02, shuffle=False)
 print(x_train.shape)
 print(y_train.shape)
 print(x_validation.shape)
 print(x_validation.shape)
 model.fit(x_train, y_train, batch_size=20000, epochs=40, validation_data=(x_validation, y_validation),callbacks=[csv_logger])
 model.save(str(name))

X = getFeaturesHead()
Y = getwinHeadTargetChance()
createModel(X,Y,'HeadTargetChance.h5')

Y = getwinHeadTargetME()
createModel(X,Y,'HeadTargetME.h5')

X = getFeaturesMiddle()
Y = getwinMiddleTargetChance()
createModel(X,Y,'MiddleTargetChance.h5')

Y = getwinMiddleTargetME()
createModel(X,Y,'MiddleTargetME.h5')

X = getFeaturesTail()
Y = getwinTailTargetChance()
createModel(X,Y,'TailTargetChance.h5')

Y = getwinTailTargetME()
createModel(X,Y,'TailTargetME.h5')