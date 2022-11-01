import numpy as np # Библиотека работы с массивами
import pandas as pd # Библиотека pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('winHead.csv', sep=',')
print(df.shape) # Размерность данных

dataset = df.values                 # Берем только значения массива(без индексов)
X = dataset[:,:-1].astype(int)   # НЕ Присваиваем им тип данных - float(с плавающей точкой) данным с 0 по 60 колонки

Y = dataset[:,-1]                   # Присваеваем значению Y данные из столбца с индексом 60

print(X.shape)                      # Выводим размерность X
print(Y.shape)                      # Выводим размерность Y
print(Y)

#Y=Y/7
Y=Y/100
print(Y.max())
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)


x_train_new, x_val, y_train_new, y_val = train_test_split(x_train,y_train,test_size=0.2)
x_train_new = x_train_new.astype(int)
x_val = x_val.astype(int)
print(x_train_new.shape)
print(x_val.shape)
print(y_train_new.shape)
print(y_val.shape)



model = DecisionTreeRegressor()
model.fit(x_train_new, y_train_new)
sss = model.predict(x_test)
print(sss)

model = RandomForestRegressor()
model.fit(x_train_new, y_train_new)
sss = model.predict(x_test)
print(sss)

model = Ridge()
model.fit(x_train_new, y_train_new)
sss = model.predict(x_test)
print(sss)

model = RidgeCV()
model.fit(x_train_new, y_train_new)
sss = model.predict(x_test)
print(sss)

model = LassoCV()
model.fit(x_train_new, y_train_new)
sss = model.predict(x_test)
print(sss)

model = ElasticNetCV()
model.fit(x_train_new, y_train_new)
sss = model.predict(x_test)
print(sss)

print(y_test)