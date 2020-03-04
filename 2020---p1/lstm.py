# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt

# from __future__ import absolute_import

# from functools import partial
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Conv2D #SimpleCNN

from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


from keras.layers import SimpleRNN
from keras.layers.sru import SRU
# from keras.layers.sru import SRU_tensorflow
#import torch

from sklearn.metrics import r2_score

import numpy
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#matplotlib inline


def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score
dataframe = read_csv('test1.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

numpy.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print('testX',type(testX))

model = Sequential()

#model.add(Conv2D(64, (2,2), activation='relu'))
#LSTM
#GRU
#
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
#20190704npf-A
model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(trainX, trainY, epochs=20, batch_size=128, verbose=2)

print(model.summary())#20190704npf-A

fig = plt.figure()
plt.plot()
plt.plot(history.history['loss'])

plt.title('model - loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()

#plt.savefig()



trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

y_pred = testY[0]
y_true = testPredict[:,0]
score = performance_metric(y_true, y_pred)
print ("R^2: {:.3f}.".format(score))


# 在具有二元标签指示符的多标签分类案例中
#print(accuracy_score(numpy.array([[0, 1], [1, 1]]), numpy.ones((2, 2)))) # 0.5


# print(accuracy_score(y_true, y_pred)) # 0.5
# print(accuracy_score(y_true, y_pred, normalize=False)) # 2


trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

fig = plt.figure()

plt.plot(scaler.inverse_transform(dataset), linewidth=2, linestyle=":")
plt.plot(trainPredictPlot, linewidth=1, linestyle="-")
plt.plot(testPredictPlot, linewidth=1, linestyle="-")
fig.savefig('Final.png',dpi=600)
plt.show()