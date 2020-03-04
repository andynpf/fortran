# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:13:37 2019

@author: jason
"""

# univariate cnn-lstm example
from numpy import array
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#index
# define dataset
dataframe = read_csv('testCnnLstm--.csv', usecols=[3],engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')


dataframe_test = read_csv('testCnnLstmAll.csv', usecols=[3],engine='python')
dataset_test = dataframe_test.values
dataset_test = dataset_test.astype('float32')

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)




scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainY, testY = dataset[train_size:len(dataset),:], dataset[train_size:len(dataset),:]


look_back = 365



trainX = trainX.reshape(-1, 4, 4, 1)
#trainX = trainX.reshape(trainX.shape[0], 4, 4, 1)
trainY = trainY.reshape(-1, 4)




X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([50, 60,70,80])
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
X = X.reshape(-1, 2, 2, 1)
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,4,1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(4))
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(optimizer='adam', loss='mse')
# fit model
#model.fit(X, y, epochs=50, verbose=0)
history =model.fit(trainX, trainY, epochs=1000, verbose=0)

fig = plt.figure()
plt.plot()
plt.plot(history.history['loss'])

plt.title('model - loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()

# demonstrate prediction
#x_input = array([50, 60,70,80])
#x_input = x_input.reshape((1, 2, 2, 1))
#yhat = model.predict(x_input, verbose=0)

#testX = array([[10,10.3   ,11,11.770237],[10.3   ,11,11.7,12.770237],[11,11.7,12.7,13.770237],[11.7,12.7,13.77,14.770237]])
testX = dataset.reshape((-1, 4, 4,1))
#testX = testY.reshape((-1, 4, 4,1))
#testY = testY.reshape(-1, 4)


testPredict  = model.predict(testX, verbose=0)

testPredict = scaler.inverse_transform(testPredict)
testPredict = testPredict.reshape(-1, 1)


testScore = math.sqrt(mean_squared_error(dataset_test[len(dataset):len(dataset)+len(testPredict),0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
testPredictPlot = numpy.empty_like(dataset_test)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(dataset):len(dataset)+len(testPredict), :] = testPredict[:,:]

fig = plt.figure()

plt.plot(dataset_test, linewidth=2, linestyle=":")
#plt.plot(trainPredictPlot, linewidth=1, linestyle="-")
plt.plot(testPredictPlot, linewidth=1, linestyle="-")
fig.savefig('Final.png',dpi=600)

plt.show()