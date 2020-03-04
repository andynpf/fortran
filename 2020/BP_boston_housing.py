# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))
print('-------------------')
print(y_train_pd.head(5))
print('-------------------')
print(x_valid_pd.head(5))
print('-------------------')
print(y_valid_pd.head(5))
from sklearn.metrics import r2_score

import numpy
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#matplotlib inline
from pandas import read_csv
import math

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
x_train_pd, y_train_pd = create_dataset(train, look_back)
x_valid_pd, y_valid_pd = create_dataset(test, look_back)
x_train = x_train_pd
y_train = y_train_pd
x_valid=x_valid_pd
y_valid=y_valid_pd

# 训练集归一化
min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(x_train_pd)
# x_train = min_max_scaler.transform(x_train_pd)
#
# min_max_scaler.fit(y_train_pd)
# y_train = min_max_scaler.transform(y_train_pd)


# 验证集归一化
# min_max_scaler.fit(x_valid_pd)
# x_valid = min_max_scaler.transform(x_valid_pd)
#
# min_max_scaler.fit(y_valid_pd)
# y_valid = min_max_scaler.transform(y_valid_pd)

# 单CPU or GPU版本，若有GPU则自动切换
model = Sequential()  # 初始化，很重要！
model.add(Dense(units = 10,   # 输出大小
                activation='relu',  # 激励函数
                input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小
               )
         )

model.add(Dropout(0.2))  # 丢弃神经元链接概率

model.add(Dense(units = 15,
#                 kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
#                 activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
                activation='relu' # 激励函数
                # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
               )
         )

model.add(Dense(units = 1,
                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数
               )
         )

print(model.summary())  # 打印网络层次结构

model.compile(loss='mse',  # 损失均方误差
              optimizer='adam',  # 优化器
             )

history = model.fit(x_train, y_train,
          epochs=200,  # 迭代次数
          batch_size=128,  # 每次用来梯度下降的批处理数据大小
          verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (x_valid, y_valid)  # 验证集
        )

# 多GPU版本
# parallel_model = multi_gpu_model(model, gpus=4)
# parallel_model.compile(loss='mse',  # 多分类
#                        optimizer='adam',
#                       )

# This `fit` call will be distributed on 4 GPUs.
# Since the batch size is 50, each GPU will process 32 samples.
# batch_size = 512
# epochs = 2
# history = parallel_model.fit(
#           x_train,
#           y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split = 0.2  # 从训练集分割出20%的数据作为验证集
#         )


import matplotlib.pyplot as plt
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


from keras.utils import plot_model
from keras.models import load_model
# 保存模型
model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'

#模型可视化 pip install pydot
plot_model(model, to_file='model_MLP.png', show_shapes=True)

# 加载模型
model = load_model('model_MLP.h5')


# 预测
x_valid = scaler.inverse_transform(x_valid_pd)
y_new = model.predict(x_valid)
# 反归一化
# scaler.fit(y_valid_pd)
y_new = scaler.inverse_transform(y_new)

trainY = y_train
testY =y_valid
trainX = x_train
testX =x_valid
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