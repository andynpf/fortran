from sklearn import preprocessing
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
# x = rng.uniform(1, 100, (100, 1))
x = np.linspace(-2, 2, 1000)
# y = 5 * x + np.sin(x) * 5000 + 2 + np.square(x) + rng.rand(100, 1) * 5000
y = 5 * x + np.sin(2 * 3.14 * x) + 2
plt.plot(x, y)
x = x.reshape((-1, 1))
#####s随机打乱
n_index = np.random.permutation(len(x))

# ###########利用随机索引划分验证集和测试集
train_x_disorder = x[n_index[0:800]]
test_x_disorder = x[n_index[800:len(x) + 1]]
train_y_disorder = y[n_index[0:800]]
test_y_disorder = y[n_index[800:len(x) + 1]]


# # # ss_y = preprocessing.StandardScaler()
ss_y =  preprocessing.MinMaxScaler()
train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))
test_y_disorder = ss_y.transform(test_y_disorder.reshape(-1, 1))
#



def build_model():
    model = models.Sequential()
    model.add(layers.Dense(80,activation='relu',input_dim=1))
    model.add(layers.Dense(1))
    opt = optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt,loss='mse',metrics=['mae'])
    return model

model = build_model()
t1 = time.time()
his = model.fit(train_x_disorder,train_y_disorder,epochs=100,batch_size=len(train_x_disorder),verbose=100)
print('train time is : ',time.time()-t1)


pre = model.predict(train_x_disorder,batch_size=1)
pre_p = ss_y.inverse_transform(pre)
fig=plt.figure()
plt.title('model - predict')
# plt.plot(x,y,'r')
plt.plot(train_x_disorder,pre,'bo')
plt.plot(train_x_disorder,train_y_disorder,'r*')
fig.savefig('BP.png',dpi=600)

yy = ss_y.inverse_transform(train_y_disorder)
error = pre_p.ravel()-yy.ravel()
plt.figure()
plt.plot(error)
plt.title('model - error')
# plt.ylabel('loss')
# plt.xlabel('epoch')
plt.show()

