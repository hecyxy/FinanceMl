# -*- coding: UTF-8 -*-
import os
import numpy as np
from matplotlib import pyplot as plt
data_dir = './deeplearning/data'
fname = os.path.join(data_dir,'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
# 共有 420552 行数据
# 每行记录一个日期和14个与天气有关的值

#将数据转为一个Numpy数据
float_data = np.zeros((len(lines),len(header) - 1))
for i,line in enumerate(lines):
    if i > 0 :
     values = [float(x) for x in line.split(',')[1:]]
     float_data[i,:] = values

temp = float_data[:,1]#温度
plt.plot(range(len(temp)),temp)
plt.show()
#数据预处理 每个时间序列减去其平均值 除以标准差 
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std
#数据生成器 data:浮点数原始数组  lookcback: 包括多少个时间步 delay: 应该在未来多少个时间步后
#min_index max_index:data数组中的索引,用于界定需要抽取哪些时间步 shuffle:是否打乱样本
#batch_size:每个批量样本数  step:数据采样的周期，将其设为6
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback,max_index,size= batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i,min(i+batch_size,max_index))
            i += len(rows)

        samples = np.zeros((len(rows),lookback,data.shape[-1]))

        targets = np.zeros((len(rows)))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples,targets

#使用generator实例三个生成器分别用于训练 验证 测试
lookback = 1440
step = 6
delay = 144
batch_size = 128
#训练集
train_gen = generator(float_data,
lookback = lookback,delay = delay,min_index = 0,
max_index = 200000,shuffle = True,
step = step,batch_size = batch_size)
#验证集
val_gen = generator(float_data,
lookback = lookback,delay = delay,min_index = 200001,
max_index = 300000,shuffle = True,
step = step,batch_size = batch_size)
#测试集
train_gen = generator(float_data,
lookback = lookback,delay = delay,min_index = 300001,
max_index = None,shuffle = True,
step = step,batch_size = batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)
#使用平均绝对误差MAE评估这种方法
#np.mean(np.abs(preds - targets))
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples,targets = next(val_gen)
        preds = samples[:,-1,1]
        mae = np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()

#先训练一个基本的机器学习方法
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape = (lookback // step,float_data.shape[-1])))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)
#绘制验证和训练的损失曲线
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.figure()

plt.plot(epochs,loss,'bo',label = 'Train loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('traing and validation loss')
plt.legend()
plt.show()

#全连接效果不太好 将时间序列铺平，删除了时间的概念，时间序列中因果、顺序都很重要
#GRU网络

model1 = Sequential()
model1.add(layers.GRU(32,input_shape = (None,float_data.shape[-1])))
model1.add(layers.Dense(1))
model1.compile(optimizer=RMSprop(),loss='mae')
history = model1.fit_generator(train_gen,steps_per_epoch=500,epochs = 20,validation_data = val_gen,validation_steps=val_steps)

#每个时间步使用相同的dropout掩码(dropout mask，相同模式的舍弃单元) 
#而不是让dropout掩码随着时间步增加而随机变化
#对GRU LSTM等循环层得到的表示做正则化，将不随时间变化的dropout掩码应用于层的内部循环激活（循环dropout掩码）
#对每个时间步使用相同的 dropout 掩码，可以让网络 沿着时间正确地传播其学习误差，而随时间随机变化的 dropout 掩码则会破坏这个误差信号，并 7 且不利于学习过程。
advance = Sequential()
advance.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
advance.add(layers.Dense(1))
advance.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs = 40,validation_data=val_gen,validation_steps=val_steps)

#训练并评估一个使用dropout正则化的堆叠GRU模型
advance1=Sequential()
advance1.add(layers.GRU(32,dropout=0.1,recurrent_sequences=True,input_shape=(None,float_data.shape[-1])))
advance1.add(layers.GRU(64,activation='relu',dropout=0.1,recurrent_dropout=0.5))
advance1.add(layers.Dense(1))
advance1.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

#使用双向RNN
biModel = Sequential()
biModel.add(layers.Bidirectional(layers.GRU(32),input_shape = (None,float_data.shape[-1])))
biModel.add(layers.Dense(1))
biModel.compile(optimizer = RMSprop(),loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps = val_steps)

