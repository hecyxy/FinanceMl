# -*- coding: UTF-8 -*-
from keras.datasets import boston_housing
from keras import models
from keras import layers
(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    #常见的回归指标是平均绝对误差MAE
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model



