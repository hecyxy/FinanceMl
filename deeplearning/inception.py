# -*- coding: UTF-8 -*-
from keras import layers
import numpy as np
x = np.array((None,4))
branch_a = layers.Conv2D(128,1,activation='relu',strides=2)(x)
branch_b = layers.Conv2D(128,1,activation='relu')(x)
branch_b = layers.Conv2D(128,3,activation='relu',strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3,strides=2)(x)
branch_c = layers.Conv2D(128,3,activation='relu')(branch_c)

branch_d = layers.Conv2D(128,1,activation='relu')(x)
branch_d = layers.Conv2D(128,3,activation='relu')(branch_d)
branch_d = layers.Conv2D(128,3,activation='relu',strides=2)(branch_d)
#将分支连接在一起得到模块输出
output = layers.concatenate([branch_a,branch_b,branch_c,branch_d],axis = -1)

#残差连接
y = layers.Conv2D(128,3,activation='relu',padding='same')(x)
y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
y = layers.Conv2D(128,3,activation='relu',padding='same')(y)
#将原始x与输出特征相加
y = layers.add([y,x]) 

from keras import Input
from keras.models import Model
#共享LSTM模型
lstm = layers.LSTM(32)
#左分支
left_input = Input(shape=(None,128))
left_output = lstm(left_input)
#右分支
right_input = Input(shape=(None,128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output,right_output],axis=-1)
predictions = layers.Dense(1,activation='sigmoid')(merged)

model = Model([left_input,right_input],predictions)

#model.fit([left_data,right_data],targets)
#可以多次重复使用，调用多次，每次都重复使用一组权重

