# -*- coding: UTF-8 -*-
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,),dtype='int32',name='posts')
embeded_posts = layers.Embedding(256,vocabulary_size)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(embeded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256,5,activation = 'relu')(x)
x=layers.Conv1D(256,5,activation = 'relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128,activation = 'relu')(x)
#输出层都有名称 name
age_predication = layers.Dense(1,name='age')(x) 
income_prediction = layers.Dense(num_income_groups,
                            activation = 'softmax',
                            name = 'income')(x)
gender_prediction = layers.Dense(1,activation = 'sigmoid',name='gender')(x)
model = Model(posts_input,[age_predication,income_prediction,gender_prediction])

#多输出模型的多重损失
model.compile(optimizer='rmsprop',loss=['mse','categorical_crossentropy','binary_crossentropy'])
#与上述模型等效
model.compile(optimizer='rmsprop',loss={'age':'mse','incode':'categorical_crossentropy','gender':'binary_crossentropy'})



