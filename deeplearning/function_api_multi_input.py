# -*- coding: UTF-8 -*-
from keras import Input,layers
from keras.models import Model
#函数式API实现双输入问答模型
text_vocabulary_size = 10000 
question_vocabulary_size = 10000 
answer_vocabulary_size = 500

text_input = Input(shape=(None,),dtype='int32',name='text')
embedded_text = layers.Embedding(text_vocabulary_size,64)(text_input)#将输入嵌入长度为64的向量

encoded_text = layers.LSTM(32)(embedded_text) #利用LSTM将向量编码为单个向量
#对问题进行相同处理 使用不同的层实例
question_input = Input(shape=(None,),dtype = 'int32' ,name = 'question')

embedded_question = layers.Embedding(question_vocabulary_size,32)(question_input)

encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text,encoded_question],axis= - 1)#将编码后的问题和文本连接起来

#添加softmax分类器
answer = layers.Dense(answer_vocabulary_size,activation = 'softmax')(concatenated)

#模型实例化时，指定两个输入和输出
model = Model([text_input,question_input],answer)
model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics=['acc'])

import numpy as np
from keras import utils as np_utils
num_samples = 1000
max_length = 100
text = np.random.randint(1,text_vocabulary_size,size=(num_samples,max_length))
question = np.random.randint(1,question_vocabulary_size,size=(num_samples,max_length))

answers = np.random.randint(answer_vocabulary_size,size=(num_samples))
answers = np_utils.to_categorical(answers,answer_vocabulary_size)

model.fit([text,question],answers,epochs=10,batch_size=128)
model.fit({'text':text,'question':question},answers,epochs=10,batch_size=128)
