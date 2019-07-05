# -*- coding: UTF-8 -*-
#采样过程 控制随机性大小 用softmax温度参数 表示采样概率分布的熵
import numpy as np
#original_ids 概率值数组 和为1
#返回原始分布重新加权后的结果
#更高的温度更随机  更低的温度更确定
def reweight_distribution(original_dis,temperature=0.5):
    distribution = np.log(original_dis)/temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

text = open('./deeplearning/data/nietzsche.txt').read().lower()
#字符串向量化 提取60个字符组成的序列
maxlen = 60
step = 3#每3个字符采样一个新序列
sentences = []#保存提取的序列
next_chars = []#保存下一个字符
for i in range(0,len(text) - maxlen,step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print('sentences length:',len(sentences))
chars = sorted(list(set(text)))#语料中唯一字符组成的列表
char_indices = dict((char,chars.index(char)) for char in chars)
#one-hot编码为二进制数组
x = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y = np.zeros((len(sentences),len(chars)),dtype=np.bool)
for i,sentences in enumerate(sentences):
    for t,char in enumerate(sentences):
        x[i,t,char_indices[char]] = 1
    y[i,char_indices[next_chars[i]]] = 1

from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.LSTM(128,input_shape=(maxlen,len(chars))))
model.add(layers.Dense(len(chars),activation='softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)
#给定训练好的模型和一个种子文本片段  给定目前已生成的文本 从模型得到下一个字符的概率分布
#定义采样函数
def sample(preds,temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

import random
import sys
for epoch in range(1,60):
    model.fit(x,y,batch_size=128,epochs=1)
    start_index = random.randint(0,len(text) - maxlen -1)
    generated_text = text[start_index:start_index+maxlen]
    print('--- generating with seed:"'+generated_text+'"')

for temperature in [0.2,0.5,1.0,1.2]:
    sys.stdout.write(generated_text)
    for i in range(400):
        sampled = np.zeros((1,maxlen,len(chars)))
        for t,char in enumerate(generated_text):
            sampled[0,t,char_indices[char]] = 1 #对生成字符进行one——hot编码
        preds = model.predict(sampled,verbose=0)[0]
        next_index = sample(preds,temperature)#对下个字符进行采样
        next_char = chars[next_index]
        generated_text += next_char
        generated_text = generated_text[1:]
        sys.stdout.write(next_char)

#较小的温度，会得到计算重复和可预测的文本，局部结构非常真实
#温度越来越大，生成的文本更有趣，更出人意料，更有创造性，局部模式开始分解
