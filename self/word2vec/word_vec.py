import numpy as np
import pandas as pd
from gensim.models import word2vec


def load(path):
    data = np.load(path)
    return data


def get_sent_vec(size,sent,model):
    vec = np.zeros(size).reshape(1,size)
    count = 0
    for word in sent:
        try:
            vec += model[word].reshape(1,size)
            count += 1
        except:
            continue
    if count != 0:
        vec /= count
    return vec

def train(x_train,train_model):
    train_vec = np.concatenate([get_sent_vec(300,sent,train_model) for sent in x_train])
    np.save('',train_vec)
    return train_vec

x_train_data = load('')
train_model = word2vec.Word2Vec.load('.model')
train_vec = train(x_train_data,train_model)