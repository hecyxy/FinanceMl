import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform

def load_data():
    train = _parse_data2(open('ner/data/train_name_pro.data','r',encoding = 'utf-8'))
    test = _parse_data2(open('ner/data/test_name_pro.data','r',encoding = 'utf-8'))
    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w,f in iter(word_counts.items()) if f>=2]
    # chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
    chunk_tags = ['B-PRO', 'I-PRO', "B-ORG", "I-ORG"]# 'B-LOC', 'I-LOC',
    with open('ner/model/search2.pkl','wb') as outp:
        pickle.dump((vocab,chunk_tags),outp)
    train = _process_data(train,vocab,chunk_tags)
    test = _process_data(test,vocab,chunk_tags)
    return train,test,(vocab,chunk_tags)
def _parse_data(fn):
    if platform.system() == 'Windows':
        split_text='\r\n'
    else:
        split_text = '\n'
    string = fn.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for 
        sample in string.strip().split(split_text+split_text)]
    fn.close()
    return data

def _parse_data2(fn):
    line = fn.readline()
    data = []
    j = 0
    while j < 1000000:
        line = fn.readline()
        if not line:
            break
        data.append([[ t for t in s.split(' ') ] for s in line.strip().split('|') ])
        j+=1
    fn.close()
    return data
    


def _process_data(data,vocab,chunk_tags,maxlen=None,onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w,i) for i,w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(),1) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x,maxlen)
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk
        

def process_data(data,vocab,maxlen=100):
    word2idx = dict((w,i) for i,w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(),1) for w in data]
    length = len(x)
    x = pad_sequences([x],maxlen)
    return x,length

load_data()