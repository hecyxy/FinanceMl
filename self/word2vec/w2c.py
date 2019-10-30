from gensim.models import fasttext
from gensim.models import word2vec
import pandas as pd
import logging
import jieba
import os
import sys

# 此函数作用是对初始语料进行分词处理后，作为训练模型的语料
def pre_txt(old_file):
    global cut_file     # 分词之后保存的文件名
    cut_file = old_file + '_cut.txt'
    try:
        fi = open(old_file, 'r', encoding='utf-8',HMM = True)
    except BaseException as e:  # 因BaseException是所有错误的基类，用它可以获得所有错误类型
        print(Exception, ":", e)    # 追踪错误详细信息

    text = fi.read()  # 获取文本内容
    new_text = jieba.cut(text, cut_all=False)  # 精确模式
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')     # 去掉标点符号
    fo = open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)

#sg: 如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型
def skipgram_model_train(train_file_name):  # model_file_name为训练语料的路径,save_model为保存模型名
    # 模型训练，生成词向量
    save_model_file = './self/word2vec/skip_xyj.model'
    save_model_name = './self/word2vec/skip_xyj.model'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = word2vec.Word2Vec(sentences, sg = 1,size=200)  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)   # 以二进制类型保存模型以便重用

def cbow_model_train(train_file_name):
    save_model_file = './self/word2vec/cbow_xyj.model'
    save_model_name = './self/word2vec/cbow_xyj.model'
    sentences = word2vec.Text8Corpus(train_file_name)
    model = word2vec.Word2Vec(sentences,sg = 0,size = 200) #c-bow模型
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)   # 以二进制类型保存模型以便重用

def train_model(model_name,mode):
    print(os.path.exists(model_name))
    cut_file = './self/word2vec/xyj.txt_cut.txt'
    if not os.path.exists(model_name):     # 判断文件是否存在
        if mode == 1:
            skipgram_model_train(cut_file)
        else:
            cbow_model_train(cut_file)
    else:
        print('此训练模型已经存在，不用再次训练')
if __name__ == "__main__":
    #pre_txt('./self/word2vec/xyj.txt')  # 须注意文件必须先另存为utf-8编码格式
    param1 = sys.argv[1]
    print(param1)
    if param1 == 'skip':
        save_model_name = './self/word2vec/skip_xyj.model'  
        train_model(save_model_name,1)
    elif param1 == 'cbow':
        save_model_name = './self/word2vec/cbow_xyj.model'  
        train_model(save_model_name,2)
    else:
        print('please send param')
        sys.exit(0)
    # 加载已训练好的模型
    print("name",save_model_name)
    model_1 = word2vec.Word2Vec.load(save_model_name)
    print('vector:',model_1['悟空'])
    # 计算两个词的相似度/相关程度
    y1 = model_1.similarity("孙悟空", "唐僧")

    print(u"孙悟空和唐僧的相似度为：", y1)
    print("-------------------------------\n")

    # 计算某个词的相关词列表
    y2 = model_1.most_similar("唐僧", topn=10)  # 10个最相关的
    print(u"和唐僧最相关的词有：\n")
    for item in y2:
        print(item[0], item[1])
    print("-------------------------------\n")