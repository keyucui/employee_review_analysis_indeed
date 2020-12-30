import pandas as pd
import numpy as np
import sys
import time
import multiprocessing
from gensim.models import Word2Vec
import gensim
from gensim.models.word2vec import LineSentence
from collections import Counter
import string

from clean_str import *


if __name__ == '__main__':
    t0 = time.time()
    print('load sentences in train data to train word2vec ------')
    # inp: 输入数据路径
    # outp1: 训练好的模型存储路径
    # outp2: 得到的词向量存储路径
    data_name = 'Indeed'
    text_col = 'text'
    inp = data_name + '/data/train.csv'
    outp1 = data_name + '/embedding/word2vec_text.model'
    outp2 = data_name + '/embedding/word2vec_text_wv.model'

    data = pd.read_csv(inp)
    data = data.dropna(subset=[text_col])
    sentences = list(data[text_col])
    print('一共有 {} 条文本'.format(len(sentences)))
    # sentences = [sen.lower() for sen in sentences]
    # rem = str.maketrans('','', string.punctuation)      # 消除标点符号
    # sentences = [sen.translate(rem) for sen in sentences]
    # sentences = [sen.split(' ') for sen in sentences]
    sentences = [process_sentencce_tokenizer(sen) for sen in sentences]

    print('耗费{} s clean over ------ start to train word2vec --------'.format(time.time() - t0))
    # sentences = [['and', 'I', 'is', 'are'], ['and', 'is', 'me']]

    '''
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    size：是每个词的向量维度；
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词；
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃；
    workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
    alpha (float, optional) – 初始学习率
    iter (int, optional) – 迭代次数，默认为5
    '''
    model = Word2Vec(sentences, size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())

    model.save(outp1)
    #不以C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format(outp2, binary=False)
    print('训练完毕， 耗时 {} s'.format(time.time() - t0))
