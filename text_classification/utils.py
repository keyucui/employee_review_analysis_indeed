# coding: UTF-8
import os
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from clean_str import *


MAX_VOCAB_SIZE = 30000  # 词表长度限制
MIN_FREQUENCY = 5
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}

    data = pd.read_csv(file_path)
    for row in data.itertuples():
        content = getattr(row, 'text')
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    # with open(file_path, 'r', encoding='UTF-8') as f:
    #     for line in tqdm(f):
    #         lin = line.strip()
    #         if not lin:
    #             continue
    #         content = lin.split('\t')[0]
    #         for word in tokenizer(content):
    #             vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_vocab_cons(file_path, tokenizer, max_size=30000, min_freq=5):
    vocab_dic = {}
    print('cons 词表最大为 {}'.format(max_size + 2))
    data = pd.read_csv(file_path)
    for row in data.itertuples():
        content = getattr(row, 'cons')
        if content != "MISS":
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        content = getattr(row, 'pros')              # cons 和pros 共用一个词典
        if content != "MISS":
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_vocab_individual(file_path, idx_col='company_id'):
    # 为individual创建embedding的vocab。如indeed 每个公司id 对应第几个embedding
    data = pd.read_csv(file_path)
    # print(list(data[idx_col].unique())[:10])
    individuals = list(data[idx_col].unique())
    # print(len(individuals))
    # print(type(individuals[0]))

    vocab_dic = {ind: ii for ii, ind in enumerate(individuals)}
    vocab_dic.update({PAD: len(vocab_dic)})
    return vocab_dic


def build_dataset(config, fix_effect=False, cons=False):
    tokenizer = process_sentencce_tokenizer             # 预处理
    print('building vocab......')
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQUENCY)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    if fix_effect:
        # 如果要控制固定效应
        if os.path.exists(config.vocab_individual_path):
            vocab_individual = pkl.load(open(config.vocab_individual_path, 'rb'))
        else:
            vocab_individual = build_vocab_individual(config.train_path, idx_col=config.individual_col)
            pkl.dump(vocab_individual, open(config.vocab_individual_path, 'wb'))

    if cons:
        print('building vocab of cons......')
        if os.path.exists(config.vocab_cons_path):
            vocab_cons = pkl.load(open(config.vocab_cons_path, 'rb'))
        else:
            vocab_cons = build_vocab_cons(config.train_path, tokenizer)
            pkl.dump(vocab_cons, open(config.vocab_cons_path, 'wb'))

    def load_dataset_cons(path, pad_size, pad_size_cons=10):
        contents = []
        data = pd.read_csv(path)
        for row in data.itertuples():
            content, label = getattr(row, 'text'), getattr(row, 'overall_rating') - 1
            content_cons = getattr(row, 'cons')
            content_pros = getattr(row, 'pros')

            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size

            words_line_cons = []
            if content_cons == 'MISS':
                token_cons = [PAD] * pad_size_cons
                seq_len_cons = pad_size_cons
            else:
                token_cons = tokenizer(content_cons)
                seq_len_cons = len(token_cons)
                if len(token_cons) < pad_size_cons:
                    token_cons.extend([PAD] * (pad_size_cons - len(token_cons)))
                else:
                    token_cons = token_cons[:pad_size_cons]
                    seq_len_cons = pad_size_cons

            words_line_pros = []
            if content_pros == 'MISS':
                token_pros = [PAD] * pad_size_cons
                seq_len_pros = pad_size_cons
            else:
                token_pros = tokenizer(content_pros)
                seq_len_pros = len(token_pros)
                if len(token_pros) < pad_size_cons:
                    token_pros.extend([PAD] * (pad_size_cons - len(token_pros)))
                else:
                    token_pros = token_pros[:pad_size_cons]
                    seq_len_pros = pad_size_cons

            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))

            for word in token_cons:
                words_line_cons.append(vocab_cons.get(word, vocab_cons.get(UNK)))

            for word in token_pros:
                words_line_pros.append(vocab_cons.get(word, vocab_cons.get(UNK)))

            contents.append((words_line, int(label), seq_len, words_line_cons, words_line_pros, seq_len_cons))
        print(len(contents[0]))
        return contents  # [([...], 0), ([...], 1), ...]


    def load_dataset_fix_effect(path, pad_size):
        '''
        加上了individual fix effect ， 比如indeed的firm embedding
        '''
        contents = []
        data = pd.read_csv(path)
        for row in data.itertuples():
            content, label = getattr(row, 'text'), getattr(row, 'overall_rating') - 1
            individual = getattr(row, config.individual_col)     # 固定效应的individual

            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len, [vocab_individual[individual]]))
            # 最后一项是该individual表对应的index（该评论是哪个公司的员工说的）
        return contents

    def load_dataset(path, pad_size=32):
        contents = []
        data = pd.read_csv(path)
        for row in data.itertuples():
            content, label = getattr(row, 'text'), getattr(row, 'overall_rating') - 1

            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
        # with open(path, 'r', encoding='UTF-8') as f:
        #     for line in tqdm(f):
        #         lin = line.strip()
        #         if not lin:
        #             continue
        #         content, label = lin.split('\t')
        #         words_line = []
        #         token = tokenizer(content)
        #         seq_len = len(token)
        #         if pad_size:
        #             if len(token) < pad_size:
        #                 token.extend([PAD] * (pad_size - len(token)))
        #             else:
        #                 token = token[:pad_size]
        #                 seq_len = pad_size
        #         # word to id
        #         for word in token:
        #             words_line.append(vocab.get(word, vocab.get(UNK)))
        #         contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    if not fix_effect and not cons:
        train = load_dataset(config.train_path, config.pad_size)
        dev = load_dataset(config.dev_path, config.pad_size)
        test = load_dataset(config.test_path, config.pad_size)
        return vocab, train, dev, test
    elif not fix_effect and cons:
        # 不固定firm， 使用cons
        print('load dataset with cons')
        train = load_dataset_cons(config.train_path, config.pad_size)
        dev = load_dataset_cons(config.dev_path, config.pad_size)
        test = load_dataset_cons(config.test_path, config.pad_size)
        return vocab, train, dev, test, vocab_cons
    else:
        train = load_dataset_fix_effect(config.train_path, config.pad_size)
        dev = load_dataset_fix_effect(config.dev_path, config.pad_size)
        test = load_dataset_fix_effect(config.test_path, config.pad_size)
        return vocab, train, dev, test, vocab_individual


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, individual=False, cons=False):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数

        self.individual = individual
        self.cons = cons            # 是否加入cons embedding

        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def _to_tensor_individual(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x_individual = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, x_individual), y

    def _to_tensor_cons(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x_cons = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        x_pros = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, x_cons, x_cons), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            if not self.cons and not self.individual:
                batches = self._to_tensor(batches)
            elif self.cons and not self.individual:
                batches = self._to_tensor_cons(batches)
            else:
                batches = self._to_tensor_individual(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            # batches = self._to_tensor(batches)
            if not self.cons:
                batches = self._to_tensor(batches)
            else:
                batches = self._to_tensor_cons(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, individual=False, cons=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, individual, cons)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
