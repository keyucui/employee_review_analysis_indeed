import pandas as pd
import random
import time
from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch

random.seed(2020)


def file_test(path='Indeed/data/train.csv'):
    t1 = time.time()
    k = 0
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            print(lin)
            if k > 2:
                break
            k += 1
    print('file open 耗时 {} s'.format(time.time() - t1))

def csv_test2(path='Indeed/data/train.csv'):
    t2 = time.time()
    k = 0
    for row in pd.read_csv(path, iterator=True):
        print(row)
        if k >= 1:
            break
        k += 1


def csv_test(path='Indeed/data/train.csv'):
    t2 = time.time()
    data = pd.read_csv(path)
    t3 = time.time()
    k = 0

    for row in data.itertuples():  # 按行遍历最快
        # pass
        # print(getattr(row, 'text'), getattr(row, 'overall_rating'))
        print(type(getattr(row, 'overall_rating')))
        print(type(getattr(row, 'text')))
        if k >= 1:
            break
        k += 1
    # for idx, row in data.iterrows():
    #     pass
        # if k >= 1:
        #     break
        # k += 1
        # print(row)
    # for idx in data.index:
    #     line = data.loc[idx]['text'].strip()
    print('file open 耗时 {} s'.format(t3 - t2))
    print('file process 耗时 {} s'.format(time.time() - t3))


def vocab_show(path='Indeed/data/vocab.pkl'):
    vocab = pkl.load(open(path, 'rb'))
    # print(vocab)
    return vocab


def split_train_dev_test(data_path=None, dev_rate=0.1, test_rate=0.1, min_count=10):
    '''
    对总的数据集进行划分，由于关注firm embedding，因此应该对每个firm应该按同样比例的采样
    '''
    t0 = time.time()
    train_rate = 1 - dev_rate - test_rate
    print('读取数据集 --------- \n')
    if not data_path:
        data = pd.read_csv('Indeed/DL_indeed_review.csv')
        data = data.dropna(subset=['text', 'cons', 'text'])
        data['text_length'] = data['text'].apply(lambda x: len(x.split(' ')))
        data = data.loc[data['text_length'] > 10]
    else:
        # 待修改
        print('不是最开始indeed数据集')
        data = pd.read_csv(data_path)

    train_data = pd.DataFrame(columns=data.columns)
    test_data = pd.DataFrame(columns=data.columns)
    dev_data = pd.DataFrame(columns=data.columns)
    counts = 0
    print('划分数据集 --------- \n ')
    for firm_info in tqdm(data.groupby('company_id')):
        # 对每个公司的评论随机打乱顺序，去前80%作为训练集，10%作为验证集，10%作为测试集
        firm_shuffle_data = firm_info[1].sample(frac=1).reset_index(drop=True)
        nums = firm_shuffle_data.shape[0]
        if nums >= min_count:
            train_data = pd.concat([train_data, firm_shuffle_data.loc[:int(train_rate * nums)]])
            dev_data = pd.concat(
                [dev_data, firm_shuffle_data.loc[int(train_rate * nums): int((train_rate + dev_rate) * nums)]]
            )
            test_data = pd.concat(
                [test_data, firm_shuffle_data.loc[int((train_rate + dev_rate) * nums):]]
            )
            counts += 1
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    dev_data = dev_data.sample(frac=1).reset_index(drop=True)
    print('评论数超过 {} 的公司一共有 {} 个'.format(min_count, counts))

    if not data_path:
        train_data.to_csv('Indeed/data/train.csv')
        test_data.to_csv('Indeed/data/test.csv')
        dev_data.to_csv('Indeed/data/dev.csv')
    else:
        save_path = '/'.join(data_path.split('/')[:-1])
        train_data.to_csv(save_path + '/data/train.csv')
        test_data.to_csv(save_path + '/data/test.csv')
        dev_data.to_csv(save_path + '/data/dev.csv')
    print('评论数一共有 {} 个'.format(int(train_data.shape[0]/0.8)))
    print('训练集测试集划分完成, 耗时 {}秒 ------- \n'.format(round(time.time() - t0, 2)))


def load_train_model(path='Indeed/saved_dict/TextCNN.ckpt'):
    model = torch.load(path)
    #print(model['embedding_individual.weight'].cpu().numpy())
    return model['embedding_individual.weight'].cpu().numpy()

def vec_sim(v1, v2):
    return np.dot(v1, v2) / np.sqrt(np.sum(v1 ** 2) * np.sum(v2 **2))

if __name__ == "__main__":
    data_source_path = 'Indeed/DL_indeed_review_with_miss_cons.csv'
    split_train_dev_test(data_path=data_source_path, dev_rate=0.1, test_rate=0.1, min_count=10)
    # file_test()
    # csv_test()
    # show 词典
    # vocab = vocab_show('Indeed/data/vocab_individual.pkl')
    # company_list = list(vocab.keys())
    # print(len(company_list))
    # print(company_list[10])
    # print(vocab['E2341'])
    # # print(vocab[])
    # for key in vocab:
    #     pass
    # print(vocab['E2363'])
    # pass

    # 测试模型参数
    # item_embedding = load_train_model()
    #
    # print(item_embedding.shape)
    # idx1 = vocab['E10233']
    # idx2 = vocab['E1017414']
    # v1, v2 = item_embedding[idx1], item_embedding[idx2]
    # print(company_list[idx1], company_list[idx2])
    # print(vec_sim(v1, v2))
    # print()
