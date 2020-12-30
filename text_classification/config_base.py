import numpy as np
import torch

class Config_Base(object):

    """所有模型的基础配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.csv'                                # 训练集
        self.dev_path = dataset + '/data/dev.csv'                                    # 验证集
        self.test_path = dataset + '/data/test.csv'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.vocab_individual_path = dataset + '/data/vocab_individual.pkl'          # 固定效应的表
        self.vocab_cons_path = dataset + '/data/vocab_cons.pkl'                      # cons和pros的词典
        self.individual_col = 'company_id'                                           # 固定效应的列名

        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = None
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                             # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.n_vocab_individual = 0                                     # individual表大小，在运行时赋值
        self.n_vocab_cons = 0                                           # cons表大小，在运行时赋值
        self.num_epochs = 20                                           # epoch数
        self.batch_size = 128                                          # mini-batch大小
        self.pad_size = 80                                          # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 200           # 字向量维度
        self.embed_individual = 10                                      # individual embedding维度
        self.embed_cons = 200
