# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config_base import Config_Base


class Config(Config_Base):
    """配置参数"""
    def __init__(self, dataset, embedding):
        super().__init__(dataset, embedding)
        self.model_name = 'TextRNN_Att'                              # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name

        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64
        self.hidden_size_cons = 32

'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''
'''Bi-GRU'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out


class Model_cons(nn.Module):
    def __init__(self, config):
        super(Model_cons, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.gru = nn.GRU(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)

        self.embedding_cons = nn.Embedding(config.n_vocab_cons, config.embed_cons, padding_idx=config.n_vocab_cons - 1)
        # self.fc_cons = nn.Linear(config.embed_cons, config.cons_hidden_size)
        self.embedding_pros = nn.Embedding(config.n_vocab_cons, config.embed_cons, padding_idx=config.n_vocab_cons - 1)

        self.gru_cons = nn.GRU(config.embed_cons, config.hidden_size_cons, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc_cons = nn.Linear(config.hidden_size_cons * config.num_layers, config.hidden_size_cons)

        self.gru_pros = nn.GRU(config.embed_cons, config.hidden_size_cons, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc_pros = nn.Linear(config.hidden_size_cons * config.num_layers, config.hidden_size_cons)

        self.fc = nn.Linear(config.hidden_size2 + config.hidden_size_cons * 2, config.num_classes)


    def forward(self, x):
        # x, _ = x
        emb = self.embedding(x[0])  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.gru(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)

        emb_cons = self.embedding_cons(x[2])
        emb_pros = self.embedding_pros(x[3])
        gru_cons, _ = self.gru_cons(emb_cons)
        cons_out = self.fc_cons(gru_cons[:, -1, :])  # 句子最后时刻的 hidden state
        gru_pros, _ = self.gru_pros(emb_pros)
        pros_out = self.fc_pros(gru_cons[:, -1, :])  # 句子最后时刻的 hidden state

        out = torch.cat([out, cons_out, pros_out],
                        1)

        out = self.fc(out)  # [128, 64 + 32 + 32]
        return out
