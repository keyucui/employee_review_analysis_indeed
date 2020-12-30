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
        self.model_name = 'TextCNN'
        self.filter_sizes = (1, 2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 128                                         # 卷积核数量(channels数)
        self.individual_hidden_size = 64
        self.num_filters_cons = 128
'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class Model_individual(nn.Module):
    def __init__(self, config):
        super(Model_individual, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.embedding_individual = nn.Embedding(config.n_vocab_individual, config.embed_individual, padding_idx=config.n_vocab_individual - 1)
        self.fc_individual = nn.Linear(config.embed_individual, config.individual_hidden_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes) + config.individual_hidden_size,
                            config.num_classes)

        #self.fc_final = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        individaul = self.embedding_individual(x[-1])
        individaul = individaul.squeeze(1)
        individaul = self.fc_individual(individaul)
        individaul = torch.relu(individaul)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs] + [individaul], 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

class Model_cons(nn.Module):
    def __init__(self, config):
        super(Model_cons, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.embedding_cons = nn.Embedding(config.n_vocab_cons, config.embed_cons, padding_idx=config.n_vocab_cons - 1)
        # self.fc_cons = nn.Linear(config.embed_cons, config.cons_hidden_size)
        self.embedding_pros = nn.Embedding(config.n_vocab_cons, config.embed_cons, padding_idx=config.n_vocab_cons - 1)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])

        self.convs_cons = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters_cons, (k, config.embed_cons)) for k in config.filter_sizes])

        self.convs_pros = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters_cons, (k, config.embed_cons)) for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes) +
                            2 * config.num_filters_cons * len(config.filter_sizes),
                            config.num_classes)

        #self.fc_final = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def conv_and_pool_cons(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def conv_and_pool_pros(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x :  [embedding, seq_len, cons_embedding, pros_embedding]
        # print(len(x))
        out = self.embedding(x[0])
        out = out.unsqueeze(1)

        out_cons = self.embedding_cons(x[2])
        out_cons = out_cons.unsqueeze(1)

        out_pros = self.embedding_pros(x[3])
        out_pros = out_pros.unsqueeze(1)

        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs] +
                        [self.conv_and_pool_cons(out_cons, conv) for conv in self.convs_cons] +
                        [self.conv_and_pool_pros(out_pros, conv) for conv in self.convs_pros],
                        1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

