# coding: UTF-8
import time
import torch
from gensim.models import Word2Vec
import numpy as np
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Indeed Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Indeed'  # 数据集
    # fix_effect = True
    fix_effect = False
    cons = True
    # cons = False
    embedding = 'word2vec'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()

    print("Loading data...")
    if not fix_effect and cons:
        print('training with cons and pros......')
        vocab, train_data, dev_data, test_data, vocab_cons = build_dataset(config, fix_effect=False, cons=cons)
        config.n_vocab_cons = len(vocab_cons)
    elif not fix_effect and not cons:
        print('without individual fix effect or cons......')
        vocab, train_data, dev_data, test_data = build_dataset(config, fix_effect=False, cons=cons)                 # 赋值cons词典长度
    else:
        print('with individual fix effect')
        vocab, train_data, dev_data, test_data, vocab_individual = build_dataset(config, fix_effect=True)
        config.n_vocab_individual = len(vocab_individual)

    # 加载pre-train词向量
    if embedding != 'random':
        print('使用pre-train词向量...........')
        count_unk = 0
        word2vec = Word2Vec.load('Indeed/embedding/word2vec_text.model')
        embedding_matrix = np.zeros((len(vocab), word2vec.vector_size))
        config.embed = word2vec.vector_size
        for word in vocab:
            if word in word2vec.wv.vocab:
                embedding_matrix[vocab[word]] = word2vec.wv.get_vector(word)
            else:
                count_unk += 1
        print('有{}个词不在pre-train词向量中'.format(count_unk))
        config.embedding_pretrained = torch.tensor(embedding_matrix.astype('float32'))

    train_iter = build_iterator(train_data, config, fix_effect, cons)
    dev_iter = build_iterator(dev_data, config, fix_effect, cons)
    test_iter = build_iterator(test_data, config, fix_effect, cons)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    if not fix_effect and not cons:
        model = x.Model(config).to(config.device)
    elif not fix_effect and cons:
        model = x.Model_cons(config).to(config.device)
    else:
        model = x.Model_individual(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)

    print('model config : ++++++++++++')
    print(config.batch_size, config.learning_rate, config.dropout)
    print('+'*20)
    print(model.parameters)

    train(config, model, train_iter, dev_iter, test_iter)
