# coding:utf-8
from torch.utils import data
import os
import torch
import nltk
import numpy as np
from gensim.models import KeyedVectors


class IMDB_Data(data.DataLoader):
    def __init__(self, data_name, min_count, word2id=None, max_sentence_length=100, batch_size=64, is_pretrain=False):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.data_name = "/imdb/" + data_name
        self.min_count = min_count
        self.word2id = word2id
        self.max_sentence_length = max_sentence_length
        self.batch_size = batch_size
        self.datas, self.labels = self.load_data()
        if is_pretrain:
            self.get_word2vec()
        else:
            self.weight = None
        for i in range(len(self.datas)):
            self.data[i] = np.array(self.datas[i])

    def load_data(self):
        datas = open(self.path + self.data_name, encoding="utf-8")
        datas = [data.split("		")[-1].split() + [data.split("		")[2]] for data in datas]
        datas = sorted(datas, key = lambda x: len(x))