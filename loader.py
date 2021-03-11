import random

import torch
import torch.nn as nn
import numpy as np

import utils

class JTESLoader:
    def __init__(self, male_li, female_li, utt_li, emo_li, feat_path, feature_dim, emo_label, batch_size=40):
        self.male_li = male_li
        self.female_li = female_li
        self.utt_li = utt_li
        self.emo_li = emo_li
        self.feat_path = feat_path
        self.feature_dim = feature_dim
        self.emo_label = emo_label
        self.batch_size = batch_size
        self.data = []
        self.emos = []
        self.idx = 0
        self.ord = []
        self.sz = 0
    
    def load(self, mean=None, std=None):
        """
        load and z-score normalization
        """
        print("- loading")
        print(self.male_li)
        print(self.female_li)
        print(self.utt_li)

        for p in self.male_li:
            for u in self.utt_li:
                for e in self.emo_li:
                    name = self.feat_path + "/M-" + e + "-" + p[1:3] + "-" + u + ".feat"
                    x = np.fromfile(name, np.float64)
                    x = x.reshape(-1, self.feature_dim).transpose((1, 0))
                    self.data.append(x)
                    self.emos.append(self.emo_label[e])

        for p in self.female_li:
            for u in self.utt_li:
                for e in self.emo_li:
                    name = self.feat_path + "/F-" + e + "-" + p[1:3] + "-" + u + ".feat"
                    x = np.fromfile(name, np.float64)
                    x = x.reshape(-1, self.feature_dim).transpose((1, 0)) # (feature, len)
                    self.data.append(x)
                    self.emos.append(self.emo_label[e])
        self.sz = len(self.emos)
        self.ord = list(range(self.sz))

        if mean is None:
            tmp = np.concatenate(self.data, axis=1)
            mean = np.mean(tmp, axis=1, keepdims=True)
            std = np.std(tmp, axis=1, keepdims=True)

        for i in range(self.sz):
            self.data[i] = (self.data[i] - mean) / (std + 1e-8)
        print("- done")
        return mean, std

    def shuffle(self):
        """
        shuffle indices
        """
        self.idx = 0
        self.ord = list(range(self.sz))
        random.shuffle(self.ord)

    def return_batch(self):
        """
        ret : (data, mask, label) or None
        """
        if self.idx >= self.sz:
            return None

        len_b = []
        data_b = []
        label_b = []

        sz_b = min(self.batch_size, self.sz-self.idx)
        idx_b = self.ord[self.idx:self.idx+sz_b]
        self.idx += sz_b

        for i in idx_b:
            data_b.append(torch.from_numpy(self.data[i].transpose((1, 0))).float()) # append (len, feature)
            len_b.append(self.data[i].shape[1])
            label_b.append(self.emos[i])
        
        mask = self.make_mask(len_b)
        data_b = nn.utils.rnn.pad_sequence(data_b, batch_first=True)
        label_b = torch.LongTensor(label_b)
        return data_b, mask, label_b

    def make_mask(self, length):
        max_length = max(length)
        mask = np.zeros((len(length), max_length))
        for i in range(len(length)):
            mask[i][0:length[i]] = 1.0
        mask = torch.from_numpy(mask).float()
        return mask