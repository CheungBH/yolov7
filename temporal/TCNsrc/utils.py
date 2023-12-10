#!/usr/bin/env python
# coding: utf-8
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader



class TCNLoader(data.Dataset):
    def __init__(self, datas, labels, maxlabel):
        self.datas = datas  # [N, 30*34]
        self.labels = labels  # [N, 1]
        self.max = maxlabel
        self.vector = [0] * self.max

    def get_one_hot_num(self, label):
        # self.vector[label] = 1
        # one_hot_vector = self.vector
        # self.vector = [0] * self.max
        # return torch.LongTensor(one_hot_vector)
        return torch.tensor(label)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        keypoint = np.asarray(self.datas[idx]).astype(np.float32)  # [30*34]#.reshape(-1,1,17,2)
        keypoint = torch.from_numpy(keypoint)
        one_hot_label = self.get_one_hot_num(self.labels[idx])
        return (keypoint, one_hot_label)

    
if __name__ == '__main__':
    data = get_data(data_pth)
    #write_txt(data, output_pth)
    dataset = TCNData(data)
    #print(dataset.get_labels)
    dataloader = DataLoader(dataset)
    dataiter = iter(dataloader)
    data, label = dataiter.next()
    print(data.shape)
    #print(label.data.max(1,keepdim=True)[1])
