
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader


class Loader(data.Dataset):
    def __init__(self, datas, labels, maxlabel):
        self.data = datas
        self.labels = labels
        self.max = maxlabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoint = np.asarray(self.data[idx]).astype(np.float32)
        keypoint = torch.from_numpy(keypoint)
        label = self.labels[idx]
        label = torch.tensor(label)
        return (keypoint, label)


if __name__ == '__main__':
    datas = [[0 ] * 30 * 34 for _ in range(5)]
    labels = [1 ,1 ,0 ,0 ,0]
    dataset = BiLstmLoader(datas)
    dataloader = DataLoader(dataset)
    dataiter = iter(dataloader)
    data, label = dataiter.next()
    print(data.shape)


