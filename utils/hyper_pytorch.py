import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    def __init__(self, dataset, transform):
        #self.data = dataset[0].astype(np.float32)
        self.data = dataset[0]
        self.transform = transform
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels



