import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import pandas as pd


class Cifar10CDataset(Dataset):
    def __init__(self, root, name='cmnist', split='train', transform=None, conflict_pct=5):
        super(Cifar10CDataset, self).__init__()
        self.name = name
        self.transform = transform
        self.root = root
        if conflict_pct >= 1:
            conflict_pct = int(conflict_pct)
        self.conflict_token = f'{conflict_pct}pct'

        if split == 'train':
            self.header_dir = os.path.join(root, self.conflict_token)
            self.align = glob(os.path.join(self.header_dir, 'align', "*", "*"))
            self.conflict = glob(os.path.join(self.header_dir, 'conflict', "*", "*"))

            self.data = self.align + self.conflict

            train_target_attr = []
            for data in self.data:
                fname = os.path.relpath(data, self.header_dir)
                train_target_attr.append(int(fname.split('_')[-2]))
            self.y_array = torch.LongTensor(train_target_attr)

        elif split == 'valid':
            self.header_dir = os.path.join(root, self.conflict_token)
            self.data = glob(os.path.join(self.header_dir, 'valid', "*", "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, 'test', "*", "*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index] # attr=(class_label, bias_label)