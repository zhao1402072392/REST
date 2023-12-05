import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.Cifar10CDataset import Cifar10CDataset
from glob import glob


class BffhqDatasetForSparse(Cifar10CDataset):
    def __init__(self, root, name='bffhq', split='train', transform=None, conflict_pct=5):
        # super(Cifar10CDatasetForSparse, self).__init__()
        self.name = name
        self.transform = transform
        self.root = root
        if conflict_pct >= 1:
            conflict_pct = int(conflict_pct)
        self.conflict_token = f'{conflict_pct}pct'
        self.split = split

        if split == 'train':
            self.header_dir = os.path.join(root, self.conflict_token)
            # print('header_dir', root, self.name, self.header_dir)
            # root: data / bffhq
            # name: bffhq
            # header_dir: data / bffhq / 0.5pct
            self.align = glob(os.path.join(self.header_dir, 'align', "*", "*"))
            self.conflict = glob(os.path.join(self.header_dir, 'conflict', "*", "*"))
            self.data = self.align + self.conflict
        elif split == 'valid':
            self.header_dir = os.path.join(root, self.conflict_token)
            self.data = glob(os.path.join(self.root, 'valid', "*"))
        elif split == 'test':
            self.data = glob(os.path.join(self.root, 'test', "*"))

        train_target_attr = []
        attr_names = []
        for data in self.data:
            # fname = os.path.relpath(data, self.header_dir)
            fname = data
            # print('fname',data, self.header_dir, fname)
            train_target_attr.append(int(fname.split('_')[-2]))
            attr_names.append(int(fname.split('_')[-1].split('.')[0]))

        self.y_array = torch.LongTensor(train_target_attr)
        self.attr_names = torch.LongTensor(attr_names)

        # TODO make shape compatiable
        self.group_array = (self.y_array == self.attr_names).int()
        self.transform = get_transform_bffhq(split)
        self.n_groups = 2
        self.n_classes = 2

# rewrite getitem
    def __getitem__(self, idx):
        # img_filename = os.path.join(
        #     self.data_dir,
        #     self.filename_array[idx])
        img_filename = self.data[idx]
        # print(img_filename)
        img = Image.open(img_filename).convert('RGB')
        # print('img', img)
        img =self.transform(img)
        y = self.y_array[idx]
        g = self.group_array[idx]
        return img, y, g

    def group_str(self, group_idx):
        # y = group_idx // (self.n_groups/self.n_classes)
        # c = group_idx % (self.n_groups//self.n_classes)

        # group_name = f'{self.target_name} = {int(y)}'
        # bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        # for attr_idx, attr_name in enumerate(self.confounder_names):
        #     group_name += f', {attr_name} = {bin_str[attr_idx]}'
        if group_idx ==1:
            group_name = 'align'
        elif group_idx == 0:
            group_name = 'conflict'
        return group_name


def get_transform_bffhq(train):
    # orig_w = 178
    # orig_h = 218
    # orig_min_dim = min(orig_w, orig_h)
    # if model_attributes[model_type]['target_resolution'] is not None:
    #     target_resolution = model_attributes[model_type]['target_resolution']
    # else:
    #     target_resolution = (orig_w, orig_h)

    if not train:
        transform = transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
    return transform


