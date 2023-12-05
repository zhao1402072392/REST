import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.cifar10_dataset import Cifar10CDatasetForSparse
from data.bffhq_dataset import BffhqDatasetForSparse
from data.cmnist_dataset import CmnistDatasetForSparse

################
### SETTINGS ###
################

confounder_settings = {
    'cmnist':{
        'constructor': CmnistDatasetForSparse
    },
    'bffhq':{
        'constructor': BffhqDatasetForSparse
    },
    'cifar10c':{
        'constructor': Cifar10CDatasetForSparse
    },
    'CelebA':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    }
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(       
        splits,
        train_frac=args.fraction,
        subsample_to_minority=args.subsample_to_minority)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str) \
        for split in splits]
    return dro_subsets

def prepare_confounder_data_cifar10c(args, split, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        args.root_dir,
        name=args.dataset,
        split=split,
        transform=None,
        conflict_pct=args.conflict_pct)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    # if train:
    #     splits = ['train', 'val', 'test']
    # else:
    #     splits = ['test']
    # subsets = full_dataset.get_splits(
    #     splits,
    #     train_frac=args.fraction,
    #     subsample_to_minority=args.subsample_to_minority)
    # dro_subsets = [
    #     DRODataset(
    #         subsets[split],
    #         process_item_fn=None,
    #         n_groups=2,
    #         n_classes=10,
    #         group_str_fn=full_dataset.group_str) \
    #     for split in splits]
    tmp = DRODataset(
        full_dataset,
        process_item_fn=None,
        n_groups=full_dataset.n_groups,
        n_classes=full_dataset.n_classes,
        group_str_fn=full_dataset.group_str)

    return tmp