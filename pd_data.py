import torch
import numpy as np
import os
import pandas as pd

import cv2
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import WeightedRandomSampler, random_split



def load_data(batch_size, num_workers, pic_path, csv_path):
    """
    Loading dataset.
    """

    pd_dataset = pdData(pic_path, csv_path)
    train_dataset, test_dataset = random_split(pd_dataset, lengths=[int(len(pd_dataset)*0.8), len(pd_dataset)-int(len(pd_dataset)*0.8)])
    print(len(train_dataset), len(test_dataset))
    train_sampler = WeightedRandomSampler(weights=np.ones(len(train_dataset)), num_samples=50000)
    test_sampler = WeightedRandomSampler(weights=np.ones(len(test_dataset)), num_samples=10000)
    # pd_dataloader = DataLoader(
    #     pd_dataset,
    #     batch_size=batch_size,
    #     shuffle = True,
    #     pin_memory=True,
    #     num_workers=num_workers,
    # )
    train_dataloader = DataLoader(
        pd_dataset,
        batch_size=batch_size,
        shuffle = False,
        pin_memory=True,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    test_dataloader = DataLoader(
        pd_dataset,
        batch_size=batch_size,
        shuffle = False,
        pin_memory=True,
        num_workers=num_workers,
        sampler=test_sampler,
    )
    print(len(train_dataset), len(test_dataset))
    print("data load success")
    return train_dataloader, test_dataloader


class pdData(Dataset):
    """
    pdData dataset.
    """

    def __init__(self, pic_path, csv_path, eval=False):
        
        self.pic_path = pic_path
        self.beforeSenDir = []
        self.afterSenDir = []
        self.totDir = []
        
        csv_data_path = os.path.join(csv_path,'data.csv')
        df = pd.read_csv(csv_data_path)
        self.beforeSenDir = list(df['record_Before'])
        self.afterSenDir = list(df['record_After'])
        self.labels = list(df['Label'])
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        
        img_before_path = self.beforeSenDir[index]
        img_after_path = self.afterSenDir[index]

        img_before = cv2.imread(os.path.join(self.pic_path, img_before_path))
        img_after = cv2.imread(os.path.join(self.pic_path, img_after_path))

        img_before = torch.from_numpy(img_before.transpose((2,0,1))).type(torch.FloatTensor)
        img_after = torch.from_numpy(img_after.transpose((2,0,1))).type(torch.FloatTensor)
        
        label = self.labels[index]
        return img_before, img_after, index, label

    def __len__(self):
        return len(self.afterSenDir)
