from torch.utils.data import Dataset
import random
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import math
import cv2
from torch.utils.data import DataLoader
from itertools import  product
from glob import glob
from pathlib import Path

class WhaleDataset(Dataset):
    def __init__(self):
        super(WhaleDataset, self).__init__()

        df_enc = pd.read_csv('encoding_org_img.csv', header=None)
        df_enc = df_enc.rename(columns={0:'Image'})
        self.df_enc = df_enc.set_index('Image')
        self.com_train = pd.read_csv('com_train.csv')
        self.balance_train()
        
    def get_encoding(self, img_name):
        encoding = self.df_enc.loc[img_name]
        return encoding.values

    def balance_train(self):
        positive_com = self.com_train[self.com_train.target == 1]
        negative_com = self.com_train[self.com_train.target == 0].sample(positive_com.shape[0])
        com_train = positive_com.append(negative_com).sample(frac=1)
        self.com_train = com_train.reset_index(drop=True)
        
    def __len__(self):
        return len(self.com_train)

    def __getitem__(self, index):
        enc = self.com_train.iloc[index]
        encoding = np.concatenate([self.get_encoding(enc.iloc[0]), self.get_encoding(enc.iloc[1])])
        label = enc.iloc[2]
        return encoding, label


class WhaleValidDataset(Dataset):
    def __init__(self):
        super(WhaleValidDataset, self).__init__()

        df_enc = pd.read_csv('encoding_org_img.csv', header=None)
        df_enc = df_enc.rename(columns={0:'Image'})
        self.df_enc = df_enc.set_index('Image')
        self.com_val = pd.read_csv('com_val.csv')
        self.balance_train()
        
    def get_encoding(self, img_name):
        encoding = self.df_enc.loc[img_name]
        return encoding.values

    def balance_train(self):
        positive_com = self.com_val[self.com_val.target == 1]
        negative_com = self.com_val[self.com_val.target == 0].sample(positive_com.shape[0])
        com_val = positive_com.append(negative_com).sample(frac=1)
        self.com_val = com_val.reset_index(drop=True)
        
    def __len__(self):
        return len(self.com_val)

    def __getitem__(self, index):
        enc = self.com_val.iloc[index]
        encoding = np.concatenate([self.get_encoding(enc.iloc[0]), self.get_encoding(enc.iloc[1])])
        label = enc.iloc[2]
        return encoding, label


class WhaleTestDataset(Dataset):
    def __init__(self):
        super(WhaleTestDataset, self).__init__()

        df_enc = pd.read_csv('encoding_org_img.csv', header=None)
        df_enc = df_enc.rename(columns={0:'Image'})
        self.df_enc = df_enc.set_index('Image')
        train = pd.read_csv('train_split_5.csv')
        val = pd.read_csv('valid_split_5.csv')
        all_df = train.append(val)
        test = [Path(e).stem+'.jpg' for e in glob('../WC_input/test/*.jpg')]
        self.com_test = list(product(test, all_df.Image.tolist()))        
        
    def get_encoding(self, img_name):
        encoding = self.df_enc.loc[img_name]
        return encoding.values
        
    def __len__(self):
        return len(self.com_test)

    def __getitem__(self, index):
        enc = self.com_test[index]
        encoding = np.concatenate([self.get_encoding(enc[0]), self.get_encoding(enc[1])])
        return encoding, comb