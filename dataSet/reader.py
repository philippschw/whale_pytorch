from torch.utils.data import Dataset
import random
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
import re
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
import torch
import ipdb
BASE_SIZE = 256

def img_name_extract_year(img):
    try:
        s = re.findall(r"\D(\d{8})\D", img)[0]
        return datetime.strptime(s, '%Y%m%d').year
    except:
        try:
            s = re.findall(r"\D(\d{9})\D", img)[0]
            return datetime.strptime(s, '%Y%m0%d').year
        except:
            return 2005

def img_name_extract_month(img):
    try:
        s = re.findall(r"\D(\d{8})\D", img)[0]
        return datetime.strptime(s, '%Y%m%d').month
    except:
        try:
            s = re.findall(r"\D(\d{9})\D", img)[0]
            return datetime.strptime(s, '%Y%m0%d').month
        except:
            return 7

def do_length_decode(rle, H=192, W=384, fill_value=255):
    mask = torch.zeros((H,W), np.uint8)
    if type(rle).__name__ == 'float': return mask
    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

def add_margin(x0,y0,x1,y1):
    crop_margin = 0.05
    dx = x1-x0
    dy = y1-y0
    x0 = x0-dx*crop_margin
    x1 = x1+dx*crop_margin+1
    y0 = y0-dy*crop_margin
    y1 = y1+dy*crop_margin+1
    if (x0<0): x0=0
    if (y0<0): y0=0
    return x0,y0,x1,y1

class WhaleDataset(Dataset):
    def __init__(self, names, labels=None, mode='train', transform_train=None,  min_num_classes=0):
        super(WhaleDataset, self).__init__()
        self.pairs = 2
        self.names = names
        self.labels = labels
        self.mode = mode
        self.transform_train = transform_train
        self.labels_dict = self.load_labels()
        self.bbox_dict = self.load_bbox()
        self.rle_masks = self.load_mask()
        self.get_year_encoder()
        self.get_month_encoder()
        self.year2enc = dict(zip(range(2005, 2019), self.binned_year.flatten()))
        self.get_month_encoder()
        self.id_labels = {Image:Id for Image, Id in zip(self.names, self.labels)}
        labels = []
        for label in self.labels:
            if label.find(' ') > -1:
                labels.append(label.split(' ')[0])
            else:
                labels.append(label)
        self.labels = labels

        if mode in ['train', 'valid']:
            self.dict_train = self.balance_train()
            # self.labels = list(self.dict_train.keys())
            self.labels = [k for k in self.dict_train.keys()
                            if len(self.dict_train[k]) >= min_num_classes]

    def get_year_encoder(self):
        self.binned_year = pd.cut(np.array(range(2005, 2019)), 5, labels=False).reshape(-1, 1)
        self.yearonehotencoder = OneHotEncoder(sparse=False)
        _ = self.yearonehotencoder.fit_transform(self.binned_year)

    def get_month_encoder(self):
        self.monthonehotencoder = OneHotEncoder(sparse=False)
        _ = self.monthonehotencoder.fit_transform(np.array(range(1,13)).reshape(-1, 1))

    def load_mask(self):
        print('loading mask...')
        rle_masks = pd.read_csv('./WC_input/model_50A_slim_ensemble.csv')
        rle_masks = rle_masks[rle_masks['rle_mask'].isnull() == False]
        rle_masks.index = rle_masks['id']
        del rle_masks['id']
        rle_masks = rle_masks.to_dict('index')
        return rle_masks

    def load_bbox(self):
        # Image,x0,y0,x1,y1
        print('loading bbox...')
        bbox = pd.read_csv('./WC_input/bboxs.csv')
        Images = bbox['Image'].tolist()
        x0s = bbox['x0'].tolist()
        y0s = bbox['y0'].tolist()
        x1s = bbox['x1'].tolist()
        y1s = bbox['y1'].tolist()
        bbox_dict = {}
        for Image,x0,y0,x1,y1 in zip(Images,x0s,y0s,x1s,y1s):
            x0,y0,x1,y1 = add_margin(x0,y0,x1,y1)
            bbox_dict[Image] = [x0, y0, x1, y1]     

        return bbox_dict

    def load_labels(self):
        dtypes = {'id': 'int', 'name': 'str'}
        label = pd.read_csv('./WC_input/label.csv', dtype=dtypes)
        labelName = label['name'].tolist()
        dict_label = {}
        id = 0
        for name in labelName:
            if name == '-1':
                dict_label[name] = 2233 * 2
                continue
            dict_label[name] = id
            id += 1
        return dict_label

    def balance_train(self):
        dict_train = {}
        for name, label in zip(self.names, self.labels):
            if not label in dict_train.keys():
                dict_train[label] = [name]
            else:
                dict_train[label].append(name)
        return dict_train
    def __len__(self):
        return len(self.labels)

    def get_image(self, name, transform, label, mode='train'):
        image = cv2.imread('./WC_input/{}/{}'.format(mode, name))

        if image is None:
            image = cv2.imread('./WC_input/test/{}'.format(name))
        try:
            mask = do_length_decode(self.rle_masks[name.split('.')[0]]['rle_mask'])
            mask = cv2.resize(mask, image.shape[:2][::-1])
        except:
            mask = cv2.imread('./WC_input/masks/' + name, cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = self.bbox_dict[name]
        if mask is None:
            mask = np.zeros_like(image[:,:,0])
        image = image[int(y0):int(y1), int(x0):int(x1)]
        mask = mask[int(y0):int(y1), int(x0):int(x1)]
        image, add_ = transform(image, mask, label)
        return image, add_

    def __getitem__(self, index):
        label = self.labels[index]
        names = self.dict_train[label]
        nums = len(names)
        if nums == 1:
            anchor_name = names[0]
            positive_name = names[0]
        else:
            anchor_name, positive_name = random.sample(names, 2)
        negative_label = random.choice(list(set(self.labels) ^ set([label, '-1'])))
        negative_name = random.choice(self.dict_train[negative_label])
        negative_label2 = '-1'
        negative_name2 = random.choice(self.dict_train[negative_label2])

        anchor_year = self.yearonehotencoder.transform(np.array([self.year2enc[img_name_extract_year(anchor_name)]]).reshape(-1, 1))
        positive_year = self.yearonehotencoder.transform(np.array([self.year2enc[img_name_extract_year(positive_name)]]).reshape(-1, 1))
        negative_year = self.yearonehotencoder.transform(np.array([self.year2enc[img_name_extract_year(negative_name)]]).reshape(-1, 1))
        negative_year2 = self.yearonehotencoder.transform(np.array([self.year2enc[img_name_extract_year(negative_name2)]]).reshape(-1, 1))

        anchor_month= self.monthonehotencoder.transform(np.array([img_name_extract_month(anchor_name)]).reshape(-1, 1))
        positive_month = self.monthonehotencoder.transform(np.array([img_name_extract_month(positive_name)]).reshape(-1, 1))
        negative_month = self.monthonehotencoder.transform(np.array([img_name_extract_month(negative_name)]).reshape(-1, 1))
        negative_month2 = self.monthonehotencoder.transform(np.array([img_name_extract_month(negative_name2)]).reshape(-1, 1))

        # anchor = torch.zeros([256, 512])
        # anchor[0, :17]=
        anchor = np.concatenate([np.array([self.labels_dict[label]]), anchor_year.flatten(), anchor_month.flatten()])
        # positive = torch.zeros([256, 512])
        positive= np.concatenate([np.array([self.labels_dict[label]]), positive_year.flatten(), positive_month.flatten()])
        # negative = torch.zeros([256, 512])
        negative = np.concatenate([np.array([self.labels_dict[negative_label]]), negative_year.flatten(), negative_month.flatten()])
        # negative2 = torch.zeros([256, 512])
        negative2= np.concatenate([np.array([self.labels_dict[negative_label2]]), negative_year2.flatten(), negative_month2.flatten()])

        anchor_image, anchor_add = self.get_image(anchor_name, self.transform_train, label)
        positive_image, positive_add = self.get_image(positive_name, self.transform_train, label)
        negative_image,  negative_add = self.get_image(negative_name, self.transform_train, negative_label)
        negative_image2, negative_add2 = self.get_image(negative_name2, self.transform_train, negative_label2)
        # ipdb.set_trace()
        assert anchor_name != negative_name
        # return [torch.cat((anchor_image, anchor.unsqueeze(0))),
        #         torch.cat((positive_image, positive.unsqueeze(0))),
        #         torch.cat((negative_image, negative.unsqueeze(0))),
        #         torch.cat((negative_image2, negative2.unsqueeze(0)))], \
        return [anchor_image,
                positive_image,
                negative_image,
                negative_image2],\
                [anchor,
                 positive,
                 negative, 
                 negative2]

class WhaleTestDataset(Dataset):
    def __init__(self, names, labels=None, mode='test',transform=None):
        super(WhaleTestDataset, self).__init__()
        self.names = names
        self.labels = labels
        self.mode = mode
        self.bbox_dict = self.load_bbox()
        self.labels_dict = self.load_labels()
        self.rle_masks = self.load_mask()
        self.get_year_encoder()
        self.get_month_encoder()
        self.year2enc = dict(zip(range(2005, 2019), self.binned_year.flatten()))
        self.get_month_encoder()
        self.transform = transform

    def get_year_encoder(self):
        self.binned_year = pd.cut(np.array(range(2005, 2019)), 5, labels=False).reshape(-1, 1)
        self.yearonehotencoder = OneHotEncoder(sparse=False)
        _ = self.yearonehotencoder.fit_transform(self.binned_year)

    def get_month_encoder(self):
        self.monthonehotencoder = OneHotEncoder(sparse=False)
        _ = self.monthonehotencoder.fit_transform(np.array(range(1,13)).reshape(-1, 1))
        
    def __len__(self):
        return len(self.names)

    def get_image(self, name, transform, mode='train'):
        image = cv2.imread('./WC_input/{}/{}'.format(mode, name))

        try:
            mask = do_length_decode(self.rle_masks[name.split('.')[0]]['rle_mask'])
            mask = cv2.resize(mask, image.shape[:2][::-1])
        except:
            mask = cv2.imread('./WC_input/masks/' + name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros_like(image[:, :, 0])
        x0, y0, x1, y1 = self.bbox_dict[name]
        x0, y0, x1, y1 = [e if e > 0 else 0 for e in self.bbox_dict[name]]
        image = image[int(y0):int(y1), int(x0):int(x1)]
        mask = mask[int(y0):int(y1), int(x0):int(x1)]
        image = transform(image, mask)
        return image

    def load_labels(self):
        dtypes = {'id': 'int', 'name': 'str'}
        label = pd.read_csv('./WC_input/label.csv', dtype=dtypes)
        labelName = label['name'].tolist()
        dict_label = {}
        id = 0
        for name in labelName:
            if name == '-1':
                dict_label[name] = 2233 * 2
                continue
            dict_label[name] = id
            id += 1
        return dict_label

    def load_mask(self):
        rle_masks = pd.read_csv('./WC_input/model_50A_slim_ensemble.csv')
        rle_masks = rle_masks[rle_masks['rle_mask'].isnull() == False]
        rle_masks.index = rle_masks['id']
        del rle_masks['id']
        rle_masks = rle_masks.to_dict('index')
        return rle_masks

    def load_bbox(self):
        print('loading bbox...')
        bbox = pd.read_csv('./WC_input/bboxs.csv')
        Images = bbox['Image'].tolist()
        x0s = bbox['x0'].tolist()
        y0s = bbox['y0'].tolist()
        x1s = bbox['x1'].tolist()
        y1s = bbox['y1'].tolist()
        bbox_dict = {}
        for Image, x0, y0, x1, y1 in zip(Images, x0s, y0s, x1s, y1s):
            x0,y0,x1,y1 = add_margin(x0,y0,x1,y1)
            bbox_dict[Image] = [x0, y0, x1, y1]
        return bbox_dict
    
    def __getitem__(self, index):
        if self.mode in ['test', 'data']:
            name = self.names[index]
            anchor_year = self.yearonehotencoder.transform(np.array([self.year2enc[img_name_extract_year(name)]]).reshape(-1, 1))
            anchor_month= self.monthonehotencoder.transform(np.array([img_name_extract_month(name)]).reshape(-1, 1))
            label = np.concatenate([np.array([0]), anchor_year.flatten(), anchor_month.flatten()])
            image = self.get_image(name, self.transform, mode=self.mode)
            # image = [torch.cat((img1, anchor.unsqueeze(0))), torch.cat((img2, anchor.unsqueeze(0)))]
            return image, label, name
        elif self.mode in ['valid', 'train']:
            name = self.names[index]
            anchor_year = self.yearonehotencoder.transform(np.array([self.year2enc[img_name_extract_year(name)]]).reshape(-1, 1))
            anchor_month= self.monthonehotencoder.transform(np.array([img_name_extract_month(name)]).reshape(-1, 1))
            label = np.concatenate([np.array([self.labels_dict[self.labels[index]]]), anchor_year.flatten(), anchor_month.flatten()])
            image = self.get_image(name, self.transform)
            # image = [torch.cat((img1, anchor.unsqueeze(0))), torch.cat((img2, anchor.unsqueeze(0)))]
            return image, label, name
