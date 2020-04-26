import numpy as np
import pandas as pd

seed = 66
np.random.seed(seed)
import cv2
import json
import glob
import os
from tqdm import *
from shutil  import copyfile, rmtree 
from pathlib import Path

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, SpatialDropout2D, Input
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

import tensorflow.keras as keras
import segmentation_models as sm

from tqdm import tqdm_notebook, tqdm

from albumentations import (
    Compose
)

preprocess_input = sm.get_preprocessing('seresnet34')

class TestDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, batch_size=32,
                 dim=(299,299),  shuffle=True, 
                 preprocess_input=preprocess_input, 
                 aug=Compose([]), min_mask=2 ):
        'Initialization'
        self.X = X
        self.batch_size = batch_size
        self.n_classes = 1
        self.shuffle = shuffle
        self.preprocess_input = preprocess_input
        self.aug = aug
        self.dim = dim
        self.on_epoch_end()

    def set_aug(self, aug):
        self.aug = aug
        self.on_epoch_end()
      
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.X) / self.batch_size) / 1) )

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        end_index = min((index+1)*self.batch_size, len(self.indexes))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs

        # Generate data
        xx = self.__data_generation(indexes)

        return xx

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        batch_size = len(indexes)
        
        # Initialization
        XX = np.empty((batch_size, self.dim[1], self.dim[0], 3), dtype='float32')

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            img = self.X[ID]
            if img.shape[0] != self.dim[0]:
                img = cv2.resize(img, self.dim, cv2.INTER_CUBIC)
            
            
            # Store class
            augmented = self.aug(image=img)
            aug_img = augmented['image']
            if aug_img.shape[1] != self.dim[1]:
                aug_img = cv2.resize(aug_img, self.dim, cv2.INTER_CUBIC)
            XX[i,] = aug_img.astype('float32')
    
       
        XX = self.preprocess_input(XX)

        return XX