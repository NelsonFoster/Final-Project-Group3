#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24 April 2019

@author: Darius Bailey | Nelson
DATS 6203-11
Machine Learning II

Final Project

ColorFERET


"""
#standard packages

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import seaborn as sb
import pandas as pd



#pyTorch Packages
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader


#other packages

import glob
import requests
import cv2
import xml
#import untangle
import json
import xml.etree.ElementTree as ET
import seaborn as sb
import os
import cv2
from skimage import io, transform
from PIL import Image

# --------------------------------------------------------------------------------------------


#modifying to run on GPU
dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")




# --------------------------------------------------------------------------------------------
#Loading Image Data

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


os.listdir('data/images')


class ImageDataSet(Dataset):

    def __init__(self, root='images', image_loader=None, transform=None):
        self.root = root
        self.image_files = [os.listdir(os.path.join(self.root, 'folder_{}'.format(i)) for i in range(1, 9))]
        self.loader = image_loader
        self.transform = transform
    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return sum([len(folder) for folder in self.image_files])

    def __getitem__(self, index):
        images = [self.loader(os.path.join(root, 'folder_{}'.format(i), self.image_files[i][index])) for i in range(1, 739)]
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        return images

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
#input_size = 784
#hidden_size = 500
#num_classes = 10
#num_epochs = 5
#batch_size = 100
#learning_rate = 0.001
# --------------------------------------------------------------------------------------------
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------

print(Dataset)



#Reading in ground truth files

tree = ET.parse('subjects.xml')
root = tree.getroot()

for child in root:
    print (child.tag, child.attrib)


for neighbor in root.iter('neighbor'):
    print (neighbor.attrib)

tree = ET.parse('recordings.xml')
root = tree.getroot()


# --------------------------------------------------------------------------------------------
#mapping ground truth files to image files

#with open('subjects.xml', 'r') as f:
#    image_to_name = xml.load(f)
#    
#print(len(image_to_name)) 
#print(image_to_name)


#convert xml to json

import xmljson




# --------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------











# --------------------------------------------------------------------------------------------