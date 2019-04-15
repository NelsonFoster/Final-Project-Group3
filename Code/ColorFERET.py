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

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import seaborn as sb
import pandas as pd



#pyTorch Packages
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models



#other packages

import glob
from PIL import Image
import requests
import cv2
import xml
import untangle
import json

# --------------------------------------------------------------------------------------------

#modifying to run on GPU
dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")


# --------------------------------------------------------------------------------------------


