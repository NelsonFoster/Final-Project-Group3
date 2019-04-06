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

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import glob
from PIL import Image
from torch.utils.data import Dataset
import requests

# --------------------------------------------------------------------------------------------

#modifying to run on GPU
dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")


# --------------------------------------------------------------------------------------------


