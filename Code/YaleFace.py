#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24 April 2019

@author: Darius Bailey | Nelson
DATS 6203-11
Machine Learning II

Final Project

YaleFace


"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


# --------------------------------------------------------------------------------------------

#modifying to run on GPU
dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------




