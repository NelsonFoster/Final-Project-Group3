import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch import optim
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from collections import Counter

CUDA = torch.cuda.is_available()

# This function was taken from https://github.com/apsdehal/Face-Recognition/blob/master/model.py
# It was used with the debugger to help determine appropriate size
def get_convnet_output_size(network, input_size):
    input_size = input_size

    if type(network) != list:
        network = [network]

    in_channels = network[0].conv1.in_channels

    output = Variable(torch.ones(1, in_channels, input_size, input_size))
    output.require_grad = False
    for conv in network:
        output = conv.forward(output)

    return np.asscalar(np.prod(output.data.shape)), output.data.size()[2]

# Add convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, max_pool_stride=2,
                 dropout_ratio=0.5):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size)
        self.max_pool2d = nn.MaxPool2d(max_pool_stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_ratio)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.max_pool2d(x)
        return self.dropout(self.max_pool2d(x))


#Original CNN
class FaceModule(nn.Module):
    """Some Information about FaceModule"""
    def __init__(self):
        super(FaceModule, self).__init__()
        self.conv1 = ConvLayer(1, 16, kernel_size=5)
        conv_output_size, _ = get_convnet_output_size(self.conv1, 128)
        self.fully_connected1 = nn.Linear(conv_output_size, 1024)
        self.fully_connected2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected1(x)
        x = nn.functional.log_softmax(self.fully_connected2(x))
        return x


# 2 Convolutional Layers
class FaceModule2(nn.Module):
    """Some Information about FaceModule"""
    def __init__(self):
        super(FaceModule2, self).__init__()
        self.convs = []
        self.conv1 = ConvLayer(1, 32, kernel_size=3)
        self.conv2 = ConvLayer(32, 64, kernel_size=3)
        self.fully_connected1 = nn.Linear(57600, 1024)
        self.fully_connected2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected1(x)
        x = nn.functional.log_softmax(self.fully_connected2(x))
        return x


# 2 Convolutional Layers + Batch Normalization
class FaceModule3(nn.Module):
    """Some Information about FaceModule"""
    def __init__(self):
        super(FaceModule3, self).__init__()
        self.convs = []
        self.conv1 = ConvLayer(1, 32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3)
        self.fully_connected1 = nn.Linear(57600, 1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.fully_connected2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected1(x)
        x = self.batchnorm2(x)
        x = nn.functional.log_softmax(self.fully_connected2(x))
        return x



# 3 Convolutional Layerrs + Batch Normalization 
class FaceModule4(nn.Module):
    """Some Information about FaceModule"""
    def __init__(self):
        super(FaceModule4, self).__init__()
        self.conv1 = ConvLayer(1, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = ConvLayer(16, 32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = ConvLayer(32, 64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fully_connected1 = nn.Linear(12544, 1024)
        self.batchnorm4 = nn.BatchNorm1d(1024)
        self.fully_connected2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected1(x)
        x = self.batchnorm4(x)
        x = nn.functional.log_softmax(self.fully_connected2(x))
        return x
