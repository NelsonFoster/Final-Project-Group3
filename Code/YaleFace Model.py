#!/usr/bin/env python
# coding: utf-8

# In[26]:


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
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from collections import Counter
from prepare_dataset import FaceDataSet
from facelayers import FaceModule, FaceModule2, FaceModule3, FaceModule4
import random 
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# In[5]:


CUDA = torch.cuda.is_available()

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(7)

face_files = glob.glob('yalefaces/*.*')


# In[8]:

#haarcascadefiles
#https://github.com/opencv/opencv/tree/master/data/haarcascades

#Darius
#cascadePath = r"/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"

#Nelson
cascadePath = 'haarcascade_frontalface_default.xml'

#harcascade helps to crop the faces from the images

faceCascade = cv2.CascadeClassifier(cascadePath)


# In[69]:


def get_images_and_labels(image_paths):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    #     image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = os.path.split(image_path)[1].split(".")[1] #.replace("subject", "")
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            timage = cv2.resize(image[y: y + h, x: x + w], (128, 128)).reshape((1,128,128))
            images.append(timage/255.0)
            labels.append(nbr)
    # return the images list and labels list
    return images, labels

fac_exp = ['glasses', 'happy', 'leftlight', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink', 'centerlight','noglasses']
images, labels = get_images_and_labels(face_files)

#Conver the labels to indexes (a numeric value)
labels = [fac_exp.index(label)  for label in labels]


# In[70]:


#Splitting train test set
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=.2)

#Creating the pytorch train and test loader 
train_loader = torch.utils.data.DataLoader(FaceDataSet(X_train, y_train), batch_size=16)
test_loader = torch.utils.data.DataLoader(FaceDataSet(X_test, y_test), len(X_test), shuffle=False)


# In[71]:


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
plt.show()

print(' '.join('%5s' % fac_exp[labels[j]] for j in range(8)))


# In[72]:


# The 4 Networks with different number of layers each defined in facelayers.py
# net = FaceModule()
# net = FaceModule2()
# net = FaceModule3()
net = FaceModule4()

if CUDA:
    net = net.cuda()

# Definind the criterion and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# In[73]:


# 10 epoch for comparing the 4 networks
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        
        if CUDA:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

print('Finished Training')


# In[67]:


print('Now Evaluating')
net.eval()
test_inputs, labels = next(iter(test_loader))
if CUDA:
    test_inputs = test_inputs.cuda()
    labels = labels.cuda()
output = net(test_inputs)
pred = output.data.max(1, keepdim=True)[1]
correct = pred.eq(labels.data.view_as(pred)).cpu().sum()
print("Accuracy = {}%".format(100. * correct / len(test_loader.dataset)))

y_pred = pred.cpu().numpy().flatten()
y_actual = labels.cpu().numpy().reshape((33,1))

y_probs = torch.nn.functional.softmax(output, dim=1)

## Plot ROC AUC Curve
skplt.metrics.plot_roc(y_actual, y_probs.cpu().detach().numpy())
plt.show()


# In[68]:


skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)


# In[82]:


net.load_state_dict(torch.load('model.pkl'))
outputs = net(image)
print(outputs.cpu().detach().numpy())
_, predicted = torch.max(outputs.data, 1)

print("Actual target = {}".format(label))
print("Predicted target = {}".format(predicted))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




