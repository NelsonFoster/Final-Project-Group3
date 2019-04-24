#!/usr/bin/env python
# coding: utf-8

# In[1]:

##############See Line


# In[ ]:
#code adapted from https://github.com/sar-gupta/convisualize_nb/blob/master/cnn-visualize.ipynb

#preprocessing image
images=glob.glob('./yalefaces/subject04.happy')
for image in images:
    img = Image.open(image)
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    plt.imshow(trans(trans1(img)))
    plt.show()


# In[38]:


#Visualization of model convolutional layers
model = FaceModule4()
if CUDA:
    model = model.cuda()


# In[42]:


model.eval()


# In[44]:


#preprocessing image
images=glob.glob('./yalefaces/subject04.happy')
for image in images:
    img = Image.open(image)
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    plt.imshow(trans(trans1(img)))
    plt.show()


# In[67]:


###Visualizing what the convnet learned

import keras
import os, shutil
from keras import layers, models
img_path = './yalefaces/subject04.happy'
from keras.preprocessing import image
import numpy as np


# In[ ]:





# In[72]:


img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /=255
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()


# In[119]:


print(model_weights)


# In[154]:


kernels = model.conv1.conv1.weight.detach()
fig, axarr = plt.subplots(kernels.size(0))
for idx in range(kernels.size(0)):
    axarr[idx].imshow(kernels[idx].squeeze(0))


# In[222]:


#adapting code from https://github.com/sar-gupta/convisualize_nb/blob/master/cnn-visualize.ipynb


# In[267]:


def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image


# In[304]:





# In[271]:


def normalize(image):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])
    image = Variable(preprocess(image).unsqueeze(0).cuda())
    return image


def predict(image):
    _, index = vgg(image).data[0].max(0)
    return str(index[0]), labels[str(index[0])][1]
    
def deprocess(image):
    return image * torch.Tensor([0.229, 0.224, 0.225]).cuda()  + torch.Tensor([0.485, 0.456, 0.406]).cuda()

def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image


# In[305]:


face_1 = load_image('./yalefaces/subject04.happy')


# In[316]:


images=glob.glob('./yalefaces/subject04.happy')
for image in images:
    img = Image.open(image)
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    face_2 = (trans(trans1(face_1)))
    plt.imshow(trans(trans1(img)))
    plt.show()


# In[323]:


trans3 = transforms.ToTensor()
face_3 = trans3(face_1)


# In[306]:


from torchvision import models
vgg = models.vgg16(pretrained=True)


# In[307]:


modulelist = list(vgg.features.modules())  #model_weights


# In[324]:


def layer_outputs(image):
    outputs = []
    names = []
    for layer in modulelist[1:]:
        image = layer(face_3)
        outputs.append(image)
        names.append(str(layer))
        
    output_im = []
    for i in outputs:
        i = i.squeeze(0)
        temp = to_grayscale(i)
        output_im.append(temp.data.cpu().numpy())
        
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (30, 50)


    for i in range(len(output_im)):
        a = fig.add_subplot(8,4,i+1)
        imgplot = plt.imshow(output_im[i])
        plt.axis('off')
        a.set_title(names[i].partition('(')[0], fontsize=30)

    plt.savefig('layer_outputs.jpg', bbox_inches='tight')


# In[311]:


layer_outputs(face_2)


# In[ ]:





# In[190]:





# In[ ]:




