#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24 April 2019

@author: Darius Bailey | Nelson Foster
DATS 6203-11
Machine Learning II

Final Project

ColorFERET using Keras 

Primary research & reference credit: Francois Chollet's Deep Learning with Python,
Chapter 5 - "Deep Learning for Computer Vision"

"""


#load packages

import keras
import os, shutil

from keras import layers
from keras import models


#Setting intiial directories

original_dataset_dir = 'images'

base_dir = 'images_small'
os.mkdir(base_dir)

#train/test/split directories

train_dir   = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)


# --------------------------------------------------------------------------------------------


#From the colorFERET documentation:
#There are 13 different poses. (The orientation "right" means
#facing the photographer's right.)
#	fa	regular frontal image
#	fb	alternative frontal image, taken shortly after the
#			corresponding fa image
#	pl	profile left
#	hl	half left - head turned about 67.5 degrees left
#	ql	quarter left - head turned about 22.5 degrees left
#	pr	profile right
#	hr	half right - head turned about 67.5 degrees right
#	qr	quarter right - head turned about 22.5 degrees right
#	ra	random image - head turned about 45 degree left
#	rb	random image - head turned about 15 degree left
#	rc	random image - head turned about 15 degree right
#	rd	random image - head turned about 45 degree right
#	re	random image - head turned about 75 degree right

#for this exercise, we are focusing on the following orientations: frontal, 
#profile left, profile right, half left, quarter left, half right, and quarter right. 
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
#Creating directories for train, test and validation datasets
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
#Training 
# --------------------------------------------------------------------------------------------

#frontal image directory

train_front_dir = os.path.join(train_dir, 'front')
os.mkdir(train_front_dir)

#profile left image directory

train_left_dir = os.path.join(train_dir, 'left')
os.mkdir(train_left_dir)

#profile right image directory

train_right_dir = os.path.join(train_dir, 'right')
os.mkdir(train_right_dir)

#profile half-left image directory

train_halfleft_dir = os.path.join(train_dir, 'halfleft')
os.mkdir(train_halfleft_dir)

#profile half-right image directory

train_halfright_dir = os.path.join(train_dir, 'halfright')
os.mkdir(train_halfright_dir)

#profile quarter-left image directory

train_quarterleft_dir = os.path.join(train_dir, 'quarterleft')
os.mkdir(train_quarterleft_dir)

#profile quarter-right mage directory

train_quarterright_dir = os.path.join(train_dir, 'quarterright')
os.mkdir(train_quarterright_dir)

# --------------------------------------------------------------------------------------------
#Validation
# --------------------------------------------------------------------------------------------

#frontal image directory

validation_front_dir = os.path.join(validation_dir, 'front')
os.mkdir(validation_front_dir)

#profile left image directory

validation_left_dir = os.path.join(validation_dir, 'left')
os.mkdir(validation_left_dir)

#profile right image directory

validation_right_dir = os.path.join(validation_dir, 'right')
os.mkdir(validation_right_dir)

#profile half-left image directory

validation_halfleft_dir = os.path.join(validation_dir, 'halfleft')
os.mkdir(validation_halfleft_dir)

#profile half-right image directory

validation_halfright_dir = os.path.join(validation_dir, 'halfright')
os.mkdir(validation_halfright_dir)

#profile quarter-left image directory

validation_quarterleft_dir = os.path.join(validation_dir, 'quarterleft')
os.mkdir(validation_quarterleft_dir)

#profile quarter-right mage directory

validation_quarterright_dir = os.path.join(validation_dir, 'quarterright')
os.mkdir(validation_quarterright_dir)

# --------------------------------------------------------------------------------------------
#Test
# --------------------------------------------------------------------------------------------


test_front_dir = os.path.join(test_dir, 'front')
os.mkdir(test_front_dir)

#profile left image directory

test_left_dir = os.path.join(test_dir, 'left')
os.mkdir(test_left_dir)

#profile right image directory

test_right_dir = os.path.join(test_dir, 'right')
os.mkdir(test_right_dir)

#profile half-left image directory

test_halfleft_dir = os.path.join(test_dir, 'halfleft')
os.mkdir(test_halfleft_dir)

#profile half-right image directory

test_halfright_dir = os.path.join(test_dir, 'halfright')
os.mkdir(test_halfright_dir)

#profile quarter-left image directory

test_quarterleft_dir = os.path.join(test_dir, 'quarterleft')
os.mkdir(test_quarterleft_dir)

#profile quarter-right mage directory

test_quarterright_dir = os.path.join(test_dir, 'quarterright')
os.mkdir(test_quarterright_dir)

# --------------------------------------------------------------------------------------------
#copying files into the respective train, validation and test directories
# --------------------------------------------------------------------------------------------
##front
# --------------------------------------------------------------------------------------------
#train
fnames = ['{}_fa.jpg'.format(i) for i in range (1000)] 
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_front_dir, fname)
	shutil.copyfile(src, dst)

#validate
fnames = ['{}_fa.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_front_dir, fname)
	shutil.copyfile(src, dst)
#test
fnames = ['{}_fa.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_front_dir, fname)
	shutil.copyfile(src, dst)

# --------------------------------------------------------------------------------------------
##left
# --------------------------------------------------------------------------------------------
#train
fnames = ['{}_pl.jpg'.format(i) for i in range (1000)] 
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_left_dir, fname)
	shutil.copyfile(src, dst)

#validate
fnames = ['{}_pl.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_left_dir, fname)
	shutil.copyfile(src, dst)

#test
fnames = ['{}_pl.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_left_dir, fname)
	shutil.copyfile(src, dst)
# --------------------------------------------------------------------------------------------
##right
# --------------------------------------------------------------------------------------------    
#train
fnames = ['{}_pr.jpg'.format(i) for i in range (1000)] 
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_right_dir, fname)
	shutil.copyfile(src, dst)
    
    
#validate
fnames = ['{}_pr.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_right_dir, fname)
	shutil.copyfile(src, dst)


#test
fnames = ['{}_pr.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_right_dir, fname)
	shutil.copyfile(src, dst)
 
# --------------------------------------------------------------------------------------------
##half-left
# --------------------------------------------------------------------------------------------   
#train
fnames = ['{}_hl.jpg'.format(i) for i in range (1000)]
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_halfleft_dir, fname)
	shutil.copyfile(src, dst)   
    
    
    
#validate
fnames = ['{}_hl.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_halfleft_dir, fname)
	shutil.copyfile(src, dst)



#test
fnames = ['{}_hl.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_halfleft_dir, fname)
	shutil.copyfile(src, dst)
# --------------------------------------------------------------------------------------------   
##half-right
# --------------------------------------------------------------------------------------------
#train
fnames = ['{}_hr.jpg'.format(i) for i in range (1000)]
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_halfright_dir, fname)
	shutil.copyfile(src, dst)
    
    
    
#validate
fnames = ['{}_hr.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_halfright_dir, fname)
	shutil.copyfile(src, dst)



#test
fnames = ['{}_hr.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_halfright_dir, fname)
	shutil.copyfile(src, dst)
# --------------------------------------------------------------------------------------------
##quarter-left
# --------------------------------------------------------------------------------------------  
#train
fnames = ['{}_ql.jpg'.format(i) for i in range (1000)]
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_quarterleft_dir, fname)
	shutil.copyfile(src, dst)    
    
    
    
#validate
fnames = ['{}_ql.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_quarterleft_dir, fname)
	shutil.copyfile(src, dst)



#test
fnames = ['{}_ql.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_quarterleft_dir, fname)
	shutil.copyfile(src, dst)    
    
# --------------------------------------------------------------------------------------------
##quarter-right
# --------------------------------------------------------------------------------------------   
#train
fnames = ['{}_qr.jpg'.format(i) for i in range (1000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_quarterright_dir, fname)
	shutil.copyfile(src, dst)    
    
    
    
#validate
fnames = ['{}_qr.jpg'.format(i) for i in range (1000, 1500)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_quarterright_dir, fname)
	shutil.copyfile(src, dst)



#test
fnames = ['{}_qr.jpg'.format(i) for i in range (1500, 2000)] # set to correct range
for fnames in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_quarterright_dir, fname)
	shutil.copyfile(src, dst)

# --------------------------------------------------------------------------------------------
###Verify directories copies successfully
# --------------------------------------------------------------------------------------------
#training
# --------------------------------------------------------------------------------------------
print('total training frontal images:', len(os.listdir(train_front_dir)))
print('total training left images:', len(os.listdir(train_left_dir)))
print('total training right images:', len(os.listdir(train_right_dir)))
print('total training halfleft images:', len(os.listdir(train_halfleft_dir)))
print('total training halfright images:', len(os.listdir(train_halfright_dir)))
print('total training quarterleft images:', len(os.listdir(train_quarterleft_dir)))
print('total training quarterright images:', len(os.listdir(train_quarterright_dir)))
# --------------------------------------------------------------------------------------------
#validation
# --------------------------------------------------------------------------------------------
print('total validation frontal images:', len(os.listdir(tvalidation_front_dir)))
print('total validation left images:', len(os.listdir(validation_left_dir)))
print('total validation right images:', len(os.listdir(validation_right_dir)))
print('total validation halfleft images:', len(os.listdir(validation_halfleft_dir)))
print('total validation halfright images:', len(os.listdir(validation_halfright_dir)))
print('total validation quarterleft images:', len(os.listdir(validation_quarterleft_dir)))
print('total validation quarterright images:', len(os.listdir(validation_quarterright_dir)))
# --------------------------------------------------------------------------------------------
#test
# --------------------------------------------------------------------------------------------
print('total test frontal images:', len(os.listdir(test_front_dir)))
print('total test left images:', len(os.listdir(test_left_dir)))
print('total test right images:', len(os.listdir(test_right_dir)))
print('total test halfleft images:', len(os.listdir(test_halfleft_dir)))
print('total test halfright images:', len(os.listdir(test_halfright_dir)))
print('total test quarterleft images:', len(os.listdir(test_quarterleft_dir)))
print('total test quarterright images:', len(os.listdir(test_quarterright_dir)))



# --------------------------------------------------------------------------------------------
#instantiating a convnet
# --------------------------------------------------------------------------------------------
from keras import layers
from keras import models


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3,), activation='relu',
				     input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# --------------------------------------------------------------------------------------------
#examining feature map dimensions
# --------------------------------------------------------------------------------------------
model.summary()

# --------------------------------------------------------------------------------------------
###configuring the model for training
# --------------------------------------------------------------------------------------------
from keras import optimizers

model.compile(loss = 'binary_crossentropy',
		      optimizer=optimizers.RMSprop(lr=le-4),
		      metrics=['acc'])

# --------------------------------------------------------------------------------------------
###preprocessing with ImageDataGenerator
# --------------------------------------------------------------------------------------------
from keras.processing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, 
	    target_size(150, 150),
	    batch_size=20,
        class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
	    validation_dir, 
	    target_size(150, 150),
	    batch_size=20,
        class_mode='binary')

# --------------------------------------------------------------------------------------------
#review output of data generator
# --------------------------------------------------------------------------------------------
for data_batch, labels_batch in train_generator:
     print('data batch shape:', data_batch.shape)
     print('labels batch shape:', labels_batch.shape)
     break

# --------------------------------------------------------------------------------------------
#fitting the model using batch generator
# --------------------------------------------------------------------------------------------
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps = 50)
# --------------------------------------------------------------------------------------------
#saving model
# --------------------------------------------------------------------------------------------
model.save('images_small_1.h5')

# --------------------------------------------------------------------------------------------
#display model performance
# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

acc = history.history[ 'acc' ]
val_acc = history.history[ 'val_acc' ]
loss = history.history[ 'loss' ]
val_loss = history.history[ 'val_loss' ]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

plt.show()
# --------------------------------------------------------------------------------------------
#using data augmentation to mitigate overfitting
# --------------------------------------------------------------------------------------------
datagen = ImageDataGenerator(
	rotation_range=40,
         width_shift_range=0.2,
         height_Shift_range=0.2,
         shear_range=0.2,
         zoom_range=0.2,
         horizontal_flip=True,
         fill_mode= 'nearest')


# --------------------------------------------------------------------------------------------
#display randomly augmented training images
# --------------------------------------------------------------------------------------------
from keras.preprocessing import image

fnames = [os.path.join(train_front_dir, fname) for
         fname in os.listdir(train_front_dir)]

img_path = fnames[00043_931230_fa]  #verify

img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)

x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
     plt.figure(i)
     img.plot = plt.imshow(image.array_to_img(batch[0]))
     i += 1
     if += 1
     if i % 4 == 0
     break



plt.show()
# --------------------------------------------------------------------------------------------
#Defininng a new convnet with dropout to further mitigate overfitting
# --------------------------------------------------------------------------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3,) activation= 'relu',
				     input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten(())
model.add(layers.Dropout((0.5))
model.add(layers.Dense(512, activation ='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss = ‘binary_crossentropy’,
		      optimizer=optimizers.RMSprop(lr=le-4),
		      metrics=[‘acc’])
# --------------------------------------------------------------------------------------------
###training the convnet using data augmentation generators
# --------------------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(
         rescale=1./255,
	rotation_range=40,
         width_shift_range=0.2,
         height_Shift_range=0.2,
         shear_range=0.2,
         zoom_range=0.2,
         horizontal_flip=True, )


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
             train_dir,
             target_size(150, 150),
             batch_size=32,
             class_mode=‘binary’)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,  
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50

model.save('images_small_2.h5')


# --------------------------------------------------------------------------------------------
###using a pretrained network for feature extraction - VGG16
# --------------------------------------------------------------------------------------------

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(150, 150, 3))
# --------------------------------------------------------------------------------------------
#examine structure of VGG
# --------------------------------------------------------------------------------------------
conv_base.summary()


# -------------------------------------------------------------------------------------------- 
#feature extraction without data augmentation
# --------------------------------------------------------------------------------------------
import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

base_dir = 'images'
train_dir = os.path.join(base_dir, 'train')
validation_dir  = os.path.base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
     features = np.zeros(shape=(sample_count, 4, 4, 512))
     labels = np.zeros(shape=(sample_count))
     generator = datagen.flow_from_directory(
             directory,
             target_size=(150, 150),
             batch_size=batch_size,
             class_mode=‘binary’)

    i = 0
    for inputs_batch, labels_batch in generator:
    features_batch  = conv_base.predict(inputs_batch)
    features[i * batch_size : (i + 1) * batch_size] = features_batch
    labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >=  sample_count
         break
    return 

return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features =  np.reshape(test_features, (1000, 4 * 4 * 512))

# --------------------------------------------------------------------------------------------
#define and training the densely connected classifier
# --------------------------------------------------------------------------------------------
from keras import models
from keras import layers
from keras import optimizers

model = models.sequential()
model.add(layers.dense(256, activation='relu', imput_dim= 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                        loss='binary_crossentropy',
                        metrics=['acc'])


history = model.fit(train_features, train_labels,
                             epochs=30
                             batch_size=20
                             validation_data=(validation_features, validation_labels))


# --------------------------------------------------------------------------------------------
#plot the results
# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b',  label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(eppchs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# --------------------------------------------------------------------------------------------
###feature extraction with Data augmentation (FOR GPU ONLY)
# --------------------------------------------------------------------------------------------
#Comment out if not on GPU
# --------------------------------------------------------------------------------------------
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base_
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
# --------------------------------------------------------------------------------------------
#examine model
# --------------------------------------------------------------------------------------------
model.summary()

# --------------------------------------------------------------------------------------------
#implement freezing convolutional base to preserve previously learned representations.
# --------------------------------------------------------------------------------------------
print('This is the number of trainable weights'  
          'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of treingable weights' 
         'after freezing the conv base:' len(model.trainable_weights))

# --------------------------------------------------------------------------------------------
#training the model end to end with a frozen convolutional base
# --------------------------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator (
          rescale=1./255,
          rotation_range=40,  
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=02,
          horizontal_flip=True
          fill_mode=‘nearest’)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
         train_dir,
         target_size=(150, 150),
         batch_size=20,
         class_mode=‘binary’

validation_generator = test_dategen_flow_from_directory(
              validation_dir,
              target_size=(150, 150),
              batch_size=20
              class_mode=‘binary’)

model.compile(loss='binary_crossentropy',
                        optimizer=optimizersRMSprop(lr=2e-5),
                        metrics=[‘acc’])

history = model.fit_generator(
          train_generator,
          steps_per_epoch=100
          epochs=30
          validation_data=validation_generator
          validation_steps=50) 
# --------------------------------------------------------------------------------------------
#plot the resulds
# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(eppchs, val_loss, 'bo', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# --------------------------------------------------------------------------------------------
###fine tuning
# --------------------------------------------------------------------------------------------
#taking another look at the convolutional base
# --------------------------------------------------------------------------------------------
conv_base.summary()
# --------------------------------------------------------------------------------------------
#freezing all layers up to a secific one
# --------------------------------------------------------------------------------------------
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers: 
     if layer.name == 'block5_conv1':
        set_trainable = True
     else
        layer.trainable = False

# --------------------------------------------------------------------------------------------
# fine tuning the model
# --------------------------------------------------------------------------------------------
model.compile(loss='binary_crossentropy',
                       optimizer=optimizers.RMSrop(lr=le-5),
                       metrics=[‘acc’])

history = model.fit_generator (
         train_generator,
         steps_per_epoch=100
         epochs=100,
         validation_data=validation_generator,
         validation_steps=50)

# --------------------------------------------------------------------------------------------
#plot the resulds
# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history[‘val_loss’]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b',  label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(eppchs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# --------------------------------------------------------------------------------------------
#smoothing the plots
# --------------------------------------------------------------------------------------------def smooth_curve(points, factor=0.8):
     smoothed_points = [ ]
     for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))

        else:
          smoothed_points.append(point)
     return  smoothed_points


plt.plot(epochs,  
            smooth_curve(acc), 'bo', label=‘Smoothed training acc’)
plt.title(’Training and validation accuracy’)
plt.legend()

plt.figure()

plt.plot(epochs,
           smooth_curve(loss), 'bo', label=‘Smoothed training loss’
plt.plot(epochs, 
           smooth_curve(val_loss), 'b', label='Smoothed validations loss')
plt.title(‘Training and validation loss’)
plt.legend
plt.show()

# --------------------------------------------------------------------------------------------
###Visualizing what the convnet learned
# --------------------------------------------------------------------------------------------

from keras.models import load_model
model = load_model(‘image_small_2.h5’)
model.summary()
# --------------------------------------------------------------------------------------------
#preprocessing an image
# --------------------------------------------------------------------------------------------
img_path =  'images/00043_931230_fa.jpg'  #find an image

from keras.preprocessing import image
import numpy as np
image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matlotlib.pyplot as plt
plt.imshow(img_tesnor[0])
plt.show()
# --------------------------------------------------------------------------------------------
### instantiating a model from an imput tensor and list of output tensors
# --------------------------------------------------------------------------------------------
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs
# --------------------------------------------------------------------------------------------
#run model in predict mode
# --------------------------------------------------------------------------------------------activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)
# --------------------------------------------------------------------------------------------
#visualizizing fourth channel
# --------------------------------------------------------------------------------------------
plt. matshow(first_layer_activation[0, :, :, 4], cmap = 'viridis')
# --------------------------------------------------------------------------------------------
#visualizing seventh channel
# --------------------------------------------------------------------------------------------
plt.matshow(first_layer_activation[0, :, :, 7], cmap = 'viridis')
# --------------------------------------------------------------------------------------------
#visualizing every channel in every intermediate activation
# --------------------------------------------------------------------------------------------
layer_names = [ ] 
for layers in model.layers[:8]:
     n_features = layer_activation.shape[-1]
     
     size = layer_activation.shape[1]
     

     n_cols = n_ features // images_per_row
     dispay_grid = np.zeros((size * n_cols, images_per_row * size))

     for col in range(n_cols):
          for row in range(images_per_row):
               channel_image = layer_activation[0, 
                                                :, :, 
                                                col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128 
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col   * size : (col + 1) * size,
                                    row *  size : (row + 1) * size] = channel_image

       scale = 1. /size
       plt.figure(figsize=(scale * display_grid.shape[1],
                                    scale * display_grid.shape[0]))

        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect=‘auto,’ cmap='viridis')

# --------------------------------------------------------------------------------------------


                                             
                




































# --------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------











# --------------------------------------------------------------------------------------------