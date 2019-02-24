#!/usr/bin/env python
# coding: utf-8

# # Trail Cam Animal Classification

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import keras
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications import VGG16


# In[2]:


deer_dir = "./deer"
nodeer_dir = "./nodeer"
target_size = 224
img_list = []
label = []

for img_name in os.listdir(deer_dir):
    img = load_img(deer_dir + "/" + img_name)
    img = img.resize((target_size,target_size))
    img_array = img_to_array(img)
    img_list.append(img_array)
    label.append(1)

for img_name in os.listdir(nodeer_dir):
    img = load_img(nodeer_dir + "/" + img_name)
    img = img.resize((target_size,target_size))
    img_array = img_to_array(img)
    img_list.append(img_array)
    label.append(0)


# In[4]:


img_array, img_labels = shuffle(img_list,label)


# In[5]:


img_array = np.array(img_array)
img_labels = np.array(img_labels)


# In[6]:


original_img_array = np.copy(img_array)


# In[7]:


img_array[:][:][:][0] -= np.mean(img_array[:][:][:][0], axis = 0)
img_array[:][:][:][1] -= np.mean(img_array[:][:][:][1], axis = 0)
img_array[:][:][:][2] -= np.mean(img_array[:][:][:][2], axis = 0)


# In[8]:


img_array[:][:][:][0] -= np.std(img_array[:][:][:][0], axis = 0)
img_array[:][:][:][1] -= np.std(img_array[:][:][:][1], axis = 0)
img_array[:][:][:][2] -= np.std(img_array[:][:][:][2], axis = 0)


# In[9]:


train_percent = .5
train_count = int(train_percent*len(img_array))
test_count = len(img_array) - train_count

train_labels = img_labels[0:train_count]
test_labels = img_labels[train_count:]

train_images = img_array[0:train_count]
test_images = img_array[train_count:]


# In[10]:


num_category = 2
# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_category)
test_labels = keras.utils.to_categorical(test_labels, num_category)


# In[ ]:





# In[11]:


#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=["accuracy"])


# In[ ]:





# In[44]:


test_images.shape


# In[14]:


vgg_conv = VGG16(weights = 'imagenet',
                include_top=True,
                input_shape=(224,224,3))
x = Dense(2, activation='sigmoid', name='predictions')(vgg_conv.layers[-2].output)
my_model = keras.Model(inputs=vgg_conv.input, outputs=x)
my_model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=["accuracy"])


# In[ ]:


#conv_model.fit(train_images,train_labels,epochs=5)
batch_size = 10
num_epoch = 10
#model training
model_log = my_model.fit(train_images,train_labels,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(test_images,test_labels))


# In[ ]:




