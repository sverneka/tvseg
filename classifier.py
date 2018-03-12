import numpy as np
seed = 1
np.random.seed(seed)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import cv2
import math
import glob
import os

# frames are resized to 64x64

input_dim = 64


train_X = np.load('train_X_64_binary.npy')
train_Y = np.load('train_Y_64_binary.npy')

train_Y = np.transpose(np.array([train_Y, 1-train_Y]))
val_split = 0.2
val_split = int((1-val_split)*train_X.shape[0])

val_X = train_X[val_split:,]
val_Y = train_Y[val_split:,]

train_X = train_X[:val_split,]
train_Y = train_Y[:val_split,]

arr = np.arange(train_X.shape[0])
np.random.shuffle(arr)
train_X = train_X[arr,]
train_Y = train_Y[arr,]


def build_model():
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3,3), border_mode='same',input_shape=(input_dim,input_dim,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model




# train
model = build_model()

nb_epoch = 5
adma = Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])



gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)

train_generator = gen.flow(train_X, train_Y, batch_size=64)

model.fit_generator(train_generator, class_weight={1:1.0, 0:0.00005}, epochs=nb_epoch, shuffle = True,  validation_data=(val_X, val_Y))
