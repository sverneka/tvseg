# Create dataset to train siamese network

import numpy as np
seed = 1
np.random.seed(seed)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout

import numpy as np
import pandas as pd
import cv2
import math
import glob
import os

input_dim = 64

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)



def create_similar_image_data():
    count = 0
    img0=[]
    img1=[]
    labels = []
    for image in glob.iglob('images/*/*.png'):
        img = cv2.imread(image)
        img = cv2.resize(img, (input_dim,input_dim)).astype('float32')/255.0
        for i in xrange(0,5):
            img0.append(img)
            img1.append(datagen.random_transform(img, seed=seed))
            labels.append(1)
            count += 1
            if(count%1000==0):
                print(count)
    return np.array([img0, img1]),  np.array(labels)  



def create_dissimilar_image_data():
    img0 = []
    img1 = []
    shows = []
    labels = []
    count  = 0
    for fldr in glob.iglob('images/*'):
        shows.append([])
        print "folder is: "+fldr
        for image in glob.iglob(fldr+'/*.png'):
            #print(image)
            img = cv2.imread(image)
            img = cv2.resize(img, (input_dim,input_dim)).astype('float32')/255.0
            shows[-1].append(img)
            count += 1
            if(count%1000 == 0):
                print count
    print "show size ", len(shows)
    print("creating_pairs")
    for i in xrange(0,50000):
        x1 = random.randint(0, len(shows)-1)
        x2 = random.randint(0, len(shows)-1)
        y1 = random.randint(0, len(shows[x1])-1)
        y2 = random.randint(0, len(shows[x2])-1)
        img0.append(shows[x1][y1])
        img1.append(datagen.random_transform(shows[x2][y2], seed=seed))
        labels.append(0)
        if(i%1000==0):
            print i
    return np.array([img0, img1]), np.array(labels)



def create_pairs():
    similar_pairs, labels_similar = create_similar_image_data()
    dissimilar_pairs, labels_dissimilar = create_dissimilar_image_data()
    return np.concatenate((similar_pairs, dissimilar_pairs), axis=1), np.concatenate((labels_similar, labels_dissimilar))
    #return dissimilar_pairs, labels_dissimilar

# Note it needs around 12GB memory to create 112490 samples
train_X, train_Y = create_pairs()
np.save('train_X.npy', train_X)
np.save('train_Y.npy', train_Y)

