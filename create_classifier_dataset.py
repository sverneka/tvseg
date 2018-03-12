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

def create_boundary_data():
    images=[]
    labels = []
    for image in glob.iglob('images/boundaries/*.png'):
        img = cv2.imread(image)
        img = cv2.resize(img, (input_dim,input_dim)).astype('float32')/255.0
        images.append(img)
        labels.append(1)
    return np.array(images), np.array(labels)



def create_non_boundary_data():
    shows = []
    labels = []
    images = []
    count  = 0
    for fldr in glob.iglob('images/*'):
        if(fldr == 'images/boundaries'):
            print fldr
            continue
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
    print("randomly picking images")
    for i in xrange(0,50000):
        x = random.randint(0, len(shows)-1)
        y = random.randint(0, len(shows[x])-1)
        images.append(shows[x][y])
        labels.append(0)
        if(i%1000==0):
            print i
    return np.array(images), np.array(labels)


def create_dataset():
    boundary_data, labels_boundary = create_boundary_data()
    non_boundary_data, labels_non_boundary = create_non_boundary_data()
    return np.concatenate((boundary_data, non_boundary_data), axis=0), np.concatenate((labels_boundary, labels_non_boundary))


train_X, train_Y = create_dataset()
np.save('train_X_64_binary.npy', train_X)
np.save('train_Y_64_binary.npy', train_Y)
