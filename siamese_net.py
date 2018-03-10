# siamese network modified from: https://github.com/NVIDIA/keras/blob/master/examples/mnist_siamese_graph.py

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


train_X = np.load('train_X_64.npy')
train_Y = np.load('train_Y_64.npy')
arr = np.arange(train_X.shape[1])
np.random.shuffle(arr)
train_X = train_X[:,arr,:,:,:]
train_Y = train_Y[arr]

val_split = 0.2
val_split = int((1-val_split)*train_X.shape[1])

val_X = train_X[:,val_split:,:,:,:]
val_Y = train_Y[val_split:]

train_X = train_X[:,0:val_split,:,:,:]
train_Y = train_Y[:val_split]

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Conv2D(8, kernel_size = (3,3), border_mode='same',input_shape=(input_dim,input_dim,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.summary()
    return model


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.where(labels == np.array([ int(i<0.5) for i in predictions]))[0].shape[0]*1.0/len(labels)


nb_epoch = 5


# network definition
base_network = create_base_network()

input_a = Input(shape=(input_dim,input_dim,3))
input_b = Input(shape=(input_dim,input_dim,3))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

#auxilary model to extract features

model_fe = Model(input_a, processed_a)

# train
rms = RMSprop(lr=0.01)
model.compile(loss=contrastive_loss, optimizer=rms)



model.fit([train_X[0,],train_X[1,]], train_Y, validation_data=([val_X[0,], val_X[1,]], val_Y), batch_size = 128, shuffle = True, epochs=nb_epoch)


# compute final accuracy on training and test sets
pred = model.predict([train_X[0,],train_X[1,]])
tr_acc = compute_accuracy(pred, train_Y)

pred = model.predict([val_X[0,],val_X[1,]])
val_acc = compute_accuracy(pred, val_Y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * val_acc))

# save model

model_fe.save('model_test.h5')

