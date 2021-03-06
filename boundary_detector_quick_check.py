import numpy as np
import pandas as pd
import cv2
import math
import glob
import os
from keras.models import load_model
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

input_dim=64

# Convert time to frame number
def get_frame(t):
    return (t.hour*3600+t.minute*60+t.second)*30


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



model = load_model('model_test.h5', custom_objects={'contrastive_loss': contrastive_loss})


# Create pred vectors for boundary images for comparison
def create_boundary_vectors():
    images = []
    for image in glob.iglob('images/boundaries/*.png'):
        img = cv2.imread(image)
        img = cv2.resize(img, (input_dim,input_dim)).astype('float32')/255.0
        images.append(img)
    return model.predict(np.array(images))

pred_a = create_boundary_vectors()


# Return True if frame is a boundary
def is_boundary(pred_a, pred_b):
    pred_b = np.repeat(pred_b, len(pred_a), axis=0)
    dis = K.eval(euclidean_distance([pred_a, pred_b])).ravel()
    if np.any([dis < 0.05]):
        #print(np.min(dis))
        return True
    return False


# Detect and save boundary frames

def detect_boundary_frame(start, end, cap, fldr_name):
    i=max(1,start)
    if not os.path.exists(fldr_name):
        os.makedirs(fldr_name)
    while(i<=end):
        cap.set(1,i)
        success, img_full = cap.read()
        if(success):
            img_full = cv2.resize(img_full, (input_dim,input_dim))
            img = img_full.astype('float32')/255.0
            pred_b = model.predict(np.array(img).reshape(1, input_dim, input_dim, 3))
        if(is_boundary(pred_a, pred_b)):
            boundaries[-1].append(i)
            # save boundary frame
            cv2.imwrite(fldr_name+str(i)+'.png',img_full)
            print('boundary ', i)
        i=i+5


# Run through all the videos in "videos" folder and puts boundary images into "detected_boundaries/[video_name]/[show_name]/[begin/end]" folder.

df = pd.ExcelFile('sample_split_annotations.xlsx').parse('Sheet1')
columns = df.columns
start = df['Start']
finish = df['Finish']
show = df['Show']

boundaries = []
i=1
detection_fldr = 'detected_boundaries/'
if not os.path.exists(detection_fldr):
        os.mkdir(detection_fldr)


i=0
for video in glob.iglob('videos/*.mp4'):
    fldr_name = os.path.join(detection_fldr, video.split('/')[1])
    if not os.path.exists(fldr_name):
        os.mkdir(fldr_name)
    cap = cv2.VideoCapture(video)
    while(i<len(start) and not pd.isnull(start[i])):
        start_frame = get_frame(start[i])
        end_frame = get_frame(finish[i])
        detect_boundary_frame(start_frame-500, start_frame+500, cap, os.path.join(fldr_name, show[i], 'begin'))
        detect_boundary_frame(end_frame-500, end_frame+500, cap, os.path.join(fldr_name, show[i], 'end'))
        i+=1
        print("%d shows processed" % i)
    i+=1
    print("%d shows processed" % i)

