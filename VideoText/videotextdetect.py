import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

import os
import glob
import sys
sys.path.append("crnn.pytorch")

import torch
from torch.autograd import Variable
import dataset
import models.crnn as crnn

score_threshold = 0.8

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata())[:,0:3].reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def decode_box_coordinates(image, file_boxes):
    width, height = image.size
    new_boxes = []
    for box in file_boxes:
        #print("box" + str(box))
        ymin = box[0] * height
        xmin = box[1] * width
        ymax = box[2] * height
        xmax = box[3] * width
        new_boxes.append((xmin,ymin,xmax,ymax))
    return new_boxes


def get_tf_session():
    # Path to frozen detection graph. This is the model retrained on text data
    PATH_TO_CKPT = './weights/frozen_inference_graph.pb'
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    detection_graph.as_default()
    sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    fetches = [detection_boxes, detection_scores, detection_classes, num_detections]
    return sess, fetches, image_tensor


def get_detections(image, sess, fetches, image_tensor):
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (file_boxes, file_scores, file_classes, file_num) = sess.run(fetches, feed_dict={image_tensor: image_np_expanded})
    top_box_indices = file_scores[0] > score_threshold
    file_boxes = file_boxes[0][top_box_indices]
    new_file_boxes = decode_box_coordinates(image, file_boxes)
    #file_scores = file_scores[0][top_box_indices]
    return new_file_boxes

if __name__ == '__main__':
    fldr_name = '../VidTextExtraction'
    if not os.path.exists(fldr_name):
        os.mkdir(fldr_name)
    sess, fetches, image_tensor = get_tf_session()
    for video in glob.iglob('../videos/*.mp4'):
        detections = []
        file_name = os.path.join(fldr_name, video.split('/')[-1].split('.')[0]+'.p')
        cap = cv2.VideoCapture(video)
        i = 1
        cap.set(1,i)
        success, image = cap.read()
        while(success):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            boxes = get_detections(image, sess, fetches, image_tensor)
            if(len(boxes)>0):
                detections.append([i,boxes])
            i = i+10
            cap.set(1,i)
            success, image = cap.read()
            if((i-1)%1000==0):
                print("%d frames processed" % i)
                #break
        pickle.dump(detections, open(file_name,"wb"))
