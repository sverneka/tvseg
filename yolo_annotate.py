# Create video annotations using YOLO from https://github.com/marvis/pytorch-yolo2.git on videos
# and write annotations in "yolo_annotations/[video_file_name].txt" file
# in the format, [frame_no, label, box_co-ordinates{4 numbers}, prediction_confidence]

import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import cv2
import glob
import os

def detect(m, img, class_names, log_file, frame):
    use_cuda = 1
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    for box in boxes:
        log_file.write(str(frame)+','+class_names[box[-1]]+','+",".join(map(str, box[0:4]))+','+str(box[5])+'\n')
        log_file.flush()


if __name__ == '__main__':
    cfgfile = 'cfg/yolo.cfg'
    weightfile = 'yolo.weights'
    fldr_name = '../yolo_annotations'
    if not os.path.exists(fldr_name):
        os.mkdir(fldr_name)
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    namesfile = 'data/coco.names'
    class_names = load_class_names(namesfile)
    m.cuda()
    for video in glob.iglob('../videos/*.mp4'):
        file_name = os.path.join(fldr_name, video.split('/')[-1].split('.')[0]+'.txt')
        log_file = open(file_name, 'w+')
        cap = cv2.VideoCapture(video)
        i = 1
        cap.set(1,i)
        success, image = cap.read()
        while(success):
            detect(m, image, class_names, log_file, i)
            i = i+10
            cap.set(1,i)
            success, image = cap.read()
            if((i-1)%10000==0):
                print("%d frames processed" % i)
        log_file.close()

        

