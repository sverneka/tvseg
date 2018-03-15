import numpy as np
from PIL import Image
import cv2

import os
import glob
import sys
sys.path.append("crnn.pytorch")

import torch
from torch.autograd import Variable
import utils
import dataset
import models.crnn as crnn

import pickle

# Recognition model
def get_recognition_model():
    model_path = './weights/crnn.pth'
    #Intial model
    model = crnn.CRNN(32, 1, 37, 256)
    try:
        model = model.cuda()
    except:
        #there is some issue with model loading, 
        #and loading it twice solves the problem
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))
    return model


def recognize_cropped(image,  model, converter, transformer):
    image = transformer(image.convert('L'))
    image = image.cuda()
    #Reshaping by adding another axis with length 1 at the beginning
    # [1, 32, 100] -> [1, 1, 32, 100]
    image = image.view(1, *image.size())
    #Create pytorch variable and evaluate(infer) the model
    image = Variable(image, volatile=True)
    #model.eval()
    preds = model(image)
    # preds.size() = [26, 1, 37]
    #Process preds for decoding
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    #Decode
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def get_recogized_texts(image, bb, model, converter, transformer):
    #print("box is",bb)
    return recognize_cropped(image.crop(bb), model, converter, transformer)


if __name__ == '__main__':
    model = get_recognition_model()
    #For decoding model output
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    converter = utils.strLabelConverter(alphabet)
    #For preprocessing the input
    transformer = dataset.resizeNormalize((100, 32))

    fldr_name = '../VidTextExtraction'
    if not os.path.exists(fldr_name):
        os.mkdir(fldr_name)
    for file_name in glob.iglob(fldr_name+"/*.p"):
        detections = pickle.load(open(file_name, "rb"))
        print file_name
        video = "../videos/"+file_name.split('/')[-1][0:-2]+".mp4"
        print(video)
        cap = cv2.VideoCapture(video)
        file_name = file_name[0:-2]+'.txt'
        log_file = open(file_name, 'w+')
        print "log_file_name is ",file_name
        print("there are %d detections" % len(detections))
        for j in range(0,len(detections)):
            boxes = detections[j]
            i=boxes[0]
            cap.set(1,i)
            success, image = cap.read()
            if(success):
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                for box in boxes[1]:
                    text = get_recogized_texts(image, box, model, converter, transformer)
                    if(text):
                        #image.save(str(count)+'.png')
                        log_file.write(str(i)+','+text+'\n')
                        log_file.flush()
            if j%1000==0:
                print("%d detections processed" % j)
        log_file.close()
