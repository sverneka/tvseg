import numpy as np
import pandas as pd
import cv2
import math
import glob
import os
img_count = 0

# Convert time to frame number
def get_frame(t):
    return (t.hour*3600+t.minute*60+t.second)*30

# Sample frames from start to end at every 200th frame

def sample_random_frames(start, end, cap, show):
    global img_count
    fldr_name = "images/"+ show
    if not os.path.exists(fldr_name):
        os.mkdir(fldr_name)
    i = max(1,start)
    while(i<=end):
        cap.set(1,i)
        success, image = cap.read()
        if(success):
            cv2.imwrite(os.path.join(fldr_name, str(img_count)+'.png'), image)
            img_count += 1
            if(img_count % 1000 == 0):
                print img_count/1000
        i=i+200


# Sample boundary frames at every 10th frame from start to end

def sample_boundary_frames(start, end, cap, fldr_name):
    i=max(1,start)
    if not os.path.exists(fldr_name):
        os.mkdir(fldr_name)
    while(i<=end):
        cap.set(1,i)
        success, image = cap.read()
        if(success):
            cv2.imwrite(os.path.join(fldr_name, str(i)+'.png'), image)
        i=i+10


count = 0

# You need to install xlrd before using this by running "sudo pip install xlrd"
# Extract frames around boundary timing listed on annoation file
# Manually discard the ones that aren't exactly the boundaries
# going into each folder created and finally put them all into "images/boundaries"
# folder

df = pd.ExcelFile('sample_split_annotations.xlsx').parse('Sheet1')
columns = df.columns
start = df['Start']
finish = df['Finish']
show = df['Show']

i=0
for video in glob.iglob('videos/*.mp4'):
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    while(i<len(start) and not pd.isnull(start[i])):
        start_frame = get_frame(start[i])
        end_frame = get_frame(finish[i])
        sample_boundary_frames(start_frame-300, start_frame+300, cap, video.split('/')[1]+show[i]+'start')
        sample_boundary_frames(end_frame-300, end_frame+300, cap, video.split('/')[1]+show[i]+'end')
        i+=1
    i+=1




# Extract non boundary frames and put them into 'images/[show]' folder

i=0
for video in glob.iglob('videos/*.mp4'):
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    while(i<len(start) and not pd.isnull(start[i])):
        start_frame = get_frame(start[i])
        end_frame = get_frame(finish[i])
        sample_random_frames(start_frame+1000, end_frame-1000, cap, show[i])
        i+=1
    i+=1


	










