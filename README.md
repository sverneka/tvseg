# tvseg

This is for GSoC-2018 project, TV Show Segmentation in collaboration with Red Hen Lab, http://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/the-barnyard/tv-show-segmentation

# Requirements
- Install python 2.7
- Install pandas
- Install keras version 2.1.2 with tensorflow backend
- Install xlrd
- Install pytorch
- Install lmdb
- Install tqdm


First put all videos to be segmented into "videos" folder.

# Run sample_frames.py
This is to sample boundary frames as well as randomly sample non-boundary frames.

It puts boundary frames into "images/boundaries" folder and non-boundary frames into 'images/[show_name]' folder.

You need to then manually discard frames from "images/boundaries" that aren't exactly the boundaries.

```bash
$ python sample_frames.py
```

# Run create_dataset.py
This creates numpy arrays in the format needed to train siamese network by randomly creating pairs from sampled frames.

It saves these numpy arrays into files train_X_64.npy and train_Y_64.npy

Note: You need 12GB memory to run this, if you don't have enough memory, modify create_dataset.py to create a smaller dataset.

```bash
$ python create_dataset.py
```

# Run siamese_net.py
It runs for 5 epochs and saves the model trained into model_test.h5

It gives around 97% accuracy on train and validation data.

```bash
$ python siamese_net.py
```

# To test the network on videos:  run boundary_detector.py
It runs through all the videos in "videos" folder and puts boundary images into "detected_boundaries/[video_name]" folder.
Boundary images are named by their frame numbers.

```bash
$ python boundary_detector.py
```

# Just test the provided model, model_test.h5 file
If you don't want to create dataset and train the network, you can just use the provided model_test.h5 file to detect boundaries for videos in "videos" folder.

```bash
$ mkdir images
$ mv boundaries images/.
$ python boundary_detector.py
```

# Validated boundary detection
To test boundary detection around pre-annotated boundary frames, i.e, around boundary-500 and boundary+500 region, run boundary_detectory_quick_check.py. This will put boundary frames in "detected_boundaries/[video_name]/[show_name]/[begin/end]" folder.

```bash
$ python boundary_detector_quick_check.py
```



# Building a binary classifier
I just gave 2-class classifier a try that just says given a frame, if it's a boundary frame or not. As expected it didn't work, no matter what classifier you use.
The reason being severe class imbalance, where we have very few boundary frames, and very very large set of non boundary frames.
This type of problem is not learnable as corroborated by the paper, "Severe Class Imbalance: Why Better Algorithms Aren’t the Answer", https://webdocs.cs.ualberta.ca/~holte/Publications/ecml05.pdf

To test you can run below scripts and you notice that the classifier does no better than majority classification

```bash
$ python create_classifier_dataset.py
$ python classifier.py
```

# Run YOLO to create annotations
Creates video annotations using YOLO from https://github.com/marvis/pytorch-yolo2.git on videos and writes annotations in "yolo_annotations/[video_file_name].txt" file in the format, [frame_no, label, box_co-ordinates{4 numbers}, prediction_confidence]

Install Yolo
```bash
$ git clone https://github.com/sverneka/pytorch-yolo2.git
$ cd pytorch-yolo2
```
Download pre-trained YOLO weights - 80 class detection
```bash
$ wget http://pjreddie.com/media/files/yolo.weights
```
Copy yolo_annotate.py file from root to pytorch-yolo2
```bash
$ cp ../yolo_annotate.py .
```
Run yolo_annoate.py
```bash
$ python yolo_annotate.py
```

# Run video text detector and recognizer
Extracts video screen text for every 10th frame in videos and puts the corresponding .txt files in VidTextExtraction folder.
Source code modified from: https://github.com/sravya8/VideoText

Since, there are 2 models, one for detection and other for recognition, I am running them separately as you can't create 2 tensorflow sessions on single GPU through the same process.

First detect text in videos and put the bounding boxes in pickle format in VidTextExtraction folder
```bash
$ cd VideoText
$ python videotextdetect.py
```

Run videotextrecognize.py to read the bounding boxes from pickle files and recognize text from videos and put them into .txt files in the format [frame_number, text], in VidTextExtraction folder
```bash
$ python videotextrecogize.py
```
