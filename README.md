# tvseg

This is for GSoC-2018 project, TV Show Segmentation in collaboration with Red Hen Lab, http://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/the-barnyard/tv-show-segmentation

# Requirements
python 2.7
keras version 2.1.2 with tensorflow backend
xlrd



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
It runs for 20 epochs and saves the model trained into model_test.h5

It gives around 99% accuracy on train and validation data.

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

