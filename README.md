# tvseg

This is for GSoC-2018 project, TV Show Segmentation in collaboration with Red Hen Lab, http://www.redhenlab.org/home/the-cognitive-core-research-topics-in-red-hen/the-barnyard/tv-show-segmentation

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

It saves these numpy arrays into numpy train_X_64.py and train_Y_64.py

Note: You need 12GB memory to create this, if you don't have enough memory, modify create_dataset.py to create smaller dataset.

```bash
$ python create_dataset.py
```

# Run siamese_net.py
It runs for 20 epochs and saves the model trained into model_test.h5

It gives around 99% accuracy on train and validation data.

```bash
$ python siamese_net.py
```

# Run boundary_detector.py
It runs through all the videos in "videos" folder and puts boundary images into "detected_boundaries/[video_name]" folder.
Boundary images are named by their frame number.

```bash
$ python boundary_detector.py
```






















































