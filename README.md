# Hand Gesture Recognition System
Hand Gesture Recognition System that uses MediaPipe and a custom Neural Network model to detect hand gestures from a webcam feed and apply different image filters in real time.

## Introduction to the Project

This project is a hand gesture recognition system built using MediaPipe and PyTorch. The system detects hand gestures from images and applies gesture-based filters in real-time. The model has been trained on right-hand gestures and works best for right-hand input. A pre-trained model is included to allow users to run the system without retraining.

## Why This Project is Useful

Hand gesture recognition has various applications, including human-computer interaction, sign language interpretation, and augmented reality interfaces. This project provides an easy-to-use framework for recognizing gestures and applying effects based on detected gestures.

## Methods and Process

### Data Collection
* Used the Hand Gesture Recognition dataset, which contains infrared images of ten different hand gestures performed by ten subjects.
* Preprocessed images using image augmentations (mirroring and rotation transformations).
* Extracted landmarks from images using MediaPipe.

### Model Training
* Implemented a custom Convolutional Neural Network (CNN) for classification.
* Trained the model using PyTorch with a dataset primarily consisting of right-hand gestures.
* Applied data augmentations to improve generalization.
* Optimized the model using categorical cross-entropy loss and Adam optimizer.

### System Implementation
* The trained model is included as a pre-trained file, so users can directly run the system without retraining.
* Implemented gesture-based image filters that apply real-time effects based on recognized gestures.
* The system loads the pre-trained model and processes user input images to classify gestures.

