# Hand Gesture Recognition System
A Hand Gesture Recognition System that uses MediaPipe and a custom Neural Network model to detect hand gestures from a webcam feed and apply different image filters in real time.

## Introduction to the Project

This project implements a hand gesture recognition system using MediaPipe Hands to extract landmark data and a custom Convolutional Neural Network (CNN) to classify gestures in real time and change the camera filter. The model was trained on right-hand landmarks and performs best on right-hand inputs. Left-hand recognition is a limitation that can be improved with further training.

## Why This Project is Useful

This project enables real-time hand gesture classification making it suitable for applications such as
Hand gesture recognition has various applications, including human-computer interaction, sign language interpretation, and augmented reality interfaces. This project provides an easy-to-use framework for recognizing gestures and applying effects based on detected gestures.

## Methodology

### Data Collection
* Collected hand landmark data on six different hand gestures performed by me,  primarily consisting of right-hand gestures.
* Extracted landmarks from images using MediaPipe.
* Each gesture is represented as a set of landmark coordinates.

### Model Training
* Implemented a custom Convolutional Neural Network (CNN) for classification and trained the model using PyTorch
* Optimized the model using categorical cross-entropy loss and Adam optimizer.

### System Implementation
* The trained model is included as a pre-trained file, so users can directly run the system without retraining.
* Implemented gesture-based image filters that apply real-time effects based on recognized gestures.

## Outcome

* The model accurately classifies right-hand gestures with high performance.
* Left-hand gestures are less reliable and require further improvement.
