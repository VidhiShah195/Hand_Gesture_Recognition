# Hand Gesture Recognition System
A Hand Gesture Recognition System that uses MediaPipe and a custom Neural Network model to detect hand gestures from a webcam feed and apply different image filters in real time.

## Introduction to the Project

This project implements a hand gesture recognition system that leverages MediaPipe Hands to extract landmark data and a custom Convolutional Neural Network (CNN) to classify gestures in real-time and trigger different camera filters. The system is designed to identify specific hand gestures, which then correspond to unique image filters, enhancing the user experience in applications like augmented reality and human-computer interaction. Currently, the model is optimized for right-hand gestures, with left-hand gesture recognition as a potential area for future improvement.

## Why This Project is Useful

Hand gesture recognition is crucial for various fields like:
* Human-computer interaction: Enabling hands-free control over devices and applications.
* Sign language interpretation: Translating sign language gestures into text or speech.
* Augmented reality (AR): Enhancing AR applications with gesture-based controls and interactive features.

This project provides an easy-to-use framework for gesture-based control, ideal for use in gaming, education, accessibility tools, and entertainment applications.

## Methodology

### Data Collection
* The dataset consists of six distinct hand gestures: Peace Sign, OK Sign, Thumbs Up, Fist, Open Palm, and Index Finger.
* The gestures were performed by me (right hand only) and captured with a webcam.
* MediaPipe was used to extract landmark coordinates for each hand, with each gesture represented by a unique set of landmarks. These landmarks capture the position of key points like the wrist, fingertips, and knuckles.

### Model Training
* A custom Convolutional Neural Network (CNN) was built and trained using PyTorch, optimized for classifying the six gestures.
* The model uses categorical cross-entropy as the loss function and the Adam optimizer for training.

### System Implementation
* The trained model is included as a pre-trained file, so users can directly run the system without retraining.
* Implemented gesture-based image filters that apply real-time effects based on recognized gestures.


## System Implementation
* A pre-trained model file is included, enabling users to run the gesture recognition system without the need for retraining.
* The system applies different image filters in real time based on the recognized gesture:
    * Open Palm → No filter.
    * Peace Sign → Applies a sepia filter.
    * OK Sign → Applies a cartoon filter.
    * Thumbs Up → Applies a blur effect.
    * Fist → Applies a grayscale filter.
    * Index Finger → Triggers edge detection.

## Outcome

* The model successfully classifies hand gestures with high accuracy when using right-hand gestures.
* The system applies various image filters in real time based on the detected gestures, enhancing user interaction.
* The project demonstrates the potential for integrating hand gesture recognition with real-time visual effects, providing a novel user experience for various applications.

## Limitations

* The model is optimized for right-hand gestures and does not perform as well for left-hand gestures due to a lack of training data for the left hand.
* Future improvements include:
  * Expanding the dataset to include left-hand gestures for better overall accuracy.
  * Enhancing the model with more gesture categories for broader applicability.
  * Incorporating real-time hand pose tracking improvements to better handle occlusions and dynamic gestures.
