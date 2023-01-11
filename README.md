# AMLS Assignment 22/23 - SN 19043467

## Running the Code
The code for this assignment was tested using a Conda environment with the necessary packages contained with in the [environment.yml](environment.yml) file. Once Conda is installed, the Conda environment can be created with the following command in the terminal: 

    conda env create -f environment.yml

The Conda environment must then be activated using the following command in the terminal:

    conda activate ml-final

The tasks can then be run by running [main.py](main.py) as one normally would. **Note this may take some time to run** depending on the particular machine as some models require extensive time for feature extraction and parameter tuning. In the case that certain elements take an excessive amount of time, these lines of code are highlighted with comments and can be uncommented if desired.

In the case that a Conda enviornment is not used, the required packages and dependencies are listed here for convenience:

- python=3.9
- pip
- numpy
- pandas
- matplotlib
- scikit-learn
- cmake
- opencv-python
- keras
- tensorflow
- scikit-image
- seaborn
- Pillow
- dlib

# File Structure and Roles

For this assignment, each task is separated into its own respective directory with the code being written as if the tasks are entirely separate. 

## A1 - Binary Gender Classificaiton

This directory contains the code for task A1 and is contains two code files and a pre-trained model used for feature extraction:

 - [gender_landmarks.py](A1/gender_landmarks.py) - Performs feature and label extraction on celeba dataset, based of dlib implementation of paper referenced in report - also utilises some modified code from wk6 lab
 - [gender_detection.py](A1/gender_detection.py) - Contains the code to run the SVM classifier for gender classification
 - [shape_predictor_68_face_landmarks.dat](A1/shape_predictor_68_face_landmarks.dat) - pre -trained facial detector model to detect coordinates of 68 facial landmarks 

 ## A2 - Binary Emotion Detection

This directory contains the code for task A2 and contains two code files and a pre-trained model used for feature extraction:

- [smile_landmarks.py](A2/smile_landmarks.py) - Performs feature extraction and label extraction based from celeba dataset, based off dlib implementation of paper referenced in report - also utilises some modified code from wk6 lab
- [emotion_detection.py](A2/emotion_detection.py) - Contains code to run SVM classifier for emotion detection
- [shape_predictor_68_face_landmarks.dat](A2/shape_predictor_68_face_landmarks.dat) - pre -trained facial detector model to detect coordinates of 68 facial landmarks 

## B1 - 5 Class Facial Shape Classification

This directory contains four code files for B1 of which **two are for the final solution to the task** and two are for an alternate solution:

### Final Optimised Solution:
- [face_shape_recognition_svm.py](B1/face_shape_recognition_svm.py) - Utilise HOG features and linear SVM for 5 class face shape classification
- [hog_extraction.py](B1/hog_extraction.py) - Extracts HOG features and labels from cartoon dataset to be used for linear SVM classifier

### Alternate Un-Optimised Solution

- [face_shape_recognition_nn.py](B1/face_shape_recognition_nn.py) - Testing of ANN and CNN for face shape classification. Model performed poorly and HOG + SVM was used instead.
- [feature_extraction.py](B1/feature_extraction.py) - Extracts image data and labels from cartoon dataset to be used with neural networks.

## B2