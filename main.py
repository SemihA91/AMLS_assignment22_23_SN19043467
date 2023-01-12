from A1 import gender_detection as a1
from A2 import emotion_detection as a2
from B1 import face_shape_recognition_nn as a3_nn
from B1 import face_shape_recognition_svm as a3_hog_svm
from B2 import eye_colour_recognition as a4
import os

def create_datasets_dir():   
    if not os.path.exists('Datasets/'):
        os.makedirs('Datasets')
    return

def main():
    create_datasets_dir()
    a1.run_classifier()
    a2.run_classifier() 
       
    # a3_nn.run_classifier() # Can uncomment to test neural network for B1
    
    a3_hog_svm.run_classifier()
    a4.run_classifier()


if __name__ == "__main__":
    main()
