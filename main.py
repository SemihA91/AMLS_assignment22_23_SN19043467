from A1 import gender_detection as a1
from A2 import emotion_detection as a2
from B1 import face_shape_recognition as a3
import os

def create_datasets_dir():   
    if not os.path.exists('Datasets/'):
        os.makedirs('Datasets')
    return

def main():
    create_datasets_dir()
    #a1.run_classifier()
    #a2.run_classifier() 
       
    a3.run_classifier()
    # a4.run_classifier()

if __name__ == "__main__":
    main()
