import os 
from keras.preprocessing import image
import cv2
import json
import numpy as np
import dlib

def get_filename(line):
    """Gets filename from line of CSV

    Args:
        line: line from CSV
    Returns:
        filename: string representing filename of images 
    """
    split = line.split('\t')
    filename = split[-1]
    return filename

def get_face_shape_label(line):
    """Gets face shape label from line of CSV
    Args:
        line: line from CSV
    Returns:
        filename: int from 0 to 4 indicating face shape class

    """
    split = line.split('\t')
    shape = split[-2]
    return shape

def get_labels(basedir, images_dir, labels_filename, testing):
    """Returns image data and labels from dataset
    Args:
        basedir: base directory of images
        images_dir: directory of images
        labels_filename: directory of file containing labels for images
        testing: boolean denoting whether data is test or train data
    Returns:
        all_features: numpy array of image data
        all_labels: numpy array of image labels 

    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    lines = [line.strip('\n').strip('"') for line in lines]
    face_shape_labels = {get_filename(line.replace('"','').replace("'",'')): get_face_shape_label(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for idx, img_path in enumerate(image_paths):
            file_name= img_path.split('\\')[-1]
            img_data = cv2.imread(img_path)
            img_data = cv2.resize(img_data, (125,125))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            
            all_features.append(img_data)
            all_labels.append(face_shape_labels[file_name])
            progress = round((idx+1)/(len(image_paths)+1)*100, 1)
            if progress % 5 == 0:
                print('Progress = {}'.format(progress))

    return np.array(all_features), np.array(all_labels).astype(int)