import os 
from keras.preprocessing import image
import cv2
import json
import numpy as np
import dlib

def get_filename(line):
    split = line.split('\t')
    filename = split[-1]
    return filename

def get_face_shape_label(line):
    split = line.split('\t')
    shape = split[-2]
    return shape

def get_grey_image(image):
    image = image.astype('uint8')
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('uint8')
    return grey_image

def get_labels(basedir, images_dir, labels_filename, testing):
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
            # print('{}% processed'.format(round((idx+1)/(len(image_paths)+1)*100, 4)))
            # img_data = image.image_utils.img_to_array(
            #     image.image_utils.load_img(img_path,
            #                    target_size=target_size,
            #                    interpolation='bicubic')).astype('uint8')
            ### img_data = get_grey_image(img_data)


            # img_data = image.image_utils.load_img(img_path,
            #                    target_size=target_size,
            #                    interpolation='bicubic')
            img_data = cv2.imread(img_path)
            img_data = cv2.resize(img_data, (125,125))
            
            all_features.append(img_data)
            all_labels.append(face_shape_labels[file_name])
            progress = round((idx+1)/(len(image_paths)+1)*100, 1)
            if progress % 5 == 0:
                print('Progress = {}'.format(progress))

    return np.array(all_features), np.array(all_labels).astype(int)