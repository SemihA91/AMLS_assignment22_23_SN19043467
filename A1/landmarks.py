import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import tensorflow as tf
import pandas as pd
import json


# PATH TO ALL IMAGES
global basedir, image_paths, target_size
# basedir = os.path.abspath(os.path.join('./Datasets', os.pardir))
# basedir = './Datasets'
# images_dir = os.path.join(basedir,'celeba\img')
# labels_filename = 'celeba\labels.csv'

detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(os.sys.path[0], 'A1\shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
def get_gender(line):
    split = line.split('\t')
    if split[-2] == '-1':
        return -1
    return 1

def get_filename(line):
    split = line.split('.')[0]
    split = split[:len(split)//2]
    filename = split + '.jpg'
    return filename


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(basedir, images_dir, labels_filename, testing):
    """
    This funtion extracts the landmarks features for all images in the folder provided by 'images_dir'.
    It also extracts the gender label for each image from the folder passed in by 'labels_filename'.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {get_filename(line.replace('"','').replace("'",'')): get_gender(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for idx, img_path in enumerate(image_paths):
            file_name= img_path.split('\\')[-1]
            print('Processing {}...'.format(file_name))
            # load image
            img = image.image_utils.img_to_array(
                image.image_utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[file_name])
            
            # if idx == 20:
            #     print('HIT {}, STOPPING TO MAKE IT FASTER'.format(idx))
            #     break

    
    # all_features = [list(feature) for feature in all_features]
    # print(type(all_labels))
    # print(type(all_features[0]))
    # print(all_features[0])
    all_features = [feature.tolist() for feature in all_features]
    all_labels = [(label + 1)/2 for label in all_labels]

    data = {
        'features': all_features,
        'labels': all_labels
    }

    if testing:
        filename = 'A1/test_data.json'
    else:
        filename = 'A1/training_data.json'
    
    outfile = open(filename, 'w')
    json.dump(data, outfile, indent= 3)
    outfile.close()

  
    landmark_features = np.array(all_features)
    # all_labels = [(label + 1)/2 for label in all_labels]
    gender_labels = (np.array(all_labels))

    return landmark_features, gender_labels

