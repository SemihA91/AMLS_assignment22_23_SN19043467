from . import landmarks
import os
import numpy as np

basedir = './Datasets'
train_images_dir = os.path.join(basedir,'celeba\img')
train_labels_filename = 'celeba\labels.csv'

test_images_dir = os.path.join(basedir, 'celeba_test\img')
test_labels_filename = 'celeba_test\labels.csv'

def test():
    x_train, y_train = landmarks.extract_features_labels(basedir, train_images_dir, train_labels_filename)
    x_test, y_test = landmarks.extract_features_labels(basedir, test_images_dir, test_labels_filename)
    
    print(y_test)
    # print(y_train[0])
    # print(x_train[0])
    # print(x_test[0])

