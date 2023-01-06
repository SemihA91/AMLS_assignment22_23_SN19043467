from B1 import feature_extraction
import os
import json
import numpy as np
import keras
from keras.preprocessing import image
import cv2
import tensorflow as tf


basedir = './Datasets'
train_images_dir = os.path.join(basedir,'cartoon_set\img')
train_labels_filename = 'cartoon_set\labels.csv'

test_images_dir = os.path.join(basedir, 'cartoon_set_test\img')
test_labels_filename = 'cartoon_set_test\labels.csv'



def run_classifier():

    x_train, y_train = feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = feature_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)
    print(len(y_train), len(y_test))
    print(x_train.shape)
    print(x_train[0])

    x_train = x_train.reshape(x_train.shape[0], 500*500*3)
    x_test = x_test.reshape(x_test.shape[0], 500*500*3)
    # x_train = x_train/255
    # x_test = x_test/255

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    model = keras.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(750000,), activation='sigmoid', ))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(x_train, y_train, epochs=5)

    predicted = model.predict(x_test)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)