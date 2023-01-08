from B1 import feature_extraction
import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report


basedir = './Datasets'
train_images_dir = os.path.join(basedir,'cartoon_set\img')
train_labels_filename = 'cartoon_set\labels.csv'

test_images_dir = os.path.join(basedir, 'cartoon_set_test\img')
test_labels_filename = 'cartoon_set_test\labels.csv'

def plot_performance(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])
    plt.savefig('B1/TRAININGFIG.png')

def get_label_split(labels, train):
    zero = np.sum(labels == 0)
    one = np.sum(labels == 1)
    two = np.sum(labels == 2)
    three = np.sum(labels == 3)
    four = np.sum(labels == 4)

    if train: 
        print('TRAIN: Zeros: {}, Ones: {}, Twos: {}, Threes: {}, Fours: {}'.format(zero, one, two, three, four))
    else: 
        print('TEST: Zeros: {}, Ones: {}, Twos: {}, Threes: {}, Fours: {}'.format(zero, one, two, three, four))

def process_data(x_train, y_train, x_test, test_size):
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    x_train = np.array(x_train)/255
    x_test = np.array(x_test)/255
    x_validaton = np.array(x_validation)/255

    return x_train, x_validation, y_train, y_validation

def get_ann(activation, optimizer, learning_rate, epochs):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(125,125,3)))
    model.add(keras.layers.Dense(5, activation='softmax'))
    return model


def run_classifier():

    x_train, y_train = feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = feature_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)
    
    get_label_split(y_train, True)
    get_label_split(y_test, False)

    x_train, x_validation, y_train, y_validation = process_data(x_train, y_train, x_test, test_size=0.2)


    epochs = range(5, 21, 2)
    learning_rates = [0.1 0.01 0.001 0.0001]



    model = get_ann()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(x_train, y_train, epochs=15, validation_data=(x_validation, y_validation))
    predicted = model.predict(x_test)
    predicted_labels = [np.argmax(label) for label in predicted]

    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predicted_labels)
    print(confusion_matrix)

    print('Classificaton Report: \n', classification_report(y_test, predicted_labels))
    print(y_test[:25], predicted_labels[:25])