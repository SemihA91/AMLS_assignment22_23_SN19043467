from B2 import feature_extraction
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import seaborn as sns

basedir = './Datasets'
train_images_dir = os.path.join(basedir,'cartoon_set\img')
train_labels_filename = 'cartoon_set\labels.csv'

test_images_dir = os.path.join(basedir, 'cartoon_set_test\img')
test_labels_filename = 'cartoon_set_test\labels.csv'


def plot_performance(history):
    """Plots the training and validation accuracy and losses of classifier

    Args:
        history: Keras model history
    Returns:
        None
    """
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])
    plt.savefig('B2/accuracy.png')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'])
    plt.savefig('B2/losses.png')

def get_label_split(labels, is_training):
    """Prints the number of samples from each class

    Args:
        labels: array of labels for data
        is_training: boolean to denote whether data is for training
    """
    zero = np.sum(labels == 0)
    one = np.sum(labels == 1)
    two = np.sum(labels == 2)
    three = np.sum(labels == 3)
    four = np.sum(labels == 4)
    if is_training: 
        print('TRAIN: Zeros: {}, Ones: {}, Twos: {}, Threes: {}, Fours: {}'.format(zero, one, two, three, four))
    else: 
        print('TEST: Zeros: {}, Ones: {}, Twos: {}, Threes: {}, Fours: {}'.format(zero, one, two, three, four))
    
    return

def run_classifier():
    """Runs CNN classifier
    """
    x_train, y_train = feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = feature_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)

    get_label_split(y_train, True)
    get_label_split(y_test, False)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=1, stratify=y_train)

    # Normalising the pixel values
    x_train = x_train/255
    x_test = x_test/255
    x_validation = x_validation/255

    
    FILTERS = 8
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=FILTERS, kernel_size=(3,3), activation='relu', input_shape=(125,125, 3)))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten(input_shape=(125,125,3)))
    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    hist = model.fit(x_train, y_train, epochs=15, validation_data=(x_validation, y_validation))
    predicted = model.predict(x_test)
    predicted_labels = [np.argmax(label) for label in predicted]
    plot_performance(hist)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    print('Classificaton Report: \n', classification_report(y_test, predicted_labels, digits=6))
    print('Filters: {}'.format(FILTERS))
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1', '2', '3', '4'])
    disp.plot()
    plt.savefig('B2/confusion_mat.png') 
    return