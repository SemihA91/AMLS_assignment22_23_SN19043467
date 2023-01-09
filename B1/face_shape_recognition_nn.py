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
# np.random.seed(3)

def plot_performance(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'])
    plt.savefig('B1/accuracy.png')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'])
    plt.savefig('B1/losses.png')


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
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=1, stratify=y_train)
    x_train = np.array(x_train)/255
    x_test = np.array(x_test)/255
    x_validaton = np.array(x_validation)/255

    return x_train, x_validation, y_train, y_validation

def get_ann():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(125,125,1)))
    model.add(keras.layers.Dense(5, activation='softmax'))
    return model

def get_cnn():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(125,125,1)))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(5, activation='softmax'))
    return model

def run_classifier():

    x_train, y_train = feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = feature_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)
    
    get_label_split(y_train, True)
    get_label_split(y_test, False)

    x_train, x_validation, y_train, y_validation = process_data(x_train, y_train, x_test, test_size=0.2)

    model = get_ann()
    # model = get_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(x_train, y_train, epochs=25, validation_data=(x_validation, y_validation))
    plot_performance(hist)
    predicted = model.predict(x_test)
    predicted_labels = [np.argmax(label) for label in predicted]

    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predicted_labels)
    print(confusion_matrix)

    print('Classificaton Report: \n', classification_report(y_test, predicted_labels))
    print(y_test[:25], predicted_labels[:25])