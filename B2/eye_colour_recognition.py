from B2 import feature_extraction
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
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

def get_model_params(x_train, y_train, x_test, y_test):
    """Finds optimal SVM parameters 

    Utilises Sklearn GridSearch to find optimal SVM paramters using lists of 
    provided parameters

    Args:
        x_train: Array of training data
        y_train: Array of training labels
        x_test: Array of test data
        y_test: Array of test labels to compare model against
    """
    classifier = KNeighborsClassifier()
    param_grid = {'n_neighbors': [100, 500, 1000]}
  
    grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
    grid.fit(x_train, y_train)

    grid_predictions = grid.predict(x_test)
    print(classification_report(y_test, grid_predictions, digits=6))
    print(grid.best_params_)
    print(grid.best_estimator_)
    return


def run_classifier():
    x_train, y_train = feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = feature_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)

    # feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=1, stratify=y_train)
    # x_train = np.array(x_train)/255
    # x_test = np.array(x_test)/255
    # x_validaton = np.array(x_validation)/255

    x_train = x_train.reshape(x_train.shape[0], 125*125*3)
    x_test = x_test.reshape(x_test.shape[0], 125*125*3)

    # get_model_params(x_train, y_train, x_test, y_test)
    print('CLASSIFYING')
    model = KNeighborsClassifier(n_neighbors=5000)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    return

def nn_classifier():
    x_train, y_train = feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = feature_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=1, stratify=y_train)

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=8, kernel_size=(2,2), activation='relu', input_shape=(125,125, 3)))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten(input_shape=(125,125,3)))
    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    hist = model.fit(x_train, y_train, epochs=15, validation_data=(x_validation, y_validation))
    predicted = model.predict(x_test)
    predicted_labels = [np.argmax(label) for label in predicted]
    plot_performance(hist)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predicted_labels)
    print(confusion_matrix)
    


    print('Classificaton Report: \n', classification_report(y_test, predicted_labels))
    plt.figure()
    sns.heatmap(confusion_matrix, annot=True)
    plt.show()
    return