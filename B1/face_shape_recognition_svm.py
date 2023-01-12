from B1 import hog_extraction
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

basedir = './Datasets'
train_images_dir = os.path.join(basedir,'cartoon_set\img')
train_labels_filename = 'cartoon_set\labels.csv'

test_images_dir = os.path.join(basedir, 'cartoon_set_test\img')
test_labels_filename = 'cartoon_set_test\labels.csv'

def get_label_split(labels, is_training):
    """Prints number of files from each class

    Args:
        labels: list of labels for data
        is_training: boolean denoting whether data is the training data
    Returns: None
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
    classifier = LinearSVC(max_iter=5000)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
  
    grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
    grid.fit(x_train, y_train)

    grid_predictions = grid.predict(x_test)
    print(classification_report(y_test, grid_predictions, digits=6))
    print(grid.best_params_)
    print(grid.best_estimator_)
    return


def SVM(x_train, y_train, x_test, y_test):
    """Retuns prediction using provided data and linear SVM classifier for task B1

    Args:
        x_train: Array of training data
        y_train: Array of training labels
        x_test: Array of test data
        y_test: Array of test labels to compare model against
    Returns:
        pred: array of predictions formed from SVM using traning data
    """

    classifier = LinearSVC(C=1, max_iter=5000)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print('Classification report\n', classification_report(y_test, pred, zero_division=0, digits=6))
    cm = confusion_matrix(y_test, pred)
    # tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('B1/confusion_mat.png')
    print(cm)

def run_classifier():
    """Runs classifier for task B1
    """
    x_train, y_train = hog_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)
    x_test, y_test = hog_extraction.get_labels(basedir, test_images_dir, test_labels_filename, testing=True)
    get_label_split(y_train, True)
    get_label_split(y_test, False)
    # get_model_params(x_train, y_train, x_test, y_test) # CAN UNCOMMENT TO GET MODEL PARAMETERS WITH GRIDSEARCH
    SVM(x_train, y_train, x_test, y_test)

    return