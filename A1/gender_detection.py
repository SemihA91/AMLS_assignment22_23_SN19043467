from . import gender_landmarks
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import json
import os
from os import path
from sklearn.model_selection import GridSearchCV
import seaborn as sns

basedir = './Datasets'
train_images_dir = os.path.join(basedir,'celeba\img')
train_labels_filename = 'celeba\labels.csv'

test_images_dir = os.path.join(basedir, 'celeba_test\img')
test_labels_filename = 'celeba_test\labels.csv'

def get_gender_split(training_labels, testing_labels):
    """Prints split of data between male and female from labels

    Args:
        training_labels: array of 0s or 1s corresponding to female or male respecitvely from training data
        testing_labels: array of 0s or 1s corresponding to female or male respecitvely from testing data
    Returns:
        None
    """
    male_training = np.count_nonzero(training_labels)
    female_training = len(training_labels) - male_training
    male_testing = np.count_nonzero(testing_labels)
    female_testing = len(testing_labels) - male_testing
    print('Training split: {} male, {} female'.format(male_training, female_training))
    print('Testing split: {} male, {} female'.format(male_testing, female_testing))

def get_model_params(x_train, y_train, x_test, y_test):
    """Set optimals SVM parameters 

    Utilises Sklearn GridSearch to find optimal SVM paramters using lists of 
    provided parameters

    Args:
        x_train: Array of training data
        y_train: Array of training labels
        x_test: Array of test data
        y_test: Array of test labels to compare model against
    """
    classifier = SVC()
    param_grid = [{'C': [0.01, 0.1, 1], 
              'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
              'kernel': ['linear', 'rbf']},
              {'C': [0.01, 0.1, 1], 
                'kernel': ['poly'],
                'degree': [3, 4, 5]
              }]
    grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
    grid.fit(x_train, y_train)
    grid_predictions = grid.predict(x_test)
    print(classification_report(y_test, grid_predictions, digits=6))
    print(grid.best_params_)
    print(grid.best_estimator_)
    return


def A1_SVM(x_train, y_train, x_test, y_test):
    """Retuns prediction using provided data and SVM classifier for task A1

    Args:
        x_train: Array of training data
        y_train: Array of training labels
        x_test: Array of test data
        y_test: Array of test labels to compare model against
    Returns:
        pred: array of predictions formed from SVM using traning data
    """

    classifier = SVC(C=0.1, kernel='poly', degree=4)
    classifier.fit(x_train, y_train)
    print(classifier._gamma)
    pred = classifier.predict(x_test)
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print('Classification report\n', classification_report(y_test, pred, zero_division=0, digits=6))
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.savefig('A1/confusion_mat.png')
    print('True Negatives: {}, False Positives: {}, False Negatives: {}, True Positives: {} '.format(tn, fp, fn, tp))
    return pred

def run_classifier():
    if not os.path.exists('A1/training_data.json'):
        x_train, y_train = gender_landmarks.extract_features_labels(basedir, train_images_dir, train_labels_filename, testing=False)
        x_test, y_test = gender_landmarks.extract_features_labels(basedir, test_images_dir, test_labels_filename, testing=True)
    
    else:
        train = open('A1/training_data.json')
        training_data = json.load(train)
        x_train = np.array(training_data['features'])
        y_train = np.array(training_data['labels'])
        test = open('A1/test_data.json')
        testing_data = json.load(test)
        x_test = np.array(testing_data['features'])
        y_test = np.array(testing_data['labels'])

    print('A1 CLASSIFIER')
    get_gender_split(y_train, y_test)
    x_train = x_train.reshape((x_train.shape[0], 68*2))
    x_test = x_test.reshape((x_test.shape[0], 68*2))

    
    #get_model_params(x_train, y_train, x_test, y_test) # Cross validation to obtain the best model, 
                                                    # UNCOMMENT TO RUN as it has a considerable runtime
    A1_SVM(x_train, y_train, x_test, y_test)

 