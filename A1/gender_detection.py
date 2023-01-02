from . import landmarks
import os
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import json
import os
from os import path
from sklearn.model_selection import GridSearchCV

basedir = './Datasets'
train_images_dir = os.path.join(basedir,'celeba\img')
train_labels_filename = 'celeba\labels.csv'

test_images_dir = os.path.join(basedir, 'celeba_test\img')
test_labels_filename = 'celeba_test\labels.csv'

def img_SVM(x_train, y_train, x_test, y_test):
    classifier = SVC(C=0.01, kernel='linear', gamma=1)
    # ss = StandardScaler()
    # ss.fit(x_train)
    # x_train_std = ss.transform(x_train)
    # x_test_std = ss.transform(x_test)
    # classifier.fit(x_train_std, y_train)
    # pred = classifier.predict(x_test_std)
    # print("\nAccuracy:", accuracy_score(y_test, pred))
    # print('Classification report\n', classification_report(y_test, pred, zero_division=0))
    # tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    # print('True Negatives: {}, False Positives: {}, False Negatives: {}, True Positives: {} '.format(tn, fp, fn, tp))

    # param_grid = {'C': [0.01, 0.1, 1], 
    #           'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #           'kernel': ['linear']} 
  
    # grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
    # grid.fit(x_train, y_train)

    # grid_predictions = grid.predict(x_test)
    # print(classification_report(y_test, grid_predictions))
    # print(grid.best_params_)
    # print(grid.best_estimator_)

    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print('Classification report\n', classification_report(y_test, pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print('True Negatives: {}, False Positives: {}, False Negatives: {}, True Positives: {} '.format(tn, fp, fn, tp))
    return pred

def test():
    if not os.path.exists('A1/training_data.json'):
        x_train, y_train = landmarks.extract_features_labels(basedir, train_images_dir, train_labels_filename, testing=False)
        x_test, y_test = landmarks.extract_features_labels(basedir, test_images_dir, test_labels_filename, testing=True)
    
    else:
        train = open('A1/training_data.json')

        training_data = json.load(train)
        x_train = np.array(training_data['features'])
        y_train = np.array(training_data['labels'])
        test = open('A1/test_data.json')
        testing_data = json.load(test)
        x_test = np.array(testing_data['features'])
        y_test = np.array(testing_data['labels'])


    # # # Scikit learn library results
    # # print('testing shape: {}'.format(y_test.shape))
    # # print('training shape: {}'.format(y_train.shape))

    


    pred=img_SVM(x_train.reshape((x_train.shape[0], 68*2)), y_train, x_test.reshape((x_test.shape[0], 68*2)), y_test)

 