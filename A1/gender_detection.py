from . import landmarks
import os
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC

basedir = './Datasets'
train_images_dir = os.path.join(basedir,'celeba\img')
train_labels_filename = 'celeba\labels.csv'

test_images_dir = os.path.join(basedir, 'celeba_test\img')
test_labels_filename = 'celeba_test\labels.csv'

def img_SVM(x_train, y_train, x_test, y_test):
    classifier = SVC()
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print('Classification report\n', classification_report(y_test, pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print('True Negatives: {}, False Positives: {}, False Negatives: {}, True Positives: {} '.format(tn, fp, fn, tp))
    return pred

def test():
    x_train, y_train = landmarks.extract_features_labels(basedir, train_images_dir, train_labels_filename)
    x_test, y_test = landmarks.extract_features_labels(basedir, test_images_dir, test_labels_filename)
    y_train = np.array([int(y) for y in y_train])
    y_test = np.array([int(y) for y in y_test])
    pred=img_SVM(x_train.reshape((x_train.shape[0], 68*2)), y_train, x_test.reshape((x_test.shape[0], 68*2)), y_test)

    #print(len(x_test))
    # print(len(y_test)) 
    # lengths = [get_len(data) for data in [x_train, y_train, x_test, y_test]]
    # print(lengths)
    # y_test = np.array([y_test, -(y_test - 1)]).T
    # y_train = np.array([y_train, -(y_train - 1)]).T

    

    # # Scikit learn library results
    # print('testing shape: {}'.format(y_test.shape))
    # print('training shape: {}'.format(y_train.shape))

    
    # pred=img_SVM(x_train.reshape((x_train.shape[0], 68*2)), list(zip(*y_train))[0], x_test.reshape((x_test.shape[0], 68*2)), list(zip(*y_test))[0])
    # # pred = img_SVM(x_train, y_train, x_test, y_test)