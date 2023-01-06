from B1 import feature_extraction
import os


basedir = './Datasets'
train_images_dir = os.path.join(basedir,'cartoon_set\img')
train_labels_filename = 'cartoon_set\labels.csv'

test_images_dir = os.path.join(basedir, 'cartoon_set_test\img')
test_labels_filename = 'cartoon_set_test\labels.csv'


def run_classifier():
    feature_extraction.get_labels(basedir, train_images_dir, train_labels_filename, testing=False)