import os 
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def get_filename(line):
    split = line.split('\t')
    filename = split[-1]
    return filename

def get_eye_colour_label(line):
    split = line.split('\t')
    shape = split[1]
    return shape

def get_labels(basedir, images_dir, labels_filename, testing):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    lines = [line.strip('\n').strip('"') for line in lines]
    eye_colour_labels = {get_filename(line.replace('"','').replace("'",'')): get_eye_colour_label(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for idx, img_path in enumerate(image_paths):
            file_name= img_path.split('\\')[-1]
            img_data = cv2.imread(img_path)
            img_data = cv2.resize(img_data, (125,125))
            # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            # reshaped = img_data.reshape((-1, 3))
            # if idx >5 and idx < 10:
            #     print(file_name)
            #     kmeans = KMeans(n_clusters=7, random_state=1).fit(reshaped)
            #     labels = kmeans.labels_
            #     centers = kmeans.cluster_centers_
            #     img = centers[labels].reshape(img_data.shape).astype('uint8')
            #     cv2.imshow('test', img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            
            # elif idx > 10:
            #     break

            all_features.append(img_data)
            all_labels.append(eye_colour_labels[file_name])
              
            progress = round((idx+1)/(len(image_paths)+1)*100, 1)
            if progress % 5 == 0:
                print('Progress = {}'.format(progress))

    return np.array(all_features), np.array(all_labels).astype(int)