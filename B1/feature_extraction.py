import os 
from keras.preprocessing import image

def get_filename(line):
    split = line.split('\t')
    filename = split[-1]
    return filename

def get_face_shape_label(line):
    split = line.split('\t')
    shape = split[-2]
    return shape

def get_labels(basedir, images_dir, labels_filename, testing):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    lines = [line.strip('\n').strip('"') for line in lines]
    face_shape_labels = {get_filename(line.replace('"','').replace("'",'')): get_face_shape_label(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for idx, img_path in enumerate(image_paths):
            file_name= img_path.split('\\')[-1]
            print(file_name)
            img_data = image.image_utils.img_to_array(
                image.image_utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            print(img_data.shape)
            if idx == 0:
                break



    # print(list(face_shape_labels.keys())[:10], list(face_shape_labels.values())[:10])
    return