from . import landmarks


def test():
    x_train, y_train = landmarks.extract_features_labels()
    print(x_train[:100])
    print(y_train[:100])

