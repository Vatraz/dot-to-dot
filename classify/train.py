import numpy as np
import joblib

from sklearn.svm import LinearSVC

from image_processing.hog import HOG
from image_processing.utils import center


def load_digits(dataset_path):
    data = np.genfromtxt(dataset_path, delimiter=",", dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)

    return (data, target)


def train():
    dataset_path = 'data/digits.csv'
    model_path = 'models/svm.cpickle'

    (digits, target) = load_digits(dataset_path)
    data = []

    hog = HOG(orientations=18, pixels_per_cell=(10, 10), cells_per_block=(1, 1), transform=True)

    for image in digits:
        image = center(image, (20, 20))

        hist = hog.describe(image)
        data.append(hist)

    model = LinearSVC(random_state=42)
    model.fit(data, target)

    joblib.dump(model, model_path)


if __name__ == '__main__':
    train()
