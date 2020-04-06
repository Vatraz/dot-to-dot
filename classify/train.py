import numpy as np
import joblib

from sklearn.svm import LinearSVC

from classify.hog import HOG
from image_processing.utils_br import filter_bounding_rect, crop_bounding_rect
from image_processing.utils_img import center


def load_digits(dataset_path):
    data = np.genfromtxt(dataset_path, delimiter=",", dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 16, 16)
    return (data, target)


def crop_digit(image):
    bounding_rect = (0, 0, image.shape[1], image.shape[0])
    bounding_rect = filter_bounding_rect(image, bounding_rect)
    image = crop_bounding_rect(image, bounding_rect)
    return image


def train(model_path, dataset_path):
    digits, target = load_digits(dataset_path)
    data = []

    hog = HOG()

    for n, image in enumerate(digits):
        image = center(image, (16, 16))
        hist = hog.describe(image)
        data.append(hist)

    model = LinearSVC(random_state=42)
    model.fit(data, target)

    joblib.dump(model, model_path)
