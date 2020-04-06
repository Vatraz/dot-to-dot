import joblib

from classify.hog import HOG


class Classifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.hog = HOG()

    def predict(self, img):
        hist = self.hog.describe(img)
        digit = self.model.predict([hist])[0]
        return digit
