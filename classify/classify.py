import joblib
from image_processing.hog import HOG


model_path = 'classify/models/svm.cpickle'
model = joblib.load(model_path)
hog = HOG(orientations=18, pixels_per_cell=(10, 10), cells_per_block=(1, 1), transform=True)


def classify(img):
    hist = hog.describe(img)
    digit = model.predict([hist])[0]
    return digit
