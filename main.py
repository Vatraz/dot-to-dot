import cv2

from image_processing.dots_detection import detect_dots
from image_processing.number_detection import detect_numbers
from image_processing.utils import resize


image = cv2.imread('./img/(1).jpg', cv2.IMREAD_COLOR)
h, w = image.shape[:2]

if w > h:
    image = resize(image, width=1200)
else:
    image = resize(image, height=1200)

dots = detect_dots(image)
numerated_dots = detect_numbers(image, dots)
