import cv2

from image_processing.circle_detction import detect_circles
from image_processing.number_detection import detect_numbers
from image_processing.utils import resize


image = cv2.imread('./img/(1).jpg', cv2.IMREAD_COLOR)
h, w = image.shape[:2]

if w > h:
    image = resize(image, width=1200)
else:
    image = resize(image, height=1200)
# cv2.imshow('haha', image)
# cv2.waitKey()

circles = detect_circles(image)

detect_numbers(image, circles)
