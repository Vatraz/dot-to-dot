import cv2
import numpy as np

from circle_detction import detect_circles, draw_circles, erase_circles
from number_detection import detect_numbers



image = cv2.imread('img\(1).jpg', cv2.IMREAD_COLOR)
h, w = image.shape[:2]

if w > h:
    image = cv2.resize(image, (1200, int(h * float(1200 / w))), cv2.INTER_LINEAR)
else:
    image = cv2.resize(image, (int(w * float(1200 / h)), 1200))

# M = cv2.getRotationMatrix2D((10, 100), 45, 1)
# print(M)
circles = detect_circles(image)
# image = erase_circles(image, circles)
# draw_circles(image, circles)
detect_numbers(image, circles)
