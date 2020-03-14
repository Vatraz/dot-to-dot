import cv2
import numpy as np

from circle_detction import detect_circles, draw_circles

image = cv2.imread('img\image2.jpg', cv2.IMREAD_COLOR)

circles = detect_circles(image)
draw_circles(image, circles)
