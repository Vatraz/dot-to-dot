import cv2
import numpy as np


def circle_filled(image, x, y, r):
    if (image.shape[1]< x+r or x-r < 0) or (image.shape[0] < y+r < 0 or y-r < 0):
        return False
    img_crop = image[y-r:y+r+1, x-r:x+r+1]
    mask = np.zeros((img_crop.shape[0], img_crop.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, color=255, thickness=cv2.FILLED)
    circle_field = np.count_nonzero(mask)
    found_circle = cv2.bitwise_and(img_crop, img_crop, mask=mask)
    if np.count_nonzero(found_circle) > int(0.7 * circle_field):
        return True
    else:
        return False


def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 126, 255, cv2.THRESH_BINARY_INV)
    gray_blur = cv2.blur(gray, (3, 3))
    # cv2.imshow('gray', gray_blur)
    circles = cv2.HoughCircles(gray_blur, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=50, param2=4,
                               minRadius=4, maxRadius=7)
    circle_list = []
    if circles is None:
        return circle_list

    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        x, y, r = circle[0], circle[1], circle[2]
        if not circle_filled(gray, x, y, r):
            continue
        circle_list.append((x, y, r))
    return circle_list


def draw_circles(image, circles):
    image = image.copy()
    for circle in circles:
        x, y, r = circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 1)
    cv2.imshow("Detected Circle", image)
    cv2.waitKey(0)
