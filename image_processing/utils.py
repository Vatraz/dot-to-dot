import numpy as np
import cv2


def distance_to_bounding_rect(bounding_rect, point):
    (x, y, w, h) = bounding_rect
    (br_center_x, br_center_y) = (x + w//2, y+h//2)
    return np.sqrt((br_center_x - point[0])**2 + (br_center_y - point[1])**2)


def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv2.resize(image, dim, interpolation=interpolation)


def center(image, des_shape):
    des_image = np.zeros(des_shape, dtype="uint8")

    (h, w) = image.shape[:2]
    if h > w:
        image = resize(image, height=des_shape[0])
    else:
        image = resize(image, width=des_shape[1])

    offset_x = (des_shape[1] - image.shape[1]) // 2
    offset_y = (des_shape[0] - image.shape[0]) // 2

    des_image[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1]] = image
    return des_image
