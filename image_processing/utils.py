import numpy as np
import cv2


def bounding_rect_centroid(bounding_rect):
    (x, y, w, h) = bounding_rect
    x_br, y_br = int(x + w / 2), int(y + h / 2)
    return x_br, y_br


def bounding_rect_rel_pos(bounding_rect, point):
    x_br, y_br = bounding_rect_centroid(bounding_rect)
    x_p, y_p = point[0],  point[1]
    return (x_br-x_p, y_br-y_p)


def distance_to_point(bounding_rect, point):
    x, y = bounding_rect_rel_pos(bounding_rect, point)
    distance = np.sqrt(x**2 + y**2)
    return distance


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
