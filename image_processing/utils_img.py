import cv2
import numpy as np


def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]

    if h == 0:
        print('halo')
    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv2.resize(image, dim, interpolation=interpolation)


def normalize(img):
    f = 255/img.max()
    img = np.multiply(img, f).astype(np.uint8)
    return img


def center(image, des_shape):
    des_image = np.zeros(des_shape, dtype="uint8")
    h, w = image.shape[:2]
    if h > w:
        image = resize(image, height=des_shape[0])
    else:
        image = resize(image, width=des_shape[1])

    offset_x = (des_shape[1] - image.shape[1]) // 2
    offset_y = (des_shape[0] - image.shape[0]) // 2

    des_image[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1]] = image
    return des_image


def erode_thresh(img_thresh, iter=1):
    """
    Returns the eroded img_thresh image.

    :param img_thresh: grayscale image
    :param iter: number of iterations
    :return: eroded image
    """
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = cv2.erode(img_thresh, element)
    while iter > 1:
        img = cv2.erode(img_thresh, element)
        iter -= 1
    return img


def calculate_indent(elements):
    indent = 0
    for elem in elements:
        if elem == 0:
            indent += 1
        else:
            break
    return indent


def vertical_indent(image, x_pos):
    column = image[:,x_pos].ravel()
    left = calculate_indent(column)
    right = calculate_indent(reversed(column))
    return [left, right]


def horizontal_indent(image, y_pos):
    row = image[y_pos, :].ravel()
    left = calculate_indent(row)
    right = calculate_indent(reversed(row))
    return [left, right]


def get_vertical_indents(image):
    indents = []
    for x_pos in range(image.shape[1]):
        indents.append(sum(vertical_indent(image, x_pos)))
    return np.array(indents)


def indents_exc_thresh(image, threshold=9, trim=0):
    image = image[:,trim:image.shape[1]-trim]
    indents = get_vertical_indents(image)
    indents_exc = []
    prev = False
    for indent in indents:
        if indent > threshold and prev == False:
            indents_exc.append(indent)
            prev = True
        if indent <= threshold:
            prev = False

    indents = indents[indents > threshold]
    return indents