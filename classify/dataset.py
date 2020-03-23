from classify.models import imutils
import numpy as np
import mahotas
import cv2





def center_extent(image, size):
    # Grab the extent width and height
    (w, h) = size

    # When the width is greater than the height
    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=w)
    # When the height is greater than the width
    else:
        image = imutils.resize(image, height=h)

    # Save memory for the extent of the image and grab it
    extent = np.zeros((h, w), dtype="uint8")
    offset_x = (w - image.shape[1]) // 2
    offset_y = (h - image.shape[0]) // 2
    extent[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1]] = image

    # Compute the center of mass of the image and then move the center of mass to the center of the image
    (c_y, c_x) = np.round(mahotas.center_of_mass(extent)).astype("int32")
    (d_x, d_y) = ((size[0] // 2) - c_x, (size[1] // 2) - c_y)
    matrix = np.float32([[1, 0, d_x], [0, 1, d_y]])
    extent = cv2.warpAffine(extent, matrix, size)

    # Return the extent of the image
    return extent