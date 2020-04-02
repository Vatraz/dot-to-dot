import cv2
import numpy as np

DOT_MIN_DIST = 10


def erase_circles(image, circles):
    """
    Returns a copy of the image without circles from the circles list.

    :param image: image
    :param circles: list of circles in (x, y, r) format
    :return: image with circles erased
    """
    color = (255, 255, 255) if image.shape[2] > 1 else 255

    image = image.copy()
    for circle in circles:
        x, y, r = circle
        cv2.circle(image, (x+1, y+1), r, color=color, thickness=cv2.FILLED)
    return image


def is_circle(image, circle):
    """
    Checks if the image contains a circle in an area described by (x, y, r).

    :param image: image in greyscale
    :param circle: circle - (x, y, r)
    :return: whether (x, y, r) circle is found in the image
    """
    x, y, r = circle
    offset = r + 3
    # if found circle is on the edge of the image
    if (image.shape[1] < x+offset or x-offset < 0) or (image.shape[0] < y+offset < 0 or y-offset < 0):
        return False

    # part of the image that should contain a circle
    img_crop = image[y-offset:y+1+offset, x-offset:x+1+offset]

    # calculate the reference circle area
    mask_circle = np.zeros((img_crop.shape[0], img_crop.shape[1]), dtype=np.uint8)
    cv2.circle(mask_circle, (offset, offset), r, color=255, thickness=cv2.FILLED)
    mask_circle_inv = np.bitwise_not(mask_circle)
    ref_circle_field = np.count_nonzero(mask_circle)
    ref_circle_inv_field = img_crop.size - ref_circle_field

    # calculate the area of the circle described by (x, y, r)
    found_circle_field = np.count_nonzero(cv2.bitwise_and(img_crop, img_crop, mask=mask_circle))
    found_circle_inv_field = np.count_nonzero(cv2.bitwise_and(img_crop, img_crop, mask=mask_circle_inv))

    if found_circle_field > int(0.7 * ref_circle_field) and found_circle_inv_field < int(ref_circle_inv_field * 0.3):
        return True
    else:
        return False


def get_circle_list(image):
    """
    Returns a list of circles found in the BGR image.

    :param image: image on which circles will be searched
    :return: list of circles in (x, y, r) format
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (3, 3))
    _, gray_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    circles = cv2.HoughCircles(gray_blur, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=50, param2=5,
                               minRadius=1, maxRadius=3)
    circles = np.int16(np.around(circles))[0]

    circles = [circle for circle in circles if is_circle(gray_thresh, circle)]

    return circles


def detect_dots(image):
    """
    Returns a list containing position (x, y) and radius r of dots found in the image.

    :param image: processed image
    :return: list of dots in (x, y, r) format
    """
    circles = get_circle_list(image)
    return circles
    dots_list = []
    for circle in circles:
        found_bigger = False
        x, y, r = circle
        for n, dot in enumerate(dots_list):
            if abs(x - dot[0]) < DOT_MIN_DIST and abs(y - dot[1]) < DOT_MIN_DIST:
                if dot[2] > r:
                    found_bigger = True
                else:
                    dots_list.pop(n)
        if not found_bigger:
            dots_list.append((x, y, r))

    return dots_list


def draw_circles(image, circles):
    """
    Returns a copy of the image with circles drawn on it.

    :param image: image on which circles will be drawn
    :param circles: list of circles in (x, y, r) format
    :return: image
    """
    image = image.copy()
    for circle in circles:
        x, y, r = circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 1)
    return image
