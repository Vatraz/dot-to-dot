import cv2
import numpy as np

from classify.classify import classify
from image_processing.utils import center, distance_to_point, bounding_rect_centroid, erode_thresh, \
    bounding_rect_list_contains

SHAPE_X, SHAPE_Y = (128, 128)
DIGIT_W_MIN, DIGIT_W_MAX = (6, 10)
DIGIT_H_MIN, DIGIT_H_MAX = (13, 20)
CIRCLE_MAX = 6


def contour_is_circle(bounding_rect):
    """
    Checks if the bounding rect contains a circle.

    :param bounding_rect: tested bounding rect (x, y, width, height)
    :return: whether bounding rect contains a circle
    """
    (x, y, w, h) = bounding_rect
    if abs(h - w) <= 3 and w <= CIRCLE_MAX and h <= CIRCLE_MAX:
        return True
    return False


def get_circle_image(image, circle):
    """
    Returns a segment of the image copy that contains the circle.

    :param image: processed image
    :param circle: circle in (x, y, radius) format
    :return: a fragment of the image containing the circle
    """
    height, width = image.shape[:2]
    x, y, _ = circle
    offset = int(width / 30)
    offset_x_left = offset_x_right = offset_y_up = offset_y_bottom = offset
    temp = np.full((offset * 2, offset * 2), fill_value=255, dtype=np.uint8)

    if width - offset > x > offset and offset < y < height - offset:
        cropped = image[y - offset:y + offset, x - offset:x + offset]
        return cropped

    if x < offset:
        offset_x_left = x
    if x > width - offset:
        offset_x_right = width - x
    if y < offset:
        offset_y_up = y
    if y > height - offset:
        offset_y_bottom = height - y
    cropped = image[y - offset_y_up:y + offset_y_up, x - offset_x_left:x + offset_x_right]
    temp[offset-offset_y_up:offset+offset_y_bottom, offset-offset_x_left:offset+offset_x_right] = cropped
    return temp


def split_digits(bounding_rect, n=2):
    """
    Returns bounding rectangles obtained from the vertically divided bounding rectangle.

    :param bounding_rect: bounding rectangle (x, y, width, height)
    :param i: number of slices
    :return: list of bounding rectangles
    """
    x, y, w, h = bounding_rect
    w_div = int(w//n)

    digit_br_list = []
    for i in range(n):
        digit_br_list.append((x + i*w_div, y, w_div, h))
    return digit_br_list


def erase_center_dot(image, radius, erase_color=0):
    """
    Removes dot in the center of the image, by drawing circle filled with erase_color.

    :param image: image
    :param radius: radius of dot
    :param erase_color: background color - default: 0
    """
    shape = image.shape
    mid_x, mid_y = shape[1]//2, shape[0]//2
    cv2.circle(image, (mid_x, mid_y), radius, color=erase_color, thickness=cv2.FILLED)


def circle_in_center(contours, image_shape, erase_img_list=None, erase_color=0):
    """
    Checks if at least one of the contours is a circle in the image center. If erase_img_list is not None,
    fills the contour with erase_color in each image in the list.

    :param contours: list of contours
    :param image_shape: shape of the image
    :param erase_img_list: list of images from which circles will be removed.
    :param erase_color: background color of images in erase_img_list
    :return: whether a circle is in the image center
    """
    shape_y, shape_x = image_shape[:2]
    found_circle = True
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if int(shape_x * 0.4) < int(x + w / 2) < int(shape_x * 0.6) and \
                int(shape_y * 0.4) < int(y + h / 2) < int(shape_y * 0.6):
            if not contour_is_circle([x, y, w, h]):
                found_circle = False
            if erase_img_list:
                for image in erase_img_list:
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=erase_color, thickness=cv2.FILLED)
    return found_circle


def classify_digit(image, bounding_rect):
    """
    Returns predicted value of a digit in the image at the location described by the bounding rectangle.

    :param image: image
    :param bounding_rect: bounding rectangle (x, y, width, height)
    :return: number
    """
    (x, y, w, h) = bounding_rect
    number_image = center(image[y:y + h, x:x + w], (28, 28))
    digit = classify(number_image)
    return digit


def get_number(digit_br_list, image):
    """
    Returns a number closest to the center of the image

    :param digit_br_list: list of bounding rects that contain digit
    :param image: grayscale image
    :return: number
    """
    if not digit_br_list:
        return None
    circle_point = (image.shape[1] // 2, image.shape[0] // 2)
    if len(digit_br_list) == 1:
        number = classify_digit(image, digit_br_list[0])
    else:
        nearest_idx = int(np.argmin([distance_to_point(br, circle_point) for br in digit_br_list]))
        nearest = digit_br_list.pop(nearest_idx)
        nearest_digit = classify_digit(image, nearest)
        nearest_point = bounding_rect_centroid(nearest)

        second_idx = int(np.argmin([distance_to_point(br, nearest_point) for br in digit_br_list]))
        second = digit_br_list.pop(second_idx)
        second_point = bounding_rect_centroid(second)

        # if it is a two-digit number
        if abs(second_point[1] - nearest_point[1]) < DIGIT_H_MAX and \
                distance_to_point(second, nearest_point) < DIGIT_W_MAX:
            second_digit = classify_digit(image, second)
            if second[0] < nearest[0]:
                number = second_digit * 10 + nearest_digit
            else:
                number = nearest_digit * 10 + second_digit
        # if the digits are too far
        else:
            number = nearest_digit
    return number


def detect_numbers(image, dots):
    """
    Assigns numbers to dots in the list.

    :param image: BGR image
    :param dots: list of dots in (x, y, radius) format
    :return: list of dicts: {'dot': (x, y, radius), 'number': assigned number}
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # invert the image
    image = 255 - image
    # get radius of dots
    dot_radius = int(np.median(np.array(dots)[:, 2]))*3

    numerated_dots = []
    for dot in dots:
        cropped = get_circle_image(image, dot)
        circle_image = cv2.resize(cropped, (SHAPE_X, SHAPE_Y))

        _, thresh = cv2.threshold(circle_image, 60, 255, cv2.THRESH_BINARY)
        erase_center_dot(thresh, dot_radius)

        # Find contours of numbers
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rect_list = [cv2.boundingRect(contour) for contour in contours]
        # cv2.drawContours(circle_image, contours, -1, color=255)

        # create list of digits bounding rectangles
        digit_br_list = []
        for bounding_rect in bounding_rect_list:
            (x, y, w, h) = bounding_rect

            if bounding_rect_list_contains(bounding_rect_list, bounding_rect):
                continue

            # If number contains two digits
            if w > int(DIGIT_W_MAX):
                for br in split_digits(bounding_rect, n=2):
                    digit_br_list.append(br)
            elif w > int(DIGIT_W_MAX * 3/2):
                for br in split_digits(bounding_rect, n=3):
                    digit_br_list.append(br)
            else:
                digit_br_list.append(bounding_rect)

        for bounding_rect in digit_br_list:
            (x, y, w, h) = bounding_rect
            cv2.rectangle(circle_image, (x,y), (x+w,y+h), 255)

        number_img = circle_image
        number = get_number(digit_br_list, number_img)
        # print('--')
        # print(number)
        # cv2.imshow('hihi', number_img)
        # cv2.waitKey(0)

        numerated_dots.append({'dot': dot[:2], 'number': number})
    return numerated_dots
