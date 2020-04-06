import cv2
import numpy as np

from image_processing.split_number import guess_digits_number, get_number_image_splits, split_br
from image_processing.utils_br import distance_br_to_point, bounding_rect_centroid, bounding_rect_list_contains, \
    crop_bounding_rect, filter_bounding_rect
from image_processing.utils_img import center, normalize


def get_circle_image(image, circle):
    """
    Returns a segment of the image copy that contains the circle.

    :param image: processed image
    :param circle: circle in (x, y, radius) format
    :return: a fragment of the image containing the circle
    """
    height, width = image.shape[:2]
    x, y, _ = circle
    offset = width // 40
    offset_x_left = offset_x_right = offset_y_up = offset_y_bottom = offset
    circle_img = np.full((offset * 2, offset * 2), fill_value=255, dtype=np.uint8)

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
    circle_img[offset-offset_y_up:offset+offset_y_bottom, offset-offset_x_left:offset+offset_x_right] = cropped
    return circle_img


def classify_digit_br(image, bounding_rect, classifier):
    """
    Returns predicted value of a digit in the image at the location described by the bounding rectangle.

    :param image: image
    :param bounding_rect: bounding rectangle (x, y, width, height)
    :param classifier: classifier
    :return: number
    """
    img = crop_bounding_rect(image, bounding_rect)
    img = normalize(img)
    number_image = center(img, (16, 16))
    digit = classifier.predict(number_image)
    return digit


def filter_numbers_in_row(br_ref, br_list, thresh=2):
    """
    Returns list of bounding rectangles that are in row with ref bounding rectangle.

    :param br_ref: reference bounding rectangle
    :param br_list: list of bounding rectangles
    :param thresh: threshold in y-axis - number of pixels
    :return: list of bounding rectangles
    """
    x, y = bounding_rect_centroid(br_ref)
    row = []
    for br in br_list:
        if y - thresh <= bounding_rect_centroid(br)[1] <= y + thresh:
            row.append(br)
    row.append(br_ref)
    return row


def get_center_number_br_list(image, br_list):
    circle_point = (image.shape[1] // 2, image.shape[0] // 2)
    candidates = br_list.copy()
    nearest_idx = np.argmin([distance_br_to_point(br, circle_point) for br in candidates])
    nearest = candidates.pop(nearest_idx)

    # reduce candidates to digits in row with nearest
    candidates = filter_numbers_in_row(nearest, candidates)

    # sort digits in row
    candidates.sort(key=lambda br: br[0])

    # select nearest number series
    number_br_list = []
    prev_br = candidates[0]
    nearest_series = False
    max_digit_distance = image.shape[1] // 10
    for br in candidates:
        if br[0] - prev_br[0] < max_digit_distance:
            number_br_list.append(br)
            # if found series contains nearest br
            if br[0] == nearest[0]:
                nearest_series = True
        elif not nearest_series:
            number_br_list = [br]
        else:
            break
        prev_br = br

    return number_br_list


def get_number(image, br_list, classifier):
    """
    Returns a number closest to the center of the image.

    :param br_list: list of bounding rectangles that contain the number
    :param image: grayscale image
    :param classifier: classifier
    :return: number
    """
    # classify row digits
    digit_list = [classify_digit_br(image, br, classifier) for br in br_list]

    number = 0
    for n, digit in enumerate(reversed(digit_list)):
        number += digit * 10**n
    return number


def get_dot_number(image, dot, dot_radius, classifier):
    circle_image = get_circle_image(image, dot)

    _, thresh_br = cv2.threshold(circle_image, 70, 255, cv2.THRESH_BINARY)
    _, thresh_number = cv2.threshold(circle_image, 50, 255, cv2.THRESH_TOZERO)

    # Find contours of numbers
    contours, _ = cv2.findContours(thresh_br, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rect_list = [cv2.boundingRect(contour) for contour in contours]

    # create list of bounding rectangles of digits
    found_digits_br_list = []
    for bounding_rect in bounding_rect_list:
        h = bounding_rect[3]
        if h <= dot_radius * 2 or bounding_rect_list_contains(bounding_rect_list, bounding_rect):
            continue

        bounding_rect = filter_bounding_rect(thresh_number, bounding_rect)
        if not bounding_rect:
            continue

        image_br = crop_bounding_rect(thresh_number, bounding_rect)

        n_digits = guess_digits_number(image_br)
        if n_digits == 1:
            found_digits_br_list.append(bounding_rect)
        else:
            splits = get_number_image_splits(image_br, n=n_digits)
            split_br_list = split_br(bounding_rect, splits)
            found_digits_br_list.extend(split_br_list)

    number_br_list = get_center_number_br_list(thresh_number, found_digits_br_list)
    number = get_number(thresh_number, number_br_list, classifier)
    return number


def detect_numbers(image, dots, classifier):
    """
    Assigns numbers to dots in the list.

    :param image: BGR image
    :param dots: list of dots in (x, y, radius) format
    :param classifier: classifier
    :return: list of dicts: {'dot': (x, y, radius), 'number': assigned number}
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = 255 - image

    dot_radius = int(np.median(np.array(dots)[:, 2]))

    for dot in dots:
        cv2.circle(image, tuple(dot[:2]), int(dot_radius * 1.5), color=0, thickness=cv2.FILLED)

    numerated_dots = []
    for dot in dots:
        number = get_dot_number(image, dot, dot_radius, classifier)
        numerated_dots.append({'dot': tuple(dot[:2]), 'number': number})

    return numerated_dots
