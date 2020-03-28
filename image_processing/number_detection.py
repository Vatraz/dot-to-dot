import cv2
import numpy as np

from classify.classify import classify
from image_processing.utils import center, distance_to_point, bounding_rect_centroid

SHAPE_X, SHAPE_Y = (128, 128)
DIGIT_W_MIN, DIGIT_W_MAX = (6, 55)
DIGIT_H_MIN, DIGIT_H_MAX = (17, 65)
CIRCLE_MAX = 20


def contour_is_circle(bounding_rect):
    (x, y, w, h) = bounding_rect
    if abs(h - w) <= 3 and w <= CIRCLE_MAX and h <= CIRCLE_MAX:
        return True
    return False


def get_circle_image(image, circle):
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


def split_digits(bounding_rect):
    (x, y, w, h) = bounding_rect
    w_div = int(w/2)
    digit_br_list = [(x, y, w_div, h), (x+w_div, y, w_div, h)]
    return digit_br_list


def circle_bounding_rect(contours, erase_img_list=(), erase_color=0):
    found_circle = True
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if int(SHAPE_X * 0.4) < int(x + w / 2) < int(SHAPE_X * 0.6) and int(SHAPE_Y * 0.4) < int(y + h / 2) < int(
                SHAPE_Y * 0.6):
            if not contour_is_circle([x, y, w, h]):
                found_circle = False
            for image in erase_img_list:
                cv2.rectangle(image, (x, y), (x + w, y + h), color=erase_color, thickness=cv2.FILLED)
    return found_circle


def classify_digit(image, bounding_rect):
    (x, y, w, h) = bounding_rect
    number_image = center(image[y:y + h, x:x + w], (20, 20))
    digit = classify(number_image)
    return digit


def get_number(digit_br_list, image):
    if not digit_br_list:
        return None
    circle_point = (SHAPE_X // 2, SHAPE_Y // 2)
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
        if all([abs(second_point[1] - nearest_point[1]) < DIGIT_H_MAX,
                distance_to_point(second, nearest_point) < DIGIT_W_MAX]):
            second_digit = classify_digit(image, second)
            if second[0] < nearest[0]:
                number = second_digit * 10 + nearest_digit
            else:
                number = nearest_digit * 10 + second_digit
        else:
            number = nearest_digit
    return number


def detect_numbers(image, circles):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    numerated_circles = []
    for circle in circles:
        cropped = get_circle_image(image, circle)
        circle_image = cv2.resize(cropped, (SHAPE_X, SHAPE_Y))

        _, thresh = cv2.threshold(circle_image, 160, 255, cv2.THRESH_BINARY_INV)

        # Erase object in the centre of the image, and check if it is a circle
        (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not circle_bounding_rect(contours, erase_img_list=[thresh], erase_color=0):
            continue

        # Find contours of numbers
        (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rects = [cv2.boundingRect(contour) for contour in contours]
        # cv2.drawContours(circle_image, contours, -1, color=(21, 32, 255))

        # create list of numbers
        digit_br_list = []
        for bounding_rect in bounding_rects:
            (x, y, w, h) = bounding_rect
            # Ignore if found object is not a number
            if w > DIGIT_W_MAX or h > DIGIT_H_MAX or h < DIGIT_H_MIN or w < DIGIT_W_MIN \
                    or contour_is_circle([x, y, w, h]):
                continue
            # If number contains two digits
            if w > int(DIGIT_W_MAX * 2 / 3):
                for br in split_digits(bounding_rect):
                    digit_br_list.append(br)
            else:
                digit_br_list.append(bounding_rect)

        number = get_number(digit_br_list, thresh)

        numerated_circles.append({'circle': circle[:2], 'number': number})
    return numerated_circles
