import cv2
import numpy as np

from classify.classify import classify
from image_processing.utils import center, bounding_rect_rel_pos

shape_x, shape_y = (128, 128)
digit_w_min, digit_w_max = (6, 55)
digit_h_min, digit_h_max = (17, 65)
circle_max = 20


def contour_is_circle(bounding_rect):
    (x, y, w, h) = bounding_rect
    if abs(h - w) <= 3 and w <= circle_max and h <= circle_max:
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
        if int(shape_x * 0.4) < int(x + w / 2) < int(shape_x * 0.6) and int(shape_y * 0.4) < int(y + h / 2) < int(
                shape_y * 0.6):
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


def detect_numbers(image, circles):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for circle in circles:
        cropped = get_circle_image(image, circle)
        circle_image = cv2.resize(cropped, (shape_x, shape_y))

        _, thresh = cv2.threshold(circle_image, 160, 255, cv2.THRESH_BINARY_INV)

        # Erase object in the centre of the image, and check if it is a circle
        (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not circle_bounding_rect(contours, erase_img_list=(thresh), erase_color=0):
            continue

        (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rects = [cv2.boundingRect(contour) for contour in contours]
        found_numbers = []
        # cv2.drawContours(circle_image, contours, -1, color=(21, 32, 255))
        for bounding_rect in bounding_rects:
            (x, y, w, h) = bounding_rect
            # Ignore if found object is not a number
            if w > digit_w_max or h > digit_h_max or h < digit_h_min or w < digit_w_min \
                    or contour_is_circle([x, y, w, h]):
                continue
            # If number contains two digits
            if w > int(digit_w_max * 2/3):
                for br in split_digits(bounding_rect):
                    found_numbers.append([
                        classify_digit(thresh, br),
                        bounding_rect_rel_pos(br, circle[:2]),
                    ])
            else:
                found_numbers.append([
                        classify_digit(thresh, bounding_rect),
                        bounding_rect_rel_pos(bounding_rect, circle[:2]),
                ])

            pass
            # check value of found digit
            # cv2.rectangle(circle_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # number_image = center(thresh[y:y+h, x:x+w], (20, 20))
            # digit = classify(number_image)
            # print('->> ', digit)
            # cv2.imshow(':>', number_image)
            # cv2.waitKey()
