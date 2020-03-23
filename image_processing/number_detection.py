import numpy as np
import argparse
import time
import cv2
from classify import classify

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


def circle_bounding_rect(contours, erase_img_list=(), erase_color=0):
    found_circle = True
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if int(shape_x * 0.4) < int(x + w / 2) < int(shape_x * 0.6) and int(shape_y * 0.4) < int(y + h / 2) < int(
                shape_y * 0.6):
            print('JESTEM W SRODECZKU :>')
            print((x, y, w, h))
            if not contour_is_circle([x, y, w, h]):
                print('I TO NIE JEST  KUŁECZKO :<')
                found_circle = False
            for image in erase_img_list:
                cv2.rectangle(image, (x, y), (x + w, y + h), color=erase_color, thickness=cv2.FILLED)
    return found_circle


def detect_numbers(image, circles):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for circle in circles:
        x, y, _ = circle

        cropped = get_circle_image(image, circle)
        circle_image = cv2.resize(cropped, (shape_x, shape_y))
        # gray_blur = cv2.blur(gray, (3, 3))

        _, gray_thresh = cv2.threshold(circle_image, 160, 255, cv2.THRESH_BINARY_INV)

        # cv2.imshow('to to', circle_image)

        # gray_canny = cv2.Canny(gray_thresh, 0, 50)
        print('==========================================================================================')

        (contours, _) = cv2.findContours(gray_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not circle_bounding_rect(contours, erase_img_list=(gray_thresh), erase_color=0):
            continue

        (contours, _) = cv2.findContours(gray_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bounding_rects = [cv2.boundingRect(contour) for contour in contours]
        # cv2.drawContours(circle_image, contours, -1, color=(21, 32, 255))
        for bounding_rect in bounding_rects:
            (x, y, w, h) = bounding_rect
            if w > digit_w_max or h > digit_h_max or h < digit_h_min or w < digit_w_min \
                    or contour_is_circle([x, y, w, h]):
                continue

            cv2.rectangle(circle_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            number = center(gray_thresh[y:y+h, x:x+w], (20, 20))
            digit = classify(number)
            print('mysle ze to jest ', digit)
            cv2.imshow('okienko :>', number)
            cv2.waitKey()
        # cv2.imshow('haha', circle_image)
        # cv2.waitKey()
        # cv2.imwrite(f'out\\{number}.jpg', temp)
        print('taka o ', bounding_rects)
    pass


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

    h, w = image.shape[:2]
    if h > w:
        image = resize(image, height=des_shape[0])
    else:
        image = resize(image, width=des_shape[1])

    offset_x = (des_shape[1] - image.shape[1]) // 2
    offset_y = (des_shape[0] - image.shape[0]) // 2

    des_image[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1]] = image
    return des_image


if __name__ == '__main__':
    image2 = cv2.imread('img/image2.jpg')
    image = cv2.imread('img/image2.jpg', 0)
    _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('tr', thresh)
    image = cv2.blur(image, (3,3))

    # lines = cv2.HoughLinesP(image, 1000, np.pi / 180, 80, maxLineGap=10)
    # print(lines)
    # for line in lines:
    #     print(line)
    #
    #     x1, y1, x2, y2 = line[0]
    #     if (np.sqrt((x1-x2)**2 + (y1-y2)**2)>50):
    #         continue
    #     cv2.line(image2, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # cv2.imshow('okienko :>', image2)
    cv2.waitKey()
    # orig = image.copy()
    # (H, W) = image.shape[:2]