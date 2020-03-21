import numpy as np
import argparse
import time
import cv2


shape_x, shape_y = (128, 128)
digit_w_min, digit_w_max = (6, 50)
digit_h_min, digit_h_max = (17, 35)

def erase_circle(image):
    # shape = image.shape[0]
    # color = 255
    # cv2.rectangle(image, (shape, y-r), (x+r, y+r), color=color, thickness=cv2.FILLED)
    return image


def contour_is_circle(bounding_rect):
    (x, y, w, h) = bounding_rect
    if abs(h - w) < 2 and w <= digit_w_min and h < digit_h_min:
        return True
    return False


def detect_numbers(image, circles):
    image = image.copy()
    width, height, _ = image.shape
    offset = int(width / 30)
    for circle in circles:
        break_loop = False
        x, y, _ = circle
        try:
            temp = cv2.resize(image[y - offset:y + offset, x - offset:x + offset], (shape_x, shape_y))
            gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            gray = erase_circle(gray)
            gray_blur = cv2.blur(gray, (3, 3))

            _, gray_thresh = cv2.threshold(gray_blur, 160, 255, cv2.THRESH_BINARY_INV)

            cv2.imshow('thrash', gray_thresh)

            # gray_canny = cv2.Canny(gray_thresh, 0, 50)

            (contours, _) = cv2.findContours(gray_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if int(shape_x*0.4) < int(x+w/2) < int(shape_x*0.6) and int(shape_y*0.4) < int(y+h/2) < int(shape_y*0.6):
                    if contour_is_circle([x, y, w, h]):
                        break_loop = True
                    cv2.rectangle(gray_thresh, (x,y), (x+w, y+h), color=0, thickness=cv2.FILLED)
            if break_loop:
                continue
            (contours, _) = cv2.findContours(gray_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(temp, contours, -1, color=(21, 32, 255))

            cv2.imshow('gray', gray_thresh)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                print(x, y, w, h)
                if w > digit_w_max or h > digit_h_max or h < digit_h_min or w < digit_w_min :
                    print('no ladnie')
                    continue
                if contour_is_circle([x, y, w, h]):
                    continue
                print('to spoko')
                cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow('okienko :>', temp)
            cv2.waitKey()

            # cv2.imwrite(f'out\\{number}.jpg', temp)
            # number += 1
        except:
            print('tu mi nie wyszlo')
            pass
    cv2.imshow('okienko :>', image)
    cv2.waitKey()



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