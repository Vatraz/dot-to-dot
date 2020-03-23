import cv2
import numpy as np
import glob


def erase_circles(image, circles):
    color = (255, 255, 255) if image.shape[2] > 1 else 255

    image = image.copy()
    for circle in circles:
        x, y, r = circle
        cv2.circle(image, (x+1, y+1), r, color=color, thickness=cv2.FILLED)
    return image


def circle_filled(image, x, y, r):
    offset = r + 3
    if (image.shape[1]< x+offset or x-offset < 0) or (image.shape[0] < y+offset < 0 or y-offset < 0):
        return False
    img_crop = image[y-offset:y+1+offset, x-offset:x+1+offset]

    mask_circle = np.zeros((img_crop.shape[0], img_crop.shape[1]), dtype=np.uint8)
    cv2.circle(mask_circle, (offset, offset), r, color=255, thickness=cv2.FILLED)
    mask_circle_inv = np.bitwise_not(mask_circle)
    des_circle_field = np.count_nonzero(mask_circle)
    des_circle_inv_field = img_crop.shape[0] * img_crop.shape[1] - des_circle_field

    found_circle_field = np.count_nonzero(cv2.bitwise_and(img_crop, img_crop, mask=mask_circle))
    found_circle_inv_field = np.count_nonzero(cv2.bitwise_and(img_crop, img_crop, mask=mask_circle_inv))

    if found_circle_field > int(0.7 * des_circle_field) and found_circle_inv_field < int(des_circle_inv_field * 0.3):
        return True
    else:
        return False


def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (3, 3))
    _, gray_thresh = cv2.threshold(gray_blur, 125, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('gray', gray)
    circles = cv2.HoughCircles(gray_thresh, method=cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=50, param2=4,
                               minRadius=3, maxRadius=7)
    circle_list = []
    if circles is None:
        return circle_list

    circles = np.int16(np.around(circles))

    for circle in circles[0, :]:
        found_bigger = False
        x, y, r = circle[0], circle[1], circle[2]
        if not circle_filled(gray_thresh, x, y, r):
            continue
        for n, circle in enumerate(circle_list):
            if abs(x - circle[0]) < 10 and abs(y - circle[1]) < 10:
                if circle[2] > r:
                    found_bigger = True
                else:
                    circle_list.pop(n)
        if not found_bigger:
            circle_list.append((x, y, r))

    # radius_median = np.median(np.asarray(circle_list))

    return circle_list


def draw_circles(image, circles):
    image = image
    for circle in circles:
        x, y, r = circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 1)
    cv2.imshow("Detected Circle", image)
    cv2.waitKey(0)

#
# def nadus_dataset():
#     itemy = glob.glob('./img/*')
#     number = 1
#     for item in itemy:
#         img = cv2.imread(item)
#         circles = detect_circles(img)
#         width, height, _ = img.shape
#         s = int(width / 30)
#         for circle in circles:
#             x, y, _ = circle
#             try:
#                 temp = img[y-s:y+s, x-s:x+s]
#                 cv2.imwrite(f'out\\{number}.jpg', temp)
#                 number += 1
#             except:
#                 pass
#
#
# def resampluj():
#     itemy = glob.glob('./out/*')
#     number = 1
#     for item in itemy:
#         img = cv2.imread(item)
#         temp = cv2.resize(img, (64, 64))
#         cv2.imwrite(f'data\\{number}.jpg', temp)
#         number += 1


if __name__ == '__main__':
    pass
    # resampluj()
    # nadus_dataset()
    # i = cv2.imread(itemy[0])
    # cv2.imshow('hihi', i)
    # cv2.waitKey()
    # print(itemy)
    # nadus_dataset()