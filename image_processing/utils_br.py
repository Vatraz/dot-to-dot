import numpy as np

from image_processing.utils_img import vertical_indent, horizontal_indent


def bounding_rect_list_contains(bounding_rect_list, bounding_rect):
    x, y, w, h = bounding_rect
    for br in bounding_rect_list:
        x_br, y_br, w_br, h_br = br
        if br == bounding_rect:
            continue
        if x_br <= x <= x_br + w_br - w and y_br <= y <= y_br + h_br - h:
            return True
    return False


def bounding_rect_centroid(bounding_rect):
    (x, y, w, h) = bounding_rect
    x_br, y_br = int(x + w / 2), int(y + h / 2)
    return x_br, y_br


def bounding_rect_rel_pos(bounding_rect, point):
    x_br, y_br = bounding_rect_centroid(bounding_rect)
    x_p, y_p = point[0],  point[1]
    return (x_br-x_p, y_br-y_p)


def distance_to_point(bounding_rect, point):
    x, y = bounding_rect_rel_pos(bounding_rect, point)
    distance = np.sqrt(x**2 + y**2)
    return distance


def crop_bounding_rect(image, bounding_rect):
    x, y, w, h = bounding_rect
    return image[y:y+h, x:x+w]


def filter_bounding_rect(image, bounding_rect):
    br_filtered = bounding_rect
    x, y, w, h = bounding_rect
    l_offset = r_offset = u_offset = b_offset = 0
    while True:
        img = crop_bounding_rect(image, br_filtered)
        if 0 in img.shape[:2]:
            return None
        h_img, w_img = img.shape[:2]
        h_thresh = h_img - 1
        changed = False
        # left, right offset
        if sum(vertical_indent(img, 0)) >= h_thresh:
            l_offset += 1
            changed = True
        if sum(vertical_indent(img, w_img - 1)) >= h_thresh:
            r_offset += 1
            changed = True
        # upper, bottom offset
        if sum(horizontal_indent(img, 0)) >= w_img:
            u_offset += 1
            changed = True
        if sum(horizontal_indent(img, h_img - 1)) >= w_img:
            b_offset += 1
            changed = True
        # stop filtering if there was no change
        if not changed:
            break
        # update output bounding rectangle
        br_filtered = (x+l_offset, y+u_offset, w - (l_offset+r_offset), h - (u_offset+b_offset))
    return br_filtered