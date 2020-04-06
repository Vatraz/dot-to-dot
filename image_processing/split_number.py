from image_processing.utils_img import get_vertical_indents, indents_exc_thresh


def get_number_image_splits(image, n=2):
    width = image.shape[1]
    indents = get_vertical_indents(image)

    splits = []
    for i in range(1, n):
        estimated = int(width * i/n)
        cur_max = indents[estimated]
        if indents[estimated + 1] > cur_max:
            split = estimated + 1
        elif indents[estimated - 1] > cur_max:
            split = estimated - 1
        else:
            split = estimated
        splits.append(split)

    return splits


def split_br(bounding_rect, splits):
    x, y, w, h = bounding_rect
    n_br = len(splits) + 1
    edges = [x] + [split+x for split in splits] + [x+w]

    br_list = []
    for i in range(n_br):
        br = (edges[i], y, edges[i+1] - edges[i], h)
        br_list.append(br)

    return br_list


def guess_digits_number(image, threshold=0.9):
    h, w = image.shape[:2]
    wh_ratio = w/h
    indents = indents_exc_thresh(image, threshold=threshold, trim=1)
    n_indents = len(indents)
    if wh_ratio > 1.5 and n_indents >= 2 or wh_ratio > 1.7:
        number = 3
    elif wh_ratio >= 1.3 or n_indents >= 1 and wh_ratio >= 1.2:
        number = 2
    else:
        number = 1

    return number
