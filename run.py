import argparse
import cv2
from time import strftime

from classify.classify import Classifier
from image_processing.dots_detection import detect_dots
from image_processing.number_detection import detect_numbers


def run(model_path, image_path, out_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    dots = detect_dots(image)
    classifier = Classifier(model_path)
    num_dots = detect_numbers(image, dots, classifier)

    num_dots.sort(key=lambda dot: dot['number'])
    for i in range(1, len(num_dots)):
        cv2.line(image, num_dots[i-1]['dot'], num_dots[i]['dot'], (255, 10, 10), 2)

    cv2.imwrite(f'{out_path}out_{strftime("%m_%d__%H_%M")}.png', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help="a model path", type=str, default='classify/models/svm.cpickle')
    parser.add_argument('--image', '-i', help="an input image path", type=str, default='img/1.jpg')
    parser.add_argument('--out', '-o', help="a output image path", type=str, default='img/')

    args = parser.parse_args()
    model_path = args.model
    image_path = args.image
    out_path = args.out

    run(model_path, image_path, out_path)