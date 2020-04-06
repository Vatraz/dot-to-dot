import argparse
from classify.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help="a model path", type=str, default='classify/models/svm.cpickle')
    parser.add_argument('--dataset', '-d', help="a dataset .csv path", type=str, default='classify/data/digits16x16.csv')

    args = parser.parse_args()
    model_path = args.model
    dataset_path = args.dataset

    train(model_path, dataset_path)
