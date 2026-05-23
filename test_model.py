import argparse
import os
from struct import unpack
import gzip

import numpy as np

import mynn as nn


def load_test_mnist(data_dir='./dataset/MNIST'):
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--checkpoint', default='./best_models/mlp_sgd_multistep/best_model.pickle')
    parser.add_argument('--data-dir', default='./dataset/MNIST')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    model = nn.models.Model_MLP() if args.model == 'mlp' else nn.models.Model_CNN()
    model.load_model(args.checkpoint)

    test_imgs, test_labs = load_test_mnist(args.data_dir)
    total_correct = 0
    total = 0
    for start in range(0, test_imgs.shape[0], args.batch_size):
        batch_X = test_imgs[start:start + args.batch_size]
        batch_y = test_labs[start:start + args.batch_size]
        logits = model(batch_X)
        total_correct += (np.argmax(logits, axis=-1) == batch_y).sum()
        total += batch_X.shape[0]

    print(f'test accuracy: {total_correct / total:.5f}')


if __name__ == '__main__':
    main()
