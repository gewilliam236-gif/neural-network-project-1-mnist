# Error analysis and model visualization for Part C.
import argparse
import os
from struct import unpack
import gzip

import matplotlib.pyplot as plt
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


def predict_in_batches(model, images, batch_size):
    logits_list = []
    for start in range(0, images.shape[0], batch_size):
        logits_list.append(model(images[start:start + batch_size]))
    return np.vstack(logits_list)


def confusion_matrix(labels, preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        matrix[true_label, pred_label] += 1
    return matrix


def plot_confusion(matrix, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_misclassified(images, labels, preds, save_path, count=16):
    wrong_idx = np.where(labels != preds)[0][:count]
    if wrong_idx.size == 0:
        return
    cols = 4
    rows = int(np.ceil(wrong_idx.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, idx in zip(axes, wrong_idx):
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'T:{labels[idx]} P:{preds[idx]}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_model_weights(model, model_type, save_path):
    if model_type == 'mlp':
        first_linear = next(layer for layer in model.layers if layer.optimizable)
        weights = first_linear.params['W'].T[:16]
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for ax, weight in zip(axes.reshape(-1), weights):
            ax.imshow(weight.reshape(28, 28), cmap='RdBu')
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        return

    first_conv = next(layer for layer in model.layers if layer.__class__.__name__ == 'conv2D')
    kernels = first_conv.params['W'][:, 0]
    cols = min(4, kernels.shape[0])
    rows = int(np.ceil(kernels.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, kernel in zip(axes, kernels):
        ax.imshow(kernel, cmap='RdBu')
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--checkpoint', default='./best_models/mlp_sgd_multistep/best_model.pickle')
    parser.add_argument('--data-dir', default='./dataset/MNIST')
    parser.add_argument('--fig-dir', default='./figs')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    model = nn.models.Model_MLP() if args.model == 'mlp' else nn.models.Model_CNN()
    model.load_model(args.checkpoint)
    images, labels = load_test_mnist(args.data_dir)
    logits = predict_in_batches(model, images, args.batch_size)
    preds = np.argmax(logits, axis=-1)

    matrix = confusion_matrix(labels, preds)
    prefix = os.path.join(args.fig_dir, args.model)
    plot_confusion(matrix, f'{prefix}_confusion_matrix.png')
    plot_misclassified(images, labels, preds, f'{prefix}_misclassified.png')
    plot_model_weights(model, args.model, f'{prefix}_weights.png')

    print(f'accuracy: {(preds == labels).mean():.5f}')
    print(f'figures saved with prefix: {prefix}')


if __name__ == '__main__':
    main()
