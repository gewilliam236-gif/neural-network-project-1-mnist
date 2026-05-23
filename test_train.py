# Train MLP/CNN models on MNIST with reproducible settings.
import argparse
import os
from struct import unpack
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot


def load_mnist(data_dir='./dataset/MNIST'):
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels


def build_model(args, input_dim, num_classes):
    if args.model == 'mlp':
        return nn.models.Model_MLP(
            [input_dim, args.hidden_dim, num_classes],
            'ReLU',
            [args.weight_decay, args.weight_decay]
        )
    return nn.models.Model_CNN(
        input_shape=(1, 28, 28),
        num_classes=num_classes,
        weight_decay=args.weight_decay > 0,
        weight_decay_lambda=args.weight_decay
    )


def build_optimizer(args, model):
    if args.optimizer == 'momentum':
        return nn.optimizer.MomentGD(init_lr=args.lr, model=model, mu=args.momentum)
    return nn.optimizer.SGD(init_lr=args.lr, model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--optimizer', choices=['sgd', 'momentum'], default='sgd')
    parser.add_argument('--scheduler', choices=['none', 'multistep', 'exponential'], default='multistep')
    parser.add_argument('--data-dir', default='./dataset/MNIST')
    parser.add_argument('--save-dir', default='./best_models')
    parser.add_argument('--fig-dir', default='./figs')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--eval-batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.06)
    parser.add_argument('--hidden-dim', type=int, default=600)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--valid-size', type=int, default=10000)
    parser.add_argument('--sample-limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=309)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    images, labels = load_mnist(args.data_dir)
    idx = np.random.permutation(np.arange(images.shape[0]))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

    images = images[idx]
    labels = labels[idx]
    valid_imgs = images[:args.valid_size]
    valid_labs = labels[:args.valid_size]
    train_imgs = images[args.valid_size:]
    train_labs = labels[args.valid_size:]

    if args.sample_limit > 0:
        train_imgs = train_imgs[:args.sample_limit]
        train_labs = train_labs[:args.sample_limit]

    model = build_model(args, train_imgs.shape[-1], int(labels.max()) + 1)
    optimizer = build_optimizer(args, model)

    scheduler = None
    if args.scheduler == 'multistep':
        scheduler = nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[800, 2400, 4000],
            gamma=0.5
        )
    elif args.scheduler == 'exponential':
        scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=int(labels.max()) + 1)
    runner = nn.runner.RunnerM(
        model,
        optimizer,
        nn.metric.accuracy,
        loss_fn,
        batch_size=args.batch_size,
        scheduler=scheduler
    )

    run_name = f'{args.model}_{args.optimizer}_{args.scheduler}'
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=args.epochs,
        log_iters=args.eval_interval,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        save_dir=save_dir
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_tight_layout(True)
    plot(runner, axes)
    fig.savefig(os.path.join(args.fig_dir, f'{run_name}_learning_curve.png'), dpi=160)

    print(f'best validation accuracy: {runner.best_score:.5f}')
    print(f'model saved to: {os.path.join(save_dir, "best_model.pickle")}')
    print(f'curve saved to: {os.path.join(args.fig_dir, f"{run_name}_learning_curve.png")}')


if __name__ == '__main__':
    main()

