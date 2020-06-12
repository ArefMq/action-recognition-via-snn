#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import matplotlib

from gpu_checker import wait_until_gpu_is_free

from utils import plot_metrics
from scnn import SNN
from scnn.optim import RAdam

from data.data_augmentor import batchify
from tools.time_expector import TimeExpector
# from tools.notify import notify

matplotlib.use('Agg')
SAVE_PLOTS = False
time_expector = TimeExpector()

# configurations
batch_size = 4
nb_frame = 40
CACHE_FOLDER_PATH = "/home/aref/dataset/dvs-dataset"
DATASET_FOLDER_PATH = "/home/aref/dataset/dvs-dataset"


def small_dataset_generator():
    original_size = 128
    for __xb, __yb in batchify(
            'train',
            DATASET_FOLDER_PATH,
            CACHE_FOLDER_PATH,
            condition_limit=['natural'],
            batch_size=original_size,
            augmentation=False,
            max_augmentation=1,
            frame=nb_frame
    ):
        break

    _hist = {i: 0 for i in range(12)}
    for i in __yb:
        _hist[i] += 1
    max_value = max(_hist.values())

    aug_xb = []
    aug_yb = []
    for i in range(12):
        idx = np.where(__yb == i)[0][0]
        to_add = max_value - _hist[i]
        for _ in range(to_add):
            aug_yb.append(i)
            aug_xb.append(__xb[idx, :, :, :])
    __yb = np.concatenate([__yb, np.array(aug_yb)])
    __xb = np.concatenate([__xb, np.array(aug_xb)])

    def load_data(trail):
        begin = 0
        end = begin + batch_size
        while end <= __yb.shape[0]:
            yield __xb[begin:end, :, :, :], __yb[begin:end]
            begin = end
            end = begin + batch_size

    return load_data


def medium_dataset_generator():
    original_size = 700
    for __xb, __yb in batchify(
            'train',
            DATASET_FOLDER_PATH,
            CACHE_FOLDER_PATH,
            condition_limit=['natural'],
            batch_size=original_size,
            augmentation=False,
            max_augmentation=1,
            frame=nb_frame
    ):
        break

    _hist = {i: 0 for i in range(12)}
    for i in __yb:
        _hist[i] += 1
    max_value = max(_hist.values())

    aug_xb = []
    aug_yb = []
    for i in range(12):
        idx = np.where(__yb == i)[0][0]
        to_add = max_value - _hist[i]
        for _ in range(to_add):
            aug_yb.append(i)
            aug_xb.append(__xb[idx, :, :, :])

    __yb = np.concatenate([__yb, np.array(aug_yb)])
    __xb = np.concatenate([__xb, np.array(aug_xb)])

    def load_data(trail):
        begin = 0
        end = begin + batch_size
        while end <= __yb.shape[0]:
            yield __xb[begin:end, :, :, :], __yb[begin:end]
            begin = end
            end = begin + batch_size

    return load_data


def large_dataset_generator():
    def load_data(trail):
        if trail.startswith('acc'):
            max_augmentation = 1
            augmentation = False
        else:
            max_augmentation = 2 if trail == 'train' else 1
            augmentation = True

        trail = trail.replace('acc_', '')
        return batchify(
            trail,
            DATASET_FOLDER_PATH,
            CACHE_FOLDER_PATH,
            condition_limit=['natural'],
            batch_size=batch_size,
            augmentation=augmentation,
            max_augmentation=max_augmentation,
            frame=nb_frame
        )

    return load_data


def main(epochs, load_data, opt_adam, lr, weight_decay, read_from_saved, read_from_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    print("Device:", device)

    if read_from_file is not None:
        network = SNN.from_file(read_from_file)
    elif read_from_saved:
        network = SNN.from_file('logs/save_network.net')
    else:
        network = SNN.from_last_checkpoint()
    network.time_expector = time_expector
    # network.notifier = notify # FIXME

    if opt_adam:
        opt = RAdam(network.get_trainable_parameters(lr, weight_decay))
    else:
        opt = torch.optim.SGD(network.get_trainable_parameters(lr, weight_decay), lr=lr, momentum=0.9)

    with open('logs/results.log', 'w+') as f:
        res_metrics = network.fit(
            load_data,
            epochs=epochs,
            optimizer=opt,
            result_file=f,
            save_checkpoints=True
        )
        plot_metrics(res_metrics, save_plot_path='./logs/metrics_C_' if SAVE_PLOTS else None)

    network.save('logs/save_network.net')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, help="Number of iteration to train the network (default=10)",
                        default=10)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("-s", "--small", action="store_true", help="Set dataset size to small")
    group1.add_argument("-m", "--medium", action="store_true", help="Set dataset size to medium")
    group1.add_argument("-x", "--large", action="store_true", help="Set dataset size to large (default)")

    parser.add_argument("-g", "--sgd", action="store_true", help="Use SGD optimizer instead of Adam")
    parser.add_argument("-l", "--lr", type=float, help="Set learning rate (default=0.001)", default=0.001)
    parser.add_argument("-w", "--weight_decay", type=float, help="Set learning rate (default=0.00001)", default=0.00001)

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("-f", "--file", type=str, help="Read network from file")
    group2.add_argument("-u", action="store_true", help="Read network from last saved network")

    args = parser.parse_args()

    if args.small:
        data_loader = small_dataset_generator()
    elif args.medium:
        data_loader = medium_dataset_generator()
    elif args.large:
        data_loader = large_dataset_generator()
    else:
        data_loader = None

    wait_until_gpu_is_free()
    main(args.epochs, data_loader, not args.sgd, args.lr, args.weight_decay, args.u, args.file)
