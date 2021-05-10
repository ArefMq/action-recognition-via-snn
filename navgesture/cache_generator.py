import os
import torch
from . import load_all
from utils import equalizer
import glob
import json
from random import random


def check_folder(**kwargs):
    cache_root = kwargs.get('cache_root_dir', './navgesture')
    addr = '{4}/cache_{0}_{1}_{2}_{3}'.format(kwargs['batch_size'], kwargs['frames'], kwargs.get('max_read_file', 'all'), kwargs['polarity_mode'], cache_root)
    res = os.path.exists(addr)
    return res, addr


def load_cache(force=False, **kwargs):
    res, addr = check_folder(**kwargs)
    if force and res:
        for f in glob.glob(addr + "/*"):
            os.remove(f)
        os.rmdir(addr)
        res = False

    if res:
        return addr
    print('creating cache directory "%s"' % addr)
    os.makedirs(addr)

    counter = 0
    data_list = {}
    for xc, yc in load_all(**kwargs):
        file_name = '{0}/d_{1}.data'.format(addr, counter)
        torch.save(xc, file_name)
        data_list[file_name] = yc

        counter += 1
        print('\rWorking on: %s...  ' % file_name, end='')
        # if counter > 50:
        #     break

    file_name = '{0}/labels.json'.format(addr)
    with open(file_name, 'w') as f:
        json.dump(data_list, f, indent=2)
    return addr


def get_load_access(addr, num_of_classes=None, test_size=.2, classifier_limit=None):
    file_name = '{0}/labels.json'.format(addr)
    with open(file_name, 'r') as f:
        data_list = json.load(f)

    if classifier_limit is not None:
        num_of_classes = len(classifier_limit)
    elif num_of_classes is None:
        num_of_classes = max(data_list.values())+1

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for f in data_list.keys():
        if classifier_limit is not None and data_list[f] not in classifier_limit:
            continue

        if random() > test_size:
            x_train.append(f)
            y_train.append(data_list[f])
        else:
            x_test.append(f)
            y_test.append(data_list[f])

    x_train, y_train = equalizer(x_train, y_train, num_of_classes)
    x_test, y_test = equalizer(x_test, y_test, num_of_classes)

    return {'files': x_train, 'labels': y_train}, {'files': x_test, 'labels': y_test}

# def load_all(trail):
#     pass


def test_main():
    data_loading_config = {
        'batch_size': 16,
        'image_size': (128, 128),
        'image_scale': (.4, .4),
        'frames': 20,
        'frame_len': 5000,
        'polarity_mode': 'accumulative',
        'data_path': './navgesture/user*/*',
        'silent': True,
        'max_read_file': 30,
    }
    # print(check_folder(**data_loading_config))
    # load_cache(force=True, **data_loading_config)
    # print(check_folder(**data_loading_config))

    addr = load_cache(**data_loading_config)
    train_access, test_access = get_load_access(addr)


if __name__ == "__main__":
    test_main()
