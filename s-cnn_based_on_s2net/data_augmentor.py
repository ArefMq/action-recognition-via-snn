import numpy as np
import matplotlib.pyplot as plt
from random import randint
from cache_generator import data_loader, GESTURE_MAPPING


def data_augment(*args, **kwargs):
    max_augmentation = kwargs.pop('max_augmentation', 1)
    aug_enabled = kwargs.pop('augmentation', True)
    aug_x_offset = kwargs.pop('aug_x_offset', 10)
    aug_y_offset = kwargs.pop('aug_y_offset', 3)
    aug_frame_offset = kwargs.pop('aug_f_offset', 10)
    frame = kwargs.pop('frame', 100)

    for data_x, data_y in data_loader(*args, **kwargs):
        label_histogram = {i: 0 for i in GESTURE_MAPPING.keys()}
        max_hist = None
        output_x = []
        output_y = []

        last_label = None
        buffer_x = []

        # here we augment data until all classes have the same amount of data
        while max_hist is None or any([i < max_hist for _, i in label_histogram.items()]):
            if aug_enabled and max_hist is not None:
                x_offset = randint(-aug_x_offset, aug_x_offset)
                y_offset = randint(-aug_y_offset, aug_y_offset)
                frame_offset = randint(0, aug_frame_offset)
            else:
                x_offset = 0
                y_offset = 0
                frame_offset = 0

            for i in range(frame_offset, data_x.shape[0]):
                current_x = np.reshape(data_x[i], (64, 64))
                current_y = data_y[i]

                # image augmentation
                current_x = np.roll(current_x, x_offset, axis=0)
                current_x = np.roll(current_x, y_offset, axis=1)

                if last_label is None:
                    last_label = current_y
                elif len(buffer_x) >= frame:
                    if max_hist is None or label_histogram[current_y] < max_hist:
                        output_x.append(np.array(buffer_x))
                        output_y.append(current_y)
                        label_histogram[current_y] += 1
                    buffer_x = []
                elif last_label != current_y:
                    buffer_x = []
                    last_label = current_y
                buffer_x.append(current_x)

            # if the augmentation is false, then we ignore adding any more data
            if not aug_enabled:
                break

            if max_hist is None:
                max_hist = label_histogram[max(label_histogram, key=label_histogram.get)] * max_augmentation

        output_x = np.array(output_x)
        output_y = np.array(output_y)

        shuffle_indices = np.random.permutation(output_x.shape[0])
        for i in shuffle_indices:
            yield output_x[i, ...], output_y[i]


def batchify(*args, **kwargs):
    batch_size = kwargs.pop('batch_size', 16)

    batch_x = []
    batch_y = []
    for x, y in data_augment(*args, **kwargs):
        batch_x.append(x)
        batch_y.append(y)

        if len(batch_x) >= batch_size:
            yield np.array(batch_x), np.array(batch_y)
            batch_x = []
            batch_y = []


if __name__ == "__main__":
    CACHE_FOLDER_PATH = "/Users/aref/dvs-dataset/Cached/"
    DATASET_FOLDER_PATH = "/Users/aref/dvs-dataset/DvsGesture/"
    FRAME = 100

    plt.figure(0)
    plt.ion()
    plt.show()

    imager = 5
    img = np.zeros((64, 64))
    for x, y in data_augment('test', DATASET_FOLDER_PATH, CACHE_FOLDER_PATH, frame=FRAME, condition_limit=['natural']):
        for i in range(FRAME):
            img *= 0.7
            img += x[i, :, :]

            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.draw()
            plt.title('%d - %s' % (i, GESTURE_MAPPING[y]))
            plt.pause(0.00001)
            plt.clf()

        if imager == 0:
            break
        imager -= 1
    print('done')
