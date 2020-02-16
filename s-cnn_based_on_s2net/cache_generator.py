import numpy as np
import os
from collections import OrderedDict

from time_expector import TimeExpector
from notify import notify
from reader import read_file_all

ORIGINAL_IMAGE_HEIGHT = 128
ORIGINAL_IMAGE_WIDTH = 128

GESTURE_MAPPING = {
    0: 'no_gesture',
    1: 'hand_clapping',
    2: 'right_hand_wave',
    3: 'left_hand_wave',
    4: 'right_arm_clockwise',
    5: 'right_arm_counter_clockwise',
    6: 'left_arm_clockwise',
    7: 'left_arm_counter_clockwise',
    8: 'arm_roll',
    9: 'air_drums',
    10: 'air_guitar',
    11: 'other_gestures',
}


def get_label(event_labels, timestamp):
    for t in event_labels.keys():
        if t > timestamp:
            return event_labels[t]
    return 0


def get_label_text(event_labels, timestamp):
    return GESTURE_MAPPING[get_label(event_labels, timestamp)]


def load_trail_files(trail_file, dataset_folder_path):
    file_list = []
    with open(os.path.join(dataset_folder_path, trail_file), 'r') as f:
        for line in f.readlines():
            aedat_file = line.strip()
            if not line:
                continue
            csv_file = aedat_file.replace('.aedat', '_labels.csv')
            file_list.append((
                os.path.join(dataset_folder_path, aedat_file),
                os.path.join(dataset_folder_path, csv_file)
            ))
    return file_list


def read_event_labels(path):
    label_began_time = None
    event_labels = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            label, start, end = line.strip().split(',')
            try:
                label = int(label)
                start = int(start)
                end = int(end)
            except ValueError:
                continue
            if label_began_time is None:
                label_began_time = start

            event_labels[start] = 0
            event_labels[end] = label
    return event_labels, label_began_time


def serialize_events(x_data, y_data, ev_times, image_reduce_factor, frame_length_us):
    x_train = []
    y_train = []

    frames = 0
    max_time = np.max(ev_times)
    current_time = np.min(ev_times)

    while current_time < max_time:
        ev_lb = ev_times > current_time
        ev_ub = ev_times < (current_time + frame_length_us)

        event_x = x_data[ev_lb & ev_ub, :]
        event_y = y_data[ev_lb & ev_ub]

        desired_width = ORIGINAL_IMAGE_WIDTH / image_reduce_factor
        desired_height = ORIGINAL_IMAGE_HEIGHT / image_reduce_factor

        retina = np.zeros([desired_width, desired_height])
        retina[event_x[:, 1] / image_reduce_factor, event_x[:, 0] / image_reduce_factor] = 1

        frame_y = np.round(np.mean(event_y))
        current_time += frame_length_us

        frames += 1
        x_train.append(retina.flatten())
        y_train.append(frame_y)

    return np.array(x_train), np.array(y_train)


def read_and_process_file(file_name, trail, counter, image_reduce_factor, frame_length_us, output_data_path):
    ev_x = []
    ev_y = []
    ev_z = []

    print('---> reading "%s"' % file_name[0])
    event_labels, label_began_time = read_event_labels(file_name[1])
    event_list = read_file_all(file_name[0])

    print('---> pre-processing file...')
    stream_begin_time = None
    stream_end_time = None

    frame_counter = 0
    for e in event_list:
        if e == 'clear':
            frame_counter += 1
            continue

        # calculate time information
        event_time = e['timestamp']
        if stream_begin_time is None:
            stream_begin_time = event_time
        event_corrected_time = event_time - stream_begin_time + label_began_time
        event_label = get_label(event_labels, event_corrected_time)

        # append event
        ev = (e['x'], e['y'])
        ev_x.append(ev)
        ev_y.append(event_label)
        ev_z.append(e['timestamp'])

    print('---> processing file...')
    x_train, y_train = serialize_events(np.array(ev_x), np.array(ev_y), np.array(ev_z), image_reduce_factor,
                                        frame_length_us)

    print('---> saving file...')
    x_train = np.array(x_train, dtype='uint8')
    y_train = np.array(y_train, dtype='uint8')
    np.save(file="%s_%s/x_%s_%d" % (output_data_path, trail, trail, counter), arr=x_train)
    np.save(file="%s_%s/y_%s_%d" % (output_data_path, trail, trail, counter), arr=y_train)


#     return x_train, y_train


def download_dataset(dataset_folder_path):
    print('downloading into "%s"' % dataset_folder_path)
    raise NotImplementedError('can not access data folder.')


def cache_data(trail,
               dataset_folder_path,
               output_data_path,
               image_reduce_factor=2,
               frame_length_us=9900,  # almost the same delay in each frame in the DVS data
               ):
    t = TimeExpector()

    if not os.path.exists(dataset_folder_path):
        download_dataset(dataset_folder_path)
        notify('download complete')

    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    if trail == 'train':
        file_list = load_trail_files('trials_to_train.txt', dataset_folder_path)
    elif trail == 'test':
        file_list = load_trail_files('trials_to_test.txt', dataset_folder_path)

    counter = 0
    for file_name_set in file_list:
        t.tick(iteration_left=len(file_list) - counter)
        counter += 1
        print('     ------------------------- %d out of %d -------------------------' % (counter, len(file_list)))
        read_and_process_file(file_name_set, trail, counter, image_reduce_factor, frame_length_us, output_data_path)
        notify('chunk done: %d / %d' % (counter, len(file_list)))
    notify('all done')


def data_loader(trail, dataset_folder_path, cache_folder_path, condition_limit=None):
    dataset_path = dataset_folder_path + 'cleaned_cache_' + trail
    file_list = load_trail_files('trials_to_%s.txt' % trail, dataset_folder_path)

    for counter, f in enumerate(file_list):
        # FIXME: This line is not satisfactory...
        light_condition = f[0].split('/')[-1].split('.')[0][len('userXX_'):]
        if condition_limit is not None and light_condition not in condition_limit:
            continue

        x_data = np.load(file='%s/x_%s_%d.npy' % (cache_folder_path, trail, counter + 1))
        y_data = np.load(file='%s/y_%s_%d.npy' % (cache_folder_path, trail, counter + 1))

        yield x_data, y_data


__all__ = ['data_loader', 'cache_data', 'GESTURE_MAPPING']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from random import randint

    CACHE_FOLDER_PATH = "/Users/aref/dvs-dataset/Cached/"
    DATASET_FOLDER_PATH = "/Users/aref/dvs-dataset/DvsGesture/"
    FRAME_TO_SHOW = 100

    plt.figure(0)
    plt.ion()
    plt.show()

    img = np.zeros((64, 64))
    for x, y in data_loader('test', DATASET_FOLDER_PATH, CACHE_FOLDER_PATH, condition_limit=['natural']):
        idx = randint(0, x.shape[0] - FRAME_TO_SHOW - 1)
        for i in range(FRAME_TO_SHOW):
            img *= 0.7
            img += np.reshape(x[idx + i, :], (64, 64))
            label = y[idx + i]

            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.draw()
            plt.title('%d - %s' % (i, GESTURE_MAPPING[label]))
            plt.pause(0.00001)
            plt.clf()
        break
    print('done')
