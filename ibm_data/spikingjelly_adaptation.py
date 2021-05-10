from torch.utils.data import Dataset

import os
import glob
import time
import struct
import zipfile
import threading
import multiprocessing

import torch
import numpy as np



labels_dict = {
'hand_clapping': 1,
'right_hand_wave': 2,
'left_hand_wave': 3,
'right_arm_clockwise': 4,
'right_arm_counter_clockwise': 5,
'left_arm_clockwise': 6,
'left_arm_counter_clockwise': 7,
'arm_roll': 8,
'air_drums': 9,
'air_guitar': 10,
'other_gestures': 11
}  # gesture_mapping.csv
# url md5
resource = ['https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '8a5c71fb11e24e5ca5b11866ca6c00a1']



class FunctionThread(threading.Thread):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs
    def run(self):
        self.f(*self.args, **self.kwargs)        
        

class EventsFramesDatasetBase(Dataset):
    @staticmethod
    def get_wh():
        raise NotImplementedError

    @staticmethod
    def read_bin(file_name: str):
        raise NotImplementedError

    @staticmethod
    def get_events_item(file_name):
        raise NotImplementedError

    @staticmethod
    def get_frames_item(file_name):
        raise NotImplementedError

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        raise NotImplementedError

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str, normalization: str or None):
        raise NotImplementedError



def integrate_events_to_frames(events, height, width, frames_num=10, split_by='time', normalization=None):
    frames = np.zeros(shape=[frames_num, 2, height * width])

    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    if split_by == 'time':
        events['t'] -= events['t'][0]
        assert events['t'][-1] > frames_num
        dt = events['t'][-1] // frames_num
        idx = np.arange(events['t'].size)
        for i in range(frames_num):
            t_l = dt * i
            t_r = t_l + dt
            mask = np.logical_and(events['t'] >= t_l, events['t'] < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1 if i < frames_num - 1 else events['t'].size

    elif split_by == 'number':
        di = events['t'].size // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di if i < frames_num - 1 else events['t'].size
    else:
        raise NotImplementedError

    for i in range(frames_num):
        x = events['x'][j_l[i]:j_r[i]]
        y = events['y'][j_l[i]:j_r[i]]
        p = events['p'][j_l[i]:j_r[i]]
        mask = []
        mask.append(p == 0)
        mask.append(np.logical_not(mask[0]))
        for j in range(2):
            position = y[mask[j]] * height + x[mask[j]]
            events_number_per_pos = np.bincount(position)
            frames[i][j][np.arange(events_number_per_pos.size)] += events_number_per_pos

        if normalization == 'frequency':
            if split_by == 'time':
                if i < frames_num - 1:
                    frames[i] /= dt
                else:
                    frames[i] /= (dt + events['t'][-1] % frames_num)
            elif split_by == 'number':
                    frames[i] /= (events['t'][j_r[i]] - events['t'][j_l[i]])  # 表示脉冲发放的频率

            else:
                raise NotImplementedError
    return frames.reshape((frames_num, 2, height, width))





def convert_events_dir_to_frames_dir(events_data_dir, frames_data_dir, suffix, read_function, height, width,
                                              frames_num=10, split_by='time', normalization=None, thread_num=1, compress=False):
    def cvt_fun(events_file_list):
        for events_file in events_file_list:
            frames = integrate_events_to_frames(read_function(events_file), height, width, frames_num, split_by,
                                                normalization)
            if compress:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npz')
                np.savez_compressed(frames_file, frames)
            else:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npy')
                np.save(frames_file, frames)
                
    print('A) here:', events_data_dir, '~', suffix)
    events_file_list = glob.glob(events_data_dir + '/*' + suffix)#utils.list_files(events_data_dir, suffix, True)
    if thread_num == 1:
        cvt_fun(events_file_list)
    else:
        thread_list = []
        block = events_file_list.__len__() // thread_num
        for i in range(thread_num - 1):
            thread_list.append(FunctionThread(cvt_fun, events_file_list[i * block: (i + 1) * block]))
            thread_list[-1].start()
            print(f'thread {i} start, processing files index: {i * block} : {(i + 1) * block}.')
        thread_list.append(FunctionThread(cvt_fun, events_file_list[(thread_num - 1) * block:]))
        thread_list[-1].start()
        print(f'thread {thread_num} start, processing files index: {(thread_num - 1) * block} : {events_file_list.__len__()}.')
        for i in range(thread_num):
            thread_list[i].join()
            print(f'thread {i} finished.')



def normalize_frame(frames: np.ndarray or torch.Tensor, normalization: str):
    eps = 1e-5
    for i in range(frames.shape[0]):
        if normalization == 'max':
            frames[i][0] /= max(frames[i][0].max(), eps)
            frames[i][1] /= max(frames[i][1].max(), eps)

        elif normalization == 'norm':
            frames[i][0] = (frames[i][0] - frames[i][0].mean()) / np.sqrt(max(frames[i][0].var(), eps))
            frames[i][1] = (frames[i][1] - frames[i][1].mean()) / np.sqrt(max(frames[i][1].var(), eps))

        elif normalization == 'sum':
            frames[i][0] /= max(frames[i][0].sum(), eps)
            frames[i][1] /= max(frames[i][1].sum(), eps)

        else:
            raise NotImplementedError
    return frames


class DVS128Gesture(EventsFramesDatasetBase):
    @staticmethod
    def get_wh():
        return 128, 128

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        file_name = os.path.join(download_root, 'DvsGesture.tar.gz')
        if os.path.exists(file_name):
            print('DvsGesture.tar.gz already exists, check md5')
            if utils.check_md5(file_name, resource[1]):
                print('md5 checked, extracting...')
                utils.extract_archive(file_name, extract_root)
                return
            else:
                print(f'{file_name} corrupted.')


        print(f'Please download from {resource[0]} and save to {download_root} manually.')
        raise NotImplementedError


    @staticmethod
    def read_bin(file_name: str):
        # https://gitlab.com/inivation/dv/dv-python/
        with open(file_name, 'rb') as bin_f:
            # skip ascii header
            line = bin_f.readline()
            while line.startswith(b'#'):
                if line == b'#!END-HEADER\r\n':
                    break
                else:
                    line = bin_f.readline()

            txyp = {
                't': [],
                'x': [],
                'y': [],
                'p': []
            }
            while True:
                header = bin_f.read(28)
                if not header or len(header) == 0:
                    break

                # read header
                e_type = struct.unpack('H', header[0:2])[0]
                e_source = struct.unpack('H', header[2:4])[0]
                e_size = struct.unpack('I', header[4:8])[0]
                e_offset = struct.unpack('I', header[8:12])[0]
                e_tsoverflow = struct.unpack('I', header[12:16])[0]
                e_capacity = struct.unpack('I', header[16:20])[0]
                e_number = struct.unpack('I', header[20:24])[0]
                e_valid = struct.unpack('I', header[24:28])[0]

                data_length = e_capacity * e_size
                data = bin_f.read(data_length)
                counter = 0

                if e_type == 1:
                    while data[counter:counter + e_size]:
                        aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                        timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                        x = (aer_data >> 17) & 0x00007FFF
                        y = (aer_data >> 2) & 0x00007FFF
                        pol = (aer_data >> 1) & 0x00000001
                        counter = counter + e_size
                        txyp['x'].append(x)
                        txyp['y'].append(y)
                        txyp['t'].append(timestamp)
                        txyp['p'].append(pol)
                else:
                    # non-polarity event packet, not implemented
                    pass
            txyp['x'] = np.asarray(txyp['x'])
            txyp['y'] = np.asarray(txyp['y'])
            txyp['t'] = np.asarray(txyp['t'])
            txyp['p'] = np.asarray(txyp['p'])
            return txyp


    @staticmethod
    def convert_aedat_dir_to_npy_dir(aedat_data_dir: str, events_npy_train_root: str, events_npy_test_root: str):
        def cvt_files_fun(aedat_file_list, output_dir):
            for aedat_file in aedat_file_list:
                base_name = aedat_file[0: -6]
                events = DVS128Gesture.read_bin(os.path.join(aedat_data_dir, aedat_file))
                events_csv = np.loadtxt(os.path.join(aedat_data_dir, base_name + '_labels.csv'),
                                        dtype=np.uint32, delimiter=',', skiprows=1)
                index = 0
                index_l = 0
                index_r = 0
                for i in range(events_csv.shape[0]):
                    label = events_csv[i][0]
                    t_start = events_csv[i][1]
                    t_end = events_csv[i][2]

                    while True:
                        t = events['t'][index]
                        if t < t_start:
                            index += 1
                        else:
                            index_l = index  # 左闭
                            break
                    while True:
                        t = events['t'][index]
                        if t < t_end:
                            index += 1
                        else:
                            index_r = index  # 右开
                            break
                    # [index_l, index_r)
                    j = 0
                    while True:
                        file_name = os.path.join(output_dir, f'{base_name}_{label}_{j}.npy')
                        if os.path.exists(file_name):
                            j += 1
                        else:
                            np.save(file=file_name, arr={
                                't': events['t'][index_l:index_r],
                                'x': events['x'][index_l:index_r],
                                'y': events['y'][index_l:index_r],
                                'p': events['p'][index_l:index_r]
                            })
                            break

        with open(os.path.join(aedat_data_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_data_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            train_list = []
            for fname in trials_to_train_txt.readlines():
                fname = fname.strip()
                if fname.__len__() > 0:
                    train_list.append(fname)
            test_list = []
            for fname in trials_to_test_txt.readlines():
                fname = fname.strip()
                if fname.__len__() > 0:
                    test_list.append(fname)

        print('convert events data from aedat to numpy format.')

        npy_data_num = train_list.__len__() + test_list.__len__()
        thread_num = max(multiprocessing.cpu_count(), 2)
        block = train_list.__len__() // (thread_num - 1)
        thread_list = []
        for i in range(thread_num - 1):

            thread_list.append(FunctionThread(cvt_files_fun, train_list[i * block: (i + 1) * block], events_npy_train_root))
            print(f'thread {i} start')
            thread_list[-1].start()

        thread_list.append(FunctionThread(cvt_files_fun, test_list, events_npy_test_root))
        print(f'thread {thread_num - 1} start')
        thread_list[-1].start()

        while True:
            working_thread = []
            finished_thread = []
            for i in range(thread_list.__len__()):
                if thread_list[i].is_alive():
                    working_thread.append(i)
                else:
                    finished_thread.append(i)
#             pbar.update(utils.list_files(events_npy_train_root, '.npy').__len__() + utils.list_files(events_npy_test_root, '.npy').__len__())
            print('wroking thread:', working_thread)
            print('finished thread:', finished_thread)
            if finished_thread.__len__() == thread_list.__len__():
                return
            else:
                time.sleep(10)


    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str, normalization: str or None):
        width, height = DVS128Gesture.get_wh()
        def read_fun(file_name):
            return np.load(file_name, allow_pickle=True).item()
        convert_events_dir_to_frames_dir(events_data_dir, frames_data_dir, '.npy',
                                                               read_fun, height, width, frames_num, split_by,
                                                               normalization, thread_num=4)

    @staticmethod
    def get_events_item(file_name):
        return np.load(file_name, allow_pickle=True).item(), int(os.path.basename(file_name).split('_')[-2]) - 1

    @staticmethod
    def get_frames_item(file_name):
        return torch.from_numpy(np.load(file_name)).float(), int(os.path.basename(file_name).split('_')[-2]) - 1

    def __init__(self, root: str, train: bool, use_frame=True, frames_num=10, split_by='number', normalization='max'):
        super().__init__()
        events_npy_root = os.path.join(root, 'events_npy')
        events_npy_train_root = os.path.join(events_npy_root, 'train')
        events_npy_test_root = os.path.join(events_npy_root, 'test')
        if os.path.exists(events_npy_train_root) and os.path.exists(events_npy_test_root):
            print(f'npy format events data root {events_npy_train_root}, {events_npy_test_root} already exists')
        else:

            extracted_root = os.path.join(root, 'extracted')
            if os.path.exists(extracted_root):
                print(f'extracted root {extracted_root} already exists.')
            else:
                self.download_and_extract(root, extracted_root)
            if not os.path.exists(events_npy_root):
                os.mkdir(events_npy_root)
                print(f'mkdir {events_npy_root}')
            os.mkdir(events_npy_train_root)
            print(f'mkdir {events_npy_train_root}')
            os.mkdir(events_npy_test_root)
            print(f'mkdir {events_npy_test_root}')
            print('read events data from *.aedat and save to *.npy...')
            self.convert_aedat_dir_to_npy_dir(os.path.join(extracted_root, 'DvsGesture'), events_npy_train_root, events_npy_test_root)


        self.file_name = []
        self.use_frame = use_frame
        self.data_dir = None
        if use_frame:
            self.normalization = normalization
            if normalization == 'frequency':
                dir_suffix = normalization
            else:
                dir_suffix = None
            frames_root = os.path.join(root, f'frames_num_{frames_num}_split_by_{split_by}_normalization_{dir_suffix}')
            frames_train_root = os.path.join(frames_root, 'train')
            frames_test_root = os.path.join(frames_root, 'test')
            if os.path.exists(frames_root):
                print(f'frames data root {frames_root} already exists.')
            else:
                os.mkdir(frames_root)
                os.mkdir(frames_train_root)
                os.mkdir(frames_test_root)
                print(f'mkdir {frames_root}, {frames_train_root}, {frames_test_root}.')
                print('creating frames data..')
                self.create_frames_dataset(events_npy_train_root, frames_train_root, frames_num, split_by, normalization)
                self.create_frames_dataset(events_npy_test_root, frames_test_root, frames_num, split_by, normalization)
            if train:
                self.data_dir = frames_train_root
            else:
                self.data_dir = frames_test_root

            self.file_name = glob.glob(self.data_dir + '/*.npy')#utils.list_files(self.data_dir, '.npy', True)

        else:
            if train:
                self.data_dir = events_npy_train_root
            else:
                self.data_dir = events_npy_test_root
            self.file_name = glob.glob(self.data_dir + '/*.npy')#utils.list_files(self.data_dir, '.npy', True)


    def __len__(self):
        return self.file_name.__len__()
    def __getitem__(self, index):
        if self.use_frame:
            frames, labels = self.get_frames_item(self.file_name[index])
            if self.normalization is not None and self.normalization != 'frequency':
                frames = normalize_frame(frames, self.normalization)
            return frames, labels
        else:
            return self.get_events_item(self.file_name[index])


