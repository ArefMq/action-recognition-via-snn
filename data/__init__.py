from .spikingjelly_adaptation import labels_dict, DVS128Gesture

classes = list(labels_dict.keys())
num_of_classes = len(classes)


def load_all(**kwargs):
    frames = kwargs.get('frames', 20)
    data_path = kwargs.get('data_path')
    polarity_mode = kwargs.get('polarity_mode', None)
    c = 0
    
    if polarity_mode is not None and polarity_mode != 'accumulative':
        raise Exception('Spiking Jelly only supports "accumulative" polarity mode.')
    
    for x_data, y_data in DVS128Gesture(root=data_path, train=True,  use_frame=True, frames_num=frames, split_by='number', normalization=None):
        yield x_data, y_data
    for x_data, y_data in DVS128Gesture(root=data_path, train=False, use_frame=True, frames_num=frames, split_by='number', normalization=None):
        c += 1
        if c < 10:
            yield x_data, y_data
        else:
            break

